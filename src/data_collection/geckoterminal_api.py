"""GeckoTerminal API wrapper for historical OHLCV data"""
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger


class Timeframe(Enum):
    """OHLCV timeframe options"""
    MINUTE_1 = ("minute", 1)
    MINUTE_5 = ("minute", 5)
    MINUTE_15 = ("minute", 15)
    HOUR_1 = ("hour", 1)
    HOUR_4 = ("hour", 4)
    HOUR_12 = ("hour", 12)
    DAY_1 = ("day", 1)

    @property
    def period(self) -> str:
        return self.value[0]

    @property
    def aggregate(self) -> int:
        return self.value[1]


@dataclass
class OHLCV:
    """Single OHLCV candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_api_data(cls, data: List) -> 'OHLCV':
        """Create from GeckoTerminal API response [timestamp, open, high, low, close, volume]"""
        return cls(
            timestamp=datetime.fromtimestamp(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def wait_if_needed(self):
        """Wait if needed to respect rate limit"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class GeckoTerminalAPI:
    """
    Async wrapper for GeckoTerminal API (CoinGecko's DEX aggregator)

    Provides:
    - Historical OHLCV data (up to 6 months)
    - Pool/pair information
    - Multi-chain support (200+ networks)

    Rate limits:
    - Free tier: 30 calls/minute
    - Pro tier: Higher limits available

    API Docs: https://www.geckoterminal.com/dex-api
    """

    BASE_URL = "https://api.geckoterminal.com/api/v2"

    # Network ID mappings (GeckoTerminal network IDs)
    NETWORK_IDS = {
        'solana': 'solana',
        'ethereum': 'eth',
        'base': 'base',
        'arbitrum': 'arbitrum',
        'polygon': 'polygon_pos',
        'bsc': 'bsc',
        'avalanche': 'avax',
        'optimism': 'optimism',
    }

    def __init__(self, rate_limit_per_minute: int = 30):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self._session_owned = False
        self.logger = get_logger("GeckoTerminal")

    async def __aenter__(self):
        """Async context manager entry"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._session_owned = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session_owned and self.session:
            await self.session.close()
            self.session = None

    async def _ensure_session(self):
        """Ensure we have an active session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._session_owned = True

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make rate-limited GET request"""
        await self._ensure_session()
        await self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            async with self.session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 429:
                    self.logger.warning("Rate limit hit, waiting 60s...")
                    await asyncio.sleep(60)
                    return await self._get(endpoint, params)

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            self.logger.error(f"API request error for {url}: {e}")
            return {}
        except asyncio.TimeoutError:
            self.logger.error(f"API request timeout for {url}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
            return {}

    def _get_network_id(self, chain: str) -> str:
        """Convert chain name to GeckoTerminal network ID"""
        return self.NETWORK_IDS.get(chain.lower(), chain.lower())

    async def get_networks(self) -> List[Dict]:
        """
        Get list of supported networks

        Returns:
            List of network data dicts
        """
        data = await self._get("/networks")
        return data.get('data', [])

    async def search_pools(
        self,
        query: str,
        network: Optional[str] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        Search for pools by token name/symbol

        Args:
            query: Search term
            network: Optional network filter
            page: Page number for pagination

        Returns:
            List of pool data
        """
        params = {'query': query, 'page': page}
        if network:
            params['network'] = self._get_network_id(network)

        data = await self._get("/search/pools", params)
        return data.get('data', [])

    async def get_pool_info(self, network: str, pool_address: str) -> Optional[Dict]:
        """
        Get pool information

        Args:
            network: Network ID (e.g., 'solana', 'ethereum')
            pool_address: Pool/pair contract address

        Returns:
            Pool data or None
        """
        network_id = self._get_network_id(network)
        data = await self._get(f"/networks/{network_id}/pools/{pool_address}")
        return data.get('data')

    async def get_pool_ohlcv(
        self,
        network: str,
        pool_address: str,
        timeframe: Timeframe = Timeframe.HOUR_1,
        before_timestamp: Optional[int] = None,
        limit: int = 1000,
        currency: str = "usd"
    ) -> List[OHLCV]:
        """
        Get historical OHLCV data for a pool

        Args:
            network: Network ID (e.g., 'solana', 'ethereum')
            pool_address: Pool/pair contract address
            timeframe: Candle timeframe
            before_timestamp: Unix timestamp to fetch data before (for pagination)
            limit: Number of candles (max 1000)
            currency: Quote currency ('usd' or 'token')

        Returns:
            List of OHLCV candles (oldest first)
        """
        network_id = self._get_network_id(network)
        endpoint = f"/networks/{network_id}/pools/{pool_address}/ohlcv/{timeframe.period}"

        params = {
            'aggregate': timeframe.aggregate,
            'limit': min(limit, 1000),
            'currency': currency
        }

        if before_timestamp:
            params['before_timestamp'] = before_timestamp

        data = await self._get(endpoint, params)

        if not data or 'data' not in data:
            return []

        ohlcv_list = data['data'].get('attributes', {}).get('ohlcv_list', [])

        # Convert to OHLCV objects
        candles = [OHLCV.from_api_data(c) for c in ohlcv_list]

        # Sort by timestamp (oldest first)
        candles.sort(key=lambda x: x.timestamp)

        return candles

    async def get_extended_ohlcv(
        self,
        network: str,
        pool_address: str,
        timeframe: Timeframe = Timeframe.HOUR_1,
        days: int = 30,
        currency: str = "usd"
    ) -> List[OHLCV]:
        """
        Get extended historical OHLCV by paginating through multiple requests

        Args:
            network: Network ID
            pool_address: Pool address
            timeframe: Candle timeframe
            days: Number of days of history to fetch (max ~180 days for free tier)
            currency: Quote currency

        Returns:
            List of OHLCV candles for the entire period
        """
        all_candles = []
        before_timestamp = None
        target_date = datetime.now() - timedelta(days=days)

        self.logger.info(f"Fetching {days} days of {timeframe.name} data for {pool_address[:8]}...")

        while True:
            candles = await self.get_pool_ohlcv(
                network=network,
                pool_address=pool_address,
                timeframe=timeframe,
                before_timestamp=before_timestamp,
                limit=1000,
                currency=currency
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Check if we've reached our target date
            oldest_candle = min(candles, key=lambda x: x.timestamp)
            if oldest_candle.timestamp < target_date:
                # Filter to only include candles within our date range
                all_candles = [c for c in all_candles if c.timestamp >= target_date]
                break

            # Set before_timestamp for next request (oldest candle timestamp)
            before_timestamp = int(oldest_candle.timestamp.timestamp())

            self.logger.debug(f"Fetched {len(candles)} candles, total: {len(all_candles)}")

            # Small delay between requests
            await asyncio.sleep(0.5)

        # Remove duplicates and sort
        seen = set()
        unique_candles = []
        for c in all_candles:
            key = (c.timestamp, c.open, c.close)
            if key not in seen:
                seen.add(key)
                unique_candles.append(c)

        unique_candles.sort(key=lambda x: x.timestamp)

        self.logger.info(f"Retrieved {len(unique_candles)} candles from {unique_candles[0].timestamp if unique_candles else 'N/A'} to {unique_candles[-1].timestamp if unique_candles else 'N/A'}")

        return unique_candles

    async def get_top_pools(
        self,
        network: str,
        page: int = 1,
        sort: str = "h24_volume_usd_liquidity_desc"
    ) -> List[Dict]:
        """
        Get top pools on a network

        Args:
            network: Network ID
            page: Page number
            sort: Sort order (h24_volume_usd_liquidity_desc, h24_tx_count_desc, etc.)

        Returns:
            List of pool data
        """
        network_id = self._get_network_id(network)
        params = {'page': page, 'sort': sort}
        data = await self._get(f"/networks/{network_id}/pools", params)
        return data.get('data', [])

    async def get_trending_pools(
        self,
        network: Optional[str] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        Get trending pools

        Args:
            network: Optional network filter
            page: Page number

        Returns:
            List of trending pool data
        """
        if network:
            network_id = self._get_network_id(network)
            endpoint = f"/networks/{network_id}/trending_pools"
        else:
            endpoint = "/networks/trending_pools"

        data = await self._get(endpoint, {'page': page})
        return data.get('data', [])

    async def get_new_pools(
        self,
        network: Optional[str] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        Get newly created pools

        Args:
            network: Optional network filter
            page: Page number

        Returns:
            List of new pool data
        """
        if network:
            network_id = self._get_network_id(network)
            endpoint = f"/networks/{network_id}/new_pools"
        else:
            endpoint = "/networks/new_pools"

        data = await self._get(endpoint, {'page': page})
        return data.get('data', [])

    def parse_pool_data(self, pool_data: Dict) -> Dict:
        """
        Parse and normalize pool data from API response

        Args:
            pool_data: Raw pool data from API

        Returns:
            Normalized pool data dict
        """
        if not pool_data:
            return {}

        attrs = pool_data.get('attributes', {})

        # Extract network from pool ID (format: "network_pooladdress")
        pool_id = pool_data.get('id', '')
        network = pool_id.split('_')[0] if '_' in pool_id else ''

        return {
            'pool_address': attrs.get('address', ''),
            'name': attrs.get('name', ''),
            'network': network,
            'dex': pool_data.get('relationships', {}).get('dex', {}).get('data', {}).get('id', ''),
            'base_token_address': attrs.get('base_token_price_native_currency', ''),
            'base_token_symbol': attrs.get('name', '').split('/')[0] if '/' in attrs.get('name', '') else '',
            'quote_token_symbol': attrs.get('name', '').split('/')[1] if '/' in attrs.get('name', '') else '',
            'price_usd': float(attrs.get('base_token_price_usd', 0) or 0),
            'price_native': float(attrs.get('base_token_price_native_currency', 0) or 0),
            'fdv_usd': float(attrs.get('fdv_usd', 0) or 0),
            'market_cap_usd': float(attrs.get('market_cap_usd', 0) or 0),
            'reserve_in_usd': float(attrs.get('reserve_in_usd', 0) or 0),  # Liquidity
            'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
            'volume_6h_usd': float(attrs.get('volume_usd', {}).get('h6', 0) or 0),
            'volume_1h_usd': float(attrs.get('volume_usd', {}).get('h1', 0) or 0),
            'volume_5m_usd': float(attrs.get('volume_usd', {}).get('m5', 0) or 0),
            'price_change_24h': float(attrs.get('price_change_percentage', {}).get('h24', 0) or 0),
            'price_change_6h': float(attrs.get('price_change_percentage', {}).get('h6', 0) or 0),
            'price_change_1h': float(attrs.get('price_change_percentage', {}).get('h1', 0) or 0),
            'price_change_5m': float(attrs.get('price_change_percentage', {}).get('m5', 0) or 0),
            'transactions_24h_buys': attrs.get('transactions', {}).get('h24', {}).get('buys', 0),
            'transactions_24h_sells': attrs.get('transactions', {}).get('h24', {}).get('sells', 0),
            'transactions_1h_buys': attrs.get('transactions', {}).get('h1', {}).get('buys', 0),
            'transactions_1h_sells': attrs.get('transactions', {}).get('h1', {}).get('sells', 0),
            'pool_created_at': attrs.get('pool_created_at'),
        }


# Example usage and testing
async def main():
    """Example usage of GeckoTerminal API"""
    async with GeckoTerminalAPI() as api:
        print("Testing GeckoTerminal API...")

        # Test 1: Search for pools
        print("\n1. Searching for BONK pools...")
        pools = await api.search_pools("BONK", network="solana")
        if pools:
            print(f"Found {len(pools)} pools")
            pool = pools[0]
            parsed = api.parse_pool_data(pool)
            print(f"First result: {parsed['name']} on {parsed['network']}")
            print(f"Price: ${parsed['price_usd']:.8f}")
            print(f"Liquidity: ${parsed['reserve_in_usd']:,.2f}")

            # Test 2: Get OHLCV data
            print(f"\n2. Getting 7 days of hourly OHLCV for {parsed['name']}...")
            ohlcv = await api.get_extended_ohlcv(
                network="solana",
                pool_address=parsed['pool_address'],
                timeframe=Timeframe.HOUR_1,
                days=7
            )
            if ohlcv:
                print(f"Got {len(ohlcv)} candles")
                print(f"Date range: {ohlcv[0].timestamp} to {ohlcv[-1].timestamp}")
                print(f"Price range: ${min(c.low for c in ohlcv):.8f} - ${max(c.high for c in ohlcv):.8f}")

        # Test 3: Get trending pools
        print("\n3. Getting trending pools...")
        trending = await api.get_trending_pools(network="solana")
        print(f"Found {len(trending)} trending pools")
        for i, pool in enumerate(trending[:3]):
            parsed = api.parse_pool_data(pool)
            print(f"  {i+1}. {parsed['name']} - ${parsed['price_usd']:.8f} (24h vol: ${parsed['volume_24h_usd']:,.0f})")

        print("\nAPI tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
