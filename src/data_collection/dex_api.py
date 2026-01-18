import asyncio
import aiohttp
import time
from typing import Dict, List, Optional
from datetime import datetime


class RateLimiter:
    """Rate limiter to prevent API overuse"""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0

    async def wait_if_needed(self):
        """Wait if needed to respect rate limit"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class DexScreenerAPI:
    """
    Async wrapper for DexScreener API

    Rate limits:
    - Main endpoints: 300 requests/minute
    - Token boosts: 60 requests/minute
    """

    BASE_URL = "https://api.dexscreener.com"

    def __init__(self, rate_limit_per_minute: int = 300):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self._session_owned = False

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
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"API request error for {url}: {e}")
            return {}
        except asyncio.TimeoutError:
            print(f"API request timeout for {url}")
            return {}
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")
            return {}

    async def get_token_pairs(self, chain_id: str, token_address: str) -> Dict:
        """
        Get pairs for a specific token

        Args:
            chain_id: Chain identifier (e.g., 'solana', 'ethereum', 'base')
            token_address: Token contract address

        Returns:
            Dict with 'pairs' list and token information
        """
        endpoint = f"/latest/dex/tokens/{chain_id}/{token_address}"
        return await self._get(endpoint)

    async def search_pairs(self, query: str) -> Dict:
        """
        Search for pairs by query

        Args:
            query: Search term (token symbol, name, or address)

        Returns:
            Dict with 'pairs' list matching the query
        """
        endpoint = "/latest/dex/search"
        return await self._get(endpoint, params={"q": query})

    async def get_boosted_tokens(self) -> List[Dict]:
        """
        Get tokens with active boosts (promoted tokens)
        Rate limit: 60 requests/minute

        Returns:
            List of boosted token data
        """
        endpoint = "/token-boosts/latest/v1"
        data = await self._get(endpoint)
        return data if isinstance(data, list) else []

    async def get_token_profiles(self) -> List[Dict]:
        """
        Get token profiles

        Returns:
            List of token profiles
        """
        endpoint = "/token-profiles/latest/v1"
        data = await self._get(endpoint)
        return data if isinstance(data, list) else []

    async def get_token_profile(self, chain_id: str, token_address: str) -> Optional[Dict]:
        """
        Get specific token profile

        Args:
            chain_id: Chain identifier
            token_address: Token contract address

        Returns:
            Token profile dict or None if not found
        """
        profiles = await self.get_token_profiles()

        for profile in profiles:
            if (profile.get('chainId') == chain_id and
                profile.get('tokenAddress', '').lower() == token_address.lower()):
                return profile

        return None

    async def get_pair_by_address(self, chain_id: str, pair_address: str) -> Optional[Dict]:
        """
        Get specific pair by its address

        Args:
            chain_id: Chain identifier
            pair_address: Pair contract address

        Returns:
            Pair data or None
        """
        endpoint = f"/latest/dex/pairs/{chain_id}/{pair_address}"
        data = await self._get(endpoint)

        if data and 'pair' in data:
            return data['pair']
        elif data and 'pairs' in data and len(data['pairs']) > 0:
            return data['pairs'][0]

        return None

    def parse_token_data(self, pair_data: Dict) -> Dict:
        """
        Parse and normalize token data from pair response

        Args:
            pair_data: Raw pair data from API

        Returns:
            Normalized token data dict
        """
        if not pair_data:
            return {}

        base_token = pair_data.get('baseToken', {})
        quote_token = pair_data.get('quoteToken', {})

        # Calculate token age
        pair_created = pair_data.get('pairCreatedAt')
        token_age_hours = 0
        if pair_created:
            created_time = pair_created / 1000  # Convert from ms
            token_age_hours = (time.time() - created_time) / 3600

        return {
            # Basic info
            'chain_id': pair_data.get('chainId', ''),
            'dex_id': pair_data.get('dexId', ''),
            'pair_address': pair_data.get('pairAddress', ''),
            'token_address': base_token.get('address', ''),
            'token_symbol': base_token.get('symbol', ''),
            'token_name': base_token.get('name', ''),
            'token_age_hours': token_age_hours,

            # Price and market data
            'price_usd': float(pair_data.get('priceUsd', 0)),
            'price_native': float(pair_data.get('priceNative', 0)),
            'liquidity_usd': pair_data.get('liquidity', {}).get('usd', 0),
            'market_cap': pair_data.get('marketCap', 0) or pair_data.get('fdv', 0),

            # Volume data
            'volume_5m': pair_data.get('volume', {}).get('m5', 0),
            'volume_1h': pair_data.get('volume', {}).get('h1', 0),
            'volume_6h': pair_data.get('volume', {}).get('h6', 0),
            'volume_24h': pair_data.get('volume', {}).get('h24', 0),

            # Transaction counts
            'txns_5m_buys': pair_data.get('txns', {}).get('m5', {}).get('buys', 0),
            'txns_5m_sells': pair_data.get('txns', {}).get('m5', {}).get('sells', 0),
            'txns_1h_buys': pair_data.get('txns', {}).get('h1', {}).get('buys', 0),
            'txns_1h_sells': pair_data.get('txns', {}).get('h1', {}).get('sells', 0),
            'txns_6h_buys': pair_data.get('txns', {}).get('h6', {}).get('buys', 0),
            'txns_6h_sells': pair_data.get('txns', {}).get('h6', {}).get('sells', 0),
            'txns_24h_buys': pair_data.get('txns', {}).get('h24', {}).get('buys', 0),
            'txns_24h_sells': pair_data.get('txns', {}).get('h24', {}).get('sells', 0),

            # Price changes
            'price_change_5m': pair_data.get('priceChange', {}).get('m5', 0),
            'price_change_1h': pair_data.get('priceChange', {}).get('h1', 0),
            'price_change_6h': pair_data.get('priceChange', {}).get('h6', 0),
            'price_change_24h': pair_data.get('priceChange', {}).get('h24', 0),

            # Additional metadata
            'url': pair_data.get('url', ''),
            'labels': pair_data.get('labels', []),
            'website': pair_data.get('info', {}).get('websites', []),
            'socials': pair_data.get('info', {}).get('socials', []),
        }


# Example usage and testing
async def main():
    """Example usage of DexScreener API"""
    async with DexScreenerAPI() as api:
        print("Testing DexScreener API...")

        # Test 1: Search for a popular token
        print("\n1. Searching for PEPE...")
        results = await api.search_pairs("PEPE")
        if results and 'pairs' in results:
            print(f"Found {len(results['pairs'])} pairs")
            if len(results['pairs']) > 0:
                pair = results['pairs'][0]
                print(f"First result: {pair.get('baseToken', {}).get('symbol')} on {pair.get('dexId')}")
                parsed = api.parse_token_data(pair)
                print(f"Price: ${parsed['price_usd']:.8f}")
                print(f"Liquidity: ${parsed['liquidity_usd']:,.2f}")
                print(f"Volume 24h: ${parsed['volume_24h']:,.2f}")

        # Test 2: Get boosted tokens
        print("\n2. Getting boosted tokens...")
        boosted = await api.get_boosted_tokens()
        print(f"Found {len(boosted)} boosted tokens")

        # Test 3: Get specific token (example with Solana USDC)
        print("\n3. Getting Solana USDC token data...")
        token_data = await api.get_token_pairs(
            "solana",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        )
        if token_data and 'pairs' in token_data:
            print(f"Found {len(token_data['pairs'])} pairs for USDC")

        print("\nAPI tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
