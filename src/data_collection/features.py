from dataclasses import dataclass, asdict
from typing import Optional, Dict
import pandas as pd
import time


@dataclass
class TokenFeatures:
    """Container for token features used in ML model"""

    # Basic info
    token_address: str
    token_symbol: str
    chain_id: str
    token_age_hours: float

    # Liquidity & Market
    liquidity_usd: float
    market_cap: float
    price_usd: float

    # Volume metrics
    volume_5min: float
    volume_1hour: float
    volume_24hour: float

    # Transaction counts
    buy_count_5min: int
    sell_count_5min: int
    buy_count_1hour: int
    sell_count_1hour: int
    unique_buyers: int = 0  # Optional - requires blockchain data
    unique_sellers: int = 0  # Optional - requires blockchain data

    # Holder analysis (optional - requires blockchain data)
    total_holders: int = 0
    top1_holder_pct: float = 0
    top10_holder_pct: float = 0

    # Price dynamics
    price_change_5min: float = 0
    price_change_1hour: float = 0
    price_change_24hour: float = 0

    # Safety features (optional - requires contract analysis)
    liquidity_locked: bool = False
    mint_authority_revoked: bool = False
    has_blacklist_function: bool = False
    contract_verified: bool = False

    # Derived features (calculated properties)
    @property
    def buy_sell_ratio_5min(self) -> float:
        """Buy/sell ratio for 5min window"""
        if self.sell_count_5min == 0:
            return float('inf') if self.buy_count_5min > 0 else 0
        return self.buy_count_5min / self.sell_count_5min

    @property
    def buy_sell_ratio_1hour(self) -> float:
        """Buy/sell ratio for 1hour window"""
        if self.sell_count_1hour == 0:
            return float('inf') if self.buy_count_1hour > 0 else 0
        return self.buy_count_1hour / self.sell_count_1hour

    @property
    def liquidity_to_mcap_ratio(self) -> float:
        """Liquidity to market cap ratio"""
        if self.market_cap == 0:
            return 0
        return self.liquidity_usd / self.market_cap

    @property
    def safety_score(self) -> int:
        """Count of safety features present (0-4)"""
        return sum([
            self.liquidity_locked,
            self.mint_authority_revoked,
            not self.has_blacklist_function,
            self.contract_verified
        ])

    @property
    def volume_acceleration(self) -> float:
        """Volume acceleration (5min vs 1hour rate)"""
        if self.volume_5min == 0:
            return 0
        # Normalize to per-minute rate
        rate_5min = self.volume_5min / 5
        rate_1hour = self.volume_1hour / 60
        if rate_1hour == 0:
            return float('inf') if rate_5min > 0 else 0
        return rate_5min / rate_1hour

    @property
    def whale_dominance(self) -> float:
        """Whale dominance (top holders relative to total)"""
        if self.total_holders == 0:
            return 1.0  # Assume high risk if no data
        return self.top10_holder_pct / 100  # Convert to 0-1 scale

    @property
    def launch_phase_risk(self) -> float:
        """Risk score based on token age (1.0 = new, 0.3 = established)"""
        if self.token_age_hours < 24:
            return 1.0  # Very new
        elif self.token_age_hours < 168:  # 1 week
            return 0.7  # New
        else:
            return 0.3  # Established

    def to_dict(self) -> Dict:
        """Convert to dictionary for ML model"""
        # Get all dataclass fields
        base_dict = asdict(self)

        # Add derived features
        derived = {
            'buy_sell_ratio_5min': self.buy_sell_ratio_5min,
            'buy_sell_ratio_1hour': self.buy_sell_ratio_1hour,
            'liquidity_to_mcap_ratio': self.liquidity_to_mcap_ratio,
            'safety_score': self.safety_score,
            'volume_acceleration': self.volume_acceleration,
            'whale_dominance': self.whale_dominance,
            'launch_phase_risk': self.launch_phase_risk,
        }

        # Convert boolean to int for ML
        base_dict['liquidity_locked'] = int(base_dict['liquidity_locked'])
        base_dict['mint_authority_revoked'] = int(base_dict['mint_authority_revoked'])
        base_dict['has_blacklist_function'] = int(base_dict['has_blacklist_function'])
        base_dict['contract_verified'] = int(base_dict['contract_verified'])

        # Combine
        base_dict.update(derived)

        return base_dict

    def to_ml_features(self) -> Dict:
        """
        Get only the features needed for ML (exclude identifiers)
        """
        all_features = self.to_dict()

        # Remove non-feature fields
        exclude_fields = ['token_address', 'token_symbol', 'chain_id']
        return {k: v for k, v in all_features.items() if k not in exclude_fields}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame (single row)"""
        return pd.DataFrame([self.to_dict()])


class FeatureCollector:
    """Collect features from API data and external sources"""

    def __init__(self, dex_api):
        self.dex_api = dex_api

    async def collect_features(self, token_address: str, chain_id: str) -> Optional[TokenFeatures]:
        """
        Collect all features for a token

        Args:
            token_address: Token contract address
            chain_id: Chain identifier (e.g., 'solana', 'ethereum')

        Returns:
            TokenFeatures object or None if data unavailable
        """
        try:
            # Get token data from DexScreener
            data = await self.dex_api.get_token_pairs(chain_id, token_address)

            if not data or 'pairs' not in data or len(data['pairs']) == 0:
                print(f"No data found for token {token_address} on {chain_id}")
                return None

            # Use the primary (most liquid) pair
            pair = data['pairs'][0]

            # Parse using the API's built-in parser
            parsed = self.dex_api.parse_token_data(pair)

            # Create features object
            features = TokenFeatures(
                # Basic info
                token_address=token_address,
                token_symbol=parsed['token_symbol'],
                chain_id=chain_id,
                token_age_hours=parsed['token_age_hours'],

                # Liquidity & Market
                liquidity_usd=parsed['liquidity_usd'],
                market_cap=parsed['market_cap'],
                price_usd=parsed['price_usd'],

                # Volume
                volume_5min=parsed['volume_5m'],
                volume_1hour=parsed['volume_1h'],
                volume_24hour=parsed['volume_24h'],

                # Transactions
                buy_count_5min=parsed['txns_5m_buys'],
                sell_count_5min=parsed['txns_5m_sells'],
                buy_count_1hour=parsed['txns_1h_buys'],
                sell_count_1hour=parsed['txns_1h_sells'],

                # Price changes
                price_change_5min=parsed['price_change_5m'],
                price_change_1hour=parsed['price_change_1h'],
                price_change_24hour=parsed['price_change_24h'],

                # Holders - TODO: Add blockchain data collection
                unique_buyers=0,
                unique_sellers=0,
                total_holders=0,
                top1_holder_pct=0,
                top10_holder_pct=0,

                # Safety - TODO: Add contract analysis
                liquidity_locked=False,
                mint_authority_revoked=False,
                has_blacklist_function=False,
                contract_verified=False,
            )

            return features

        except Exception as e:
            print(f"Error collecting features for {token_address}: {e}")
            return None

    async def collect_features_batch(self, token_addresses: list, chain_id: str) -> list:
        """
        Collect features for multiple tokens

        Args:
            token_addresses: List of token addresses
            chain_id: Chain identifier

        Returns:
            List of TokenFeatures objects
        """
        features_list = []

        for address in token_addresses:
            features = await self.collect_features(address, chain_id)
            if features:
                features_list.append(features)
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

        return features_list

    def _calculate_age_hours(self, created_timestamp: int) -> float:
        """Calculate token age in hours from timestamp"""
        current_time = time.time()
        created_time = created_timestamp / 1000  # Convert from ms
        age_seconds = current_time - created_time
        return age_seconds / 3600


# Example usage
async def main():
    """Example of feature collection"""
    from .dex_api import DexScreenerAPI

    async with DexScreenerAPI() as api:
        collector = FeatureCollector(api)

        print("Testing Feature Collection...")

        # Test with a known Solana token (USDC)
        print("\nCollecting features for Solana USDC...")
        features = await collector.collect_features(
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "solana"
        )

        if features:
            print(f"\nToken: {features.token_symbol}")
            print(f"Age: {features.token_age_hours:.1f} hours")
            print(f"Price: ${features.price_usd:.8f}")
            print(f"Liquidity: ${features.liquidity_usd:,.2f}")
            print(f"Market Cap: ${features.market_cap:,.2f}")
            print(f"Volume 24h: ${features.volume_24hour:,.2f}")
            print(f"Buy/Sell Ratio (5min): {features.buy_sell_ratio_5min:.2f}")
            print(f"Liquidity/MCap Ratio: {features.liquidity_to_mcap_ratio:.4f}")
            print(f"Launch Phase Risk: {features.launch_phase_risk:.2f}")
            print(f"Safety Score: {features.safety_score}/4")

            # Show ML features
            print("\nML Features:")
            ml_features = features.to_ml_features()
            for key, value in list(ml_features.items())[:10]:
                print(f"  {key}: {value}")

            print(f"\nTotal ML features: {len(ml_features)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
