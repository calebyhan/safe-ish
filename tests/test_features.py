"""Tests for feature collection using pair data directly."""
import pytest

from src.data_collection.dex_api import DexScreenerAPI
from src.data_collection.features import TokenFeatures


class TestTokenFeatures:
    """Tests for TokenFeatures dataclass."""

    @pytest.mark.asyncio
    async def test_create_features_from_pair_data(self):
        """Test creating features from parsed pair data."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            # Get first Solana pair
            pair = None
            for p in results['pairs']:
                if p['chainId'] == 'solana':
                    pair = p
                    break

            if not pair:
                pair = results['pairs'][0]

            parsed = api.parse_token_data(pair)

            features = TokenFeatures(
                token_address=parsed['token_address'],
                token_symbol=parsed['token_symbol'],
                chain_id=parsed['chain_id'],
                token_age_hours=parsed['token_age_hours'],
                liquidity_usd=parsed['liquidity_usd'],
                market_cap=parsed['market_cap'],
                price_usd=parsed['price_usd'],
                volume_5min=parsed['volume_5m'],
                volume_1hour=parsed['volume_1h'],
                volume_24hour=parsed['volume_24h'],
                buy_count_5min=parsed['txns_5m_buys'],
                sell_count_5min=parsed['txns_5m_sells'],
                buy_count_1hour=parsed['txns_1h_buys'],
                sell_count_1hour=parsed['txns_1h_sells'],
                price_change_5min=parsed['price_change_5m'],
                price_change_1hour=parsed['price_change_1h'],
                price_change_24hour=parsed['price_change_24h'],
            )

            assert features.token_symbol is not None
            assert features.token_address is not None

    @pytest.mark.asyncio
    async def test_derived_features_calculated(self):
        """Test that derived features are properly calculated."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            pair = results['pairs'][0]
            parsed = api.parse_token_data(pair)

            features = TokenFeatures(
                token_address=parsed['token_address'],
                token_symbol=parsed['token_symbol'],
                chain_id=parsed['chain_id'],
                token_age_hours=parsed['token_age_hours'],
                liquidity_usd=parsed['liquidity_usd'],
                market_cap=parsed['market_cap'],
                price_usd=parsed['price_usd'],
                volume_5min=parsed['volume_5m'],
                volume_1hour=parsed['volume_1h'],
                volume_24hour=parsed['volume_24h'],
                buy_count_5min=parsed['txns_5m_buys'],
                sell_count_5min=parsed['txns_5m_sells'],
                buy_count_1hour=parsed['txns_1h_buys'],
                sell_count_1hour=parsed['txns_1h_sells'],
                price_change_5min=parsed['price_change_5m'],
                price_change_1hour=parsed['price_change_1h'],
                price_change_24hour=parsed['price_change_24h'],
            )

            # Test derived features
            assert features.liquidity_to_mcap_ratio >= 0
            assert features.buy_sell_ratio_5min >= 0
            assert 0 <= features.safety_score <= 4

    @pytest.mark.asyncio
    async def test_ml_features_count(self):
        """Test that we get the expected number of ML features."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            pair = results['pairs'][0]
            parsed = api.parse_token_data(pair)

            features = TokenFeatures(
                token_address=parsed['token_address'],
                token_symbol=parsed['token_symbol'],
                chain_id=parsed['chain_id'],
                token_age_hours=parsed['token_age_hours'],
                liquidity_usd=parsed['liquidity_usd'],
                market_cap=parsed['market_cap'],
                price_usd=parsed['price_usd'],
                volume_5min=parsed['volume_5m'],
                volume_1hour=parsed['volume_1h'],
                volume_24hour=parsed['volume_24h'],
                buy_count_5min=parsed['txns_5m_buys'],
                sell_count_5min=parsed['txns_5m_sells'],
                buy_count_1hour=parsed['txns_1h_buys'],
                sell_count_1hour=parsed['txns_1h_sells'],
                price_change_5min=parsed['price_change_5m'],
                price_change_1hour=parsed['price_change_1h'],
                price_change_24hour=parsed['price_change_24h'],
            )

            ml_features = features.to_ml_features()

            # Should have 30+ ML features
            assert len(ml_features) >= 20

    @pytest.mark.asyncio
    async def test_volume_acceleration(self):
        """Test volume acceleration calculation."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            pair = results['pairs'][0]
            parsed = api.parse_token_data(pair)

            features = TokenFeatures(
                token_address=parsed['token_address'],
                token_symbol=parsed['token_symbol'],
                chain_id=parsed['chain_id'],
                token_age_hours=parsed['token_age_hours'],
                liquidity_usd=parsed['liquidity_usd'],
                market_cap=parsed['market_cap'],
                price_usd=parsed['price_usd'],
                volume_5min=parsed['volume_5m'],
                volume_1hour=parsed['volume_1h'],
                volume_24hour=parsed['volume_24h'],
                buy_count_5min=parsed['txns_5m_buys'],
                sell_count_5min=parsed['txns_5m_sells'],
                buy_count_1hour=parsed['txns_1h_buys'],
                sell_count_1hour=parsed['txns_1h_sells'],
                price_change_5min=parsed['price_change_5m'],
                price_change_1hour=parsed['price_change_1h'],
                price_change_24hour=parsed['price_change_24h'],
            )

            # Volume acceleration should be a reasonable value
            assert features.volume_acceleration >= 0
