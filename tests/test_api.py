"""Tests for API integrations."""
import pytest

from src.data_collection.dex_api import DexScreenerAPI
from src.data_collection.features import FeatureCollector, TokenFeatures


class TestDexScreenerAPI:
    """Tests for DexScreener API wrapper."""

    @pytest.mark.asyncio
    async def test_search_pairs(self):
        """Test searching for token pairs."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results is not None
            assert 'pairs' in results
            assert len(results['pairs']) > 0

    @pytest.mark.asyncio
    async def test_parse_token_data(self):
        """Test parsing token data from API response."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            pair = results['pairs'][0]
            parsed = api.parse_token_data(pair)

            # Check required fields exist
            assert 'token_address' in parsed
            assert 'token_symbol' in parsed
            assert 'chain_id' in parsed
            assert 'price_usd' in parsed
            assert 'liquidity_usd' in parsed

    @pytest.mark.asyncio
    async def test_find_solana_pair(self):
        """Test finding a Solana pair specifically."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results

            # Find a Solana pair
            solana_pair = None
            for p in results['pairs']:
                if p['chainId'] == 'solana':
                    solana_pair = p
                    break

            # BONK is primarily on Solana, so we should find one
            assert solana_pair is not None
            assert solana_pair['chainId'] == 'solana'


class TestFeatureCollector:
    """Tests for feature collection."""

    @pytest.mark.asyncio
    async def test_collect_features_from_pair(self):
        """Test collecting features from pair data directly."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            # Find Solana pair
            pair = None
            for p in results['pairs']:
                if p['chainId'] == 'solana':
                    pair = p
                    break

            if not pair:
                pair = results['pairs'][0]

            # Parse and create features from pair data directly
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

            assert features is not None
            assert features.token_address == parsed['token_address']
            assert features.token_symbol is not None

    @pytest.mark.asyncio
    async def test_feature_values_valid(self):
        """Test that collected feature values are valid."""
        async with DexScreenerAPI() as api:
            results = await api.search_pairs("BONK")

            assert results and 'pairs' in results and len(results['pairs']) > 0

            pair = results['pairs'][0]
            parsed = api.parse_token_data(pair)

            # Create features from parsed data
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

            # Check derived features are calculated
            assert features.liquidity_to_mcap_ratio >= 0
            assert features.safety_score >= 0

    @pytest.mark.asyncio
    async def test_to_ml_features(self):
        """Test converting features to ML format."""
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

            assert isinstance(ml_features, dict)
            assert len(ml_features) > 20  # Should have 30+ features
