"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def sample_token_data():
    """Sample token data for testing strategies."""
    return {
        'token_address': '0x123',
        'token_symbol': 'TEST',
        'chain_id': 'solana',
        'price_usd': 0.001,
        'price_change_5min': 10.0,
        'price_change_1hour': 20.0,
        'price_change_24hour': 50.0,
        'volume_5min': 5000,
        'volume_1hour': 10000,
        'volume_24hour': 50000,
        'buy_count_5min': 50,
        'sell_count_5min': 20,
        'buy_count_1hour': 200,
        'sell_count_1hour': 100,
        'liquidity_usd': 50000,
        'market_cap': 500000,
        'token_age_hours': 48,
        'ml_risk_score': 0.20
    }


@pytest.fixture
def scam_token_features():
    """Scam-like token features for ML testing."""
    return {
        'token_age_hours': 12,
        'liquidity_usd': 2000,
        'market_cap': 10000,
        'price_usd': 0.0005,
        'volume_5min': 100,
        'volume_1hour': 500,
        'volume_24hour': 5000,
        'buy_count_5min': 5,
        'sell_count_5min': 25,
        'buy_count_1hour': 30,
        'sell_count_1hour': 120,
        'price_change_5min': -30,
        'price_change_1hour': -60,
        'price_change_24hour': -85,
        'buy_sell_ratio_5min': 0.2,
        'buy_sell_ratio_1hour': 0.25,
        'liquidity_to_mcap_ratio': 0.02,
        'volume_acceleration': 1.5,
        'launch_phase_risk': 0.9,
        'safety_score': 0,
        'volume_to_liquidity_ratio': 25,
    }


@pytest.fixture
def legit_token_features():
    """Legitimate-like token features for ML testing."""
    return {
        'token_age_hours': 720,
        'liquidity_usd': 250000,
        'market_cap': 2000000,
        'price_usd': 1.5,
        'volume_5min': 5000,
        'volume_1hour': 50000,
        'volume_24hour': 500000,
        'buy_count_5min': 40,
        'sell_count_5min': 30,
        'buy_count_1hour': 200,
        'sell_count_1hour': 150,
        'price_change_5min': 5,
        'price_change_1hour': 15,
        'price_change_24hour': 35,
        'buy_sell_ratio_5min': 1.33,
        'buy_sell_ratio_1hour': 1.33,
        'liquidity_to_mcap_ratio': 0.125,
        'volume_acceleration': 1.2,
        'launch_phase_risk': 0,
        'safety_score': 4,
        'volume_to_liquidity_ratio': 2,
    }
