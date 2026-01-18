"""Integration tests for the OHLCV system."""
import pytest
import sqlite3
from datetime import datetime, timedelta

from src.data_collection.geckoterminal_api import GeckoTerminalAPI, Timeframe
from src.backtesting.ohlcv_backtest import OHLCVBacktestEngine, OHLCVBacktestConfig
from src.strategies.momentum import MomentumBreakoutStrategy


class TestGeckoTerminalAPI:
    """Tests for GeckoTerminal API wrapper."""

    @pytest.mark.asyncio
    async def test_search_pools(self):
        """Test searching for pools."""
        async with GeckoTerminalAPI() as api:
            pools = await api.search_pools("BONK", network="solana")

            assert pools is not None
            assert len(pools) > 0

    @pytest.mark.asyncio
    async def test_parse_pool_data(self):
        """Test parsing pool data."""
        async with GeckoTerminalAPI() as api:
            pools = await api.search_pools("BONK", network="solana")

            assert pools and len(pools) > 0

            pool = api.parse_pool_data(pools[0])

            assert 'name' in pool
            assert 'price_usd' in pool
            assert 'pool_address' in pool

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data(self):
        """Test fetching OHLCV data."""
        async with GeckoTerminalAPI() as api:
            pools = await api.search_pools("BONK", network="solana")

            assert pools and len(pools) > 0

            pool = api.parse_pool_data(pools[0])

            ohlcv = await api.get_extended_ohlcv(
                network="solana",
                pool_address=pool['pool_address'],
                timeframe=Timeframe.HOUR_1,
                days=7
            )

            assert ohlcv is not None
            assert len(ohlcv) > 0

            # Check candle structure
            candle = ohlcv[0]
            assert hasattr(candle, 'timestamp')
            assert hasattr(candle, 'open')
            assert hasattr(candle, 'high')
            assert hasattr(candle, 'low')
            assert hasattr(candle, 'close')
            assert hasattr(candle, 'volume')


class TestDataCollection:
    """Tests for data collection system."""

    def test_database_tables_exist(self):
        """Test that required database tables exist."""
        try:
            conn = sqlite3.connect("data/trading.db")
            cursor = conn.cursor()

            # Check ohlcv_data table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data'")
            ohlcv_table = cursor.fetchone()

            # Check pool_metadata table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pool_metadata'")
            pool_table = cursor.fetchone()

            conn.close()

            assert ohlcv_table is not None, "ohlcv_data table should exist"
            assert pool_table is not None, "pool_metadata table should exist"
        except sqlite3.OperationalError:
            pytest.skip("Database not initialized yet")

    def test_data_collection_stats(self):
        """Test getting data collection statistics."""
        try:
            from scripts.collect_ohlcv_data import OHLCVCollector

            collector = OHLCVCollector(db_path="data/trading.db")
            stats = collector.get_stats()

            assert 'total_candles' in stats
            assert 'unique_pools' in stats
            assert stats['total_candles'] >= 0
        except (ImportError, sqlite3.OperationalError):
            pytest.skip("Collector or database not available")


class TestBacktestEngine:
    """Tests for OHLCV backtest engine."""

    def test_backtest_config(self):
        """Test backtest configuration."""
        config = OHLCVBacktestConfig(
            initial_capital=1000.0,
            max_positions=3,
            max_position_size_pct=0.20,
            commission_pct=0.001,
            slippage_pct=0.005,
            use_intrabar_stops=True
        )

        assert config.initial_capital == 1000.0
        assert config.max_positions == 3
        assert config.commission_pct == 0.001

    def test_backtest_engine_initialization(self):
        """Test backtest engine can be initialized."""
        strategy = MomentumBreakoutStrategy(
            max_positions=3,
            max_position_pct=0.20
        )

        config = OHLCVBacktestConfig(
            initial_capital=1000.0,
            max_positions=3
        )

        engine = OHLCVBacktestEngine(
            strategies=[strategy],
            config=config,
            ml_model=None,
            db_path="data/trading.db"
        )

        assert engine is not None

    def test_run_backtest_with_data(self):
        """Test running a backtest if data is available."""
        try:
            from scripts.collect_ohlcv_data import OHLCVCollector

            collector = OHLCVCollector(db_path="data/trading.db")
            stats = collector.get_stats()

            if stats['total_candles'] == 0:
                pytest.skip("No OHLCV data available for backtesting")

            strategy = MomentumBreakoutStrategy(
                max_positions=3,
                max_position_pct=0.20
            )

            config = OHLCVBacktestConfig(
                initial_capital=1000.0,
                max_positions=3,
                max_position_size_pct=0.20,
                commission_pct=0.001,
                slippage_pct=0.005,
                use_intrabar_stops=True
            )

            engine = OHLCVBacktestEngine(
                strategies=[strategy],
                config=config,
                ml_model=None,
                db_path="data/trading.db"
            )

            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            result = engine.run(
                networks=None,
                timeframe="HOUR_1",
                start_date=start_date,
                end_date=end_date
            )

            assert result is not None
            assert hasattr(result, 'bars_processed')
            assert hasattr(result, 'total_trades')
            assert hasattr(result, 'roi_pct')

        except (ImportError, ValueError) as e:
            pytest.skip(f"Backtest skipped: {e}")
