"""Tests for trading components."""
from datetime import datetime

import pytest

from src.strategies.base import Signal, SignalType, Position, ExitReason
from src.strategies.momentum import MomentumBreakoutStrategy
from src.strategies.dip_buying import DipBuyingStrategy
from src.trading.portfolio import PortfolioManager
from src.trading.risk import RiskManager, RiskConfig


class TestMomentumStrategy:
    """Tests for momentum breakout strategy."""

    def test_strong_momentum_generates_signal(self):
        """Strong momentum conditions should generate a buy signal."""
        strategy = MomentumBreakoutStrategy()

        strong_momentum = {
            'token_address': '0x123',
            'token_symbol': 'MOON',
            'chain_id': 'solana',
            'price_usd': 0.001,
            'price_change_5min': 15.0,
            'price_change_1hour': 25.0,
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

        signal = strategy.analyze(strong_momentum)

        assert signal is not None
        assert signal.token_symbol == 'MOON'
        assert signal.confidence > 0
        assert signal.entry_price == 0.001
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit_1 > signal.entry_price

    def test_weak_momentum_no_signal(self):
        """Weak momentum should not generate a signal."""
        strategy = MomentumBreakoutStrategy()

        weak_momentum = {
            'token_address': '0x456',
            'token_symbol': 'WEAK',
            'chain_id': 'solana',
            'price_usd': 0.001,
            'price_change_5min': 1.0,
            'price_change_1hour': 2.0,
            'price_change_24hour': 5.0,
            'volume_5min': 100,
            'volume_1hour': 5000,
            'volume_24hour': 20000,
            'buy_count_5min': 5,
            'sell_count_5min': 5,
            'buy_count_1hour': 50,
            'sell_count_1hour': 50,
            'liquidity_usd': 5000,
            'market_cap': 10000,
            'token_age_hours': 1,
            'ml_risk_score': 0.20
        }

        signal = strategy.analyze(weak_momentum)
        assert signal is None

    def test_high_ml_risk_rejected(self):
        """High ML risk score should reject the token."""
        strategy = MomentumBreakoutStrategy()

        high_risk = {
            'token_address': '0x123',
            'token_symbol': 'RISKY',
            'chain_id': 'solana',
            'price_usd': 0.001,
            'price_change_5min': 15.0,
            'price_change_1hour': 25.0,
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
            'ml_risk_score': 0.70  # High risk
        }

        signal = strategy.analyze(high_risk)
        assert signal is None


class TestDipBuyingStrategy:
    """Tests for dip buying strategy."""

    def test_good_dip_generates_signal(self):
        """Good dip with recovery should generate a signal."""
        strategy = DipBuyingStrategy()

        good_dip = {
            'token_address': '0x789',
            'token_symbol': 'DIP',
            'chain_id': 'ethereum',
            'price_usd': 0.05,
            'price_change_5min': 5.0,
            'price_change_1hour': -25.0,
            'price_change_24hour': 10.0,
            'volume_5min': 3000,
            'volume_1hour': 20000,
            'volume_24hour': 100000,
            'buy_count_5min': 30,
            'sell_count_5min': 20,
            'buy_count_1hour': 150,
            'sell_count_1hour': 200,
            'liquidity_usd': 40000,
            'market_cap': 400000,
            'token_age_hours': 72,
            'ml_risk_score': 0.25
        }

        signal = strategy.analyze(good_dip)

        assert signal is not None
        assert signal.token_symbol == 'DIP'

    def test_no_dip_no_signal(self):
        """No dip (positive 1h) should not generate a signal."""
        strategy = DipBuyingStrategy()

        no_dip = {
            'token_address': '0x789',
            'token_symbol': 'NODIP',
            'chain_id': 'ethereum',
            'price_usd': 0.05,
            'price_change_5min': 5.0,
            'price_change_1hour': 10.0,  # No dip
            'price_change_24hour': 10.0,
            'volume_5min': 3000,
            'volume_1hour': 20000,
            'volume_24hour': 100000,
            'buy_count_5min': 30,
            'sell_count_5min': 20,
            'buy_count_1hour': 150,
            'sell_count_1hour': 200,
            'liquidity_usd': 40000,
            'market_cap': 400000,
            'token_age_hours': 72,
            'ml_risk_score': 0.25
        }

        signal = strategy.analyze(no_dip)
        assert signal is None

    def test_too_deep_dip_rejected(self):
        """Too deep dip should be rejected."""
        strategy = DipBuyingStrategy()

        dead_dip = {
            'token_address': '0x789',
            'token_symbol': 'DEAD',
            'chain_id': 'ethereum',
            'price_usd': 0.05,
            'price_change_5min': 5.0,
            'price_change_1hour': -60.0,  # Too deep
            'price_change_24hour': 10.0,
            'volume_5min': 3000,
            'volume_1hour': 20000,
            'volume_24hour': 100000,
            'buy_count_5min': 30,
            'sell_count_5min': 20,
            'buy_count_1hour': 150,
            'sell_count_1hour': 200,
            'liquidity_usd': 40000,
            'market_cap': 400000,
            'token_age_hours': 72,
            'ml_risk_score': 0.25
        }

        signal = strategy.analyze(dead_dip)
        assert signal is None


class TestPortfolioManager:
    """Tests for portfolio manager."""

    def test_open_position(self):
        """Test opening a position."""
        portfolio = PortfolioManager(
            initial_capital=1000.0,
            max_positions=3,
            paper_trading=True
        )

        signal = Signal(
            signal_type=SignalType.BUY,
            token_address='0xtest',
            token_symbol='TEST',
            chain_id='solana',
            strategy_name='momentum_breakout',
            confidence=0.80,
            entry_price=0.001,
            stop_loss=0.00085,
            take_profit_1=0.0013,
            take_profit_2=0.0015,
            position_size_pct=0.10
        )

        position = portfolio.open_position(signal, ml_risk_score=0.20)

        assert position is not None
        assert position.token_symbol == 'TEST'
        # Position size is dynamically calculated based on confidence and risk
        assert position.size_usd > 0
        assert position.size_usd <= 200  # Max 20% of 1000
        assert portfolio.capital == 1000.0 - position.size_usd

    def test_close_position_with_profit(self):
        """Test closing a position with profit."""
        portfolio = PortfolioManager(
            initial_capital=1000.0,
            max_positions=3,
            paper_trading=True
        )

        signal = Signal(
            signal_type=SignalType.BUY,
            token_address='0xtest',
            token_symbol='TEST',
            chain_id='solana',
            strategy_name='momentum_breakout',
            confidence=0.80,
            entry_price=0.001,
            stop_loss=0.00085,
            take_profit_1=0.0013,
            take_profit_2=0.0015,
            position_size_pct=0.10
        )

        position = portfolio.open_position(signal, ml_risk_score=0.20)

        # Update price to simulate profit
        portfolio.update_position_price(position.id, 0.0012)

        # Close position
        trade = portfolio.close_position(position.id, ExitReason.TAKE_PROFIT_1, 0.0012)

        assert trade is not None
        assert trade.pnl_usd > 0
        assert trade.exit_reason == ExitReason.TAKE_PROFIT_1

    def test_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        portfolio = PortfolioManager(
            initial_capital=1000.0,
            paper_trading=True
        )

        stats = portfolio.get_statistics()

        assert 'initial_capital' in stats
        assert 'current_capital' in stats
        assert 'unrealized_pnl' in stats


class TestRiskManager:
    """Tests for risk manager."""

    def test_can_trade_initially(self):
        """Should be able to trade initially."""
        portfolio = PortfolioManager(initial_capital=1000.0, paper_trading=True)
        risk_config = RiskConfig(
            max_drawdown_pct=30.0,
            max_daily_loss_pct=10.0,
            max_positions=3
        )
        risk_manager = RiskManager(portfolio, risk_config)

        can_trade, reason = risk_manager.check_can_trade()

        assert can_trade is True

    def test_validate_good_signal(self):
        """Good signal should pass validation."""
        portfolio = PortfolioManager(initial_capital=1000.0, paper_trading=True)
        risk_config = RiskConfig(
            max_drawdown_pct=30.0,
            max_daily_loss_pct=10.0,
            max_positions=3
        )
        risk_manager = RiskManager(portfolio, risk_config)

        good_signal = {
            'position_size_pct': 0.10,
            'stop_loss_pct': 0.15,
            'chain_id': 'solana',
            'ml_risk_score': 0.20
        }

        is_valid, reason = risk_manager.validate_signal(good_signal)

        assert is_valid is True

    def test_reject_high_ml_risk(self):
        """High ML risk should be rejected."""
        portfolio = PortfolioManager(initial_capital=1000.0, paper_trading=True)
        risk_config = RiskConfig(
            max_drawdown_pct=30.0,
            max_daily_loss_pct=10.0,
            max_positions=3
        )
        risk_manager = RiskManager(portfolio, risk_config)

        bad_signal = {
            'position_size_pct': 0.10,
            'stop_loss_pct': 0.15,
            'chain_id': 'solana',
            'ml_risk_score': 0.80
        }

        is_valid, reason = risk_manager.validate_signal(bad_signal)

        assert is_valid is False

    def test_record_trade_result(self):
        """Test recording trade results."""
        portfolio = PortfolioManager(initial_capital=1000.0, paper_trading=True)
        risk_config = RiskConfig(
            max_drawdown_pct=30.0,
            max_daily_loss_pct=10.0,
            max_positions=3
        )
        risk_manager = RiskManager(portfolio, risk_config)

        risk_manager.record_trade_result(50.0, True)  # Win
        risk_manager.record_trade_result(-30.0, False)  # Loss

        metrics = risk_manager.get_risk_metrics()

        assert metrics['daily_trades'] == 2
        assert metrics['daily_pnl'] == 20.0


class TestPositionExitConditions:
    """Tests for position exit conditions."""

    def test_stop_loss_triggered(self):
        """Stop loss should trigger when price drops below threshold."""
        strategy = MomentumBreakoutStrategy()

        position = Position(
            id='test123',
            token_address='0xtest',
            token_symbol='TEST',
            chain_id='solana',
            strategy_name='momentum_breakout',
            entry_price=0.001,
            current_price=0.001,
            size_usd=100.0,
            stop_loss=0.00085,
            take_profit_1=0.0013,
            take_profit_2=0.0015,
            entry_time=datetime.utcnow(),
            ml_risk_score=0.20
        )

        position.current_price = 0.0008  # Below stop
        should_exit, reason = strategy.should_exit(position, {'price_usd': 0.0008})

        assert should_exit is True
        assert reason == ExitReason.STOP_LOSS

    def test_take_profit_1_triggered(self):
        """Take profit 1 should trigger when price rises above threshold."""
        strategy = MomentumBreakoutStrategy()

        position = Position(
            id='test123',
            token_address='0xtest',
            token_symbol='TEST',
            chain_id='solana',
            strategy_name='momentum_breakout',
            entry_price=0.001,
            current_price=0.001,
            size_usd=100.0,
            stop_loss=0.00085,
            take_profit_1=0.0013,
            take_profit_2=0.0015,
            entry_time=datetime.utcnow(),
            ml_risk_score=0.20
        )

        position.current_price = 0.0014  # Above TP1
        should_exit, reason = strategy.should_exit(position, {'price_usd': 0.0014})

        assert should_exit is True
        assert reason == ExitReason.TAKE_PROFIT_1

    def test_take_profit_2_after_partial_exit(self):
        """Take profit 2 should trigger after partial exit at TP1."""
        strategy = MomentumBreakoutStrategy()

        position = Position(
            id='test123',
            token_address='0xtest',
            token_symbol='TEST',
            chain_id='solana',
            strategy_name='momentum_breakout',
            entry_price=0.001,
            current_price=0.001,
            size_usd=100.0,
            stop_loss=0.00085,
            take_profit_1=0.0013,
            take_profit_2=0.0015,
            entry_time=datetime.utcnow(),
            ml_risk_score=0.20
        )

        position.partial_exits = 1  # Already hit TP1
        position.current_price = 0.0016  # Above TP2
        should_exit, reason = strategy.should_exit(position, {'price_usd': 0.0016})

        assert should_exit is True
        assert reason == ExitReason.TAKE_PROFIT_2
