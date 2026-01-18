"""Simple and aggressive technical strategies for testing OHLCV backtesting"""
from typing import Dict, Optional, Tuple

from .base import BaseStrategy, Signal, Position, ExitReason


class SimpleRSI(BaseStrategy):
    """
    Ultra-simple RSI strategy for testing

    Entry: RSI < 40 (relaxed oversold)
    Exit: Stop loss or take profit
    """

    def __init__(self):
        super().__init__(
            name="simple_rsi",
            max_position_pct=0.20,
            stop_loss_pct=0.12,
            take_profit_1_pct=0.25,
            take_profit_2_pct=0.50,
            max_positions=5,
            min_confidence=0.40  # Low threshold
        )

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """Simple RSI-based entry"""
        # Check basic conditions (liquidity, etc.)
        if not self.validate_entry_conditions(token_data):
            return None

        rsi = token_data.get('rsi_14')

        # Skip if RSI not available (NaN or None)
        if rsi is None or rsi != rsi:  # NaN check
            return None

        # Relaxed oversold condition
        if rsi < 40:
            confidence = 0.5 + (40 - rsi) / 80  # 0.5-1.0 range

            return self.create_signal(
                token_data=token_data,
                confidence=min(confidence, 0.95),
                metadata={'rsi': rsi, 'entry_reason': f'RSI {rsi:.1f} < 40'}
            )

        return None

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """Standard exits only"""
        position.current_price = current_data.get('price_usd', position.current_price)

        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            return True, ExitReason.TAKE_PROFIT_1

        # RSI overbought exit (aggressive)
        rsi = current_data.get('rsi_14', 50)
        if rsi > 65 and position.pnl_pct > 5:
            return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.MANUAL


class BuyAndHold(BaseStrategy):
    """
    Simple buy-and-hold test strategy

    Buys on any positive price momentum
    Just for testing the backtest engine
    """

    def __init__(self):
        super().__init__(
            name="buy_and_hold_test",
            max_position_pct=0.15,
            stop_loss_pct=0.15,
            take_profit_1_pct=0.20,
            take_profit_2_pct=0.40,
            max_positions=3,
            min_confidence=0.30
        )

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """Buy on any positive momentum"""
        if not self.validate_entry_conditions(token_data):
            return None

        price_change = token_data.get('price_change_1hour', 0)

        # Buy on any positive hourly change
        if price_change > 0:
            confidence = min(0.5 + price_change / 20, 0.80)

            return self.create_signal(
                token_data=token_data,
                confidence=confidence,
                metadata={'price_change': price_change}
            )

        return None

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """Standard exits"""
        position.current_price = current_data.get('price_usd', position.current_price)

        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            return True, ExitReason.TAKE_PROFIT_1

        return False, ExitReason.MANUAL
