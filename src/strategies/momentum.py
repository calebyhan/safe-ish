"""Momentum Breakout Strategy for meme coin trading"""
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType, Position, ExitReason


class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy

    Entry Conditions:
    - Volume spike (5m volume > 2x 1h average)
    - Positive price momentum (5m change > +5%)
    - Buy pressure (buy/sell ratio > 1.5)
    - ML filter passed (risk score < 0.40)

    Exit Conditions:
    - Stop loss at -15%
    - Take profit 1 at +30% (exit 50%)
    - Take profit 2 at +50% (exit remaining)
    - Time exit after 4 hours
    - Trailing stop after TP1 hit
    """

    def __init__(
        self,
        # Entry thresholds
        volume_spike_multiplier: float = 2.0,
        min_price_momentum_pct: float = 5.0,
        min_buy_sell_ratio: float = 1.5,
        max_ml_risk_score: float = 0.40,
        # Position management
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.15,
        take_profit_1_pct: float = 0.30,
        take_profit_2_pct: float = 0.50,
        max_positions: int = 3,
        time_exit_hours: float = 4.0,
        trailing_stop_pct: float = 0.10,
        min_confidence: float = 0.60
    ):
        """
        Initialize Momentum Breakout Strategy

        Args:
            volume_spike_multiplier: Required volume spike ratio
            min_price_momentum_pct: Minimum 5m price change percentage
            min_buy_sell_ratio: Minimum buy/sell ratio
            max_ml_risk_score: Maximum acceptable ML risk score
            max_position_pct: Maximum position size as % of capital
            stop_loss_pct: Stop loss percentage
            take_profit_1_pct: First take profit percentage
            take_profit_2_pct: Second take profit percentage
            max_positions: Maximum concurrent positions
            time_exit_hours: Maximum hold time before forced exit
            trailing_stop_pct: Trailing stop percentage after TP1
            min_confidence: Minimum confidence to generate signal
        """
        super().__init__(
            name="momentum_breakout",
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_1_pct=take_profit_1_pct,
            take_profit_2_pct=take_profit_2_pct,
            max_positions=max_positions,
            min_confidence=min_confidence
        )

        self.volume_spike_multiplier = volume_spike_multiplier
        self.min_price_momentum_pct = min_price_momentum_pct
        self.min_buy_sell_ratio = min_buy_sell_ratio
        self.max_ml_risk_score = max_ml_risk_score
        self.time_exit_hours = time_exit_hours
        self.trailing_stop_pct = trailing_stop_pct

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """
        Analyze token for momentum breakout entry

        Args:
            token_data: Token market data including:
                - price_usd, price_change_5m, price_change_1h
                - volume_5m, volume_1h
                - buy_count_5m, sell_count_5m
                - liquidity_usd, market_cap
                - ml_risk_score (from ML filter)

        Returns:
            Signal if entry conditions met, None otherwise
        """
        # Basic entry validation
        if not self.validate_entry_conditions(token_data):
            return None

        # Check ML risk score
        ml_risk = token_data.get('ml_risk_score', 1.0)
        if ml_risk > self.max_ml_risk_score:
            return None

        # Calculate momentum metrics
        confidence, reasons = self._calculate_momentum_score(token_data)

        # Check minimum confidence
        if confidence < self.min_confidence:
            return None

        # Create and return signal
        return self.create_signal(
            token_data=token_data,
            confidence=confidence,
            metadata={
                'entry_reasons': reasons,
                'ml_risk_score': ml_risk,
                'volume_spike': token_data.get('volume_spike_ratio', 0),
                'buy_sell_ratio': token_data.get('buy_sell_ratio_5min', 0)
            }
        )

    def _calculate_momentum_score(self, token_data: Dict) -> Tuple[float, list]:
        """
        Calculate momentum score and entry reasons

        Returns:
            Tuple of (confidence_score, list_of_reasons)
        """
        score = 0.0
        max_score = 0.0
        reasons = []

        # 1. Volume Spike (25% weight)
        volume_5m = token_data.get('volume_5min', 0)
        volume_1h = token_data.get('volume_1hour', 0)

        if volume_1h > 0:
            # Normalize 1h volume to 5m equivalent
            volume_1h_per_5m = volume_1h / 12
            volume_spike = volume_5m / volume_1h_per_5m if volume_1h_per_5m > 0 else 0

            if volume_spike >= self.volume_spike_multiplier:
                spike_score = min(1.0, volume_spike / (self.volume_spike_multiplier * 2))
                score += 0.25 * spike_score
                reasons.append(f"Volume spike: {volume_spike:.1f}x")

            max_score += 0.25

        # 2. Price Momentum (30% weight)
        price_change_5m = token_data.get('price_change_5min', 0)

        if price_change_5m >= self.min_price_momentum_pct:
            momentum_score = min(1.0, price_change_5m / (self.min_price_momentum_pct * 3))
            score += 0.30 * momentum_score
            reasons.append(f"Price momentum: +{price_change_5m:.1f}%")

        max_score += 0.30

        # 3. Buy Pressure (25% weight)
        buy_count = token_data.get('buy_count_5min', 0)
        sell_count = token_data.get('sell_count_5min', 1)

        buy_sell_ratio = buy_count / sell_count if sell_count > 0 else buy_count

        if buy_sell_ratio >= self.min_buy_sell_ratio:
            pressure_score = min(1.0, buy_sell_ratio / (self.min_buy_sell_ratio * 2))
            score += 0.25 * pressure_score
            reasons.append(f"Buy pressure: {buy_sell_ratio:.1f}x")

        max_score += 0.25

        # 4. Liquidity & Market Cap (20% weight)
        liquidity = token_data.get('liquidity_usd', 0)
        market_cap = token_data.get('market_cap', 0)

        liq_score = 0.0
        if liquidity >= 50000:
            liq_score = 1.0
            reasons.append(f"Strong liquidity: ${liquidity:,.0f}")
        elif liquidity >= 20000:
            liq_score = 0.7
            reasons.append(f"Good liquidity: ${liquidity:,.0f}")
        elif liquidity >= 10000:
            liq_score = 0.4

        mcap_score = 0.0
        if market_cap >= 500000:
            mcap_score = 1.0
        elif market_cap >= 100000:
            mcap_score = 0.6
        elif market_cap >= 50000:
            mcap_score = 0.3

        score += 0.20 * ((liq_score + mcap_score) / 2)
        max_score += 0.20

        # Normalize score
        confidence = score / max_score if max_score > 0 else 0

        return confidence, reasons

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """
        Check exit conditions for position

        Args:
            position: Current position
            current_data: Current market data

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Update current price
        position.current_price = current_data.get('price_usd', position.current_price)

        # 1. Check time-based exit
        if position.hold_time_hours >= self.time_exit_hours:
            return True, ExitReason.TIME_EXIT

        # 2. Check stop loss (including trailing)
        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        # 3. Check take profit levels
        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            # After TP1, activate trailing stop
            position.update_trailing_stop(position.current_price, self.trailing_stop_pct)
            return True, ExitReason.TAKE_PROFIT_1

        # 4. Update trailing stop on new highs
        if position.partial_exits > 0:
            position.update_trailing_stop(position.current_price, self.trailing_stop_pct)

        # 5. Check for momentum reversal (optional exit signal)
        if self._check_momentum_reversal(current_data):
            # Only exit on reversal if we're in profit
            if position.pnl_pct > 5:
                return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.HOLD if hasattr(ExitReason, 'HOLD') else ExitReason.MANUAL

    def _check_momentum_reversal(self, current_data: Dict) -> bool:
        """
        Check if momentum has reversed

        Args:
            current_data: Current market data

        Returns:
            True if momentum has reversed
        """
        price_change_5m = current_data.get('price_change_5min', 0)
        buy_count = current_data.get('buy_count_5min', 0)
        sell_count = current_data.get('sell_count_5min', 1)

        buy_sell_ratio = buy_count / sell_count if sell_count > 0 else buy_count

        # Reversal: negative momentum + selling pressure
        if price_change_5m < -5 and buy_sell_ratio < 0.7:
            return True

        return False

    def get_statistics(self) -> Dict:
        """Get strategy-specific statistics"""
        stats = super().get_statistics()
        stats.update({
            'volume_spike_threshold': self.volume_spike_multiplier,
            'momentum_threshold': self.min_price_momentum_pct,
            'buy_sell_threshold': self.min_buy_sell_ratio,
            'time_exit_hours': self.time_exit_hours
        })
        return stats
