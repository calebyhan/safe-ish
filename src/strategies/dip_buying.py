"""Dip Buying Strategy for meme coin trading"""
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType, Position, ExitReason


class DipBuyingStrategy(BaseStrategy):
    """
    Dip Buying Strategy

    Entry Conditions:
    - Significant recent dip (1h price down 20-40%)
    - Recovery signal (5m momentum turning positive)
    - Volume returning (5m volume > 1h average)
    - Strong historical pattern (was up before the dip)
    - ML filter passed (risk score < 0.40)

    Exit Conditions:
    - Stop loss at -20% (wider for dip buying)
    - Take profit 1 at +25% (exit 50%)
    - Take profit 2 at +40% (exit remaining)
    - Time exit after 6 hours
    - Trailing stop after TP1 hit
    """

    def __init__(
        self,
        # Entry thresholds
        min_dip_pct: float = 20.0,
        max_dip_pct: float = 40.0,
        min_recovery_momentum_pct: float = 2.0,
        min_volume_ratio: float = 1.0,
        max_ml_risk_score: float = 0.40,
        # Position management
        max_position_pct: float = 0.08,  # Smaller positions for dip buying
        stop_loss_pct: float = 0.20,  # Wider stop for volatility
        take_profit_1_pct: float = 0.25,
        take_profit_2_pct: float = 0.40,
        max_positions: int = 2,
        time_exit_hours: float = 6.0,
        trailing_stop_pct: float = 0.12,
        min_confidence: float = 0.55
    ):
        """
        Initialize Dip Buying Strategy

        Args:
            min_dip_pct: Minimum dip percentage to consider
            max_dip_pct: Maximum dip percentage (avoid dead tokens)
            min_recovery_momentum_pct: Minimum 5m recovery percentage
            min_volume_ratio: Minimum 5m/1h volume ratio
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
            name="dip_buying",
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_1_pct=take_profit_1_pct,
            take_profit_2_pct=take_profit_2_pct,
            max_positions=max_positions,
            min_confidence=min_confidence
        )

        self.min_dip_pct = min_dip_pct
        self.max_dip_pct = max_dip_pct
        self.min_recovery_momentum_pct = min_recovery_momentum_pct
        self.min_volume_ratio = min_volume_ratio
        self.max_ml_risk_score = max_ml_risk_score
        self.time_exit_hours = time_exit_hours
        self.trailing_stop_pct = trailing_stop_pct

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """
        Analyze token for dip buying entry

        Args:
            token_data: Token market data including:
                - price_usd, price_change_5m, price_change_1h, price_change_24h
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

        # Calculate dip buying score
        confidence, reasons = self._calculate_dip_score(token_data)

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
                'dip_pct': token_data.get('price_change_1hour', 0),
                'recovery_pct': token_data.get('price_change_5min', 0)
            }
        )

    def _calculate_dip_score(self, token_data: Dict) -> Tuple[float, list]:
        """
        Calculate dip buying score and entry reasons

        Returns:
            Tuple of (confidence_score, list_of_reasons)
        """
        score = 0.0
        max_score = 0.0
        reasons = []

        # 1. Dip Detection (30% weight)
        price_change_1h = token_data.get('price_change_1hour', 0)

        if price_change_1h < 0:
            dip_pct = abs(price_change_1h)

            if self.min_dip_pct <= dip_pct <= self.max_dip_pct:
                # Sweet spot for dip buying
                dip_score = 1.0 - abs(dip_pct - 30) / 20  # Peak at 30% dip
                dip_score = max(0.5, min(1.0, dip_score))
                score += 0.30 * dip_score
                reasons.append(f"Dip detected: -{dip_pct:.1f}%")
            elif dip_pct > self.max_dip_pct:
                # Too deep, might be dead
                return 0, []
            else:
                # Not enough dip
                return 0, []

        else:
            # No dip, not a dip buying opportunity
            return 0, []

        max_score += 0.30

        # 2. Recovery Signal (30% weight)
        price_change_5m = token_data.get('price_change_5min', 0)

        if price_change_5m >= self.min_recovery_momentum_pct:
            recovery_score = min(1.0, price_change_5m / (self.min_recovery_momentum_pct * 4))
            score += 0.30 * recovery_score
            reasons.append(f"Recovery signal: +{price_change_5m:.1f}%")
        else:
            # No recovery signal yet
            recovery_score = max(0, price_change_5m / self.min_recovery_momentum_pct)
            score += 0.30 * recovery_score * 0.5  # Partial credit

        max_score += 0.30

        # 3. Volume Returning (20% weight)
        volume_5m = token_data.get('volume_5min', 0)
        volume_1h = token_data.get('volume_1hour', 0)

        if volume_1h > 0:
            volume_1h_per_5m = volume_1h / 12
            volume_ratio = volume_5m / volume_1h_per_5m if volume_1h_per_5m > 0 else 0

            if volume_ratio >= self.min_volume_ratio:
                vol_score = min(1.0, volume_ratio / 2.0)
                score += 0.20 * vol_score
                reasons.append(f"Volume returning: {volume_ratio:.1f}x")
            else:
                vol_score = volume_ratio / self.min_volume_ratio
                score += 0.20 * vol_score * 0.5

        max_score += 0.20

        # 4. Historical Strength (20% weight)
        price_change_24h = token_data.get('price_change_24hour', 0)
        liquidity = token_data.get('liquidity_usd', 0)

        historical_score = 0.0

        # Was strong before the dip (24h still positive or only slightly negative)
        if price_change_24h > 0:
            historical_score += 0.5
            reasons.append(f"24h still positive: +{price_change_24h:.1f}%")
        elif price_change_24h > -20:
            historical_score += 0.3

        # Good liquidity indicates established token
        if liquidity >= 30000:
            historical_score += 0.5
            reasons.append(f"Strong liquidity: ${liquidity:,.0f}")
        elif liquidity >= 15000:
            historical_score += 0.3

        score += 0.20 * historical_score
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

        # 5. Check for failed recovery (dip continuing)
        if self._check_failed_recovery(position, current_data):
            if position.pnl_pct < -10:  # Cut losses early on failed recovery
                return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.MANUAL

    def _check_failed_recovery(self, position: Position, current_data: Dict) -> bool:
        """
        Check if recovery has failed and dip is continuing

        Args:
            position: Current position
            current_data: Current market data

        Returns:
            True if recovery has failed
        """
        price_change_5m = current_data.get('price_change_5min', 0)
        price_change_1h = current_data.get('price_change_1hour', 0)

        # Still falling after entry
        if price_change_5m < -5 and price_change_1h < -30:
            return True

        # No recovery after 1 hour
        if position.hold_time_hours > 1.0 and position.pnl_pct < -5:
            if price_change_5m < 0:
                return True

        return False

    def validate_entry_conditions(self, token_data: Dict) -> bool:
        """
        Override to add dip-specific validation

        Args:
            token_data: Token market data

        Returns:
            True if basic conditions met
        """
        # Call parent validation
        if not super().validate_entry_conditions(token_data):
            return False

        # For dip buying, we want slightly older tokens
        age_hours = token_data.get('token_age_hours', 0)
        if age_hours < 12:  # At least 12 hours old for dip buying
            return False

        # Need sufficient liquidity for dip buying (safety)
        liquidity = token_data.get('liquidity_usd', 0)
        if liquidity < 10000:  # Higher minimum for dip buying
            return False

        return True

    def get_statistics(self) -> Dict:
        """Get strategy-specific statistics"""
        stats = super().get_statistics()
        stats.update({
            'dip_range': f"{self.min_dip_pct}-{self.max_dip_pct}%",
            'recovery_threshold': self.min_recovery_momentum_pct,
            'volume_threshold': self.min_volume_ratio,
            'time_exit_hours': self.time_exit_hours
        })
        return stats
