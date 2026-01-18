"""Pure Technical Analysis Strategy for OHLCV backtesting"""
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType, Position, ExitReason


class TechnicalStrategy(BaseStrategy):
    """
    Pure Technical Analysis Strategy (OHLCV Compatible)

    Uses only OHLCV-derived indicators:
    - RSI (Relative Strength Index)
    - Moving Average crossovers
    - Volume analysis
    - Volatility/ATR

    Entry Conditions:
    - RSI oversold (< 30) with bullish divergence OR
    - Golden cross (SMA20 > SMA50) with volume confirmation OR
    - Price breakout above resistance with high volume

    Exit Conditions:
    - Stop loss at -12%
    - Take profit 1 at +25% (exit 50%)
    - Take profit 2 at +50% (exit remaining)
    - RSI overbought (> 70) exit signal
    - Trailing stop after TP1
    """

    def __init__(
        self,
        # RSI settings
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        # Volume settings
        min_volume_spike: float = 1.5,  # 1.5x average volume
        # Position management
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.12,
        take_profit_1_pct: float = 0.25,
        take_profit_2_pct: float = 0.50,
        max_positions: int = 4,
        trailing_stop_pct: float = 0.10,
        min_confidence: float = 0.50
    ):
        """
        Initialize Technical Analysis Strategy

        Args:
            rsi_oversold: RSI level considered oversold (buy signal)
            rsi_overbought: RSI level considered overbought (sell signal)
            min_volume_spike: Minimum volume spike ratio for entry
            max_position_pct: Maximum position size as % of capital
            stop_loss_pct: Stop loss percentage
            take_profit_1_pct: First take profit percentage
            take_profit_2_pct: Second take profit percentage
            max_positions: Maximum concurrent positions
            trailing_stop_pct: Trailing stop percentage after TP1
            min_confidence: Minimum confidence to generate signal
        """
        super().__init__(
            name="technical_analysis",
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_1_pct=take_profit_1_pct,
            take_profit_2_pct=take_profit_2_pct,
            max_positions=max_positions,
            min_confidence=min_confidence
        )

        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_volume_spike = min_volume_spike
        self.trailing_stop_pct = trailing_stop_pct

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """
        Analyze token using pure technical analysis

        Args:
            token_data: Token data including OHLCV-derived indicators:
                - price_usd, price_change_5m, price_change_1h
                - volume_1hour, volume_sma_20
                - sma_20, sma_50
                - rsi_14
                - atr_14, volatility
                - liquidity_usd, market_cap

        Returns:
            Signal if entry conditions met, None otherwise
        """
        # Skip if indicators not computed yet (early bars)
        if not self._has_required_indicators(token_data):
            return None

        # Basic entry validation
        if not self.validate_entry_conditions(token_data):
            return None

        # Calculate technical score
        confidence, reasons = self._calculate_technical_score(token_data)

        # Check minimum confidence
        if confidence < self.min_confidence:
            return None

        # Create and return signal
        return self.create_signal(
            token_data=token_data,
            confidence=confidence,
            metadata={
                'entry_reasons': reasons,
                'rsi': token_data.get('rsi_14'),
                'sma_20': token_data.get('sma_20'),
                'sma_50': token_data.get('sma_50'),
                'volume_ratio': token_data.get('volume_1hour', 0) / token_data.get('volume_sma_20', 1)
            }
        )

    def _has_required_indicators(self, token_data: Dict) -> bool:
        """Check if required technical indicators are available"""
        required = ['rsi_14', 'sma_20', 'sma_50', 'volume_sma_20']

        for indicator in required:
            value = token_data.get(indicator)
            if value is None or (isinstance(value, float) and value != value):  # None or NaN
                return False

        return True

    def _calculate_technical_score(self, token_data: Dict) -> Tuple[float, list]:
        """
        Calculate technical analysis score

        Returns:
            Tuple of (confidence_score, list_of_reasons)
        """
        score = 0.0
        max_score = 0.0
        reasons = []

        price = token_data.get('price_usd', 0)
        rsi = token_data.get('rsi_14', 50)
        sma_20 = token_data.get('sma_20', 0)
        sma_50 = token_data.get('sma_50', 0)
        volume = token_data.get('volume_1hour', 0)
        volume_sma = token_data.get('volume_sma_20', 1)

        # 1. RSI Analysis (35% weight)
        max_score += 0.35

        if rsi < self.rsi_oversold:
            # Oversold - strong buy signal
            rsi_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
            score += 0.35 * min(1.0, rsi_strength + 0.5)
            reasons.append(f"RSI oversold: {rsi:.1f}")
        elif rsi < 40:
            # Approaching oversold - moderate buy signal
            score += 0.20
            reasons.append(f"RSI low: {rsi:.1f}")

        # 2. Moving Average Analysis (30% weight)
        max_score += 0.30

        if sma_20 > 0 and sma_50 > 0:
            # Golden cross: SMA20 > SMA50
            if sma_20 > sma_50:
                crossover_strength = (sma_20 - sma_50) / sma_50
                score += 0.30 * min(1.0, crossover_strength * 10)
                reasons.append(f"Golden cross: SMA20 ${sma_20:.8f} > SMA50 ${sma_50:.8f}")

            # Price above MAs - bullish
            if price > sma_20:
                score += 0.10
                reasons.append(f"Price above SMA20")

        # 3. Volume Analysis (25% weight)
        max_score += 0.25

        if volume_sma > 0:
            volume_ratio = volume / volume_sma

            if volume_ratio >= self.min_volume_spike:
                volume_score = min(1.0, (volume_ratio - 1) / 2)
                score += 0.25 * volume_score
                reasons.append(f"Volume spike: {volume_ratio:.1f}x avg")
            elif volume_ratio >= 1.0:
                score += 0.10
                reasons.append(f"Above avg volume: {volume_ratio:.1f}x")

        # 4. Price Momentum (10% weight)
        max_score += 0.10

        price_change_1h = token_data.get('price_change_1hour', 0)

        if price_change_1h > 0:
            momentum_score = min(1.0, price_change_1h / 10)  # Normalize to 10%
            score += 0.10 * momentum_score
            if price_change_1h > 3:
                reasons.append(f"Positive momentum: +{price_change_1h:.1f}%")

        # Normalize score
        confidence = score / max_score if max_score > 0 else 0

        return confidence, reasons

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """
        Check technical exit conditions

        Args:
            position: Current position
            current_data: Current market data with technical indicators

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Update current price
        position.current_price = current_data.get('price_usd', position.current_price)

        # 1. Check stop loss (including trailing)
        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        # 2. Check take profit levels
        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            # After TP1, activate trailing stop
            position.update_trailing_stop(position.current_price, self.trailing_stop_pct)
            return True, ExitReason.TAKE_PROFIT_1

        # 3. Update trailing stop on new highs
        if position.partial_exits > 0:
            position.update_trailing_stop(position.current_price, self.trailing_stop_pct)

        # 4. Check technical exit signals
        if self._check_technical_exit(position, current_data):
            # Only exit on technical signal if in profit
            if position.pnl_pct > 5:
                return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.MANUAL

    def _check_technical_exit(self, position: Position, current_data: Dict) -> bool:
        """
        Check for technical exit signals

        Args:
            position: Current position
            current_data: Current market data

        Returns:
            True if should exit based on technicals
        """
        # RSI overbought exit
        rsi = current_data.get('rsi_14', 50)
        if rsi > self.rsi_overbought:
            return True

        # Death cross: SMA20 crosses below SMA50
        sma_20 = current_data.get('sma_20', 0)
        sma_50 = current_data.get('sma_50', 0)

        if sma_20 > 0 and sma_50 > 0:
            if sma_20 < sma_50:
                # Bearish crossover
                return True

        # Significant negative momentum
        price_change = current_data.get('price_change_1hour', 0)
        if price_change < -8:
            return True

        return False

    def get_statistics(self) -> Dict:
        """Get strategy-specific statistics"""
        stats = super().get_statistics()
        stats.update({
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'volume_spike_threshold': self.min_volume_spike,
            'trailing_stop_pct': self.trailing_stop_pct
        })
        return stats


class RSIStrategy(BaseStrategy):
    """
    Simple RSI Mean Reversion Strategy (OHLCV Compatible)

    Entry: RSI < 30 (oversold)
    Exit: RSI > 70 (overbought) or stop loss/take profit

    Pure momentum/mean reversion based on RSI only.
    """

    def __init__(
        self,
        rsi_oversold: float = 25.0,
        rsi_overbought: float = 75.0,
        max_position_pct: float = 0.12,
        stop_loss_pct: float = 0.15,
        take_profit_1_pct: float = 0.20,
        take_profit_2_pct: float = 0.40,
        max_positions: int = 3,
        min_confidence: float = 0.60
    ):
        super().__init__(
            name="rsi_mean_reversion",
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_1_pct=take_profit_1_pct,
            take_profit_2_pct=take_profit_2_pct,
            max_positions=max_positions,
            min_confidence=min_confidence
        )

        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """Analyze based on RSI only"""
        # Check basic conditions
        if not self.validate_entry_conditions(token_data):
            return None

        rsi = token_data.get('rsi_14')

        # Skip if RSI not available
        if rsi is None or (isinstance(rsi, float) and rsi != rsi):
            return None

        # Entry: RSI oversold
        if rsi < self.rsi_oversold:
            # Confidence based on how oversold
            confidence = min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold + 0.6)

            if confidence >= self.min_confidence:
                return self.create_signal(
                    token_data=token_data,
                    confidence=confidence,
                    metadata={
                        'entry_reason': f'RSI oversold: {rsi:.1f}',
                        'rsi': rsi
                    }
                )

        return None

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """Exit on RSI overbought or standard stops"""
        position.current_price = current_data.get('price_usd', position.current_price)

        # Check stops
        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            return True, ExitReason.TAKE_PROFIT_1

        # RSI overbought exit
        rsi = current_data.get('rsi_14', 50)
        if rsi > self.rsi_overbought and position.pnl_pct > 5:
            return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.MANUAL


class MovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy (OHLCV Compatible)

    Entry: Golden cross (SMA20 crosses above SMA50)
    Exit: Death cross (SMA20 crosses below SMA50) or stop loss/take profit

    Classic trend-following strategy.
    """

    def __init__(
        self,
        max_position_pct: float = 0.18,
        stop_loss_pct: float = 0.10,
        take_profit_1_pct: float = 0.30,
        take_profit_2_pct: float = 0.60,
        max_positions: int = 3,
        min_confidence: float = 0.55
    ):
        super().__init__(
            name="ma_crossover",
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_1_pct=take_profit_1_pct,
            take_profit_2_pct=take_profit_2_pct,
            max_positions=max_positions,
            min_confidence=min_confidence
        )

    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """Analyze based on MA crossover"""
        if not self.validate_entry_conditions(token_data):
            return None

        sma_20 = token_data.get('sma_20')
        sma_50 = token_data.get('sma_50')
        price = token_data.get('price_usd', 0)

        # Skip if MAs not available
        if not sma_20 or not sma_50:
            return None

        # Golden cross: SMA20 > SMA50
        if sma_20 > sma_50:
            # Strength of crossover
            crossover_pct = ((sma_20 - sma_50) / sma_50) * 100

            # Additional confirmation: price above both MAs
            price_above_mas = price > sma_20 and price > sma_50

            if price_above_mas:
                confidence = min(0.9, 0.55 + crossover_pct * 2)
            else:
                confidence = min(0.7, 0.50 + crossover_pct * 2)

            if confidence >= self.min_confidence:
                return self.create_signal(
                    token_data=token_data,
                    confidence=confidence,
                    metadata={
                        'entry_reason': f'Golden cross: {crossover_pct:.2f}%',
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'price_above_mas': price_above_mas
                    }
                )

        return None

    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """Exit on death cross or standard stops"""
        position.current_price = current_data.get('price_usd', position.current_price)

        # Check stops
        if position.should_stop_loss():
            return True, ExitReason.STOP_LOSS

        if position.should_take_profit_2():
            return True, ExitReason.TAKE_PROFIT_2

        if position.should_take_profit_1():
            return True, ExitReason.TAKE_PROFIT_1

        # Death cross exit
        sma_20 = current_data.get('sma_20', 0)
        sma_50 = current_data.get('sma_50', 0)

        if sma_20 > 0 and sma_50 > 0:
            if sma_20 < sma_50 and position.pnl_pct > 3:
                return True, ExitReason.SIGNAL_EXIT

        return False, ExitReason.MANUAL
