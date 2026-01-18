"""Base strategy class for trading strategies"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExitReason(Enum):
    """Reasons for exiting a position"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"
    SIGNAL_EXIT = "signal_exit"


@dataclass
class Signal:
    """Trading signal from a strategy"""
    signal_type: SignalType
    token_address: str
    token_symbol: str
    chain_id: str
    strategy_name: str
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_pct: float  # Percentage of capital to use
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)
        if risk == 0:
            return 0
        return reward / risk


@dataclass
class Position:
    """Active trading position"""
    id: str
    token_address: str
    token_symbol: str
    chain_id: str
    strategy_name: str
    entry_price: float
    current_price: float
    size_usd: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    entry_time: datetime
    ml_risk_score: float
    partial_exits: int = 0  # Track partial take profits
    trailing_stop: Optional[float] = None

    @property
    def pnl_usd(self) -> float:
        """Calculate unrealized P&L in USD"""
        price_change_pct = (self.current_price - self.entry_price) / self.entry_price
        return self.size_usd * price_change_pct

    @property
    def pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage"""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def hold_time_hours(self) -> float:
        """Calculate how long position has been held"""
        delta = datetime.utcnow() - self.entry_time
        return delta.total_seconds() / 3600

    def should_stop_loss(self) -> bool:
        """Check if stop loss should trigger"""
        effective_stop = self.trailing_stop if self.trailing_stop else self.stop_loss
        return self.current_price <= effective_stop

    def should_take_profit_1(self) -> bool:
        """Check if first take profit level reached"""
        return self.current_price >= self.take_profit_1 and self.partial_exits == 0

    def should_take_profit_2(self) -> bool:
        """Check if second take profit level reached"""
        return self.current_price >= self.take_profit_2 and self.partial_exits == 1

    def update_trailing_stop(self, new_price: float, trail_pct: float = 0.10):
        """Update trailing stop based on new high"""
        if new_price > self.entry_price:
            new_trailing = new_price * (1 - trail_pct)
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(
        self,
        name: str,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.15,
        take_profit_1_pct: float = 0.30,
        take_profit_2_pct: float = 0.50,
        max_positions: int = 3,
        min_confidence: float = 0.60
    ):
        """
        Initialize strategy

        Args:
            name: Strategy identifier
            max_position_pct: Maximum capital per position (default 10%)
            stop_loss_pct: Stop loss percentage (default 15%)
            take_profit_1_pct: First take profit target (default 30%)
            take_profit_2_pct: Second take profit target (default 50%)
            max_positions: Maximum concurrent positions
            min_confidence: Minimum confidence to generate signal
        """
        self.name = name
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_1_pct = take_profit_1_pct
        self.take_profit_2_pct = take_profit_2_pct
        self.max_positions = max_positions
        self.min_confidence = min_confidence

        # Track strategy state
        self.active_positions: List[Position] = []
        self.signals_generated: int = 0
        self.signals_acted_on: int = 0

    @abstractmethod
    def analyze(self, token_data: Dict) -> Optional[Signal]:
        """
        Analyze token data and generate trading signal

        Args:
            token_data: Dict containing token features and market data

        Returns:
            Signal if entry conditions met, None otherwise
        """
        pass

    @abstractmethod
    def should_exit(self, position: Position, current_data: Dict) -> Tuple[bool, ExitReason]:
        """
        Determine if position should be exited

        Args:
            position: Current position
            current_data: Current market data for the token

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        pass

    def calculate_position_size(
        self,
        capital: float,
        confidence: float,
        ml_risk_score: float
    ) -> float:
        """
        Calculate position size based on confidence and risk

        Args:
            capital: Available capital
            confidence: Signal confidence (0-1)
            ml_risk_score: ML model risk score (0-1, higher = riskier)

        Returns:
            Position size in USD
        """
        # Base size from max position percentage
        base_size = capital * self.max_position_pct

        # Adjust for confidence (scale 0.5-1.0)
        confidence_factor = 0.5 + (confidence * 0.5)

        # Adjust for ML risk (reduce size for higher risk)
        risk_factor = 1 - (ml_risk_score * 0.5)  # 0.5-1.0

        position_size = base_size * confidence_factor * risk_factor

        return min(position_size, capital * 0.20)  # Never exceed 20% of capital

    def calculate_stops(
        self,
        entry_price: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Entry price
            volatility: Optional volatility measure for dynamic stops

        Returns:
            Tuple of (stop_loss, take_profit_1, take_profit_2)
        """
        # Adjust for volatility if provided
        vol_factor = 1.0
        if volatility and volatility > 0:
            vol_factor = min(1.5, max(0.7, volatility / 100))

        stop_loss = entry_price * (1 - self.stop_loss_pct * vol_factor)
        take_profit_1 = entry_price * (1 + self.take_profit_1_pct * vol_factor)
        take_profit_2 = entry_price * (1 + self.take_profit_2_pct * vol_factor)

        return stop_loss, take_profit_1, take_profit_2

    def create_signal(
        self,
        token_data: Dict,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> Signal:
        """
        Create a trading signal from token data

        Args:
            token_data: Token market data
            confidence: Signal confidence (0-1)
            metadata: Additional signal metadata

        Returns:
            Trading signal
        """
        entry_price = token_data.get('price_usd', 0)
        volatility = abs(token_data.get('price_change_24h', 0))

        stop_loss, tp1, tp2 = self.calculate_stops(entry_price, volatility)

        signal = Signal(
            signal_type=SignalType.BUY,
            token_address=token_data.get('token_address', ''),
            token_symbol=token_data.get('token_symbol', ''),
            chain_id=token_data.get('chain_id', ''),
            strategy_name=self.name,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            position_size_pct=self.max_position_pct * confidence,
            metadata=metadata or {}
        )

        self.signals_generated += 1
        return signal

    def validate_entry_conditions(self, token_data: Dict) -> bool:
        """
        Validate basic entry conditions are met

        Args:
            token_data: Token market data

        Returns:
            True if basic conditions met
        """
        # Check minimum liquidity
        liquidity = token_data.get('liquidity_usd', 0)
        if liquidity < 5000:
            return False

        # Check minimum market cap
        market_cap = token_data.get('market_cap', 0)
        if market_cap < 10000:
            return False

        # Check token age (avoid very new tokens)
        age_hours = token_data.get('token_age_hours', 0)
        if age_hours < 2:  # At least 2 hours old
            return False

        # Check we're not at max positions
        if len(self.active_positions) >= self.max_positions:
            return False

        return True

    def get_statistics(self) -> Dict:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'signals_generated': self.signals_generated,
            'signals_acted_on': self.signals_acted_on,
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'conversion_rate': self.signals_acted_on / self.signals_generated if self.signals_generated > 0 else 0
        }
