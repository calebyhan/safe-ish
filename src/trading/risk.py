"""Risk Manager for trading safety controls"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .portfolio import PortfolioManager
from ..utils.logger import get_logger


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Drawdown limits
    max_drawdown_pct: float = 30.0  # Stop trading if down 30%
    warning_drawdown_pct: float = 20.0  # Warning at 20%

    # Daily limits
    max_daily_loss_pct: float = 10.0  # Stop for day if down 10%
    max_daily_trades: int = 20  # Max trades per day

    # Per-trade limits
    max_loss_per_trade_pct: float = 15.0  # Max loss per trade
    max_position_size_pct: float = 20.0  # Max single position

    # Concentration limits
    max_positions: int = 5
    max_per_chain_pct: float = 50.0  # Max exposure per chain

    # Time-based limits
    max_hold_time_hours: float = 12.0
    cooldown_after_loss_minutes: int = 5

    # Recovery settings
    recovery_required_pct: float = 5.0  # Need 5% recovery to resume


class RiskManager:
    """
    Manages trading risk and circuit breakers

    Responsibilities:
    - Monitor portfolio drawdown
    - Enforce daily loss limits
    - Manage circuit breakers
    - Validate trade safety
    - Track risk metrics
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        config: Optional[RiskConfig] = None
    ):
        """
        Initialize risk manager

        Args:
            portfolio: Portfolio manager instance
            config: Risk configuration
        """
        self.portfolio = portfolio
        self.config = config or RiskConfig()

        # Logger
        self.logger = get_logger("RiskManager")

        # State tracking
        self.circuit_breaker_state = CircuitBreakerState.NORMAL
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.day_start = datetime.utcnow().date()

        # Risk events
        self.risk_events: List[Dict] = []

    def check_can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        # Reset daily counters if new day
        self._check_day_reset()

        # Check circuit breaker
        if self.circuit_breaker_state == CircuitBreakerState.TRIGGERED:
            return False, f"Circuit breaker triggered at {self.circuit_breaker_triggered_at}"

        # Check drawdown
        drawdown = self.portfolio.get_drawdown()
        if drawdown >= self.config.max_drawdown_pct:
            self._trigger_circuit_breaker("max_drawdown")
            return False, f"Max drawdown exceeded: {drawdown:.1f}%"

        # Check daily loss
        if self._check_daily_loss_limit():
            return False, f"Daily loss limit exceeded: ${self.daily_pnl:.2f}"

        # Check daily trade count
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"

        # Check cooldown after loss
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(minutes=self.config.cooldown_after_loss_minutes)
            if datetime.utcnow() < cooldown_end:
                remaining = (cooldown_end - datetime.utcnow()).seconds
                return False, f"Loss cooldown: {remaining}s remaining"

        # Check position limits
        if len(self.portfolio.positions) >= self.config.max_positions:
            return False, f"Max positions reached: {len(self.portfolio.positions)}"

        return True, "Trading allowed"

    def validate_signal(self, signal_data: Dict) -> Tuple[bool, str]:
        """
        Validate a trading signal against risk rules

        Args:
            signal_data: Signal data including:
                - position_size_pct: Requested position size
                - stop_loss_pct: Stop loss percentage
                - chain_id: Blockchain ID
                - ml_risk_score: ML model risk score

        Returns:
            Tuple of (is_valid, reason_if_not)
        """
        # Check basic trading permission
        can_trade, reason = self.check_can_trade()
        if not can_trade:
            return False, reason

        # Check position size
        position_size_pct = signal_data.get('position_size_pct', 0)
        if position_size_pct > self.config.max_position_size_pct:
            return False, f"Position size too large: {position_size_pct:.1f}%"

        # Check potential loss
        stop_loss_pct = signal_data.get('stop_loss_pct', 0.15)
        potential_loss_pct = position_size_pct * stop_loss_pct * 100

        if potential_loss_pct > self.config.max_loss_per_trade_pct:
            return False, f"Potential loss too high: {potential_loss_pct:.1f}%"

        # Check chain concentration
        chain_id = signal_data.get('chain_id', '')
        chain_exposure = self._calculate_chain_exposure(chain_id)
        if chain_exposure > self.config.max_per_chain_pct:
            return False, f"Chain concentration too high: {chain_id} at {chain_exposure:.1f}%"

        # Check ML risk score
        ml_risk = signal_data.get('ml_risk_score', 0)
        if ml_risk > 0.40:  # Hard limit
            return False, f"ML risk score too high: {ml_risk:.2f}"

        return True, "Signal validated"

    def record_trade_result(self, pnl_usd: float, is_win: bool):
        """
        Record a trade result for risk tracking

        Args:
            pnl_usd: Trade P&L in USD
            is_win: Whether the trade was profitable
        """
        self.daily_trades += 1
        self.daily_pnl += pnl_usd

        if not is_win:
            self.last_loss_time = datetime.utcnow()

        # Check if we hit daily limit
        if self._check_daily_loss_limit():
            self._record_risk_event("daily_loss_limit", {
                'daily_pnl': self.daily_pnl,
                'trades': self.daily_trades
            })

        # Update drawdown state
        drawdown = self.portfolio.get_drawdown()
        if drawdown >= self.config.warning_drawdown_pct:
            if self.circuit_breaker_state == CircuitBreakerState.NORMAL:
                self.circuit_breaker_state = CircuitBreakerState.WARNING
                self._record_risk_event("drawdown_warning", {'drawdown': drawdown})

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        initial = self.portfolio.initial_capital
        daily_loss_limit = initial * (self.config.max_daily_loss_pct / 100)
        return self.daily_pnl < -daily_loss_limit

    def _calculate_chain_exposure(self, chain_id: str) -> float:
        """Calculate exposure to a specific chain as percentage of portfolio"""
        chain_value = 0.0
        for pos in self.portfolio.positions.values():
            if pos.chain_id == chain_id:
                chain_value += pos.size_usd + pos.pnl_usd

        total_value = self.portfolio.get_portfolio_value()
        if total_value == 0:
            return 0

        return (chain_value / total_value) * 100

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger the circuit breaker"""
        self.circuit_breaker_state = CircuitBreakerState.TRIGGERED
        self.circuit_breaker_triggered_at = datetime.utcnow()

        self._record_risk_event("circuit_breaker_triggered", {
            'reason': reason,
            'drawdown': self.portfolio.get_drawdown(),
            'portfolio_value': self.portfolio.get_portfolio_value()
        })

        self.logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        self.logger.critical(f"Trading halted at {self.circuit_breaker_triggered_at}")

    def reset_circuit_breaker(self, force: bool = False) -> Tuple[bool, str]:
        """
        Attempt to reset circuit breaker

        Args:
            force: Force reset without recovery check

        Returns:
            Tuple of (success, message)
        """
        if self.circuit_breaker_state != CircuitBreakerState.TRIGGERED:
            return True, "Circuit breaker not triggered"

        if not force:
            # Check if portfolio has recovered
            current_value = self.portfolio.get_portfolio_value()
            trigger_value = self.portfolio.initial_capital * (1 - self.config.max_drawdown_pct / 100)
            required_recovery = trigger_value * (1 + self.config.recovery_required_pct / 100)

            if current_value < required_recovery:
                return False, f"Need {self.config.recovery_required_pct}% recovery to reset"

        self.circuit_breaker_state = CircuitBreakerState.NORMAL
        self.circuit_breaker_triggered_at = None

        self._record_risk_event("circuit_breaker_reset", {
            'forced': force
        })

        return True, "Circuit breaker reset"

    def _check_day_reset(self):
        """Reset daily counters if new day"""
        today = datetime.utcnow().date()
        if today != self.day_start:
            self.day_start = today
            self.daily_trades = 0
            self.daily_pnl = 0.0

    def _record_risk_event(self, event_type: str, data: Dict):
        """Record a risk event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'data': data
        }
        self.risk_events.append(event)

        # Keep only last 100 events
        if len(self.risk_events) > 100:
            self.risk_events = self.risk_events[-100:]

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        drawdown = self.portfolio.get_drawdown()

        return {
            'circuit_breaker_state': self.circuit_breaker_state.value,
            'drawdown_pct': drawdown,
            'drawdown_limit': self.config.max_drawdown_pct,
            'drawdown_warning': self.config.warning_drawdown_pct,
            'daily_trades': self.daily_trades,
            'daily_trades_limit': self.config.max_daily_trades,
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.portfolio.initial_capital * (self.config.max_daily_loss_pct / 100),
            'active_positions': len(self.portfolio.positions),
            'max_positions': self.config.max_positions,
            'recent_events': self.risk_events[-5:] if self.risk_events else []
        }

    def get_position_risk(self, position_id: str) -> Optional[Dict]:
        """Get risk metrics for a specific position"""
        position = self.portfolio.get_position(position_id)
        if not position:
            return None

        return {
            'position_id': position_id,
            'token': position.token_symbol,
            'size_usd': position.size_usd,
            'size_pct': (position.size_usd / self.portfolio.get_portfolio_value()) * 100,
            'pnl_usd': position.pnl_usd,
            'pnl_pct': position.pnl_pct,
            'distance_to_stop_pct': ((position.current_price - position.stop_loss) / position.current_price) * 100,
            'hold_time_hours': position.hold_time_hours,
            'max_hold_time': self.config.max_hold_time_hours,
            'ml_risk_score': position.ml_risk_score
        }

    def print_status(self):
        """Print risk manager status"""
        metrics = self.get_risk_metrics()

        self.logger.info("=" * 50)
        self.logger.info("RISK MANAGER STATUS")
        self.logger.info("=" * 50)

        # Circuit breaker status
        state = metrics['circuit_breaker_state']
        if state == 'triggered':
            self.logger.critical("CIRCUIT BREAKER: TRIGGERED")
        elif state == 'warning':
            self.logger.warning("CIRCUIT BREAKER: WARNING")
        else:
            self.logger.info("CIRCUIT BREAKER: NORMAL")

        # Drawdown
        self.logger.info(
            f"Drawdown: Current={metrics['drawdown_pct']:.1f}% | "
            f"Warning={metrics['drawdown_warning']:.1f}% | Limit={metrics['drawdown_limit']:.1f}%"
        )

        # Daily limits
        self.logger.info(
            f"Daily Limits: Trades={metrics['daily_trades']}/{metrics['daily_trades_limit']} | "
            f"P&L=${metrics['daily_pnl']:.2f} (limit: -${metrics['daily_loss_limit']:.2f})"
        )

        # Positions
        self.logger.info(f"Positions: {metrics['active_positions']}/{metrics['max_positions']}")

        # Recent events
        if metrics['recent_events']:
            self.logger.info("Recent Risk Events:")
            for event in metrics['recent_events'][-3:]:
                self.logger.info(f"  [{event['timestamp'][:19]}] {event['type']}")
