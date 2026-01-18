"""Portfolio Manager for tracking capital and positions"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from ..strategies.base import Position, Signal, ExitReason
from ..utils.logger import get_logger


@dataclass
class Trade:
    """Completed trade record"""
    id: str
    token_address: str
    token_symbol: str
    chain_id: str
    strategy_name: str
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    pnl_usd: float
    pnl_pct: float
    ml_risk_score: float


class PortfolioManager:
    """
    Manages trading capital and positions

    Responsibilities:
    - Track available capital
    - Open and close positions
    - Calculate portfolio metrics
    - Enforce position limits
    - Record trade history
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_positions: int = 5,
        max_position_size_pct: float = 0.20,
        paper_trading: bool = True
    ):
        """
        Initialize portfolio manager

        Args:
            initial_capital: Starting capital in USD
            max_positions: Maximum concurrent positions
            max_position_size_pct: Maximum single position as % of capital
            paper_trading: Whether in paper trading mode
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size_pct = max_position_size_pct
        self.paper_trading = paper_trading

        # Logger
        self.logger = get_logger("Portfolio")

        # Active positions
        self.positions: Dict[str, Position] = {}  # position_id -> Position

        # Trade history
        self.trades: List[Trade] = []

        # Tracking
        self.total_pnl = 0.0
        self.peak_capital = initial_capital
        self.trades_won = 0
        self.trades_lost = 0

    def open_position(
        self,
        signal: Signal,
        ml_risk_score: float,
        actual_price: Optional[float] = None,
        current_time: Optional[datetime] = None
    ) -> Optional[Position]:
        """
        Open a new position based on signal

        Args:
            signal: Trading signal
            ml_risk_score: ML model risk score
            actual_price: Actual execution price (or use signal price)
            current_time: Current time (for backtesting) or None for live trading

        Returns:
            Position if opened, None if rejected
        """
        # Check position limits
        if len(self.positions) >= self.max_positions:
            return None

        # Check if already in position for this token
        for pos in self.positions.values():
            if pos.token_address == signal.token_address:
                return None

        # Calculate position size
        position_size = self._calculate_position_size(
            signal.position_size_pct,
            signal.confidence,
            ml_risk_score
        )

        # Check minimum position size
        if position_size < 10:  # $10 minimum
            return None

        # Check available capital
        if position_size > self.capital:
            position_size = self.capital * 0.9  # Use 90% of remaining

        # Use actual price or signal price
        entry_price = actual_price if actual_price else signal.entry_price

        # Create position
        position_id = str(uuid.uuid4())[:8]
        position = Position(
            id=position_id,
            token_address=signal.token_address,
            token_symbol=signal.token_symbol,
            chain_id=signal.chain_id,
            strategy_name=signal.strategy_name,
            entry_price=entry_price,
            current_price=entry_price,
            size_usd=position_size,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            entry_time=current_time if current_time else datetime.utcnow(),
            ml_risk_score=ml_risk_score
        )

        # Deduct capital
        self.capital -= position_size

        # Store position
        self.positions[position_id] = position

        mode = "[PAPER] " if self.paper_trading else ""
        self.logger.info(
            f"{mode}OPENED: {signal.token_symbol} | Entry: ${entry_price:.6f} | "
            f"Size: ${position_size:.2f} | Stop: ${signal.stop_loss:.6f} | "
            f"TP1: ${signal.take_profit_1:.6f} | TP2: ${signal.take_profit_2:.6f}"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_reason: ExitReason,
        exit_price: Optional[float] = None,
        partial_pct: float = 1.0,
        current_time: Optional[datetime] = None
    ) -> Optional[Trade]:
        """
        Close a position

        Args:
            position_id: Position ID to close
            exit_reason: Reason for exit
            exit_price: Exit price (or use current price)
            partial_pct: Percentage of position to close (1.0 = full)
            current_time: Current time (for backtesting) or None for live trading

        Returns:
            Trade record if closed, None if position not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        # Use exit price or current price
        actual_exit = exit_price if exit_price else position.current_price

        # Calculate exit size
        exit_size = position.size_usd * partial_pct

        # Calculate P&L
        price_change_pct = (actual_exit - position.entry_price) / position.entry_price
        pnl_usd = exit_size * price_change_pct
        pnl_pct = price_change_pct * 100

        # Create trade record
        trade = Trade(
            id=str(uuid.uuid4())[:8],
            token_address=position.token_address,
            token_symbol=position.token_symbol,
            chain_id=position.chain_id,
            strategy_name=position.strategy_name,
            entry_price=position.entry_price,
            exit_price=actual_exit,
            size_usd=exit_size,
            entry_time=position.entry_time,
            exit_time=current_time if current_time else datetime.utcnow(),
            exit_reason=exit_reason,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            ml_risk_score=position.ml_risk_score
        )

        # Record trade
        self.trades.append(trade)

        # Update capital
        self.capital += exit_size + pnl_usd

        # Update stats
        self.total_pnl += pnl_usd
        if pnl_usd > 0:
            self.trades_won += 1
        else:
            self.trades_lost += 1

        # Update peak capital
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Handle partial or full close
        if partial_pct >= 1.0:
            # Full close - remove position
            del self.positions[position_id]
        else:
            # Partial close - reduce position size
            position.size_usd -= exit_size
            position.partial_exits += 1

        mode = "[PAPER] " if self.paper_trading else ""
        emoji = "+" if pnl_usd > 0 else ""
        self.logger.info(
            f"{mode}CLOSED: {position.token_symbol} | Exit: ${actual_exit:.6f} | "
            f"P&L: ${emoji}{pnl_usd:.2f} ({pnl_pct:+.1f}%) | Reason: {exit_reason.value}"
        )

        return trade

    def update_position_price(self, position_id: str, new_price: float):
        """Update current price for a position"""
        if position_id in self.positions:
            self.positions[position_id].current_price = new_price

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)

    def get_position_by_token(self, token_address: str) -> Optional[Position]:
        """Get position by token address"""
        for pos in self.positions.values():
            if pos.token_address == token_address:
                return pos
        return None

    def get_all_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())

    def _calculate_position_size(
        self,
        base_pct: float,
        confidence: float,
        ml_risk_score: float
    ) -> float:
        """
        Calculate position size based on factors

        Args:
            base_pct: Base position size percentage
            confidence: Signal confidence
            ml_risk_score: ML risk score

        Returns:
            Position size in USD
        """
        # Start with base percentage
        size = self.capital * base_pct

        # Adjust for confidence (0.5-1.0 multiplier)
        confidence_factor = 0.5 + (confidence * 0.5)
        size *= confidence_factor

        # Adjust for ML risk (0.5-1.0 multiplier, lower for higher risk)
        risk_factor = 1 - (ml_risk_score * 0.5)
        size *= risk_factor

        # Cap at max position size
        max_size = self.capital * self.max_position_size_pct
        size = min(size, max_size)

        return size

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(p.size_usd + p.pnl_usd for p in self.positions.values())
        return self.capital + positions_value

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(p.pnl_usd for p in self.positions.values())

    def get_drawdown(self) -> float:
        """Get current drawdown from peak"""
        current_value = self.get_portfolio_value()
        if self.peak_capital == 0:
            return 0
        return (self.peak_capital - current_value) / self.peak_capital * 100

    def get_statistics(self) -> Dict:
        """Get portfolio statistics"""
        total_trades = self.trades_won + self.trades_lost
        win_rate = self.trades_won / total_trades if total_trades > 0 else 0

        # Calculate average win/loss
        wins = [t.pnl_usd for t in self.trades if t.pnl_usd > 0]
        losses = [t.pnl_usd for t in self.trades if t.pnl_usd <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else total_wins

        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'portfolio_value': self.get_portfolio_value(),
            'total_pnl': self.total_pnl,
            'roi_pct': (self.total_pnl / self.initial_capital) * 100,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'drawdown_pct': self.get_drawdown(),
            'peak_capital': self.peak_capital,
            'active_positions': len(self.positions),
            'max_positions': self.max_positions,
            'total_trades': total_trades,
            'trades_won': self.trades_won,
            'trades_lost': self.trades_lost,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'paper_trading': self.paper_trading
        }

    def print_summary(self):
        """Print portfolio summary"""
        stats = self.get_statistics()
        mode = "[PAPER]" if self.paper_trading else "[LIVE]"

        self.logger.info("=" * 50)
        self.logger.info(f"PORTFOLIO SUMMARY {mode}")
        self.logger.info("=" * 50)
        self.logger.info(
            f"Capital: Initial=${stats['initial_capital']:.2f} | "
            f"Current=${stats['portfolio_value']:.2f} | "
            f"P&L=${stats['total_pnl']:.2f} ({stats['roi_pct']:+.1f}%) | "
            f"Drawdown={stats['drawdown_pct']:.1f}%"
        )
        self.logger.info(
            f"Positions: {stats['active_positions']}/{stats['max_positions']} | "
            f"Unrealized P&L: ${stats['unrealized_pnl']:.2f}"
        )
        self.logger.info(
            f"Trades: {stats['total_trades']} | Won: {stats['trades_won']} | "
            f"Lost: {stats['trades_lost']} | Win Rate: {stats['win_rate']:.1%} | "
            f"Profit Factor: {stats['profit_factor']:.2f}"
        )

        # Log active positions
        if self.positions:
            self.logger.info("ACTIVE POSITIONS:")
            for pos in self.positions.values():
                self.logger.info(
                    f"  {pos.token_symbol}: ${pos.size_usd:.2f} | "
                    f"P&L: ${pos.pnl_usd:.2f} ({pos.pnl_pct:+.1f}%) | "
                    f"Hold: {pos.hold_time_hours:.1f}h"
                )
