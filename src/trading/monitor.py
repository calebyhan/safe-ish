"""Position Monitor for tracking and managing open positions"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from ..strategies.base import Position, ExitReason, BaseStrategy
from .portfolio import PortfolioManager
from ..utils.logger import get_logger


@dataclass
class MonitorConfig:
    """Configuration for position monitoring"""
    update_interval_seconds: float = 30.0
    max_hold_time_hours: float = 12.0
    enable_trailing_stops: bool = True
    trailing_stop_pct: float = 0.10
    partial_exit_pct: float = 0.50  # Exit 50% at TP1


class PositionMonitor:
    """
    Monitors open positions and triggers exits

    Responsibilities:
    - Periodically check position prices
    - Trigger stop losses and take profits
    - Manage trailing stops
    - Enforce time-based exits
    - Handle partial exits
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        strategies: Dict[str, BaseStrategy],
        config: Optional[MonitorConfig] = None,
        price_fetcher: Optional[Callable] = None
    ):
        """
        Initialize position monitor

        Args:
            portfolio: Portfolio manager instance
            strategies: Dict of strategy name -> strategy instance
            config: Monitor configuration
            price_fetcher: Async function to fetch current prices
        """
        self.portfolio = portfolio
        self.strategies = strategies
        self.config = config or MonitorConfig()
        self.price_fetcher = price_fetcher

        # Logger
        self.logger = get_logger("PositionMonitor")

        # Tracking
        self.is_running = False
        self.last_update = None
        self.updates_count = 0
        self.exits_triggered = 0

    async def start(self):
        """Start the position monitoring loop"""
        self.is_running = True
        self.logger.info(f"Position Monitor started (interval: {self.config.update_interval_seconds}s)")

        while self.is_running:
            try:
                await self.update_positions()
                await asyncio.sleep(self.config.update_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    def stop(self):
        """Stop the position monitoring loop"""
        self.is_running = False
        self.logger.info("Position Monitor stopped")

    async def update_positions(self):
        """Update all positions and check exit conditions"""
        positions = self.portfolio.get_all_positions()

        if not positions:
            return

        self.last_update = datetime.utcnow()
        self.updates_count += 1

        for position in positions:
            try:
                await self._update_position(position)
            except Exception as e:
                self.logger.error(f"Error updating position {position.id}: {e}")

    async def _update_position(self, position: Position):
        """
        Update a single position

        Args:
            position: Position to update
        """
        # Fetch current price
        if self.price_fetcher:
            current_data = await self.price_fetcher(
                position.token_address,
                position.chain_id
            )
            if current_data:
                position.current_price = current_data.get('price_usd', position.current_price)
        else:
            current_data = {'price_usd': position.current_price}

        # Check exit conditions via strategy
        strategy = self.strategies.get(position.strategy_name)
        if strategy:
            should_exit, exit_reason = strategy.should_exit(position, current_data)

            if should_exit:
                await self._execute_exit(position, exit_reason, current_data)
                return

        # Check global exit conditions
        await self._check_global_exits(position, current_data)

    async def _execute_exit(
        self,
        position: Position,
        exit_reason: ExitReason,
        current_data: Dict
    ):
        """
        Execute position exit

        Args:
            position: Position to exit
            exit_reason: Reason for exit
            current_data: Current market data
        """
        exit_price = current_data.get('price_usd', position.current_price)

        # Handle partial exits at TP1
        if exit_reason == ExitReason.TAKE_PROFIT_1 and position.partial_exits == 0:
            # Partial exit - close 50%
            self.portfolio.close_position(
                position.id,
                exit_reason,
                exit_price,
                partial_pct=self.config.partial_exit_pct
            )
            self.exits_triggered += 1

            # Activate trailing stop for remaining position
            if self.config.enable_trailing_stops:
                position.update_trailing_stop(exit_price, self.config.trailing_stop_pct)

        else:
            # Full exit
            self.portfolio.close_position(
                position.id,
                exit_reason,
                exit_price,
                partial_pct=1.0
            )
            self.exits_triggered += 1

    async def _check_global_exits(self, position: Position, current_data: Dict):
        """
        Check global exit conditions not handled by strategy

        Args:
            position: Position to check
            current_data: Current market data
        """
        # 1. Maximum hold time
        if position.hold_time_hours >= self.config.max_hold_time_hours:
            await self._execute_exit(position, ExitReason.TIME_EXIT, current_data)
            return

        # 2. Update trailing stop on new highs
        if self.config.enable_trailing_stops and position.partial_exits > 0:
            position.update_trailing_stop(
                position.current_price,
                self.config.trailing_stop_pct
            )

    def get_position_status(self, position_id: str) -> Optional[Dict]:
        """
        Get detailed status for a position

        Args:
            position_id: Position ID

        Returns:
            Status dict or None if not found
        """
        position = self.portfolio.get_position(position_id)
        if not position:
            return None

        return {
            'id': position.id,
            'token': position.token_symbol,
            'chain': position.chain_id,
            'strategy': position.strategy_name,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'size_usd': position.size_usd,
            'pnl_usd': position.pnl_usd,
            'pnl_pct': position.pnl_pct,
            'hold_time_hours': position.hold_time_hours,
            'stop_loss': position.stop_loss,
            'trailing_stop': position.trailing_stop,
            'take_profit_1': position.take_profit_1,
            'take_profit_2': position.take_profit_2,
            'partial_exits': position.partial_exits
        }

    def get_all_statuses(self) -> List[Dict]:
        """Get status for all positions"""
        return [
            self.get_position_status(pos.id)
            for pos in self.portfolio.get_all_positions()
        ]

    def get_statistics(self) -> Dict:
        """Get monitor statistics"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'updates_count': self.updates_count,
            'exits_triggered': self.exits_triggered,
            'update_interval': self.config.update_interval_seconds,
            'max_hold_time': self.config.max_hold_time_hours,
            'trailing_stops_enabled': self.config.enable_trailing_stops
        }

    def print_status(self):
        """Print current monitor status"""
        self.logger.info("=" * 50)
        self.logger.info("POSITION MONITOR STATUS")
        self.logger.info("=" * 50)

        stats = self.get_statistics()
        self.logger.info(f"Running: {'Yes' if stats['is_running'] else 'No'}")
        self.logger.info(f"Last Update: {stats['last_update'] or 'Never'}")
        self.logger.info(f"Updates: {stats['updates_count']} | Exits Triggered: {stats['exits_triggered']}")

        positions = self.get_all_statuses()
        if positions:
            self.logger.info(f"Active Positions ({len(positions)}):")
            for pos in positions:
                trailing = f" | Trailing: ${pos['trailing_stop']:.6f}" if pos['trailing_stop'] else ""
                self.logger.info(
                    f"  {pos['token']} ({pos['strategy']}) | "
                    f"Entry: ${pos['entry_price']:.6f} | Current: ${pos['current_price']:.6f} | "
                    f"P&L: ${pos['pnl_usd']:.2f} ({pos['pnl_pct']:+.1f}%) | "
                    f"Hold: {pos['hold_time_hours']:.1f}h | Stop: ${pos['stop_loss']:.6f}{trailing}"
                )
        else:
            self.logger.info("No active positions")
