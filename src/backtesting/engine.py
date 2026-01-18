"""Backtesting engine for testing strategies on historical data"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..strategies.base import BaseStrategy, Signal, Position, ExitReason
from ..trading.portfolio import PortfolioManager, Trade
from ..ml.models import RugPullDetector
from ..utils.logger import get_logger


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 1000.0
    max_positions: int = 5
    max_position_size_pct: float = 0.20
    commission_pct: float = 0.001  # 0.1% commission per trade
    slippage_pct: float = 0.005  # 0.5% slippage
    ml_risk_threshold: float = 0.40


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    # Performance metrics
    initial_capital: float
    final_capital: float
    total_pnl: float
    roi_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Time metrics
    backtest_start: datetime
    backtest_end: datetime
    days_tested: float

    # Trade history
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary of results"""
        return f"""
{'='*60}
BACKTEST RESULTS
{'='*60}

Period: {self.backtest_start.date()} to {self.backtest_end.date()} ({self.days_tested:.1f} days)

Capital:
  Initial: ${self.initial_capital:,.2f}
  Final: ${self.final_capital:,.2f}
  P&L: ${self.total_pnl:+,.2f} ({self.roi_pct:+.2f}%)

Trades:
  Total: {self.total_trades}
  Won: {self.winning_trades}
  Lost: {self.losing_trades}
  Win Rate: {self.win_rate:.1%}

Performance:
  Average Win: ${self.avg_win:.2f}
  Average Loss: ${self.avg_loss:.2f}
  Profit Factor: {self.profit_factor:.2f}
  Max Drawdown: {self.max_drawdown_pct:.2f}%
  Sharpe Ratio: {self.sharpe_ratio:.2f}

{'='*60}
"""


class BacktestEngine:
    """
    Backtesting engine for strategy validation

    Replays historical data through strategies to validate performance
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        config: Optional[BacktestConfig] = None,
        ml_model: Optional[RugPullDetector] = None
    ):
        """
        Initialize backtest engine

        Args:
            strategies: List of strategies to test
            config: Backtest configuration
            ml_model: Optional ML model for filtering
        """
        self.strategies = strategies
        self.config = config or BacktestConfig()
        self.ml_model = ml_model
        self.logger = get_logger("Backtesting")

        # State
        self.portfolio = None
        self.equity_curve = []

    def run(
        self,
        historical_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            historical_data: DataFrame with columns:
                - timestamp: datetime
                - token_address: str
                - token_symbol: str
                - chain_id: str
                - price_usd: float
                - liquidity_usd: float
                - volume_5min, volume_1hour, volume_24hour: float
                - buy_count_5min, sell_count_5min: int
                - buy_count_1hour, sell_count_1hour: int
                - price_change_5min, price_change_1hour, price_change_24hour: float
                - (and other features as needed)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with performance metrics and trade history
        """
        self.logger.info("Starting backtest...")

        # Filter by date range
        if start_date:
            historical_data = historical_data[historical_data['timestamp'] >= start_date]
        if end_date:
            historical_data = historical_data[historical_data['timestamp'] <= end_date]

        if len(historical_data) == 0:
            raise ValueError("No data in specified date range")

        # Initialize portfolio
        self.portfolio = PortfolioManager(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            max_position_size_pct=self.config.max_position_size_pct,
            paper_trading=True
        )

        # Get unique timestamps (bars)
        timestamps = sorted(historical_data['timestamp'].unique())

        self.logger.info(f"Testing {len(timestamps)} time periods")
        self.logger.info(f"Date range: {timestamps[0]} to {timestamps[-1]}")

        # Process each timestamp
        for i, current_time in enumerate(timestamps):
            # Get data at this timestamp
            current_data = historical_data[historical_data['timestamp'] == current_time]

            # Update positions (check stops, TPs)
            self._update_positions(current_data, current_time)

            # Generate new signals
            self._generate_signals(current_data, current_time)

            # Record equity
            equity = self.portfolio.get_portfolio_value()
            self.equity_curve.append((current_time, equity))

            # Log progress
            if (i + 1) % 100 == 0:
                self.logger.info(
                    f"Progress: {i+1}/{len(timestamps)} | "
                    f"Capital: ${equity:.2f} | "
                    f"Trades: {len(self.portfolio.trades)}"
                )

        # Close any remaining positions at final price
        self._close_all_positions("BACKTEST_END")

        # Calculate results
        result = self._calculate_results(timestamps[0], timestamps[-1])

        self.logger.info("Backtest complete!")
        self.logger.info(result.summary())

        return result

    def _generate_signals(self, current_data: pd.DataFrame, current_time: datetime):
        """Generate trading signals for current data"""
        for _, token_data in current_data.iterrows():
            # Convert to dict
            token_dict = token_data.to_dict()

            # Apply ML filter if available
            if self.ml_model:
                ml_risk = self._assess_ml_risk(token_dict)
                if ml_risk > self.config.ml_risk_threshold:
                    continue
                token_dict['ml_risk_score'] = ml_risk
            else:
                token_dict['ml_risk_score'] = 0.0

            # Check each strategy
            for strategy in self.strategies:
                signal = strategy.analyze(token_dict)

                if signal:
                    # Apply slippage to entry price
                    slipped_price = signal.entry_price * (1 + self.config.slippage_pct)

                    # Open position
                    position = self.portfolio.open_position(
                        signal,
                        ml_risk_score=token_dict['ml_risk_score'],
                        actual_price=slipped_price
                    )

                    if position:
                        self.logger.debug(
                            f"[{current_time}] Opened {signal.token_symbol} @ ${slipped_price:.6f}"
                        )

    def _update_positions(self, current_data: pd.DataFrame, current_time: datetime):
        """Update open positions with current prices"""
        for position in list(self.portfolio.positions.values()):
            # Find current price for this token
            token_row = current_data[
                current_data['token_address'] == position.token_address
            ]

            if len(token_row) == 0:
                # Token data not available - assume liquidity dried up
                self.logger.warning(
                    f"No data for {position.token_symbol}, closing at last price"
                )
                self.portfolio.close_position(
                    position.id,
                    ExitReason.MANUAL,  # Use MANUAL as fallback
                    position.current_price
                )
                continue

            # Update current price
            current_price = float(token_row.iloc[0]['price_usd'])
            self.portfolio.update_position_price(position.id, current_price)

            # Check exit conditions for this strategy
            strategy = next(
                (s for s in self.strategies if s.name == position.strategy_name),
                None
            )

            if strategy:
                token_data = token_row.iloc[0].to_dict()
                should_exit, exit_reason = strategy.should_exit(position, token_data)

                if should_exit:
                    # Apply slippage
                    slipped_price = current_price * (1 - self.config.slippage_pct)

                    # Apply commission
                    commission = position.size_usd * self.config.commission_pct

                    # Close position
                    trade = self.portfolio.close_position(
                        position.id,
                        exit_reason,
                        slipped_price
                    )

                    if trade:
                        # Deduct commission
                        trade.pnl_usd -= commission
                        self.portfolio.capital -= commission
                        self.portfolio.total_pnl -= commission

                        self.logger.debug(
                            f"[{current_time}] Closed {trade.token_symbol} | "
                            f"P&L: ${trade.pnl_usd:.2f} | {exit_reason.value}"
                        )

    def _close_all_positions(self, reason: str):
        """Close all remaining open positions"""
        for position in list(self.portfolio.positions.values()):
            self.portfolio.close_position(
                position.id,
                ExitReason.TIME_EXIT,
                position.current_price
            )

    def _assess_ml_risk(self, token_data: Dict) -> float:
        """Assess ML risk score for token"""
        if not self.ml_model:
            return 0.0

        try:
            # Extract features
            features = {
                'token_age_hours': token_data.get('token_age_hours', 0),
                'liquidity_usd': token_data.get('liquidity_usd', 0),
                'market_cap': token_data.get('market_cap', 0),
                'price_usd': token_data.get('price_usd', 0),
                'volume_5min': token_data.get('volume_5min', 0),
                'volume_1hour': token_data.get('volume_1hour', 0),
                'volume_24hour': token_data.get('volume_24hour', 0),
                'buy_count_5min': token_data.get('buy_count_5min', 0),
                'sell_count_5min': token_data.get('sell_count_5min', 0),
                'buy_count_1hour': token_data.get('buy_count_1hour', 0),
                'sell_count_1hour': token_data.get('sell_count_1hour', 0),
                'price_change_5min': token_data.get('price_change_5min', 0),
                'price_change_1hour': token_data.get('price_change_1hour', 0),
                'price_change_24hour': token_data.get('price_change_24hour', 0),
            }

            # Add derived features (simplified)
            features['buy_sell_ratio_5min'] = (
                features['buy_count_5min'] / max(features['sell_count_5min'], 1)
            )
            features['liquidity_to_mcap_ratio'] = (
                features['liquidity_usd'] / max(features['market_cap'], 1)
            )

            return self.ml_model.predict_risk(features)

        except Exception as e:
            self.logger.error(f"ML risk assessment error: {e}")
            return 0.5

    def _calculate_results(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> BacktestResult:
        """Calculate backtest results"""
        stats = self.portfolio.get_statistics()

        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1][1]
                curr_equity = self.equity_curve[i][1]
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

            returns_array = np.array(returns)
            sharpe = (
                (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
                if np.std(returns_array) > 0 else 0
            )
        else:
            sharpe = 0.0

        # Days tested
        days = (end_time - start_time).total_seconds() / 86400

        return BacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=stats['portfolio_value'],
            total_pnl=stats['total_pnl'],
            roi_pct=stats['roi_pct'],
            total_trades=stats['total_trades'],
            winning_trades=stats['trades_won'],
            losing_trades=stats['trades_lost'],
            win_rate=stats['win_rate'],
            avg_win=stats['avg_win'],
            avg_loss=stats['avg_loss'],
            profit_factor=stats['profit_factor'],
            max_drawdown_pct=stats['drawdown_pct'],
            sharpe_ratio=sharpe,
            backtest_start=start_time,
            backtest_end=end_time,
            days_tested=days,
            trades=self.portfolio.trades.copy(),
            equity_curve=self.equity_curve.copy()
        )
