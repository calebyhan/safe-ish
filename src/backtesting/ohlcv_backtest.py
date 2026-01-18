"""OHLCV-based backtesting engine for accurate historical strategy testing"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import sqlite3

from ..strategies.base import BaseStrategy, Signal, Position, ExitReason
from ..trading.portfolio import PortfolioManager, Trade
from ..ml.models import RugPullDetector
from ..utils.logger import get_logger


@dataclass
class OHLCVBar:
    """Single OHLCV bar with computed indicators"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Computed indicators (populated by backtest engine)
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    volume_sma_20: Optional[float] = None
    atr_14: Optional[float] = None
    volatility: Optional[float] = None

    def to_token_data(self, pool_address: str, network: str, token_symbol: str) -> Dict:
        """Convert to token_data dict format expected by strategies"""
        # Estimate liquidity from volume (hourly volume * 50 for 24h rotation)
        estimated_liquidity = max(self.volume * 50, 10000)
        # Estimate market cap (assume FDV, arbitrary but above minimum)
        estimated_mcap = max(estimated_liquidity * 10, 50000)

        return {
            'token_address': pool_address,
            'token_symbol': token_symbol,
            'chain_id': network,
            'price_usd': self.close,
            'liquidity_usd': estimated_liquidity,
            'market_cap': estimated_mcap,
            'token_age_hours': 100,  # Assume established token

            # Volume (estimate from OHLCV)
            'volume_5min': self.volume / 12,  # Rough if hourly
            'volume_1hour': self.volume,
            'volume_6hour': self.volume * 6,
            'volume_24hour': self.volume * 24,

            # Transaction counts (estimate)
            'buy_count_5min': 10,
            'sell_count_5min': 10,
            'buy_count_1hour': 50,
            'sell_count_1hour': 50,

            # Price changes (computed from indicators)
            'price_change_5min': ((self.close - self.open) / self.open) * 100 if self.open else 0,
            'price_change_1hour': ((self.close - self.open) / self.open) * 100 if self.open else 0,
            'price_change_24hour': self.volatility or 0,

            # Technical indicators
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'rsi_14': self.rsi_14,
            'atr_14': self.atr_14,
            'volatility': self.volatility,
        }


@dataclass
class OHLCVBacktestConfig:
    """Configuration for OHLCV backtesting"""
    initial_capital: float = 1000.0
    max_positions: int = 5
    max_position_size_pct: float = 0.20
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.005  # 0.5% slippage
    ml_risk_threshold: float = 0.40
    use_intrabar_stops: bool = True  # Check high/low for stops within bar


@dataclass
class OHLCVBacktestResult:
    """Results from OHLCV backtest"""
    initial_capital: float
    final_capital: float
    total_pnl: float
    roi_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    backtest_start: datetime
    backtest_end: datetime
    days_tested: float
    bars_processed: int

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary report"""
        return f"""
{'='*70}
OHLCV BACKTEST RESULTS
{'='*70}

Period: {self.backtest_start.date()} to {self.backtest_end.date()} ({self.days_tested:.1f} days)
Bars Processed: {self.bars_processed:,}

CAPITAL
  Initial: ${self.initial_capital:,.2f}
  Final: ${self.final_capital:,.2f}
  P&L: ${self.total_pnl:+,.2f} ({self.roi_pct:+.2f}%)

TRADES
  Total: {self.total_trades}
  Winners: {self.winning_trades} ({self.win_rate:.1%})
  Losers: {self.losing_trades}
  Avg Win: ${self.avg_win:.2f}
  Avg Loss: ${self.avg_loss:.2f}
  Profit Factor: {self.profit_factor:.2f}

RISK METRICS
  Max Drawdown: {self.max_drawdown_pct:.2f}%
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}
  Calmar Ratio: {self.calmar_ratio:.2f}

{'='*70}
"""


class OHLCVBacktestEngine:
    """
    Backtesting engine using proper OHLCV candlestick data

    Features:
    - Accurate price simulation using OHLCV bars
    - Intra-bar stop loss and take profit checks (using high/low)
    - Technical indicator computation
    - Multiple pools/tokens simultaneously
    - Realistic commission and slippage modeling
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        config: Optional[OHLCVBacktestConfig] = None,
        ml_model: Optional[RugPullDetector] = None,
        db_path: str = "data/trading.db"
    ):
        self.strategies = strategies
        self.config = config or OHLCVBacktestConfig()
        self.ml_model = ml_model
        self.db_path = db_path
        self.logger = get_logger("OHLCVBacktest")

        # State
        self.portfolio = None
        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = 0

    def load_ohlcv_data(
        self,
        pool_addresses: Optional[List[str]] = None,
        networks: Optional[List[str]] = None,
        timeframe: str = "HOUR_1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database

        Args:
            pool_addresses: Optional list of specific pools
            networks: Optional network filter
            timeframe: Timeframe to load
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                o.pool_address,
                o.network,
                o.token_symbol,
                o.timeframe,
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv_data o
            WHERE o.timeframe = ?
        """
        params = [timeframe]

        if pool_addresses:
            placeholders = ','.join(['?' for _ in pool_addresses])
            query += f" AND o.pool_address IN ({placeholders})"
            params.extend(pool_addresses)

        if networks:
            placeholders = ','.join(['?' for _ in networks])
            query += f" AND o.network IN ({placeholders})"
            params.extend(networks)

        if start_date:
            query += " AND o.timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND o.timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY o.timestamp ASC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            raise ValueError("No OHLCV data found for specified criteria")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        self.logger.info(f"Loaded {len(df):,} OHLCV bars for {df['pool_address'].nunique()} pools")

        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators for each pool

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        result_dfs = []

        for pool_address in df['pool_address'].unique():
            pool_df = df[df['pool_address'] == pool_address].copy()
            pool_df = pool_df.sort_values('timestamp')

            # Simple Moving Averages
            pool_df['sma_20'] = pool_df['close'].rolling(window=20).mean()
            pool_df['sma_50'] = pool_df['close'].rolling(window=50).mean()

            # Volume SMA
            pool_df['volume_sma_20'] = pool_df['volume'].rolling(window=20).mean()

            # RSI (14 period)
            delta = pool_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            pool_df['rsi_14'] = 100 - (100 / (1 + rs))

            # ATR (14 period)
            high_low = pool_df['high'] - pool_df['low']
            high_close = abs(pool_df['high'] - pool_df['close'].shift())
            low_close = abs(pool_df['low'] - pool_df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            pool_df['atr_14'] = tr.rolling(window=14).mean()

            # Volatility (20-period standard deviation of returns)
            returns = pool_df['close'].pct_change()
            pool_df['volatility'] = returns.rolling(window=20).std() * 100

            result_dfs.append(pool_df)

        return pd.concat(result_dfs, ignore_index=True)

    def run(
        self,
        pool_addresses: Optional[List[str]] = None,
        networks: Optional[List[str]] = None,
        timeframe: str = "HOUR_1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> OHLCVBacktestResult:
        """
        Run backtest on OHLCV data

        Args:
            pool_addresses: Optional specific pools to test
            networks: Optional network filter
            timeframe: Candle timeframe
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Backtest results
        """
        self.logger.info("Starting OHLCV backtest...")

        # Load data
        df = self.load_ohlcv_data(
            pool_addresses=pool_addresses,
            networks=networks,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Compute indicators
        self.logger.info("Computing technical indicators...")
        df = self.compute_indicators(df)

        # Initialize portfolio
        self.portfolio = PortfolioManager(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            max_position_size_pct=self.config.max_position_size_pct,
            paper_trading=True
        )

        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = self.config.initial_capital

        # Get unique timestamps
        timestamps = sorted(df['timestamp'].unique())
        self.logger.info(f"Processing {len(timestamps):,} time periods...")
        self.logger.info(f"Date range: {timestamps[0]} to {timestamps[-1]}")

        # Process each bar
        for i, current_time in enumerate(timestamps):
            # Get all bars at this timestamp
            current_bars = df[df['timestamp'] == current_time]

            # 1. Update existing positions (check stops/TPs on high/low)
            self._update_positions(current_bars, current_time)

            # 2. Generate new signals
            self._generate_signals(current_bars, current_time)

            # 3. Record equity
            equity = self.portfolio.get_portfolio_value()
            self.equity_curve.append((current_time, equity))

            # Track drawdown
            if equity > self.peak_equity:
                self.peak_equity = equity
            drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100
            self.drawdown_curve.append((current_time, drawdown_pct))

            # Log progress
            if (i + 1) % 500 == 0:
                self.logger.info(
                    f"Progress: {i+1}/{len(timestamps)} | "
                    f"Equity: ${equity:.2f} | "
                    f"Trades: {len(self.portfolio.trades)}"
                )

        # Close remaining positions at final price
        self._close_all_positions()

        # Calculate results
        result = self._calculate_results(timestamps[0], timestamps[-1], len(timestamps))

        self.logger.info("Backtest complete!")
        self.logger.info(result.summary())

        return result

    def _generate_signals(self, current_bars: pd.DataFrame, current_time: datetime):
        """Generate trading signals from current bars"""
        for _, bar_data in current_bars.iterrows():
            # Skip if we already have a position in this pool
            pool_address = bar_data['pool_address']
            existing_position = any(
                p.token_address == pool_address
                for p in self.portfolio.positions.values()
            )
            if existing_position:
                continue

            # Create OHLCVBar object
            bar = OHLCVBar(
                timestamp=bar_data['timestamp'],
                open=float(bar_data['open']),
                high=float(bar_data['high']),
                low=float(bar_data['low']),
                close=float(bar_data['close']),
                volume=float(bar_data['volume']),
                sma_20=bar_data.get('sma_20'),
                sma_50=bar_data.get('sma_50'),
                rsi_14=bar_data.get('rsi_14'),
                volume_sma_20=bar_data.get('volume_sma_20'),
                atr_14=bar_data.get('atr_14'),
                volatility=bar_data.get('volatility')
            )

            # Convert to token_data format
            token_data = bar.to_token_data(
                pool_address=pool_address,
                network=bar_data['network'],
                token_symbol=bar_data['token_symbol'] or 'UNKNOWN'
            )

            # Apply ML filter if available
            if self.ml_model:
                try:
                    ml_risk = self.ml_model.predict_risk(token_data)
                    if ml_risk > self.config.ml_risk_threshold:
                        continue
                    token_data['ml_risk_score'] = ml_risk
                except Exception:
                    token_data['ml_risk_score'] = 0.3

            # Check each strategy
            for strategy in self.strategies:
                signal = strategy.analyze(token_data)

                if signal:
                    # Apply slippage
                    slipped_price = signal.entry_price * (1 + self.config.slippage_pct)
                    signal.entry_price = slipped_price

                    # Recalculate stops based on slipped entry
                    signal.stop_loss = slipped_price * (1 - strategy.stop_loss_pct)
                    signal.take_profit_1 = slipped_price * (1 + strategy.take_profit_1_pct)
                    signal.take_profit_2 = slipped_price * (1 + strategy.take_profit_2_pct)

                    # Open position
                    position = self.portfolio.open_position(
                        signal,
                        ml_risk_score=token_data.get('ml_risk_score', 0.0)
                    )

                    if position:
                        self.logger.debug(
                            f"[{current_time}] Opened {signal.token_symbol} @ ${slipped_price:.6f}"
                        )
                        break  # Only one signal per pool per bar

    def _update_positions(self, current_bars: pd.DataFrame, current_time: datetime):
        """Update positions with current bar data, checking intra-bar stops"""
        for position in list(self.portfolio.positions.values()):
            # Find bar for this position's pool
            bar_data = current_bars[current_bars['pool_address'] == position.token_address]

            if len(bar_data) == 0:
                continue

            bar = bar_data.iloc[0]
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])

            # Update current price
            position.current_price = close

            exit_price = None
            exit_reason = None

            if self.config.use_intrabar_stops:
                # Check if stop loss was hit (using low of bar)
                effective_stop = position.trailing_stop or position.stop_loss
                if low <= effective_stop:
                    exit_price = effective_stop
                    exit_reason = ExitReason.STOP_LOSS

                # Check if TP1 was hit (using high of bar)
                elif high >= position.take_profit_1 and position.partial_exits == 0:
                    exit_price = position.take_profit_1
                    exit_reason = ExitReason.TAKE_PROFIT_1

                # Check if TP2 was hit
                elif high >= position.take_profit_2 and position.partial_exits == 1:
                    exit_price = position.take_profit_2
                    exit_reason = ExitReason.TAKE_PROFIT_2

            else:
                # Check on close only
                if close <= (position.trailing_stop or position.stop_loss):
                    exit_price = close
                    exit_reason = ExitReason.STOP_LOSS
                elif close >= position.take_profit_1 and position.partial_exits == 0:
                    exit_price = close
                    exit_reason = ExitReason.TAKE_PROFIT_1
                elif close >= position.take_profit_2 and position.partial_exits == 1:
                    exit_price = close
                    exit_reason = ExitReason.TAKE_PROFIT_2

            if exit_price and exit_reason:
                # Apply slippage
                slipped_exit = exit_price * (1 - self.config.slippage_pct)

                # Handle partial exit at TP1
                if exit_reason == ExitReason.TAKE_PROFIT_1:
                    trade = self.portfolio.close_position(
                        position.id,
                        exit_reason,
                        slipped_exit,
                        partial_pct=0.5  # Close 50%
                    )
                    # Activate trailing stop
                    position.update_trailing_stop(exit_price, trail_pct=0.10)
                else:
                    trade = self.portfolio.close_position(
                        position.id,
                        exit_reason,
                        slipped_exit
                    )

                if trade:
                    # Apply commission
                    commission = trade.pnl_usd * self.config.commission_pct
                    trade.pnl_usd -= abs(commission)
                    self.portfolio.capital -= abs(commission)

                    self.logger.debug(
                        f"[{current_time}] Closed {trade.token_symbol} | "
                        f"P&L: ${trade.pnl_usd:.2f} | {exit_reason.value}"
                    )

            else:
                # Update trailing stop if price made new high
                if high > position.entry_price and position.partial_exits > 0:
                    position.update_trailing_stop(high, trail_pct=0.10)

    def _close_all_positions(self):
        """Close all remaining positions at end of backtest"""
        for position in list(self.portfolio.positions.values()):
            self.portfolio.close_position(
                position.id,
                ExitReason.TIME_EXIT,
                position.current_price
            )

    def _calculate_results(
        self,
        start_time: datetime,
        end_time: datetime,
        bars_processed: int
    ) -> OHLCVBacktestResult:
        """Calculate backtest results and metrics"""
        stats = self.portfolio.get_statistics()

        # Calculate returns for risk metrics
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev = self.equity_curve[i-1][1]
                curr = self.equity_curve[i][1]
                ret = (curr - prev) / prev if prev > 0 else 0
                returns.append(ret)

            returns_array = np.array(returns)

            # Sharpe Ratio (annualized, assuming hourly bars)
            periods_per_year = 365 * 24
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0

            # Sortino Ratio (downside deviation only)
            negative_returns = returns_array[returns_array < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
            sortino = (mean_return / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0

        else:
            sharpe = 0.0
            sortino = 0.0

        # Max drawdown
        max_dd = max(dd for _, dd in self.drawdown_curve) if self.drawdown_curve else 0

        # Calmar Ratio (annual return / max drawdown)
        days = (end_time - start_time).total_seconds() / 86400
        annual_return = (stats['roi_pct'] / days * 365) if days > 0 else 0
        calmar = (annual_return / max_dd) if max_dd > 0 else 0

        return OHLCVBacktestResult(
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
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            backtest_start=start_time,
            backtest_end=end_time,
            days_tested=days,
            bars_processed=bars_processed,
            trades=self.portfolio.trades.copy(),
            equity_curve=self.equity_curve.copy(),
            drawdown_curve=self.drawdown_curve.copy()
        )

    def generate_report(self, result: OHLCVBacktestResult, output_path: str = None) -> str:
        """
        Generate detailed backtest report

        Args:
            result: Backtest result
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = result.summary()

        # Add trade details
        report += "\nTRADE HISTORY\n"
        report += "-" * 70 + "\n"

        for trade in result.trades:
            pnl_emoji = "+" if trade.pnl_usd > 0 else ""
            report += (
                f"{trade.exit_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{trade.token_symbol:10s} | "
                f"Entry: ${trade.entry_price:.6f} | "
                f"Exit: ${trade.exit_price:.6f} | "
                f"P&L: ${pnl_emoji}{trade.pnl_usd:.2f} ({trade.pnl_pct:+.1f}%) | "
                f"{trade.exit_reason.value}\n"
            )

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")

        return report
