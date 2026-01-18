#!/usr/bin/env python3
"""
Run backtest and save results to database for dashboard display
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.ohlcv_backtest import OHLCVBacktestEngine, OHLCVBacktestConfig
from src.strategies.simple_technical import SimpleRSI


def save_trades_to_db(engine, db_path="data/trading.db"):
    """Save backtest trades to database"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Clear old backtest results
    cursor.execute("DELETE FROM trades WHERE chain_id = 'BACKTEST'")

    saved_count = 0

    # Get closed trades from portfolio
    for trade in engine.portfolio.trades:
        try:
            # Calculate hold time in hours
            hold_time_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600

            cursor.execute("""
                INSERT INTO trades (
                    timestamp, token_address, token_symbol, chain_id,
                    strategy, action, price, size_usd, ml_risk_score,
                    stop_loss, take_profit_1, take_profit_2,
                    confidence, close_reason, pnl_usd, pnl_pct, hold_time_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.exit_time.isoformat(),
                trade.token_address,
                trade.token_symbol,
                'BACKTEST',  # Mark as backtest data
                trade.strategy_name,
                'CLOSE',
                trade.exit_price,
                trade.size_usd,
                trade.ml_risk_score,
                0,  # stop_loss (not in Trade dataclass)
                0,  # take_profit_1 (not in Trade dataclass)
                0,  # take_profit_2 (not in Trade dataclass)
                0.5,  # confidence (default)
                trade.exit_reason.value,  # ExitReason enum to string
                trade.pnl_usd,
                trade.pnl_pct,
                hold_time_hours
            ))
            saved_count += 1
        except Exception as e:
            print(f"Error saving trade: {e}")
            import traceback
            traceback.print_exc()

    conn.commit()
    conn.close()

    return saved_count


def main():
    print("Running backtest and saving to database...")
    print("="*60)

    # Configure backtest
    config = OHLCVBacktestConfig(
        initial_capital=10000,
        max_positions=5,
        max_position_size_pct=0.20
    )

    # Create strategies
    strategies = [SimpleRSI()]

    # Create engine
    engine = OHLCVBacktestEngine(
        strategies=strategies,
        config=config,
        ml_model=None,
        db_path="data/trading.db"
    )

    # Run backtest on last 30 days
    result = engine.run(
        pool_addresses=None,
        networks=None,
        timeframe='HOUR_1',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    # Print summary
    print("\nBacktest Results:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1f}%")
    print(f"  P&L: ${result.total_pnl:,.2f} ({result.roi_pct:.2f}%)")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")

    # Save to database
    print("\nSaving trades to database...")
    count = save_trades_to_db(engine)
    print(f"✓ Saved {count} trades to database")
    print("\nTrades are now visible in the dashboard at http://localhost:3000")


if __name__ == '__main__':
    main()
