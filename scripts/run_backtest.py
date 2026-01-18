#!/usr/bin/env python3
"""CLI script for running backtests"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strategies.momentum import MomentumBreakoutStrategy
from src.strategies.dip_buying import DipBuyingStrategy
from src.ml.models import RugPullDetector
from src.utils.database import Database
from src.utils.logger import setup_logging


def load_historical_data(db_path: str, days_back: int = 30) -> pd.DataFrame:
    """
    Load historical token snapshots from database

    Args:
        db_path: Path to database
        days_back: Number of days to load

    Returns:
        DataFrame with historical token data
    """
    db = Database(db_path)

    cutoff_date = datetime.now() - timedelta(days=days_back)

    with db.get_connection() as conn:
        query = """
            SELECT
                snapshot_time as timestamp,
                token_address,
                token_symbol,
                chain_id,
                price_usd,
                liquidity_usd,
                market_cap,
                volume_5min,
                volume_1hour,
                volume_24hour,
                buy_count_5min,
                sell_count_5min,
                buy_count_1hour,
                sell_count_1hour,
                price_change_5min,
                price_change_1hour,
                price_change_24hour,
                token_age_hours
            FROM token_snapshots
            WHERE snapshot_time >= ?
            AND snapshot_type = 'initial'
            ORDER BY snapshot_time
        """

        df = pd.read_sql_query(query, conn, params=(cutoff_date,))

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def run_backtest(args):
    """Run backtest with specified configuration"""
    print("\n" + "=" * 60)
    print("BACKTESTING ENGINE")
    print("=" * 60)

    # Setup logging
    logger = setup_logging(console_level=args.log_level)

    # Load historical data
    logger.info(f"Loading {args.days_back} days of historical data...")
    try:
        historical_data = load_historical_data(args.db_path, args.days_back)
        logger.info(f"Loaded {len(historical_data)} data points")
    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        return

    if len(historical_data) == 0:
        logger.error("No historical data found. Run data collection first.")
        return

    # Load ML model if specified
    ml_model = None
    if args.use_ml and args.ml_model_path:
        try:
            ml_model = RugPullDetector()
            ml_model.load(args.ml_model_path)
            logger.info(f"Loaded ML model from {args.ml_model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
            logger.warning("Continuing without ML filter")

    # Initialize strategies
    strategies = []

    if args.strategy in ['all', 'momentum']:
        strategies.append(MomentumBreakoutStrategy())
        logger.info("Added Momentum Breakout strategy")

    if args.strategy in ['all', 'dip']:
        strategies.append(DipBuyingStrategy())
        logger.info("Added Dip Buying strategy")

    if not strategies:
        logger.error("No strategies selected")
        return

    # Create backtest config
    config = BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        max_position_size_pct=args.position_size / 100,
        commission_pct=args.commission / 100,
        slippage_pct=args.slippage / 100,
        ml_risk_threshold=args.ml_threshold
    )

    # Create and run backtest
    engine = BacktestEngine(strategies, config, ml_model)

    logger.info("\n" + "=" * 60)
    logger.info("Starting backtest...")
    logger.info("=" * 60)

    try:
        result = engine.run(historical_data)

        # Print results
        print(result.summary())

        # Save results if requested
        if args.save_report:
            report_path = Path(args.output_dir) / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                f.write(result.summary())
                f.write("\n\nTrade History:\n")
                f.write("=" * 60 + "\n")

                for i, trade in enumerate(result.trades, 1):
                    f.write(f"\n{i}. {trade.token_symbol}\n")
                    f.write(f"   Strategy: {trade.strategy_name}\n")
                    f.write(f"   Entry: ${trade.entry_price:.6f} at {trade.entry_time}\n")
                    f.write(f"   Exit: ${trade.exit_price:.6f} at {trade.exit_time}\n")
                    f.write(f"   P&L: ${trade.pnl_usd:.2f} ({trade.pnl_pct:+.1f}%)\n")
                    f.write(f"   Reason: {trade.exit_reason.value}\n")

            logger.info(f"\nReport saved to: {report_path}")

        # Save equity curve
        if args.save_equity:
            equity_path = Path(args.output_dir) / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            equity_df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"Equity curve saved to: {equity_path}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Run strategy backtests on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest momentum strategy on last 30 days
  python scripts/run_backtest.py --strategy momentum --days-back 30

  # Backtest all strategies with ML filter
  python scripts/run_backtest.py --strategy all --use-ml --days-back 60

  # Backtest with custom capital and save report
  python scripts/run_backtest.py --capital 5000 --save-report --save-equity

  # Backtest dip buying only
  python scripts/run_backtest.py --strategy dip --days-back 45
        """
    )

    parser.add_argument(
        '--strategy',
        choices=['momentum', 'dip', 'all'],
        default='all',
        help='Strategy to test (default: all)'
    )

    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Days of historical data to test (default: 30)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Initial capital (default: 1000)'
    )

    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions (default: 5)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=20.0,
        help='Max position size as % of capital (default: 20)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.1,
        help='Commission per trade in % (default: 0.1)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.5,
        help='Slippage per trade in % (default: 0.5)'
    )

    parser.add_argument(
        '--use-ml',
        action='store_true',
        help='Use ML filter for token screening'
    )

    parser.add_argument(
        '--ml-model-path',
        default='data/models/rug_detector.pkl',
        help='Path to ML model (default: data/models/rug_detector.pkl)'
    )

    parser.add_argument(
        '--ml-threshold',
        type=float,
        default=0.40,
        help='ML risk threshold (default: 0.40)'
    )

    parser.add_argument(
        '--db-path',
        default='data/trading.db',
        help='Database path (default: data/trading.db)'
    )

    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save backtest report to file'
    )

    parser.add_argument(
        '--save-equity',
        action='store_true',
        help='Save equity curve to CSV'
    )

    parser.add_argument(
        '--output-dir',
        default='data/backtests',
        help='Output directory for reports (default: data/backtests)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    run_backtest(args)


if __name__ == '__main__':
    main()
