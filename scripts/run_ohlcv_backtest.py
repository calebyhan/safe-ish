#!/usr/bin/env python3
"""
Run OHLCV-based backtests on historical candlestick data

This script uses proper OHLCV data for more accurate backtesting results.
"""
import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.ohlcv_backtest import OHLCVBacktestEngine, OHLCVBacktestConfig
from src.strategies.momentum import MomentumBreakoutStrategy
from src.strategies.dip_buying import DipBuyingStrategy
from src.strategies.technical import TechnicalStrategy, RSIStrategy, MovingAverageCrossover
from src.strategies.simple_technical import SimpleRSI, BuyAndHold
from src.ml.models import RugPullDetector


def load_ml_model(model_path: str = "models/ml_model.pkl"):
    """Load ML model if available"""
    try:
        model = RugPullDetector()
        model.load_model(model_path)
        print(f"✓ ML model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"⚠ Could not load ML model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run OHLCV-based backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest last 30 days with default strategies
  python scripts/run_ohlcv_backtest.py --days 30

  # Backtest specific date range
  python scripts/run_ohlcv_backtest.py --start 2024-01-01 --end 2024-02-01

  # Backtest specific pools
  python scripts/run_ohlcv_backtest.py --pools POOL_ADDRESS_1,POOL_ADDRESS_2 --days 60

  # Test with specific strategy
  python scripts/run_ohlcv_backtest.py --strategy momentum --days 30

  # Test on Solana only
  python scripts/run_ohlcv_backtest.py --network solana --days 30

  # Run with custom parameters
  python scripts/run_ohlcv_backtest.py --days 30 --capital 5000 --max-positions 3
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to backtest (default: 30)'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--network',
        default=None,
        help='Network filter (solana, ethereum, base, etc.)'
    )

    parser.add_argument(
        '--pools',
        default=None,
        help='Comma-separated list of pool addresses to test'
    )

    parser.add_argument(
        '--strategy',
        choices=['momentum', 'dip', 'technical', 'rsi', 'ma', 'simple', 'test', 'all'],
        default='simple',
        help='Strategy to test (default: simple)'
    )

    parser.add_argument(
        '--timeframe',
        choices=['MINUTE_1', 'MINUTE_5', 'MINUTE_15', 'HOUR_1', 'HOUR_4', 'HOUR_12', 'DAY_1'],
        default='HOUR_1',
        help='OHLCV timeframe (default: HOUR_1)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Initial capital in USD (default: 1000)'
    )

    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Max concurrent positions (default: 5)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=0.20,
        help='Max position size as % of capital (default: 0.20)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission per trade (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.005,
        help='Slippage per trade (default: 0.005 = 0.5%%)'
    )

    parser.add_argument(
        '--ml-model',
        default='models/ml_model.pkl',
        help='Path to ML model (default: models/ml_model.pkl)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML filtering'
    )

    parser.add_argument(
        '--no-intrabar-stops',
        action='store_true',
        help='Disable intra-bar stop checks (use close only)'
    )

    parser.add_argument(
        '--output',
        help='Path to save report'
    )

    parser.add_argument(
        '--db-path',
        default='data/trading.db',
        help='Database path (default: data/trading.db)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OHLCV BACKTEST ENGINE")
    print("=" * 70)

    # Parse date range
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)

    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Max Positions: {args.max_positions}")
    print(f"Position Size: {args.position_size*100:.0f}%")
    print(f"Commission: {args.commission*100:.2f}%")
    print(f"Slippage: {args.slippage*100:.2f}%")

    # Parse pools
    pool_addresses = None
    if args.pools:
        pool_addresses = [p.strip() for p in args.pools.split(',')]
        print(f"Testing {len(pool_addresses)} specific pools")

    # Parse networks
    networks = None
    if args.network:
        networks = [args.network]
        print(f"Network: {args.network}")

    # Load ML model
    ml_model = None
    if not args.no_ml:
        ml_model = load_ml_model(args.ml_model)

    # Initialize strategies
    strategies = []

    # Simple test strategies (very aggressive for testing)
    if args.strategy == 'simple':
        strategies.append(SimpleRSI())

    if args.strategy == 'test':
        strategies.append(BuyAndHold())

    # Technical strategies (OHLCV-compatible)
    if args.strategy == 'all' or args.strategy == 'technical':
        strategies.append(TechnicalStrategy(
            max_positions=args.max_positions,
            max_position_pct=args.position_size
        ))

    if args.strategy == 'all' or args.strategy == 'rsi':
        strategies.append(RSIStrategy(
            max_positions=args.max_positions,
            max_position_pct=args.position_size
        ))

    if args.strategy == 'all' or args.strategy == 'ma':
        strategies.append(MovingAverageCrossover(
            max_positions=args.max_positions,
            max_position_pct=args.position_size
        ))

    # Transaction-based strategies (require DexScreener data)
    if args.strategy == 'momentum':
        strategies.append(MomentumBreakoutStrategy(
            max_positions=args.max_positions,
            max_position_pct=args.position_size
        ))

    if args.strategy == 'dip':
        strategies.append(DipBuyingStrategy(
            max_positions=args.max_positions,
            max_position_pct=args.position_size
        ))

    if not strategies:
        print("\n✗ No strategies configured!")
        return

    print(f"\nStrategies: {', '.join(s.name for s in strategies)}")

    # Configure backtest
    config = OHLCVBacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        max_position_size_pct=args.position_size,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        ml_risk_threshold=0.40,
        use_intrabar_stops=not args.no_intrabar_stops
    )

    # Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70 + "\n")

    try:
        engine = OHLCVBacktestEngine(
            strategies=strategies,
            config=config,
            ml_model=ml_model,
            db_path=args.db_path
        )

        result = engine.run(
            pool_addresses=pool_addresses,
            networks=networks,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Generate report
        report = engine.generate_report(result, output_path=args.output)

        print(report)

        # Export to JSON
        if args.output:
            json_path = args.output.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'config': {
                        'initial_capital': config.initial_capital,
                        'max_positions': config.max_positions,
                        'commission_pct': config.commission_pct,
                        'slippage_pct': config.slippage_pct,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                    },
                    'results': {
                        'final_capital': result.final_capital,
                        'total_pnl': result.total_pnl,
                        'roi_pct': result.roi_pct,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'sortino_ratio': result.sortino_ratio,
                    },
                    'trades': [
                        {
                            'token_symbol': t.token_symbol,
                            'entry_time': t.entry_time.isoformat(),
                            'exit_time': t.exit_time.isoformat(),
                            'entry_price': t.entry_price,
                            'exit_price': t.exit_price,
                            'pnl_usd': t.pnl_usd,
                            'pnl_pct': t.pnl_pct,
                            'exit_reason': t.exit_reason.value
                        }
                        for t in result.trades
                    ]
                }, f, indent=2)
            print(f"\nJSON results saved to {json_path}")

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you've collected OHLCV data first:")
        print("   python scripts/collect_ohlcv_data.py --mode trending --days 30")
        print("\n2. Check available data:")
        print("   python scripts/collect_ohlcv_data.py --mode stats")
        return

    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
