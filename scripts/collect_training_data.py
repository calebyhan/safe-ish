#!/usr/bin/env python3
"""CLI script for collecting historical training data"""
import asyncio
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.dex_api import DexScreenerAPI
from src.data_collection.collector import HistoricalDataCollector
from src.utils.database import Database


async def collect_phase(args):
    """Collect initial token snapshots"""
    print("\n" + "=" * 60)
    print("PHASE 1: COLLECTING INITIAL TOKEN SNAPSHOTS")
    print("=" * 60)

    db = Database(args.db_path)

    async with DexScreenerAPI() as api:
        collector = HistoricalDataCollector(api, db)

        stats = await collector.collect_training_data(
            num_tokens=args.num_tokens,
            chains=args.chains,
            search_queries=None,  # Use defaults
            include_boosted=not args.no_boosted,
            max_token_age_hours=args.max_age_hours
        )

        print("\n✓ Collection phase complete!")
        print(f"\nNext step: Wait {args.label_hours} hours, then run:")
        print(f"  python scripts/collect_training_data.py --mode label --label-hours {args.label_hours}")

        return stats


async def label_phase(args):
    """Label collected tokens by checking their fate"""
    print("\n" + "=" * 60)
    print("PHASE 2: LABELING TOKEN OUTCOMES")
    print("=" * 60)

    db = Database(args.db_path)

    async with DexScreenerAPI() as api:
        collector = HistoricalDataCollector(api, db)

        stats = await collector.label_collected_tokens(
            hours_to_wait=args.label_hours,
            batch_size=args.batch_size
        )

        print("\n✓ Labeling phase complete!")
        print(f"\nReady for ML training! Run:")
        print(f"  python scripts/train_model.py")

        return stats


async def full_cycle(args):
    """Run full collection and labeling cycle (for testing only)"""
    print("\n" + "=" * 60)
    print("FULL CYCLE MODE (Testing)")
    print("=" * 60)
    print("\nWARNING: This will collect and immediately label tokens.")
    print("For production use, run 'collect' and 'label' modes separately with time delay.\n")

    db = Database(args.db_path)

    async with DexScreenerAPI() as api:
        collector = HistoricalDataCollector(api, db)

        # Phase 1: Collect
        print("\nPhase 1: Collecting tokens...")
        collect_stats = await collector.collect_training_data(
            num_tokens=args.num_tokens,
            chains=args.chains,
            search_queries=None,
            include_boosted=not args.no_boosted,
            max_token_age_hours=args.max_age_hours
        )

        # Brief pause
        print("\nPausing 5 seconds before labeling...")
        await asyncio.sleep(5)

        # Phase 2: Label
        print("\nPhase 2: Labeling tokens...")
        label_stats = await collector.label_collected_tokens(
            hours_to_wait=0,  # No wait in full cycle mode
            batch_size=args.batch_size
        )

        print("\n✓ Full cycle complete!")

        return {'collect': collect_stats, 'label': label_stats}


async def historical_auto_label_phase(args):
    """Collect and auto-label historical tokens in one pass"""
    print("\n" + "="*60)
    print("Historical Auto-Label Collection")
    print("="*60 + "\n")

    db = Database(args.db_path)

    async with DexScreenerAPI() as api:
        collector = HistoricalDataCollector(api, db)

        stats = await collector.collect_and_auto_label_historical(
            num_tokens=args.num_tokens,
            min_token_age_hours=args.min_age_hours,
            max_token_age_hours=args.max_age_hours,
            chains=args.chains
        )

        print("\n✓ Historical auto-label complete!")
        print(f"\nCollected and labeled {stats['tokens_labeled']} tokens")

        usable = stats['labels'].get(0, 0) + stats['labels'].get(1, 0) + stats['labels'].get(2, 0) + stats['labels'].get(3, 0)
        print(f"Usable for training (labels 0-3): {usable}")

        if usable >= 100:
            print("\n✓ Ready to train! Run:")
            print("  python scripts/train_model.py --mode binary --evaluate --report")
        else:
            print(f"\nCollect {100 - usable} more samples to reach minimum (100)")

        return stats


async def stats_phase(args):
    """Show collection and labeling statistics"""
    db = Database(args.db_path)

    print("\n" + "=" * 60)
    print("TRAINING DATA STATISTICS")
    print("=" * 60)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Token snapshots count
        cursor.execute("SELECT COUNT(*) as count FROM token_snapshots WHERE snapshot_type = 'initial'")
        snapshots = cursor.fetchone()['count']

        # Labels count by type
        cursor.execute("""
            SELECT
                label,
                COUNT(*) as count
            FROM token_labels
            GROUP BY label
            ORDER BY label
        """)
        label_rows = cursor.fetchall()

        # Unlabeled count
        cursor.execute("""
            SELECT COUNT(DISTINCT token_address) as count
            FROM token_snapshots
            WHERE snapshot_type = 'initial'
            AND token_address NOT IN (SELECT token_address FROM token_labels)
        """)
        unlabeled = cursor.fetchone()['count']

        print(f"\nToken Snapshots Collected: {snapshots}")
        print(f"Unlabeled Tokens: {unlabeled}")

        if label_rows:
            print(f"\nLabeled Tokens:")
            label_names = {
                0: 'Rug Pull',
                1: 'Pump & Dump',
                2: 'Wash Trading',
                3: 'Legitimate',
                4: 'Unknown'
            }
            total_labeled = 0
            for row in label_rows:
                label_id = row['label']
                count = row['count']
                total_labeled += count
                print(f"  {label_names.get(label_id, 'Unknown')}: {count}")

            print(f"\nTotal Labeled: {total_labeled}")

            # Calculate class distribution
            print(f"\nClass Distribution:")
            for row in label_rows:
                label_id = row['label']
                count = row['count']
                pct = (count / total_labeled * 100) if total_labeled > 0 else 0
                print(f"  {label_names.get(label_id, 'Unknown')}: {pct:.1f}%")

            # Check if ready for training
            usable = sum(row['count'] for row in label_rows if row['label'] in [0, 1, 2, 3])
            print(f"\nUsable for Training (labels 0-3): {usable}")

            if usable >= 100:
                print("✓ Sufficient data for ML training!")
            else:
                print(f"⚠ Need at least 100 labeled tokens (have {usable})")


def main():
    parser = argparse.ArgumentParser(
        description='Collect and label historical token data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Historical auto-label (instant, no waiting!)
  python scripts/collect_training_data.py --mode historical --num-tokens 500 --min-age-hours 48 --max-age-hours 720

  # Historical with specific age range for more rug pulls
  python scripts/collect_training_data.py --mode historical --num-tokens 500 --min-age-hours 48 --max-age-hours 168

  # Historical from Solana only
  python scripts/collect_training_data.py --mode historical --num-tokens 500 --chains solana

  # OLD METHOD: Collect tokens (requires waiting)
  python scripts/collect_training_data.py --mode collect --num-tokens 500 --max-age-hours 72

  # OLD METHOD: Label collected tokens (run after 24 hours)
  python scripts/collect_training_data.py --mode label --label-hours 24

  # Show statistics
  python scripts/collect_training_data.py --mode stats
        """
    )

    parser.add_argument(
        '--mode',
        choices=['collect', 'label', 'full', 'stats', 'historical'],
        default='stats',
        help='Operation mode: historical (instant auto-label), collect, label, full, or stats'
    )

    parser.add_argument(
        '--num-tokens',
        type=int,
        default=500,
        help='Number of tokens to collect (default: 500)'
    )

    parser.add_argument(
        '--chains',
        nargs='+',
        default=['solana', 'ethereum', 'base'],
        help='Blockchains to collect from (default: solana ethereum base)'
    )

    parser.add_argument(
        '--label-hours',
        type=int,
        default=24,
        help='Hours to wait before checking token fate (default: 24)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for labeling (default: 100)'
    )

    parser.add_argument(
        '--db-path',
        default='data/trading.db',
        help='Database path (default: data/trading.db)'
    )

    parser.add_argument(
        '--max-age-hours',
        type=int,
        default=None,
        help='Maximum token age in hours (filters out older tokens, default: no limit)'
    )

    parser.add_argument(
        '--no-boosted',
        action='store_true',
        help='Skip collecting boosted tokens (default: include boosted tokens)'
    )

    parser.add_argument(
        '--min-age-hours',
        type=int,
        default=48,
        help='Minimum token age in hours for historical mode (default: 48 = 2 days)'
    )

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == 'collect':
        asyncio.run(collect_phase(args))
    elif args.mode == 'label':
        asyncio.run(label_phase(args))
    elif args.mode == 'full':
        asyncio.run(full_cycle(args))
    elif args.mode == 'historical':
        asyncio.run(historical_auto_label_phase(args))
    elif args.mode == 'stats':
        asyncio.run(stats_phase(args))


if __name__ == '__main__':
    main()
