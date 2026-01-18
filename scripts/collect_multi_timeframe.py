#!/usr/bin/env python3
"""
Collect OHLCV data across multiple timeframes for trending pools
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_ohlcv_data import OHLCVCollector
from src.data_collection.geckoterminal_api import GeckoTerminalAPI, Timeframe


async def collect_multi_timeframe():
    """Collect trending pools with multiple timeframes"""

    # Timeframes to collect (balance between detail and API limits)
    timeframes_config = [
        (Timeframe.MINUTE_15, 7),   # 15min candles, 7 days
        (Timeframe.HOUR_1, 30),      # 1h candles, 30 days
        (Timeframe.HOUR_4, 60),      # 4h candles, 60 days
        (Timeframe.DAY_1, 180),      # 1d candles, 180 days
    ]

    collector = OHLCVCollector(db_path="data/trading.db")

    async with GeckoTerminalAPI() as api:
        collector.api = api

        print("Fetching trending Solana pools...")
        pools = await api.get_trending_pools(network='solana')

        if not pools:
            print("No trending pools found")
            return

        # Get top 15 trending pools
        max_pools = 15
        print(f"Found {len(pools)} trending pools, collecting top {max_pools}\n")

        for i, pool in enumerate(pools[:max_pools], 1):
            parsed = api.parse_pool_data(pool)

            print(f"\n{'='*60}")
            print(f"[{i}/{max_pools}] {parsed['name']}")
            print(f"Liquidity: ${parsed['reserve_in_usd']:,.0f}")
            print('='*60)

            # Collect each timeframe for this pool
            for timeframe, days in timeframes_config:
                try:
                    print(f"\n  Collecting {days} days of {timeframe.name} data...")

                    stats = await collector.collect_pool_history(
                        network=parsed['network'],
                        pool_address=parsed['pool_address'],
                        timeframe=timeframe,
                        days=days
                    )

                    print(f"    ✓ Saved {stats['candles_saved']} new candles")

                    # Small delay between timeframes
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"    ✗ Error: {e}")

            # Delay between pools to respect rate limits
            if i < max_pools:
                print(f"\n  Waiting 2s before next pool...")
                await asyncio.sleep(2)

        # Print final stats
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        stats = collector.get_stats()
        print(f"Total Candles: {stats['total_candles']:,}")
        print(f"Unique Pools: {stats['unique_pools']}")
        if stats['by_timeframe']:
            print("\nBy Timeframe:")
            for tf, count in stats['by_timeframe'].items():
                print(f"  {tf}: {count:,} candles")


if __name__ == '__main__':
    asyncio.run(collect_multi_timeframe())
