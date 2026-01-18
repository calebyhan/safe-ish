#!/usr/bin/env python3
"""
CLI script for collecting historical OHLCV data for backtesting

Uses GeckoTerminal API to fetch historical candlestick data for:
- Specific pools/pairs
- Trending tokens
- Top volume tokens
- Tokens from your watchlist
"""
import asyncio
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.geckoterminal_api import GeckoTerminalAPI, Timeframe, OHLCV
from src.utils.database import Database


class OHLCVCollector:
    """Collects and stores historical OHLCV data"""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self.api: Optional[GeckoTerminalAPI] = None
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure OHLCV tables exist in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create OHLCV data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_address TEXT NOT NULL,
                network TEXT NOT NULL,
                token_symbol TEXT,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pool_address, network, timeframe, timestamp)
            )
        """)

        # Create pools metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pool_metadata (
                pool_address TEXT PRIMARY KEY,
                network TEXT NOT NULL,
                name TEXT,
                base_token_symbol TEXT,
                quote_token_symbol TEXT,
                dex TEXT,
                reserve_usd REAL,
                fdv_usd REAL,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_pool_time
            ON ohlcv_data(pool_address, timeframe, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp
            ON ohlcv_data(timestamp)
        """)

        conn.commit()
        conn.close()

    def _save_pool_metadata(self, pool_data: Dict):
        """Save pool metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pool_metadata (
                pool_address, network, name, base_token_symbol,
                quote_token_symbol, dex, reserve_usd, fdv_usd, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(pool_address) DO UPDATE SET
                name = excluded.name,
                reserve_usd = excluded.reserve_usd,
                fdv_usd = excluded.fdv_usd,
                last_updated = CURRENT_TIMESTAMP
        """, (
            pool_data['pool_address'],
            pool_data['network'],
            pool_data['name'],
            pool_data['base_token_symbol'],
            pool_data['quote_token_symbol'],
            pool_data['dex'],
            pool_data['reserve_in_usd'],
            pool_data['fdv_usd']
        ))

        conn.commit()
        conn.close()

    def _save_ohlcv_data(
        self,
        pool_address: str,
        network: str,
        token_symbol: str,
        timeframe: str,
        candles: List[OHLCV]
    ) -> int:
        """Save OHLCV data to database"""
        if not candles:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved = 0
        for candle in candles:
            try:
                cursor.execute("""
                    INSERT INTO ohlcv_data (
                        pool_address, network, token_symbol, timeframe,
                        timestamp, open, high, low, close, volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pool_address, network, timeframe, timestamp) DO NOTHING
                """, (
                    pool_address,
                    network,
                    token_symbol,
                    timeframe,
                    candle.timestamp.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume
                ))
                if cursor.rowcount > 0:
                    saved += 1
            except Exception as e:
                print(f"Error saving candle: {e}")

        conn.commit()
        conn.close()
        return saved

    async def collect_pool_history(
        self,
        network: str,
        pool_address: str,
        timeframe: Timeframe = Timeframe.HOUR_1,
        days: int = 30
    ) -> Dict:
        """
        Collect historical OHLCV for a specific pool

        Args:
            network: Network ID (solana, ethereum, base, etc.)
            pool_address: Pool contract address
            timeframe: Candle timeframe
            days: Days of history to fetch

        Returns:
            Stats about collected data
        """
        print(f"\nCollecting {days} days of {timeframe.name} data for {pool_address[:16]}...")

        # Get pool info first
        pool_info = await self.api.get_pool_info(network, pool_address)
        if pool_info:
            parsed = self.api.parse_pool_data(pool_info)
            self._save_pool_metadata(parsed)
            token_symbol = parsed['base_token_symbol']
            print(f"Pool: {parsed['name']}")
        else:
            token_symbol = "UNKNOWN"
            print("Warning: Could not fetch pool info")

        # Fetch OHLCV data
        candles = await self.api.get_extended_ohlcv(
            network=network,
            pool_address=pool_address,
            timeframe=timeframe,
            days=days
        )

        if not candles:
            print("No candle data received")
            return {'candles': 0, 'saved': 0}

        # Save to database
        saved = self._save_ohlcv_data(
            pool_address, network, token_symbol, timeframe.name, candles
        )

        print(f"Fetched {len(candles)} candles, saved {saved} new records")

        return {
            'pool_address': pool_address,
            'network': network,
            'token_symbol': token_symbol,
            'timeframe': timeframe.name,
            'candles_fetched': len(candles),
            'candles_saved': saved,
            'date_range': (
                candles[0].timestamp.isoformat() if candles else None,
                candles[-1].timestamp.isoformat() if candles else None
            )
        }

    async def collect_from_search(
        self,
        query: str,
        network: Optional[str] = None,
        timeframe: Timeframe = Timeframe.HOUR_1,
        days: int = 30,
        max_pools: int = 5
    ) -> List[Dict]:
        """
        Search for pools and collect their historical data

        Args:
            query: Search term (token symbol/name)
            network: Optional network filter
            timeframe: Candle timeframe
            days: Days of history
            max_pools: Maximum pools to collect

        Returns:
            List of collection stats
        """
        print(f"\nSearching for '{query}' pools...")

        pools = await self.api.search_pools(query, network=network)

        if not pools:
            print("No pools found")
            return []

        print(f"Found {len(pools)} pools, collecting top {max_pools}")

        results = []
        for pool in pools[:max_pools]:
            parsed = self.api.parse_pool_data(pool)

            # Skip low liquidity pools
            if parsed['reserve_in_usd'] < 1000:
                print(f"Skipping {parsed['name']} - low liquidity (${parsed['reserve_in_usd']:.0f})")
                continue

            try:
                stats = await self.collect_pool_history(
                    network=parsed['network'],
                    pool_address=parsed['pool_address'],
                    timeframe=timeframe,
                    days=days
                )
                results.append(stats)

                # Small delay between pools
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error collecting {parsed['name']}: {e}")

        return results

    async def collect_trending(
        self,
        network: Optional[str] = None,
        timeframe: Timeframe = Timeframe.HOUR_1,
        days: int = 30,
        max_pools: int = 10
    ) -> List[Dict]:
        """
        Collect historical data for trending pools

        Args:
            network: Optional network filter
            timeframe: Candle timeframe
            days: Days of history
            max_pools: Maximum pools to collect

        Returns:
            List of collection stats
        """
        print(f"\nFetching trending pools...")

        pools = await self.api.get_trending_pools(network=network)

        if not pools:
            print("No trending pools found")
            return []

        print(f"Found {len(pools)} trending pools, collecting top {max_pools}")

        results = []
        for pool in pools[:max_pools]:
            parsed = self.api.parse_pool_data(pool)

            try:
                stats = await self.collect_pool_history(
                    network=parsed['network'],
                    pool_address=parsed['pool_address'],
                    timeframe=timeframe,
                    days=days
                )
                results.append(stats)
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error collecting {parsed['name']}: {e}")

        return results

    async def collect_top_volume(
        self,
        network: str,
        timeframe: Timeframe = Timeframe.HOUR_1,
        days: int = 30,
        max_pools: int = 10
    ) -> List[Dict]:
        """
        Collect historical data for top volume pools

        Args:
            network: Network to scan
            timeframe: Candle timeframe
            days: Days of history
            max_pools: Maximum pools to collect

        Returns:
            List of collection stats
        """
        print(f"\nFetching top volume pools on {network}...")

        pools = await self.api.get_top_pools(network=network)

        if not pools:
            print("No pools found")
            return []

        print(f"Found {len(pools)} pools, collecting top {max_pools}")

        results = []
        for pool in pools[:max_pools]:
            parsed = self.api.parse_pool_data(pool)

            try:
                stats = await self.collect_pool_history(
                    network=parsed['network'],
                    pool_address=parsed['pool_address'],
                    timeframe=timeframe,
                    days=days
                )
                results.append(stats)
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error collecting {parsed['name']}: {e}")

        return results

    def get_stats(self) -> Dict:
        """Get statistics about collected OHLCV data"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total candles
        cursor.execute("SELECT COUNT(*) as count FROM ohlcv_data")
        total_candles = cursor.fetchone()['count']

        # Unique pools
        cursor.execute("SELECT COUNT(DISTINCT pool_address) as count FROM ohlcv_data")
        unique_pools = cursor.fetchone()['count']

        # By network
        cursor.execute("""
            SELECT network, COUNT(*) as candles, COUNT(DISTINCT pool_address) as pools
            FROM ohlcv_data
            GROUP BY network
        """)
        by_network = [dict(row) for row in cursor.fetchall()]

        # By timeframe
        cursor.execute("""
            SELECT timeframe, COUNT(*) as count
            FROM ohlcv_data
            GROUP BY timeframe
        """)
        by_timeframe = {row['timeframe']: row['count'] for row in cursor.fetchall()}

        # Date range
        cursor.execute("""
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM ohlcv_data
        """)
        date_range = cursor.fetchone()

        conn.close()

        return {
            'total_candles': total_candles,
            'unique_pools': unique_pools,
            'by_network': by_network,
            'by_timeframe': by_timeframe,
            'date_range': {
                'earliest': date_range['earliest'],
                'latest': date_range['latest']
            }
        }


async def run_collection(args):
    """Run OHLCV collection based on arguments"""
    collector = OHLCVCollector(db_path=args.db_path)

    # Map timeframe string to enum
    timeframe_map = {
        '1m': Timeframe.MINUTE_1,
        '5m': Timeframe.MINUTE_5,
        '15m': Timeframe.MINUTE_15,
        '1h': Timeframe.HOUR_1,
        '4h': Timeframe.HOUR_4,
        '12h': Timeframe.HOUR_12,
        '1d': Timeframe.DAY_1
    }
    timeframe = timeframe_map.get(args.timeframe, Timeframe.HOUR_1)

    async with GeckoTerminalAPI() as api:
        collector.api = api

        if args.mode == 'pool':
            # Collect specific pool
            if not args.pool_address:
                print("Error: --pool-address required for pool mode")
                return

            await collector.collect_pool_history(
                network=args.network,
                pool_address=args.pool_address,
                timeframe=timeframe,
                days=args.days
            )

        elif args.mode == 'search':
            # Search and collect
            if not args.query:
                print("Error: --query required for search mode")
                return

            await collector.collect_from_search(
                query=args.query,
                network=args.network if args.network != 'all' else None,
                timeframe=timeframe,
                days=args.days,
                max_pools=args.max_pools
            )

        elif args.mode == 'trending':
            # Collect trending pools
            await collector.collect_trending(
                network=args.network if args.network != 'all' else None,
                timeframe=timeframe,
                days=args.days,
                max_pools=args.max_pools
            )

        elif args.mode == 'top':
            # Collect top volume pools
            networks = [args.network] if args.network != 'all' else ['solana', 'ethereum', 'base']

            for network in networks:
                await collector.collect_top_volume(
                    network=network,
                    timeframe=timeframe,
                    days=args.days,
                    max_pools=args.max_pools
                )

        elif args.mode == 'meme':
            # Collect popular meme coins
            meme_queries = ['BONK', 'WIF', 'PEPE', 'SHIB', 'DOGE', 'FLOKI', 'BRETT', 'MOG']

            for query in meme_queries:
                print(f"\n{'='*50}")
                print(f"Collecting {query}")
                print('='*50)

                await collector.collect_from_search(
                    query=query,
                    network=args.network if args.network != 'all' else None,
                    timeframe=timeframe,
                    days=args.days,
                    max_pools=3  # Top 3 pools per meme coin
                )

                await asyncio.sleep(2)

        elif args.mode == 'stats':
            # Show statistics
            stats = collector.get_stats()

            print("\n" + "="*60)
            print("OHLCV DATA STATISTICS")
            print("="*60)
            print(f"\nTotal Candles: {stats['total_candles']:,}")
            print(f"Unique Pools: {stats['unique_pools']}")

            if stats['by_network']:
                print("\nBy Network:")
                for net in stats['by_network']:
                    print(f"  {net['network']}: {net['candles']:,} candles, {net['pools']} pools")

            if stats['by_timeframe']:
                print("\nBy Timeframe:")
                for tf, count in stats['by_timeframe'].items():
                    print(f"  {tf}: {count:,} candles")

            if stats['date_range']['earliest']:
                print(f"\nDate Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect historical OHLCV data for backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 30 days of hourly data for trending pools
  python scripts/collect_ohlcv_data.py --mode trending --days 30

  # Collect 90 days of 4-hour data for top Solana pools
  python scripts/collect_ohlcv_data.py --mode top --network solana --days 90 --timeframe 4h

  # Collect data for specific pool
  python scripts/collect_ohlcv_data.py --mode pool --network solana --pool-address <ADDRESS> --days 60

  # Search and collect data for BONK pools
  python scripts/collect_ohlcv_data.py --mode search --query BONK --days 30

  # Collect popular meme coins (BONK, WIF, PEPE, etc.)
  python scripts/collect_ohlcv_data.py --mode meme --days 60

  # Show collection statistics
  python scripts/collect_ohlcv_data.py --mode stats
        """
    )

    parser.add_argument(
        '--mode',
        choices=['pool', 'search', 'trending', 'top', 'meme', 'stats'],
        default='stats',
        help='Collection mode'
    )

    parser.add_argument(
        '--network',
        default='solana',
        help='Network to collect from (solana, ethereum, base, all)'
    )

    parser.add_argument(
        '--pool-address',
        help='Specific pool address (for pool mode)'
    )

    parser.add_argument(
        '--query',
        help='Search query (for search mode)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of history to collect (default: 30, max: ~180)'
    )

    parser.add_argument(
        '--timeframe',
        choices=['1m', '5m', '15m', '1h', '4h', '12h', '1d'],
        default='1h',
        help='Candle timeframe (default: 1h)'
    )

    parser.add_argument(
        '--max-pools',
        type=int,
        default=10,
        help='Maximum pools to collect per query (default: 10)'
    )

    parser.add_argument(
        '--db-path',
        default='data/trading.db',
        help='Database path (default: data/trading.db)'
    )

    args = parser.parse_args()
    asyncio.run(run_collection(args))


if __name__ == '__main__':
    main()
