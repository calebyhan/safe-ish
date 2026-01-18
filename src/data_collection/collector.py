"""Historical data collector for training ML models"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .dex_api import DexScreenerAPI
from .features import TokenFeatures
from ..utils.database import Database


class HistoricalDataCollector:
    """Collects historical token data and labels outcomes for ML training"""

    def __init__(self, api: DexScreenerAPI, db: Database):
        """
        Initialize collector

        Args:
            api: DexScreener API instance
            db: Database instance for storing data
        """
        self.api = api
        self.db = db

    async def collect_training_data(
        self,
        num_tokens: int = 500,
        chains: List[str] = None,
        search_queries: List[str] = None,
        include_boosted: bool = True,
        max_token_age_hours: int = None
    ) -> Dict[str, int]:
        """
        Collect historical tokens for training

        Args:
            num_tokens: Target number of tokens to collect
            chains: List of chain IDs to collect from (default: ['solana', 'ethereum', 'base'])
            search_queries: Search queries to find tokens (default: diverse patterns)
            include_boosted: Whether to include recently boosted tokens (often newer)
            max_token_age_hours: Maximum token age in hours (None = no limit)

        Returns:
            Dict with collection statistics
        """
        if chains is None:
            chains = ['solana', 'ethereum', 'base']

        if search_queries is None:
            # Diverse search patterns to find various types of tokens
            search_queries = self._generate_diverse_queries()

        stats = {
            'tokens_collected': 0,
            'tokens_skipped': 0,
            'chains_processed': {},
            'errors': 0,
            'age_filtered': 0
        }

        collected_addresses = set()

        print(f"\n{'='*60}")
        print(f"Historical Data Collection Started")
        print(f"Target: {num_tokens} tokens")
        print(f"Chains: {', '.join(chains)}")
        if max_token_age_hours:
            print(f"Max token age: {max_token_age_hours} hours ({max_token_age_hours/24:.1f} days)")
        print(f"{'='*60}\n")

        # First, collect boosted tokens (often newer launches)
        if include_boosted:
            print("Collecting recently boosted tokens...")
            boosted_collected = await self._collect_boosted_tokens(
                chains, collected_addresses, max_token_age_hours, stats
            )
            print(f"Collected {boosted_collected} boosted tokens\n")

        # Then search using diverse queries
        for query in search_queries:
            if stats['tokens_collected'] >= num_tokens:
                break

            print(f"Searching for: {query}...")

            try:
                results = await self.api.search_pairs(query)

                if not results or 'pairs' not in results:
                    continue

                for pair in results['pairs']:
                    if stats['tokens_collected'] >= num_tokens:
                        break

                    chain_id = pair.get('chainId', '')
                    if chain_id not in chains:
                        continue

                    token_address = pair.get('baseToken', {}).get('address', '')

                    # Skip if already collected
                    if token_address in collected_addresses:
                        stats['tokens_skipped'] += 1
                        continue

                    try:
                        # Parse token data
                        parsed = self.api.parse_token_data(pair)

                        # Filter by age if specified
                        if max_token_age_hours and parsed['token_age_hours'] > max_token_age_hours:
                            stats['age_filtered'] += 1
                            continue

                        # Create features
                        features = TokenFeatures(
                            token_address=parsed['token_address'],
                            token_symbol=parsed['token_symbol'],
                            chain_id=parsed['chain_id'],
                            token_age_hours=parsed['token_age_hours'],
                            liquidity_usd=parsed['liquidity_usd'],
                            market_cap=parsed['market_cap'],
                            price_usd=parsed['price_usd'],
                            volume_5min=parsed['volume_5m'],
                            volume_1hour=parsed['volume_1h'],
                            volume_24hour=parsed['volume_24h'],
                            buy_count_5min=parsed['txns_5m_buys'],
                            sell_count_5min=parsed['txns_5m_sells'],
                            buy_count_1hour=parsed['txns_1h_buys'],
                            sell_count_1hour=parsed['txns_1h_sells'],
                            price_change_5min=parsed['price_change_5m'],
                            price_change_1hour=parsed['price_change_1h'],
                            price_change_24hour=parsed['price_change_24h'],
                        )

                        # Store initial snapshot
                        self._store_token_snapshot(features, 'initial')

                        collected_addresses.add(token_address)
                        stats['tokens_collected'] += 1
                        stats['chains_processed'][chain_id] = stats['chains_processed'].get(chain_id, 0) + 1

                        if stats['tokens_collected'] % 50 == 0:
                            print(f"Progress: {stats['tokens_collected']}/{num_tokens} tokens collected")

                    except Exception as e:
                        print(f"Error processing token {token_address}: {e}")
                        stats['errors'] += 1
                        continue

                # Rate limiting between searches
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                stats['errors'] += 1
                continue

        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"Tokens collected: {stats['tokens_collected']}")
        print(f"Tokens skipped (duplicates): {stats['tokens_skipped']}")
        if max_token_age_hours:
            print(f"Tokens filtered by age: {stats['age_filtered']}")
        print(f"Errors: {stats['errors']}")
        print(f"\nBy Chain:")
        for chain, count in stats['chains_processed'].items():
            print(f"  {chain}: {count}")
        print(f"{'='*60}\n")

        return stats

    def _generate_diverse_queries(self) -> List[str]:
        """
        Generate diverse search queries to find different types of tokens

        Returns:
            List of search query strings
        """
        import random
        import string

        queries = []

        # Popular meme/trending keywords (some legitimate, some potentially scammy)
        trending_keywords = [
            'PEPE', 'DOGE', 'SHIB', 'FLOKI', 'BONK', 'WIF', 'MEME',
            'ELON', 'WOJAK', 'AI', 'MOON', 'SAFE', 'GEM', 'ROCKET',
            'DEGEN', 'APE', 'INU', 'CAT', 'FROG', 'BOT', 'DAO',
            'CHAD', 'BEAR', 'BULL', 'PUMP', 'LAMBO', 'WAGMI'
        ]

        # Random short strings (to find obscure tokens)
        random_strings = [
            ''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(10)
        ]

        # Common scam patterns
        scam_patterns = [
            'BABY', 'MINI', 'MEGA', 'ULTRA', 'SUPER', 'HYPER',
            'MOON', 'MARS', 'SAFE', 'SECURE', 'VAULT', 'GOLD'
        ]

        # Combine established + random + scam-like patterns
        queries.extend(trending_keywords[:10])  # Some established tokens
        queries.extend(random_strings)  # Random discoveries
        queries.extend(scam_patterns)  # Potential scam patterns

        # Shuffle for variety
        random.shuffle(queries)

        return queries

    async def _collect_boosted_tokens(
        self,
        chains: List[str],
        collected_addresses: set,
        max_token_age_hours: Optional[int],
        stats: Dict
    ) -> int:
        """
        Collect recently boosted tokens (often newer launches)

        Args:
            chains: List of allowed chains
            collected_addresses: Set of already collected addresses
            max_token_age_hours: Maximum token age filter
            stats: Stats dictionary to update

        Returns:
            Number of boosted tokens collected
        """
        collected_count = 0

        try:
            boosted_tokens = await self.api.get_boosted_tokens()

            for boost_data in boosted_tokens:
                chain_id = boost_data.get('chainId', '')
                token_address = boost_data.get('tokenAddress', '')

                if chain_id not in chains:
                    continue

                if token_address in collected_addresses:
                    continue

                try:
                    # Get full token data
                    token_data = await self.api.get_token_pairs(chain_id, token_address)

                    if not token_data or 'pairs' not in token_data or len(token_data['pairs']) == 0:
                        continue

                    # Use first pair
                    pair = token_data['pairs'][0]
                    parsed = self.api.parse_token_data(pair)

                    # Filter by age if specified
                    if max_token_age_hours and parsed['token_age_hours'] > max_token_age_hours:
                        stats['age_filtered'] += 1
                        continue

                    # Create features
                    features = TokenFeatures(
                        token_address=parsed['token_address'],
                        token_symbol=parsed['token_symbol'],
                        chain_id=parsed['chain_id'],
                        token_age_hours=parsed['token_age_hours'],
                        liquidity_usd=parsed['liquidity_usd'],
                        market_cap=parsed['market_cap'],
                        price_usd=parsed['price_usd'],
                        volume_5min=parsed['volume_5m'],
                        volume_1hour=parsed['volume_1h'],
                        volume_24hour=parsed['volume_24h'],
                        buy_count_5min=parsed['txns_5m_buys'],
                        sell_count_5min=parsed['txns_5m_sells'],
                        buy_count_1hour=parsed['txns_1h_buys'],
                        sell_count_1hour=parsed['txns_1h_sells'],
                        price_change_5min=parsed['price_change_5m'],
                        price_change_1hour=parsed['price_change_1h'],
                        price_change_24hour=parsed['price_change_24h'],
                    )

                    # Store initial snapshot
                    self._store_token_snapshot(features, 'initial')

                    collected_addresses.add(token_address)
                    collected_count += 1
                    stats['tokens_collected'] += 1
                    stats['chains_processed'][chain_id] = stats['chains_processed'].get(chain_id, 0) + 1

                    # Rate limiting
                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"Error collecting boosted token {token_address}: {e}")
                    stats['errors'] += 1

        except Exception as e:
            print(f"Error fetching boosted tokens: {e}")
            stats['errors'] += 1

        return collected_count

    async def collect_and_auto_label_historical(
        self,
        num_tokens: int = 500,
        min_token_age_hours: int = 48,
        max_token_age_hours: int = 720,
        chains: List[str] = None
    ) -> Dict[str, int]:
        """
        Collect historical tokens and auto-label based on their outcomes

        This method collects tokens that are old enough to have historical data,
        then immediately labels them based on their price/liquidity history.
        No waiting period required.

        Args:
            num_tokens: Target number of tokens to collect
            min_token_age_hours: Minimum token age (default: 48hrs = 2 days)
            max_token_age_hours: Maximum token age (default: 720hrs = 30 days)
            chains: List of chain IDs (default: ['solana', 'ethereum', 'base'])

        Returns:
            Dict with collection and labeling statistics
        """
        if chains is None:
            chains = ['solana', 'ethereum', 'base']

        stats = {
            'tokens_collected': 0,
            'tokens_labeled': 0,
            'labels': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            'tokens_skipped': 0,
            'errors': 0,
            'age_filtered': 0
        }

        collected_addresses = set()

        print(f"\n{'='*60}")
        print(f"Historical Auto-Label Collection Started")
        print(f"Target: {num_tokens} tokens")
        print(f"Age range: {min_token_age_hours}-{max_token_age_hours} hours")
        max_days_str = f"{max_token_age_hours/24:.1f}" if max_token_age_hours else "present"
        print(f"           ({min_token_age_hours/24:.1f}-{max_days_str} days)")
        print(f"Chains: {', '.join(chains)}")
        print(f"{'='*60}\n")

        # Generate diverse search queries
        search_queries = self._generate_diverse_queries()

        for query in search_queries:
            if stats['tokens_collected'] >= num_tokens:
                break

            print(f"Searching for: {query}...")

            try:
                results = await self.api.search_pairs(query)

                for pair in results.get('pairs', [])[:20]:  # Limit per query
                    if stats['tokens_collected'] >= num_tokens:
                        break

                    token_address = pair.get('baseToken', {}).get('address', '')

                    if not token_address or token_address in collected_addresses:
                        stats['tokens_skipped'] += 1
                        continue

                    try:
                        parsed = self.api.parse_token_data(pair)
                        token_age = parsed['token_age_hours']

                        # Filter by age range
                        if token_age < min_token_age_hours:
                            stats['age_filtered'] += 1
                            continue

                        if max_token_age_hours and token_age > max_token_age_hours:
                            stats['age_filtered'] += 1
                            continue

                        # Create features
                        features = TokenFeatures(
                            token_address=parsed['token_address'],
                            token_symbol=parsed['token_symbol'],
                            chain_id=parsed['chain_id'],
                            token_age_hours=parsed['token_age_hours'],
                            liquidity_usd=parsed['liquidity_usd'],
                            market_cap=parsed['market_cap'],
                            price_usd=parsed['price_usd'],
                            volume_5min=parsed['volume_5m'],
                            volume_1hour=parsed['volume_1h'],
                            volume_24hour=parsed['volume_24h'],
                            buy_count_5min=parsed['txns_5m_buys'],
                            sell_count_5min=parsed['txns_5m_sells'],
                            buy_count_1hour=parsed['txns_1h_buys'],
                            sell_count_1hour=parsed['txns_1h_sells'],
                            price_change_5min=parsed['price_change_5m'],
                            price_change_1hour=parsed['price_change_1h'],
                            price_change_24hour=parsed['price_change_24h'],
                        )

                        # Store snapshot
                        self._store_token_snapshot(features, 'initial')

                        # Auto-label based on historical data
                        label = self._auto_label_from_history(parsed)

                        # Store label
                        outcome_data = {
                            'method': 'historical_auto_label',
                            'price_change_24h': parsed.get('price_change_24h', 0),
                            'liquidity_usd': parsed.get('liquidity_usd', 0),
                            'token_age_hours': parsed.get('token_age_hours', 0)
                        }
                        self._update_token_label(token_address, label, outcome_data)

                        collected_addresses.add(token_address)
                        stats['tokens_collected'] += 1
                        stats['tokens_labeled'] += 1
                        stats['labels'][label] += 1

                        # Rate limiting
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        print(f"Error processing token {token_address}: {e}")
                        stats['errors'] += 1

            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                stats['errors'] += 1

        # Print results
        print(f"\n{'='*60}")
        print(f"Historical Auto-Label Collection Complete!")
        print(f"Tokens collected: {stats['tokens_collected']}")
        print(f"Tokens labeled: {stats['tokens_labeled']}")
        print(f"\nLabel Distribution:")
        label_names = {0: 'Rug Pull', 1: 'Pump & Dump', 2: 'Wash Trading', 3: 'Legitimate', 4: 'Unknown'}
        for label_id, count in stats['labels'].items():
            if count > 0:
                pct = (count / stats['tokens_labeled'] * 100) if stats['tokens_labeled'] > 0 else 0
                print(f"  {label_names[label_id]}: {count} ({pct:.1f}%)")
        print(f"\nTokens filtered by age: {stats['age_filtered']}")
        print(f"Errors: {stats['errors']}")
        print(f"{'='*60}\n")

        return stats

    def _auto_label_from_history(self, parsed_data: Dict) -> int:
        """
        Auto-label token based on historical data already in API response

        Labels:
            0 = RUG_PULL (price crash + liquidity gone)
            1 = PUMP_DUMP (pumped then crashed)
            2 = WASH_TRADING (extreme volume/liquidity ratio)
            3 = LEGITIMATE (healthy metrics)
            4 = UNKNOWN (unclear)

        Args:
            parsed_data: Parsed token data from API

        Returns:
            Label (0-4)
        """
        price_change_1h = parsed_data.get('price_change_1h', 0)
        price_change_24h = parsed_data.get('price_change_24h', 0)
        liquidity = parsed_data.get('liquidity_usd', 0)
        volume_24h = parsed_data.get('volume_24h', 0)
        token_age = parsed_data.get('token_age_hours', 0)

        # Calculate volume to liquidity ratio
        vol_to_liq = volume_24h / liquidity if liquidity > 0 else 0

        # RUG PULL: Massive price crash + very low liquidity
        if price_change_24h < -80 and liquidity < 1000:
            return 0

        # RUG PULL: Severe crash with disappeared liquidity
        if price_change_24h < -70 and liquidity < 500:
            return 0

        # RUG PULL: Moderate crash but very old token with no liquidity
        if price_change_24h < -60 and liquidity < 200 and token_age > 48:
            return 0

        # PUMP & DUMP: Big pump followed by crash
        if price_change_1h > 100 and price_change_24h < -40:
            return 1

        # PUMP & DUMP: Severe 24h crash (likely dumped after pump)
        if price_change_24h < -70 and liquidity > 500 and token_age > 24:
            return 1

        # PUMP & DUMP: Moderate crash with some liquidity remaining
        if price_change_24h < -55 and liquidity > 1000 and token_age > 72:
            return 1

        # WASH TRADING: Extremely high volume/liquidity ratio
        if vol_to_liq > 30:
            return 2

        # WASH TRADING: High ratio with low liquidity
        if vol_to_liq > 15 and liquidity < 5000:
            return 2

        # WASH TRADING: Moderate ratio with very low liquidity
        if vol_to_liq > 10 and liquidity < 2000:
            return 2

        # LEGITIMATE: Excellent liquidity, stable price, old enough
        if liquidity > 50000 and abs(price_change_24h) < 30 and token_age > 168:
            return 3

        # LEGITIMATE: Good liquidity, reasonable price movement
        if liquidity > 10000 and abs(price_change_24h) < 60 and token_age > 72:
            return 3

        # LEGITIMATE: Decent metrics for newer tokens
        if liquidity > 5000 and abs(price_change_24h) < 70 and token_age > 48:
            return 3

        # UNKNOWN: Doesn't fit clear patterns
        return 4

    async def check_token_fate(
        self,
        token_address: str,
        chain_id: str,
        hours_later: int = 24
    ) -> Tuple[int, Dict]:
        """
        Check token outcome after specified hours

        Labels:
            0 = RUG_PULL (95% crash or 80% liquidity removal)
            1 = PUMP_DUMP (5x pump then 70% crash)
            2 = WASH_TRADING (50%+ circular volume)
            3 = LEGITIMATE (stable, still trading)
            4 = UNKNOWN (unclear outcome or error)

        Args:
            token_address: Token address to check
            chain_id: Blockchain ID
            hours_later: Hours to wait before checking

        Returns:
            Tuple of (label, outcome_data)
        """
        try:
            # Get initial snapshot
            initial = self._get_token_snapshot(token_address, 'initial')
            if not initial:
                return 4, {'error': 'No initial snapshot found'}

            # Wait for specified time (in real usage)
            # For testing, we'll fetch current state
            await asyncio.sleep(1)  # Minimal delay for API

            # Search for token to get current state
            results = await self.api.search_pairs(token_address[:8])

            if not results or 'pairs' not in results:
                return 4, {'error': 'Token not found'}

            # Find matching pair
            current_pair = None
            for pair in results['pairs']:
                if (pair.get('baseToken', {}).get('address', '') == token_address and
                    pair.get('chainId', '') == chain_id):
                    current_pair = pair
                    break

            if not current_pair:
                return 4, {'error': 'Pair not found'}

            # Parse current data
            current = self.api.parse_token_data(current_pair)

            # Calculate changes
            initial_price = initial.get('price_usd', 0)
            current_price = current.get('price_usd', 0)
            initial_liquidity = initial.get('liquidity_usd', 0)
            current_liquidity = current.get('liquidity_usd', 0)

            if initial_price == 0:
                return 4, {'error': 'Invalid initial price'}

            price_change_pct = ((current_price - initial_price) / initial_price) * 100

            if initial_liquidity > 0:
                liquidity_change_pct = ((current_liquidity - initial_liquidity) / initial_liquidity) * 100
            else:
                liquidity_change_pct = 0

            # Determine label
            outcome_data = {
                'initial_price': initial_price,
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'initial_liquidity': initial_liquidity,
                'current_liquidity': current_liquidity,
                'liquidity_change_pct': liquidity_change_pct,
                'hours_elapsed': hours_later
            }

            # RUG PULL: 95% price crash OR 80% liquidity removal
            if price_change_pct <= -95 or liquidity_change_pct <= -80:
                return 0, outcome_data

            # PUMP_DUMP: Check if there was a pump (requires historical tracking)
            # For now, detect severe crash after being listed
            if price_change_pct <= -70:
                return 1, outcome_data

            # WASH_TRADING: Detect via volume patterns (simplified)
            # This would require more sophisticated analysis
            volume_24h = current.get('volume_24h', 0)
            if volume_24h > 0 and current_liquidity > 0:
                volume_to_liquidity = volume_24h / current_liquidity
                if volume_to_liquidity > 10:  # Extremely high turnover ratio
                    return 2, outcome_data

            # LEGITIMATE: Stable and still trading
            if abs(price_change_pct) < 50 and liquidity_change_pct > -30:
                return 3, outcome_data

            # UNKNOWN: Can't clearly categorize
            return 4, outcome_data

        except Exception as e:
            return 4, {'error': str(e)}

    async def label_collected_tokens(
        self,
        hours_to_wait: int = 24,
        batch_size: int = 50
    ) -> Dict[str, int]:
        """
        Label previously collected tokens by checking their fate

        Args:
            hours_to_wait: Hours to wait before checking outcomes
            batch_size: Process this many tokens at a time

        Returns:
            Dict with labeling statistics
        """
        stats = {
            'rug_pull': 0,
            'pump_dump': 0,
            'wash_trading': 0,
            'legitimate': 0,
            'unknown': 0,
            'errors': 0
        }

        label_names = {
            0: 'rug_pull',
            1: 'pump_dump',
            2: 'wash_trading',
            3: 'legitimate',
            4: 'unknown'
        }

        # Get unlabeled tokens from database
        unlabeled = self._get_unlabeled_tokens(limit=batch_size)

        print(f"\n{'='*60}")
        print(f"Labeling Tokens")
        print(f"Tokens to label: {len(unlabeled)}")
        print(f"{'='*60}\n")

        for i, token in enumerate(unlabeled, 1):
            token_address = token['token_address']
            chain_id = token['chain_id']

            try:
                label, outcome_data = await self.check_token_fate(
                    token_address, chain_id, hours_to_wait
                )

                # Store label in database
                self._update_token_label(token_address, label, outcome_data)

                label_name = label_names[label]
                stats[label_name] += 1

                print(f"[{i}/{len(unlabeled)}] {token_address[:16]}... → {label_name.upper()}")

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Error labeling {token_address}: {e}")
                stats['errors'] += 1

        print(f"\n{'='*60}")
        print(f"Labeling Complete!")
        print(f"Rug Pull: {stats['rug_pull']}")
        print(f"Pump & Dump: {stats['pump_dump']}")
        print(f"Wash Trading: {stats['wash_trading']}")
        print(f"Legitimate: {stats['legitimate']}")
        print(f"Unknown: {stats['unknown']}")
        print(f"Errors: {stats['errors']}")
        print(f"{'='*60}\n")

        return stats

    def _store_token_snapshot(self, features: TokenFeatures, snapshot_type: str):
        """Store token snapshot in database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO token_snapshots (
                    token_address, chain_id, snapshot_type, snapshot_time,
                    token_symbol, token_age_hours, price_usd, liquidity_usd,
                    market_cap, volume_5min, volume_1hour, volume_24hour,
                    buy_count_5min, sell_count_5min, buy_count_1hour, sell_count_1hour,
                    price_change_5min, price_change_1hour, price_change_24hour
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                features.token_address,
                features.chain_id,
                snapshot_type,
                datetime.utcnow().isoformat(),
                features.token_symbol,
                features.token_age_hours,
                features.price_usd,
                features.liquidity_usd,
                features.market_cap,
                features.volume_5min,
                features.volume_1hour,
                features.volume_24hour,
                features.buy_count_5min,
                features.sell_count_5min,
                features.buy_count_1hour,
                features.sell_count_1hour,
                features.price_change_5min,
                features.price_change_1hour,
                features.price_change_24hour
            ))

    def _get_token_snapshot(self, token_address: str, snapshot_type: str) -> Optional[Dict]:
        """Retrieve token snapshot from database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM token_snapshots
                WHERE token_address = ? AND snapshot_type = ?
                ORDER BY snapshot_time DESC
                LIMIT 1
            """, (token_address, snapshot_type))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def _get_unlabeled_tokens(self, limit: int = 100) -> List[Dict]:
        """Get tokens that haven't been labeled yet"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT token_address, chain_id
                FROM token_snapshots
                WHERE snapshot_type = 'initial'
                AND token_address NOT IN (
                    SELECT token_address FROM token_labels
                )
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def _update_token_label(self, token_address: str, label: int, outcome_data: Dict):
        """Update token with outcome label"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO token_labels (
                    token_address, label, label_time,
                    outcome_data
                )
                VALUES (?, ?, ?, ?)
            """, (
                token_address,
                label,
                datetime.utcnow().isoformat(),
                str(outcome_data)
            ))

    def get_labeled_dataset(self) -> Tuple[List[Dict], List[int]]:
        """
        Get all labeled tokens for ML training

        Returns:
            Tuple of (features_list, labels_list)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, l.label
                FROM token_snapshots s
                JOIN token_labels l ON s.token_address = l.token_address
                WHERE s.snapshot_type = 'initial'
                AND l.label IN (0, 1, 2, 3)
            """)

            rows = cursor.fetchall()

            features = []
            labels = []

            for row in rows:
                row_dict = dict(row)
                label = row_dict.pop('label')
                labels.append(label)

                # Extract only ML features (exclude metadata)
                feature_dict = {
                    'token_age_hours': row_dict['token_age_hours'],
                    'liquidity_usd': row_dict['liquidity_usd'],
                    'market_cap': row_dict['market_cap'],
                    'price_usd': row_dict['price_usd'],
                    'volume_5min': row_dict['volume_5min'],
                    'volume_1hour': row_dict['volume_1hour'],
                    'volume_24hour': row_dict['volume_24hour'],
                    'buy_count_5min': row_dict['buy_count_5min'],
                    'sell_count_5min': row_dict['sell_count_5min'],
                    'buy_count_1hour': row_dict['buy_count_1hour'],
                    'sell_count_1hour': row_dict['sell_count_1hour'],
                    'price_change_5min': row_dict['price_change_5min'],
                    'price_change_1hour': row_dict['price_change_1hour'],
                    'price_change_24hour': row_dict['price_change_24hour'],
                }

                features.append(feature_dict)

            return features, labels
