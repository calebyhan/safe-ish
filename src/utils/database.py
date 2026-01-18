import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional


class Database:
    """SQLite database manager for trading data"""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Trades table - tracks all position lifecycle
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    chain_id TEXT,
                    strategy TEXT,
                    action TEXT,  -- 'OPEN' or 'CLOSE'
                    price REAL,
                    size_usd REAL,
                    ml_risk_score REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    confidence REAL,
                    close_reason TEXT,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    hold_time_hours REAL
                )
            """)

            # ML assessments table - stores ML predictions and outcomes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    chain_id TEXT,
                    risk_score REAL,
                    recommendation TEXT,  -- REJECT/CAUTION/MONITOR/PASS
                    rug_pull_risk REAL,
                    wash_trading_risk REAL,
                    pump_dump_risk REAL,
                    outcome_label INTEGER,  -- Actual outcome after 24h (0-4)
                    outcome_verified BOOLEAN DEFAULT 0,
                    features_json TEXT  -- JSON string of features for retraining
                )
            """)

            # Filter stats table - tracks filter performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filter_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_scanned INTEGER,
                    approved INTEGER,
                    rejected INTEGER,
                    rejection_rate REAL
                )
            """)

            # Portfolio snapshots table - periodic portfolio state
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    capital REAL,
                    total_pnl REAL,
                    roi_pct REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    open_positions INTEGER
                )
            """)

            # Token snapshots table - for historical data collection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,  -- 'initial', 'followup'
                    snapshot_time DATETIME NOT NULL,
                    token_symbol TEXT,
                    token_age_hours REAL,
                    price_usd REAL,
                    liquidity_usd REAL,
                    market_cap REAL,
                    volume_5min REAL,
                    volume_1hour REAL,
                    volume_24hour REAL,
                    buy_count_5min INTEGER,
                    sell_count_5min INTEGER,
                    buy_count_1hour INTEGER,
                    sell_count_1hour INTEGER,
                    price_change_5min REAL,
                    price_change_1hour REAL,
                    price_change_24hour REAL
                )
            """)

            # Token labels table - outcome labels for ML training
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL UNIQUE,
                    label INTEGER NOT NULL,  -- 0=RUG_PULL, 1=PUMP_DUMP, 2=WASH_TRADING, 3=LEGITIMATE, 4=UNKNOWN
                    label_time DATETIME NOT NULL,
                    outcome_data TEXT  -- JSON string with outcome details
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_token
                ON trades(token_address, chain_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_assessments_token
                ON ml_assessments(token_address, chain_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_assessments_verified
                ON ml_assessments(outcome_verified, timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_snapshots
                ON token_snapshots(token_address, snapshot_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_labels
                ON token_labels(token_address)
            """)

    def log_trade(self, trade_data: Dict):
        """Log a trade to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    token_address, token_symbol, chain_id, strategy,
                    action, price, size_usd, ml_risk_score,
                    stop_loss, take_profit_1, take_profit_2, confidence,
                    close_reason, pnl_usd, pnl_pct, hold_time_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('token_address'),
                trade_data.get('token_symbol'),
                trade_data.get('chain_id'),
                trade_data.get('strategy'),
                trade_data.get('action'),
                trade_data.get('price'),
                trade_data.get('size_usd'),
                trade_data.get('ml_risk_score'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit_1'),
                trade_data.get('take_profit_2'),
                trade_data.get('confidence'),
                trade_data.get('close_reason'),
                trade_data.get('pnl_usd'),
                trade_data.get('pnl_pct'),
                trade_data.get('hold_time_hours')
            ))
            return cursor.lastrowid

    def log_ml_assessment(self, assessment: Dict):
        """Log an ML assessment"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ml_assessments (
                    token_address, token_symbol, chain_id,
                    risk_score, recommendation,
                    rug_pull_risk, wash_trading_risk, pump_dump_risk,
                    features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assessment.get('token_address'),
                assessment.get('token_symbol'),
                assessment.get('chain_id'),
                assessment.get('risk_score'),
                assessment.get('recommendation'),
                assessment.get('rug_pull_risk'),
                assessment.get('wash_trading_risk'),
                assessment.get('pump_dump_risk'),
                assessment.get('features_json')
            ))
            return cursor.lastrowid

    def log_filter_results(self, timestamp: datetime, total_scanned: int,
                          approved: int, rejected: int):
        """Log filter performance stats"""
        rejection_rate = rejected / total_scanned if total_scanned > 0 else 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO filter_stats (
                    timestamp, total_scanned, approved, rejected, rejection_rate
                ) VALUES (?, ?, ?, ?, ?)
            """, (timestamp, total_scanned, approved, rejected, rejection_rate))

    def log_portfolio_snapshot(self, snapshot: Dict):
        """Log portfolio state snapshot"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio_snapshots (
                    capital, total_pnl, roi_pct, total_trades, win_rate, open_positions
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                snapshot.get('capital'),
                snapshot.get('total_pnl'),
                snapshot.get('roi_pct'),
                snapshot.get('total_trades'),
                snapshot.get('win_rate'),
                snapshot.get('open_positions')
            ))

    def get_all_trades(self, limit: Optional[int] = None) -> List[sqlite3.Row]:
        """Get all trades, optionally limited"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM trades ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            return cursor.fetchall()

    def get_verified_assessments(self, days: int = 30) -> List[sqlite3.Row]:
        """Get verified ML assessments for retraining"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ml_assessments
                WHERE outcome_verified = 1
                AND timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
            """, (days,))
            return cursor.fetchall()

    def get_unverified_assessments(self, hours_old_min: int = 24) -> List[sqlite3.Row]:
        """Get unverified assessments older than specified hours"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ml_assessments
                WHERE outcome_verified = 0
                AND timestamp <= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
            """, (hours_old_min,))
            return cursor.fetchall()

    def update_assessment_outcome(self, assessment_id: int,
                                  outcome_label: int, outcome_verified: bool = True):
        """Update ML assessment with actual outcome"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ml_assessments
                SET outcome_label = ?, outcome_verified = ?
                WHERE id = ?
            """, (outcome_label, outcome_verified, assessment_id))

    def get_portfolio_stats(self, days: int = 7) -> Dict:
        """Get portfolio performance statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get trade statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE NULL END) as avg_win,
                    AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd ELSE NULL END) as avg_loss,
                    SUM(pnl_usd) as total_pnl
                FROM trades
                WHERE action = 'CLOSE'
                AND timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))

            row = cursor.fetchone()

            total_trades = row['total_trades'] or 0
            winning_trades = row['winning_trades'] or 0
            losing_trades = row['losing_trades'] or 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_win': row['avg_win'] or 0,
                'avg_loss': row['avg_loss'] or 0,
                'total_pnl': row['total_pnl'] or 0,
                'profit_factor': abs(row['avg_win'] / row['avg_loss']) if row['avg_loss'] else 0
            }

    def get_ml_filter_accuracy(self, days: int = 7) -> Dict:
        """Calculate ML filter accuracy on verified outcomes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE
                        WHEN (recommendation = 'REJECT' AND outcome_label IN (0, 1, 2))
                        OR (recommendation = 'PASS' AND outcome_label = 3)
                        THEN 1 ELSE 0
                    END) as correct
                FROM ml_assessments
                WHERE outcome_verified = 1
                AND timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))

            row = cursor.fetchone()
            total = row['total'] or 0
            correct = row['correct'] or 0

            return {
                'total_verified': total,
                'correct_predictions': correct,
                'accuracy': correct / total if total > 0 else 0
            }
