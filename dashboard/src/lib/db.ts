import Database from 'better-sqlite3';
import path from 'path';

const DB_PATH = path.join(process.cwd(), '..', 'data', 'trading.db');

let db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!db) {
    db = new Database(DB_PATH, { readonly: true });
  }
  return db;
}

export interface OHLCVCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Pool {
  pool_address: string;
  network: string;
  name: string;
  base_token_symbol: string;
  quote_token_symbol: string;
  dex: string;
  reserve_usd: number;
  fdv_usd: number;
  first_seen: string;
  last_updated: string;
}

export interface Trade {
  id: number;
  timestamp: string;
  token_address: string;
  token_symbol: string;
  chain_id: string;
  strategy: string;
  action: string;
  price: number;
  size_usd: number;
  ml_risk_score: number;
  stop_loss: number;
  take_profit_1: number;
  take_profit_2: number;
  confidence: number;
  close_reason: string;
  pnl_usd: number;
  pnl_pct: number;
  hold_time_hours: number;
}

export interface Stats {
  total_candles: number;
  unique_pools: number;
  networks: { network: string; candles: number; pools: number }[];
  timeframes: { [key: string]: number };
  date_range: { min: string; max: string } | null;
}

export function getOHLCV(
  poolAddress: string,
  timeframe: string = 'HOUR_1',
  limit: number = 500
): OHLCVCandle[] {
  const db = getDb();
  const stmt = db.prepare(`
    SELECT
      timestamp,
      open,
      high,
      low,
      close,
      volume
    FROM ohlcv_data
    WHERE pool_address = ? AND timeframe = ?
    ORDER BY timestamp DESC
    LIMIT ?
  `);

  const rows = stmt.all(poolAddress, timeframe, limit) as OHLCVCandle[];
  return rows.reverse(); // Return in chronological order
}

export function getPools(): Pool[] {
  const db = getDb();
  const stmt = db.prepare(`
    SELECT
      pool_address,
      network,
      name,
      base_token_symbol,
      quote_token_symbol,
      dex,
      reserve_usd,
      fdv_usd,
      first_seen,
      last_updated
    FROM pool_metadata
    ORDER BY reserve_usd DESC
  `);

  return stmt.all() as Pool[];
}

export function getTrades(limit: number = 100): Trade[] {
  const db = getDb();

  // Check if trades table exists
  const tableCheck = db.prepare(`
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='trades'
  `).get();

  if (!tableCheck) {
    return [];
  }

  const stmt = db.prepare(`
    SELECT *
    FROM trades
    ORDER BY timestamp DESC
    LIMIT ?
  `);

  return stmt.all(limit) as Trade[];
}

export function getStats(): Stats {
  const db = getDb();

  // Total candles
  const totalCandles = db.prepare(`
    SELECT COUNT(*) as count FROM ohlcv_data
  `).get() as { count: number };

  // Unique pools
  const uniquePools = db.prepare(`
    SELECT COUNT(DISTINCT pool_address) as count FROM ohlcv_data
  `).get() as { count: number };

  // By network
  const byNetwork = db.prepare(`
    SELECT
      network,
      COUNT(*) as candles,
      COUNT(DISTINCT pool_address) as pools
    FROM ohlcv_data
    GROUP BY network
  `).all() as { network: string; candles: number; pools: number }[];

  // By timeframe
  const byTimeframe = db.prepare(`
    SELECT timeframe, COUNT(*) as count
    FROM ohlcv_data
    GROUP BY timeframe
  `).all() as { timeframe: string; count: number }[];

  const timeframes: { [key: string]: number } = {};
  byTimeframe.forEach(row => {
    timeframes[row.timeframe] = row.count;
  });

  // Date range
  const dateRange = db.prepare(`
    SELECT MIN(timestamp) as min, MAX(timestamp) as max
    FROM ohlcv_data
  `).get() as { min: string; max: string } | null;

  return {
    total_candles: totalCandles.count,
    unique_pools: uniquePools.count,
    networks: byNetwork,
    timeframes,
    date_range: dateRange
  };
}
