'use client';

import { useEffect, useState } from 'react';

interface Trade {
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

export default function TradesTable() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch('/api/trades?limit=50');
        const data = await response.json();
        setTrades(data.trades || []);
      } catch (error) {
        console.error('Failed to fetch trades:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-1/4 mb-4"></div>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-10 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (trades.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">Recent Trades</h2>
        <p className="text-gray-400">No trades recorded yet. Run the trading bot in paper mode to see trades here.</p>
      </div>
    );
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 overflow-x-auto">
      <h2 className="text-xl font-bold text-white mb-4">Recent Trades</h2>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2 px-2">Token</th>
            <th className="text-left py-2 px-2">Action</th>
            <th className="text-left py-2 px-2">Strategy</th>
            <th className="text-right py-2 px-2">Price</th>
            <th className="text-right py-2 px-2">Size</th>
            <th className="text-right py-2 px-2">P&L</th>
            <th className="text-right py-2 px-2">P&L %</th>
            <th className="text-right py-2 px-2">Hold Time</th>
            <th className="text-left py-2 px-2">Reason</th>
            <th className="text-left py-2 px-2">Time</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => (
            <tr
              key={trade.id}
              className="border-b border-gray-700 hover:bg-gray-700/50"
            >
              <td className="py-2 px-2 text-white font-medium">
                {trade.token_symbol || 'Unknown'}
              </td>
              <td className="py-2 px-2">
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    trade.action === 'OPEN'
                      ? 'bg-blue-500/20 text-blue-300'
                      : 'bg-purple-500/20 text-purple-300'
                  }`}
                >
                  {trade.action}
                </span>
              </td>
              <td className="py-2 px-2 text-gray-300">
                {trade.strategy?.replace('_', ' ') || 'N/A'}
              </td>
              <td className="py-2 px-2 text-right text-gray-300">
                ${trade.price?.toFixed(8) || '0'}
              </td>
              <td className="py-2 px-2 text-right text-gray-300">
                ${trade.size_usd?.toFixed(2) || '0'}
              </td>
              <td
                className={`py-2 px-2 text-right font-medium ${
                  trade.pnl_usd && trade.pnl_usd >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {trade.pnl_usd ? (
                  <>
                    {trade.pnl_usd >= 0 ? '+' : ''}${trade.pnl_usd.toFixed(2)}
                  </>
                ) : (
                  '-'
                )}
              </td>
              <td
                className={`py-2 px-2 text-right font-medium ${
                  trade.pnl_pct && trade.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {trade.pnl_pct ? (
                  <>
                    {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(1)}%
                  </>
                ) : (
                  '-'
                )}
              </td>
              <td className="py-2 px-2 text-right text-gray-300">
                {trade.hold_time_hours ? `${trade.hold_time_hours.toFixed(1)}h` : '-'}
              </td>
              <td className="py-2 px-2 text-gray-300 capitalize">
                {trade.close_reason?.replace('_', ' ') || '-'}
              </td>
              <td className="py-2 px-2 text-gray-400 text-xs">
                {formatDate(trade.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
