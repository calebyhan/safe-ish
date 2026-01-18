'use client';

import { useEffect, useState } from 'react';

interface Stats {
  total_candles: number;
  unique_pools: number;
  networks: { network: string; candles: number; pools: number }[];
  timeframes: { [key: string]: number };
  date_range: { min: string; max: string } | null;
}

export default function StatsCard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          <div className="h-4 bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-red-400">Failed to load stats</p>
      </div>
    );
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold text-white mb-4">Data Statistics</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Total Candles</p>
          <p className="text-2xl font-bold text-white">
            {stats.total_candles.toLocaleString()}
          </p>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Unique Pools</p>
          <p className="text-2xl font-bold text-white">{stats.unique_pools}</p>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Networks</p>
          <p className="text-2xl font-bold text-white">{stats.networks.length}</p>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Timeframes</p>
          <p className="text-2xl font-bold text-white">
            {Object.keys(stats.timeframes).length}
          </p>
        </div>
      </div>

      {stats.date_range && (
        <div className="text-sm text-gray-400">
          <span className="font-medium">Date Range:</span>{' '}
          {formatDate(stats.date_range.min)} - {formatDate(stats.date_range.max)}
        </div>
      )}

      {stats.networks.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium text-gray-300 mb-2">By Network:</p>
          <div className="space-y-1">
            {stats.networks.map((net) => (
              <div
                key={net.network}
                className="flex justify-between text-sm text-gray-400"
              >
                <span className="capitalize">{net.network}</span>
                <span>
                  {net.candles.toLocaleString()} candles, {net.pools} pools
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
