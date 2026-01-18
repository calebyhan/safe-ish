'use client';

import { useEffect, useState } from 'react';

interface Pool {
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

interface PoolSelectorProps {
  selectedPool: string;
  onSelectPool: (address: string, name: string) => void;
}

export default function PoolSelector({ selectedPool, onSelectPool }: PoolSelectorProps) {
  const [pools, setPools] = useState<Pool[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPools = async () => {
      try {
        const response = await fetch('/api/pools');
        const data = await response.json();
        setPools(data.pools || []);

        // Auto-select first pool if none selected
        if (!selectedPool && data.pools?.length > 0) {
          onSelectPool(data.pools[0].pool_address, data.pools[0].name);
        }
      } catch (error) {
        console.error('Failed to fetch pools:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPools();
  }, [selectedPool, onSelectPool]);

  if (loading) {
    return <div className="text-gray-400">Loading pools...</div>;
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-300">
        Select Pool
      </label>
      <select
        value={selectedPool}
        onChange={(e) => {
          const pool = pools.find(p => p.pool_address === e.target.value);
          if (pool) {
            onSelectPool(pool.pool_address, pool.name);
          }
        }}
        className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        {pools.map((pool) => (
          <option key={pool.pool_address} value={pool.pool_address}>
            {pool.base_token_symbol}/{pool.quote_token_symbol} - {pool.dex || 'Unknown DEX'}
          </option>
        ))}
      </select>
      <div className="text-xs text-gray-500">
        {pools.length} pools available
      </div>
    </div>
  );
}
