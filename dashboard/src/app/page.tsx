'use client';

import { useState } from 'react';
import CandlestickChart from '@/components/CandlestickChart';
import PoolSelector from '@/components/PoolSelector';
import StatsCard from '@/components/StatsCard';
import TradesTable from '@/components/TradesTable';

export default function Home() {
  const [selectedPool, setSelectedPool] = useState<string>('');
  const [selectedPoolName, setSelectedPoolName] = useState<string>('');

  const handleSelectPool = (address: string, name: string) => {
    setSelectedPool(address);
    setSelectedPoolName(name);
  };

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <main className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-white">Safe-ish Trading Dashboard</h1>
          <PoolSelector selectedPool={selectedPool} onSelectPool={handleSelectPool} />
        </div>

        <StatsCard />

        <div className="bg-gray-900 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">
            Price Chart {selectedPoolName && `- ${selectedPoolName}`}
          </h2>
          <CandlestickChart poolAddress={selectedPool} poolName={selectedPoolName} />
        </div>

        <div className="bg-gray-900 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Recent Trades</h2>
          <TradesTable />
        </div>
      </main>
    </div>
  );
}
