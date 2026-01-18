import { NextRequest, NextResponse } from 'next/server';
import { getOHLCV } from '@/lib/db';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const poolAddress = searchParams.get('pool');
  const timeframe = searchParams.get('timeframe') || 'HOUR_1';
  const limit = parseInt(searchParams.get('limit') || '500');

  if (!poolAddress) {
    return NextResponse.json(
      { error: 'Pool address is required' },
      { status: 400 }
    );
  }

  try {
    const candles = getOHLCV(poolAddress, timeframe, limit);

    // Transform for lightweight-charts format
    // Use Unix timestamps (seconds) for proper time handling
    const tvData = candles.map(candle => {
      // Treat database timestamps as UTC
      const timestampStr = candle.timestamp.endsWith('Z')
        ? candle.timestamp
        : candle.timestamp + 'Z';
      const unixTime = Math.floor(new Date(timestampStr).getTime() / 1000);

      return {
        time: unixTime as any,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      };
    });

    const volumeData = candles.map(candle => {
      const timestampStr = candle.timestamp.endsWith('Z')
        ? candle.timestamp
        : candle.timestamp + 'Z';
      const unixTime = Math.floor(new Date(timestampStr).getTime() / 1000);

      return {
        time: unixTime as any,
        value: candle.volume,
        color: candle.close >= candle.open ? '#26a69a' : '#ef5350',
      };
    });

    return NextResponse.json({
      candles: tvData,
      volume: volumeData,
      count: candles.length,
    });
  } catch (error) {
    console.error('Error fetching OHLCV:', error);
    return NextResponse.json(
      { error: 'Failed to fetch OHLCV data' },
      { status: 500 }
    );
  }
}
