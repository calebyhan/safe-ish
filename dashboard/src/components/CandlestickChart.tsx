'use client';

import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  CandlestickSeries,
  HistogramSeries,
  Time,
  WhitespaceData
} from 'lightweight-charts';

interface CandlestickChartProps {
  poolAddress: string;
  poolName?: string;
}

export default function CandlestickChart({ poolAddress, poolName }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<string>('HOUR_1');

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#1a1a2e' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      watermark: {
        visible: false,
      },
      timeScale: {
        borderColor: '#2B2B43',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 6,
        fixLeftEdge: false,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
        tickMarkFormatter: (time: Time) => {
          // Format tick marks on the time axis to show UTC time
          const timestamp = typeof time === 'number' ? time * 1000 : new Date(time as string).getTime();
          const date = new Date(timestamp);

          // Show time for intraday, date for daily (using UTC)
          const hours = date.getUTCHours();
          const minutes = date.getUTCMinutes();

          // If it's midnight (00:00) UTC, just show the date
          if (hours === 0 && minutes === 0) {
            return date.toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              timeZone: 'UTC'
            });
          }

          // Otherwise show time in UTC
          const hour12 = hours % 12 || 12;
          const ampm = hours >= 12 ? 'PM' : 'AM';
          const minutesStr = minutes.toString().padStart(2, '0');
          return `${hour12}:${minutesStr} ${ampm}`;
        },
      },
      localization: {
        dateFormat: 'dd MMM',
        timeFormatter: (time: Time) => {
          // This formats the crosshair label on the time axis (using UTC)
          const timestamp = typeof time === 'number' ? time * 1000 : new Date(time as string).getTime();
          const date = new Date(timestamp);
          const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
          const day = date.getUTCDate();
          const hours = date.getUTCHours();
          const minutes = date.getUTCMinutes();
          const hour12 = hours % 12 || 12;
          const ampm = hours >= 12 ? 'PM' : 'AM';
          const minutesStr = minutes.toString().padStart(2, '0');
          return `${month} ${day}, ${hour12}:${minutesStr} ${ampm}`;
        },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    chartRef.current = chart;

    // Create candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    });
    candleSeriesRef.current = candleSeries;

    // Create volume series
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // Setup crosshair move handler for custom tooltip
    chart.subscribeCrosshairMove((param) => {
      if (!tooltipRef.current || !chartContainerRef.current) return;

      if (
        param.point === undefined ||
        !param.time ||
        param.point.x < 0 ||
        param.point.y < 0
      ) {
        tooltipRef.current.style.display = 'none';
      } else {
        const data = param.seriesData.get(candleSeries) as CandlestickData | undefined;

        if (data) {
          const timestamp = typeof param.time === 'number' ? param.time * 1000 : new Date(param.time as string).getTime();
          const date = new Date(timestamp);

          tooltipRef.current.style.display = 'block';

          // Format the date for display in UTC
          const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
          const day = date.getUTCDate();
          const year = date.getUTCFullYear();
          const hours = date.getUTCHours();
          const minutes = date.getUTCMinutes();
          const hour12 = hours % 12 || 12;
          const ampm = hours >= 12 ? 'PM' : 'AM';
          const minutesStr = minutes.toString().padStart(2, '0');
          const formattedDate = `${month} ${day}, ${year} ${hour12}:${minutesStr} ${ampm} UTC`;

          tooltipRef.current.innerHTML = `
            <div style="background: rgba(0,0,0,0.9); border: 1px solid #2B2B43; border-radius: 4px; padding: 8px; color: white; font-size: 12px;">
              <div style="margin-bottom: 4px; font-weight: 600;">
                ${formattedDate}
              </div>
              <div>O: ${data.open?.toFixed(8)}</div>
              <div>H: ${data.high?.toFixed(8)}</div>
              <div>L: ${data.low?.toFixed(8)}</div>
              <div>C: ${data.close?.toFixed(8)}</div>
            </div>
          `;

          const toolipWidth = 220;
          const toolipHeight = 140;
          const x = param.point.x;
          const y = param.point.y;

          let left = x + 15;
          let top = y + 15;

          if (left > chartContainerRef.current.clientWidth - toolipWidth) {
            left = x - toolipWidth - 15;
          }

          if (top > chartContainerRef.current.clientHeight - toolipHeight) {
            top = y - toolipHeight - 15;
          }

          tooltipRef.current.style.left = left + 'px';
          tooltipRef.current.style.top = top + 'px';
        }
      }
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Fetch data when pool or timeframe changes
  useEffect(() => {
    if (!poolAddress || !candleSeriesRef.current || !volumeSeriesRef.current) return;

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/ohlcv?pool=${poolAddress}&timeframe=${timeframe}&limit=1000`);
        if (!response.ok) throw new Error('Failed to fetch data');

        const data = await response.json();

        if (data.candles && data.candles.length > 0) {
          // Get midnight UTC for today and tomorrow
          const nowDate = new Date();
          const todayMidnightUTC = Date.UTC(
            nowDate.getUTCFullYear(),
            nowDate.getUTCMonth(),
            nowDate.getUTCDate(),
            0, 0, 0, 0
          );
          const midnightTimestamp = Math.floor(todayMidnightUTC / 1000);

          const tomorrowMidnightUTC = Date.UTC(
            nowDate.getUTCFullYear(),
            nowDate.getUTCMonth(),
            nowDate.getUTCDate() + 1,
            0, 0, 0, 0
          );
          const nextMidnightTimestamp = Math.floor(tomorrowMidnightUTC / 1000);

          // Calculate interval in seconds based on timeframe
          const timeframeIntervals: Record<string, number> = {
            'MINUTE_5': 300,
            'MINUTE_15': 900,
            'HOUR_1': 3600,
            'HOUR_4': 14400,
            'DAY_1': 86400,
          };
          const intervalSeconds = timeframeIntervals[timeframe] || 3600;

          // Fill in missing intervals from last candle to midnight with whitespace
          const lastCandleTime = data.candles[data.candles.length - 1].time as number;
          const candlesWithExtension: (CandlestickData | WhitespaceData)[] = [...data.candles];

          // Add whitespace for each missing interval between last candle and midnight
          let currentTime = lastCandleTime + intervalSeconds;
          while (currentTime <= nextMidnightTimestamp) {
            candlesWithExtension.push({
              time: currentTime as Time,
            } as WhitespaceData);
            currentTime += intervalSeconds;
          }

          candleSeriesRef.current?.setData(candlesWithExtension);
          volumeSeriesRef.current?.setData(data.volume as HistogramData[]);

          // Force autoscale to recalculate price range
          candleSeriesRef.current?.priceScale().applyOptions({ autoScale: true });

          // Set visible range to show current day (from today's midnight UTC to next midnight UTC)
          setTimeout(() => {
            if (chartRef.current) {
              chartRef.current.timeScale().setVisibleRange({
                from: midnightTimestamp as Time,
                to: nextMidnightTimestamp as Time,
              });
            }
          }, 50);
        } else {
          // No data available - clear the chart
          candleSeriesRef.current?.setData([]);
          volumeSeriesRef.current?.setData([]);
          setError(`No data available for ${timeframe} timeframe`);
        }
      } catch (err) {
        console.error('Chart fetch error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [poolAddress, timeframe]);

  const timeframes = [
    { value: 'MINUTE_5', label: '5m' },
    { value: 'MINUTE_15', label: '15m' },
    { value: 'HOUR_1', label: '1H' },
    { value: 'HOUR_4', label: '4H' },
    { value: 'DAY_1', label: '1D' },
  ];

  return (
    <div className="relative">
      {poolName && (
        <h3 className="text-lg font-semibold text-white mb-2">{poolName}</h3>
      )}

      {/* Timeframe selector */}
      <div className="flex gap-2 mb-4">
        {timeframes.map((tf) => (
          <button
            key={tf.value}
            onClick={() => setTimeframe(tf.value)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              timeframe === tf.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {tf.label}
          </button>
        ))}
      </div>

      <div className="relative">
        <div
          ref={chartContainerRef}
          className="w-full rounded-lg overflow-hidden relative [&_a]:!hidden"
        />
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            display: 'none',
            padding: 0,
            boxSizing: 'border-box',
            fontSize: '12px',
            textAlign: 'left',
            zIndex: 1000,
            pointerEvents: 'none',
          }}
        />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
            <div className="text-white">Loading chart...</div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
            <div className="text-red-400">{error}</div>
          </div>
        )}
      </div>
    </div>
  );
}
