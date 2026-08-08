"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "./api";

export interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export function useCandles(
  symbol: string | null,
  timeframe: string,
  limit: number = 300,
): { candles: Candle[]; loading: boolean } {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) {
      setCandles([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    apiFetch(
      `/api/candles?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}&limit=${limit}`,
    )
      .then((res) => (res.ok ? (res.json() as Promise<Candle[]>) : Promise.resolve([])))
      .then((data) => {
        if (!cancelled) setCandles(data);
      })
      .catch(() => {
        if (!cancelled) setCandles([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, timeframe, limit]);

  return { candles, loading };
}
