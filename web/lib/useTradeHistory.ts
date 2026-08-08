"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api";
import type { ClosedPaperTrade } from "./types";

const POLL_INTERVAL_MS = 30_000;

export function useTradeHistory(): {
  trades: ClosedPaperTrade[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const [trades, setTrades] = useState<ClosedPaperTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTrades = useCallback(() => {
    apiFetch("/api/trades?limit=250")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<ClosedPaperTrade[]>;
      })
      .then((data) => {
        setTrades(data);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load trade history");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchTrades]);

  return { trades, loading, error, refresh: fetchTrades };
}
