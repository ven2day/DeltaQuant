"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api";
import type { SignalRecord } from "./types";

const POLL_INTERVAL_MS = 30_000;

export function useSignalHistory(days: number = 7): {
  signals: SignalRecord[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSignals = useCallback(() => {
    apiFetch(`/api/signals?days=${days}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<SignalRecord[]>;
      })
      .then((data) => {
        setSignals(data);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load signal history");
      })
      .finally(() => setLoading(false));
  }, [days]);

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchSignals]);

  return { signals, loading, error, refresh: fetchSignals };
}
