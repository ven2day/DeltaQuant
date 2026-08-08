"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api";
import type { SystemHealth } from "./types";

const POLL_INTERVAL_MS = 60_000;

export function useSystemHealth(): {
  health: SystemHealth | null;
  loading: boolean;
  error: string | null;
  runFullCheck: () => void;
} {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = useCallback((full: boolean) => {
    setLoading(true);
    apiFetch(`/api/health?full=${full}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<SystemHealth>;
      })
      .then((data) => {
        setHealth(data);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load system health");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchHealth(false);
    const interval = setInterval(() => fetchHealth(false), POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  return { health, loading, error, runFullCheck: () => fetchHealth(true) };
}
