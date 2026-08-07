"use client";

import { useCallback, useEffect, useState } from "react";
import type { SystemHealth } from "./types";

const POLL_INTERVAL_MS = 60_000;

// Shares a host/port with the WebSocket (both served by the same FastAPI app) —
// derive the base URL from NEXT_PUBLIC_WS_URL rather than adding a second env var.
function apiBaseUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws";
  return wsUrl.replace(/^ws/, "http").replace(/\/ws$/, "");
}

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
    fetch(`${apiBaseUrl()}/api/health?full=${full}`)
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
