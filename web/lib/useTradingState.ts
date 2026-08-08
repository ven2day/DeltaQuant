"use client";

import { useEffect, useRef, useState } from "react";
import type { StateMessage, TradingStats } from "./types";

const RECONNECT_DELAYS_MS = [2000, 5000, 10000];
const HISTORY_LENGTH = 60;

export interface HistoryPoint {
  balance: number;
  totalPnl: number;
}

export function useTradingState(): {
  state: TradingStats | null;
  connected: boolean;
  history: HistoryPoint[];
} {
  const [state, setState] = useState<TradingStats | null>(null);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const attemptRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let cancelled = false;

    const url = process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws";

    function connect() {
      if (cancelled) return;

      socket = new WebSocket(url);

      socket.onopen = () => {
        attemptRef.current = 0;
        setConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as StateMessage;
          if (message.type === "state") {
            setState(message.data);
            // Client-side rolling buffer for stat-tile sparklines — the
            // backend only pushes the current snapshot, so trend history is
            // accumulated here as messages arrive rather than fetched.
            setHistory((prev) => {
              const next = [
                ...prev,
                { balance: message.data.current_balance, totalPnl: message.data.total_pnl },
              ];
              return next.length > HISTORY_LENGTH ? next.slice(-HISTORY_LENGTH) : next;
            });
          }
        } catch {
          // Ignore malformed messages rather than tearing down the connection.
        }
      };

      const scheduleReconnect = (event: CloseEvent) => {
        if (cancelled) return;
        setConnected(false);
        // 1008 = policy violation: the backend's /ws handler sends this specifically
        // for a missing/expired/invalid session cookie (src/webui/server.py). Retrying
        // with the same stale cookie would just loop forever — send the user to log in
        // again instead.
        if (event.code === 1008) {
          window.location.assign("/login");
          return;
        }
        const delay =
          RECONNECT_DELAYS_MS[Math.min(attemptRef.current, RECONNECT_DELAYS_MS.length - 1)];
        attemptRef.current += 1;
        timerRef.current = setTimeout(connect, delay);
      };

      socket.onclose = scheduleReconnect;
      socket.onerror = () => socket?.close();
    }

    connect();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      socket?.close();
    };
  }, []);

  return { state, connected, history };
}
