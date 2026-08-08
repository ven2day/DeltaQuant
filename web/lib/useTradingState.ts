"use client";

import { useEffect, useRef, useState } from "react";
import { checkAuthenticated } from "./api";
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

        // The backend's /ws handler rejects a missing/expired/invalid session
        // cookie by closing with code 1008 BEFORE completing the WebSocket
        // handshake (src/webui/server.py). Browsers do not expose that code (or
        // the underlying HTTP 403) for a handshake that never completed — they
        // report a generic 1006 instead, so checking `event.code === 1008` here
        // would never actually match in a real browser. Ask the backend
        // directly via /api/session instead, which is unambiguous either way:
        // an ordinary network blip (server restart, brief connectivity loss)
        // still reports the session as valid and should just retry the socket,
        // while a genuinely dead/expired session sends the user back to log in
        // instead of retrying forever with a cookie the server will never accept.
        checkAuthenticated().then((authenticated) => {
          if (cancelled) return;
          if (!authenticated) {
            window.location.assign("/login");
            return;
          }
          const delay =
            RECONNECT_DELAYS_MS[Math.min(attemptRef.current, RECONNECT_DELAYS_MS.length - 1)];
          attemptRef.current += 1;
          timerRef.current = setTimeout(connect, delay);
        });
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
