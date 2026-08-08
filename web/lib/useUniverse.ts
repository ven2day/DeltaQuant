"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "./api";

/** NSE symbols chartable from the Charts tab -- the NIFTY50+midcap universe the
 * discovery/signal loop scans, not limited to symbols with an open position. */
export function useUniverse(): string[] {
  const [symbols, setSymbols] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/universe")
      .then((res) => (res.ok ? (res.json() as Promise<string[]>) : Promise.resolve([])))
      .then((data) => {
        if (!cancelled) setSymbols(data);
      })
      .catch(() => {
        if (!cancelled) setSymbols([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return symbols;
}
