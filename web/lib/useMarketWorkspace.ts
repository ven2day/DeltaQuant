"use client";

import { useEffect, useState } from "react";
import {
  getMarketPositions,
  getMarketCandidates,
  getMarketModels,
  getMarketSignals,
  getMarketStatus,
  getMarketStrategies,
  type MarketDecision,
  type MarketTechnicalCandidate,
  type MarketModelStatus,
  type MarketName,
  type MarketStatus,
  type MarketStrategyStatus,
} from "./marketApi";

export function useMarketWorkspace(market: MarketName) {
  const [status, setStatus] = useState<MarketStatus | null>(null);
  const [signals, setSignals] = useState<MarketDecision[]>([]);
  const [candidates, setCandidates] = useState<MarketTechnicalCandidate[]>([]);
  const [positions, setPositions] = useState<MarketDecision[]>([]);
  const [strategies, setStrategies] = useState<MarketStrategyStatus[]>([]);
  const [models, setModels] = useState<MarketModelStatus[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      const values = await Promise.allSettled([
        getMarketStatus(market),
        getMarketCandidates(market),
        getMarketSignals(market),
        getMarketPositions(market),
        getMarketStrategies(market),
        getMarketModels(market),
      ]);
      if (cancelled) return;
      if (values[0].status === "fulfilled") setStatus(values[0].value);
      if (values[1].status === "fulfilled") setCandidates(values[1].value);
      if (values[2].status === "fulfilled") setSignals(values[2].value);
      if (values[3].status === "fulfilled") setPositions(values[3].value);
      if (values[4].status === "fulfilled") setStrategies(values[4].value);
      if (values[5].status === "fulfilled") setModels(values[5].value);
      const failed = values.find((value) => value.status === "rejected");
      setError(failed?.status === "rejected" ? String(failed.reason) : null);
    };
    void refresh();
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [market]);

  return { status, candidates, signals, positions, strategies, models, error };
}
