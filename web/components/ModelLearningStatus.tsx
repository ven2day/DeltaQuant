"use client";

import { BrainCircuit } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import {
  getMarketModels,
  getMarketStrategies,
  type MarketModelStatus,
  type MarketName,
  type MarketStrategyStatus,
} from "@/lib/marketApi";

export function ModelLearningStatus({ market }: { market: MarketName }) {
  const [models, setModels] = useState<MarketModelStatus[]>([]);
  const [validation, setValidation] = useState<MarketStrategyStatus[]>([]);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const [modelResult, validationResult] = await Promise.allSettled([
        getMarketModels(market),
        getMarketStrategies(market),
      ]);
      if (!active) return;
      if (modelResult.status === "fulfilled") setModels(modelResult.value);
      if (validationResult.status === "fulfilled") setValidation(validationResult.value);
    };
    void refresh();
    const timer = window.setInterval(refresh, 30_000);
    return () => { active = false; window.clearInterval(timer); };
  }, [market]);

  return (
    <Card title={`${market} Model / Learning Status`} icon={BrainCircuit} accent="var(--cat-4)">
      <p className="mb-4 text-xs text-ink-muted">Training and walk-forward validation run outside the trading worker. Runtime inference uses only ACTIVE approved models.</p>
      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase text-ink-muted">Validation</h3>
          {validation.length === 0 ? <p className="text-xs text-ink-muted">No published validator state.</p> : validation.map((item, index) => (
            <div key={`${item.strategy_name}-${item.timeframe}-${index}`} className="grid grid-cols-[1fr_auto_auto] gap-2 border-t border-border py-2 text-xs">
              <span>{item.strategy_name ?? item.strategy}</span><span>{item.timeframe}</span><span>{item.eligibility ?? item.validation_status}</span>
            </div>
          ))}
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase text-ink-muted">Models</h3>
          {models.length === 0 ? <p className="text-xs text-ink-muted">No ACTIVE {market} model.</p> : models.map((item, index) => (
            <div key={`${item.strategy_name}-${item.timeframe}-${index}`} className="grid grid-cols-[1fr_auto_auto] gap-2 border-t border-border py-2 text-xs">
              <span>{item.strategy_name}</span><span>{item.timeframe}</span><span>{item.model_version ?? item.status}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
