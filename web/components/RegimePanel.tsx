import { Layers } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";
import { Meter } from "./ui/Meter";

export function RegimePanel({ stats }: { stats: TradingStats }) {
  const conf = stats.regime_confidence;
  const confColor =
    conf >= 0.7 ? "var(--status-good)" : conf >= 0.5 ? "var(--status-warning)" : "var(--status-critical)";

  return (
    <Card title="Active Strategies" icon={Layers} accent="var(--cat-7)">
      <div className="mb-3">
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="text-ink-muted">Regime confidence</span>
          <span className="tabular text-ink-primary">{(conf * 100).toFixed(0)}%</span>
        </div>
        <Meter fraction={conf} color={confColor} />
      </div>
      {stats.active_strategies.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {stats.active_strategies.map((s) => (
            <span
              key={s}
              className="rounded-md bg-surface-raised px-2 py-1 text-xs font-medium text-ink-secondary"
            >
              {s}
            </span>
          ))}
        </div>
      ) : (
        <div className="italic text-ink-muted">No strategies active</div>
      )}
    </Card>
  );
}
