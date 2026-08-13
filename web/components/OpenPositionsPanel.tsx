import { ListOrdered } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

function _fmt(value: number | undefined): string {
  return value ? `Rs.${value.toFixed(2)}` : "—";
}

export function OpenPositionsPanel({ stats }: { stats: TradingStats }) {
  const positions = stats.open_positions ?? [];

  return (
    <Card title="Open Positions" icon={ListOrdered} accent="var(--ink-muted)">
      {positions.length === 0 ? (
        <div className="italic text-ink-muted">No open positions</div>
      ) : (
        <div className="divide-y divide-border">
          {positions.slice(0, 5).map((pos, i) => {
            const invested = pos.qty * pos.entry;
            return (
              <div key={`${pos.symbol}-${i}`} className="py-2 first:pt-0 last:pb-0">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="font-medium text-ink-primary">{pos.symbol}</span>
                    <Badge label={pos.side} tone={pos.side === "BUY" ? "good" : "critical"} />
                    {pos.timeframe ? (
                      <span className="text-xs text-ink-muted">{pos.timeframe}</span>
                    ) : null}
                  </div>
                  <span
                    className={`tabular font-medium ${
                      pos.pnl >= 0 ? "text-status-good" : "text-status-critical"
                    }`}
                  >
                    {pos.pnl >= 0 ? "+" : ""}Rs.{pos.pnl.toFixed(2)}
                  </span>
                </div>
                <div className="tabular mt-0.5 text-xs text-ink-muted">
                  Qty {pos.qty} @ {_fmt(pos.entry)} · Invested Rs.{invested.toFixed(2)}
                </div>
                <div className="tabular mt-0.5 text-xs text-ink-muted">
                  SL {_fmt(pos.stop)} · TP {_fmt(pos.target)}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
