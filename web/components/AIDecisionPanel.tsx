import { Brain } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

export function AIDecisionPanel({ stats }: { stats: TradingStats }) {
  const sig = stats.current_signal;
  const hasSignal = sig && sig.signal_type;
  const action = sig.action ?? sig.signal_type ?? "WAIT";

  return (
    <Card title="AI Decision" icon={Brain} accent="var(--cat-5)">
      {hasSignal ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge
              label={action}
              tone={action === "BUY" ? "good" : action === "REJECT" ? "critical" : "warning"}
            />
            <span className="font-semibold text-ink-primary">{sig.symbol}</span>
          </div>
          {sig.target_realistic === false && (
            <div className="rounded border border-status-warning/30 bg-status-warning/10 p-2 text-xs text-status-warning">
              The 3–4% framework target is ambitious for this symbol/timeframe.
            </div>
          )}
          {(sig.rationale ?? []).length > 0 && (
            <ul className="space-y-1 border-t border-border pt-2 text-xs text-ink-muted">
              {(sig.rationale ?? []).slice(0, 5).map((reason) => (
                <li key={reason}>• {reason}</li>
              ))}
            </ul>
          )}
          <div className="flex items-center justify-between text-xs">
            <span className="text-ink-muted">Strategy</span>
            <span className="text-ink-primary">
              {sig.strategy}
              {sig.timeframe && <span className="text-cat-5"> · {sig.timeframe}</span>}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-ink-muted">Confidence</span>
            <span className="tabular font-medium text-ink-primary">
              {((sig.confidence ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      ) : (
        <div className="italic text-ink-muted">No active signal</div>
      )}
      {stats.last_decision_reason && (
        <div className="mt-3 border-t border-border pt-2 text-xs italic text-ink-muted">
          {stats.last_decision_reason.slice(0, 140)}
        </div>
      )}
    </Card>
  );
}
