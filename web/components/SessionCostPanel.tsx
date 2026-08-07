import { Settings } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";
import { Meter } from "./ui/Meter";

export function SessionCostPanel({ stats }: { stats: TradingStats }) {
  const target = stats.goal_expected_to_date || 0;
  const frac = target > 0 ? stats.goal_mtd_pnl / target : 0;

  return (
    <Card title="Session & Cost" icon={Settings} accent="var(--ink-muted)">
      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-muted">
        LLM today (FinOps)
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="tabular text-lg font-semibold text-ink-primary">{stats.llm_calls}</div>
          <div className="text-xs text-ink-muted">Calls</div>
        </div>
        <div>
          <div className="tabular text-lg font-semibold text-ink-primary">
            {stats.llm_tokens.toLocaleString()}
          </div>
          <div className="text-xs text-ink-muted">Tokens</div>
        </div>
        <div>
          <div className="tabular text-lg font-semibold text-cat-1">
            ${stats.llm_cost_usd.toFixed(4)}
          </div>
          <div className="text-xs text-ink-muted">Cost</div>
        </div>
      </div>

      {stats.goal_enabled && (
        <div className="mt-4 border-t border-border pt-3">
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-muted">
            Profit goal
          </div>
          {!stats.goal_feasible ? (
            <div className="text-xs font-medium text-status-warning">
              ⚠ target exceeds risk budget
            </div>
          ) : (
            <>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-ink-secondary">
                  Rs.{stats.goal_mtd_pnl.toLocaleString("en-IN")} / Rs.
                  {stats.goal_expected_to_date.toLocaleString("en-IN")}
                </span>
                <span
                  className={`font-medium ${stats.goal_on_pace ? "text-status-good" : "text-status-warning"}`}
                >
                  {stats.goal_on_pace ? "On pace" : "Behind"}
                </span>
              </div>
              <Meter
                fraction={Math.max(0, frac)}
                color={stats.goal_on_pace ? "var(--status-good)" : "var(--status-warning)"}
              />
            </>
          )}
        </div>
      )}
    </Card>
  );
}
