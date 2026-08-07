import { Wallet } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";
import { Meter } from "./ui/Meter";

const inr = (n: number) =>
  n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function Row({ label, value, tone }: { label: string; value: string; tone?: "good" | "critical" }) {
  const color =
    tone === "good" ? "text-status-good" : tone === "critical" ? "text-status-critical" : "text-ink-primary";
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-ink-muted">{label}</span>
      <span className={`tabular font-medium ${color}`}>{value}</span>
    </div>
  );
}

export function AccountPanel({ stats }: { stats: TradingStats }) {
  return (
    <Card title="Account" icon={Wallet} accent="var(--cat-1)">
      <div className="divide-y divide-border">
        <Row label="Total trades" value={String(stats.total_trades)} />
        <Row label="Realized P&L" value={`Rs. ${inr(stats.realized_pnl)}`} tone={stats.realized_pnl >= 0 ? "good" : "critical"} />
        <Row label="Unrealized P&L" value={`Rs. ${inr(stats.unrealized_pnl)}`} tone={stats.unrealized_pnl >= 0 ? "good" : "critical"} />
        {stats.total_trades > 0 && (
          <>
            <Row label="Best trade" value={`+Rs. ${inr(stats.best_trade)}`} tone="good" />
            <Row label="Worst trade" value={`Rs. ${inr(stats.worst_trade)}`} tone="critical" />
          </>
        )}
      </div>

      {stats.goal_enabled && (
        <div className="mt-3 border-t border-border pt-3">
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="text-ink-muted">Monthly goal</span>
            <span className="tabular text-ink-primary">Rs. {stats.goal_target_amount.toLocaleString("en-IN")}</span>
          </div>
          {!stats.goal_feasible ? (
            <div className="text-xs font-medium text-status-warning">⚠ target exceeds risk budget</div>
          ) : (
            <>
              <Meter
                fraction={
                  stats.goal_expected_to_date > 0
                    ? stats.goal_mtd_pnl / stats.goal_expected_to_date
                    : 0
                }
                color={stats.goal_on_pace ? "var(--status-good)" : "var(--status-warning)"}
              />
              <div
                className={`mt-1 text-xs font-medium ${stats.goal_on_pace ? "text-status-good" : "text-status-warning"}`}
              >
                {stats.goal_on_pace ? "On pace" : "Behind pace"} — Rs.
                {stats.goal_mtd_pnl.toLocaleString("en-IN")} / Rs.
                {stats.goal_expected_to_date.toLocaleString("en-IN")}
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
