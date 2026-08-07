import { Bot } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";

function Row({ label, value, color = "text-ink-primary" }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between py-1">
      <span className="text-ink-muted">{label}</span>
      <span className={`tabular font-medium ${color}`}>{value}</span>
    </div>
  );
}

export function AgentActivityPanel({ stats }: { stats: TradingStats }) {
  const approvalRate =
    stats.signals_generated > 0
      ? ((stats.signals_validated / stats.signals_generated) * 100).toFixed(0) + "%"
      : "—";

  return (
    <Card title="Agent Activity" icon={Bot} accent="var(--cat-5)">
      <div className="divide-y divide-border">
        <Row label="Cycles run" value={String(stats.cycles_run)} />
        <Row label="Signals generated" value={String(stats.signals_generated)} />
        <Row label="Validated" value={String(stats.signals_validated)} color="text-status-good" />
        <Row label="Rejected" value={String(stats.signals_rejected)} color="text-status-critical" />
        <Row label="Risk blocked" value={String(stats.trades_risk_rejected)} color="text-status-warning" />
        <Row label="Approval rate" value={approvalRate} color="text-cat-1" />
      </div>
    </Card>
  );
}
