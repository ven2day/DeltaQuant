import { CheckCircle2, Info, ScrollText, TrendingUp, XCircle } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";

const LEVEL_STYLE: Record<string, { color: string; icon: LucideIcon }> = {
  INFO: { color: "var(--cat-1)", icon: Info },
  SUCCESS: { color: "var(--status-good)", icon: CheckCircle2 },
  WARNING: { color: "var(--status-warning)", icon: Info },
  ERROR: { color: "var(--status-critical)", icon: XCircle },
  TRADE: { color: "var(--cat-3)", icon: TrendingUp },
};

export function ActivityLogPanel({ stats }: { stats: TradingStats }) {
  const entries = stats.activity_log ?? [];

  return (
    <Card title="Activity Log" icon={ScrollText} accent="var(--ink-muted)">
      {entries.length === 0 ? (
        <div className="italic text-ink-muted">Waiting for activity…</div>
      ) : (
        <div className="max-h-[28rem] space-y-2 overflow-y-auto">
          {entries
            .slice()
            .reverse()
            .map((entry, i) => {
              const style = LEVEL_STYLE[entry.level] ?? { color: "var(--ink-muted)", icon: Info };
              const Icon = style.icon;
              return (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <Icon size={13} strokeWidth={2} style={{ color: style.color }} className="mt-0.5 shrink-0" />
                  <span className="tabular shrink-0 text-ink-muted">{entry.time}</span>
                  <span className="text-ink-secondary">{entry.message}</span>
                </div>
              );
            })}
        </div>
      )}
    </Card>
  );
}
