import type { LucideIcon } from "lucide-react";
import { Sparkline } from "./Sparkline";

// Stat tile contract: label, value (proportional figures, not tabular — this
// is a display-size number, not a table column), optional signed delta
// (color = direction × whether up is good), optional trend sparkline.
export function StatTile({
  label,
  value,
  icon: Icon,
  delta,
  deltaGood,
  trend,
  trendColor,
}: {
  label: string;
  value: string;
  icon?: LucideIcon;
  delta?: string;
  deltaGood?: boolean;
  trend?: number[];
  trendColor?: string;
}) {
  const deltaColor =
    deltaGood === undefined
      ? "text-ink-muted"
      : deltaGood
        ? "text-status-good"
        : "text-status-critical";

  return (
    <div className="rounded-xl border border-border bg-surface p-4 shadow-card">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-medium text-ink-muted">
          {Icon && <Icon size={14} strokeWidth={2} />}
          {label}
        </div>
        {trend && trend.length > 1 && (
          <Sparkline values={trend} color={trendColor ?? "var(--cat-1)"} />
        )}
      </div>
      <div className="mt-1 text-2xl font-semibold text-ink-primary">{value}</div>
      {delta && <div className={`mt-0.5 text-xs font-medium ${deltaColor}`}>{delta}</div>}
    </div>
  );
}
