import type { LucideIcon } from "lucide-react";
import { Sparkline } from "./Sparkline";

// Stat tile contract: label, value (tabular-nums -- this is a terminal-style
// ticker figure, values must align vertically), optional signed delta shown
// inline (not stacked, to keep the strip dense), optional trend sparkline.
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
    <div className="flex flex-1 items-center justify-between gap-3 px-4 py-2.5">
      <div className="min-w-0">
        <div className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-wide text-ink-muted">
          {Icon && <Icon size={11} strokeWidth={2.5} />}
          {label}
        </div>
        <div className="mt-1 flex items-baseline gap-2">
          <span className="tabular text-lg font-semibold leading-none text-ink-primary">
            {value}
          </span>
          {delta && (
            <span className={`tabular text-[11px] font-medium leading-none ${deltaColor}`}>
              {delta}
            </span>
          )}
        </div>
      </div>
      {trend && trend.length > 1 && (
        <Sparkline values={trend} color={trendColor ?? "var(--cat-1)"} />
      )}
    </div>
  );
}
