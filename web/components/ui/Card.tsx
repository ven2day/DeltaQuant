import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function Card({
  title,
  icon: Icon,
  accent,
  right,
  children,
  className = "",
}: {
  title: string;
  icon?: LucideIcon;
  accent?: string; // one of the --cat-N / --status-N CSS vars, applied to the icon
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-border bg-surface shadow-card ${className}`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          {Icon && (
            <Icon
              size={16}
              strokeWidth={2}
              style={{ color: accent ?? "var(--ink-muted)" }}
            />
          )}
          <h2 className="text-sm font-semibold text-ink-primary">{title}</h2>
        </div>
        {right}
      </div>
      <div className="p-4 text-sm text-ink-secondary">{children}</div>
    </div>
  );
}
