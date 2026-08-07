import type { LucideIcon } from "lucide-react";

export interface TabItem {
  id: string;
  label: string;
  icon: LucideIcon;
}

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="flex justify-center">
      <div className="inline-flex gap-1 rounded-xl border border-border bg-surface p-1">
        {tabs.map((tab) => {
          const isActive = tab.id === active;
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onChange(tab.id)}
              className={`flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-surface-raised text-ink-primary shadow-card"
                  : "text-ink-muted hover:text-ink-secondary"
              }`}
              style={isActive ? { boxShadow: "inset 0 0 0 1px var(--border)" } : undefined}
            >
              <Icon size={14} strokeWidth={2} className={isActive ? "text-cat-1" : ""} />
              {tab.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
