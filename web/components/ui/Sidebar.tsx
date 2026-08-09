import type { TabItem } from "./Tabs";

// Renders one vertical nav group (a bordered box of buttons). Positioning
// (width, stickiness) is the caller's responsibility so multiple groups --
// e.g. a market switcher stacked above the page-section nav -- can share one
// sticky column instead of each fighting for its own sticky offset.
export function Sidebar({
  tabs,
  active,
  onChange,
}: {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl border border-border bg-surface p-1.5 shadow-card">
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${
              isActive
                ? "bg-surface-raised text-ink-primary shadow-card"
                : "text-ink-muted hover:text-ink-secondary"
            }`}
            style={isActive ? { boxShadow: "inset 0 0 0 1px var(--border)" } : undefined}
          >
            <Icon size={15} strokeWidth={2} className={isActive ? "text-cat-1" : ""} />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
