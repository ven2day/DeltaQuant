import type { TabItem } from "./Tabs";

// Renders a bordered navigation group. The default remains vertical for
// compatibility; workspaces can opt into a single-row, horizontally scrollable
// section bar when they need the full page width for dashboard content.
export function Sidebar({
  tabs,
  active,
  onChange,
  orientation = "vertical",
}: {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
  orientation?: "vertical" | "horizontal";
}) {
  const horizontal = orientation === "horizontal";

  return (
    <div
      role="tablist"
      aria-orientation={orientation}
      className={`flex gap-0.5 rounded-xl border border-border bg-surface p-1.5 shadow-card ${
        horizontal ? "flex-row items-center overflow-x-auto" : "flex-col"
      }`}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${
              horizontal ? "min-h-9 flex-none whitespace-nowrap" : ""
            } ${
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
