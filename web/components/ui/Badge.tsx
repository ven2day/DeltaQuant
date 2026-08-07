import type { LucideIcon } from "lucide-react";

// Status colors always ship with an icon + label, never color alone.
const TONE: Record<string, { fg: string; bg: string }> = {
  good: { fg: "var(--status-good)", bg: "rgba(12,163,12,0.12)" },
  warning: { fg: "var(--status-warning)", bg: "rgba(250,178,25,0.12)" },
  serious: { fg: "var(--status-serious)", bg: "rgba(236,131,90,0.12)" },
  critical: { fg: "var(--status-critical)", bg: "rgba(230,103,103,0.12)" },
  neutral: { fg: "var(--ink-secondary)", bg: "rgba(255,255,255,0.06)" },
};

export function Badge({
  label,
  tone = "neutral",
  icon: Icon,
  dot = false,
  pulse = false,
}: {
  label: string;
  tone?: keyof typeof TONE;
  icon?: LucideIcon;
  dot?: boolean;
  /** Animates the dot — reserve for a genuinely live/streaming state. */
  pulse?: boolean;
}) {
  const c = TONE[tone] ?? TONE.neutral;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ color: c.fg, backgroundColor: c.bg }}
    >
      {dot && (
        <span className="relative flex h-1.5 w-1.5">
          {pulse && (
            <span
              className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
              style={{ backgroundColor: c.fg }}
            />
          )}
          <span
            className="relative inline-flex h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: c.fg }}
          />
        </span>
      )}
      {Icon && <Icon size={12} strokeWidth={2.5} />}
      {label}
    </span>
  );
}
