"use client";

import { Activity } from "lucide-react";
import { useEffect, useState } from "react";
import type { TradingStats } from "@/lib/types";

// Groups the machine-key stages set by run_live_trading.py's dashboard.set_cycle_stage()
// calls into a display tone. Kept as a lookup (not a naming convention) so a new stage
// added later fails safe to "active" rather than silently miscoloring.
const STAGE_TONE: Record<string, "active" | "waiting" | "idle" | "error"> = {
  starting: "active",
  exits: "active",
  market_data: "active",
  scanning: "active",
  ranking: "active",
  ai_review: "active",
  processing: "active",
  waiting: "waiting",
  no_signals: "idle",
  no_candidates: "idle",
  stale_data: "idle",
  budget_paused: "idle",
  market_closed: "idle",
  error: "error",
};

const TONE_COLOR: Record<string, string> = {
  active: "var(--status-good)",
  waiting: "var(--cat-1)",
  idle: "var(--ink-muted)",
  error: "var(--status-critical)",
};

function elapsedLabel(sinceIso: string, nowMs: number): string {
  if (!sinceIso) return "";
  const since = new Date(sinceIso).getTime();
  const seconds = Math.max(0, Math.floor((nowMs - since) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}

function countdownLabel(untilIso: string, nowMs: number): string {
  if (!untilIso) return "";
  const until = new Date(untilIso).getTime();
  const seconds = Math.max(0, Math.round((until - nowMs) / 1000));
  return `${seconds}s`;
}

export function CycleLifecyclePanel({ stats }: { stats: TradingStats }) {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const tone = STAGE_TONE[stats.cycle_stage] ?? "active";
  const color = TONE_COLOR[tone];
  const isWaiting = stats.cycle_stage === "waiting" && stats.next_cycle_at;
  const timerLabel = isWaiting
    ? `next cycle in ${countdownLabel(stats.next_cycle_at, nowMs)}`
    : stats.cycle_stage_started_at
      ? `running ${elapsedLabel(stats.cycle_stage_started_at, nowMs)}`
      : "";

  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-2.5 shadow-card">
      <Activity size={15} strokeWidth={2} style={{ color }} className="flex-none" />
      <span className="relative flex h-2 w-2 flex-none">
        {tone === "active" && (
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
            style={{ backgroundColor: color }}
          />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      </span>
      <span className="flex-none text-xs font-semibold text-ink-primary">
        Cycle #{stats.current_cycle_number || "—"}
      </span>
      <span className="min-w-0 flex-1 truncate text-xs text-ink-secondary">
        {stats.cycle_stage_label || "Starting up…"}
      </span>
      {timerLabel && (
        <span className="tabular flex-none text-[11px] text-ink-muted">{timerLabel}</span>
      )}
    </div>
  );
}
