"use client";

import {
  Bell,
  Brain,
  Database,
  LineChart,
  type LucideIcon,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useSystemHealth } from "@/lib/useSystemHealth";
import { formatLocalTime } from "@/lib/formatTime";
import type { ServiceHealthEntry } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

const STATUS_TONE: Record<string, "good" | "warning" | "critical"> = {
  healthy: "good",
  degraded: "warning",
  unhealthy: "critical",
};

const SERVICE_META: Record<string, { label: string; hint: string }> = {
  market_data: { label: "Market Data (DhanHQ)", hint: "Live Dhan quote feed" },
  dhan_broker: { label: "DhanHQ Broker", hint: "Live order routing (optional)" },
  groq_api: { label: "Groq LLM", hint: "Powers all agent reasoning" },
  news_feed: { label: "News Sentiment", hint: "Google RSS + LLM scoring" },
  langfuse: { label: "Langfuse", hint: "Agent tracing, evaluation & monitoring" },
  database: { label: "PostgreSQL", hint: "Agent memory storage" },
  memory_system: { label: "Memory System", hint: "Learning-from-losses loop" },
  paper_wallet: { label: "Paper Wallet", hint: "Local paper trading engine" },
  circuit_breakers: { label: "Circuit Breakers", hint: "LLM call safety guards" },
  telegram: { label: "Telegram Alerts", hint: "Trade & risk notifications" },
};

const CATEGORIES: { title: string; icon: LucideIcon; services: string[] }[] = [
  { title: "Data & Market Feeds", icon: LineChart, services: ["market_data", "dhan_broker"] },
  { title: "AI & Intelligence", icon: Brain, services: ["groq_api", "news_feed", "langfuse"] },
  {
    title: "Storage & Memory",
    icon: Database,
    services: ["database", "memory_system", "paper_wallet"],
  },
  { title: "Execution & Alerts", icon: ShieldCheck, services: ["circuit_breakers", "telegram"] },
];

// These checks only run a real network call on a "full" check (they're skipped on the
// cheap default poll) — until then they simply won't be in the services list yet.
const REQUIRES_FULL_CHECK = new Set(["market_data", "groq_api", "database"]);

function ServiceTile({ name, entry }: { name: string; entry: ServiceHealthEntry | undefined }) {
  const meta = SERVICE_META[name] ?? { label: name, hint: "" };

  if (!entry) {
    return (
      <div className="flex items-start justify-between gap-2 rounded-lg border border-border p-3">
        <div>
          <div className="text-sm font-medium text-ink-primary">{meta.label}</div>
          <div className="text-xs text-ink-muted">{meta.hint}</div>
        </div>
        <Badge label="NOT CHECKED" tone="neutral" />
      </div>
    );
  }

  const tone = STATUS_TONE[entry.status] ?? "neutral";
  return (
    <div className="flex items-start justify-between gap-2 rounded-lg border border-border p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium text-ink-primary">{meta.label}</div>
        <div className="truncate text-xs text-ink-muted" title={entry.message}>
          {entry.message}
        </div>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-1">
        <Badge label={entry.status.toUpperCase()} tone={tone} />
        {entry.latency_ms > 0 && (
          <span className="tabular text-[10px] text-ink-muted">
            {entry.latency_ms.toFixed(0)}ms
          </span>
        )}
      </div>
    </div>
  );
}

export function SystemStatusPanel() {
  const { health, loading, error, runFullCheck } = useSystemHealth();

  const byName = new Map((health?.services ?? []).map((s) => [s.name, s]));
  const overallTone = health ? (STATUS_TONE[health.status] ?? "neutral") : "neutral";
  const hasFullCheck = [...REQUIRES_FULL_CHECK].some((name) => byName.has(name));

  return (
    <Card
      title="System Status"
      icon={ShieldCheck}
      accent="var(--cat-1)"
      right={
        <button
          type="button"
          onClick={runFullCheck}
          disabled={loading}
          className="flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink-primary disabled:opacity-50"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Run Full Check
        </button>
      }
    >
      {error ? (
        <div className="italic text-status-critical">Failed to load system status: {error}</div>
      ) : !health ? (
        <div className="italic text-ink-muted">Checking system status…</div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-surface-raised p-3">
            <div className="flex items-center gap-2">
              <Badge label={health.status.toUpperCase()} tone={overallTone} dot pulse />
              <span className="text-sm text-ink-secondary">
                {health.status === "healthy"
                  ? "All checked systems operational"
                  : health.status === "degraded"
                    ? "Running, with non-fatal checks needing attention"
                    : "Action needed — a required component is unhealthy"}
              </span>
            </div>
            <span className="text-xs text-ink-muted">
              Uptime {Math.floor(health.uptime_seconds / 60)}m · checked{" "}
              {formatLocalTime(health.checked_at)}
            </span>
          </div>

          {!hasFullCheck && (
            <div className="flex items-center gap-2 rounded-lg border border-dashed border-border p-2 text-xs text-ink-muted">
              <Bell size={12} />
              Groq/DhanHQ/PostgreSQL connectivity isn&apos;t pinged automatically (avoids
              wasting API quota) — click <span className="font-medium">Run Full Check</span> to
              verify them live.
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {CATEGORIES.map((category) => {
              const Icon = category.icon;
              return (
                <div key={category.title}>
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                    <Icon size={12} />
                    {category.title}
                  </div>
                  <div className="space-y-2">
                    {category.services.map((name) => (
                      <ServiceTile key={name} name={name} entry={byName.get(name)} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}
