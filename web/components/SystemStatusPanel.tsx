"use client";

import {
  Bell,
  Brain,
  Database,
  LineChart,
  ListChecks,
  type LucideIcon,
  RefreshCw,
  Server,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { useSystemHealth } from "@/lib/useSystemHealth";
import { useSystemJobs } from "@/lib/useSystemJobs";
import { formatLocalTime } from "@/lib/formatTime";
import type { BackgroundJobStatus, LocalServiceEntry, ServiceHealthEntry } from "@/lib/types";
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

const LOCAL_SERVICE_META: Record<string, { label: string; hint: string }> = {
  backend: { label: "Backend", hint: "Trading loop + FastAPI/WebSocket" },
  frontend: { label: "Frontend", hint: "Next.js dashboard" },
  postgres: { label: "PostgreSQL", hint: "Wallet, positions, trade journal" },
  timescaledb: { label: "TimescaleDB", hint: "Market history candles" },
};

function LocalServiceTile({ entry }: { entry: LocalServiceEntry }) {
  const meta = LOCAL_SERVICE_META[entry.name] ?? { label: entry.name, hint: "" };
  const tone: "good" | "warning" | "critical" = entry.running
    ? "good"
    : entry.required
      ? "critical"
      : "warning";
  return (
    <div className="flex items-start justify-between gap-2 rounded-lg border border-border p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium text-ink-primary">{meta.label}</div>
        <div className="truncate text-xs text-ink-muted" title={entry.detail}>
          {entry.detail}
        </div>
      </div>
      <Badge label={entry.running ? "RUNNING" : "STOPPED"} tone={tone} />
    </div>
  );
}

const JOB_META: Record<string, { label: string; hint: string }> = {
  strategy_validation: { label: "Strategy Validation", hint: "Walk-forward eligibility gate" },
  scalp_training: { label: "Scalp ML Training", hint: "Candidate prediction artifacts" },
  signal_discovery: { label: "Signal Discovery", hint: "LLM-driven alpha research" },
};

function JobStateBadge({ job }: { job: BackgroundJobStatus }) {
  if (job.state === "running") return <Badge label="RUNNING" tone="warning" dot pulse />;
  if (job.state === "never_run") return <Badge label="NEVER RUN" tone="neutral" />;
  if (job.last_run_failed) return <Badge label="FAILED" tone="critical" />;
  return <Badge label={job.stale ? "STALE" : "UP TO DATE"} tone={job.stale ? "warning" : "good"} />;
}

function jobDetail(job: BackgroundJobStatus): string {
  if (job.hours_since_run == null) return "Never run";
  const age =
    job.hours_since_run < 1
      ? `${Math.round(job.hours_since_run * 60)}m ago`
      : `${job.hours_since_run.toFixed(1)}h ago`;
  return `Last run ${age} · auto every ${job.refresh_interval_hours}h (${job.enabled ? "on" : "off"})`;
}

function JobRow({
  name,
  job,
  onRun,
}: {
  name: string;
  job: BackgroundJobStatus;
  onRun: (name: string) => void;
}) {
  const meta = JOB_META[name] ?? { label: name, hint: "" };
  const running = job.state === "running";
  return (
    <div className="flex items-start justify-between gap-2 rounded-lg border border-border p-3">
      <div className="min-w-0">
        <div className="text-sm font-medium text-ink-primary">{meta.label}</div>
        <div className="truncate text-xs text-ink-muted">{jobDetail(job)}</div>
        {job.intervals && (
          <div className="mt-1 flex flex-wrap gap-1">
            {Object.entries(job.intervals).map(([interval, status]) => (
              <span
                key={interval}
                className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] text-ink-muted"
                title={status}
              >
                {interval}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-end gap-1.5">
        <JobStateBadge job={job} />
        <button
          type="button"
          onClick={() => onRun(name)}
          disabled={running}
          className="text-[11px] text-ink-muted underline decoration-dotted transition-colors hover:text-ink-primary disabled:cursor-not-allowed disabled:opacity-50"
        >
          Run now
        </button>
      </div>
    </div>
  );
}

interface RegistryRow {
  validation_status?: string;
}

function BackgroundJobsSection() {
  const { jobs, runJobNow } = useSystemJobs();
  const [swingModelCount, setSwingModelCount] = useState<number | null>(null);
  const [validationCounts, setValidationCounts] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    apiFetch("/api/models/status")
      .then((res) => (res.ok ? (res.json() as Promise<RegistryRow[]>) : []))
      .then((rows) => setSwingModelCount(rows.length))
      .catch(() => setSwingModelCount(null));
    apiFetch("/api/validation/status")
      .then((res) => (res.ok ? (res.json() as Promise<RegistryRow[]>) : []))
      .then((rows) => {
        const counts: Record<string, number> = {};
        for (const row of rows) {
          const status = row.validation_status ?? "UNKNOWN";
          counts[status] = (counts[status] ?? 0) + 1;
        }
        setValidationCounts(counts);
      })
      .catch(() => setValidationCounts(null));
  }, [jobs?.background_jobs.strategy_validation.last_run_at]);

  if (!jobs) return null;
  const { background_jobs: bg } = jobs;

  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-muted">
        <ListChecks size={12} />
        Background Jobs
      </div>
      <div className="space-y-2">
        <JobRow name="strategy_validation" job={bg.strategy_validation} onRun={runJobNow} />
        {validationCounts && (
          <div className="flex flex-wrap gap-x-3 pl-3 text-[11px] text-ink-muted">
            {Object.entries(validationCounts).map(([status, count]) => (
              <span key={status}>
                {status}: {count}
              </span>
            ))}
          </div>
        )}
        <JobRow name="scalp_training" job={bg.scalp_training} onRun={runJobNow} />
        <div className="flex items-start justify-between gap-2 rounded-lg border border-border p-3">
          <div className="min-w-0">
            <div className="text-sm font-medium text-ink-primary">Swing ML Training</div>
            <div className="text-xs text-ink-muted">
              {swingModelCount === null
                ? "Loading…"
                : swingModelCount === 0
                  ? "Never trained — run scripts/train_prediction_artifact.py manually " +
                    "with a prepared dataset (not auto-run)"
                  : `${swingModelCount} active model(s)`}
            </div>
          </div>
          <Badge
            label={swingModelCount ? "TRAINED" : "NEVER TRAINED"}
            tone={swingModelCount ? "good" : "neutral"}
          />
        </div>
        <JobRow name="signal_discovery" job={bg.signal_discovery} onRun={runJobNow} />
      </div>
    </div>
  );
}

export function SystemStatusPanel() {
  const { health, loading, error, runFullCheck } = useSystemHealth();
  const { jobs } = useSystemJobs();

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
            {jobs && (
              <div>
                <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                  <Server size={12} />
                  Local Services
                </div>
                <div className="space-y-2">
                  {jobs.local_services.map((entry) => (
                    <LocalServiceTile key={entry.name} entry={entry} />
                  ))}
                </div>
              </div>
            )}
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

          <BackgroundJobsSection />
        </div>
      )}
    </Card>
  );
}
