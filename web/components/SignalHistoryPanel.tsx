"use client";

import { History, RefreshCw, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useSignalHistory } from "@/lib/useSignalHistory";
import type { SignalRecord } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

const STATUS_TONE: Record<string, { label: string; tone: "good" | "warning" | "critical" }> = {
  approved: { label: "APPROVED", tone: "good" },
  rejected_validation: { label: "VALIDATION REJECTED", tone: "warning" },
  rejected_risk: { label: "RISK BLOCKED", tone: "critical" },
};

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Always render in IST regardless of the viewer's local timezone — timestamps here are
  // recorded from the trading loop's own IST-based clock, and market-hours context (which
  // signals get validated/rejected when) only makes sense read in IST.
  return (
    d.toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }) + " IST"
  );
}

function sourceLabel(source: SignalRecord["source"]): string {
  return source === "backfill" ? "SIMULATED" : "LIVE";
}

function istDateKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  // en-CA formats as YYYY-MM-DD -- a stable, sortable key, computed in IST (not the
  // viewer's local zone, and not raw UTC-string slicing) so a signal timestamped
  // just after midnight IST buckets under today's date, matching what formatTime()
  // already displays for it, instead of silently falling under the previous day.
  return new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Kolkata" }).format(d);
}

function distinctValues(signals: SignalRecord[], value: (signal: SignalRecord) => string): string[] {
  return [...new Set(signals.map(value).filter(Boolean))].sort();
}

export function SignalHistoryPanel() {
  const { signals, loading, error, refresh } = useSignalHistory(7);
  const [symbolQuery, setSymbolQuery] = useState("");
  const [day, setDay] = useState("all");
  const [side, setSide] = useState("all");
  const [timeframe, setTimeframe] = useState("all");
  const [strategy, setStrategy] = useState("all");
  const [source, setSource] = useState("all");
  const [status, setStatus] = useState("all");

  const filterOptions = useMemo(
    () => ({
      days: distinctValues(signals, (signal) => istDateKey(signal.timestamp)),
      timeframes: distinctValues(signals, (signal) => signal.timeframe),
      strategies: distinctValues(signals, (signal) => signal.strategy),
      sources: distinctValues(signals, (signal) => signal.source ?? "live"),
      statuses: distinctValues(signals, (signal) => signal.status),
    }),
    [signals]
  );

  const filteredSignals = useMemo(
    () =>
      signals.filter((signal) => {
        const normalizedSource = signal.source ?? "live";
        return (
          signal.symbol.toLowerCase().includes(symbolQuery.trim().toLowerCase()) &&
          (day === "all" || istDateKey(signal.timestamp) === day) &&
          (side === "all" || signal.side === side) &&
          (timeframe === "all" || signal.timeframe === timeframe) &&
          (strategy === "all" || signal.strategy === strategy) &&
          (source === "all" || normalizedSource === source) &&
          (status === "all" || signal.status === status)
        );
      }),
    [day, side, signals, source, status, strategy, symbolQuery, timeframe]
  );

  const filtersActive =
    symbolQuery ||
    day !== "all" ||
    side !== "all" ||
    timeframe !== "all" ||
    strategy !== "all" ||
    source !== "all" ||
    status !== "all";

  function clearFilters() {
    setSymbolQuery("");
    setDay("all");
    setSide("all");
    setTimeframe("all");
    setStrategy("all");
    setSource("all");
    setStatus("all");
  }

  return (
    <Card
      title="Signal History (7 days)"
      icon={History}
      accent="var(--cat-2)"
      right={
        <button
          type="button"
          onClick={refresh}
          className="flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink-primary"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      }
    >
      {error ? (
        <div className="italic text-status-critical">Failed to load signal history: {error}</div>
      ) : loading && signals.length === 0 ? (
        <div className="italic text-ink-muted">Loading…</div>
      ) : signals.length === 0 ? (
        <div className="italic text-ink-muted">
          No signals recorded in the last 7 days yet — they appear here as soon as the trading
          loop generates one, whether approved or rejected.
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 border-b border-border pb-3">
            <div className="relative min-w-44 flex-1 sm:max-w-52">
              <Search
                size={14}
                className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-ink-muted"
              />
              <input
                value={symbolQuery}
                onChange={(event) => setSymbolQuery(event.target.value)}
                aria-label="Filter by symbol"
                placeholder="Search symbol"
                className="h-8 w-full border border-border bg-surface-raised pl-7 pr-2 text-xs text-ink-primary outline-none transition-colors placeholder:text-ink-muted focus:border-cat-1"
              />
            </div>
            <select
              value={day}
              onChange={(event) => setDay(event.target.value)}
              aria-label="Filter by day"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All days</option>
              {filterOptions.days.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <select
              value={side}
              onChange={(event) => setSide(event.target.value)}
              aria-label="Filter by side"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All sides</option>
              <option value="BUY">Buy</option>
              <option value="SELL">Sell</option>
            </select>
            <select
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
              aria-label="Filter by timeframe"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All timeframes</option>
              {filterOptions.timeframes.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <select
              value={strategy}
              onChange={(event) => setStrategy(event.target.value)}
              aria-label="Filter by strategy"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All strategies</option>
              {filterOptions.strategies.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
              aria-label="Filter by source"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All sources</option>
              {filterOptions.sources.map((value) => (
                <option key={value} value={value}>
                  {sourceLabel(value)}
                </option>
              ))}
            </select>
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              aria-label="Filter by status"
              className="h-8 border border-border bg-surface-raised px-2 text-xs text-ink-secondary outline-none focus:border-cat-1"
            >
              <option value="all">All statuses</option>
              {filterOptions.statuses.map((value) => (
                <option key={value} value={value}>
                  {STATUS_TONE[value]?.label ?? value}
                </option>
              ))}
            </select>
            {filtersActive && (
              <button
                type="button"
                onClick={clearFilters}
                title="Clear filters"
                aria-label="Clear filters"
                className="grid size-8 place-items-center border border-border text-ink-muted transition-colors hover:border-ink-muted hover:text-ink-primary"
              >
                <X size={14} />
              </button>
            )}
            <output className="ml-auto whitespace-nowrap text-xs text-ink-muted">
              {filteredSignals.length} / {signals.length}
            </output>
          </div>
          {filteredSignals.length === 0 ? (
            <div className="py-12 text-center italic text-ink-muted">No signals match these filters.</div>
          ) : (
            <div className="h-[32rem] max-h-[calc(100vh-19rem)] overflow-auto">
          <table className="w-full border-collapse text-xs">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
                <th className="py-2 pr-3 font-medium">Time</th>
                <th className="py-2 pr-3 font-medium">Symbol</th>
                <th className="py-2 pr-3 font-medium">Side</th>
                <th className="py-2 pr-3 text-right font-medium">Entry</th>
                <th className="py-2 pr-3 font-medium">TF</th>
                <th className="py-2 pr-3 font-medium">Strategy</th>
                <th className="py-2 pr-3 font-medium">Source</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.map((s, i) => {
                const status = STATUS_TONE[s.status] ?? { label: s.status, tone: "warning" as const };
                return (
                  <tr key={`${s.timestamp}-${s.symbol}-${i}`} className="border-b border-border/50">
                    <td className="tabular whitespace-nowrap py-1.5 pr-3 text-ink-muted">
                      {formatTime(s.timestamp)}
                    </td>
                    <td className="py-1.5 pr-3 font-medium text-ink-primary">{s.symbol}</td>
                    <td
                      className={`py-1.5 pr-3 font-medium ${
                        s.side === "BUY" ? "text-status-good" : "text-status-critical"
                      }`}
                    >
                      {s.side}
                    </td>
                    <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                      {s.entry_price ? s.entry_price.toFixed(2) : "—"}
                    </td>
                    <td className="py-1.5 pr-3 text-ink-secondary">{s.timeframe || "—"}</td>
                    <td className="py-1.5 pr-3 text-ink-secondary">{s.strategy || "—"}</td>
                    <td className="py-1.5 pr-3">
                      {s.source === "backfill" ? (
                        <Badge label="SIMULATED" tone="warning" />
                      ) : (
                        <span className="text-ink-muted">Live</span>
                      )}
                    </td>
                    <td className="py-1.5 pr-3">
                      <Badge label={status.label} tone={status.tone} />
                    </td>
                    <td className="max-w-xs truncate py-1.5 text-ink-muted" title={s.reason}>
                      {s.reason || "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
          )}
        </div>
      )}
    </Card>
  );
}
