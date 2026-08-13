"use client";

import { Gauge, Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { ScalpSymbolStatus, TimeframeAssessment, TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

const TIMEFRAME_COLUMNS = ["5m", "15m", "30m", "1h"];

const DECISION_DOT: Record<TimeframeAssessment["decision"], string> = {
  BUY: "var(--status-good)",
  WAIT: "var(--status-warning)",
  REJECT: "var(--status-critical)",
};

const FILTERS: { value: "ALL" | ScalpSymbolStatus["decision"]; label: string }[] = [
  { value: "ALL", label: "All" },
  { value: "BUY", label: "Buy" },
  { value: "NO_BUY", label: "No Buy" },
  { value: "NO_DATA", label: "No Data" },
];

function TimeframeCell({ assessment }: { assessment: TimeframeAssessment | undefined }) {
  if (!assessment) {
    return <span className="text-ink-muted">-</span>;
  }
  return (
    <div
      className="flex items-center justify-end gap-1.5"
      title={assessment.reasons.join(" | ") || assessment.decision}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: DECISION_DOT[assessment.decision] }}
      />
      <span className="tabular text-ink-secondary">{assessment.score.toFixed(2)}</span>
    </div>
  );
}

function PriceCell({ value, tone }: { value: number; tone?: string }) {
  return (
    <td className={`tabular py-1.5 pr-3 text-right ${tone ?? "text-ink-secondary"}`}>
      {value > 0 ? `Rs.${value.toFixed(2)}` : "-"}
    </td>
  );
}

function StatusRow({ status }: { status: ScalpSymbolStatus }) {
  const tone =
    status.decision === "BUY" ? "good" : status.decision === "NO_DATA" ? "critical" : "warning";
  const reason = status.reasons[0] ?? "No reason recorded.";

  return (
    <tr className="border-b border-border/50 align-top">
      <td className="py-1.5 pr-3">
        <div className="font-medium text-ink-primary">{status.symbol}</div>
        <div className="text-[10px] text-ink-muted">
          {status.primary_strategy || "no setup"}
          {status.primary_timeframe ? ` | ${status.primary_timeframe}` : ""}
        </div>
      </td>
      <td className="py-1.5 pr-3">
        <Badge label={status.decision.replace(/_/g, " ")} tone={tone} dot />
        {status.selected_for_review && (
          <div className="mt-1 text-[9px] uppercase tracking-wide text-ink-muted">
            review shortlist
          </div>
        )}
      </td>
      <td className="max-w-[150px] py-1.5 pr-3 text-[10px] uppercase text-ink-secondary">
        {status.stage.replace(/_/g, " ")}
      </td>
      {TIMEFRAME_COLUMNS.map((tf) => (
        <td key={tf} className="py-1.5 pr-3">
          <TimeframeCell assessment={status.timeframe_states?.[tf]} />
        </td>
      ))}
      <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
        {status.raw_buy_signals}
      </td>
      <td className="tabular py-1.5 pr-3 text-right font-medium text-ink-primary">
        {status.score > 0 ? status.score.toFixed(2) : "-"}
      </td>
      <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
        {status.ml_probability == null ? "-" : `${(status.ml_probability * 100).toFixed(1)}%`}
      </td>
      <PriceCell value={status.entry_price} />
      <PriceCell value={status.stop_loss} tone="text-status-critical" />
      <PriceCell value={status.target_price} tone="text-status-good" />
      <td
        className="max-w-[260px] py-1.5 text-[11px] text-ink-muted"
        title={status.reasons.join(" | ")}
      >
        {reason}
        {status.reasons.length > 1 && ` +${status.reasons.length - 1} more`}
      </td>
    </tr>
  );
}

function FunnelSummary({ funnel }: { funnel: TradingStats["scalp_funnel"] }) {
  const steps: [string, number | undefined][] = [
    ["raw", funnel.raw_triggers],
    ["consolidated", funnel.consolidated],
    ["mtf-confirmed", funnel.mtf_candidates],
    ["entry-quality", funnel.entry_quality_passed],
    ["ML-scored", funnel.ml_scored],
    ["regime-ok", funnel.regime_compatible],
    ["registry eligible", funnel.registry_eligible],
    ["sent to AI", funnel.sent_to_ai],
    ["AI approved", funnel.ai_approved],
    ["executed", funnel.execution_accepted],
  ];
  if (steps.every(([, value]) => !value)) {
    return null;
  }
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-1 gap-y-1 text-[11px] text-ink-muted">
      {steps.map(([label, value], index) => (
        <span key={label} className="flex items-center gap-1">
          {index > 0 && <span className="text-ink-muted/50">-&gt;</span>}
          <span className="tabular font-medium text-ink-secondary">{value ?? 0}</span>
          <span>{label}</span>
        </span>
      ))}
    </div>
  );
}

export function ScalpDecisionTable({ stats }: { stats: TradingStats }) {
  const statuses = stats.scalp_symbol_statuses ?? [];
  const funnel = stats.scalp_funnel ?? {};
  const [activeFilter, setActiveFilter] = useState<"ALL" | ScalpSymbolStatus["decision"]>(
    "ALL"
  );
  const [query, setQuery] = useState("");

  const counts = useMemo(
    () => ({
      ALL: statuses.length,
      BUY: statuses.filter((status) => status.decision === "BUY").length,
      NO_BUY: statuses.filter((status) => status.decision === "NO_BUY").length,
      NO_DATA: statuses.filter((status) => status.decision === "NO_DATA").length,
    }),
    [statuses]
  );

  const visibleStatuses = useMemo(() => {
    const normalizedQuery = query.trim().toUpperCase();
    return statuses.filter(
      (status) =>
        (activeFilter === "ALL" || status.decision === activeFilter) &&
        (!normalizedQuery || status.symbol.toUpperCase().includes(normalizedQuery))
    );
  }, [activeFilter, query, statuses]);

  return (
    <Card title={`All Scalp Signals (${counts.ALL})`} icon={Gauge} accent="var(--cat-4)">
      <div className="mb-2 text-[11px] text-ink-muted">
        Full configured NSE universe. BUY means a deterministic technical setup exists;
        paper-order approval and risk checks remain separate.
        {stats.scalp_scan_completed_at && (
          <span> Last completed: {new Date(stats.scalp_scan_completed_at).toLocaleString()}.</span>
        )}
      </div>
      <FunnelSummary funnel={funnel} />

      {(stats.scalp_rejection_reasons ?? []).length > 0 && (
        <div className="mb-3 rounded border border-border bg-white/[0.02] px-3 py-2">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-ink-muted">
            Top scalp rejection reasons this cycle
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
            {(stats.scalp_rejection_reasons ?? []).slice(0, 6).map((row) => (
              <span key={`${row.stage}-${row.reason}`} className="text-ink-secondary">
                {row.reason} <strong className="text-ink-primary">{row.count}</strong>
                <span className="ml-1 text-ink-muted">({row.percentage.toFixed(1)}%)</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        {FILTERS.map((filter) => {
          const active = activeFilter === filter.value;
          return (
            <button
              key={filter.value}
              type="button"
              onClick={() => setActiveFilter(filter.value)}
              className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                active
                  ? "border-[var(--cat-4)] bg-white/5 text-ink-primary"
                  : "border-border text-ink-muted hover:text-ink-secondary"
              }`}
            >
              {filter.label} ({counts[filter.value]})
            </button>
          );
        })}
        <label className="ml-auto flex min-w-48 items-center gap-2 rounded border border-border px-2 py-1">
          <Search size={13} className="text-ink-muted" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find symbol"
            className="w-full bg-transparent text-xs text-ink-primary outline-none placeholder:text-ink-muted"
          />
        </label>
      </div>

      {statuses.length === 0 ? (
        <div className="italic text-ink-muted">
          Waiting for the first full-universe scalp scan. Confirm that SCALP_ENABLED is true
          and that historical candles are available.
        </div>
      ) : visibleStatuses.length === 0 ? (
        <div className="italic text-ink-muted">No symbols match this filter.</div>
      ) : (
        <div className="max-h-[680px] overflow-auto">
          <table className="w-full min-w-[1120px] border-collapse text-xs">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
                <th className="py-2 pr-3 font-medium">Symbol</th>
                <th className="py-2 pr-3 font-medium">Signal</th>
                <th className="py-2 pr-3 font-medium">Stage</th>
                {TIMEFRAME_COLUMNS.map((tf) => (
                  <th key={tf} className="py-2 pr-3 text-right font-medium">
                    {tf}
                  </th>
                ))}
                <th className="py-2 pr-3 text-right font-medium">Triggers</th>
                <th className="py-2 pr-3 text-right font-medium">Score</th>
                <th className="py-2 pr-3 text-right font-medium">ML P(up)</th>
                <th className="py-2 pr-3 text-right font-medium">Entry</th>
                <th className="py-2 pr-3 text-right font-medium">Stop</th>
                <th className="py-2 pr-3 text-right font-medium">Target</th>
                <th className="py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {visibleStatuses.map((status) => (
                <StatusRow key={status.symbol} status={status} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
