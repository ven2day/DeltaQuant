"use client";

import { Zap } from "lucide-react";
import { useMemo, useState } from "react";
import type { ScalpOpportunity, TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";

type CandidateFilter = "ALL" | ScalpOpportunity["final_decision"];

const FILTERS: { label: string; value: CandidateFilter }[] = [
  { label: "All", value: "ALL" },
  { label: "Enter now", value: "ENTER_NOW" },
  { label: "Wait pullback", value: "WAIT_PULLBACK" },
  { label: "Wait breakout", value: "WAIT_BREAKOUT" },
  { label: "Rejected", value: "REJECT" },
];

function money(value: number): string {
  return `Rs.${value.toFixed(2)}`;
}

function decisionColour(decision: ScalpOpportunity["final_decision"]): string {
  if (decision === "ENTER_NOW") return "text-status-good";
  if (decision === "REJECT") return "text-status-bad";
  return "text-status-warn";
}

export function ScalpingCandidatesPanel({ stats }: { stats: TradingStats }) {
  const candidates = Array.isArray(stats.scalp_opportunities) ? stats.scalp_opportunities : [];
  const [activeFilter, setActiveFilter] = useState<CandidateFilter>("ALL");

  const counts = useMemo(
    () => ({
      ALL: candidates.length,
      ENTER_NOW: candidates.filter((candidate) => candidate.final_decision === "ENTER_NOW")
        .length,
      WAIT_PULLBACK: candidates.filter(
        (candidate) => candidate.final_decision === "WAIT_PULLBACK"
      ).length,
      WAIT_BREAKOUT: candidates.filter(
        (candidate) => candidate.final_decision === "WAIT_BREAKOUT"
      ).length,
      REJECT: candidates.filter((candidate) => candidate.final_decision === "REJECT").length,
    }),
    [candidates]
  );

  const visibleCandidates = useMemo(
    () =>
      activeFilter === "ALL"
        ? candidates
        : candidates.filter((candidate) => candidate.final_decision === activeFilter),
    [activeFilter, candidates]
  );

  const emptyMessage = !stats.market_open
    ? "NSE is closed. Candidates will update after the next settled Dhan candle during the verified trading window."
    : stats.scalp_scan_completed_at
      ? "The latest settled-candle scan completed with no eligible scalp candidates."
      : "Waiting for the first settled-candle scalp scan.";

  return (
    <Card title={`Scalping Candidates (${candidates.length})`} icon={Zap} accent="var(--cat-3)">
      <div className="mb-3 text-[11px] text-ink-muted">
        Event-driven candidates that passed the current technical, eligibility, and regime stages.
        ML and Qwen may still abstain or be skipped; deterministic risk remains final.
        {stats.scalp_scan_completed_at && (
          <span> Last completed: {new Date(stats.scalp_scan_completed_at).toLocaleString()}.</span>
        )}
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {FILTERS.map((filter) => {
          const active = activeFilter === filter.value;
          return (
            <button
              key={filter.value}
              type="button"
              onClick={() => setActiveFilter(filter.value)}
              className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                active
                  ? "border-[var(--cat-3)] bg-white/5 text-ink-primary"
                  : "border-border text-ink-muted hover:text-ink-secondary"
              }`}
            >
              {filter.label} ({counts[filter.value]})
            </button>
          );
        })}
      </div>

      {candidates.length === 0 ? (
        <div className="italic text-ink-muted">{emptyMessage}</div>
      ) : visibleCandidates.length === 0 ? (
        <div className="italic text-ink-muted">No candidates match this filter.</div>
      ) : (
        <div className="max-h-[680px] overflow-auto">
          <table className="w-full min-w-[1120px] border-collapse text-xs">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
                <th className="py-2 pr-3 font-medium">Symbol</th>
                <th className="py-2 pr-3 font-medium">Side</th>
                <th className="py-2 pr-3 font-medium">Strategy</th>
                <th className="py-2 pr-3 font-medium">TF</th>
                <th className="py-2 pr-3 font-medium">Decision</th>
                <th className="py-2 pr-3 text-right font-medium">Score</th>
                <th className="py-2 pr-3 text-right font-medium">ML</th>
                <th className="py-2 pr-3 text-right font-medium">Entry</th>
                <th className="py-2 pr-3 text-right font-medium">Stop</th>
                <th className="py-2 pr-3 text-right font-medium">Target</th>
                <th className="py-2 pr-3 text-right font-medium">Expected R</th>
                <th className="py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {visibleCandidates.map((candidate) => (
                <tr
                  key={`${candidate.symbol}-${candidate.primary_timeframe}-${candidate.direction}`}
                  className="border-b border-border/50"
                >
                  <td className="py-1.5 pr-3 font-medium text-ink-primary">{candidate.symbol}</td>
                  <td
                    className={`py-1.5 pr-3 font-semibold ${
                      candidate.direction === "BUY" ? "text-status-good" : "text-status-bad"
                    }`}
                  >
                    {candidate.direction}
                  </td>
                  <td className="py-1.5 pr-3 text-ink-secondary">
                    {candidate.primary_strategy}
                  </td>
                  <td className="py-1.5 pr-3 text-ink-secondary">
                    {candidate.primary_timeframe}
                  </td>
                  <td
                    className={`py-1.5 pr-3 font-medium ${decisionColour(candidate.final_decision)}`}
                  >
                    {candidate.final_decision.replaceAll("_", " ")}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {candidate.score.toFixed(3)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {candidate.ml_probability == null
                      ? "ABSTAIN"
                      : `${(candidate.ml_probability * 100).toFixed(1)}%`}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {money(candidate.entry_price)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {money(candidate.stop_loss)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {money(candidate.target_price)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
                    {candidate.expected_r.toFixed(3)}
                  </td>
                  <td className="max-w-72 py-1.5 text-ink-muted">
                    {candidate.reason.join(", ") || "Qualified technical setup"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
