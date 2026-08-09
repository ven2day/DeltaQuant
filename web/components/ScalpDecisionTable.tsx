"use client";

import { Gauge } from "lucide-react";
import type { ScalpOpportunity, TimeframeAssessment, TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

// req 10's role order: 5m=execution, 15m=primary, 30m=directional, 1h=context.
// 4h (optional macro filter) is shown in the expanded reason row instead of its
// own column -- it's the one role that can be disabled entirely
// (scalp_macro_filter_enabled), so a permanently-empty column would be noise.
const TIMEFRAME_COLUMNS = ["5m", "15m", "30m", "1h"];

const DECISION_DOT: Record<TimeframeAssessment["decision"], string> = {
  BUY: "var(--status-good)",
  WAIT: "var(--status-warning)",
  REJECT: "var(--status-critical)",
};

const FINAL_DECISION_TONE: Record<ScalpOpportunity["final_decision"], "good" | "warning" | "critical"> = {
  ENTER_NOW: "good",
  WAIT_PULLBACK: "warning",
  WAIT_BREAKOUT: "warning",
  REJECT: "critical",
};

function TimeframeCell({ assessment }: { assessment: TimeframeAssessment | undefined }) {
  if (!assessment) {
    return <span className="text-ink-muted">—</span>;
  }
  return (
    <div
      className="flex items-center justify-end gap-1.5"
      title={assessment.reasons.join(" · ") || assessment.decision}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: DECISION_DOT[assessment.decision] }}
      />
      <span className="tabular text-ink-secondary">{assessment.score.toFixed(2)}</span>
    </div>
  );
}

function OpportunityRow({ opportunity }: { opportunity: ScalpOpportunity }) {
  const eq = opportunity.entry_quality;
  return (
    <tr className="border-b border-border/50 align-top">
      <td className="py-1.5 pr-3">
        <div className="font-medium text-ink-primary">{opportunity.symbol}</div>
        <div className="text-[10px] text-ink-muted">
          {opportunity.primary_strategy || "—"} · {opportunity.primary_timeframe || "—"}
        </div>
      </td>
      {TIMEFRAME_COLUMNS.map((tf) => (
        <td key={tf} className="py-1.5 pr-3">
          <TimeframeCell assessment={opportunity.timeframe_states?.[tf]} />
        </td>
      ))}
      <td className="tabular py-1.5 pr-3 text-right font-medium text-ink-primary">
        {opportunity.score.toFixed(2)}
      </td>
      <td className="py-1.5 pr-3">
        {eq ? (
          <span
            className="whitespace-nowrap text-[11px]"
            style={{ color: DECISION_DOT[eq.status === "ENTER_NOW" ? "BUY" : eq.status === "REJECT" ? "REJECT" : "WAIT"] }}
            title={eq.reasons.join(" · ")}
          >
            {eq.status.replace("_", " ")}
          </span>
        ) : (
          <span className="text-ink-muted">—</span>
        )}
      </td>
      <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
        Rs.{opportunity.preferred_entry_low.toFixed(2)}–{opportunity.preferred_entry_high.toFixed(2)}
      </td>
      <td className="tabular py-1.5 pr-3 text-right text-status-critical">
        Rs.{opportunity.stop_loss.toFixed(2)}
      </td>
      <td className="tabular py-1.5 pr-3 text-right text-status-good">
        Rs.{opportunity.target_price.toFixed(2)}
      </td>
      <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
        {opportunity.expected_r.toFixed(2)}R
      </td>
      <td className="py-1.5 pr-3">
        <Badge
          label={opportunity.final_decision.replace("_", " ")}
          tone={FINAL_DECISION_TONE[opportunity.final_decision]}
        />
      </td>
      <td className="max-w-[220px] py-1.5 text-[11px] text-ink-muted">
        {opportunity.reason[0] ?? "—"}
        {opportunity.reason.length > 1 && (
          <span title={opportunity.reason.slice(1).join(" · ")}>
            {" "}
            +{opportunity.reason.length - 1} more
          </span>
        )}
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
    ["regime-ok", funnel.regime_compatible],
    ["H-8 admitted", funnel.h8_admitted],
    ["sent to AI", funnel.sent_to_ai],
    ["AI approved", funnel.ai_approved],
    ["executed", funnel.execution_accepted],
  ];
  if (steps.every(([, v]) => !v)) {
    return null;
  }
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-1 gap-y-1 text-[11px] text-ink-muted">
      {steps.map(([label, value], i) => (
        <span key={label} className="flex items-center gap-1">
          {i > 0 && <span className="text-ink-muted/50">→</span>}
          <span className="tabular font-medium text-ink-secondary">{value ?? 0}</span>
          <span>{label}</span>
        </span>
      ))}
    </div>
  );
}

export function ScalpDecisionTable({ stats }: { stats: TradingStats }) {
  const opportunities = stats.scalp_opportunities ?? [];
  const funnel = stats.scalp_funnel ?? {};

  return (
    <Card title="Scalp Decisions" icon={Gauge} accent="var(--cat-4)">
      <FunnelSummary funnel={funnel} />
      {opportunities.length === 0 ? (
        <div className="italic text-ink-muted">
          No scalp opportunities this cycle — either scalping is disabled, or nothing
          cleared entry-quality/regime/multi-timeframe confirmation yet.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-xs">
            <thead>
              <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
                <th className="py-2 pr-3 font-medium">Symbol</th>
                {TIMEFRAME_COLUMNS.map((tf) => (
                  <th key={tf} className="py-2 pr-3 text-right font-medium">
                    {tf}
                  </th>
                ))}
                <th className="py-2 pr-3 text-right font-medium">Score</th>
                <th className="py-2 pr-3 font-medium">Entry Quality</th>
                <th className="py-2 pr-3 text-right font-medium">Preferred Entry</th>
                <th className="py-2 pr-3 text-right font-medium">Stop</th>
                <th className="py-2 pr-3 text-right font-medium">Target</th>
                <th className="py-2 pr-3 text-right font-medium">Exp. R</th>
                <th className="py-2 pr-3 font-medium">Decision</th>
                <th className="py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {opportunities.map((o) => (
                <OpportunityRow key={o.symbol} opportunity={o} />
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-3 text-[10px] text-ink-muted">
        Dots show each timeframe&apos;s decision (green=BUY, amber=WAIT, red=REJECT);
        hover a cell or the reason column for the full explanation. A row reading here
        still has to independently clear the H-8 strategy-admission gate and every
        risk check before it can become a real order.
      </div>
    </Card>
  );
}
