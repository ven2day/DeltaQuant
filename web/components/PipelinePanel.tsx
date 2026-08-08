"use client";

import { GitBranch } from "lucide-react";
import { useMemo } from "react";
import { useSignalHistory } from "@/lib/useSignalHistory";
import type { TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

// Fixed, closed set (src/market/signals.py StrategyType) -- safe to hardcode here rather
// than thread a fifth field through TradingStats just to enumerate what's NOT admitted.
const ALL_STRATEGIES = ["momentum", "mean_reversion", "breakout", "trend_following"];

type StageState = "pass" | "fail" | "pending" | "neutral";

const DOT_COLOR: Record<StageState, string> = {
  pass: "var(--status-good)",
  fail: "var(--status-critical)",
  pending: "var(--status-warning)",
  neutral: "var(--ink-muted)",
};

function StatusDot({ state }: { state: StageState }) {
  const color = DOT_COLOR[state];
  return (
    <span
      className="inline-block h-2 w-2 flex-none rounded-full"
      style={{ backgroundColor: color, boxShadow: state !== "neutral" ? `0 0 0 3px ${color}26` : undefined }}
    />
  );
}

function Stage({ label, state, detail }: { label: string; state: StageState; detail: string }) {
  return (
    <div className="flex min-w-[128px] flex-1 basis-32 flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <StatusDot state={state} />
        <span className="text-[10.5px] font-semibold uppercase tracking-wide text-ink-muted">
          {label}
        </span>
      </div>
      <div className="tabular text-xs leading-snug text-ink-secondary">{detail}</div>
    </div>
  );
}

export function PipelinePanel({ stats }: { stats: TradingStats }) {
  const { signals } = useSignalHistory(1);
  const sig = stats.current_signal;
  const hasSignal = Boolean(sig?.symbol);

  // current_signal is only ever the framework pre-check's pick (stage 5 -- the "AI
  // Decision" data). Signal History is written by the actual approval pipeline
  // (validation/risk), so it's the only source for the real, final verdict + reason.
  const verdictRecord = useMemo(() => {
    if (!sig?.symbol) return null;
    const matches = signals.filter(
      (s) => s.symbol === sig.symbol && s.strategy === sig.strategy && s.timeframe === sig.timeframe,
    );
    return matches[0] ?? null;
  }, [signals, sig]);

  const isOpenPosition = (stats.open_positions ?? []).some((p) => p.symbol === sig?.symbol);

  const admitted = stats.active_strategies ?? [];
  const candidateAdmitted = sig?.strategy ? admitted.includes(sig.strategy) : null;

  const regimeConf = stats.regime_confidence ?? 0;
  const regimeState: StageState = regimeConf === 0 ? "neutral" : regimeConf >= 0.3 ? "pass" : "fail";
  const scanState: StageState = stats.signals_generated > 0 ? "pass" : "neutral";
  const precheckState: StageState = !hasSignal
    ? "neutral"
    : sig.action === "BUY"
      ? "pass"
      : sig.action === "REJECT"
        ? "fail"
        : "pending";
  const admissionState: StageState =
    candidateAdmitted === null ? "neutral" : candidateAdmitted ? "pass" : "fail";
  const validationState: StageState = !hasSignal
    ? "neutral"
    : verdictRecord
      ? verdictRecord.status === "approved"
        ? "pass"
        : "fail"
      : admissionState === "fail"
        ? "fail"
        : "pending";
  const riskState: StageState = isOpenPosition
    ? "pass"
    : verdictRecord?.status === "rejected_risk"
      ? "fail"
      : "neutral";

  const verdict: { label: string; tone: "good" | "critical" | "warning" | "neutral" } = isOpenPosition
    ? { label: "FILLED", tone: "good" }
    : verdictRecord?.status === "approved"
      ? { label: "APPROVED", tone: "good" }
      : verdictRecord
        ? { label: "REJECTED", tone: "critical" }
        : hasSignal && sig.action === "BUY"
          ? { label: "IN REVIEW", tone: "warning" }
          : hasSignal
            ? { label: "PRE-CHECK: WAIT", tone: "neutral" }
            : { label: "SCANNING", tone: "neutral" };

  const reasonText =
    verdictRecord?.reason ||
    (!verdictRecord && candidateAdmitted === false
      ? `"${sig?.strategy}" is not an admitted strategy this cycle`
      : "");

  return (
    <Card title="Signal Pipeline" icon={GitBranch} accent="var(--cat-1)">
      <div className="flex flex-wrap gap-x-5 gap-y-4 pb-4">
        <Stage
          label="Scan"
          state={scanState}
          detail={`${stats.cycles_run} cycle${stats.cycles_run === 1 ? "" : "s"} · ${stats.signals_generated} raw`}
        />
        <Stage
          label="Pre-check"
          state={precheckState}
          detail={hasSignal ? `${sig.symbol} · ${((sig.confidence ?? 0) * 100).toFixed(0)}%` : "No candidate"}
        />
        <Stage
          label="Regime"
          state={regimeState}
          detail={`${stats.current_regime} · ${(regimeConf * 100).toFixed(0)}%`}
        />
        <Stage
          label="Admission"
          state={admissionState}
          detail={admitted.length > 0 ? admitted.join(", ") : "None admitted"}
        />
        <Stage
          label="Validation"
          state={validationState}
          detail={`${stats.signals_validated} passed · ${stats.signals_rejected} rejected`}
        />
        <Stage
          label="Risk"
          state={riskState}
          detail={`${stats.trades_approved} approved · ${stats.trades_risk_rejected} blocked`}
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-border pt-3">
        <div className="flex items-center gap-2">
          <Badge label={verdict.label} tone={verdict.tone} />
          {hasSignal && <span className="font-semibold text-ink-primary">{sig.symbol}</span>}
          {sig?.strategy && (
            <span className="text-xs text-ink-muted">
              {sig.strategy}
              {sig.timeframe && ` · ${sig.timeframe}`}
            </span>
          )}
        </div>
        {reasonText && (
          <span
            className={`max-w-md text-right text-xs ${
              verdict.tone === "critical" ? "text-status-critical" : "text-ink-muted"
            }`}
          >
            {reasonText}
          </span>
        )}
      </div>
    </Card>
  );
}
