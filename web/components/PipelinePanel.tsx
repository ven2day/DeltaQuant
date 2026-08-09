"use client";

import { GitBranch, Newspaper, Gauge, Brain, Compass, ListChecks } from "lucide-react";
import { useMemo } from "react";
import type { ReactNode } from "react";
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

// Signal History timestamps are naive local (server) time, no offset -- same
// convention SignalHistoryPanel already assumes elsewhere in this codebase.
function ageLabel(timestamp: string): { text: string; stale: boolean } {
  const ageMs = Date.now() - new Date(timestamp).getTime();
  const minutes = Math.floor(ageMs / 60_000);
  if (minutes < 1) return { text: "just now", stale: false };
  if (minutes < 60) return { text: `${minutes}m ago`, stale: minutes >= 10 };
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return { text: `${hours}h ago`, stale: true };
  const days = Math.floor(hours / 24);
  return { text: `${days}d ago`, stale: true };
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

function DetailSection({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Newspaper;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border/60 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-wide text-ink-muted">
        <Icon size={11} strokeWidth={2.5} />
        {label}
      </div>
      {children}
    </div>
  );
}

const MOOD_COLOR: Record<string, string> = {
  extreme_fear: "var(--status-critical)",
  fear: "var(--status-serious)",
  neutral: "var(--ink-muted)",
  greed: "var(--status-good)",
  extreme_greed: "var(--status-good)",
};

function MoodGauge({ mood }: { mood: TradingStats["market_mood"] }) {
  if (!mood || !("mood_index" in mood)) {
    return <div className="text-xs italic text-ink-muted">No mood data yet this session</div>;
  }
  const color = MOOD_COLOR[mood.mood_label] ?? "var(--ink-muted)";
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-ink-primary">
          {mood.mood_index}/100 · {mood.mood_label.replace("_", " ")}
        </span>
        <span className="text-ink-muted">confidence {(mood.confidence * 100).toFixed(0)}%</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full transition-[width]"
          style={{ width: `${mood.mood_index}%`, backgroundColor: color }}
        />
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-ink-muted">
        <span>news {mood.news_score >= 0 ? "+" : ""}{mood.news_score.toFixed(2)}</span>
        <span>volatility {mood.volatility_score.toFixed(0)}</span>
        <span>breadth {mood.breadth_score >= 0 ? "+" : ""}{mood.breadth_score.toFixed(2)}</span>
      </div>
    </div>
  );
}

function PredictionRow({ p }: { p: TradingStats["prediction_signals"][number] }) {
  if (p.abstained) {
    return (
      <div className="flex items-center justify-between py-0.5 text-xs">
        <span className="font-medium text-ink-primary">{p.symbol}</span>
        <span className="italic text-ink-muted">abstained — not enough evidence</span>
      </div>
    );
  }
  const tone = p.direction === "up" ? "good" : p.direction === "down" ? "critical" : "neutral";
  return (
    <div className="flex items-center justify-between py-0.5 text-xs">
      <span className="font-medium text-ink-primary">{p.symbol}</span>
      <Badge label={`${p.direction.toUpperCase()} ${(p.confidence * 100).toFixed(0)}%`} tone={tone} />
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

  const approvalRate =
    stats.signals_generated > 0
      ? ((stats.signals_validated / stats.signals_generated) * 100).toFixed(0) + "%"
      : "—";

  return (
    <Card title="Signal Pipeline & Agent Activity" icon={GitBranch} accent="var(--cat-1)">
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
          detail={`${stats.signals_validated} passed · ${stats.signals_rejected} rejected · ${approvalRate}`}
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
            {verdictRecord && (() => {
              const age = ageLabel(verdictRecord.timestamp);
              return (
                <span className="text-ink-muted">
                  {" "}
                  ({age.text}
                  {age.stale ? " — from an earlier cycle, not this one" : ""})
                </span>
              );
            })()}
          </span>
        )}
      </div>

      {stats.agent_fallback_notice && (
        <div
          className="mt-3 rounded-lg border px-3 py-2 text-xs"
          style={{
            borderColor: "rgba(250,178,25,0.35)",
            backgroundColor: "rgba(250,178,25,0.08)",
            color: "var(--status-warning)",
          }}
        >
          <span className="font-semibold">AI review skipped this cycle:</span>{" "}
          {stats.agent_fallback_notice}. A simplified backup rule ran instead — expect
          rougher decisions until this clears.
        </div>
      )}

      <div className="mt-3 grid grid-cols-1 gap-3 border-t border-border pt-3 md:grid-cols-2 xl:grid-cols-3">
        <DetailSection icon={Newspaper} label="News Analyst">
          {stats.news_headlines.length > 0 ? (
            <div>
              <div className="mb-1 text-[11px] text-ink-muted">
                avg sentiment{" "}
                <span className={stats.news_sentiment >= 0 ? "text-status-good" : "text-status-critical"}>
                  {stats.news_sentiment >= 0 ? "+" : ""}
                  {stats.news_sentiment.toFixed(2)}
                </span>
              </div>
              <ul className="space-y-1">
                {stats.news_headlines.slice(0, 4).map((h, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-ink-secondary">
                    <span
                      className="mt-1 h-1.5 w-1.5 flex-none rounded-full"
                      style={{
                        backgroundColor:
                          h.sentiment === "positive" ? "var(--status-good)" : "var(--status-critical)",
                      }}
                    />
                    <span className="line-clamp-2">{h.title}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="text-xs italic text-ink-muted">No headlines fetched this cycle</div>
          )}
        </DetailSection>

        <DetailSection icon={Gauge} label="Sentiment Agent · Market Mood">
          <MoodGauge mood={stats.market_mood} />
        </DetailSection>

        <DetailSection icon={Brain} label="Prediction Agent · ML Ensemble">
          {stats.prediction_signals.length > 0 ? (
            <div>
              {stats.prediction_signals.map((p, i) => (
                <PredictionRow key={i} p={p} />
              ))}
            </div>
          ) : (
            <div className="text-xs italic text-ink-muted">No predictions this cycle</div>
          )}
        </DetailSection>

        <DetailSection icon={Compass} label="Market Regime">
          <p className="line-clamp-4 text-xs leading-relaxed text-ink-secondary">
            {stats.regime_reasoning || (
              <span className="italic text-ink-muted">No reasoning recorded yet</span>
            )}
          </p>
        </DetailSection>

        <DetailSection icon={ListChecks} label="Strategy Selection">
          <p className="line-clamp-4 text-xs leading-relaxed text-ink-secondary">
            {stats.strategy_reasoning || (
              <span className="italic text-ink-muted">No reasoning recorded yet</span>
            )}
          </p>
        </DetailSection>
      </div>
    </Card>
  );
}
