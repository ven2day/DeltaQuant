"use client";

import { GitBranch, Newspaper, Gauge, Brain, Compass, ListChecks } from "lucide-react";
import { useMemo, useState } from "react";
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

function RejectionReasons({ stats }: { stats: TradingStats }) {
  const [strategy, setStrategy] = useState("");
  const [timeframe, setTimeframe] = useState("");
  const [symbol, setSymbol] = useState("");
  const details = stats.rejection_drilldown ?? [];
  const strategies = [...new Set(details.map((row) => row.strategy).filter(Boolean))].sort();
  const timeframes = [...new Set(details.map((row) => row.timeframe).filter(Boolean))].sort();
  const filtered = useMemo(() => {
    if (!strategy && !timeframe && !symbol.trim()) return stats.rejection_reasons ?? [];
    const normalizedSymbol = symbol.trim().toUpperCase();
    const grouped = new Map<string, { stage: string; reason: string; count: number }>();
    for (const row of details) {
      if (strategy && row.strategy !== strategy) continue;
      if (timeframe && row.timeframe !== timeframe) continue;
      if (normalizedSymbol && !(row.symbol ?? "").toUpperCase().includes(normalizedSymbol)) continue;
      const key = `${row.stage}|${row.reason}`;
      const current = grouped.get(key) ?? { stage: row.stage, reason: row.reason, count: 0 };
      current.count += row.count;
      grouped.set(key, current);
    }
    const total = [...grouped.values()].reduce((sum, row) => sum + row.count, 0);
    return [...grouped.values()]
      .map((row) => ({ ...row, percentage: total ? (100 * row.count) / total : 0 }))
      .sort((a, b) => b.count - a.count);
  }, [details, stats.rejection_reasons, strategy, timeframe, symbol]);

  if (!filtered.length && !details.length) return null;
  return (
    <div className="mt-3 rounded-lg border border-border/60 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="mr-auto text-[10.5px] font-semibold uppercase tracking-wide text-ink-muted">
          Top rejection reasons this cycle
        </span>
        <select
          value={strategy}
          onChange={(event) => setStrategy(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-[11px] text-ink-secondary"
        >
          <option value="">All strategies</option>
          {strategies.map((value) => <option key={value}>{value}</option>)}
        </select>
        <select
          value={timeframe}
          onChange={(event) => setTimeframe(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-[11px] text-ink-secondary"
        >
          <option value="">All timeframes</option>
          {timeframes.map((value) => <option key={value}>{value}</option>)}
        </select>
        <input
          value={symbol}
          onChange={(event) => setSymbol(event.target.value)}
          placeholder="Symbol"
          className="w-24 rounded border border-border bg-transparent px-2 py-1 text-[11px] text-ink-secondary outline-none"
        />
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-x-3 gap-y-1 text-xs">
        {filtered.slice(0, 8).map((row) => (
          <div key={`${row.stage}-${row.reason}`} className="contents">
            <span className="truncate text-ink-secondary">{row.reason}</span>
            <span className="tabular text-right text-ink-primary">{row.count}</span>
            <span className="tabular text-right text-ink-muted">{row.percentage.toFixed(1)}%</span>
          </div>
        ))}
      </div>
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

  const funnel = stats.signal_funnel && "raw_signals" in stats.signal_funnel
    ? stats.signal_funnel
    : null;
  const scanState: StageState = (funnel?.symbols_scanned ?? 0) > 0 ? "pass" : "neutral";
  const validationState: StageState = (funnel?.signal_validation_pass ?? 0) > 0
    ? "pass"
    : (funnel?.signal_validation_reject ?? 0) > 0 ? "fail" : "neutral";
  const riskState: StageState = (funnel?.risk_approved ?? 0) > 0
    ? "pass"
    : (funnel?.risk_blocked ?? 0) > 0 ? "fail" : "neutral";

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
    stats.ai_review_reason;

  return (
    <Card title="Signal Pipeline & Agent Activity" icon={GitBranch} accent="var(--cat-1)">
      <div className="flex flex-wrap gap-x-5 gap-y-4 pb-4">
        <Stage
          label="Scan"
          state={scanState}
          detail={`${funnel?.symbols_scanned ?? 0}/${funnel?.symbols_requested ?? 0} symbols · ${funnel?.strategy_evaluations ?? 0} evaluations`}
        />
        <Stage
          label="Raw"
          state={(funnel?.raw_signals ?? 0) > 0 ? "pass" : "neutral"}
          detail={`${funnel?.raw_signals ?? 0} this cycle · ${stats.signals_generated} session total`}
        />
        <Stage
          label="Technical setup"
          state={(funnel?.technical_setups ?? 0) > 0 ? "pass" : "neutral"}
          detail={`${funnel?.technical_setups ?? 0} setups`}
        />
        <Stage
          label="Strategy eligibility"
          state={(funnel?.registry_eligible ?? 0) > 0 ? "pass" : (funnel?.registry_blocked ?? 0) > 0 ? "fail" : "neutral"}
          detail={`${funnel?.registry_eligible ?? 0} eligible · ${funnel?.registry_shadow_only ?? 0} shadow · ${funnel?.registry_blocked ?? 0} blocked`}
        />
        <Stage
          label="Qualified"
          state={(funnel?.deterministic_qualified ?? 0) > 0 ? "pass" : (funnel?.deterministic_rejected ?? 0) > 0 ? "fail" : "neutral"}
          detail={`${funnel?.deterministic_qualified ?? 0} passed · ${funnel?.deterministic_rejected ?? 0} rejected`}
        />
        <Stage
          label="ML overlay"
          state={(funnel?.ml_pass ?? 0) > 0 ? "pass" : (funnel?.ml_abstain ?? 0) > 0 ? "fail" : "neutral"}
          detail={`${funnel?.ml_candidates ?? 0} candidates · ${funnel?.ml_inference ?? 0} inference · ${funnel?.ml_abstain ?? 0} abstain`}
        />
        <Stage
          label="Qwen context"
          state={(funnel?.qwen_external_calls ?? 0) > 0 || (funnel?.qwen_cache_hits ?? 0) > 0 ? "pass" : "neutral"}
          detail={`${funnel?.qwen_required ?? 0} required · ${funnel?.qwen_external_calls ?? 0} calls · ${funnel?.qwen_cache_hits ?? 0} cache`}
        />
        <Stage
          label="Validation"
          state={validationState}
          detail={`${funnel?.signal_validation_pass ?? 0} passed · ${funnel?.signal_validation_reject ?? 0} rejected`}
        />
        <Stage
          label="Risk"
          state={riskState}
          detail={`${funnel?.risk_approved ?? 0} approved · ${funnel?.risk_blocked ?? 0} blocked`}
        />
        <Stage
          label="Final"
          state={(funnel?.final_buy ?? 0) > 0 ? "pass" : (funnel?.final_reject ?? 0) > 0 ? "fail" : "neutral"}
          detail={`${funnel?.final_buy ?? 0} orders · ${funnel?.final_hold ?? 0} hold · ${funnel?.final_wait ?? 0} wait · ${funnel?.final_reject ?? 0} reject`}
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

      {stats.ai_review_status && stats.ai_review_status !== "completed" && (
        <div className="mt-3 rounded-lg border border-border/60 bg-white/[0.02] px-3 py-2 text-xs text-ink-secondary">
          <span className="font-semibold">AI activity:</span>{" "}
          {stats.ai_review_reason || "Waiting for a qualifying candidate"}
        </div>
      )}

      <RejectionReasons stats={stats} />

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
            <div className="text-xs italic text-ink-muted">
              {stats.ai_review_reason || "No headlines fetched yet"}
            </div>
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
            <div className="text-xs italic text-ink-muted">
              {stats.ai_review_reason || "No ML predictions yet"}
            </div>
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
              <span className="italic text-ink-muted">
                {stats.ai_review_reason ||
                  "Candidate-context reasoning did not run; see the funnel and rejection counts above."}
              </span>
            )}
          </p>
        </DetailSection>
      </div>
    </Card>
  );
}
