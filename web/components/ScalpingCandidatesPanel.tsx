"use client";

import { Zap } from "lucide-react";
import { useState } from "react";
import type { ScalpingCandidate, TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";

// Column order matches src/market/scalping_screener.py's DEFAULT_TIMEFRAMES (finest first).
const TIMEFRAME_COLUMNS = ["15m", "30m", "1h"];

// Price tiers a symbol's last_price is bucketed into — the swing/day counts alone don't
// tell you if a symbol is actually affordable to trade in size; a ₹14,000 stock and a
// ₹300 stock can have the same swing count but need very different capital per lot.
const PRICE_TIERS: { label: string; min: number; max: number }[] = [
  { label: "Small (< Rs.500)", min: 0, max: 500 },
  { label: "Medium (Rs.500 – Rs.2,000)", min: 500, max: 2000 },
  { label: "Large (Rs.2,000 – Rs.5,000)", min: 2000, max: 5000 },
  { label: "Very Large (> Rs.5,000)", min: 5000, max: Infinity },
];

function TimeframeCell({ stats }: { stats: ScalpingCandidate["timeframes"][string] | undefined }) {
  if (!stats) {
    return <span className="text-ink-muted">—</span>;
  }
  return (
    <div className="tabular text-right">
      <span className="font-medium text-status-good">{stats.avg_swings_per_day.toFixed(1)}</span>
      <span className="ml-1 text-[10px] text-ink-muted">Rs.{stats.avg_swing_size.toFixed(0)}</span>
    </div>
  );
}

function CandidatesTable({ candidates }: { candidates: ScalpingCandidate[] }) {
  if (candidates.length === 0) {
    return <div className="py-3 italic text-ink-muted">No candidates in this price range.</div>;
  }
  return (
    <table className="w-full border-collapse text-xs">
      <thead>
        <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
          <th className="py-2 pr-3 font-medium">Symbol</th>
          {TIMEFRAME_COLUMNS.map((tf) => (
            <th key={tf} className="py-2 pr-3 text-right font-medium">
              {tf}
            </th>
          ))}
          <th className="py-2 pr-3 text-right font-medium">Avg Range</th>
          <th className="py-2 text-right font-medium">Last Price</th>
        </tr>
      </thead>
      <tbody>
        {candidates.map((c) => (
          <tr key={c.symbol} className="border-b border-border/50">
            <td className="py-1.5 pr-3 font-medium text-ink-primary">{c.symbol}</td>
            {TIMEFRAME_COLUMNS.map((tf) => (
              <td key={tf} className="py-1.5 pr-3">
                <TimeframeCell stats={c.timeframes?.[tf]} />
              </td>
            ))}
            <td className="tabular py-1.5 pr-3 text-right text-ink-secondary">
              Rs.{(c.avg_daily_range ?? 0).toFixed(2)}
            </td>
            <td className="tabular py-1.5 text-right text-ink-secondary">
              Rs.{(c.last_price ?? 0).toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ScalpingCandidatesPanel({ stats }: { stats: TradingStats }) {
  const candidates = stats.scalping_candidates ?? [];
  const scanStatus = stats.scalping_screener_status ?? "pending";
  const dataSource = stats.scalping_screener_data_source || stats.data_source;
  const [activeTier, setActiveTier] = useState(0);

  const emptyMessage =
    scanStatus === "ready"
      ? `Latest ${dataSource} scan completed; no symbols met the configured 0.5% swing-frequency threshold.`
      : scanStatus === "error"
        ? `Scalping scanner is unavailable in ${dataSource} mode. See Activity Log for details.`
        : "Waiting for the first scalping scan…";

  const tierCandidates = candidates.filter(
    (c) =>
      (c.last_price ?? 0) >= PRICE_TIERS[activeTier].min &&
      (c.last_price ?? 0) < PRICE_TIERS[activeTier].max
  );

  return (
    <Card title="Scalping Candidates" icon={Zap} accent="var(--cat-3)">
      {candidates.length === 0 ? (
        <div className="italic text-ink-muted">{emptyMessage}</div>
      ) : (
        <div>
          <div className="mb-3 flex gap-1 border-b border-border">
            {PRICE_TIERS.map((tier, i) => {
              const count = candidates.filter(
                (c) => (c.last_price ?? 0) >= tier.min && (c.last_price ?? 0) < tier.max
              ).length;
              const active = i === activeTier;
              return (
                <button
                  key={tier.label}
                  type="button"
                  onClick={() => setActiveTier(i)}
                  className={`border-b-2 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide transition-colors ${
                    active
                      ? "border-[var(--cat-3)] text-ink-primary"
                      : "border-transparent text-ink-muted hover:text-ink-secondary"
                  }`}
                >
                  {tier.label} ({count})
                </button>
              );
            })}
          </div>
          <div className="overflow-x-auto">
            <CandidatesTable candidates={tierCandidates} />
          </div>
          <div className="mt-3 text-[10px] text-ink-muted">
            Each cell: swings/day at that timeframe, and the average ₹ size of each swing.
            Ranked by 15m swing frequency within each price tier — the fastest-moving
            symbols first.
          </div>
        </div>
      )}
    </Card>
  );
}
