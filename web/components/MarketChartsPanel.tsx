"use client";

import { CandlestickChart } from "lucide-react";
import { useMemo, useState } from "react";
import type { ChartSeries, TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

const TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"];
const WIDTH = 920;
const HEIGHT = 280;
const PAD_LEFT = 34;
const PAD_RIGHT = 112;
const PAD_Y = 34;

function PriceChart({ series }: { series: ChartSeries }) {
  const geometry = useMemo(() => {
    const points = series.points ?? [];
    const levels = [series.entry, series.current, series.target, series.stop].filter(
      (value): value is number => typeof value === "number" && Number.isFinite(value),
    );
    const prices = points.flatMap((point) => [point.low, point.high]).concat(levels);
    if (points.length < 2 || prices.length === 0) return null;
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = Math.max(max - min, max * 0.005, 0.01);
    const x = (index: number) =>
      PAD_LEFT + (index / (points.length - 1)) * (WIDTH - PAD_LEFT - PAD_RIGHT);
    const y = (value: number) =>
      HEIGHT - PAD_Y - ((value - min) / range) * (HEIGHT - PAD_Y * 2);
    const path = points
      .map((point, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(point.close).toFixed(1)}`)
      .join(" ");
    return { points, x, y, path, min, max };
  }, [series]);

  if (!geometry) return <div className="py-12 text-center italic text-ink-muted">No chart data</div>;

  const markers = [
    ["Entry", series.entry, "var(--cat-1)"],
    ["Current", series.current, "var(--ink-primary)"],
    ["Target", series.target, "var(--status-good)"],
    ["Stop", series.stop, "var(--status-critical)"],
  ] as const;

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full" role="img">
      {[0.25, 0.5, 0.75].map((fraction) => (
        <line
          key={fraction}
          x1={PAD_LEFT}
          x2={WIDTH - PAD_RIGHT}
          y1={PAD_Y + fraction * (HEIGHT - PAD_Y * 2)}
          y2={PAD_Y + fraction * (HEIGHT - PAD_Y * 2)}
          stroke="var(--gridline)"
        />
      ))}
      {geometry.points.map((point, index) => {
        const x = geometry.x(index);
        const up = point.close >= point.open;
        return (
          <line
            key={`${point.time}-${index}`}
            x1={x}
            x2={x}
            y1={geometry.y(point.high)}
            y2={geometry.y(point.low)}
            stroke={up ? "var(--status-good)" : "var(--status-critical)"}
            opacity="0.28"
          />
        );
      })}
      <path d={geometry.path} fill="none" stroke="var(--cat-1)" strokeWidth="2.2" />
      {markers.map(([label, value, color]) => {
        if (typeof value !== "number" || !Number.isFinite(value)) return null;
        const y = geometry.y(value);
        return (
          <g key={label}>
            <line
              x1={PAD_LEFT}
              x2={WIDTH - PAD_RIGHT}
              y1={y}
              y2={y}
              stroke={color}
              strokeDasharray="6 5"
            />
            <text x={WIDTH - PAD_RIGHT + 5} y={y + 4} fill={color} fontSize="11">
              {label} {value.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function MarketChartsPanel({ stats }: { stats: TradingStats }) {
  const [timeframe, setTimeframe] = useState("5m");
  const series = stats.chart_timeframes?.[timeframe];
  return (
    <Card title="Simulated Market Structure" icon={CandlestickChart} accent="var(--cat-1)">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-ink-primary">{stats.chart_symbol || "No symbol"}</span>
          {series?.status && <Badge label={series.status} tone={series.status === "OPEN" ? "good" : "neutral"} />}
          <span className="text-xs text-ink-muted">Event {stats.simulation_event_time || "—"}</span>
        </div>
        <div className="flex gap-1">
          {TIMEFRAMES.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setTimeframe(value)}
              className={`rounded px-2 py-1 text-xs ${
                timeframe === value
                  ? "bg-cat-1 text-white"
                  : "bg-surface-raised text-ink-muted hover:text-ink-primary"
              }`}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      {series ? <PriceChart series={series} /> : <div className="py-12 text-center italic text-ink-muted">Waiting for a candidate</div>}
    </Card>
  );
}
