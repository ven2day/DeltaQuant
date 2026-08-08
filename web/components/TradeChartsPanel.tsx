"use client";

import {
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  ColorType,
  createChart,
} from "lightweight-charts";
import { CandlestickChart, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useCandles } from "@/lib/useCandles";
import type { Position, TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

const TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"];

// Matches web/app/globals.css's dark palette -- lightweight-charts takes literal
// colors, not CSS var references, so these are duplicated here deliberately.
const COLORS = {
  ink: "#e5e4e0",
  muted: "#898781",
  border: "rgba(255, 255, 255, 0.1)",
  good: "#0ca30c",
  critical: "#e66767",
  cat1: "#3987e5",
};

function PositionChart({ symbol, position }: { symbol: string; position: Position | null }) {
  const [timeframe, setTimeframe] = useState("15m");
  const { candles, loading } = useCandles(symbol, timeframe);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Chart + series are created once per mount, not per data update — updates go
  // through series.setData()/createPriceLine() in the effect below instead of
  // recreating the whole chart on every candle refresh.
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: COLORS.muted,
      },
      grid: {
        vertLines: { color: COLORS.border },
        horzLines: { color: COLORS.border },
      },
      rightPriceScale: { borderColor: COLORS.border },
      timeScale: { borderColor: COLORS.border, timeVisible: true },
      height: 320,
      autoSize: true,
    });
    const series = chart.addCandlestickSeries({
      upColor: COLORS.good,
      downColor: COLORS.critical,
      borderVisible: false,
      wickUpColor: COLORS.good,
      wickDownColor: COLORS.critical,
    });
    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => chart.applyOptions({ width: containerRef.current?.clientWidth });
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Data + price-line updates, separate from chart setup so they don't tear down
  // and rebuild the chart (which would lose zoom/scroll state) on every refresh.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    const chartData: CandlestickData[] = candles.map((c) => ({
      ...c,
      time: c.time as UTCTimestamp,
    }));
    series.setData(chartData);

    // createPriceLine has no built-in "replace" -- track and remove the
    // previous set before drawing the current entry/target/stop so stale
    // lines from a prior position/candle refresh don't accumulate.
    const priceLines: ReturnType<ISeriesApi<"Candlestick">["createPriceLine"]>[] = [];
    if (position) {
      const lines: [number | undefined, string, string][] = [
        [position.entry, "Entry", COLORS.cat1],
        [position.target, "Target", COLORS.good],
        [position.stop, "Stop", COLORS.critical],
        [position.current, "Current", COLORS.ink],
      ];
      for (const [price, title, color] of lines) {
        if (typeof price !== "number" || !Number.isFinite(price)) continue;
        priceLines.push(
          series.createPriceLine({
            price,
            color,
            lineWidth: 1,
            lineStyle: 2, // dashed
            axisLabelVisible: true,
            title,
          }),
        );
      }
    }
    return () => {
      for (const line of priceLines) series.removePriceLine(line);
    };
  }, [candles, position]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-end gap-1">
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
      <div ref={containerRef} className="w-full" style={{ height: 320 }} />
      {!loading && candles.length === 0 && (
        <div className="py-8 text-center italic text-ink-muted">
          No history available for {symbol} @ {timeframe} yet
        </div>
      )}
    </div>
  );
}

function formatEntryTime(iso: string | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function TradeChartsPanel({ stats }: { stats: TradingStats }) {
  const positions = useMemo(() => stats.open_positions ?? [], [stats.open_positions]);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  const activeSymbol = selectedSymbol ?? positions[0]?.symbol ?? null;
  const activePosition = positions.find((p) => p.symbol === activeSymbol) ?? null;

  return (
    <div className="space-y-4">
      <Card title="Trade Chart" icon={CandlestickChart} accent="var(--cat-1)">
        {positions.length > 1 && (
          <div className="mb-3 flex flex-wrap gap-1">
            {positions.map((p) => (
              <button
                key={p.symbol}
                type="button"
                onClick={() => setSelectedSymbol(p.symbol)}
                className={`rounded px-2 py-1 text-xs font-medium ${
                  p.symbol === activeSymbol
                    ? "bg-cat-1 text-white"
                    : "bg-surface-raised text-ink-muted hover:text-ink-primary"
                }`}
              >
                {p.symbol}
              </button>
            ))}
          </div>
        )}
        {activeSymbol ? (
          <PositionChart symbol={activeSymbol} position={activePosition} />
        ) : (
          <div className="py-16 text-center italic text-ink-muted">
            No open positions to chart yet
          </div>
        )}
      </Card>

      <Card title="Open Trades" icon={CandlestickChart} accent="var(--ink-muted)">
        {positions.length === 0 ? (
          <div className="italic text-ink-muted">No open positions</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full whitespace-nowrap">
              <thead>
                <tr className="text-left text-xs text-ink-muted">
                  <th className="pb-2 pr-4 font-medium">Symbol</th>
                  <th className="pb-2 pr-4 font-medium">Side</th>
                  <th className="pb-2 pr-4 text-right font-medium">Qty</th>
                  <th className="pb-2 pr-4 text-right font-medium">Entry</th>
                  <th className="pb-2 pr-4 text-right font-medium">Current</th>
                  <th className="pb-2 pr-4 text-right font-medium">Stop</th>
                  <th className="pb-2 pr-4 text-right font-medium">Target</th>
                  <th className="pb-2 pr-4 text-right font-medium">P&amp;L</th>
                  <th className="pb-2 pr-4 font-medium">Entry Time (IST)</th>
                  <th className="pb-2 pr-4 font-medium">Signal</th>
                  <th className="pb-2 font-medium">Timeframe</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {positions.map((pos, i) => (
                  <tr
                    key={`${pos.symbol}-${i}`}
                    className={pos.symbol === activeSymbol ? "bg-surface-raised" : undefined}
                  >
                    <td className="py-1.5 pr-4 font-medium text-ink-primary">{pos.symbol}</td>
                    <td className="py-1.5 pr-4">
                      <Badge label={pos.side} tone={pos.side === "BUY" ? "good" : "critical"} />
                    </td>
                    <td className="tabular py-1.5 pr-4 text-right">{pos.qty}</td>
                    <td className="tabular py-1.5 pr-4 text-right">Rs.{pos.entry.toFixed(2)}</td>
                    <td className="tabular py-1.5 pr-4 text-right">
                      {typeof pos.current === "number" ? `Rs.${pos.current.toFixed(2)}` : "—"}
                    </td>
                    <td className="tabular py-1.5 pr-4 text-right text-status-critical">
                      {typeof pos.stop === "number" ? `Rs.${pos.stop.toFixed(2)}` : "—"}
                    </td>
                    <td className="tabular py-1.5 pr-4 text-right text-status-good">
                      {typeof pos.target === "number" ? `Rs.${pos.target.toFixed(2)}` : "—"}
                    </td>
                    <td
                      className={`tabular py-1.5 pr-4 text-right font-medium ${
                        pos.pnl >= 0 ? "text-status-good" : "text-status-critical"
                      }`}
                    >
                      {pos.pnl >= 0 ? "+" : ""}Rs.{pos.pnl.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-4 text-ink-muted">{formatEntryTime(pos.entry_time)}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{pos.strategy || "—"}</td>
                    <td className="py-1.5 text-ink-muted">{pos.timeframe || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="OpenCharts"
        icon={CandlestickChart}
        accent="var(--cat-1)"
        right={
          <a
            href="/opencharts/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-ink-muted hover:text-ink-primary"
          >
            Open in new tab <ExternalLink size={12} />
          </a>
        }
      >
        <div className="text-xs text-ink-muted mb-2">
          Full-featured self-hosted charting terminal (github.com/dylanpersonguy/OpenCharts) —
          its own bundled demo market data and paper-trading sandbox, not connected to
          DeltaQuant&apos;s live trades.
        </div>
        <iframe
          src="/opencharts/"
          title="OpenCharts"
          className="w-full rounded-lg border border-border"
          style={{ height: 640 }}
        />
      </Card>
    </div>
  );
}
