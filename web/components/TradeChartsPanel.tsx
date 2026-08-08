"use client";

import {
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
  ColorType,
  createChart,
  TickMarkType,
} from "lightweight-charts";
import { CandlestickChart, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { type Candle, useCandles } from "@/lib/useCandles";
import { useUniverse } from "@/lib/useUniverse";
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

// NSE trades in IST regardless of who's looking at the dashboard --
// lightweight-charts formats Time using the browser's own local timezone by
// default, which silently shows the wrong clock hours for any viewer not
// physically in IST. Our candle timestamps are already correct absolute epoch
// seconds (see src/webui/candles.py); only the *display* needs pinning to IST.
const IST_TIME_ZONE = "Asia/Kolkata";

function toDate(time: Time): Date {
  // Candles from useCandles() are always UTCTimestamp (epoch seconds) -- never
  // the BusinessDay-object form of Time -- so this cast is safe here.
  return new Date((time as UTCTimestamp) * 1000);
}

function istTickMarkFormatter(time: Time, tickMarkType: TickMarkType): string {
  const date = toDate(time);
  switch (tickMarkType) {
    case TickMarkType.Year:
      return new Intl.DateTimeFormat("en-IN", { year: "numeric", timeZone: IST_TIME_ZONE }).format(
        date,
      );
    case TickMarkType.Month:
      return new Intl.DateTimeFormat("en-IN", { month: "short", timeZone: IST_TIME_ZONE }).format(
        date,
      );
    case TickMarkType.DayOfMonth:
      return new Intl.DateTimeFormat("en-IN", {
        day: "2-digit",
        month: "short",
        timeZone: IST_TIME_ZONE,
      }).format(date);
    case TickMarkType.TimeWithSeconds:
      return new Intl.DateTimeFormat("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: IST_TIME_ZONE,
      }).format(date);
    case TickMarkType.Time:
    default:
      return new Intl.DateTimeFormat("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: IST_TIME_ZONE,
      }).format(date);
  }
}

function formatVolume(volume: number): string {
  if (volume >= 1_000_000) return `${(volume / 1_000_000).toFixed(2)}M`;
  if (volume >= 1_000) return `${(volume / 1_000).toFixed(1)}K`;
  return volume.toFixed(0);
}

function istCrosshairFormatter(time: Time): string {
  const date = toDate(time);
  const formatted = new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: IST_TIME_ZONE,
  }).format(date);
  return `${formatted} IST`;
}

function PositionChart({ symbol, position }: { symbol: string; position: Position | null }) {
  const [timeframe, setTimeframe] = useState("15m");
  // Explicit opt-in preview of the simulated pipeline for a symbol with no
  // simulated position of its own -- proves the mock data is genuinely alive
  // without waiting for a trade to open. Reset on symbol change so switching
  // to a different stock never carries the preview over by accident.
  const [previewSimulated, setPreviewSimulated] = useState(false);
  useEffect(() => setPreviewSimulated(false), [symbol]);
  const isSimulated = position?.entry_data_source === "simulated" || previewSimulated;
  const { candles, loading } = useCandles(symbol, timeframe, 300, previewSimulated);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  // OHLCV readout: the hovered candle under the crosshair, or the latest one
  // when the pointer isn't over the chart -- same "always show real numbers"
  // convention TradingView-style charts use, not just the candle shapes.
  const [legendCandle, setLegendCandle] = useState<Candle | null>(null);
  // Crosshair subscription is registered once at setup and must never see a
  // stale `candles` array from its mount-time closure -- the data effect below
  // keeps this ref current on every refresh instead.
  const candlesRef = useRef<Candle[]>([]);

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
      // Leave the bottom 22% of the pane clear for the volume histogram below,
      // so candle wicks never overlap the volume bars.
      rightPriceScale: { borderColor: COLORS.border, scaleMargins: { top: 0.1, bottom: 0.25 } },
      timeScale: {
        borderColor: COLORS.border,
        timeVisible: true,
        tickMarkFormatter: istTickMarkFormatter,
      },
      localization: { timeFormatter: istCrosshairFormatter },
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
    // Its own price scale (not the candlesticks' one) pinned to the bottom
    // strip, so volume bars are readable without distorting the price axis.
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    chartRef.current = chart;
    seriesRef.current = series;
    volumeSeriesRef.current = volumeSeries;

    chart.subscribeCrosshairMove((param) => {
      const latest = candlesRef.current[candlesRef.current.length - 1] ?? null;
      if (!param.time) {
        setLegendCandle(latest);
        return;
      }
      const hovered = candlesRef.current.find((c) => c.time === param.time);
      setLegendCandle(hovered ?? latest);
    });

    const handleResize = () => chart.applyOptions({ width: containerRef.current?.clientWidth });
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      volumeSeriesRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Data + price-line updates, separate from chart setup so they don't tear down
  // and rebuild the chart (which would lose zoom/scroll state) on every refresh.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    candlesRef.current = candles;
    setLegendCandle(candles[candles.length - 1] ?? null);
    const chartData: CandlestickData[] = candles.map((c) => ({
      ...c,
      time: c.time as UTCTimestamp,
    }));
    series.setData(chartData);

    const volumeSeries = volumeSeriesRef.current;
    if (volumeSeries) {
      const volumeData: HistogramData[] = candles.map((c) => ({
        time: c.time as UTCTimestamp,
        value: c.volume,
        color: c.close >= c.open ? "rgba(12,163,12,0.5)" : "rgba(230,103,103,0.5)",
      }));
      volumeSeries.setData(volumeData);
    }

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
      <div className="mb-2 flex items-center justify-between gap-1">
        {position?.entry_data_source === "simulated" ? (
          <span />
        ) : (
          <button
            type="button"
            onClick={() => setPreviewSimulated((v) => !v)}
            title="Preview the simulated (mock) data pipeline for this symbol -- proves it's alive without waiting for a trade to open on it"
            className="rounded px-2 py-1 text-xs font-medium text-ink-muted hover:text-ink-primary"
            style={
              previewSimulated
                ? { backgroundColor: "var(--status-warning)", color: "#1a1a1a" }
                : { backgroundColor: "var(--surface-raised)" }
            }
          >
            {previewSimulated ? "Showing simulated" : "Preview simulated"}
          </button>
        )}
        <div className="flex items-center gap-1">
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
      <div className="relative">
        <div ref={containerRef} className="w-full" style={{ height: 320 }} />
        {legendCandle && (
          <div
            className="tabular pointer-events-none absolute left-2 top-2 flex flex-wrap gap-x-3 gap-y-0.5 rounded px-2 py-1 text-[11px] text-ink-muted backdrop-blur-sm"
            style={{ backgroundColor: "rgba(32, 31, 30, 0.8)" }}
          >
            <span>
              O <span className="text-ink-primary">{legendCandle.open.toFixed(2)}</span>
            </span>
            <span>
              H <span className="text-status-good">{legendCandle.high.toFixed(2)}</span>
            </span>
            <span>
              L <span className="text-status-critical">{legendCandle.low.toFixed(2)}</span>
            </span>
            <span>
              C{" "}
              <span
                className={
                  legendCandle.close >= legendCandle.open
                    ? "text-status-good"
                    : "text-status-critical"
                }
              >
                {legendCandle.close.toFixed(2)}
              </span>
            </span>
            <span>
              Vol <span className="text-ink-primary">{formatVolume(legendCandle.volume)}</span>
            </span>
          </div>
        )}
        {isSimulated && (
          <div
            className="pointer-events-none absolute inset-0 flex items-center justify-center select-none"
            aria-hidden="true"
          >
            <span
              className="text-4xl font-bold uppercase tracking-widest opacity-10"
              style={{ color: "var(--status-warning)", transform: "rotate(-20deg)" }}
            >
              Simulated
            </span>
          </div>
        )}
      </div>
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

function SymbolSearch({
  universe,
  onSelect,
}: {
  universe: string[];
  onSelect: (symbol: string) => void;
}) {
  const [query, setQuery] = useState("");

  const submit = (raw: string) => {
    const symbol = raw.trim().toUpperCase();
    if (symbol) onSelect(symbol);
    setQuery("");
  };

  return (
    <form
      className="mb-3 flex items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        submit(query);
      }}
    >
      <div className="relative flex-1 max-w-xs">
        <Search size={14} className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-ink-muted" />
        <input
          type="text"
          list="nse-universe"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Chart any NSE stock (e.g. RELIANCE)"
          className="w-full rounded bg-surface-raised py-1.5 pl-7 pr-2 text-xs text-ink-primary placeholder:text-ink-muted focus:outline-none"
        />
        <datalist id="nse-universe">
          {universe.map((symbol) => (
            <option key={symbol} value={symbol} />
          ))}
        </datalist>
      </div>
      <button
        type="submit"
        className="rounded bg-surface-raised px-2 py-1.5 text-xs text-ink-muted hover:text-ink-primary"
      >
        Chart
      </button>
    </form>
  );
}

export function TradeChartsPanel({ stats }: { stats: TradingStats }) {
  const positions = useMemo(() => stats.open_positions ?? [], [stats.open_positions]);
  const universe = useUniverse();
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  // With no open position and nothing searched yet, default to a real symbol
  // (RELIANCE if it's in the universe, else whatever loads first) instead of an
  // empty "search above" prompt -- so the chart always has something real on
  // screen to prove the feature actually works, not just a blank state.
  const defaultSymbol = universe.includes("RELIANCE") ? "RELIANCE" : (universe[0] ?? null);
  const activeSymbol = selectedSymbol ?? positions[0]?.symbol ?? defaultSymbol;
  const activePosition = positions.find((p) => p.symbol === activeSymbol) ?? null;

  return (
    <div className="space-y-4">
      <Card title="Trade Chart" icon={CandlestickChart} accent="var(--cat-1)">
        <SymbolSearch universe={universe} onSelect={setSelectedSymbol} />
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
        {activePosition?.entry_data_source === "simulated" && (
          <div className="mb-2">
            <Badge label="SIMULATED ENTRY — market was closed" tone="warning" />
          </div>
        )}
        {activeSymbol ? (
          <PositionChart symbol={activeSymbol} position={activePosition} />
        ) : (
          <div className="py-16 text-center italic text-ink-muted">Loading symbol list…</div>
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
                    <td className="py-1.5 pr-4 font-medium text-ink-primary">
                      <div className="flex items-center gap-1.5">
                        {pos.symbol}
                        {pos.entry_data_source === "simulated" && (
                          <Badge label="SIM" tone="warning" />
                        )}
                      </div>
                    </td>
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
    </div>
  );
}
