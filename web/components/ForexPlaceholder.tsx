import { ArrowLeftRight } from "lucide-react";
import { Card } from "./ui/Card";

const PLAN_STEPS = [
  "OANDA quotes + historical candles (mirrors the DhanHQ integration)",
  "Durable state partitioned by market -- separate wallet, positions, and risk limits from NSE",
  "Regime signals (Hurst exponent, Kaufman Efficiency Ratio) feeding the existing Market Regime agent",
  "Pip/lot-aware position sizing and a spread-based cost model",
  "A second, parallel agent pipeline cycle -- same agents, own cadence and risk limits",
  "Walk-forward validation (H-8 gate) on Forex data before anything trades live-paper",
];

export function ForexPlaceholder() {
  return (
    <Card title="Forex" icon={ArrowLeftRight} accent="var(--cat-3)">
      <div className="flex flex-col items-center gap-4 py-10 text-center">
        <div
          className="flex h-12 w-12 items-center justify-center rounded-full"
          style={{ backgroundColor: "rgba(124, 58, 237, 0.12)" }}
        >
          <ArrowLeftRight size={22} strokeWidth={2} style={{ color: "var(--cat-3)" }} />
        </div>
        <div>
          <div className="text-sm font-semibold text-ink-primary">Not connected yet</div>
          <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-ink-muted">
            Forex runs as its own book with its own paper wallet and risk limits, never
            mixed with NSE capital. It reuses the same agent pipeline and strategies once
            the data layer is wired up.
          </p>
        </div>
        <ul className="mx-auto mt-2 max-w-md space-y-1.5 text-left">
          {PLAN_STEPS.map((step, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-ink-secondary">
              <span className="mt-1.5 h-1 w-1 flex-none rounded-full bg-ink-muted" />
              {step}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}
