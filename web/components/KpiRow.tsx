import { Activity, TrendingUp, Wallet, Percent } from "lucide-react";
import type { HistoryPoint } from "@/lib/useTradingState";
import type { TradingStats } from "@/lib/types";
import { StatTile } from "./ui/StatTile";

const REGIME_LABEL: Record<string, string> = {
  trending_up: "Bull",
  trending_down: "Bear",
  ranging: "Range",
  volatile: "Volatile",
  unknown: "Unknown",
};

const inr = (n: number) =>
  n.toLocaleString("en-IN", { maximumFractionDigits: 0 });

export function KpiRow({ stats, history }: { stats: TradingStats; history: HistoryPoint[] }) {
  const pnlPositive = stats.total_pnl >= 0;

  return (
    <div className="col-span-full grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatTile
        label="Balance"
        icon={Wallet}
        value={`Rs. ${inr(stats.current_balance)}`}
        trend={history.map((h) => h.balance)}
        trendColor="var(--cat-1)"
      />
      <StatTile
        label="Total P&L"
        icon={TrendingUp}
        value={`${pnlPositive ? "+" : ""}Rs. ${inr(stats.total_pnl)}`}
        delta={`${pnlPositive ? "+" : ""}${stats.pnl_percent.toFixed(2)}%`}
        deltaGood={pnlPositive}
        trend={history.map((h) => h.totalPnl)}
        trendColor={pnlPositive ? "var(--status-good)" : "var(--status-critical)"}
      />
      <StatTile
        label="Win Rate"
        icon={Percent}
        value={`${stats.win_rate.toFixed(1)}%`}
        delta={`${stats.winning_trades}W / ${stats.losing_trades}L`}
        deltaGood={stats.win_rate >= 50}
      />
      <StatTile
        label="Market Regime"
        icon={Activity}
        value={REGIME_LABEL[stats.current_regime] ?? "Unknown"}
        delta={`${(stats.regime_confidence * 100).toFixed(0)}% confidence`}
        deltaGood={stats.regime_confidence >= 0.5}
      />
    </div>
  );
}
