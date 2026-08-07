"use client";

import { History, RefreshCw } from "lucide-react";
import { useTradeHistory } from "@/lib/useTradeHistory";
import { formatLocalDateTime } from "@/lib/formatTime";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

function formatCurrency(value: number): string {
  return `Rs. ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

const EXIT_REASON_LABELS: Record<string, string> = {
  stop_loss: "Stop Loss",
  target_hit: "Target Hit",
  trailing_stop: "Trailing Stop",
  time_exit: "Time Exit",
  stale_exit: "Stale Exit",
  regime_change: "Regime Change",
  partial: "Partial",
};

function ExitReasonBadge({ reason }: { reason: string }) {
  if (!reason) return <span className="text-ink-muted">—</span>;
  const tone = reason === "target_hit" ? "good" : reason === "stop_loss" ? "critical" : "neutral";
  return <Badge label={EXIT_REASON_LABELS[reason] ?? reason} tone={tone} />;
}

export function TradeHistoryPanel() {
  const { trades, loading, error, refresh } = useTradeHistory();

  return (
    <Card
      title="Paper Trade History"
      icon={History}
      accent="var(--cat-3)"
      right={
        <button
          type="button"
          onClick={refresh}
          className="flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-ink-primary"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      }
    >
      {error ? (
        <div className="italic text-status-critical">Failed to load trade history: {error}</div>
      ) : loading && trades.length === 0 ? (
        <div className="italic text-ink-muted">Loading...</div>
      ) : trades.length === 0 ? (
        <div className="italic text-ink-muted">No closed paper trades yet.</div>
      ) : (
        <div className="max-h-[calc(100vh-18rem)] overflow-auto">
          <table className="w-full border-collapse text-xs">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-border text-left uppercase tracking-wide text-ink-muted">
                <th className="py-2 pr-3 font-medium">Closed</th>
                <th className="py-2 pr-3 font-medium">Symbol</th>
                <th className="py-2 pr-3 font-medium">Side</th>
                <th className="py-2 pr-3 text-right font-medium">Qty</th>
                <th className="py-2 pr-3 text-right font-medium">Entry</th>
                <th className="py-2 pr-3 text-right font-medium">Exit</th>
                <th className="py-2 pr-3 font-medium">Exit Reason</th>
                <th className="py-2 text-right font-medium">Net P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => {
                const profitable = trade.net_pnl > 0;
                return (
                  <tr key={trade.close_order_id} className="border-b border-border/50">
                    <td className="whitespace-nowrap py-2 pr-3 text-ink-muted">
                      {formatLocalDateTime(trade.timestamp)}
                    </td>
                    <td className="py-2 pr-3 font-medium text-ink-primary">{trade.symbol}</td>
                    <td className="py-2 pr-3">
                      <Badge label={trade.side} tone={trade.side === "BUY" ? "good" : "critical"} />
                    </td>
                    <td className="tabular py-2 pr-3 text-right">{trade.quantity}</td>
                    <td className="tabular py-2 pr-3 text-right">{trade.entry_price.toFixed(2)}</td>
                    <td className="tabular py-2 pr-3 text-right">{trade.exit_price.toFixed(2)}</td>
                    <td className="py-2 pr-3">
                      <ExitReasonBadge reason={trade.exit_reason} />
                    </td>
                    <td className={`tabular py-2 text-right font-medium ${profitable ? "text-status-good" : "text-status-critical"}`}>
                      {formatCurrency(trade.net_pnl)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
