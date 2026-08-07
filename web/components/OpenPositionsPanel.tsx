import { ListOrdered } from "lucide-react";
import type { TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

export function OpenPositionsPanel({ stats }: { stats: TradingStats }) {
  const positions = stats.open_positions ?? [];

  return (
    <Card title="Open Positions" icon={ListOrdered} accent="var(--ink-muted)">
      {positions.length === 0 ? (
        <div className="italic text-ink-muted">No open positions</div>
      ) : (
        <table className="w-full">
          <thead>
            <tr className="text-left text-xs text-ink-muted">
              <th className="pb-2 font-medium">Symbol</th>
              <th className="pb-2 font-medium">Side</th>
              <th className="pb-2 text-right font-medium">Qty</th>
              <th className="pb-2 text-right font-medium">Entry</th>
              <th className="pb-2 text-right font-medium">P&amp;L</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {positions.slice(0, 5).map((pos, i) => (
              <tr key={`${pos.symbol}-${i}`}>
                <td className="py-1.5 font-medium text-ink-primary">{pos.symbol}</td>
                <td className="py-1.5">
                  <Badge label={pos.side} tone={pos.side === "BUY" ? "good" : "critical"} />
                </td>
                <td className="tabular py-1.5 text-right">{pos.qty}</td>
                <td className="tabular py-1.5 text-right">Rs.{pos.entry.toFixed(2)}</td>
                <td
                  className={`tabular py-1.5 text-right font-medium ${
                    pos.pnl >= 0 ? "text-status-good" : "text-status-critical"
                  }`}
                >
                  {pos.pnl >= 0 ? "+" : ""}Rs.{pos.pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
