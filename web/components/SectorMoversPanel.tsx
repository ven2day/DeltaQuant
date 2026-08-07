import { PieChart, TrendingDown, TrendingUp } from "lucide-react";
import type { SectorMover, TradingStats } from "@/lib/types";
import { Card } from "./ui/Card";

// A real <table> per gainers/losers list so the symbol/percentage columns align
// regardless of how long individual symbol names are — a flex row per entry can't do
// that since each row sizes independently of its siblings. Gainers and losers are
// stacked (not side by side) so each list gets the card's full width — halving it
// wasn't enough room for longer names ("SHRIRAMFIN", "BAJFINANCE") without wrapping
// or truncating them.
function MoverTable({ movers, positive }: { movers: SectorMover[]; positive: boolean }) {
  if (movers.length === 0) {
    return <div className="text-xs italic text-ink-muted">—</div>;
  }
  const color = positive ? "text-status-good" : "text-status-critical";
  return (
    <table className="w-full">
      <tbody>
        {movers.map((m) => (
          <tr key={m.symbol}>
            <td className="whitespace-nowrap py-0.5 pr-3 text-xs text-ink-primary">
              {m.symbol}
            </td>
            <td
              className={`tabular w-full whitespace-nowrap py-0.5 text-right text-xs font-medium ${color}`}
            >
              {positive ? "+" : ""}
              {m.change_percent.toFixed(2)}%
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SectorBlock({
  sector,
  gainers,
  losers,
}: {
  sector: string;
  gainers: SectorMover[];
  losers: SectorMover[];
}) {
  return (
    <div className="mb-3 break-inside-avoid rounded-lg border border-border p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
        {sector}
      </div>
      <div className="space-y-3">
        <div>
          <div className="mb-1 flex items-center gap-1 text-[10px] uppercase text-ink-muted">
            <TrendingUp size={11} className="text-status-good" />
            Gainers
          </div>
          <MoverTable movers={gainers} positive />
        </div>
        <div>
          <div className="mb-1 flex items-center gap-1 text-[10px] uppercase text-ink-muted">
            <TrendingDown size={11} className="text-status-critical" />
            Losers
          </div>
          <MoverTable movers={losers} positive={false} />
        </div>
      </div>
    </div>
  );
}

export function SectorMoversPanel({
  stats,
  expanded = false,
}: {
  stats: TradingStats;
  /** True when this panel is the sole content of its own tab — drop the
   * inner scroll cap and let it use all the room the page gives it. */
  expanded?: boolean;
}) {
  const sectors = Object.entries(stats.sector_movers ?? {});
  const scanStatus = stats.sector_movers_status ?? "pending";
  const dataSource = stats.sector_movers_data_source || stats.data_source;

  const emptyMessage =
    scanStatus === "ready"
      ? `Latest ${dataSource} sector scan completed; no non-flat movers were found.`
      : scanStatus === "error"
        ? `Sector scanner is unavailable in ${dataSource} mode. See Activity Log for details.`
        : "Waiting for the first sector scan (this runs every few minutes, separate from the trading loop)…";

  return (
    <Card title="Sector Movers" icon={PieChart} accent="var(--cat-4)">
      {sectors.length === 0 ? (
        <div className="italic text-ink-muted">{emptyMessage}</div>
      ) : (
        <div
          className={`columns-1 gap-3 sm:columns-2 lg:columns-3 xl:columns-4 ${
            expanded ? "" : "max-h-[36rem] overflow-y-auto"
          }`}
          style={{ columnFill: "balance" }}
        >
          {sectors.map(([sector, { gainers, losers }]) => (
            <SectorBlock key={sector} sector={sector} gainers={gainers} losers={losers} />
          ))}
        </div>
      )}
    </Card>
  );
}
