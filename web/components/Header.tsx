"use client";

import { LogOut, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { logout } from "@/lib/api";
import type { TradingStats } from "@/lib/types";
import { Badge } from "./ui/Badge";

// Just a ticking IST clock for display — whether new entries are actually allowed
// comes from the backend (stats.market_open), which is the same is_trading_window()/
// force_trading_window check the trading loop itself gates on. Guessing that
// independently here risked disagreeing with what the system was really doing (e.g.
// showing "NSE CLOSED" while force_trading_window had the loop actively trading).
function useISTClock(): string {
  const [clock, setClock] = useState("--:--:-- IST");
  useEffect(() => {
    const tick = () => {
      const ist = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
      setClock(`${ist.toTimeString().slice(0, 8)} IST`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return clock;
}

export function Header({
  stats,
  connected,
}: {
  stats: TradingStats | null;
  connected: boolean;
}) {
  const clock = useISTClock();
  const router = useRouter();

  async function handleLogout() {
    await logout();
    router.replace("/login");
  }

  const mode = stats?.trading_mode ?? "paper";
  const dataSource = stats?.data_source ?? "simulated";
  const marketOpen = stats?.market_open ?? false;
  const forced = stats?.force_trading_window ?? false;

  return (
    <div className="col-span-full flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-2.5">
        <ShieldCheck size={22} className="text-cat-1" strokeWidth={2} />
        <div>
          <div className="text-base font-semibold leading-tight text-ink-primary">
            ₹DeltaQuant
          </div>
          <div className="text-xs text-ink-muted">Agentic NSE Trading</div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge label={mode.toUpperCase()} tone={mode === "paper" ? "good" : "critical"} />
        <Badge label={dataSource.toUpperCase()} tone="neutral" dot />
        <Badge
          label={forced ? "TRADING (forced, test mode)" : marketOpen ? "NSE OPEN" : "NSE CLOSED"}
          tone={forced ? "critical" : marketOpen ? "good" : "neutral"}
          dot
        />
        <Badge
          label={connected ? "Connected" : "Disconnected"}
          tone={connected ? "good" : "critical"}
          dot
          pulse={connected}
        />
        <span className="tabular text-xs text-ink-muted">{clock}</span>
        <button
          type="button"
          onClick={handleLogout}
          title="Sign out"
          className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-ink-muted transition-colors hover:border-status-critical hover:text-status-critical"
        >
          <LogOut size={12} strokeWidth={2} />
          Sign out
        </button>
      </div>
    </div>
  );
}
