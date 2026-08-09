"use client";

import {
  CandlestickChart,
  Gauge,
  History,
  LayoutDashboard,
  PieChart,
  ShieldCheck,
  TrendingUp,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { AccountPanel } from "@/components/AccountPanel";
import { ActivityLogPanel } from "@/components/ActivityLogPanel";
import { CycleLifecyclePanel } from "@/components/CycleLifecyclePanel";
import { ForexPlaceholder } from "@/components/ForexPlaceholder";
import { Header } from "@/components/Header";
import { KpiRow } from "@/components/KpiRow";
import { OpenPositionsPanel } from "@/components/OpenPositionsPanel";
import { PipelinePanel } from "@/components/PipelinePanel";
import { RegimePanel } from "@/components/RegimePanel";
import { ScalpDecisionTable } from "@/components/ScalpDecisionTable";
import { ScalpingCandidatesPanel } from "@/components/ScalpingCandidatesPanel";
import { SectorMoversPanel } from "@/components/SectorMoversPanel";
import { SessionCostPanel } from "@/components/SessionCostPanel";
import { SignalHistoryPanel } from "@/components/SignalHistoryPanel";
import { SystemStatusPanel } from "@/components/SystemStatusPanel";
import { TradeChartsPanel } from "@/components/TradeChartsPanel";
import { TradeHistoryPanel } from "@/components/TradeHistoryPanel";
import { Sidebar } from "@/components/ui/Sidebar";
import { useAuthGuard } from "@/lib/useAuthGuard";
import { useTradingState } from "@/lib/useTradingState";

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "charts", label: "Charts", icon: CandlestickChart },
  { id: "sectors", label: "Sector Movers", icon: PieChart },
  { id: "scalping", label: "Scalping Candidates", icon: Zap },
  { id: "scalp-decisions", label: "Scalp Decisions", icon: Gauge },
  { id: "signals", label: "Signal History", icon: History },
  { id: "trades", label: "Trade History", icon: TrendingUp },
  { id: "status", label: "System Status", icon: ShieldCheck },
];

export default function Home() {
  const { checking } = useAuthGuard();
  const { state, connected, history } = useTradingState();
  const [activeMarket, setActiveMarket] = useState("nse");
  const [activeTab, setActiveTab] = useState("overview");

  if (checking) {
    return (
      <main className="mx-auto flex max-w-7xl flex-col gap-4 p-4 sm:p-6">
        <div className="py-24 text-center italic text-ink-muted">Checking session…</div>
      </main>
    );
  }

  const showSectionNav = activeMarket === "nse" && Boolean(state);

  return (
    <main className="mx-auto flex max-w-[96rem] flex-col gap-4 p-4 sm:p-6">
      <Header
        stats={state}
        connected={connected}
        activeMarket={activeMarket}
        onMarketChange={setActiveMarket}
      />

      <div className="flex items-start gap-4">
        {showSectionNav && (
          <div className="sticky top-4 w-48 flex-none">
            <Sidebar tabs={TABS} active={activeTab} onChange={setActiveTab} />
          </div>
        )}

        <div className="min-w-0 flex-1 space-y-4">
          {activeMarket === "forex" ? (
            <ForexPlaceholder />
          ) : !state ? (
            <div className="py-24 text-center italic text-ink-muted">
              Waiting for the backend at{" "}
              {process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws"}…
            </div>
          ) : (
            <>
              <CycleLifecyclePanel stats={state} />
              <KpiRow stats={state} history={history} />

              {activeTab === "overview" && (
                <div className="space-y-4">
                  {/* PipelinePanel now includes what used to be the separate
                      AgentActivityPanel (news/mood/prediction/regime/strategy detail) --
                      they showed overlapping cycle/validation/risk counters, and
                      AgentActivityPanel had grown tall enough to throw off the masonry
                      grid below's height-balancing among shorter cards. */}
                  <PipelinePanel stats={state} />
                  <div
                    className="columns-1 gap-4 md:columns-2 xl:columns-3 [&>*]:mb-4 [&>*]:break-inside-avoid"
                    style={{ columnFill: "balance" }}
                  >
                    <ActivityLogPanel stats={state} />
                    <OpenPositionsPanel stats={state} />
                    <SessionCostPanel stats={state} />
                    <RegimePanel stats={state} />
                    <AccountPanel stats={state} />
                  </div>
                </div>
              )}

              {activeTab === "charts" && <TradeChartsPanel stats={state} />}

              {activeTab === "sectors" && <SectorMoversPanel stats={state} expanded />}

              {activeTab === "scalping" && <ScalpingCandidatesPanel stats={state} />}

              {activeTab === "scalp-decisions" && <ScalpDecisionTable stats={state} />}

              {activeTab === "signals" && <SignalHistoryPanel />}

              {activeTab === "trades" && <TradeHistoryPanel />}

              {activeTab === "status" && <SystemStatusPanel />}
            </>
          )}
        </div>
      </div>
    </main>
  );
}
