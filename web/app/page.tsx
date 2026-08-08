"use client";

import {
  CandlestickChart,
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
import { AgentActivityPanel } from "@/components/AgentActivityPanel";
import { AIDecisionPanel } from "@/components/AIDecisionPanel";
import { Header } from "@/components/Header";
import { KpiRow } from "@/components/KpiRow";
import { MarketChartsPanel } from "@/components/MarketChartsPanel";
import { OpenPositionsPanel } from "@/components/OpenPositionsPanel";
import { RegimePanel } from "@/components/RegimePanel";
import { ScalpingCandidatesPanel } from "@/components/ScalpingCandidatesPanel";
import { SectorMoversPanel } from "@/components/SectorMoversPanel";
import { SessionCostPanel } from "@/components/SessionCostPanel";
import { SignalHistoryPanel } from "@/components/SignalHistoryPanel";
import { SystemStatusPanel } from "@/components/SystemStatusPanel";
import { TradeChartsPanel } from "@/components/TradeChartsPanel";
import { TradeHistoryPanel } from "@/components/TradeHistoryPanel";
import { Tabs } from "@/components/ui/Tabs";
import { useAuthGuard } from "@/lib/useAuthGuard";
import { useTradingState } from "@/lib/useTradingState";

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "charts", label: "Charts", icon: CandlestickChart },
  { id: "sectors", label: "Sector Movers", icon: PieChart },
  { id: "scalping", label: "Scalping Candidates", icon: Zap },
  { id: "signals", label: "Signal History", icon: History },
  { id: "trades", label: "Trade History", icon: TrendingUp },
  { id: "status", label: "System Status", icon: ShieldCheck },
];

export default function Home() {
  const { checking } = useAuthGuard();
  const { state, connected, history } = useTradingState();
  const [activeTab, setActiveTab] = useState("overview");

  if (checking) {
    return (
      <main className="mx-auto flex max-w-7xl flex-col gap-4 p-4 sm:p-6">
        <div className="py-24 text-center italic text-ink-muted">Checking session…</div>
      </main>
    );
  }

  return (
    <main className="mx-auto flex max-w-7xl flex-col gap-4 p-4 sm:p-6">
      <Header stats={state} connected={connected} />

      {!state ? (
        <div className="py-24 text-center italic text-ink-muted">
          Waiting for the backend at {process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws"}…
        </div>
      ) : (
        <>
          <KpiRow stats={state} history={history} />

          <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

          {activeTab === "overview" && (
            <div className="space-y-4">
              <MarketChartsPanel stats={state} />
              <div
                className="columns-1 gap-4 md:columns-2 lg:columns-3 [&>*]:mb-4 [&>*]:break-inside-avoid"
                style={{ columnFill: "balance" }}
              >
                <ActivityLogPanel stats={state} />
                <OpenPositionsPanel stats={state} />
                <SessionCostPanel stats={state} />
                <RegimePanel stats={state} />
                <AgentActivityPanel stats={state} />
                <AccountPanel stats={state} />
                <AIDecisionPanel stats={state} />
              </div>
            </div>
          )}

          {activeTab === "charts" && <TradeChartsPanel stats={state} />}

          {activeTab === "sectors" && <SectorMoversPanel stats={state} expanded />}

          {activeTab === "scalping" && <ScalpingCandidatesPanel stats={state} />}

          {activeTab === "signals" && <SignalHistoryPanel />}

          {activeTab === "trades" && <TradeHistoryPanel />}

          {activeTab === "status" && <SystemStatusPanel />}
        </>
      )}
    </main>
  );
}
