"use client";

import {
  CandlestickChart,
  Gauge,
  History,
  LayoutDashboard,
  BrainCircuit,
  PieChart,
  ShieldCheck,
  TrendingUp,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { AccountPanel } from "@/components/AccountPanel";
import { ActivityLogPanel } from "@/components/ActivityLogPanel";
import { CycleLifecyclePanel } from "@/components/CycleLifecyclePanel";
import { KpiRow } from "@/components/KpiRow";
import { ModelLearningStatus } from "@/components/ModelLearningStatus";
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
import type { TradingStats } from "@/lib/types";
import type { HistoryPoint } from "@/lib/useTradingState";

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "charts", label: "Charts", icon: CandlestickChart },
  { id: "sectors", label: "Sector Movers", icon: PieChart },
  { id: "scalping", label: "Scalping Candidates", icon: Zap },
  { id: "scalp-decisions", label: "All Scalp Signals", icon: Gauge },
  { id: "signals", label: "Signal History", icon: History },
  { id: "trades", label: "Trade History", icon: TrendingUp },
  { id: "status", label: "System Status", icon: ShieldCheck },
  { id: "models", label: "Model / Learning", icon: BrainCircuit },
];

export function NseWorkspace({ stats, history }: { stats: TradingStats; history: HistoryPoint[] }) {
  const [activeTab, setActiveTab] = useState("overview");
  return (
    <div className="min-w-0 space-y-4">
      <nav aria-label="NSE workspace sections" className="w-full">
        <Sidebar
          tabs={TABS}
          active={activeTab}
          onChange={setActiveTab}
          orientation="horizontal"
        />
      </nav>
      <div className="min-w-0 space-y-4">
        <CycleLifecyclePanel stats={stats} />
        <KpiRow stats={stats} history={history} />
        {activeTab === "overview" && (
          <div className="space-y-4">
            <PipelinePanel stats={stats} />
            <div className="columns-1 gap-4 md:columns-2 xl:columns-3 [&>*]:mb-4 [&>*]:break-inside-avoid">
              <ActivityLogPanel stats={stats} />
              <OpenPositionsPanel stats={stats} />
              <SessionCostPanel stats={stats} />
              <RegimePanel stats={stats} />
              <AccountPanel stats={stats} />
            </div>
          </div>
        )}
        {activeTab === "charts" && <TradeChartsPanel stats={stats} />}
        {activeTab === "sectors" && <SectorMoversPanel stats={stats} expanded />}
        {activeTab === "scalping" && <ScalpingCandidatesPanel stats={stats} />}
        {activeTab === "scalp-decisions" && <ScalpDecisionTable stats={stats} />}
        {activeTab === "signals" && <SignalHistoryPanel />}
        {activeTab === "trades" && <TradeHistoryPanel />}
        {activeTab === "status" && <SystemStatusPanel />}
        {activeTab === "models" && <ModelLearningStatus market="NSE" />}
      </div>
    </div>
  );
}
