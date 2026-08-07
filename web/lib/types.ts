// Mirrors `stats_to_dict()` (src/webui/schema.py) which serializes
// `TradingStats` (src/dashboard/cli.py) 1:1, plus its three computed properties.

export interface Position {
  symbol: string;
  side: string;
  qty: number;
  entry: number;
  pnl: number;
  position_id?: string;
  current?: number;
  target?: number;
  stop?: number;
  status?: string;
}

export interface Quote {
  symbol: string;
  last_price: number;
  open: number;
  high: number;
  low: number;
  close: number;
  change: number;
  change_percent: number;
  volume: number;
  is_live: boolean;
}

export interface CurrentSignal {
  signal_type?: string;
  symbol?: string;
  strategy?: string;
  confidence?: number;
  timeframe?: string;
  action?: "BUY" | "WAIT" | "REJECT" | string;
  rationale?: string[];
  target_realistic?: boolean;
}

export interface CandidateDecision {
  symbol: string;
  action: "BUY" | "WAIT" | "REJECT";
  rationale: string[];
  entry_price: number;
  stop_loss: number;
  target_price: number;
  target_pct: number;
  target_realistic: boolean;
  volume_ratio: number;
  vwap: number;
  higher_timeframes_aligned: number;
  ml_direction: string;
  ml_confidence: number;
}

export interface ChartPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartSeries {
  points: ChartPoint[];
  entry?: number;
  current?: number;
  target?: number;
  stop?: number;
  status?: string;
}

export interface ActivityEntry {
  time: string;
  level: "INFO" | "SUCCESS" | "WARNING" | "ERROR" | "TRADE" | string;
  message: string;
}

export interface SectorMover {
  symbol: string;
  last_price: number;
  change_percent: number;
}

export interface SectorMovers {
  gainers: SectorMover[];
  losers: SectorMover[];
}

export interface TimeframeSwingStats {
  avg_swings_per_day: number;
  avg_swing_size: number;
  days_analyzed: number;
}

export interface ScalpingCandidate {
  symbol: string;
  last_price: number;
  avg_daily_range: number;
  // Keyed by timeframe value: "15m" | "30m" | "1h" | "4h"
  timeframes: Record<string, TimeframeSwingStats>;
}

export interface TradingStats {
  session_start: string;
  trading_mode: string;
  data_source: string;
  // Backend-computed: the same is_trading_window()/force_trading_window gate the
  // cycle loop itself uses for new entries — not a client-guessed clock.
  market_open: boolean;
  force_trading_window: boolean;

  starting_balance: number;
  current_balance: number;

  total_trades: number;
  winning_trades: number;
  losing_trades: number;

  realized_pnl: number;
  unrealized_pnl: number;
  best_trade: number;
  worst_trade: number;

  open_positions: Position[];

  cycles_run: number;
  signals_generated: number;
  signals_validated: number;
  signals_rejected: number;
  trades_approved: number;
  trades_risk_rejected: number;

  current_regime: string;
  regime_confidence: number;
  active_strategies: string[];

  llm_calls: number;
  llm_tokens: number;
  llm_cost_usd: number;

  goal_enabled: boolean;
  goal_feasible: boolean;
  goal_target_amount: number;
  goal_mtd_pnl: number;
  goal_expected_to_date: number;
  goal_on_pace: boolean;
  goal_status: string;

  market_quotes: Record<string, Quote>;
  top_movers: Quote[];
  sector_movers: Record<string, SectorMovers>;
  sector_movers_status: "disabled" | "pending" | "ready" | "error" | string;
  sector_movers_data_source: string;
  scalping_candidates: ScalpingCandidate[];
  scalping_screener_status: "disabled" | "pending" | "ready" | "error" | string;
  scalping_screener_data_source: string;

  last_decision_reason: string;
  current_signal: CurrentSignal;
  candidate_decisions: CandidateDecision[];
  chart_symbol: string;
  chart_timeframes: Record<string, ChartSeries>;
  simulation_event_time: string;
  daily_entry_cap: number;
  daily_entries: number;

  activity_log: ActivityEntry[];

  // Computed properties (not dataclass fields — added by stats_to_dict())
  win_rate: number;
  total_pnl: number;
  pnl_percent: number;
}

export interface StateMessage {
  type: "state";
  data: TradingStats;
}

// Shape of one entry from GET /api/signals (src/execution/signal_log.py SignalRecord),
// independent of the TradingStats websocket snapshot.
export interface SignalRecord {
  timestamp: string;
  symbol: string;
  side: string;
  entry_price: number;
  timeframe: string;
  strategy: string;
  confidence: number;
  status: "approved" | "rejected_validation" | "rejected_risk";
  reason: string;
  source?: "live" | "backfill" | string;
}

export interface ClosedPaperTrade {
  close_order_id: string;
  timestamp: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entry_price: number;
  entry_time: string;
  exit_price: number;
  // "" for trades closed before this field existed, or an entry-side leg.
  exit_reason: string;
  gross_pnl: number;
  entry_charges: number;
  exit_charges: number;
  net_pnl: number;
}

// Shape of GET /api/health (src/api/health.py SystemHealth.to_dict() / ServiceHealth.to_dict()).
export interface ServiceHealthEntry {
  name: string;
  status: "healthy" | "degraded" | "unhealthy";
  latency_ms: number;
  message: string;
  details: Record<string, unknown>;
  checked_at: string;
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  uptime_seconds: number;
  services: ServiceHealthEntry[];
  checked_at: string;
}
