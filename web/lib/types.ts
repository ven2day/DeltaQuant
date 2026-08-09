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
  entry_time?: string;
  strategy?: string;
  timeframe?: string;
  /** "simulated" if opened while NSE was closed (weekend/off-hours pipeline test
   * data), "real" if opened on genuine DhanHQ quotes. Sticks to how the position
   * was first opened even after the live pipeline switches back to real data. */
  entry_data_source?: string;
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

// News Analyst agent output: one Google-RSS headline scored -1..+1 by Groq, collapsed
// to a coarse label (src/agents/graph.py's _news_analyst_node wrapper).
export interface NewsHeadline {
  title: string;
  sentiment: "positive" | "negative" | string;
}

// Sentiment Agent output (SentimentSignal.to_dict(), src/agents/sentiment.py) -- a
// deterministic blend of news score, volatility, and market breadth, not an LLM call.
export interface MarketMood {
  mood_index: number; // 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
  mood_label: "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed" | string;
  news_score: number; // -1..+1
  volatility_score: number; // 0-100, higher = less volatile
  breadth_score: number; // -1..+1, positive = more advancers than decliners
  confidence: number;
  reasoning: string;
  timestamp: string;
}

// Prediction Agent output (PredictionSignal.to_dict(), src/agents/prediction.py) -- a
// walk-forward-validated, Platt-calibrated ensemble of 3 regressors (Linear/RandomForest/
// GradientBoosting), not a single model. Can abstain rather than guess.
export interface PredictionSignal {
  symbol: string;
  direction: "up" | "down" | "flat";
  confidence: number;
  predicted_change_pct: number;
  reasoning: string;
  timestamp: string;
  abstained: boolean;
  feature_version: string;
  model_version: string;
  oos_samples: number;
  calibration_by_regime: Record<string, number>;
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

// One symbol/timeframe cell of the scalp assessment matrix
// (src/market/assessment_matrix.py TimeframeAssessment.to_dict()). `decision` is
// descriptive only, never an admission decision -- H-8 and risk_compliance still
// gate every trade independently regardless of what a cell reads here.
export interface TimeframeAssessment {
  timeframe: string;
  decision: "BUY" | "WAIT" | "REJECT";
  score: number;
  strategy_consensus: number;
  ml_probability: number | null;
  regime_compatible: boolean;
  reasons: string[];
}

// src/market/entry_quality.py EntryQualityResult.to_dict()
export interface EntryQualityResult {
  status: "ENTER_NOW" | "WAIT_PULLBACK" | "WAIT_BREAKOUT" | "REJECT";
  preferred_entry_low: number;
  preferred_entry_high: number;
  vwap_distance_pct: number;
  ema9_distance_pct: number;
  atr_extension: number;
  nearest_swing_support: number | null;
  nearest_swing_resistance: number | null;
  breakout_state: "none" | "breaking_out" | "retesting";
  relative_volume: number;
  upper_wick_ratio: number;
  lower_wick_ratio: number;
  risk_reward: number;
  reasons: string[];
}

// src/market/scalp_confirmation.py ScalpConfirmationResult.to_dict() -- req 10's
// 5m=execution/15m=primary/30m=directional/1h=context/4h=optional-macro roles.
export interface ScalpConfirmationResult {
  execution_ok: boolean;
  primary_ok: boolean;
  directional_ok: boolean;
  context_ok: boolean;
  macro_ok: boolean | null; // null when the 4h macro filter is disabled, not evaluated
  aligned_count: number;
  required: number;
  passed: boolean;
  reasons: string[];
}

// src/market/scalp_opportunity.py ScalpOpportunity.to_dict() -- the canonical
// scan->rank->agent->UI->execution scalp candidate object (req 11).
export interface ScalpOpportunity {
  symbol: string;
  direction: "BUY" | "SELL";
  // Keyed by timeframe value: "5m" | "15m" | "30m" | "1h" | "4h"
  timeframe_states: Record<string, TimeframeAssessment>;
  primary_strategy: string;
  primary_timeframe: string;
  entry_quality: EntryQualityResult | null;
  mtf_confirmation: ScalpConfirmationResult | null;
  regime_compatible: boolean;
  ml_probability: number | null;
  historical_scalp_expectancy: number | null;
  score: number;
  final_decision: "ENTER_NOW" | "WAIT_PULLBACK" | "WAIT_BREAKOUT" | "REJECT";
  reason: string[];
  entry_price: number;
  preferred_entry_low: number;
  preferred_entry_high: number;
  stop_loss: number;
  target_price: number;
  expected_r: number;
}

// src/market/scalp_scan.py FUNNEL_KEYS -- req 13's funnel observability counters,
// one full cycle's worth: raw strategy triggers -> ... -> execution accepted.
export interface ScalpFunnel {
  raw_triggers: number;
  consolidated: number;
  mtf_candidates: number;
  entry_quality_passed: number;
  regime_compatible: number;
  h8_admitted: number;
  sent_to_ai: number;
  ai_approved: number;
  execution_accepted: number;
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

  // What each agent actually decided on the most recently completed cycle --
  // populated every cycle regardless of whether it produced a trade.
  regime_reasoning: string;
  strategy_reasoning: string;
  news_headlines: NewsHeadline[];
  news_sentiment: number;
  market_mood: MarketMood | Record<string, never>;
  prediction_signals: PredictionSignal[];
  agent_fallback_notice: string;

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
  // Top-ranked scalp opportunities, refreshed every cycle the same way
  // candidate_decisions is -- empty unless the backend's scalp_enabled is on.
  scalp_opportunities: ScalpOpportunity[];
  scalp_funnel: ScalpFunnel | Record<string, never>;
  chart_symbol: string;
  chart_timeframes: Record<string, ChartSeries>;
  simulation_event_time: string;
  daily_entry_cap: number;
  daily_entries: number;

  activity_log: ActivityEntry[];

  // Live cycle lifecycle -- what the in-progress cycle is doing right now, instead of
  // only learning it from raw backend logs.
  current_cycle_number: number;
  cycle_stage: string;
  cycle_stage_label: string;
  cycle_stage_started_at: string;
  /** Set only while cycle_stage === "waiting"; "" the rest of the time. */
  next_cycle_at: string;

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
