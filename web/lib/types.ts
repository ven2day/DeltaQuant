// Mirrors `stats_to_dict()` (src/webui/schema.py) which serializes
// `TradingStats` (src/dashboard/stats.py) 1:1, plus its three computed properties.

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
// descriptive only, never an admission decision -- strategy eligibility and risk still
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

// src/market/scalp_scan.py ScalpSymbolStatus.to_dict(). This is the uncapped,
// full-universe technical scan. BUY is a setup indication, not order approval.
export interface ScalpSymbolStatus {
  symbol: string;
  decision: "BUY" | "NO_BUY" | "NO_DATA";
  stage: string;
  reasons: string[];
  available_timeframes: string[];
  missing_timeframes: string[];
  raw_buy_signals: number;
  timeframe_states: Record<string, TimeframeAssessment>;
  primary_strategy: string;
  primary_timeframe: string;
  entry_quality: EntryQualityResult | null;
  mtf_confirmation: ScalpConfirmationResult | null;
  regime_compatible: boolean | null;
  score: number;
  selected_for_review: boolean;
  entry_price: number;
  stop_loss: number;
  target_price: number;
  expected_r: number;
  ml_probability: number | null;
}

// src/market/scalp_scan.py FUNNEL_KEYS -- req 13's funnel observability counters,
// one full cycle's worth: raw strategy triggers -> ... -> execution accepted.
export interface ScalpFunnel {
  raw_triggers: number;
  consolidated: number;
  mtf_candidates: number;
  mtf_confirmed: number;
  mtf_conflict: number;
  mtf_missing_data: number;
  entry_quality_passed: number;
  ml_scored: number;
  regime_compatible: number;
  registry_eligible: number;
  confidence_reduced: number;
  sent_to_ai: number;
  ai_approved: number;
  execution_accepted: number;
}

export interface SignalFunnel {
  cycle_id: number;
  trade_horizon: string;
  symbols_requested: number;
  symbols_ready: number;
  symbols_scanned: number;
  strategy_evaluations: number;
  raw_signals: number;
  technical_setups: number;
  deterministic_qualified: number;
  deterministic_rejected: number;
  registry_eligible: number;
  registry_shadow_only: number;
  registry_blocked: number;
  registry_unvalidated: number;
  regime_blocked: number;
  confidence_reduced: number;
  ml_candidates: number;
  ml_inference: number;
  ml_abstain: number;
  ml_pass: number;
  qwen_required: number;
  qwen_skipped_clear: number;
  qwen_cache_hits: number;
  qwen_external_calls: number;
  qwen_deferred: number;
  qwen_rejected: number;
  signal_validation_pass: number;
  signal_validation_reject: number;
  risk_approved: number;
  risk_blocked: number;
  final_buy: number;
  final_hold: number;
  final_wait: number;
  final_reject: number;
  reconciled: boolean;
  reconciliation_errors: string[];
}

export interface RejectionBreakdown {
  stage: string;
  reason: string;
  count: number;
  percentage: number;
  strategy?: string;
  timeframe?: string;
  symbol?: string;
}

export interface StrategyFunnelRow {
  strategy: string;
  timeframe: string;
  trade_horizon: string;
  raw?: number;
  technical_setups?: number;
  qualified?: number;
  registry_eligible?: number;
  shadow_only?: number;
  ml_evaluated?: number;
  ml_abstained?: number;
  qwen_required?: number;
  validation_passed?: number;
  validation_rejected?: number;
  risk_approved?: number;
  risk_rejected?: number;
  final_orders?: number;
  rejected?: number;
}

export interface DataLineage {
  execution_mode?: string;
  runtime_mode?: string;
  market_data_source?: string;
  quote_source?: string;
  candle_source?: string;
  is_simulated?: boolean;
  exchange_timestamp?: string;
  symbols_in_current_quote_batch?: number;
  simulated_data_executable?: boolean;
  broker_orders_enabled?: boolean;
}

export interface QwenCallAuditRow {
  agent: string;
  purpose: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_hits: number;
  candidate_related: boolean;
  symbols: string[];
  cycle_id: string;
}

export interface StrategyEligibilityRegistryRow {
  strategy: string;
  timeframe: string;
  model_version: string;
  validation_status: string;
  validated_at: string;
  allowed_regimes: string[];
  disabled_regimes: string[];
  minimum_model_confidence: number;
  minimum_strategy_confidence: number;
  oos_trade_count: number;
  oos_profit_factor: number | null;
  oos_max_drawdown: number | null;
  oos_win_rate: number | null;
  status_reason: string;
}

export interface FinOpsBucket {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

export interface FinOpsSummary {
  date: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  candidate_review_calls: number;
  context_support_calls: number;
  by_market: Record<string, FinOpsBucket>;
  by_reason: Record<string, FinOpsBucket>;
  by_component: Record<string, FinOpsBucket>;
  by_model: Record<string, FinOpsBucket>;
}

export interface TradingStats {
  session_start: string;
  trading_mode: string;
  data_source: string;
  execution_mode: string;
  quote_source: string;
  candle_source: string;
  broker_orders_enabled: boolean;
  data_lineage: DataLineage;
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
  ai_review_status: string;
  ai_review_reason: string;

  llm_calls: number;
  llm_tokens: number;
  llm_cost_usd: number;
  llm_finops: FinOpsSummary | Record<string, never>;
  llm_session_metrics: FinOpsSummary | Record<string, never>;
  llm_cycle_metrics: FinOpsSummary | Record<string, never>;

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
  scalp_symbol_statuses: ScalpSymbolStatus[];
  scalp_scan_completed_at: string;
  scalp_funnel: ScalpFunnel | Record<string, never>;
  scalp_rejection_reasons: RejectionBreakdown[];
  signal_funnel: SignalFunnel | Record<string, never>;
  rejection_reasons: RejectionBreakdown[];
  rejection_drilldown: RejectionBreakdown[];
  strategy_funnel: StrategyFunnelRow[];
  strategy_eligibility_registry: StrategyEligibilityRegistryRow[];
  qwen_call_audit: QwenCallAuditRow[];
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
  markets?: Record<
    string,
    {
      provider: string;
      environment?: string;
      healthy?: boolean;
      market_data: string;
      pricing_stream?: string;
      execution: string;
      kill_switch?: boolean;
      message?: string;
      latest_price_age_seconds?: number | null;
    }
  >;
  checked_at: string;
}

export interface LocalServiceEntry {
  name: string;
  required: boolean;
  running: boolean;
  detail: string;
}

export interface BackgroundJobStatus {
  enabled: boolean;
  last_run_at: string | null;
  hours_since_run: number | null;
  refresh_interval_hours: number;
  stale: boolean;
  state: "idle" | "running" | "never_run";
  last_run_failed: boolean;
  intervals?: Record<string, string>;
}

export interface SystemJobsStatus {
  local_services: LocalServiceEntry[];
  background_jobs: {
    signal_discovery: BackgroundJobStatus;
    strategy_validation: BackgroundJobStatus;
    scalp_training: BackgroundJobStatus;
  };
}
