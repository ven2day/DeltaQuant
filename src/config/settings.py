"""
Application settings and configuration management.

Uses pydantic-settings for environment variable loading with validation.
Includes cross-field validation to ensure configuration consistency.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.env_loader import resolve_env_files


def resolve_effective_execution_mode(
    requested_mode: str,
    *,
    allow_live_orders: bool,
    trading_mode: str,
    has_dhan_credentials: bool,
) -> str:
    """Pure resolution of the execution mode actually in effect (C-2 fix).

    A single source of truth for "can this configuration ever reach a real broker
    order", reused by ``ExecutionService._resolve_mode`` (the stateful runtime
    wrapper) and ``scripts/check_config.py`` (a static preview with no engine to
    construct). Kept dependency-free (plain strings, no ``ExecutionMode`` import)
    to avoid a settings <-> execution circular import.

    Invariants:
    - ``dhan_paper`` NEVER reaches a live route. There is no verified Dhan sandbox
      endpoint (``dhan_base_url`` defaults to Dhan's live API host), so it is
      redefined to mean "simulate against Dhan-shaped data/mechanics only" — not
      "live route gated by a flag". This is the exact hazard C-2 described:
      TRADING_MODE=paper + EXECUTION_MODE=dhan_paper + ALLOW_LIVE_ORDERS=true with
      valid credentials used to reach a genuine broker order.
    - ``live`` reaches a real broker route only under the full, explicit
      conjunction: ``trading_mode == "live"`` AND ``allow_live_orders`` AND Dhan
      credentials present. Missing any one of those resolves to ``shadow``
      (mirrors the decision/sizing, simulates the fill, sends nothing) rather
      than silently downgrading to local paper.
    """
    if requested_mode == "dhan_paper":
        return "shadow"
    if requested_mode == "live":
        if not (allow_live_orders and trading_mode == "live" and has_dhan_credentials):
            return "shadow"
    return requested_mode


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Cross-field validation warnings collected by validate_configuration().
    # Populated rather than raised so invalid config degrades instead of failing
    # startup; tooling (e.g. check_config.py) can surface these to the operator.
    _config_warnings: list[str] = PrivateAttr(default_factory=list)

    @property
    def config_warnings(self) -> list[str]:
        """Cross-field configuration warnings detected at load time (may be empty)."""
        return self._config_warnings

    # ===========================================
    # Market worker / multi-market isolation
    # ===========================================
    market: Literal["NSE", "FOREX", "CRYPTO"] = Field(default="NSE")
    app_env: str = Field(default="local")
    nse_enabled: bool = Field(default=True)
    forex_enabled: bool = Field(default=False)
    crypto_enabled: bool = Field(default=False)
    global_kill_switch: bool = Field(default=False)
    nse_kill_switch: bool = Field(default=False)
    forex_kill_switch: bool = Field(default=False)
    crypto_kill_switch: bool = Field(default=False)
    runtime_id: str = Field(default="deltaquant")
    market_snapshot_root: str = Field(default="run")
    strategy_config_root: str = Field(default="config")
    ml_model_root: str = Field(default="artifacts")

    nse_broker: Literal["dhan"] = Field(default="dhan")
    nse_timezone: str = Field(default="Asia/Kolkata")
    nse_execution_enabled: bool = Field(default=False)
    nse_db_schema: str = Field(default="nse")
    nse_config_root: str = Field(default="config/nse")
    nse_artifact_root: str = Field(default="artifacts/nse")
    nse_log_root: str = Field(default="logs/nse")

    forex_provider: Literal["oanda"] = Field(default="oanda")
    forex_environment: Literal["practice", "live"] = Field(default="practice")
    oanda_environment: Literal["practice", "live"] = Field(default="practice")
    oanda_account_id: SecretStr = Field(default="")
    oanda_access_token: SecretStr = Field(default="")
    oanda_rest_url: str = Field(default="")
    oanda_stream_url: str = Field(default="")
    oanda_request_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    oanda_max_read_retries: int = Field(default=3, ge=0, le=8)
    oanda_stream_stale_seconds: float = Field(default=15.0, gt=1.0, le=300.0)
    oanda_stream_startup_timeout_seconds: float = Field(default=60.0, gt=1.0, le=300.0)
    forex_cycle_wake_seconds: int = Field(default=60, ge=10, le=300)
    forex_execution_enabled: bool = Field(default=False)
    forex_live_trading_ack: str = Field(default="")
    forex_timezone: str = Field(default="UTC")
    forex_symbols: str = Field(
        default="EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,USD_CHF,NZD_USD,EUR_GBP,EUR_JPY,GBP_JPY"
    )
    forex_max_positions: int = Field(default=5, ge=1, le=100)
    forex_max_portfolio_exposure: float = Field(default=2.0, gt=0.0)
    forex_max_currency_exposure: float = Field(default=1.0, gt=0.0)
    forex_max_correlated_currency_positions: int = Field(default=3, ge=1, le=20)
    forex_max_spread_pips: float = Field(default=5.0, gt=0.0)
    forex_max_spread_atr_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    forex_risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.05)
    forex_paper_account_equity: float = Field(default=100000.0, gt=0.0)
    forex_daily_loss_limit_fraction: float = Field(default=0.02, gt=0.0, le=0.20)
    forex_max_margin_utilization: float = Field(default=0.50, gt=0.0, le=1.0)
    forex_account_home_currency: str = Field(default="USD")
    forex_orb_sessions: str = Field(default="LONDON,NEW_YORK")
    forex_vwap_mode: Literal["disabled", "tick_volume_approximation"] = Field(default="disabled")
    forex_regime_trend_adx_min: float = Field(default=25.0, ge=0.0, le=100.0)
    forex_regime_high_volatility_atr_ratio: float = Field(default=0.01, gt=0.0, le=1.0)
    forex_regime_low_volatility_atr_ratio: float = Field(default=0.002, ge=0.0, le=1.0)
    forex_db_schema: str = Field(default="forex")
    forex_config_root: str = Field(default="config/forex")
    forex_artifact_root: str = Field(default="artifacts/forex")
    forex_log_root: str = Field(default="logs/forex")
    # Trade horizons. Intraday is the original swing/intraday pass; scalp is a
    # parallel, independently-governed 5m pass (additive-only: with scalp disabled
    # the worker behaves exactly as before this feature). Both run the same shared
    # strategy core and deterministic risk engine; scalp only swaps its own
    # timeframe set + tighter risk/spread limits and tags decisions SCALP.
    forex_intraday_timeframes: str = Field(default="15m,30m,1h,4h")
    forex_scalp_enabled: bool = Field(default=False)
    forex_scalp_timeframes: str = Field(default="5m")
    forex_scalp_risk_per_trade: float = Field(default=0.0025, gt=0.0, le=0.05)
    forex_scalp_max_positions: int = Field(default=3, ge=1, le=100)
    forex_scalp_max_spread_pips: float = Field(default=2.0, gt=0.0)
    # PAPER-ONLY research override. When True, SHADOW (unvalidated) strategies are
    # allowed to produce paper decisions instead of being blocked by the
    # walk-forward/PAPER_APPROVED eligibility gate. The deterministic risk engine,
    # MTF-conflict resolution, ML and LLM review still apply; only the "must be
    # validated first" gate is lifted. It has NO effect on live execution (Forex
    # never routes real OANDA orders) and defaults off so behaviour stays fail-closed.
    forex_allow_unvalidated_paper: bool = Field(default=False)

    crypto_provider: str = Field(default="")
    crypto_api_key: SecretStr = Field(default="")
    crypto_api_secret: SecretStr = Field(default="")
    crypto_execution_enabled: bool = Field(default=False)
    crypto_db_schema: str = Field(default="crypto")
    crypto_config_root: str = Field(default="config/crypto")
    crypto_artifact_root: str = Field(default="artifacts/crypto")
    crypto_log_root: str = Field(default="logs/crypto")

    # ===========================================
    # LLM Provider - Groq
    # ===========================================
    groq_api_key: SecretStr = Field(default="", description="Groq API key for LLM access")
    groq_model_primary: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary Groq model for agent reasoning",
    )
    groq_model_fallback: str = Field(
        default="llama-3.1-8b-instant",
        description="Fallback Groq model for rate limit scenarios",
    )
    groq_temperature: float = Field(
        default=0.1,
        description="Temperature for LLM responses (low for consistency)",
    )
    groq_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens per LLM response",
    )

    # ===========================================
    # LLM Provider selection (Groq / Gemini / DeepSeek)
    # ===========================================
    # Every LLM-backed agent node (market_regime, strategy_selection, signal_validation,
    # news_analyst) goes through src/agents/llm_factory.py rather than hardcoding a
    # provider, so switching this one setting moves every agent at once. Groq stays the
    # default -- it's what's been validated against this codebase's prompts; Gemini/
    # DeepSeek/Qwen (Alibaba Cloud DashScope) are available, cheaper alternatives for
    # future use (see llm_factory.create_chat_model).
    llm_provider: Literal["groq", "gemini", "deepseek", "qwen"] = Field(
        default="groq",
        description="Which LLM provider every agent node uses. Switching this requires "
        "the matching API key to be set (see validate_configuration) -- there is no "
        "silent fallback to Groq if e.g. GOOGLE_API_KEY is missing.",
    )
    google_api_key: SecretStr = Field(
        default="",
        description="Google AI Studio API key for Gemini (required when LLM_PROVIDER=gemini).",
    )
    gemini_model_primary: str = Field(
        default="gemini-2.0-flash",
        description="Primary Gemini model for agent reasoning.",
    )
    gemini_model_fallback: str = Field(
        default="gemini-1.5-flash",
        description="Fallback Gemini model for rate-limit scenarios.",
    )
    deepseek_api_key: SecretStr = Field(
        default="",
        description="DeepSeek API key (required when LLM_PROVIDER=deepseek). DeepSeek is "
        "OpenAI-API-compatible; served via langchain-openai pointed at api.deepseek.com.",
    )
    deepseek_model_primary: str = Field(
        default="deepseek-chat",
        description="Primary DeepSeek model for agent reasoning.",
    )
    deepseek_model_fallback: str = Field(
        default="deepseek-chat",
        description="Fallback DeepSeek model for rate-limit scenarios (DeepSeek does not "
        "publish a distinct lighter tier the way Groq/Gemini do, so this defaults to the "
        "same model; override if that changes).",
    )
    # Field/env-var name matches DashScope's own convention (DASHSCOPE_API_KEY) rather
    # than a DeltaQuant-invented QWEN_API_KEY -- same precedent as google_api_key above
    # (GOOGLE_API_KEY, not GEMINI_API_KEY): the key is issued by the platform, not the
    # model family.
    dashscope_api_key: SecretStr = Field(
        default="",
        description="Alibaba Cloud DashScope API key for Qwen (required when "
        "LLM_PROVIDER=qwen). Qwen is OpenAI-API-compatible via DashScope's "
        "compatible-mode endpoint; served via langchain-openai pointed at qwen_base_url.",
    )
    qwen_base_url: str = Field(
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI-compatible-mode base URL. Defaults to the "
        "international endpoint; accounts provisioned in mainland China should override "
        "this to https://dashscope.aliyuncs.com/compatible-mode/v1.",
    )
    qwen_model_primary: str = Field(
        default="qwen3.7-plus",
        description="Primary Qwen model for agent reasoning.",
    )
    qwen_model_fallback: str = Field(
        default="qwen3.7-plus",
        description="Fallback Qwen model for rate-limit scenarios. Defaults to the same "
        "model as primary (like DeepSeek above) -- override once a confirmed lighter "
        "DashScope tier is picked for this account.",
    )
    qwen_fast_model: str = Field(
        default="",
        description="Optional cheaper/faster Qwen model for compact routine tasks. Empty "
        "reuses qwen_model_fallback so existing deployments do not change model silently.",
    )
    qwen_review_model: str = Field(
        default="",
        description="Optional stronger Qwen model for ambiguous candidate review. Empty "
        "reuses qwen_model_primary.",
    )
    qwen_max_output_tokens_news: int = Field(default=96, ge=32, le=2048)
    qwen_max_output_tokens_regime: int = Field(default=256, ge=32, le=2048)
    qwen_max_output_tokens_strategy: int = Field(default=256, ge=32, le=2048)
    qwen_max_output_tokens_fast: int = Field(default=256, ge=32, le=4096)
    qwen_max_output_tokens_review: int = Field(default=512, ge=32, le=4096)
    qwen_max_input_tokens_per_review: int = Field(
        default=900,
        ge=128,
        le=8192,
        description="Maximum compact contextual input per reviewed candidate. Raw candles "
        "are never included.",
    )
    qwen_max_reviews_per_event: int = Field(
        default=20,
        ge=1,
        le=272,
        description="Operational capacity ceiling for context-required candidates in one "
        "settled-candle event; overflow is marked deferred, never bad signal.",
    )
    qwen_regime_uncertain_below: float = Field(default=0.60, ge=0.0, le=1.0)
    qwen_ml_conflict_below: float = Field(default=0.45, ge=0.0, le=1.0)
    qwen_max_output_tokens_learning: int = Field(default=256, ge=32, le=2048)
    qwen_max_output_tokens_discovery: int = Field(default=1800, ge=128, le=8192)

    # LLM cost optimization is opt-in so deployments can prove before/after behavior
    # against the same replay fixture. The individual switches below are inert while
    # this master switch is false.
    llm_cost_optimization_enabled: bool = Field(default=False)
    llm_batch_candidates: bool = Field(
        default=False,
        description="Reserved opt-in for a future Trade Critic batch. The existing "
        "signal_validation node's compatible batch response is not changed by this flag.",
    )
    llm_optional_fast_path: bool = Field(
        default=False,
        description="Allow a future zero-Qwen approval path. Disabled by default; current "
        "implementation never treats this flag as approval authority.",
    )
    llm_cache_ttl_scalp_seconds: int = Field(default=300, ge=0)
    llm_cache_ttl_swing_seconds: int = Field(default=1800, ge=0)
    llm_cache_material_price_move_bps: float = Field(default=20.0, ge=1.0)
    llm_cache_ml_probability_step: float = Field(default=0.05, gt=0.0, le=1.0)
    llm_cache_expected_r_step: float = Field(default=0.10, gt=0.0)
    llm_cache_entry_score_step: float = Field(default=5.0, gt=0.0)
    market_context_ttl_seconds: int = Field(default=600, ge=0)
    market_news_ttl_seconds: int = Field(default=600, ge=0)
    sector_context_ttl_seconds: int = Field(default=900, ge=0)
    market_context_breadth_invalidation_step: float = Field(default=0.10, gt=0.0, le=1.0)
    market_context_volatility_invalidation_step: float = Field(default=0.25, gt=0.0)
    market_context_sentiment_invalidation_step: float = Field(default=0.20, gt=0.0, le=2.0)

    # ===========================================
    # Broker API - DhanHQ (Optional for free tier)
    # ===========================================
    dhan_client_id: str | None = Field(
        default=None,
        description="DhanHQ client ID (optional for local paper trading)",
    )
    dhan_access_token: SecretStr | None = Field(
        default=None,
        description="DhanHQ access token. Only needed if you're NOT using the PIN+TOTP "
        "auto-login below — get_valid_access_token() prefers a fresh auto-generated "
        "token and falls back to this static value (useful for a quick manual token, "
        "though it expires in ~24h and has to be re-pasted by hand)",
    )
    dhan_pin: SecretStr | None = Field(
        default=None,
        description="DhanHQ account PIN, used with dhan_totp_secret to auto-generate a "
        "fresh access token (see src/markets/nse/broker/dhan/auth.py) instead of pasting a "
        "manually-generated one every ~24h",
    )
    dhan_totp_secret: SecretStr | None = Field(
        default=None,
        description="Base32 TOTP secret for DhanHQ's 2FA (the same secret an "
        "authenticator app would use), needed alongside dhan_pin for automatic "
        "access-token generation",
    )
    dhan_token_cache_file: str = Field(
        default="run/nse/dhan/token_cache.json",
        description="Where the auto-generated DhanHQ access token is cached between "
        "restarts (avoids hitting the login endpoint every process start; a token is "
        "reused until close to its documented 24h expiry)",
    )
    dhan_api_key: str | None = Field(
        default=None,
        description="DhanHQ Partner API key (not required for the PIN+TOTP "
        "generateAccessToken flow itself; kept for other Partner API use)",
    )
    dhan_api_secret: SecretStr | None = Field(
        default=None,
        description="DhanHQ Partner API secret, paired with dhan_api_key",
    )
    dhan_auth_base_url: str = Field(
        default="https://auth.dhan.co",
        description="DhanHQ authentication host — where generateAccessToken lives "
        "(separate from dhan_base_url, which is the trading/data API host)",
    )
    dhan_base_url: str = Field(
        default="https://api.dhan.co/v2",
        description="DhanHQ API base URL (use https://api.dhan.co/v2 for live)",
    )
    dhan_feed_url: str = Field(
        default="wss://api-feed.dhan.co",
        description="DhanHQ WebSocket live-market-feed host",
    )
    dhan_exchange_segment: str = Field(
        default="NSE_EQ",
        description="DhanHQ exchange segment for equity requests (historical data, "
        "quotes, orders) — NSE_EQ for NSE cash equities",
    )
    dhan_product_type: str = Field(
        default="CNC",
        description="DhanHQ product type for equity delivery orders (CNC = Cash and "
        "Carry). Not used for paper trading, only if allow_live_orders is ever enabled",
    )
    dhan_instrument: str = Field(
        default="EQUITY",
        description="DhanHQ instrument type for historical-data/quote requests",
    )
    dhan_instrument_master_url: str = Field(
        default="https://images.dhan.co/api-data/api-scrip-master.csv",
        description="DhanHQ's public instrument-master CSV, used to resolve symbol -> "
        "security ID (see src/markets/nse/broker/dhan/instruments.py)",
    )
    enable_dhan_instrument_lookup: bool = Field(
        default=True,
        description="Resolve NSE equity security IDs by fetching DhanHQ's public "
        "instrument master at startup, instead of relying on a small hardcoded "
        "watchlist — lets any symbol in your discovery universe (e.g. a custom "
        "STOCK_UNIVERSE_CSV_PATH) get live Dhan quotes, not just a hardcoded ~20. "
        "Disable if your network can't reach DhanHQ's instrument-list host and you "
        "want to skip straight to the small built-in fallback watchlist",
    )
    enable_dhan_historical_data: bool = Field(
        default=True,
        description="Source historical OHLCV candles (for indicators and the scalping "
        "screener) from DhanHQ. Disabling this leaves no production historical data "
        "source — only the separately gated testing-only synthetic history remains "
        "available (YFinance was removed).",
    )
    enable_synthetic_history: bool = Field(
        default=False,
        description="TESTING ONLY: seed synthetic daily OHLCV when Dhan history is disabled. "
        "The live loop additionally requires forced trading, local-paper execution, paper "
        "trading, disabled live orders, and disabled Dhan quote/history calls before honoring "
        "this switch.",
    )
    enable_dhan_quotes: bool = Field(
        default=True,
        description="Source current quotes (sector movers' gainers/losers scan) from "
        "DhanHQ's batched quote_data endpoint. Disabling this leaves no live-quote "
        "source at all — DhanHQ is the only one this codebase supports (YFinance was "
        "removed). change_percent is computed against each symbol's real previous "
        "close (fetched via DhanHQ's historical daily data and cached for the day — "
        "see dhan_previous_close_cache_file) — NOT DhanHQ's own 'net_change' quote "
        "field, which was verified live to not track day-over-day change at all (it "
        "read 0 for a stock that had genuinely moved ~3.5%)",
    )
    dhan_previous_close_cache_file: str = Field(
        default="run/nse/dhan/previous_close_cache.json",
        description="Where each symbol's previous daily close is cached between "
        "sector-movers scans (refreshed once per IST calendar day — fetching it is "
        "~275 sequential DhanHQ calls, too slow to repeat every 3-minute scan)",
    )

    # ===========================================
    # Observability - Langfuse
    # ===========================================
    langfuse_public_key: SecretStr = Field(
        default="",
        description="Optional Langfuse public API key",
    )
    langfuse_secret_key: SecretStr = Field(
        default="",
        description="Optional Langfuse secret API key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse Cloud or self-hosted base URL",
    )
    langfuse_environment: str = Field(
        default="paper",
        description="Environment label attached to Langfuse traces",
    )
    langfuse_tracing_enabled: bool = Field(
        default=False,
        description="Enable optional Langfuse tracing",
    )

    # ===========================================
    # Database - PostgreSQL
    # ===========================================
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/trading_agent",
        description="PostgreSQL connection URL",
    )

    # ===========================================
    # Local market-history store (Postgres / TimescaleDB)
    # ===========================================
    market_history_store_enabled: bool = Field(
        default=True,
        description="Persist normalized OHLCV candles and read them locally for indicators/ML.",
    )
    market_history_database_url: str = Field(
        default="",
        description="Dedicated Postgres/TimescaleDB URL for candles. Empty reuses DATABASE_URL.",
    )
    market_history_enable_timescale: bool = Field(
        default=True,
        description="Convert market_candles to a hypertable when TimescaleDB is installed.",
    )
    market_history_require_timescale: bool = Field(
        default=False,
        description="Fail history-store initialization when TimescaleDB is unavailable.",
    )
    market_history_auto_sync: bool = Field(
        default=True,
        description="Run the resumable Dhan historical ingestion job in the background.",
    )
    market_history_lookback_days: int = Field(default=730, ge=30, le=1825)
    market_history_overlap_days: int = Field(default=3, ge=1, le=30)
    market_history_timeframes: str = Field(
        default="5m,15m,1h,1d",
        description="Native candle series to persist; 30m and 4h are derived locally.",
    )
    market_history_refresh_hours: float = Field(default=6.0, ge=0.25, le=168.0)
    market_history_sync_during_market: bool = Field(
        default=False,
        description="Allow bulk history synchronization during NSE hours.",
    )
    ml_predictions_enabled: bool = Field(
        default=True,
        description="Master switch for the ML prediction layer (PredictionAgent inference "
        "in support_agents and the ML term in rank_signals' scoring blend). This is a soft "
        "signal only -- it never gates paper/live execution (that's the separate walk-forward "
        "StrategyEligibilityRegistry check). False = no sklearn inference calls anywhere; "
        "ranking and agent state fall back to their existing no-prediction-available path "
        "(already exercised whenever the agent abstains), so behavior stays well-defined.",
    )
    ml_history_bars: int = Field(
        default=3000,
        ge=200,
        le=20000,
        description="Maximum locally stored candles used by each walk-forward ML prediction.",
    )
    llm_prediction_context_max_symbols: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum compact precomputed ML predictions supplied to one LLM cycle.",
    )

    # ===========================================
    # Optional: Redis (not required for basic functionality)
    # ===========================================
    redis_url: str | None = Field(
        default=None,
        description="Optional Redis URL for market data caching",
    )

    # ===========================================
    # Trading Configuration
    # ===========================================
    trading_mode: Literal["paper", "live"] = Field(
        default="paper",
        description="Trading mode - paper for simulation, live for real trading",
    )
    max_daily_trades: int = Field(
        default=50,
        description="Legacy outer maximum. Permanent paper trading additionally applies "
        "paper_daily_entry_cap, which is never allowed above six.",
    )
    paper_daily_entry_cap: int = Field(
        default=6,
        ge=1,
        le=6,
        description="Hard cap on new simulated-paper entries per IST day (maximum six).",
    )
    max_position_size: float = Field(
        default=100000.0,
        description="Maximum position size in INR",
    )
    daily_loss_limit: float = Field(
        default=10000.0,
        description="Maximum daily loss limit in INR (kill switch trigger)",
    )

    # ===========================================
    # Agent Memory Configuration
    # ===========================================
    memory_top_n_lessons: int = Field(
        default=5,
        description="Number of top lessons to inject into agent context",
    )
    memory_decay_days: int = Field(
        default=30,
        description="Days after which lesson relevance starts decaying",
    )
    enable_learning: bool = Field(
        default=True,
        description="Close the learn-from-losses loop: classify closed trades into lessons and "
        "mark injected lessons as successful/not (adds an LLM call per loss). Disable to skip.",
    )

    # ===========================================
    # Free Tier Configuration
    # ===========================================
    market_data_source: Literal["dhan", "oanda"] = Field(
        default="dhan",
        description="Market data source. DhanHQ only — YFinance support was removed "
        "(it required no account but was unreliable under load and duplicated "
        "DhanHQ's own data). When Dhan is unavailable the live loop automatically "
        "falls back to simulated data rather than failing (see MarketDataManager).",
    )
    execution_mode: Literal["local_paper", "shadow", "dhan_paper", "live"] = Field(
        default="local_paper",
        description="Execution mode: local_paper (free), shadow (mirror live, send nothing), "
        "dhan_paper (simulate against Dhan-shaped data/mechanics only — there is no verified "
        "Dhan sandbox endpoint, so this NEVER reaches a real broker route regardless of "
        "allow_live_orders or credentials; see resolve_effective_execution_mode / C-2 in "
        "docs/audits/DeltaQuant-Quant-Risk-Review.md), or live (the only mode that can ever place a real "
        "order, and only when trading_mode=live AND allow_live_orders=true AND Dhan "
        "credentials are present).",
    )
    paper_execution_mode: Literal["market_paper", "mock"] = Field(
        default="market_paper",
        description="Paper-runtime environment. market_paper accepts executable decisions "
        "only from current Dhan quotes during a verified NSE session; mock explicitly uses "
        "simulated data and an isolated persistence namespace. This is distinct from "
        "execution_mode, which selects the broker-routing implementation.",
    )
    allow_live_orders: bool = Field(
        default=False,
        description="Master safety gate: real broker orders are only ever sent when this is "
        "True AND execution_mode=live AND trading_mode=live. With execution_mode=live but this "
        "False, execution runs in SHADOW (no orders sent). dhan_paper never sends real orders "
        "regardless of this flag.",
    )
    long_only: bool = Field(
        default=True,
        description="Cash-investing mode: buy opens a holding; sell may only close shares already held.",
    )
    enable_news_analysis: bool = Field(
        default=True,
        description="Enable AI-powered news sentiment analysis",
    )
    paper_wallet_balance: float = Field(
        default=1000000.0,
        description="Starting balance for local paper trading (INR)",
    )

    # ===========================================
    # Telegram Notifications
    # ===========================================
    telegram_bot_token: str | None = Field(
        default=None,
        description="Telegram bot token from @BotFather",
    )
    telegram_chat_id: str | None = Field(
        default=None,
        description="Your Telegram chat ID from @userinfobot",
    )
    telegram_enabled: bool = Field(
        default=True,
        description="Enable Telegram notifications",
    )

    # ===========================================
    # Web UI (optional; terminal dashboard is unaffected either way)
    # ===========================================
    enable_web_ui: bool = Field(
        default=False,
        description="Serve a live web dashboard (FastAPI+WebSocket) alongside the CLI dashboard",
    )
    web_ui_host: str = Field(default="127.0.0.1", description="Web UI bind host")
    web_ui_port: int = Field(default=8000, description="Web UI bind port")
    web_frontend_port: int = Field(
        default=3000,
        description="Next.js frontend dev-server port, used by the local-services "
        "liveness probe (GET /api/system/jobs) -- not the port the frontend binds "
        "itself with (that's Next.js's own -p flag / PORT env var).",
    )
    web_ui_cors_origins: str = Field(
        default=(
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:3001,http://127.0.0.1:3001"
        ),
        description="Comma-separated origins allowed to open the web UI WebSocket "
        "(Next.js dev server)",
    )
    web_ui_username: str = Field(
        default="admin",
        description="Dashboard login username (single operator account).",
    )
    web_ui_password_hash: str = Field(
        default="",
        description="Dashboard login password hash, generated by "
        "`uv run python scripts/set_dashboard_password.py`. Required whenever "
        "ENABLE_WEB_UI=true (see validate_configuration) — the dashboard exposes "
        "wallet balance, positions, and trading activity and must never be served "
        "without authentication (M-8, docs/audits/DeltaQuant-Quant-Risk-Review.md).",
    )
    web_ui_session_secret: str = Field(
        default="",
        description="Random secret signing dashboard session tokens, generated by "
        "`uv run python scripts/set_dashboard_password.py`. Required whenever "
        "ENABLE_WEB_UI=true. Rotating it logs out every existing session.",
    )
    web_ui_session_ttl_minutes: int = Field(
        default=720,
        description="How long a dashboard login session stays valid (default 12h).",
    )
    web_ui_cookie_secure: bool = Field(
        default=False,
        description="Mark the session cookie Secure (HTTPS-only). Set true once the "
        "dashboard is served over HTTPS (e.g. behind the nse.ventoday.com Nginx/TLS "
        "reverse proxy) -- false only makes sense for plain-HTTP/loopback access, "
        "where the cookie would otherwise never be sent at all.",
    )
    web_ui_login_max_attempts: int = Field(
        default=5,
        description="Failed dashboard login attempts (per client IP) before a temporary lockout.",
    )
    web_ui_login_lockout_minutes: int = Field(
        default=15,
        description="How long a client IP is locked out after web_ui_login_max_attempts failures.",
    )
    stock_universe_csv_path: str | None = Field(
        default="config/nse/symbols.csv",
        description="Optional path to a CSV file with a 'symbol' column, overriding the "
        "built-in NIFTY50+midcap list as the stock-discovery universe (feeds the "
        "trading-signal loop, sector movers, and the scalping screener alike). Falls "
        "back to the built-in list if unset, missing, or malformed.",
    )
    enable_sector_movers: bool = Field(
        default=True,
        description="Scan the full NIFTY50+midcap universe for sector-wise top gainers/"
        "losers, separate from the trading-signal loop",
    )
    sector_movers_refresh_seconds: int = Field(
        default=180,
        description="How often to refresh sector-wide movers (seconds) — kept well above "
        "the trading-cycle interval since it scans ~70 symbols",
    )
    enable_scalping_screener: bool = Field(
        default=True,
        description="Scan the discovery universe for symbols that oscillate by a "
        "meaningful rupee amount multiple times a day (zigzag swing count) — useful for "
        "scalping, separate from the trading-signal loop",
    )
    scalping_screener_refresh_seconds: int = Field(
        default=1800,
        description="How often to rescan for scalping candidates (seconds). Much longer "
        "than sector movers since each scan fetches ~10 days of 5-minute candles per "
        "symbol and the underlying multi-day pattern doesn't shift within a session",
    )
    scalping_swing_threshold_pct: float = Field(
        default=0.5,
        description="Minimum move, as a % of the symbol's own price, to count as one "
        "zigzag swing in the scalping screener. Percentage rather than a flat rupee "
        "amount so cheap and expensive stocks are held to a comparably-sized bar "
        "(a flat rupee threshold structurally favors expensive stocks)",
    )
    scalping_screener_lookback_days: int = Field(
        default=7,
        description="How many recent trading days the scalping screener averages over, "
        "at each of the 15m/30m/1h timeframes it scans",
    )

    # ===========================================
    # Market Hours Configuration
    # ===========================================
    market_open_time: str = Field(
        default="09:15",
        description="Market open time (HH:MM) in IST",
    )
    market_close_time: str = Field(
        default="15:30",
        description="Market close time (HH:MM) in IST",
    )
    no_trading_before: str = Field(
        default="09:15",
        description="No trading before this time (HH:MM)",
    )
    no_trading_after: str = Field(
        default="15:15",
        description="No trading after this time (HH:MM)",
    )
    force_trading_window: bool = Field(
        default=False,
        description="TESTING ONLY: bypass the weekday/trading-hours check so cycles run "
        "regardless of when it actually is. Real market data (Dhan REST last-known quotes, "
        "historical OHLCV) already works outside market hours, so this is the only gate "
        "standing between a closed market and a full end-to-end pipeline run. Loudly logged "
        "whenever active; never enable this for a session with real capital or intent to "
        "place live orders.",
    )
    intraday_square_off_enabled: bool = Field(
        default=True,
        description="Force-close any open position on an intraday timeframe (see "
        "intraday_timeframes) at intraday_square_off_time, regardless of its stop/target/"
        "max-hold state -- so a 5m/15m position is never carried overnight into a session "
        "it was never evaluated against. Independent of, and overrides, ExitManager's "
        "generic max_hold_minutes time exit.",
    )
    intraday_square_off_time: str = Field(
        default="15:15",
        description="HH:MM (IST) at which open intraday-timeframe positions are force-closed.",
    )
    intraday_timeframes: str = Field(
        default="5m,15m",
        description="Comma-separated timeframes treated as intraday for square-off purposes.",
    )

    # ===========================================
    # Position Sizing Configuration
    # ===========================================
    max_position_pct: float = Field(
        default=0.10,
        description="Maximum position size as fraction of capital (0.10 = 10%)",
    )
    risk_per_trade: float = Field(
        default=0.002,
        description="Maximum risk per trade as fraction of capital (0.002 = 0.2%)",
    )
    max_total_risk: float = Field(
        default=0.008,
        description="Maximum aggregate open stop risk as a fraction of account equity. "
        "Must fit inside the configured daily-loss budget after its safety buffer.",
    )
    open_risk_daily_loss_buffer: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description="Fraction of the hard daily-loss limit available to aggregate open stop "
        "risk. The remaining headroom covers costs, slippage, and gap-through-stop risk.",
    )
    max_concurrent_positions: int = Field(
        default=5,
        description="Maximum number of open positions at once, across all symbols. The "
        "risk_compliance gate blocks new entries once this many are open.",
    )
    max_total_exposure_pct: float = Field(
        default=50.0,
        description="Maximum total capital deployed across all open positions, as a "
        "percent of the account. Distinct from max_total_risk, which bounds risk-at-stake "
        "(stop distance x size), not notional exposure.",
    )
    paper_target_pct: float = Field(
        default=3.5,
        ge=3.0,
        le=4.0,
        description="Default long-only paper target, expressed as a percent above entry.",
    )
    paper_min_ml_confidence: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum upward ML confidence required by the local confirmation gate.",
    )
    paper_min_volume_ratio: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum current 5-minute volume divided by its 20-bar average.",
    )
    paper_required_higher_timeframes: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Minimum aligned bullish frames among 30m, 1h, and 4h.",
    )
    paper_averaging_enabled: bool = Field(
        default=False,
        description="Enable the isolated, capped paper-only averaging experiment.",
    )
    paper_averaging_max_adds: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Maximum add fills per lifecycle; averaging can never continue indefinitely.",
    )
    paper_averaging_trigger_pct: float = Field(
        default=1.0,
        ge=0.25,
        le=3.0,
        description="Adverse move from weighted entry before the experiment may add.",
    )
    paper_averaging_add_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=0.5,
        description="Maximum add quantity as a fraction of the original quantity.",
    )
    simulated_seed: int = Field(
        default=20260807,
        description="Deterministic seed for the coherent paper-market simulator.",
    )
    simulated_history_5m_bars: int = Field(
        default=3000,
        ge=500,
        le=10000,
        description="Five-minute bars retained per simulated symbol for multi-timeframe views.",
    )
    mean_reversion_stop_loss_pct: float = Field(
        default=1.2,
        description="Fixed % stop-loss for mean_reversion signals (always percentage-based, "
        "never ATR — it's a bounded snap-back bet, not a trend-following one, so the exit "
        "should match the swing size it's actually catching)",
    )
    mean_reversion_target_pct: float = Field(
        default=2.5,
        description="Fixed % take-profit for mean_reversion signals. Tune this to the "
        "typical swing size of the symbols you're trading it on (see the Scalping "
        "Candidates screener's avg_swing_size/last_price ratio) — a target well above the "
        "real swing amplitude means holding through multiple swings just to get there",
    )

    # ===========================================
    # Scalp trading horizon (SCALP, distinct from the SWING defaults above).
    # Everything here is off/inert until scalp_enabled=True — see CLAUDE.md "Scalp
    # horizon" section. None of this weakens eligibility or any risk_compliance
    # check; it only supplies horizon-specific thresholds those gates read.
    # ===========================================
    scalp_enabled: bool = Field(
        default=False,
        description="Master switch for the scalp scan/rank/gate pipeline. False = "
        "byte-identical behavior to before this feature existed.",
    )
    enable_signal_consolidation: bool = Field(
        default=False,
        description="Consolidate multiple strategies agreeing on the same "
        "symbol+timeframe+direction into one stronger signal before ranking, instead of "
        "treating each as an independent candidate. Off by default so the existing swing "
        "candidate mix is provably unchanged until explicitly enabled.",
    )
    scalp_signal_timeframes: str = Field(
        default="5m,15m",
        description="Comma-separated timeframes the scalp signal scan runs strategies on "
        "(same format/parser as signal_timeframes).",
    )
    scalp_confirmation_timeframes: str = Field(
        default="5m,15m,30m,1h,4h",
        description="Comma-separated timeframes read for multi-timeframe scalp "
        "confirmation. Semantics: 5m=execution, 15m=primary setup/confirmation, "
        "30m=directional confirmation, 1h=context, 4h=optional macro filter.",
    )
    scalp_macro_filter_enabled: bool = Field(
        default=True,
        description="Whether the optional 4h macro-context filter participates in "
        "multi-timeframe scalp confirmation (scalp_confirmation_timeframes' 4h leg).",
    )
    scalp_required_mtf_alignment: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Minimum number of confirmation timeframes that must agree "
        "(execution/primary/directional/context/macro) for a scalp candidate to pass "
        "multi-timeframe confirmation.",
    )
    scalp_risk_per_trade: float = Field(
        default=0.001,
        description="Max risk per SCALP-horizon trade as a fraction of capital — tighter "
        "than swing's risk_per_trade since scalp stops are much closer.",
    )
    scalp_max_position_pct: float = Field(
        default=0.05,
        description="Max SCALP-horizon position size as a fraction of capital — tighter "
        "than swing's max_position_pct (0.10).",
    )
    scalp_target_pct: float = Field(
        default=0.8,
        description="Default SCALP-horizon take-profit, percent above entry — an order of "
        "magnitude tighter than swing's paper_target_pct (3.5), matching a 5-15m horizon.",
    )
    scalp_stop_loss_pct: float = Field(
        default=0.4,
        description="Default SCALP-horizon stop-loss, percent below entry.",
    )
    scalp_min_rr: float = Field(
        default=1.5,
        description="Minimum risk-reward ratio for the SCALP-horizon signal_validation "
        "fallback rule. Deliberately shipped equal to the swing fallback's hardcoded 1.5 "
        "so this stage only makes the bar horizon-selectable, never lowers it.",
    )
    scalp_min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum signal confidence for the SCALP-horizon signal_validation "
        "fallback rule. Deliberately shipped equal to the swing fallback's hardcoded 0.6.",
    )
    scalp_min_volume_ratio: float = Field(
        default=1.2,
        ge=0.0,
        description="Minimum current-bar volume divided by its 20-bar average, required by "
        "the scalp EntryQualityEvaluator's relative-volume check.",
    )
    scalp_matrix_reject_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Per-timeframe assessment-matrix score below which a symbol/timeframe "
        "cell is labeled REJECT outright (below this, WAIT up to scalp_min_confidence, "
        "at/above scalp_min_confidence, BUY -- subject to regime compatibility). This is a "
        "descriptive label only; it never substitutes for strategy eligibility or "
        "signal_validation, both of which still run independently before any trade.",
    )
    scalp_vwap_max_distance_pct: float = Field(
        default=0.5,
        ge=0.0,
        description="Max acceptable distance from session VWAP (percent of price) for the "
        "EntryQualityEvaluator to consider an entry not yet extended.",
    )
    scalp_ema9_max_distance_pct: float = Field(
        default=0.6,
        ge=0.0,
        description="Max acceptable distance from EMA9 (percent of price) for the "
        "EntryQualityEvaluator to consider an entry not yet extended.",
    )
    scalp_atr_extension_max_multiple: float = Field(
        default=1.5,
        gt=0.0,
        description="Max acceptable distance from VWAP expressed as a multiple of ATR — "
        "an ATR-normalized extension check independent of the flat percent checks above, "
        "so the same threshold behaves consistently across low- and high-volatility names.",
    )
    scalp_wick_ratio_max: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="Max acceptable wick-to-range ratio on the triggering candle before "
        "the EntryQualityEvaluator treats it as rejection/indecision rather than a clean "
        "trigger.",
    )
    scalp_swing_lookback_bars: int = Field(
        default=40,
        ge=10,
        description="Bars of 5m history scanned for recent swing support/resistance in the "
        "EntryQualityEvaluator (reuses scalping_screener's zigzag detector).",
    )
    scalp_breakout_retest_lookback_bars: int = Field(
        default=12,
        ge=3,
        description="Bars of 5m history scanned for breakout/retest state detection in the "
        "EntryQualityEvaluator.",
    )
    scalp_resistance_min_distance_pct: float = Field(
        default=0.3,
        ge=0.0,
        description="Minimum distance (percent of price) a BUY entry must clear from the "
        "nearest detected swing resistance.",
    )
    scalp_max_active_symbols: int = Field(
        default=5,
        ge=1,
        description="Max symbols retained in the scalp ranker's shortlist per cycle "
        "(scalp analogue of max_active_stocks).",
    )
    scalp_ranking_weight_entry_quality: float = Field(default=0.30, ge=0.0)
    scalp_ranking_weight_mtf_alignment: float = Field(default=0.20, ge=0.0)
    scalp_ranking_weight_volume_liquidity: float = Field(default=0.15, ge=0.0)
    scalp_ranking_weight_regime: float = Field(default=0.10, ge=0.0)
    scalp_ranking_weight_historical_expectancy: float = Field(default=0.15, ge=0.0)
    scalp_ranking_weight_ml_probability: float = Field(default=0.10, ge=0.0)

    # ===========================================
    # Tail-risk guards (deterministic; independent of the LLM stack)
    # ===========================================
    kill_switch_flatten: bool = Field(
        default=True,
        description="When the kill switch fires, also flatten open positions (not just block "
        "new entries) to stop the bleed",
    )
    circuit_guard_enabled: bool = Field(
        default=True,
        description="Skip new entries in scrips at/through their NSE circuit band",
    )
    default_circuit_band_pct: float = Field(
        default=10.0,
        description="Assumed NSE circuit band % when per-scrip data is unavailable (2/5/10/20)",
    )
    max_sector_exposure: float = Field(
        default=0.30,
        description="Maximum exposure to single sector (0.30 = 30%)",
    )
    max_pairwise_correlation: float = Field(
        default=0.80,
        description="Warn when a candidate's daily-return correlation with an existing "
        "open position exceeds this (0.80 = 80%). Sector caps alone miss cross-sector "
        "names that move together; this measures actual co-movement.",
    )
    pairwise_correlation_lookback_days: int = Field(
        default=60,
        description="Trading days of daily closes used to compute the pairwise "
        "return-correlation risk check.",
    )

    # ===========================================
    # Rate Limiting Configuration
    # ===========================================
    groq_requests_per_minute: int = Field(
        default=30,
        description="Groq API rate limit (requests per minute)",
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for API calls",
    )
    enable_llm_agents: bool = Field(
        default=True,
        description="Master switch for the Groq-backed market_regime/strategy_selection/"
        "signal_validation agents. When false, each skips the Groq call entirely and goes "
        "straight to its existing deterministic fallback — useful when Groq is rate-limited "
        "or the daily budget is exhausted and repeated failed calls/retries are just adding "
        "latency without changing the outcome. Does not affect the separate news/sentiment/"
        "prediction support agents.",
    )

    # ===========================================
    # Circuit Breaker Configuration
    # ===========================================
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Failures before circuit breaker opens",
    )
    circuit_breaker_recovery_time: float = Field(
        default=60.0,
        description="Seconds before circuit breaker attempts recovery",
    )

    # ===========================================
    # Cache Configuration
    # ===========================================
    cache_news_ttl: int = Field(
        default=300,
        description="News cache TTL in seconds (5 minutes)",
    )
    cache_quotes_ttl: int = Field(
        default=60,
        description="Quote cache TTL in seconds (1 minute)",
    )
    cache_sentiment_ttl: int = Field(
        default=600,
        description="Sentiment cache TTL in seconds (10 minutes)",
    )

    # ===========================================
    # FinOps - Cost tracking & budgets
    # ===========================================
    finops_enabled: bool = Field(
        default=True,
        description="Enable LLM cost/token accounting and budget alerts",
    )
    daily_token_budget: int = Field(
        default=0,
        description="Max Groq tokens per IST day across all agents (0 = unlimited)",
    )
    daily_cost_budget_usd: float = Field(
        default=0.0,
        description="Max paid-tier-equivalent LLM spend per IST day in USD (0 = unlimited)",
    )
    llm_daily_budget_total: int = Field(
        default=0,
        description="Optional total Qwen token budget per IST day. When non-zero it is "
        "authoritative together with (and never looser than) daily_token_budget.",
    )
    llm_daily_budget_scalp: int = Field(
        default=0,
        description="Optional SCALP Qwen token sub-budget per IST day (0 = unlimited).",
    )
    llm_daily_budget_swing: int = Field(
        default=0,
        description="Optional SWING Qwen token sub-budget per IST day (0 = unlimited).",
    )
    finops_budget_soft_pct: float = Field(
        default=0.8,
        description="Soft-alert threshold as a fraction of a daily budget (0.8 = 80%)",
    )

    # ===========================================
    # Paper trading costs (slippage + NSE-style charges; configurable approximations)
    # ===========================================
    paper_slippage_bps: float = Field(
        default=2.0,
        description="Adverse slippage applied to paper fills, in basis points (2 = 0.02%)",
    )
    paper_brokerage_bps: float = Field(
        default=3.0,
        description="Brokerage as basis points of notional (3 = 0.03%)",
    )
    paper_brokerage_max: float = Field(
        default=20.0,
        description="Per-order brokerage cap in INR (0 = uncapped); mirrors discount brokers",
    )
    paper_statutory_bps: float = Field(
        default=5.0,
        description="Combined STT + exchange txn + SEBI + stamp charges, in basis points",
    )
    paper_gst_pct: float = Field(
        default=18.0,
        description="GST as a percentage of brokerage",
    )

    # ===========================================
    # Data ingestion / efficiency
    # ===========================================
    signals_exclude_forming_bar: bool = Field(
        default=True,
        description="Compute indicators/signals on settled bars only (drop the still-forming "
        "current-day bar) to avoid intra-bar repainting / look-ahead",
    )
    max_quote_staleness_seconds: int = Field(
        default=60,
        description="Skip NEW entries when the freshest quote is older than this many seconds "
        "(0 = disabled). Exits still run on the last known price. 0 is only permitted when "
        "enable_dhan_quotes=False (a declared simulation profile with no live quote feed at "
        "all) — a live-data profile with 0 fails startup, see validate_configuration/H-10.",
    )
    signal_timeframes: str = Field(
        default="15m,30m,1h,4h",
        description="Comma-separated candle timeframes signals are generated on "
        "(any of 15m,30m,1h,4h,1d). Each timeframe is fetched and scored independently, "
        "and every generated signal is tagged with the timeframe it came from.",
    )
    signal_scan_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent symbol-history/indicator workers in a full-universe "
        "scan. Dhan API calls remain protected by the shared provider rate limiter.",
    )
    ranking_stage1_top_k: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum distinct symbol/timeframe candidates passed from deterministic "
        "technical pre-ranking into the expensive local-ML stage.",
    )
    llm_review_all_signals: bool = Field(
        default=False,
        description="Send every generated strategy signal to the LLM graph. When enabled, "
        "MAX_ACTIVE_STOCKS, LLM_REVIEW_MAX_SYMBOLS, UNIVERSE_MAX_PER_SECTOR, and "
        "MAX_SIGNALS_PER_SYMBOL do not truncate the LLM input.",
    )
    max_active_stocks: int = Field(
        default=15,
        description="Maximum number of sector-diverse symbols retained after strategy "
        "signals from the full configured universe are ranked by local ML, expected R, "
        "and measured closed-trade outcomes.",
    )
    llm_review_max_symbols: int = Field(
        default=5,
        description="Maximum locally ranked symbols sent to the LLM agent graph in one "
        "cycle. All symbols receive deterministic multi-timeframe strategy analysis; "
        "this limit bounds Groq latency and token usage only.",
    )
    universe_max_per_sector: int = Field(
        default=2,
        description="Maximum symbols from one sector in the locally ranked shortlist, "
        "preventing a single sector or falling name from dominating review.",
    )
    max_signals_per_symbol: int = Field(
        default=2,
        description="Maximum locally ranked signals retained per surviving symbol for "
        "LLM validation in one cycle.",
    )
    trading_cycle_seconds: int = Field(
        default=120,
        description="Seconds between full LLM investment-review cycles. A moderate "
        "interval protects Groq rate limits while quotes and exits remain live.",
    )

    # ===========================================
    # Quantitative signal discovery (scheduled research + live accepted-factor tilt)
    # ===========================================
    signal_discovery_enabled: bool = Field(
        default=False,
        description="Load accepted discovered signals and apply their bounded live tilt. "
        "Formula generation can run automatically on a separate background schedule.",
    )
    signal_discovery_output_dir: str = Field(default="data/nse/discovered_signals")
    signal_discovery_timeframes: str = Field(default="15m,30m,1h,4h")
    signal_discovery_auto_run: bool = Field(default=False)
    signal_discovery_refresh_hours: int = Field(default=24, ge=1, le=720)
    signal_discovery_request: str = Field(
        default="momentum, mean-reversion, breakout, volatility, and volume-price alpha signals"
    )
    signal_discovery_num_signals: int = Field(default=3, ge=1, le=10)
    signal_discovery_max_iterations: int = Field(default=3, ge=1, le=10)
    signal_discovery_forward_periods: int = Field(default=5, ge=1, le=100)
    signal_discovery_ic_threshold: float = Field(default=0.015, ge=0.0, le=1.0)
    signal_discovery_p_value_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    signal_discovery_min_periods: int = Field(default=30, ge=10)
    signal_discovery_min_cross_section: int = Field(default=8, ge=3)
    signal_discovery_max_probability_tilt: float = Field(default=0.08, ge=0.0, le=0.20)

    # Background-job auto-run schedulers -- same shape as signal_discovery_auto_run
    # above (settings-gated, staleness-driven, default OFF so existing deployments
    # don't silently start consuming Dhan API quota / CPU). See
    # AutomaticValidationScheduler / AutomaticScalpTrainingScheduler.
    strategy_validation_auto_run: bool = Field(default=False)
    strategy_validation_refresh_hours: float = Field(default=24.0, ge=0.01, le=720.0)
    scalp_training_auto_run: bool = Field(default=False)
    scalp_training_refresh_hours: float = Field(default=24.0, ge=0.01, le=720.0)

    # ===========================================
    # Legacy strict-grain validation artifacts (read-only migration source)
    # ===========================================
    strategy_registry_dir: str = Field(
        default="data/nse/strategy_registry",
        description="Read-only location of legacy strict-grain StrategyVersion records. "
        "They are preserved and conservatively migrated into StrategyEligibilityRegistry; "
        "runtime admission no longer reads them directly.",
    )
    strategy_registry_validity_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Legacy validation-expiry window retained for migration compatibility "
        "and for validate_strategy.py metadata.",
    )
    strategy_eligibility_registry_dir: str = Field(
        default="data/nse/strategy_eligibility",
        description="StrategyEligibilityRegistry records keyed by strategy name, exact "
        "timeframe, and model version. Runtime regime policy is stored inside each record "
        "and evaluated separately; trade horizon is decision metadata, not registry identity.",
    )
    strategy_eligibility_paper_gate_enabled: bool = Field(
        default=True,
        description="Master switch for the walk-forward + Bonferroni admission gate on PAPER "
        "trading. True (default): only PAPER_APPROVED/LIVE_APPROVED strategy+timeframe "
        "combinations may place a paper trade -- every other candidate is recorded as "
        "evidence only, never executed. False: every strategy trades paper capital "
        "regardless of its walk-forward verdict (an explicit opt-out of the statistical "
        "proof requirement -- you can lose paper money on a strategy with no demonstrated "
        "edge). This flag ONLY affects the PAPER environment; it has no effect on LIVE --  "
        "LIVE_APPROVED is always required to reach a real broker order no matter what this "
        "is set to, so this can never be used to bypass live-trading safety.",
    )

    # ===========================================
    # Shared-feature production strategy family
    # ===========================================
    production_strategies_enabled: bool = Field(
        default=True,
        description="Evaluate the ten production technical strategy families. Their initial "
        "eligibility remains UNVALIDATED/SHADOW until offline walk-forward approval.",
    )
    production_strategy_version: str = Field(
        default="production_rules_v1",
        description="Rule/parameter version written into validation metadata.",
    )
    production_strategy_timeframes_json: str = Field(
        default="",
        description="Optional JSON overrides for strategy -> eligible timeframe list. Empty "
        "uses the documented conservative default matrix.",
    )
    production_strategy_parameters_json: str = Field(
        default="",
        description="Optional nested JSON threshold overrides by strategy name.",
    )
    production_strategy_score_weights_json: str = Field(
        default="",
        description="Optional nested JSON technical-score component weights by strategy.",
    )
    strategy_benchmark_symbol: str = Field(
        default="",
        description="Optional benchmark symbol for relative-strength momentum. Empty uses the "
        "configured fallback and never hard-codes a broker instrument.",
    )
    strategy_benchmark_fallback: Literal["cross_sectional_median", "disabled"] = Field(
        default="cross_sectional_median",
        description="Fallback when the configured benchmark is unavailable. The median proxy "
        "is labeled explicitly in FeatureSnapshot lineage.",
    )

    # ===========================================
    # Profit-target goal engine
    # ===========================================
    monthly_profit_target_pct: float = Field(
        default=0.0,
        description="Monthly profit target as a fraction of capital (0.05 = 5%/mo; 0 = disabled)",
    )
    monthly_profit_target_amount: float = Field(
        default=0.0,
        description="Monthly profit target in INR (alternative to pct; the larger of the two wins)",
    )
    trading_days_per_month: int = Field(
        default=21,
        description="Assumed NSE trading days per month, used to derive the daily pace target",
    )
    expected_trades_per_day: int = Field(
        default=5,
        description="Expected trades/day, used to derive the win-rate the target requires",
    )
    goal_assumed_win_rate: float = Field(
        default=0.5,
        description="Assumed win rate, used to derive the trade frequency the target requires",
    )
    goal_reward_risk_ratio: float = Field(
        default=1.5,
        description="Assumed reward:risk per trade for goal math (matches risk min_risk_reward)",
    )
    goal_off_pace_tolerance: float = Field(
        default=0.2,
        description="How far below straight-line pace before flagged off-pace (0.2 = 20%)",
    )

    # ===========================================
    # Validation
    # ===========================================

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """Validate configuration consistency.

        Two tiers, deliberately kept separate (see docs/audits/DeltaQuant-Quant-Risk-Review.md C-2/M-5):

        - ``warnings`` — benign cross-field inconsistencies (risk-param sanity, market-hours
          ordering, Telegram config) that degrade gracefully. These are collected but never
          raise, so local/dev iteration stays unobstructed.
        - ``fatal`` — the narrow, explicit set of hazardous combinations that could let a
          paper-labeled config reach a real broker order, or silently disable the live-data
          freshness gate. These raise ``ValueError`` (fail-closed at startup) instead of
          logging, because a typo here changes the safety posture of the whole process.
        """
        warnings: list[str] = []
        fatal: list[str] = []

        # ---------------------------------------------------------------
        # Fatal: hazardous trading_mode x execution_mode x allow_live_orders
        # x Dhan-credential combinations (C-2). The only combination that may
        # ever reach a real broker order is the unambiguous conjunction
        # trading_mode=live AND execution_mode=live AND allow_live_orders=true
        # AND valid Dhan credentials. Anything that half-declares live intent
        # (the master gate armed, or execution_mode=live, without the rest of
        # the conjunction agreeing) fails startup instead of quietly resolving
        # to shadow at runtime, so a mismatched .env is caught immediately
        # rather than discovered by reading multiple settings together.
        # (execution_mode=dhan_paper is intentionally NOT part of this matrix:
        # it is redefined to simulate Dhan-shaped data only and can never reach
        # a live route at all — see resolve_effective_execution_mode below.)
        # ---------------------------------------------------------------
        # Retain only the selected market's broker credentials in this process. This
        # reduces accidental disclosure through diagnostics and prevents an unrelated
        # provider from being initialized merely because its variables exist on the host.
        if self.market == "FOREX":
            object.__setattr__(self, "dhan_client_id", None)
            object.__setattr__(self, "dhan_access_token", None)
            object.__setattr__(self, "dhan_pin", None)
            object.__setattr__(self, "dhan_totp_secret", None)
            object.__setattr__(self, "crypto_api_key", SecretStr(""))
            object.__setattr__(self, "crypto_api_secret", SecretStr(""))
            object.__setattr__(self, "market_data_source", "oanda")
        elif self.market == "NSE":
            object.__setattr__(self, "oanda_account_id", SecretStr(""))
            object.__setattr__(self, "oanda_access_token", SecretStr(""))
            object.__setattr__(self, "crypto_api_key", SecretStr(""))
            object.__setattr__(self, "crypto_api_secret", SecretStr(""))
        else:
            object.__setattr__(self, "dhan_client_id", None)
            object.__setattr__(self, "dhan_access_token", None)
            object.__setattr__(self, "dhan_pin", None)
            object.__setattr__(self, "dhan_totp_secret", None)
            object.__setattr__(self, "oanda_account_id", SecretStr(""))
            object.__setattr__(self, "oanda_access_token", SecretStr(""))

        if self.market == "CRYPTO" and self.crypto_execution_enabled:
            fatal.append(
                "Crypto execution is not implemented. Set CRYPTO_EXECUTION_ENABLED=false."
            )

        has_dhan_credentials = bool(self.dhan_client_id and self.dhan_access_token)
        has_oanda_credentials = bool(
            self.oanda_account_id.get_secret_value()
            and self.oanda_access_token.get_secret_value()
        )
        if self.market == "FOREX" and self.forex_enabled and not has_oanda_credentials:
            warnings.append(
                "FOREX is enabled without OANDA_ACCOUNT_ID/OANDA_ACCESS_TOKEN; "
                "the Forex worker will fail closed as unhealthy."
            )
        if self.market == "FOREX" and self.oanda_environment == "live":
            warnings.append(
                "OANDA LIVE environment selected; broker execution remains disabled."
            )
        if self.market == "FOREX" and self.forex_environment != self.oanda_environment:
            fatal.append(
                "FOREX_ENVIRONMENT and OANDA_ENVIRONMENT must match; refusing an "
                "ambiguous practice/live configuration."
            )
        if self.market == "FOREX":
            expected_rest = {
                "practice": "https://api-fxpractice.oanda.com",
                "live": "https://api-fxtrade.oanda.com",
            }[self.oanda_environment]
            expected_stream = {
                "practice": "https://stream-fxpractice.oanda.com",
                "live": "https://stream-fxtrade.oanda.com",
            }[self.oanda_environment]
            test_environment = self.app_env.strip().lower() in {"test", "testing", "ci"}
            if (
                self.oanda_rest_url
                and self.oanda_rest_url.rstrip("/") != expected_rest
                and not test_environment
            ):
                fatal.append("OANDA_REST_URL override is allowed only in APP_ENV=test/ci")
            if (
                self.oanda_stream_url
                and self.oanda_stream_url.rstrip("/") != expected_stream
                and not test_environment
            ):
                fatal.append("OANDA_STREAM_URL override is allowed only in APP_ENV=test/ci")
        if (
            self.market == "FOREX"
            and self.oanda_environment == "live"
            and self.forex_execution_enabled
        ):
            fatal.append(
                "LIVE OANDA execution is not enabled in this release. Set "
                "FOREX_EXECUTION_ENABLED=false."
            )
        if self.market == "FOREX" and self.forex_execution_enabled and (
            self.oanda_environment != "practice" or self.trading_mode != "paper"
        ):
            fatal.append(
                "FOREX execution may only target OANDA PRACTICE with TRADING_MODE=paper."
            )

        if (
            self.market == "NSE"
            and self.paper_execution_mode == "market_paper"
            and self.force_trading_window
        ):
            fatal.append(
                "PAPER_EXECUTION_MODE=market_paper cannot be combined with "
                "FORCE_TRADING_WINDOW=true. Market-paper entries require a verified NSE "
                "session and real Dhan quotes. Use PAPER_EXECUTION_MODE=mock for an explicit "
                "off-hours simulation run."
            )

        if self.execution_mode == "live" and self.trading_mode != "live":
            fatal.append(
                f"EXECUTION_MODE=live requires TRADING_MODE=live (got "
                f"TRADING_MODE={self.trading_mode!r}). A paper-labeled process must never "
                "declare a live execution mode. Set TRADING_MODE=live only if you genuinely "
                "intend to place real broker orders, otherwise change EXECUTION_MODE to "
                "local_paper, shadow, or dhan_paper."
            )

        if self.allow_live_orders and self.trading_mode != "live":
            fatal.append(
                f"ALLOW_LIVE_ORDERS=true requires TRADING_MODE=live (got "
                f"TRADING_MODE={self.trading_mode!r}) — this is the exact C-2 hazard: the "
                "master live-order gate armed on a config every label says is paper. Set "
                "TRADING_MODE=live to confirm real intent, or ALLOW_LIVE_ORDERS=false."
            )

        if self.market == "NSE" and self.trading_mode == "live" and not has_dhan_credentials:
            fatal.append(
                "TRADING_MODE=live requires Dhan credentials (DHAN_CLIENT_ID + "
                "DHAN_ACCESS_TOKEN, or DHAN_CLIENT_ID + DHAN_PIN + DHAN_TOTP_SECRET for "
                "auto-login) — refusing to start with live trading declared but no way to "
                "authenticate to the broker."
            )

        # ---------------------------------------------------------------
        # Fatal: MAX_QUOTE_STALENESS_SECONDS=0 disables the new-entry freshness
        # gate entirely (H-10). Zero is only safe when enable_dhan_quotes=False
        # — i.e. there is no live quote feed at all, so "staleness" is
        # meaningless. Any profile that pulls real Dhan quotes must declare a
        # conservative positive threshold instead of inheriting a simulation
        # profile's value unnoticed.
        # ---------------------------------------------------------------
        if (
            self.market == "NSE"
            and self.max_quote_staleness_seconds <= 0
            and self.enable_dhan_quotes
        ):
            fatal.append(
                "MAX_QUOTE_STALENESS_SECONDS=0 disables the new-entry data-freshness gate "
                "and is only permitted when ENABLE_DHAN_QUOTES=false (a declared simulation "
                "profile with no live quote feed). This profile has ENABLE_DHAN_QUOTES=true, "
                "so set MAX_QUOTE_STALENESS_SECONDS to a conservative positive value (e.g. "
                "60) or explicitly disable live quotes."
            )

        # ---------------------------------------------------------------
        # Fatal: the web dashboard must never be enabled without authentication
        # configured (M-8, docs/audits/DeltaQuant-Quant-Risk-Review.md: "API protection relies
        # mainly on localhost binding... any remote binding... would require
        # authentication"). This app is designed to be reachable over the network,
        # not just loopback, so there is no "safe because nobody can reach it"
        # fallback to rely on here.
        # ---------------------------------------------------------------
        if self.enable_web_ui and not self.web_ui_password_hash:
            fatal.append(
                "ENABLE_WEB_UI=true requires WEB_UI_PASSWORD_HASH to be set — the dashboard "
                "exposes wallet balance, positions, and trading activity and must never run "
                "without a login. Generate one with "
                "`uv run python scripts/set_dashboard_password.py` and paste the printed "
                "WEB_UI_PASSWORD_HASH / WEB_UI_SESSION_SECRET lines into .env."
            )
        if self.enable_web_ui and not self.web_ui_session_secret:
            fatal.append(
                "ENABLE_WEB_UI=true requires WEB_UI_SESSION_SECRET to be set — without it, "
                "login sessions cannot be signed. Generate one with "
                "`uv run python scripts/set_dashboard_password.py`."
            )

        # ---------------------------------------------------------------
        # Fatal: LLM_PROVIDER must have its matching API key configured. There is no
        # silent fallback to Groq -- an operator who sets LLM_PROVIDER=gemini but
        # forgets GOOGLE_API_KEY should get a clear startup failure, not agents that
        # mysteriously fail every call at runtime and fall back to degraded/deterministic
        # behavior (which the existing per-agent resilience pattern would otherwise mask
        # as "just another LLM failure").
        # ---------------------------------------------------------------
        if self.llm_provider == "groq" and not self.groq_api_key.get_secret_value():
            fatal.append("LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")
        if self.llm_provider == "gemini" and not self.google_api_key.get_secret_value():
            fatal.append(
                "LLM_PROVIDER=gemini requires GOOGLE_API_KEY to be set (get one from "
                "Google AI Studio: https://aistudio.google.com/apikey)."
            )
        if self.llm_provider == "deepseek" and not self.deepseek_api_key.get_secret_value():
            fatal.append(
                "LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY to be set (get one from "
                "https://platform.deepseek.com/api_keys)."
            )
        if self.llm_provider == "qwen" and not self.dashscope_api_key.get_secret_value():
            fatal.append(
                "LLM_PROVIDER=qwen requires DASHSCOPE_API_KEY to be set (get one from "
                "Alibaba Cloud DashScope: https://dashscope.console.aliyun.com/apiKey)."
            )

        # Validate risk parameters. Aggregate open stop risk must fit inside the hard
        # daily-loss budget after reserving a configurable buffer for slippage, costs,
        # and gap-through-stop risk. A contradictory profile is unsafe, so fail startup
        # rather than merely warning.
        daily_loss_fraction = (
            self.daily_loss_limit / self.paper_wallet_balance
            if self.paper_wallet_balance > 0
            else 0.0
        )
        buffered_daily_risk = daily_loss_fraction * self.open_risk_daily_loss_buffer
        if self.max_total_risk > buffered_daily_risk + 1e-12:
            fatal.append(
                f"MAX_TOTAL_RISK={self.max_total_risk:.4f} exceeds the buffered daily-loss "
                f"budget {buffered_daily_risk:.4f} (DAILY_LOSS_LIMIT="
                f"{self.daily_loss_limit:.2f}, PAPER_WALLET_BALANCE="
                f"{self.paper_wallet_balance:.2f}, OPEN_RISK_DAILY_LOSS_BUFFER="
                f"{self.open_risk_daily_loss_buffer:.2f})."
            )

        if self.risk_per_trade > self.max_position_pct:
            warnings.append(
                f"risk_per_trade ({self.risk_per_trade}) should not exceed max_position_pct ({self.max_position_pct})"
            )

        if self.max_total_risk < self.risk_per_trade:
            fatal.append(
                f"MAX_TOTAL_RISK ({self.max_total_risk}) cannot be less than "
                f"RISK_PER_TRADE ({self.risk_per_trade})."
            )

        if self.scalp_risk_per_trade > self.max_total_risk:
            fatal.append(
                f"SCALP_RISK_PER_TRADE ({self.scalp_risk_per_trade}) cannot exceed "
                f"MAX_TOTAL_RISK ({self.max_total_risk})."
            )

        # Scalp ranking weights are a weighted blend, not independent knobs -- warn (don't
        # fail closed) if they drift from summing to 1.0, same severity as the risk-param
        # sanity checks above.
        scalp_weight_sum = (
            self.scalp_ranking_weight_entry_quality
            + self.scalp_ranking_weight_mtf_alignment
            + self.scalp_ranking_weight_volume_liquidity
            + self.scalp_ranking_weight_regime
            + self.scalp_ranking_weight_historical_expectancy
            + self.scalp_ranking_weight_ml_probability
        )
        if abs(scalp_weight_sum - 1.0) > 0.01:
            warnings.append(
                f"scalp_ranking_weight_* fields sum to {scalp_weight_sum:.3f}, expected 1.0"
            )

        if self.scalp_matrix_reject_score >= self.scalp_min_confidence:
            warnings.append(
                f"scalp_matrix_reject_score ({self.scalp_matrix_reject_score}) should be less "
                f"than scalp_min_confidence ({self.scalp_min_confidence}), or every WAIT band "
                "collapses to zero width"
            )

        # Validate market hours
        try:
            from datetime import datetime

            open_time = datetime.strptime(self.market_open_time, "%H:%M")
            close_time = datetime.strptime(self.market_close_time, "%H:%M")
            if open_time >= close_time:
                warnings.append("market_open_time must be before market_close_time")
        except ValueError as e:
            warnings.append(f"Invalid market hours format: {e}")

        # Validate trading window
        try:
            from datetime import datetime

            no_before = datetime.strptime(self.no_trading_before, "%H:%M")
            no_after = datetime.strptime(self.no_trading_after, "%H:%M")
            if no_before >= no_after:
                warnings.append("no_trading_before must be before no_trading_after")
        except ValueError as e:
            warnings.append(f"Invalid trading window format: {e}")

        # Telegram requires both token and chat_id
        if self.telegram_enabled:
            if self.telegram_bot_token and not self.telegram_chat_id:
                warnings.append("telegram_chat_id required when telegram_bot_token is set")
            if self.telegram_chat_id and not self.telegram_bot_token:
                warnings.append("telegram_bot_token required when telegram_chat_id is set")

        # Store warnings on the instance so tooling can surface them, and log them.
        self._config_warnings = warnings
        import logging

        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        if fatal:
            for error in fatal:
                logger.error(f"Configuration FAILED (fail-closed): {error}")
            raise ValueError(
                "Refusing to start: hazardous configuration detected "
                f"({len(fatal)} issue(s)):\n" + "\n".join(f"  - {e}" for e in fatal)
            )

        effective_execution_mode = resolve_effective_execution_mode(
            self.execution_mode,
            allow_live_orders=self.allow_live_orders,
            trading_mode=self.trading_mode,
            has_dhan_credentials=has_dhan_credentials,
        )
        real_orders = effective_execution_mode == "live"
        logger.info(
            "Configuration OK | trading_mode=%s requested_execution_mode=%s "
            "effective_execution_mode=%s allow_live_orders=%s REAL_ORDERS=%s",
            self.trading_mode,
            self.execution_mode,
            effective_execution_mode,
            self.allow_live_orders,
            "YES" if real_orders else "NO",
        )

        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    files = resolve_env_files()
    return Settings(_env_file=files)


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
