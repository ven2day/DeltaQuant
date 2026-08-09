# Component Reference

## `src/market`

| Module | Responsibility |
| --- | --- |
| `manager.py` | Unified quote lifecycle and Dhan REST/WebSocket coordination. |
| `dhan_quotes_feed.py` | Batched Dhan quote requests, previous-close handling, and rate limiting. |
| `dhan_historical_feed.py` | Dhan OHLCV retrieval and interval resampling. |
| `dhan_auth.py` | PIN+TOTP auto-login: generates and caches DhanHQ access tokens. |
| `dhan_instruments.py` | Resolves NSE symbol -> Dhan security ID from the public instrument master CSV. |
| `historical_feed.py` | Thin Dhan-only historical-candle facade, gated by `ENABLE_DHAN_HISTORICAL_DATA`. |
| `websocket_feed.py` | `DhanWebSocketFeed`: live WebSocket quote subscription during market hours. |
| `live_data.py` | Direct DhanHQ REST quote fetching (`LiveMarketData`), an alternate quote path. |
| `data_feed.py` | Async WebSocket ingestion queue (`MarketDataFeed`); exported from the package, not wired into the live script's main path. |
| `simulated_data.py` | Synthetic quote generator used when Dhan is unconfigured or unreachable. |
| `sizing.py` | Position sizing (risk-per-trade + stop distance, Kelly-adjusted) and portfolio heat. |
| `history_manager.py` | Real OHLCV cache, settled-bar access, and timeframe cache management. |
| `indicators.py` | Indicator calculation and serialized `IndicatorResult`. |
| `signals.py` | Deterministic strategy signals (momentum, mean-reversion, breakout, trend-following) generated per symbol per configured timeframe. |
| `signal_ranking.py` | Blends technical confidence, ML direction agreement, sample-smoothed historical win rate, and the signal-discovery probability tilt into `expected_r`; ranks and sector-caps the shortlist sent to the agent graph. |
| `sectors.py` | Built-in and CSV-provided sector labels. |
| `sector_movers.py` | Background sector mover scan. |
| `scalping_screener.py` | Background swing-pattern scan; informational, separate from investment entries. |
| `stock_discovery.py` | Loads the configured CSV universe (`discovery.universe`); the news/mover-based `discover()` narrowing is no longer called from the live loop, kept for `scripts/backfill_signal_history.py` and `scripts/validate_strategy.py`. |
| `price_geometry.py` | Shared EMA/session-VWAP/ATR%/wick-ratio/zigzag-pivot helpers used by the scalp entry-quality evaluator, kept separate from `candidate_policy.py` so the swing evaluation path is never touched. |
| `signal_consolidation.py` | **Scalp.** Merges multiple strategies agreeing on the same symbol+timeframe+direction into one confidence-boosted `ConsolidatedSignal`, gated by `ENABLE_SIGNAL_CONSOLIDATION`. |
| `assessment_matrix.py` | **Scalp.** Builds one BUY/WAIT/REJECT `TimeframeAssessment` per symbol/timeframe, using that timeframe's own locally-inferred regime (not one label collapsed across the cycle). |
| `regime_compatibility.py` | **Scalp.** Deterministic strategy/regime compatibility table + pre-LLM cost filter — never a substitute for the H-8 admission gate. |
| `entry_quality.py` | **Scalp.** `EntryQualityEvaluator` — deterministic VWAP/EMA9/ATR-extension/swing-support-resistance/breakout-retest/volume/wick checks; returns ENTER_NOW/WAIT_PULLBACK/WAIT_BREAKOUT/REJECT plus a preferred entry range. |
| `scalp_confirmation.py` | **Scalp.** Multi-timeframe confirmation across the 5m=execution/15m=primary/30m=directional/1h=context/4h=optional-macro roles. |
| `scalp_opportunity.py` | **Scalp.** `ScalpOpportunity` — the canonical domain object carried scanner → ranker → agents → UI → execution. |
| `scalp_ranking.py` | **Scalp.** A separate weighted ranking formula from `signal_ranking.py`'s (entry quality, MTF alignment, volume, regime, `scalp::`-namespaced historical expectancy, ML probability). |
| `scalp_scan.py` | **Scalp.** Orchestrates one cycle's full scan → consolidate → matrix → confirm → entry-quality → rank pass; extracted standalone (not inline in `run_live_trading.py`) so it's unit-testable without the whole live-session object graph. |

## `src/signal_discovery`

Automated, LLM-proposed alpha research — entirely offline from the live risk/execution
path. See [System Design Decisions](4_System_Design.md) for how it's isolated.

| Module | Responsibility |
| --- | --- |
| `workflow.py` | Orchestrates the Signal Agent (Groq formula proposals) → Code Agent → Eval Agent loop, with optimization feedback on rejected iterations. |
| `operators.py` | The allowlisted OHLCV operator catalogue and `FormulaCompiler` — parses formulas as a restricted AST (never `exec`/`eval`) and evaluates them with a hand-written tree-walking interpreter. |
| `evaluator.py` | `RankICEvaluator`: cross-sectional Spearman Rank-IC of a factor against forward returns, with a t-test p-value. |
| `models.py` | Shared dataclasses (proposed signal, evaluation result, acceptance record). |
| `store.py` | Atomic on-disk persistence (`data/discovered_signals/<timeframe>/`); re-validates formulas and thresholds again at load time, not just at save time. |
| `live.py` | `DiscoveredSignalScorer`: re-evaluates accepted formulas against the live cross-sectional Dhan panel and produces a bounded per-symbol probability tilt. |
| `scheduler.py` | `AutomaticDiscoveryScheduler`: optional 24h-default background research loop, decoupled from the trading-cycle timer. |

## `src/agents`

| Module | Responsibility |
| --- | --- |
| `graph.py` | Compiles and runs the LangGraph sequence and support-agent wrapper. |
| `state.py` | Shared `TradingState` contract. |
| `market_regime.py` | LLM market-regime classification. |
| `strategy_selection.py` | LLM strategy selection for the current regime. |
| `signal_validation.py` | LLM technical-signal review. |
| `risk_compliance.py` | Deterministic final risk and compliance gate. |
| `news_analyst.py` | Google News RSS sentiment. |
| `sentiment.py` | Market mood calculation. |
| `prediction.py` | Scikit-learn direction estimate for a bounded signal set. |

## `src/execution`

| Module | Responsibility |
| --- | --- |
| `paper_engine.py` | Durable cost-aware paper wallet, fills, positions, and trade reconstruction. |
| `service.py` | Mode resolution, idempotency, and async execution submission. |
| `adapter.py` | Local and Dhan execution adapters. |
| `exit_manager.py` | Stop, target, trailing, partial, time, and regime exits. |
| `journal.py` | Durable trade journal. |
| `signal_log.py` | Signal lifecycle audit history. |
| `costs.py` | Slippage and configurable paper charges. |
| `live_executor.py` | Broker lifecycle and reconciliation for explicitly enabled live modes. |

## Supporting Packages

| Package | Responsibility |
| --- | --- |
| `config` | Pydantic settings and configuration validation. |
| `dashboard` | Session state and durable paper-account synchronization. |
| `memory` | Performance tracking, lesson storage, injection, and post-trade learning. |
| `risk` | Daily loss, drawdown, and circuit guards. |
| `finops` | Token/cost accounting, budgets, and alerts. |
| `profit` | Advisory-only monthly profit-target planning; never feeds sizing or relaxes risk. |
| `backtesting` | Historical strategy evaluation, cost-adjusted results, and walk-forward validation. |
| `db` | Shared SQLAlchemy engine/session factory used by the Postgres-backed stores. |
| `api` | Fast, config-based health checks for the System Status page. |
| `observability` | Optional Langfuse tracing wrappers and LangGraph callback adapter. |
| `webui` | FastAPI state, history, health routes, and WebSocket hub. |
| `notifications` | Telegram delivery. |
| `utils` | Rate limits, cache, IST time, errors, events, and serialization. |

## Entrypoints

| Path | Use |
| --- | --- |
| `scripts/setup.py` | Guided one-command setup: creates `.env`, checks required keys, prints a readiness checklist. Run this first. |
| `scripts/run_live_trading.py` | Main real-data local-paper workflow. |
| `scripts/check_config.py` | Configuration inspection. |
| `scripts/test_dhan_connection.py` | Dhan connectivity validation. |
| `scripts/diagnose_risk.py` | Risk diagnostics. |
| `scripts/validate_strategy.py` | Out-of-sample walk-forward validation of the live signal logic, net of realistic costs. Run before risking capital. |
| `scripts/backfill_signal_history.py` | Replays historical candles into an isolated, tagged signal log for analysis; never touches the live risk/execution path. |
| `scripts/discover_signals.py` | On-demand run of the Signal/Code/Eval Agent alpha-research loop for a given research prompt; the same loop `AutomaticDiscoveryScheduler` runs on a timer when `SIGNAL_DISCOVERY_AUTO_RUN=true`. |
| `scripts/run_trading.py` | Legacy entry point predating `run_live_trading.py`; not the maintained workflow. |
| `web/` | Next.js dashboard frontend. |

## Persistence

PostgreSQL stores agent memory, daily risk state, trade journal data, signal
history, and paper engine state. Historical Dhan OHLCV cache files live under
`data/history_cache/`.
