# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

₹DeltaQuant is an agentic paper-trading system for the Indian NSE market. A LangGraph
pipeline of LLM-backed agents (via Groq) classifies the market regime, picks strategies,
validates signals, and applies deterministic risk rules. Market data (quotes and
historical candles) comes from DhanHQ — a Dhan account is required for real prices;
without one, the live loop automatically runs on simulated data instead of failing
(YFinance support was removed — see MarketDataManager). Execution is a local virtual
wallet, and the Groq free tier powers the LLM agents. PostgreSQL (memory) is optional.
It is educational/paper-trading only.

## Commands

Dependency management is via [`uv`](https://github.com/astral-sh/uv). The distribution name is `trading-agent`; the import
package is `src` (all internal imports are `from src.<module> import ...`).

```bash
uv sync                         # install runtime deps
uv sync --extra dev             # install dev deps (pytest, ruff, mypy)

uv run python scripts/check_config.py       # validate .env / settings (run this first)
uv run python scripts/run_live_trading.py   # MAIN entry point: live/sim dashboard + paper execution
uv run python src/backtesting/engine.py     # run a backtest

# pytest/ruff/mypy live in the `dev` optional group — pass --extra dev (or `uv sync --extra dev` once).
uv run --extra dev pytest                     # full test suite (210 tests)
uv run --extra dev pytest tests/test_agents.py   # one file
uv run --extra dev pytest tests/test_agents.py::TestRiskCompliance::test_risk_compliance_no_signals   # one test
uv run --extra dev pytest --cov=src           # with coverage

uv run --extra dev ruff check .   # lint (line-length 100, rules E,F,I,N,W,UP)
uv run --extra dev ruff format .  # format
uv run --extra dev mypy src       # type-check (strict mode is enabled)
```

`pytest` is configured with `asyncio_mode = "auto"`, so `async def test_*` functions run
without an explicit `@pytest.mark.asyncio` decorator.

## Architecture

### Agent pipeline (the core)

The system is a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` built in [src/agents/graph.py](src/agents/graph.py).
A single `TradingState` (a `TypedDict` defined in [src/agents/state.py](src/agents/state.py))
flows through every node; each node returns a **partial dict** that LangGraph merges into state.
The pipeline:

1. **support_agents** — runs news, sentiment, and prediction agents to enrich state. All
   failures here are non-fatal (caught and logged); they only add context.
2. **market_regime** — LLM classifies regime (`trending_up/down`, `ranging`, `volatile`).
   Conditional edge: if `regime_confidence < 0.3` **or** the kill switch fires, the graph ends.
3. **strategy_selection** — picks active strategies for the regime.
4. **signal_validation** — filters raw signals. Conditional edge: if no signals survive, the graph ends.
5. **risk_compliance** — a **deterministic rules engine** (not an LLM) that does final approval,
   position sizing, and enforces limits. Populates `approved_trades` / `risk_rejected`.

To add an agent: write a `*_node(state) -> dict` function, register it with
`workflow.add_node(...)`, and wire edges in `create_trading_graph()`. Conditional routing
lives in `should_continue_after_*` predicate functions.

### LLM agent conventions

Every LLM node (see [src/agents/market_regime.py](src/agents/market_regime.py) as the
reference implementation) follows the same resilience pattern — **preserve it when editing or adding agents**:

- Acquire the shared rate limiter (`get_groq_limiter`) and circuit breaker
  (`get_groq_circuit_breaker`) before calling the LLM.
- Try `settings.groq_model_primary`, then fall back to `groq_model_fallback` on rate-limit (429) errors.
- On **any** failure (circuit open, rate limit, parse error), return a deterministic
  `_fallback_*` result instead of raising. The graph must never crash on a bad LLM call.
- LLM output is JSON; parsing strips ```` ```json ```` / ```` ``` ```` fences and clamps/validates fields.

**Support-agent state contracts.** The support agents enrich `TradingState` with keys the
regime/validation agents read; the *types must match* or the enrichment is silently dropped
(the consumer raises `TypeError`, which the agent's broad `except` swallows → it falls back
without the context). The canonical contracts (declared in [state.py](src/agents/state.py)):
`news_sentiment` is a **dict** `{"avg_sentiment": float}` (not a bare float), `market_mood` is
the full `SentimentSignal.to_dict()` dict (read `market_mood["mood_index"]`, not `market_mood`
itself), `news_headlines` is a list of `{"title","sentiment"}`, and `prediction_signals` is a
list of `PredictionSignal.to_dict()`. The prediction node sources from raw `signals` (populated
when support agents run), **not** `validated_signals` (still empty at that stage). When adding a
consumer, read with `isinstance`/`.get(...)` guards so a stray type never crashes the node.

### Configuration

All config is centralized in [src/config/settings.py](src/config/settings.py): a
pydantic-settings `Settings` model loaded from `.env`. Access it **only** through the cached
`get_settings()`; use `reload_settings()` to clear the cache. Secrets are `SecretStr` —
read them with `.get_secret_value()`. The `@model_validator` (`validate_configuration`) is
two-tier: benign cross-field inconsistencies (risk-param sanity, market-hours ordering,
Telegram config) are collected as warnings and logged, same as before; but the narrow,
explicit hazardous-combination matrix (trading_mode/execution_mode/allow_live_orders/Dhan
creds that could reach a real broker order, plus `MAX_QUOTE_STALENESS_SECONDS=0` on a
live-quote profile) **raises `ValueError` and fails startup** instead of degrading — see
C-2/M-5/H-10 in `DeltaQuant-Quant-Risk-Review.md`. `resolve_effective_execution_mode()`
(same module) is the single source of truth for which execution mode is actually in effect;
`scripts/check_config.py` and `ExecutionService._resolve_mode()` both call it rather than
re-deriving the conjunction.

Key switches: `market_data_source` (`dhan` only — YFinance was removed), `execution_mode`
(`local_paper`|`shadow`|`dhan_paper`|`live` — `dhan_paper` is simulate-only and can **never**
reach a live route, see below), `trading_mode` (`paper`|`live`), `enable_news_analysis`.

### Pluggable data & execution layers

- **Market data** — [src/market/manager.py](src/market/manager.py) `MarketDataManager`
  auto-selects WebSocket (live Dhan) or simulated data based on `is_market_open()` and
  connection availability. Indicators (`ta` library) are computed in
  `indicators.py`; `signals.py` `SignalEngine` turns them into signals; `stock_discovery.py`
  dynamically finds symbols to trade (no hardcoded watchlist).
  - **Data-ingestion gotchas (do not regress):** `HistoryManager.append_quote` treats a
    quote's `volume` as the **cumulative daily total** (keeps the max, never sums — summing
    inflated volume every cycle). `get_history(..., include_forming=False)` drops the
    still-forming current-day bar; the live loop computes indicators on settled bars by
    default (`signals_exclude_forming_bar`) to avoid intra-bar repainting/look-ahead.
    Indicator floats are NaN/inf-sanitized to `None` (`_safe_float`) so warm-up values never
    reach signals or the agent JSON as `NaN`. Indicator results are memoized via
    `get_indicator_cache()` (keyed by symbol/last-bar/close). The loop skips **new entries**
    when the freshest quote is older than `max_quote_staleness_seconds` (exits still run).
  - **Decision quality (do not regress):** signal **confidence is evidence-based** —
    `SignalEngine._directional_confidence` blends the strategy's base with how many independent
    indicators (RSI/MACD/DI/price-vs-MA) agree with the direction, not a hardcoded constant.
    The live loop sizes entries with the real `PositionSizer`
    ([sizing.py](src/market/sizing.py)) off `risk_per_trade` + the stop distance (Kelly when a
    strategy win-rate exists), **not** a flat % of cash. Agent prompts ask for *calibrated*
    confidence and to weigh the news/mood/ML enrichment.
- **Execution** — [src/execution/service.py](src/execution/service.py) `ExecutionService` is
  the single mode-switched entry point the live loop uses to place orders. It wraps
  `paper_engine.py` (and, later, the broker in [adapter.py](src/execution/adapter.py):
  `ExecutionAdapter` for DhanHQ, imported lazily). `exit_manager.py` handles trailing stops /
  time exits / partial profits; `journal.py` logs trade history.
  - **Execution safety (do not regress):** every `submit(...)` carries an `idempotency_key`;
    a repeat returns a `DUPLICATE` instead of placing again (an `IdempotencyStore` persists
    keys so a restart can't replay orders). `execution_mode` adds **`shadow`** (mirror the
    live decision/sizing, simulate the fill, send nothing). `dhan_paper` is simulate-only and
    **always** resolves to SHADOW — there is no verified Dhan sandbox endpoint, so it can
    never reach a live route regardless of `allow_live_orders` or credentials (C-2 fix; see
    `resolve_effective_execution_mode` in [settings.py](src/config/settings.py)). A `live`
    request **never silently downgrades**: without `trading_mode=live` **and**
    `allow_live_orders=True` (default-off master gate) **and** valid Dhan creds, it resolves to
    SHADOW with a loud warning/error naming which condition failed. Live submission goes
    through `submit_async` → `LiveBrokerExecutor` ([live_executor.py](src/execution/live_executor.py)),
    which submits then **polls `get_order_status` to a terminal fill** (never assumes PLACED ==
    filled); `reconcile_positions` checks local vs broker positions at startup (broker = source
    of truth). `ExecutionService.real_orders_active` is the one place to check whether a given
    instance can actually place a real order (effective mode LIVE + broker_executor attached).
  - **Paper engine realism (do not regress):** `LocalPaperEngine` fills through a
    [`CostModel`](src/execution/costs.py) (slippage + NSE-style brokerage/STT/GST, all
    configurable via `paper_*` settings; pass `CostModel.zero()` for ideal fills in unit
    tests). `place_order` does proper **long/short/partial** accounting: an opposite-side
    order closes/covers (FIFO) and only the remainder opens a new position; realized P&L is
    **net of both legs' charges**, and opening commits capital (no cash leak on shorts).
    State is now Postgres-backed (`PaperWalletRecord`/`PaperPositionRecord`/`PaperOrderRecord`
    SQLAlchemy models via [src/db/base.py](src/db/base.py)) — no local JSON file, no
    temp-file/`os.replace` atomicity, no `.corrupt-*` quarantine; failures just log and roll
    back the session.

### Local signal ranking (replaces the old universe screener)

`src/market/universe_screener.py` (a deterministic pullback/trend/RSI filter) was **removed —
do not resurrect it or re-add the import**. The live loop now runs `SignalEngine` on every
symbol × every `signal_timeframes` entry unconditionally, then [signal_ranking.py](src/market/signal_ranking.py)
`rank_signals()` blends technical confidence, `PredictionAgent` ML direction agreement,
sample-smoothed historical win rate per strategy/regime, and `expected_r = p*RR - (1-p)` into
a rank score; `select_diversified_signals()` then applies the sector cap and
`max_active_stocks`/`llm_review_max_symbols` truncation (skipped entirely when
`llm_review_all_signals=true`). This is the only shortlisting step before the agent graph.

### Quantitative signal discovery (offline research, isolated from execution)

[src/signal_discovery/](src/signal_discovery/) is an LLM-driven alpha-research loop, gated by
`signal_discovery_enabled` and optionally run on a timer via `AutomaticDiscoveryScheduler`
(`signal_discovery_auto_run`, default every 24h) or on demand via `scripts/discover_signals.py`.
A Groq **Signal Agent** proposes formulas over OHLCV using a fixed operator allowlist (`Rank`,
`TS_Delta`, `EMA`, etc. — see `operators.py`); the **Code Agent** (`FormulaCompiler.compile`)
parses them with `ast.parse(mode="eval")` and an allowlist-not-blocklist AST walk — no
`exec`/`eval` of generated code, no attribute/subscript/import access, unknown node types raise.
The **Eval Agent** (`RankICEvaluator`) computes cross-sectional Spearman Rank-IC against forward
returns; only formulas clearing both `signal_discovery_ic_threshold` and a **Bonferroni-corrected**
p-value bar are persisted (`DiscoveredSignalStore`, atomic write). `SignalDiscoveryWorkflow`
tests up to `num_signals * max_iterations` formulas per run against the same panel in pursuit of
one "hit" — accepting any of them at the flat nominal `signal_discovery_p_value_threshold` is the
classic multiple-comparisons trap, so `_corrected_p_threshold(num_tests)` divides the nominal
threshold by the running count of formulas actually evaluated that run (compile failures don't
count) before `_accepted()` checks it; the bar tightens as more candidates are tried instead of
loosening the odds of a false "significant" IC. The correction only affects acceptance — the
persisted `p_value` is the raw, uncorrected statistic, and load-time re-validation (below) still
checks it against the nominal (looser) threshold, which an accepted signal always clears. Formulas
are then **re-validated against current thresholds again at load time**, so tightening config
prunes stale accepted signals without deleting files. Live application (`DiscoveredSignalScorer`)
re-evaluates accepted formulas on the current cross-sectional panel and produces a small
`±signal_discovery_max_probability_tilt` (default 0.08, hard-clamped ≤0.20) added to a signal's
`estimated_win_probability` in `rank_signals()`.
  - **Isolation invariant (do not regress):** this tilt has exactly one consumer —
    `signal_ranking.py`'s ranking/shortlisting. Neither `risk_compliance.py` nor
    `calculate_position_size` (`sizing.py`) reads any signal-discovery field. Never thread a
    discovery-derived value into either of those — it must never move a stop-loss, a position
    size, or a risk-gate decision.

### Memory & learning loop

[src/memory/](src/memory/) implements a learn-from-losses loop backed by PostgreSQL via
SQLAlchemy (`AgentMemoryDB`, `agent_memory` table). `analyzer.py` + `classifier.py` turn
losing trades into lessons with time-decay relevance scoring; `injection.py` feeds the
top-N lessons back into `TradingState.memory_lessons` for the next cycle. It targets the
PostgreSQL `DATABASE_URL` from settings, but `AgentMemoryDB._initialize_db()` silently
falls back to an in-memory SQLite database if that connection fails — so memory works
(non-persistently, lost on restart) even without Postgres.

The loop is **closed at runtime** (gated by `enable_learning`): on each full close,
`run_live_trading.py` builds a `TradeOutcome` (`analyzer.compute_outcome`), classifies it into
a lesson (`MistakeClassifier`) and stores it, and marks the lessons that were *active when the
position was opened* as successful/unsuccessful (`memory.feedback` helpers — all
failure-isolated; learning must never disrupt trading). `PerformanceTracker` persists its
trade history to Postgres (`strategy_performance_records` table, no local/file fallback), so
real win-rates survive restarts instead of resetting to the hardcoded priors. The lesson-decay
formula is unified: `database.py`'s `decayed_score()` is the single source of truth, and
`scheduler.py` imports and calls that same function rather than maintaining its own.

### FinOps (cost tracking & alerts)

[src/finops/](src/finops/) accounts for LLM spend and raises operational alerts.
`cost_tracker.py` is **pure accounting** (no I/O, thread-safe), so it is safe to call from
sync agent nodes: each LLM agent calls `record_llm_response(agent, response, model=...)`
immediately after its `circuit_breaker.call(...)` (the helper never raises). It tracks
tokens + paid-tier-equivalent cost per agent and per **IST day** (rolls over via
`utils/market_time`), keyed off a Groq pricing table (free tier = $0; configurable). Budgets
come from settings (`daily_token_budget`, `daily_cost_budget_usd`, `finops_budget_soft_pct`;
`0` = unlimited). `alerts.py` (`AlertManager`, async) logs + best-effort Telegram, de-duped
per key per IST day — reuse it for drawdown/staleness/anomaly alerts too. `run_live_trading.py`
gates the agent pipeline on `is_over_hard_budget()` (a spend kill-switch: skips new LLM cycles
and entries while still running exits), surfaces today's spend on the dashboard, and fires
soft-budget + startup/shutdown Telegram messages.

### Profit-target goal engine

[src/profit/goal_engine.py](src/profit/goal_engine.py) turns a configured monthly return
target (`monthly_profit_target_pct` / `_amount`) into a **risk-bounded plan**: the daily
pace it implies, the win-rate it needs at the expected trade frequency, and the trade
frequency it needs at an assumed win-rate (using `risk_per_trade`, `goal_reward_risk_ratio`,
`daily_loss_limit`, `max_daily_trades`). `ProfitGoalEngine.build_plan(capital)` returns a
`GoalPlan`; `.evaluate(capital, realized_pnl)` reports on/off-pace vs straight-line pace.
**Guardrail (do not break this):** the engine is *advisory only* — it never feeds position
sizing and never relaxes risk. If a target is only reachable by exceeding per-trade risk,
the daily-loss limit, or the trade cap, the plan is `feasible=False` and the recommended
action is to *lower the target*, never to take more risk. `run_live_trading.py` logs the
plan at startup, shows pace on the dashboard, and alerts (via the FinOps `AlertManager`)
when off-pace or infeasible — always with the "do not increase risk" message.

### Backtesting & evaluation

[src/backtesting/](src/backtesting/) runs strategies on historical OHLCV. Prefer
`RealSignalStrategy` ([strategies.py](src/backtesting/strategies.py)) — it feeds the **real**
`calculate_indicators` + `SignalEngine` into the backtest, so results reflect live behaviour
(the other strategies are standalone re-implementations and will diverge). `BacktestResult`
includes an `expectancy` (per-trade edge); `compare_results(baseline, candidate)` produces a
before/after scorecard (return/win-rate/profit-factor/expectancy/Sharpe/drawdown deltas +
an `improved` flag) so a change can be *proven* to help before trusting it. The engine uses
strictly-prior bars (`history = data.iloc[:i]`), so no look-ahead. Pass `BacktestEngine(cost_model=...)`
to apply the audited `CostModel` (realistic slippage + NSE fees) instead of the flat
commission/slippage; `CostModel.zero()` for ideal fills.

**Edge validation (the go/no-go gate).** [walk_forward.py](src/backtesting/walk_forward.py)
(`run_walk_forward`, `aggregate_reports`, `edge_verdict`) evaluates a strategy on rolling
**out-of-sample** folds, **net of `CostModel` costs**, and returns a `VALIDATED`/`NOT VALIDATED`
verdict (needs ≥30 OOS trades, positive net expectancy *and* return, and >50% fold consistency).
`scripts/validate_strategy.py` runs it over a **fixed** universe (never the look-ahead
`StockDiscovery` output). Survivorship caveat: the universe is current-listed only — a true
production go/no-go needs a point-in-time, survivorship-free dataset (Bhavcopy/vendor) that
no free market-data API can supply. A green verdict is *necessary, not sufficient* (no
circuit/gap/liquidity modelling).

### Cross-cutting

`utils/` holds the shared `rate_limiter`, `circuit_breaker`, `cache` (TTL), `errors`
(custom exceptions like `RateLimitError`, `LLMResponseError`), `events`, and `market_time`
(IST helpers — see below). `observability/tracing.py` wires Langfuse. `dashboard/cli.py` is
the `rich` terminal UI. `notifications/telegram.py` sends trade alerts.

## Conventions & gotchas

- **Imports are `from src...`** everywhere. Most scripts in `scripts/` prepend the repo root
  to `sys.path` before importing (e.g. `run_live_trading.py`, `run_trading.py`); a few such as
  `check_config.py` omit it and rely on being run from the repo root. When adding a script,
  include the `sys.path` line so it works regardless of the working directory.
- **Graph nodes return partial state dicts**, never the full state; let LangGraph merge.
- **Never let an LLM/agent failure propagate** — return a fallback, matching existing agents.
- **Market-hour decisions use IST, not host-local time.** Use `src/utils/market_time.py`
  (`now_ist()`, `is_market_hours()`, `IST`) — never bare `datetime.now()` — for `is_market_open()`
  and the risk engine's trading-hours check. IST is a fixed UTC+05:30 offset (NSE has no DST), so
  this stays correct on a UTC cloud host / CI runner.
- **The kill switch must gate execution, not just the graph.** `check_kill_switch` ends the agent
  graph at the regime edge *and* is re-checked in `run_live_trading.py` before placing approved
  entries (exits still run, to flatten risk). Re-check it at any new order-submission site.
- Python target is **3.11** (`pyproject.toml`, ruff, mypy) even though the README says 3.12;
  prefer 3.11-compatible syntax. `mypy` runs in **strict** mode, so annotate new code fully.
  (Note: the repo currently carries pre-existing ruff/mypy debt; keep *new* code clean and avoid
  adding violations rather than boiling the ocean.)
- Timestamps in the DB use timezone-aware UTC (`datetime.now(UTC)`).
