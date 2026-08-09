# System Design Decisions

## Deterministic Screening Before LLMs

The system uses LLMs for bounded judgement, not for broad market scanning. A
full universe may contain hundreds of names, while an LLM review is useful only
when supplied with concise, comparable evidence.

This is why DeltaQuant uses a two-stage architecture:

1. Deterministic screen: all configured symbols, real Dhan data, repeatable
   scoring, sector diversification, and explicit rejection reasons.
2. LLM review: a small shortlist with multi-timeframe signals, market context,
   news, sentiment, prediction, memory lessons, and portfolio state.

The design prevents the common failure mode in which the largest intraday mover
receives repeated attention while the rest of the investment universe is never
reviewed.

## LLMs Are Not Execution Authorities

The risk engine is deterministic. It applies position, daily-loss, drawdown,
market-hours, circuit-band, sector-exposure, and risk/reward rules. Long-only
mode is enforced again at the paper engine, where a SELL cannot exceed the
owned quantity.

An LLM response can be unavailable, rate-limited, malformed, or inconsistent.
When that happens, the graph records fallback errors and the runtime blocks new
entries. A degraded cycle is allowed to update telemetry and explain its state;
it is not allowed to create a position.

## Data Integrity

The operating data source is DhanHQ. The system uses:

- Batched REST quotes for broad coverage
- A WebSocket listener when coverage and credentials permit it
- Dhan historical candles for daily and intraday analysis
- Disk and in-memory caches to respect Dhan account-wide rate limits

Synthetic price history is not permitted in the live paper workflow. A data
failure reduces the reviewable universe rather than inventing prices. This is a
deliberate availability-versus-integrity decision.

## Portfolio Construction

The screener prefers a modest pullback in an established trend rather than a
large absolute move. It applies a sector cap before agents run. The risk layer
then applies portfolio constraints using actual durable positions, not only the
current process memory.

Sector caps are a proxy for diversification, not the real measure — two names
in different sectors can still move together, and two names in the same sector
can be only weakly correlated. The risk layer also checks actual pairwise
daily-return correlation between a candidate and each open position
(`risk_compliance.compute_return_correlations`, over
`PAIRWISE_CORRELATION_LOOKBACK_DAYS` days of daily closes computed by the live
loop) and warns when it exceeds `MAX_PAIRWISE_CORRELATION`. A pair with too
little overlapping history is skipped rather than treated as uncorrelated.

This supports an investing-oriented workflow better than treating all symbols as
independent short-term trades.

## Durability And Reconciliation

Paper execution is persisted. On restart, the engine restores its wallet,
orders, open positions, realized P&L, and trade counts. The web dashboard is
updated from that engine state. Signal history and trade history are intentionally
different views:

- **Signal history** explains what was generated, validated, rejected, or
  blocked.
- **Trade history** reconstructs executed paper fills and their net P&L.

This distinction makes it possible to explain a balance change and determine
whether it represents an open holding, a realized result, or charges.

## Rate Limits And Cost Controls

Dhan data APIs and Groq calls are separately rate-limited. The design avoids
large bursts by caching historical data, batching quotes, capping the shortlist,
and pacing agent cycles with `TRADING_CYCLE_SECONDS`.

FinOps tracks LLM tokens and estimated spend. A hard budget stops new agent
cycles, while exits and safety monitoring continue. Langfuse tracing is
optional; an observability outage must not affect market data or execution
safety.

## Offline Research Direction

The implemented offline research loop extends the architecture without placing
LLM formula generation in the live execution path:

```mermaid
flowchart LR
    A[Research hypothesis] --> B[Groq Signal Agent]
    B --> C[Allowlisted local AST compiler]
    C --> D[Cross-sectional Rank IC evaluation]
    D -->|Pass IC and p-value| E[Portable accepted JSON]
    D -->|Fail| F[Groq optimization feedback]
    F --> B
    E --> G[Load-time quality gate]
    G --> H[Bounded live confidence tilt]
    H --> I[Paper validation]
```

This adapts NVIDIA's Apache-2.0 quantitative-signal-discovery-agent architecture
to Groq and Dhan. The upstream LLM Code Agent is replaced by a deterministic
AST compiler: generated Python is never executed, and formulas cannot import,
access attributes/files, or call anything outside the operator allowlist.
Accepted formulas still require subsequent paper and walk-forward validation.

**Isolation from risk and sizing.** The live confidence tilt this loop produces
only ever adjusts `estimated_win_probability` inside the local ranking step
(`src/market/signal_ranking.py`) — it decides which signals are worth showing
the agent graph, nothing more. Neither `risk_compliance.py`'s checks nor
`calculate_position_size` read any signal-discovery field; a discovered
formula has no path to change a stop-loss, a position size, or a risk-gate
outcome. This is a hard boundary, not a convention — treat any change that
threads a discovery-derived value into either of those two as a regression.

## Scalp Horizon: A Separate Pipeline, Not A Flag

Scalp (5m/15m) and swing (15m-4h) candidates need different evidence. Entry timing quality
and multi-timeframe alignment matter far more on a 5-minute chart than a multi-hour one,
while a single strategy's long-run historical win rate matters less — scalp regimes shift
faster and sample sizes are smaller. Rather than add a `horizon` parameter to the existing
`rank_signals()` formula and risk thresholds, scalping gets its own ranker
(`scalp_ranking.py`), its own deterministic entry-timing evaluator (`entry_quality.py`), and
its own weighted formula — while reusing the *same* LangGraph nodes, the *same*
`risk_compliance` checks, and the *same* execution/journal/exit-manager stack swing already
relies on. A candidate that reaches the agent graph looks like an ordinary signal dict either
way; only a `trade_horizon` tag distinguishes which ruleset governs it.

This has a direct consequence for the H-8 admission gate: **the registry's grain had to grow
with it.** Before this feature, `StrategyVersion` was keyed on strategy name alone (plus an
unused `approved_regimes` field) — one `trend_following` artifact covered every timeframe
indiscriminately, and (a separate, pre-existing gap this work also fixed) every artifact was
honestly validated only on daily bars regardless of what interval a caller nominally
requested. The registry now carries `timeframe` and `trade_horizon` alongside `strategy_name`
and `regime`. Every artifact registered before this field existed defaults to
`trade_horizon="SWING"` with an unpinned timeframe — it keeps admitting exactly what it
always did, and structurally cannot match a `SCALP` request. The strategy-admission check
(H-8, `risk_compliance.py`'s check #14) and `strategy_selection.py`'s own gate both now read
the signal's `timeframe`/`trade_horizon` before consulting the registry, so a strategy proven
only on swing/daily data can never silently admit a 5m scalp trade just because the strategy
name matches.

**Regime pre-filtering is a cost optimization, never an admission decision.** A deterministic
strategy/regime compatibility table filters obviously-mismatched candidates (e.g. a
mean-reversion strategy in a strongly trending regime) before spending an LLM call reviewing
them. That module has no import of `StrategyRegistry` at all — it structurally cannot be
"optimized" into replacing H-8, because it has no path to the registry to begin with. A
regime-compatible candidate still has to independently clear admission and every
`risk_compliance` check exactly like any other signal.
