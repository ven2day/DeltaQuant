---
name: learning-namespace-isolation
description: Use when the task is to stop DeltaQuant's mock/synthetic/backtest trade outcomes from silently contaminating the same performance and lesson stores used for live-data paper-trading ranking, Kelly sizing, and lesson injection. Trigger on "mock trades are polluting learning", "namespace performance records by environment", "isolate simulation from live-data learning", or references to H-9 of the Quant-Risk Review or C-6 of the Institutional Audit.
---

You are the model-risk/data-governance engineer for DeltaQuant (`/root/DeltaQuant`). Read
`CLAUDE.md` first, especially the "Memory & learning loop" section — the mechanism you're fixing
is `src/memory/` and `PerformanceTracker`.

## Grounding finding (already verified, do not re-derive)

`DeltaQuant-Quant-Risk-Review.md`, **H-9**, restated with more detail as
`DeltaQuant-Paper-Only-Institutional-Audit.md` **C-6**:

- `strategy_performance_records` stores strategy, regime, P&L, symbol, winner flag, and
  timestamp — but **no** data source, execution mode, simulation-run ID, model version, or
  dataset identifier (`src/memory/performance_tracker.py`, around lines 47–59 at audit time).
  `PerformanceTracker.record_trade()` accepts no environment field either (around lines
  135–179).
- The `agent_memory` lesson schema (`src/memory/database.py`, around lines 39–82 at audit time)
  has contextual trade fields but no mock/live namespace.
- These records feed **later** strategy win rates, ranking/Kelly inputs, and lesson injection.
  So a weekend trade produced by unrealistic random synthetic bars and instant simulated fills
  can silently influence a subsequent **real-data** paper-trading decision — self-reinforcing
  model risk with no visible boundary.

## Your mission

1. Add a required environment/namespace field (at minimum distinguishing `simulation` /
   `historical_replay` / `paper_live_data` — extend to `backtest` and `shadow` if those write to
   the same tables) to every record written by `PerformanceTracker.record_trade()` and to the
   `agent_memory` lesson schema. Also carry a `run_id` (or reuse whatever run-manifest concept
   `simulator-coherence` introduces, if that work has landed — coordinate/check first) so
   records from one simulated run are traceable as a group.
2. Migrate existing schema (Postgres) with a proper migration, defaulting historical rows to a
   clearly-labeled `unknown`/`legacy` namespace rather than silently assuming they were real.
3. Update every **read** path — historical win-rate lookups feeding `signal_ranking.py`
   `rank_signals()`, Kelly-input win rate in `src/market/sizing.py`, and lesson retrieval in
   `src/memory/injection.py` — to filter to the appropriate namespace(s) for the *current*
   run's environment. Production/live-data inference must query only the approved namespace(s);
   it must not be possible for a simulation run's win rates to silently leak into a live-data
   session's Kelly sizing or ranking just because both share a database.
4. Resetting the paper wallet (a fresh ₹X start) must not implicitly carry forward contaminated
   priors from a different environment — verify wallet reset doesn't interact with this in a way
   that blurs the boundary.
5. Update `CLAUDE.md`'s "Memory & learning loop" section once the schema changes, since it
   currently doesn't mention environment isolation at all.

## Code map (starting points — re-locate via grep, audit line numbers will have drifted)

- `src/memory/performance_tracker.py` — `PerformanceTracker.record_trade()`,
  `strategy_performance_records` table
- `src/memory/database.py` — `AgentMemoryDB`, `agent_memory` table, `decayed_score()`
- `src/memory/analyzer.py`, `src/memory/classifier.py` — where `TradeOutcome` is built and
  classified into lessons
- `src/memory/injection.py` — where top-N lessons are retrieved and injected into
  `TradingState.memory_lessons`
- `src/market/signal_ranking.py` — `rank_signals()`, historical win-rate consumption
- `src/market/sizing.py` — Kelly win-rate input
- `src/config/settings.py` — wherever the current run's mode (`market_data_source`,
  `execution_mode`, synthetic-history flags) is available, to derive the namespace tag
  consistently rather than inventing a parallel concept

## Non-negotiable invariants (from `CLAUDE.md`)

- The memory/learning loop stays failure-isolated — classification/storage errors must never
  interrupt trading or exit handling. Your namespace field is additive to that pattern, not a
  new way for learning code to raise into the trading loop.
- `AgentMemoryDB`'s SQLite fallback (when Postgres init fails) still needs the same schema field
  — don't let the fallback path silently drop the namespace.
- Don't change the decay formula (`decayed_score()` in `database.py`) — that's out of scope
  here; you're adding a filter dimension, not touching relevance scoring math.

## Acceptance criteria (from both audits)

- Mock/backtest/replay evidence cannot influence live-data paper decisions without an explicit,
  visible promotion step.
- Every performance/lesson record carries an environment/run identifier.
- A test proves that seeding the performance table with `simulation`-namespaced losing trades
  does **not** move `rank_signals()`'s output or Kelly sizing for a `paper_live_data`-namespaced
  query.

## Explicitly out of scope for you

- The simulator coherence fix itself (`simulator-coherence`) — you're isolating whatever the
  simulator produces, not fixing its internal realism.
- The trade-lifecycle ledger and execution-gateway work — unrelated call sites.
- ML prediction validation/leakage fixes (`ml-strategy-governance`) — related governance theme,
  different code path (`src/agents/prediction.py`).

Run `uv run --extra dev pytest tests/` after each change (expect to touch or add tests near
`tests/` memory/performance coverage), and keep `mypy src` clean (strict mode).
