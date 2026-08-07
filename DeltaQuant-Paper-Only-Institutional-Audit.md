# DeltaQuant Paper-Only Institutional Simulation Audit

**Review date:** 7 August 2026  
**Scope:** Read-only review of the current code, configuration, persisted-state contracts, tests, and observed weekend mock runtime  
**Operating assumption:** DeltaQuant will remain permanently paper-only and must never place a real-money order  
**Review perspective:** Hedge-fund risk management, quantitative research governance, and senior trading-platform engineering

## Executive conclusion

DeltaQuant is a capable **paper-trading orchestration prototype**, but it is **not yet an institutionally credible simulation platform**. Its strongest implemented features are the deterministic agent graph, local cost-aware wallet, several fail-safe LLM patterns, persisted daily-risk state, forming-bar protection, a walk-forward research utility, and a deliberately gated synthetic-history mode. Those are useful foundations.

The system's central weakness is synchronization. A trade is not represented by one durable, versioned object from market observation through signal, risk decision, fill, exit, journal, performance, and learning. Instead, the wallet, exit manager, journal, daily-risk tracker, performance tracker, and lesson store maintain partially overlapping state with different identifiers and independent commits. Several important controls are applied before the final executable price and quantity exist. The current weekend simulator further combines synthetic history and mock quotes from independent random price processes. Consequently, the platform can appear operational while the economic trade tested by risk is not the same trade filled and monitored by the paper engine.

The present weekend setup is safe from real money: it is configured as `TRADING_MODE=paper`, `EXECUTION_MODE=local_paper`, `ALLOW_LIVE_ORDERS=false`, with Dhan instruments, history, and quotes disabled. It is suitable for UI demonstrations and limited component plumbing. It is **not suitable for estimating strategy edge, validating portfolio risk, testing full-universe behavior, or training the learning loop**.

### Institutional verdict

| Capability | Verdict | Basis |
|---|---|---|
| Real-money safety in the current profile | **Pass** | Local paper mode, live-order gate off, and Dhan data calls disabled. |
| UI and orchestration demonstration | **Pass with limitations** | Cycles run and state is visible, but the data stream is artificial and incomplete. |
| Local wallet accounting mechanics | **Conditional pass** | FIFO/partial accounting and costs are implemented; lifecycle persistence and reconciliation remain weak. |
| Full-universe weekend simulation | **Fail** | The configured universe has 272 unique symbols, while the observed mock quote set covered only 19. |
| Risk-limit credibility | **Fail** | Batch approvals use a stale snapshot; aggregate exposure is not enforced; final sizing is not re-gated. |
| ML/AI validation credibility | **Fail** | Prediction alignment/validation is not cleanly out-of-sample; LLM decisions lack durable lineage and strict output constraints. |
| Strategy performance claims | **Fail** | Mock/live-data records are not namespaced, simulation is unrealistic, and offline validation is not a runtime admission gate. |
| Crash/restart recovery | **Fail** | Wallet, exits, journal, performance, and learning cannot be transactionally reconciled. |

## Review standard and severity

- **Critical:** Invalidates simulated risk/P&L or can leave the paper book materially inconsistent after ordinary operation or restart.
- **High:** Can breach an advertised limit, contaminate model decisions, or make research results unreliable.
- **Medium:** Material provenance, realism, governance, observability, or testing weakness.
- **Recommendation:** A target-state design control, not a claim that the mechanism is already implemented.

“Verified” below means the behavior is present in the reviewed implementation/configuration or was visible in the observed current runtime. It does not mean the failure has already occurred in a recorded trade.

## The implemented end-to-end path

The current runtime roughly follows this sequence:

1. Load a configured symbol CSV through `StockDiscovery`.
2. Load Dhan or synthetic history through `HistoryManager`.
3. Obtain quotes through `MarketDataManager`, which may select `SimulatedMarketData`.
4. Run `SignalEngine` strategies over each eligible symbol and configured timeframe.
5. Rank signals using technical confidence, the prediction agent, historical performance, and any accepted discovered-signal tilt.
6. Run the LangGraph support, market-regime, strategy-selection, signal-validation, and deterministic risk nodes.
7. Rebind each approved entry to the latest quote, recalculate quantity with `PositionSizer`, and submit to `ExecutionService`/`LocalPaperEngine`.
8. Register a separate `ManagedPosition` with `ExitManager`, write a journal record, and retain volatile mappings for journal and memory feedback.
9. On later cycles, check exits, place opposite-side paper orders, update dashboard/daily risk, record performance, and—on a full close—create learning feedback.

The architecture has the right conceptual stages. The issue is that several handoffs are neither atomic nor governed by one canonical trade identity and one immutable decision snapshot.

## Critical verified findings

### C-1 — Weekend history, quotes, and executable levels are different price processes

`HistoryManager._seed_synthetic()` creates 180 daily random-walk bars for every configured symbol, with an arbitrary starting price. `SimulatedMarketData` has a separate small hard-coded map of base prices and creates independent random quote movement. The two generators do not share an instrument state, seed, timeline, or price level.

Signals derive `entry_price`, `stop_loss`, and `target_price` from synthetic historical bars. Immediately before submission, the runtime replaces the entry with the current mock quote but retains the previously calculated stop and target. Risk compliance ran before this replacement. Position sizing therefore receives a mixture of unrelated price domains, and the exit manager later monitors that mixture against mock quotes.

**Impact:** Reward/risk, stop distance, quantity, exit triggers, P&L, and learned outcomes can be economically meaningless even when every component reports success.

**Required correction:** Use one deterministic, event-time instrument simulator to generate historical bars, quotes, spread, volume, and future ticks. Bind the final executable price before the final risk decision. Reject any order whose side/entry/stop/target/tick-size invariants fail, then calculate size and projected portfolio risk exactly once from that canonical snapshot.

### C-2 — Open-trade state is split across incompatible identifiers and persistence systems

The local wallet creates its own `POS-*` position identity; the exit manager is registered with an `ORD-*` order identity; the journal creates a `TRD-*` identity. The relation between these objects is partly held in process-local dictionaries such as `open_trades` and `active_lessons`.

The wallet is persisted in PostgreSQL, exit state is kept in JSON, the journal is a separate database write, and lesson/performance records have their own transactions. There is no durable canonical `trade_id` or transaction/outbox spanning an entry fill, managed exit registration, journal creation, and lesson attribution.

**Impact:** A process crash can leave a wallet position with no stop/target manager, a ghost exit record after a close, a missing journal trade, or a position whose active lessons can never be credited. Restart does not establish which store is authoritative.

**Required correction:** Introduce a durable trade lifecycle keyed by one immutable ID: `SIGNAL → RISK_APPROVED → ORDER_INTENT → FILLED → MANAGED → PARTIAL → CLOSED → RECONCILED`. Persist lifecycle state and an outbox atomically with the wallet mutation. Rebuild derived exit/learning state from that record on startup, and fail closed on unresolved mismatches.

### C-3 — Risk approves one snapshot, while execution submits another

`risk_compliance_node()` evaluates a signal's proposed percentage/levels against a static portfolio and daily-statistics snapshot. The main loop later replaces the entry price and recomputes actual shares using `PositionSizer`; it does not rerun the deterministic risk engine using the final fill-bound levels and quantity. The runtime also forces at least one share with `max(1, sizing.shares)`, so a zero-share sizing result can become an order.

**Impact:** The actual paper order may not satisfy the limit decision recorded by risk. A “do not trade” sizing result can be overridden, and actual loss-at-stop can differ from the approved exposure.

**Required correction:** Treat zero or invalid size as a hard rejection. After final quote binding, calculate the exact quantity and projected loss at stop, then run a final deterministic pre-trade check over the precise order and a versioned portfolio snapshot. The approved object must be identical to the submitted object.

### C-4 — Multi-signal batches can breach position, trade-count, and exposure constraints

Risk compliance loops through validated signals using one unchanged view of positions and daily statistics. Multiple candidates can each pass the same remaining-slot or remaining-trade calculation. The runtime then submits them sequentially without an atomic risk reservation between candidates.

In addition, the daily-risk `trades_count` is updated on exits, not on new entries/order intents. It therefore measures completed exit events rather than the daily entries the pre-trade control is intended to cap. `max_total_exposure_pct` is configured but not enforced, and aggregate loss-at-stop/portfolio heat is not a runtime gate.

**Impact:** One cycle can exceed the nominal maximum positions or daily trades. Total notional and total hard-stop loss can exceed stated portfolio limits even when every individual approval passes.

**Required correction:** Define “daily trade” precisely and increment/reserve it on accepted entry intent. Reserve cash, position slots, gross/net/sector exposure, and loss-at-stop atomically for each candidate. Recompute after each fill/rejection before considering the next candidate.

### C-5 — Partial exits and emergency closes corrupt outcome statistics

The normal exit path records each partial fill in the performance tracker as though it were a completed trade. A later final close records another trade, inflating count and potentially win rate. Learning and journal closure on full exit use the final leg's P&L rather than a durable cumulative lifecycle P&L.

The kill-switch flatten path updates the paper wallet, dashboard, and daily-risk state but does not consistently close the journal, record performance, classify learning, or credit active lessons before clearing managed-position state.

`ExitManager` also mutates `partial_taken` and remaining quantity after an earlier save in the exit-decision flow; a crash before the next save can reload pre-partial state.

**Impact:** Performance, regime win rates, Kelly inputs, journal P&L, and the self-learning loop can disagree about the same position. Forced-loss events—the most important risk events—may be absent from model feedback.

**Required correction:** Record fills as child events of one position lifecycle. Calculate cumulative net P&L only at lifecycle level, while retaining leg-level detail. Close every exit reason, including emergency flatten, through one reconciliation/finalization service. Make the finalization idempotent and transactional.

### C-6 — Simulation and live-data paper outcomes share the same learning namespace

Performance and lesson records do not carry a required `run_id`, data-source mode, simulator/dataset version, model/prompt version, or strategy version. Weekend mock outcomes can therefore feed historical win rates, signal ranking, Kelly sizing, and lesson injection in a later live-data paper session.

**Impact:** The platform can “learn” from its own unrealistic random generator and then present the contaminated estimate as observed strategy evidence.

**Required correction:** Isolate at least `simulation`, `historical_replay`, and `live_data_paper` into non-crossing namespaces. Every decision and outcome must include a run manifest, data lineage, strategy/model versions, and cost-model version. Only explicitly promoted evidence may influence another environment.

## High-severity verified findings

### H-1 — The configured universe is not the effective simulated universe

The configured CSV has 275 rows and 272 unique normalized symbols. It contains only a symbol column, with no instrument ID, effective dates, listing status, sector, tick size, liquidity, or point-in-time membership. The main runtime instantiates `StockDiscovery` and uses `.universe`; it does not call the implemented asynchronous `discover()` method. Dynamic news/mover discovery therefore exists in code but is not part of the principal cycle.

The mock quote generator supports only a small hard-coded large-cap map. The observed runtime had current quotes for 19 of the 272 unique symbols, even though synthetic history was reported for 272/272.

**Impact:** “Universe loaded,” “history loaded,” “quote eligible,” and “signal evaluated” are being conflated. Most symbols do not participate in the weekend path, and current-list membership creates survivorship bias in research.

**Required correction:** Maintain a versioned security master and point-in-time universe snapshots. Publish separate coverage metrics for configured, resolved, history-ready, quote-ready, indicator-ready, and signal-evaluated symbols. A “full-universe simulation” profile should stop if coverage falls below its declared threshold.

### H-2 — Market-data provenance and lifecycle are insufficient

`MarketDataManager` chooses its data source at startup and refreshes that selected source; it does not continuously re-evaluate market state/source eligibility. Quote timestamps primarily represent local receipt time, not an authoritative exchange/event timestamp, and the serialized quote contract omits that timestamp. The current profile sets `MAX_QUOTE_STALENESS_SECONDS=0`, disabling the entry freshness threshold.

Exit checks occur before the cycle's quote refresh and before the entry-staleness guard. Missing quotes cause managed positions to be skipped; stale prices can therefore affect exits even when entry protections would apply.

**Impact:** Replayed/event-time semantics are absent, stale or missing marks may silently defer exits, and a long-running process may retain an unsuitable source selection.

**Required correction:** Put source, exchange timestamp, receipt timestamp, sequence, quality status, and instrument-master version on every market event. Refresh marks before evaluating exits, define a conservative stale-exit policy, and alarm on coverage/age thresholds. For simulation, drive logic from event time rather than wall-clock polling.

### H-3 — The daily weekend loop does not create a meaningful sequence of new signals

The current signal timeframe is `1d`, forming bars are excluded, and synthetic history is seeded once. Mock quotes can change every cycle, but the settled daily history used for indicators remains effectively static during the same day. The signal fingerprint then prevents repeated Groq review of an unchanged candidate set.

Synthetic dates use calendar-day frequency, including weekends and holidays, and seeds rely on Python's process-randomized `hash(symbol)` while quotes use an unseeded random generator. Runs are neither exchange-calendar correct nor reproducible across restarts.

**Impact:** Weekend cycles test scheduling and UI refresh, not market evolution, signal turnover, or repeatable scenarios.

**Required correction:** Provide deterministic scenario replay with fixed seeds, NSE calendars, event-time bars, and advancing timestamps. Preserve a manifest containing seed, universe snapshot, scenario, software revision, and expected control outcomes.

### H-4 — Prediction-agent validation is not a clean forward forecast

The feature builder creates the target from next-period returns and removes the final unknown target. `predict()` withholds only the final five aligned observations and then uses the last row of that already-labelled matrix as the purported next-candle input. The models are not refit after validation. R² weights are estimated on five points, negative values are clamped to zero, and a weak/zero-weight ensemble can still emit a direction with a confidence floor.

**Impact:** The forecast used in ranking and Groq context is not a clean unseen inference and has unstable, uncalibrated quality. Although deterministic risk does not directly consume it, it can alter shortlisting and reinforce later performance priors.

**Required correction:** Maintain a genuinely unlabeled inference row; use rolling or expanding walk-forward validation with adequate sample size, embargo/purging where feature horizons overlap, probability calibration, drift checks, and an explicit abstain state. Persist feature/model versions and validation evidence.

### H-5 — Signal confidence and ranking are scores, not validated probabilities

The signal engine blends fixed strategy priors with agreement among RSI, MACD, directional indicators, and price-versus-average. These inputs are correlated and the resulting confidence is not calibrated. Ranking then combines this score with the prediction direction, a Beta-smoothed historical win rate, expected-R, and any discovered-signal tilt. It orders candidates but does not require independently validated positive expected value.

Regime state can also be inconsistent: ranking infers a local technical regime, while outcomes can be stored against the later Groq regime. Historical performance queries and updates may therefore use different regime definitions.

The current `LLM_REVIEW_ALL_SIGNALS=true` setting bypasses normal shortlist, sector, per-symbol, and breadth controls. Several risk concentration checks are warnings rather than rejections.

**Impact:** Closely related technical evidence can be double-counted, performance buckets can drift, and diversification can be bypassed while the dashboard still describes a ranked/risk-controlled process.

**Required correction:** Define calibrated prediction targets, one canonical regime taxonomy, and a documented hard/soft constraint matrix. Diversification and exposure caps must be enforced after ranking in every runnable profile.

### H-6 — Groq agents are resilient but not institutionally reproducible

The agent nodes correctly use rate limits, circuit breakers, primary/fallback models, JSON parsing, and deterministic failure fallbacks. The runtime additionally detects fallback errors and blocks new entries, which is a conservative control.

However, successful LLM validation can approve any known signal ID returned by the model without a deterministic check that the signal's strategy belongs to the selected active strategies. Model-suggested stop/target/size modifications are not subjected to a complete schema, directional, tick, and range validator. The fallback validator applies different deterministic criteria, so normal and degraded behavior are not policy-equivalent.

Raw prompt/response, prompt version, model revision, sampling parameters, input snapshot, and decision hash are not durably linked to the trade lifecycle. Provider unavailability therefore fails safely for entries but also means the strategy cannot operate independently of Groq; a previously observed quota/fallback failure prevented demonstration of a successful AI-reviewed weekend cycle.

**Impact:** Decisions cannot be reproduced or audited as model-governance artifacts, and provider behavior can change the policy boundary.

**Required correction:** Treat Groq as advisory. Validate every model output against deterministic policy and the selected strategy set. Persist a redacted decision packet and versioned prompt/model metadata. Define an explicit, tested degraded-mode policy and surface it in the UI.

### H-7 — News enrichment has weak provenance and point-in-time controls

The news component consumes Google News RSS and retains source, publication time, and link internally, but the graph adapter reduces context mainly to headline title and binary sentiment. A sentiment score of exactly zero is categorized as negative. There is no robust story deduplication, source-quality weighting, point-in-time archive, strict freshness gate, or dedicated untrusted-text boundary before headline content reaches prompts.

Failures are deliberately non-fatal and can degrade to empty/neutral context without a data-quality state that clearly distinguishes “no news” from “feed unavailable.”

**Impact:** News evidence is difficult to reproduce, can be stale or duplicated, and can silently disappear. Historical replay cannot reconstruct what the model knew at decision time.

**Required correction:** Persist point-in-time news snapshots with source, publication and ingestion times, deduplicated event ID, quality/freshness state, and prompt-safe normalized text. Make feed degradation an explicit decision attribute.

### H-8 — Local-paper fill realism is too optimistic for hedge-fund-style claims

`LocalPaperEngine` provides useful FIFO long/short/partial accounting, adverse configurable slippage, and NSE-style charges. It also blocks long-only oversells. These are meaningful mechanics.

Market orders nevertheless fill immediately from one price with static basis-point assumptions. There is no bid/ask spread process, queue, volume participation, impact curve, latency, liquidity state, opening auction, gap handling, price-band/circuit behavior, or market-driven partial fill. Limit orders can return `PENDING`, but a complete matcher/cancel/expiry lifecycle is not integrated into the main simulation.

**Impact:** The engine can validate bookkeeping, but not execution alpha, capacity, fill probability, implementation shortfall, or realistic stop behavior.

**Required correction:** Add a deterministic event-driven matching simulator with bid/ask, size/depth, participation limits, latency, partial fills, and session/circuit rules. Report gross alpha, fees, spread, slippage, impact, and opportunity cost separately.

### H-9 — Persistence errors can be hidden behind apparently successful objects

The paper engine mutates in-memory state and then attempts to persist; persistence failures are logged/rolled back without necessarily reversing the already returned logical fill. State loading can log a database failure and start fresh from configured capital. The paper health check constructs an engine whose load failure may therefore look like a healthy empty wallet.

The journal catches commit failures yet can return an identifier, and the performance tracker appends in memory before its database commit. Different subsystems thus have different failure semantics—some fail soft, some keep volatile state, and none emits one authoritative reconciliation status.

**Impact:** The dashboard can show a fill or healthy wallet that is not durable. Restart can silently change the book or erase outcome evidence.

**Required correction:** Make wallet mutation and order/fill persistence one transaction. A failed commit must produce no fill and no in-memory mutation. Health must verify durable read/write semantics and reconcile expected state, not merely instantiate a component.

## Signal discovery and the “NVIDIA strategy” question

### What is implemented

The optional signal-discovery workflow uses a Groq signal agent to propose formulas, a deterministic AST allowlist compiler, and a Rank-IC evaluator. It applies a within-run Bonferroni-style p-value correction, writes accepted artifacts atomically, rechecks configured thresholds on load, and restricts any accepted formula to a small probability tilt in signal ranking. It does **not** directly alter risk limits, stops, or position size. This isolation is a good control.

### What is not implemented

No NVIDIA security strategy, NVDA-specific model, CUDA trading component, NIM/NeMo inference pipeline, or separately governed “NVIDIA strategy” was found. NVIDIA references are architectural attribution/reading material, not an active strategy. Any description implying that NVIDIA technology currently supplies a trading model would be aspirational.

### Governance limitations

The same research panel is used for generation feedback, candidate selection, and final reported evaluation; there is no independent holdout or walk-forward promotion stage. Daily Rank-IC t-tests do not adjust for serial dependence from overlapping forward returns. The within-run multiple-testing correction does not control repeated scheduled runs/timeframes, and early candidates face a different effective family size than later candidates.

Artifacts lack a dataset hash, universe version, sample dates, code revision, model/prompt version, turnover/capacity evidence, stability tests, and formal approval state. The current automatic scheduler requests intraday panels (`15m`, `30m`, `1h`, `4h`) while the weekend synthetic history provides daily data; its recorded runs reported zero usable symbols against the minimum of eight, and no accepted formula artifacts were present. Signal discovery is therefore **implemented as a research mechanism but inactive in the current weekend workflow**.

## Strategy governance and backtesting

### Implemented controls

- Four deterministic signal families are available: momentum, mean reversion, breakout, and trend following.
- `RealSignalStrategy` can reuse the actual indicator and signal engine in backtests.
- Backtest slicing uses strictly prior bars.
- A realistic `CostModel` can be applied.
- `edge_verdict()` provides useful walk-forward criteria: enough out-of-sample trades, positive net return/expectancy, and fold consistency.

### Gaps

- The walk-forward verdict is an offline utility, not a runtime strategy admission gate.
- There is no immutable strategy registry containing owner, version, parameters, approved universe/regime, dataset, validation dates, expiry, or retirement status.
- Other backtest strategy implementations can diverge from the runtime path.
- The current universe is current-listed, not point-in-time/survivorship-free.
- Corporate actions, delistings, borrow constraints, liquidity/capacity, gaps, circuits, and many exchange mechanics are not modeled.
- No champion/challenger, shadow promotion, or rollback evidence is linked to production-like paper decisions.

**Conclusion:** The repository contains useful research scaffolding, but “validated strategy” remains aspirational unless a versioned approval artifact is required by the runtime.

## Self-healing and self-learning assessment

The system uses circuit breakers, rate limiting, model fallbacks, cached/failure-isolated support agents, health checks, Telegram alerts, and a kill switch. These improve availability and keep many external failures from crashing the loop.

“Self-learning” is narrower than the phrase suggests. It consists primarily of:

- recording strategy/regime outcomes;
- using historical win rate in ranking/sizing;
- classifying losing trades into text lessons;
- injecting relevant lessons into later agent context; and
- updating lesson success/failure feedback.

It is **not** controlled online model retraining or autonomous strategy improvement. It lacks experiment isolation, model-risk approval, rollback, and statistical proof that a lesson improves out-of-sample results. Because active-lesson mappings are volatile and mock outcomes are not namespaced, even the implemented feedback loop can break on restart or learn from invalid simulation evidence.

Failure isolation is appropriate for trading continuity, but it also allows data/persistence degradation to look like successful operation. Institutional self-healing should reconcile to an authoritative state, declare a degraded mode, and stop new risk when an invariant cannot be proven.

## Cross-component contract audit

| Handoff | Implemented contract | Synchronization issue |
|---|---|---|
| Universe → instrument | Normalized symbol string | No durable instrument ID, effective date, or master/version provenance. |
| History → indicators | OHLCV dataframe; forming bar can be excluded | Synthetic history is not tied to the quote process; calendar/timeframe semantics differ. |
| Quote → signal | Symbol/price/volume quote | Event timestamp/source quality omitted from serialized contract; many symbols lack quotes. |
| Signal → ranking | Strategy, side, levels, technical confidence | Confidence is not calibrated; inferred regime may differ from stored outcome regime. |
| Prediction → ranking/agent | Direction, magnitude, confidence | Inference row/holdout alignment is flawed; no version/calibration/abstention contract. |
| News → agents | Headlines and summarized sentiment | Important provenance/freshness fields are discarded; outage vs no-news ambiguous. |
| Strategy selection → validation | Active strategy names plus raw signals | Successful LLM validation is not deterministically constrained to selected strategies. |
| Validation → risk | Signal with potentially modified levels/size | Model changes lack complete schema and economic-level validation. |
| Risk → execution | Approved trade and percentage | Runtime rebinds price/recomputes shares without final risk approval. |
| Execution → wallet | Local order/fill and wallet mutation | Durability can fail after logical mutation; no canonical lifecycle object. |
| Wallet → exit manager | Separately registered managed position | Different IDs/stores; crash can produce unprotected or ghost state. |
| Exit → journal/performance | Exit fill/P&L | Partial/final/emergency outcome semantics disagree. |
| Outcome → learning | Strategy/regime/symbol/P&L lesson | No run/data/model namespace; active-lesson mapping lost on restart. |
| All state → UI/alerts | Periodic snapshots and activity events | No cross-store invariant status; “healthy” can coexist with incomplete coverage/durability. |

## Monitoring and alerting assessment

### Useful controls present

- Paper wallet, memory, circuit-breaker, Telegram, tracing, Dhan, and news health checks exist.
- Daily drawdown includes unrealized mark-to-market and is persisted.
- LLM token/cost budgets can pause new agent cycles while exits continue.
- Kill-switch, staleness, anomaly, startup/shutdown, and goal-state concepts are surfaced through logs/alerts.
- The UI exposes wallet, positions, signals, mode, and activity.

### Missing institutional controls

- No invariant monitor reconciles wallet positions, exit-manager entries, journal lifecycle, daily risk, performance, and active lessons.
- No mandatory alert for configured-universe versus quote/history/signal coverage.
- No price-scale consistency alarm between recent history, quote, stop, and target.
- No simulation run ID/seed/scenario or environment watermark on every UI view and record.
- No data-latency distribution, missing-bar, outlier, split/corporate-action, or cross-source price check.
- No risk-reservation ledger or projected-versus-actual exposure reconciliation.
- No explicit “persistence degraded,” “news unavailable,” “LLM fallback,” or “unreconciled book” state that universally blocks new entries.

## Testability review

The repository has useful unit tests for agents, signals, risk rules, paper persistence/accounting, daily-risk restart, exit JSON persistence, performance restart, signal-discovery compilation/statistics, and synthetic-signal plumbing. The separation of deterministic components makes the system testable.

The material missing layer is deterministic, failure-injected integration testing. No sufficient evidence was found for tests that prove:

1. one canonical trade survives a crash at every boundary between approval, fill, exit registration, journal, and learning;
2. wallet, exit manager, journal, daily risk, performance, and lessons reconcile after restart;
3. partial plus final exits produce exactly one lifecycle outcome and correct cumulative net P&L;
4. kill-switch flatten produces the same complete outcome records as a normal close;
5. a multi-order batch cannot overshoot trade count, positions, exposure, sector, or loss-at-stop limits;
6. a database commit failure cannot return or display a durable-looking fill;
7. full-universe mock bars and quotes share one price process and reproducible event timeline;
8. daily weekend cycles actually advance the signal information set;
9. the prediction agent's training, validation, and inference timestamps are strictly separated; and
10. every degraded dependency drives the documented fail-open/fail-closed policy.

## Current weekend mock profile: exact interpretation

| Setting/state | Current value or observation | Institutional interpretation |
|---|---:|---|
| Trading mode | `paper` | Correct for permanent paper-only operation. |
| Execution mode | `local_paper` | Orders remain in the local wallet. |
| Live-order permission | `false` | Important defense; should become structurally impossible, not merely configured. |
| Dhan instruments/history/quotes | disabled | Current market data is not broker/live data. |
| Forced trading window | `true` | Enables weekend orchestration; does not by itself create realistic market time. |
| Synthetic history | enabled | Safely latched to the test profile, but independent of quotes. |
| Configured symbols | 275 rows / 272 unique | Symbol list only; not a governed security master. |
| Observed mock quote coverage | 19 symbols | Most configured symbols were not in the effective simulated universe. |
| Signal timeframe | `1d` | Repeated intraday wall-clock cycles see largely unchanged settled history. |
| Signal discovery timeframes | `15m, 30m, 1h, 4h` | Current synthetic source supplied no usable panels; discovery runs failed eligibility. |
| Quote staleness threshold | `0` | Freshness entry gate disabled. |
| LLM review breadth | all signals | Normal diversification/shortlist caps bypassed. |
| Paper wallet | ₹1,000,000, no positions when observed | Appropriate clean demo state; not proof of lifecycle correctness. |

The correct description is: **a safe, local, forced-hours UI/orchestration demo using partial mock quote coverage and independently generated daily history**. It is not a historical replay, exchange simulation, or credible strategy trial.

## Implemented functionality versus aspirational wording

| Area | Implemented today | Aspirational or unsupported interpretation |
|---|---|---|
| Dynamic discovery | A `StockDiscovery.discover()` workflow exists | The main runtime does not call it; current trading universe is effectively the configured CSV. |
| Full-universe scan | History can be seeded for all configured symbols | Weekend quote coverage does not span the full universe. |
| AI validation | Groq agents enrich/select/validate with resilient fallbacks | This is not a reproducible or independently validated investment committee. |
| Deterministic risk | Several hard rules plus warnings are evaluated | Aggregate portfolio heat/exposure and atomic batch reservations are not enforced. |
| Paper execution | Immediate local fills with costs and FIFO accounting | This is not an exchange-grade matching or capacity simulator. |
| Persistence | Multiple subsystems persist state | They do not form one transactional, reconcilable trade ledger. |
| Self-learning | Outcomes and text lessons influence later decisions | No governed online retraining or proven autonomous alpha improvement exists. |
| Signal discovery | Safe formula compiler and Rank-IC research loop exist | Current weekend discovery has no usable intraday panel or accepted active formula. |
| NVIDIA-related strategy | No active NVIDIA/NVDA strategy found | NVIDIA references do not establish an implemented trading strategy or model stack. |
| Institutional backtesting | Walk-forward and cost-aware utilities exist | They are optional and do not gate runtime strategy eligibility. |

## Highest-impact corrections for a permanent paper-only platform

Because the platform will never trade real money, engineering effort should prioritize **simulation integrity and simplification**, not broker hardening.

### P0 — Establish one authoritative paper-trade ledger

1. Create one durable trade/order/fill lifecycle with canonical IDs and idempotent state transitions.
2. Commit wallet mutation, order/fill event, exit protection, journal/outbox, and risk reservation transactionally.
3. Rebuild exit and learning state from the ledger after restart; reconcile before accepting new entries.
4. Treat persistence or reconciliation uncertainty as a hard block on new risk.
5. Remove or compile out broker submission modes and credentials from the runnable product, leaving only `paper_live_data`, `paper_replay`, and `paper_synthetic` profiles.

### P0 — Make the simulator internally coherent and reproducible

1. Generate history, quotes, spread, volume, and fills from one versioned event-time process for every universe instrument.
2. Use NSE calendars and deterministic seeds; persist a complete run manifest.
3. Add price-scale/tick/side/stop/target validation at the final quote.
4. Implement scenario controls for trend, range, gap, crash, circuit, stale feed, low liquidity, and correlation shock.
5. Separate mock/replay/live-data performance and learning stores by construction.

### P0 — Enforce portfolio risk on the actual submitted order

1. Bind price, compute size, and then run final risk on exact quantity and levels.
2. Enforce aggregate loss-at-stop, gross/net notional, symbol, sector, correlation-cluster, cash, and position-count limits.
3. Reserve risk atomically across multi-signal batches.
4. Make zero size, duplicate position, invalid stop/target, stale data, and unresolved state hard rejections.
5. Define daily trade count from entry intents/fills, not exits.

### P1 — Repair outcome, ML, and strategy governance

1. Aggregate partial fills and exits into one lifecycle outcome; include emergency closes.
2. Correct prediction alignment and use adequate walk-forward validation, calibration, and abstention.
3. Require a versioned strategy approval artifact before runtime eligibility.
4. Store decision lineage: data snapshot, feature/model/prompt/strategy/cost versions, risk result, and simulator run.
5. Require minimum sample sizes and uncertainty shrinkage before learned win rates can affect Kelly or ranking.

### P1 — Add institutional data and execution semantics

1. Version the security master and point-in-time universe.
2. Preserve event/source/ingestion timestamps and quality on market/news data.
3. Add spread, depth, participation, impact, latency, partial-fill, circuit, and gap models.
4. Measure coverage, freshness, implementation shortfall, and reconciliation continuously.

### P2 — Turn resilience into observable, tested state

1. Define explicit normal, degraded, blocked, and unreconciled platform states.
2. Add a cross-store invariant monitor and UI banner.
3. Build deterministic end-to-end failure-injection tests and golden simulation scenarios.
4. Version and test every fallback policy so fallback behavior cannot change portfolio policy silently.

## Acceptance criteria for an institutional paper-simulation claim

DeltaQuant should not be described as institutionally credible until evidence demonstrates all of the following:

- Every configured instrument has governed metadata and measurable history/quote eligibility.
- A simulation can be reproduced exactly from its run manifest and event log.
- Bars, quotes, levels, and fills are generated from one coherent price process.
- Final order quantity and levels are identical across risk approval, execution, wallet, exits, journal, and learning.
- Multi-order batches cannot exceed any hard portfolio limit.
- Restart at any lifecycle boundary yields one reconciled book with no missing or duplicate outcomes.
- Partial, final, time, stop, target, trailing, and kill-switch exits produce correct cumulative net P&L.
- Mock/replay evidence cannot influence live-data paper decisions without explicit promotion.
- ML predictions are cleanly out-of-sample, calibrated, versioned, and able to abstain.
- Every active strategy has a current, cost-aware, out-of-sample approval artifact.
- The simulator models spread, liquidity, partial fills, latency, impact, gaps, and NSE session/circuit rules sufficiently for its declared use.
- Dashboard and alerts expose data lineage, run ID, coverage, stale state, risk reservation, persistence, and reconciliation health.

## Controls worth preserving

- Permanent `local_paper` execution and disabled live-order permission.
- The strict synthetic-history safety latch.
- Forming-bar exclusion, NaN/inf sanitization, and indicator caching.
- Runtime exits before new entries and kill-switch recheck before entry submission.
- Persisted daily drawdown including unrealized P&L.
- LLM circuit breakers, rate limiting, budget gates, structured parsing, and failure fallbacks.
- Position-sizer use of stop distance and the audited transaction-cost model.
- FIFO long/short/partial local accounting and long-only oversell protection.
- AST-allowlisted signal formulas and the rule that discovery tilt cannot alter risk or sizing.
- Cost-aware walk-forward research utilities.
- Advisory-only profit goals that do not relax risk.
- UI visibility and best-effort operational notifications.

## Evidence map

Primary reviewed implementation/configuration areas:

- `scripts/run_live_trading.py` — orchestration, universe use, quote refresh ordering, signal ranking, agent graph invocation, final sizing, entries, exits, kill switch, journal, performance, and learning
- `src/market/stock_discovery.py` — configured universe and optional dynamic discovery
- `src/market/manager.py`, `simulated_data.py`, and `history_manager.py` — data-source lifecycle, mock quotes, and synthetic history
- `src/market/indicators.py`, `signals.py`, `signal_ranking.py`, and `sizing.py` — features, signal levels/confidence, ranking, and sizing
- `src/agents/graph.py`, `state.py`, `prediction.py`, `signal_validation.py`, and `risk_compliance.py` — agent contracts, ML prediction, validation, and deterministic risk
- `src/execution/service.py`, `paper_engine.py`, `costs.py`, `exit_manager.py`, and `journal.py` — local paper order, accounting, costs, managed exits, and journal state
- `src/memory/performance_tracker.py`, `database.py`, `analyzer.py`, and `classifier.py` — performance and lesson feedback
- `src/signal_discovery/` and `data/discovered_signals/auto_run_state.json` — formula research and current failed intraday discovery state
- `src/backtesting/` and `scripts/validate_strategy.py` — backtest and offline edge validation
- `src/finops/`, `src/health/`, dashboard code, and notification code — monitoring, spend limits, UI, and alerts
- `src/config/settings.py`, `.env`, and the configured universe CSV — active paper/weekend profile and safety switches
- `tests/` — current component-level coverage and integration-test gaps

## Final risk statement

DeltaQuant should be positioned today as a **safe local paper-trading research prototype and workflow demonstrator**. It should not be presented as a hedge-fund-grade simulator or used to infer expected returns, risk-adjusted performance, capacity, or model improvement from the current weekend mock run.

The highest-value work is not adding more agents. It is making the paper book, risk decision, execution event, exit state, journal, and learning record one synchronized and reproducible lifecycle; then replacing the disconnected mock generators with a coherent event-time simulator. Once those foundations are proven through restart and failure-injection tests, the existing agent, risk, and research components can be evaluated credibly.

No project file, configuration value, database record, or running process was modified during this review. Only this requested audit document was created.
