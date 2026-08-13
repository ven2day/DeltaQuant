# DeltaQuant Quantitative, Risk, and Control Review

**Review date:** 7 August 2026  
**Review type:** Read-only code, configuration, runtime-state, and workflow-guide review  
**Perspective:** Hedge-fund risk management and senior quantitative engineering  
**Reviewed guide:** `docs/architecture/DeltaQuant-End-to-End-Workflow.md`  
**Decision standard:** Whether the present system is suitable for weekend simulation, controlled paper trading, or broker-connected deployment

## Executive verdict

The current weekend configuration is protected from real-money execution: it uses `TRADING_MODE=paper`, `EXECUTION_MODE=local_paper`, `ALLOW_LIVE_ORDERS=false`, disables Dhan instrument/quote/history calls, and enables the synthetic-history safety latch. The fresh paper wallet also reports ₹1,000,000 with no open positions. That is a meaningful safety baseline.

However, the codebase is **not suitable for broker-connected deployment**, and the present mock setup is **not a valid full-universe end-to-end market simulation**. Three verified issues are deployment blockers:

1. Real broker entries can be submitted through `ExecutionService`, while normal exits and kill-switch liquidation still call the local paper engine directly. In a broker-connected mode, a dashboard can therefore show a locally closed position while genuine broker exposure remains open.
2. The `dhan_paper` name does not denote a broker sandbox. With credentials and the master gate enabled, both `live` and `dhan_paper` reach the Dhan submission path; `TRADING_MODE=paper` is not independently checked at order submission.
3. Broker-order idempotency is not crash-safe. Client order IDs are derived from Python's process-randomized `hash()`, ambiguous/unconfirmed orders are not persisted as consumed, and database read failures make duplicate protection fail open.

The weekend simulator has a separate validity problem: 272 symbols receive synthetic history, but the quote simulator has a small hard-coded price map and the running dashboard currently has quotes for only 19 symbols. Synthetic history and current quotes are generated on independent price scales. Signals and risk checks use history-scale prices; just before execution, only the entry price is overwritten with the mock quote while the original stop and target remain. Consequently, the simulator can validate and size one economic trade but execute and monitor a different one.

**Go/no-go conclusions:**

| Intended use | Verdict | Reason |
|---|---|---|
| UI demonstration with no orders | **GO** | Current configuration is local-only and shows the intended pipeline state. |
| Basic local-paper mechanics on deliberately controlled inputs | **CONDITIONAL GO** | Useful for component plumbing, but not for strategy or fill-quality conclusions. |
| Full 272-symbol weekend workflow validation | **NO-GO** | Quote coverage and price-contract mismatch invalidate the test. |
| Performance, signal edge, or risk calibration based on current mock | **NO-GO** | Synthetic paths and fills are not representative; mock outcomes can contaminate shared learning records. |
| Any Dhan/broker-connected order submission | **NO-GO** | Exit-routing and order-idempotency failures can leave or duplicate real exposure. |

## Classification used in this review

- **Critical:** Can create uncontrolled or duplicated real exposure, or defeat a primary safety claim. Must be closed before any broker connectivity.
- **High:** Can materially breach stated risk limits, invalidate the trading decision, or corrupt models/learning used for capital allocation.
- **Medium:** Important robustness, realism, governance, or operational-control deficiency; may not independently create immediate loss in the current local configuration.
- **Recommendation:** A design improvement or production control, distinguished from a defect directly verified in the current code.

“Verified” means the behavior is present in the reviewed code/configuration or was observable in the running local dashboard state. It does not mean the failure has already caused a monetary loss.

## Critical verified findings

### C-1 — Broker entries and exits use different execution systems

**Status:** Verified defect; live-deployment blocker.

The main entry loop sends approved entries through `ExecutionService.submit_async()` (`src/markets/nse/runtime/live.py`, around lines 1292–1307). Normal exit decisions call `paper_engine.place_order()` directly (around line 601), and kill-switch flattening also calls `paper_engine.place_order()` directly (around line 1196).

This is safe only while the effective execution mode remains `local_paper`. If `live` or `dhan_paper` becomes effective, the entry can reach Dhan but stop-loss, take-profit, time exit, trailing-stop exit, and emergency flatten mutate only the local wallet. Broker truth and local truth can immediately diverge. Reconciliation at startup does not cure the inability to route an urgent exit to the broker during the session.

**Impact:** Unbounded real exposure after the system believes it is flat; kill switch may provide false assurance; local P&L and risk state become unreliable.

**Required control:** Put every order intent—entry, partial exit, normal exit, reversal, and emergency flatten—through one mode-switched execution gateway. Add broker-mode integration tests asserting that every exit type reaches the broker adapter and that local state changes only after a confirmed/reconciled broker outcome.

### C-2 — `dhan_paper` is not a safe sandbox, and `TRADING_MODE=paper` is not an execution gate

**Status:** Verified hazardous mode semantics; live-deployment blocker. Current configuration remains safe because `ALLOW_LIVE_ORDERS=false` and `EXECUTION_MODE=local_paper`.

`ExecutionService._resolve_mode()` (`src/markets/nse/execution/service.py`, around lines 204–225) permits both `live` and `dhan_paper` when credentials exist and `allow_live_orders` is true. It does not independently require `trading_mode == "live"`. The service comments explicitly say those modes send real orders. The adapter may log “sandbox mode” when `TRADING_MODE=paper`, but it initializes the broker client, and the default Dhan base URL is the live API endpoint (`src/config/settings.py`, around lines 103–104).

Configuration validation compounds the issue: credentials are checked only when `TRADING_MODE=live`, and cross-field validation logs warnings rather than failing startup (`src/config/settings.py`, around lines 655–706). A configuration of `TRADING_MODE=paper`, `EXECUTION_MODE=dhan_paper`, `ALLOW_LIVE_ORDERS=true`, and valid credentials can therefore submit genuine broker orders.

**Impact:** An operator can reasonably interpret “paper” or `dhan_paper` as non-monetary while enabling a real order route.

**Required control:** Remove or rename `dhan_paper` unless Dhan provides a verified sandbox endpoint. Make all broker-capable modes require an explicit, fail-closed conjunction such as `TRADING_MODE=live && EXECUTION_MODE=live && ALLOW_LIVE_ORDERS=true`, validated before process startup. Display the resolved endpoint and “REAL ORDERS” status prominently and require an independent arming mechanism.

### C-3 — Broker-order idempotency can duplicate orders after timeout, restart, or database failure

**Status:** Verified defect; live-deployment blocker.

`ExecutionService._client_order_id()` uses Python `hash(idempotency_key)` (`src/markets/nse/execution/service.py`, around lines 228–232). Python hashes are process-randomized, so the same logical request does not reliably retain the same client order ID after restart. The idempotency record is written only for filled/partially filled results (`_record_if_filled`, around lines 302–305). `LiveBrokerExecutor.place_and_confirm()` can return a placed but unconfirmed order when polling expires (`src/markets/nse/execution/live_executor.py`, around lines 42–77); the service maps non-terminal/non-filled results to rejection and does not consume the idempotency key. A retry can therefore send another order while the first is still working or later fills.

The idempotency store also catches database lookup exceptions and returns “not seen” (`IdempotencyStore.seen`, `src/markets/nse/execution/service.py`, around lines 109–124), which makes order safety fail open during a database incident.

**Impact:** Duplicate genuine orders, especially at the worst operational moment: broker latency, process crash, network partition, or database outage.

**Required control:** Persist an `ORDER_INTENT` transaction before broker submission, use a stable cryptographic digest/UUID for the client order ID, record all accepted/ambiguous states, and reconcile by that stable ID before any retry. An idempotency-store failure must block broker submission. Use a durable state machine such as `INTENT → SUBMITTED → ACKNOWLEDGED → PARTIAL/FILLED/CANCELLED/REJECTED/UNKNOWN`, with `UNKNOWN` requiring reconciliation rather than resubmission.

## High-severity verified findings

### H-1 — Declared portfolio exposure and aggregate-risk limits are not enforced

**Status:** Verified control gap.

`max_total_exposure_pct` is defined in settings and mapped into `RiskLimits`, but no risk rule consumes it. `max_total_risk` is sanity-checked against per-trade risk during configuration validation but is not enforced as aggregate open risk at runtime. The absolute `MAX_POSITION_SIZE` is used in a legacy sizing path, while the main loop's `PositionSizer` caps by percentage of capital instead (`src/markets/nse/runtime/live.py`, around lines 1277–1288). These values happen to align at ₹1,000,000 under the current 10% percentage cap, but they will diverge as wallet capital changes.

**Impact:** The system can approve a portfolio whose total notional or total loss-at-stop exceeds the advertised limit, even though each trade passes individual checks.

**Required control:** Before every submission, calculate projected gross and net notional, sector exposure, correlated-cluster exposure, and aggregate loss at hard stops—including the candidate order and outstanding intents. Reject, do not warn, when any hard limit would be breached.

### H-2 — A batch of approved trades shares a stale risk snapshot

**Status:** Verified defect.

`risk_compliance_node()` loops over validated signals using one unchanged portfolio and daily-statistics snapshot (`src/agents/risk_compliance.py`, around lines 92–165). The runtime then submits all approved trades sequentially (`src/markets/nse/runtime/live.py`, from around line 1242) without reserving limits or refreshing positions, exposure, cash risk, and daily order count after each fill.

For example, with four of five daily trades already used, multiple candidates in the same batch can each see “4” and pass. The same pattern applies to maximum concurrent positions. Available cash may incidentally stop some orders, but cash is not a substitute for a risk reservation and does not control short or broker semantics.

**Impact:** Intra-cycle breach of daily-trade, position-count, and exposure limits.

**Required control:** Reserve risk atomically per approved intent, submit one candidate at a time, reconcile outcome, then recompute the projected portfolio before the next order. The risk decision and reservation should be versioned against the portfolio snapshot.

### H-3 — Concentration, trade quality, and duplicate-position checks are advisory warnings

**Status:** Verified policy weakness.

The deterministic risk engine blocks daily trade count, daily loss, maximum single position, maximum position count, hours, and drawdown. But minimum reward/risk, stop width, duplicate position, low confidence, sector exposure, same-sector correlation, and pairwise correlation are appended as warnings and do not prevent approval (`src/agents/risk_compliance.py`, approximately lines 221–358). Tests explicitly preserve approval when pairwise correlation is only a warning.

Runtime idempotency keys include the workflow ID, so the same symbol/signal in a later cycle receives a new key. Because an existing position is only a warning and total exposure is not enforced, repeat cycles can pyramid the same security or correlated book.

**Impact:** The system can satisfy the formal deterministic gate while violating the portfolio intent described by the guide.

**Required control:** Define a documented hard/soft limit matrix. Existing-symbol aggregation, maximum symbol weight, maximum sector weight, correlated-cluster limit, stop-distance range, and minimum reward/risk should be hard blocks unless a separately authorized strategy explicitly supports scaling.

### H-4 — Weekend mock quotes cover only a small subset of the configured universe

**Status:** Verified current limitation and guide inaccuracy.

The stock-universe file contains 275 rows and 272 unique symbols. Synthetic history was created for 272 symbols. In contrast, `SimulatedMarketData` contains a fixed price map of only about 20 large-cap names (`src/markets/nse/market_data/simulated.py`, around lines 20–41) and skips symbols absent from that map (around lines 165–166). The running dashboard state exposed quotes for only 19 current-universe names, and the activity log reported ranking from 19 symbols—not 272.

**Impact:** The weekend run does not test discovery, quote freshness, signal generation, ranking, risk, or order handling for most of the configured universe. Apparent “272/272 loaded” status applies to history, not current quotes.

**Required control:** Generate a deterministic quote stream for every resolved instrument in the loaded universe and explicitly report `history_coverage`, `quote_coverage`, and `eligible_signal_coverage` as separate numbers. Startup should fail the “full simulation” profile if coverage is incomplete.

### H-5 — Mock history, quote, stop, and target prices do not share one economic price process

**Status:** Verified test-invalidating defect.

`HistoryManager._seed_synthetic()` creates independent random historical paths with arbitrary ₹100–₹3,000 bases (`src/markets/nse/market_data/history_manager.py`, from around line 197). `SimulatedMarketData` uses a separate fixed price map. Technical signals derive entry, stop, and target from the history close (`src/core/candidates/signals.py`, around line 410 onward). Just before execution, the live loop replaces only the candidate entry with the current quote; the previously computed stop and target are retained (`src/markets/nse/runtime/live.py`, around lines 1267–1269). Risk compliance ran earlier using the history-scale entry and stop.

**Impact:** A candidate can pass risk on one scale, be sized on another mixture of scales, and then register exits that are nonsensical relative to its fill. This defeats end-to-end validation of sizing, stop loss, take profit, P&L, and learning.

**Required control:** One simulated instrument state must generate historical bars, current quote, spread, and future ticks. At the execution boundary, reject any order whose entry/stop/target is inconsistent with side, price scale, tick size, or configured distance; rerun risk and sizing after final price binding.

### H-6 — Position sizing can override a “do not trade” result and can exceed risk-per-trade intent

**Status:** Verified defect.

The position sizer can use Kelly when a strategy win rate exceeds a low threshold. Those win rates include hard-coded/blended priors even when the strategy has no empirical trades. The Kelly path caps position notional by `max_position_pct`, not loss at stop by `risk_per_trade`. A very wide stop is only a warning, so notional-capped Kelly sizing can still exceed the intended loss-at-stop budget.

The main runtime then applies `quantity = max(1, sizing.shares)` and also falls back to one share when price/stop inputs are invalid (`src/markets/nse/runtime/live.py`, around lines 1286–1288). A negative-expectancy Kelly result of zero is therefore converted into an order.

**Impact:** A quantitative no-trade decision becomes a trade, and configured per-trade risk can be exceeded.

**Required control:** Zero shares must be a hard rejection with a reason. Every sizing method must be capped by the minimum of notional, liquidity, concentration, and actual loss-at-stop budgets. Kelly should require a minimum out-of-sample sample size and conservative uncertainty shrinkage.

### H-7 — The ML prediction agent does not produce a clean forward out-of-sample forecast

**Status:** Verified model-validation defect.

The prediction feature builder creates the target with `returns.shift(-1)` and drops the unknown last target (`src/agents/prediction.py`, around lines 83–149). `predict()` withholds the final five aligned rows as a test set and then calls the final row of that same matrix “next candle” features (around lines 151–277). That row already has a known target and is part of the tiny test set. The models are not refitted on all available data after evaluation. Only five observations determine model R² weights; negative values are zeroed, and when all weights are zero the ensemble numerator collapses toward zero while the classifier still produces a direction and a confidence floor.

**Impact:** Ranking and LLM context can be biased by an in-sample/holdout-confused estimate with unstable weighting and poorly calibrated confidence. The deterministic risk engine does not directly consume the prediction, which limits—but does not remove—the impact.

**Required control:** Separate feature time `t` from target `t+1`, reserve a genuinely unseen inference row, use rolling/expanding walk-forward validation, require enough observations, calibrate probabilities, record calibration/error by regime, and abstain when validation is insufficient.

### H-8 — The formal out-of-sample edge verdict is not a runtime deployment gate

**Status:** Verified governance gap.

The codebase has a sensible walk-forward `edge_verdict()` requiring enough out-of-sample trades, positive net expectancy and return, and fold consistency (`src/backtesting/walk_forward.py`, around lines 221–249). It is used by offline validation tooling, not checked by `run_live_trading.py`. A strategy can therefore be selected and traded without a current signed validation artifact, dataset identifier, cost-model version, parameter hash, expiry date, or regime scope.

**Impact:** Research controls exist as optional analysis rather than as a capital-deployment control.

**Required control:** Maintain a strategy registry with immutable versions and a fail-closed `VALIDATED` status. Runtime should reject unapproved or expired strategy versions. Include data lineage, survivorship caveat, costs, sample sizes, stability tests, and approval identity.

### H-9 — Mock outcomes contaminate shared performance and learning state

**Status:** Verified schema/design defect.

`strategy_performance_records` stores strategy, regime, P&L, symbol, winner flag, and timestamp, but no data source, execution mode, simulation run, model version, or dataset identifier (`src/memory/performance_tracker.py`, around lines 47–59). `PerformanceTracker.record_trade()` likewise accepts no environment field (around lines 135–179). The `agent_memory` lesson schema has contextual trade fields but no mock/live namespace (`src/memory/database.py`, around lines 39–82).

These records feed later strategy win rates, ranking/Kelly inputs, and lesson injection. Thus a weekend trade produced by unrealistic random bars and immediate simulated fills can influence a later live-data decision.

**Impact:** Silent data leakage from testing into portfolio decisions; self-reinforcing model risk; misleading performance statistics.

**Required control:** Physically or logically isolate `mock`, `backtest`, `paper-live-data`, `shadow`, and `broker-live` records. Key every performance/lesson record by environment, data run, strategy/model version, and cost model. Production inference must query only approved namespaces. Resetting a wallet must not implicitly preserve contaminated priors.

### H-10 — Quote-staleness protection is disabled in the current configuration

**Status:** Verified configuration risk; not immediately harmful in the current synthetic stream.

`.env` sets `MAX_QUOTE_STALENESS_SECONDS=0`. The code interprets zero as disabling the new-entry freshness gate. The workflow guide describes stale-quote protection as a general safeguard without emphasizing that it is presently disabled.

**Impact:** If Dhan quotes are later re-enabled without changing this value, the system may initiate entries from arbitrarily old last-known prices while exits continue to run.

**Required control:** Treat zero as valid only in a dedicated simulation profile. Live-data profiles should fail startup unless exchange timestamps are present and a conservative positive threshold is configured.

### H-11 — Current “review all” setting bypasses several shortlist controls

**Status:** Verified current configuration and guide omission.

`.env` sets `LLM_REVIEW_ALL_SIGNALS=true`. That bypasses the normal `max_active_stocks`, `llm_review_max_symbols`, sector cap, and per-symbol truncation in the shortlisting stage. The current 19-symbol quote subset produced seven ranked signals, but restoring a broad quote universe could substantially expand prompt volume and review latency. Previous runtime activity also showed primary-model quota failure and an unusable fallback, so natural AI approval was not demonstrated.

**Impact:** Unbounded review breadth can create provider quota exhaustion, latency, stale decisions, and correlated signal load. FinOps budget limits measure usage but do not guarantee the provider's quota or an actionable decision before prices move.

**Required control:** Use bounded deterministic shortlisting in all deployable profiles. Make “review all” an explicit test-only setting and expose the number dropped per constraint. When LLM service is degraded, choose an explicit policy—abstain or approved deterministic fallback—and display it as a degraded mode.

## Medium-severity verified findings and limitations

### M-1 — Synthetic market paths are neither reproducible nor market-realistic

The history generator derives seeds from Python `hash(symbol)`, which changes across interpreter starts. It generates calendar-day bars, including weekends/holidays. Mock quotes use an independent, unseeded random process. There is no common market factor, sector correlation, spread process, depth, liquidity, volume participation, market impact, opening auction, gap, circuit limit, halt, corporate action, split, or dividend treatment. Quote timestamps use host-local `datetime.now()` rather than the shared IST/UTC market-time utility.

**Consequence:** Test failures are difficult to reproduce, cross-sectional/risk correlations are meaningless, and the environment cannot validate exchange edge cases.

### M-2 — Local-paper fills test accounting mechanics, not execution quality

Market orders fill immediately with static slippage and fee assumptions. There is no order-book queue, variable spread, latency, volume constraint, partial fill driven by liquidity, impact curve, or price-limit rejection. Limit orders can become `PENDING`, but no complete runtime matching/cancel/expiry lifecycle was identified. The main loop currently uses market orders, so this is a supported-mode gap rather than the immediate weekend path.

**Consequence:** Simulated P&L and fill rates should not be used to estimate deployable capacity or live execution alpha.

### M-3 — The weekend loop is wall-clock polling, not event-time replay

The current configuration uses a one-day signal timeframe while cycling every 120 seconds. Most cycles inspect the same settled daily bar; unchanged-signal fingerprinting reduces repeated AI calls, which is good, but it does not simulate a sequence of new market information. The separate signal-discovery scheduler asks for intraday panels for several timeframes, while the present synthetic data does not provide a usable full intraday panel.

**Consequence:** The weekend mode exercises orchestration, not realistic temporal behavior, signal turnover, intraday exits, or session-boundary logic.

### M-4 — External news text enters model prompts without a dedicated untrusted-content boundary

Google News RSS headlines are added to agent context. The code is resilient to LLM exceptions and malformed output, but no dedicated prompt-injection sanitizer, source allowlist, duplicate-event clustering, or strict event-time freshness policy was identified. Support-agent failures are intentionally swallowed and converted into fallback/missing context.

**Consequence:** Untrusted headline text can influence model behavior, and degraded enrichment can silently look like neutral evidence rather than a data-quality incident.

### M-5 — Startup validation is warning-oriented instead of fail-closed

The settings validator catches cross-field concerns and logs warnings rather than rejecting startup. It does not implement a complete invariant matrix covering trading mode, execution mode, live-order gate, endpoint type, forced-hours setting, Dhan data switches, and staleness policy.

**Consequence:** A typo or misunderstood field can move the system into a materially different safety posture while the process continues.

### M-6 — Broker operations need a transactional recovery model

The broker executor correctly polls order status rather than assuming “placed” means “filled,” and startup reconciliation exists. However, no durable pre-submit intent/outbox and no fail-closed ambiguous-order workflow were found. This is the broader operational counterpart to C-3.

**Consequence:** Network partitions and restart timing can leave order state uncertain precisely when automation should stop and reconcile.

### M-7 — Model and lesson governance is incomplete

The learning loop is failure-isolated and uses decayed relevance, but it lacks champion/challenger separation, formal lesson approval, environment lineage, rollback, and a mechanism to quarantine lessons derived from unrealistic simulation. Strategy priors can affect Kelly sizing before enough observations exist.

**Consequence:** The system can learn confidently from its own simulator artifacts and feed them back into future decisions.

### M-8 — API protection relies mainly on localhost binding

The present dashboard binds locally and CORS is local, which is appropriate for this demo. No application-level authentication/authorization control was identified for the dashboard state/WebSocket surface.

**Consequence:** This is acceptable only while the service remains loopback-only. Any remote binding, reverse proxy, or shared-host deployment would require authentication, TLS, origin enforcement, and secret-free state payload review.

## Current configuration: what is actually safe and what is test-only

| Control | Current value | Assessment |
|---|---:|---|
| `TRADING_MODE` | `paper` | Safe only in combination with the execution controls below; not independently enforced at broker submit. |
| `EXECUTION_MODE` | `local_paper` | Keeps the current order path local. |
| `ALLOW_LIVE_ORDERS` | `false` | Effective current master protection. |
| Dhan instruments/history/quotes | all disabled | Prevents current Dhan market-data calls. |
| `FORCE_TRADING_WINDOW` | `true` | Appropriate only for an isolated test profile; unsafe as a general live setting. |
| `ENABLE_SYNTHETIC_HISTORY` | `true` | Guarded by a strong test-only latch; useful, but the generated history is not price-consistent with quotes. |
| Wallet starting balance | ₹1,000,000 | Correct current demo capital; zero open positions observed. |
| `MAX_QUOTE_STALENESS_SECONDS` | `0` | Freshness check disabled. Must not carry into live-data operation. |
| `SIGNAL_TIMEFRAMES` | `1d` | Does not reproduce intraday weekend flow. |
| `LLM_REVIEW_ALL_SIGNALS` | `true` | Bypasses shortlist breadth and diversification caps; test-only behavior. |
| Cycle interval | 120 seconds | Repeats over mostly unchanged daily information. |
| Signal discovery | enabled/auto-run | Its requested intraday panels are not fully supported by current synthetic history. |

The correct operational interpretation is: **safe from real money today, but not representative of a complete market or production execution environment.**

## Required corrections to the workflow guide

The guide is structurally strong and accurately explains most major components, but the following statements need qualification:

1. **“Each resolved symbol gets a current quote” is not true in the current mock profile.** History loads for 272 symbols; current mock quote coverage is 19 symbols in the observed run.
2. **“Every symbol × timeframe reaches signal processing” needs an eligibility qualifier.** Symbols without quotes are skipped, and the current simulator does not provide full-universe quote coverage.
3. **“Unified execution service” applies to entries, not all exits.** Normal exit and kill-switch liquidation paths bypass it and call the local paper engine.
4. **`dhan_paper` must not be described as simulated broker paper execution.** It can reach real submission when the live-order gate and credentials are present.
5. **Freshness protection is configurable, and currently disabled.** The guide should state the actual profile value.
6. **Shortlist caps are currently bypassed.** Because `LLM_REVIEW_ALL_SIGNALS=true`, the guide's displayed cap values are not the active current behavior.
7. **The statement that the mock path is “not broken” is too strong for end-to-end testing.** The backend is running and orchestration is active, but quote coverage and price-scale inconsistencies prevent a valid complete order-lifecycle test.
8. **Learning is not environment-isolated.** The guide should warn that mock outcomes can enter the same performance and lesson stores used by subsequent runs.

## Important controls that are present and should be preserved

This review is not a claim that the system lacks meaningful controls. The following are valuable:

- The current combination of local-paper execution and disabled live-order gate prevents broker submission.
- Synthetic history has a deliberate test-only configuration latch requiring forced hours, paper/local execution, disabled Dhan calls, and no live-order permission.
- Position exits run before new entries in the main cycle.
- The kill switch is rechecked immediately before new entry submission.
- Daily drawdown includes unrealized mark-to-market effects and is persisted.
- Long-only and circuit-breaker checks exist.
- Forming-bar exclusion, stale-bar concepts, NaN/inf sanitization, and indicator caching reduce common signal defects.
- The local cost model includes slippage and NSE-style charges rather than assuming frictionless fills.
- The live executor polls to a terminal outcome rather than treating broker acceptance as a fill.
- Signal-discovery formulas use an AST allowlist, multiple-comparison correction, and isolation from risk sizing.
- An out-of-sample walk-forward validation framework exists and already encodes useful minimum criteria.
- Database-backed wallet operations roll back on persistence errors.
- LLM nodes use rate limiting, circuit breaking, model fallback, and deterministic failure fallbacks rather than crashing the graph.

These controls should be retained while the gaps above are closed.

## Prioritized remediation plan

The items below are **design recommendations** based on the verified findings; they are not claims that these mechanisms already exist.

### P0 — Before any broker connectivity

1. Route every entry and exit through one execution gateway; prove emergency flatten against the broker adapter.
2. Remove ambiguous `dhan_paper` semantics and make broker arming a fail-closed multi-condition startup invariant.
3. Implement durable order intents, stable client IDs, ambiguous-state reconciliation, and fail-closed idempotency storage.
4. Enforce projected portfolio exposure, aggregate loss-at-stop, position count, sector/correlation, and daily-trade limits with atomic reservations.
5. Make duplicate-position and invalid/zero sizing outcomes hard rejections.
6. Add an independent broker-side reconciliation/flatness monitor that can page an operator and prevent further entries.

### P1 — Before claiming full weekend end-to-end validation

1. Create a deterministic, versioned, event-time simulator covering every configured instrument.
2. Generate bars, quotes, stops, targets, and fills from the same price state; rerun risk after binding the final executable quote.
3. Add realistic spread, liquidity, partial fill, latency, market impact, gaps, circuits, and session calendars.
4. Isolate mock, backtest, paper-live-data, shadow, and live performance/lesson stores.
5. Add explicit simulation scenarios for stop, target, trailing stop, partial profit, rejection, partial fill, stale quote, kill switch, restart, database outage, and ambiguous broker response.
6. Expose universe/history/quote/signal/order coverage separately in the dashboard and test assertions.

### P1 — Before using models to allocate risk

1. Correct target/inference alignment in the prediction agent and implement nested walk-forward validation and probability calibration.
2. Require a minimum effective sample before using learned win rate or Kelly; shrink estimates toward a conservative prior and allow abstention.
3. Make the offline edge verdict a signed, versioned runtime gate.
4. Track feature, model, prompt, strategy, dataset, simulator, and cost-model versions for every decision and outcome.
5. Add drift, calibration, turnover, capacity, and regime-stability monitoring with explicit disable thresholds.

### P2 — Operational hardening

1. Reject unsafe configuration combinations at startup instead of logging warnings.
2. Restore a nonzero live quote-freshness limit and use exchange timestamps.
3. Add a degraded-mode state for LLM/news/database/broker failures; surface it in the UI and notifications.
4. Treat news text as untrusted data, constrain sources, deduplicate events, and use strict event-time freshness.
5. Require authentication/TLS before any non-loopback dashboard deployment.
6. Run repeatable failure-injection tests and maintain an operator runbook for uncertain orders, reconciliation mismatch, and emergency flatten.

## Acceptance criteria for a future re-review

A broker-connected release should not pass review until evidence demonstrates all of the following:

- A test broker adapter proves that entry, every exit type, and kill-switch flatten share the same gateway.
- A crash immediately before and after submission cannot duplicate an order.
- An ambiguous broker response blocks retries until reconciliation.
- Projected aggregate risk and exposure cannot exceed limits within a multi-order batch.
- Every live strategy version has a current out-of-sample approval artifact net of audited costs.
- Mock/backtest records cannot affect production ranking, sizing, or memory queries.
- Price, stop, target, tick size, and side invariants are revalidated at the final executable quote.
- Quote coverage and freshness are complete and measurable for the approved universe.
- End-to-end tests cover partial fills, rejects, disconnects, stale data, database failure, restart recovery, and operator flatten.
- The dashboard and alerting state distinguish local paper, shadow, and real-order operation without ambiguous terminology.

## Final risk statement

The present DeltaQuant weekend profile is an appropriate **local UI/orchestration demonstration**, and its current settings materially protect against real-money orders. It should not be represented as a complete 272-symbol market simulation, a validated strategy environment, or a rehearsal of broker execution. The most urgent engineering work is not additional agent sophistication: it is unified exit execution, crash-safe order state, fail-closed mode configuration, atomic portfolio-risk enforcement, and clean separation of simulated learning data. Until those controls are verified, the correct broker-connectivity posture is **hard disabled**.

## Evidence map

Primary reviewed implementation areas:

- `src/markets/nse/runtime/live.py` — cycle orchestration, quote binding, sizing, entries, exits, kill switch
- `src/markets/nse/execution/service.py` — execution-mode resolution, idempotency, live submission
- `src/markets/nse/execution/live_executor.py` — broker placement and terminal-status polling
- `src/markets/nse/execution/paper_engine.py` and `src/markets/nse/execution/costs.py` — local fills and costs
- `src/agents/risk_compliance.py` — deterministic risk rules and severity behavior
- `src/markets/nse/market_data/simulated.py` — simulated quote universe and process
- `src/markets/nse/market_data/history_manager.py` — synthetic-history generation
- `src/core/candidates/signals.py` and `src/markets/nse/risk/sizing.py` — signal prices and position sizing
- `src/agents/prediction.py` — ML target alignment, testing, and ensemble weighting
- `src/backtesting/walk_forward.py` — offline edge-verdict logic
- `src/memory/performance_tracker.py` and `src/memory/database.py` — performance and lesson persistence
- `src/config/settings.py` and `.env` — safety gates and current test profile
- `docs/architecture/DeltaQuant-End-to-End-Workflow.md` — user-facing workflow claims

No project file, configuration value, database record, or running process was modified during this review. Only this requested review document was created.
