"""
Failure-injection tests for the canonical trade lifecycle ledger (C-2/C-5 Institutional
Audit, H-2 Quant-Risk Review): one trade_id spanning wallet -> exit manager -> journal ->
performance tracker -> learning, durable across a crash at every boundary.

Each test simulates a crash by deliberately calling only a *prefix* of the calls a real
cycle in scripts/run_live_trading.py would make, then exercises the recovery path
(``PaperTradeLifecycleStore.reconcile`` / ``TradeFinalizer.replay_unfinalized``) and asserts
the system lands in the correct, non-ambiguous state.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.config.settings import Settings
from src.execution.costs import CostModel
from src.execution.finalize import TradeFinalizer
from src.execution.journal import TradeJournal
from src.execution.lifecycle import MOCK_NAMESPACE, PaperTradeLifecycleStore
from src.execution.paper_engine import LocalPaperEngine
from src.execution.service import ExecutionMode, ExecutionService, IdempotencyStore
from src.memory.performance_tracker import PerformanceTracker


@pytest.fixture(autouse=True)
def _fake_settings(monkeypatch):
    """LocalPaperEngine and ExecutionService both call the cached get_settings() (for
    paper_wallet_balance/long_only/cost model and for _resolve_mode() respectively) even
    when most of what they need is passed explicitly. This worktree has no .env and no
    GROQ_API_KEY (a required field with no default), so — same lesson as
    tests/test_config_failclosed.py's `_base_kwargs` — construct one explicit, hermetic
    Settings object (``_env_file=None`` so it can't pick up the real .env) and patch it
    directly into the two modules that call get_settings(), rather than relying on some
    other test file's import order to have already warmed the lru_cache.
    """
    settings = Settings(groq_api_key="test-key", _env_file=None)
    monkeypatch.setattr("src.execution.paper_engine.get_settings", lambda: settings)
    monkeypatch.setattr("src.execution.service.get_settings", lambda: settings)


def _db_url(tmp_path, name: str = "") -> str:
    # A distinct sqlite file per call — src/db/base.py caches engines per URL string, so
    # reusing a URL across tests would leak state between them.
    return f"sqlite:///{tmp_path}/{name or uuid.uuid4().hex}.db"


class _FakeExitManager:
    """Minimal stand-in for ExitManager — just enough for reconcile()'s bookkeeping."""

    def __init__(self) -> None:
        self.registered: dict[str, dict[str, Any]] = {}

    def get_position(self, position_id: str) -> dict[str, Any] | None:
        return self.registered.get(position_id)

    def register_position(self, *, position_id: str, **kwargs: Any) -> None:
        self.registered[position_id] = {"position_id": position_id, **kwargs}

    def unregister_position(self, position_id: str) -> None:
        self.registered.pop(position_id, None)

    def get_managed_positions(self):
        class _Pos:
            def __init__(self, position_id: str) -> None:
                self.position_id = position_id

        return [_Pos(pid) for pid in self.registered]


class _FakeMistakeClassifier:
    """Deterministic stand-in — the real classifier calls Groq, unsuitable for a unit test."""

    def classify(self, outcome):
        return None


class _FakeMemoryDB:
    """Captures mark_used calls instead of hitting Postgres — enough to prove lesson
    crediting happened as part of finalize, without needing a real AgentMemoryDB."""

    def __init__(self) -> None:
        self.marked: list[tuple[str, bool]] = []

    def mark_used(self, lesson_id: str, was_successful: bool = False) -> None:
        self.marked.append((lesson_id, was_successful))


def _open_trade(
    lifecycle: PaperTradeLifecycleStore,
    engine: LocalPaperEngine,
    service: ExecutionService,
    *,
    trade_id: str,
    symbol: str = "TEST",
    quantity: int = 10,
    entry_price: float = 100.0,
    stop_loss: float = 98.0,
    target_price: float = 110.0,
    journal: TradeJournal | None = None,
) -> None:
    """Replicate exactly the sequence run_live_trading.py performs for a new entry."""
    key = f"{MOCK_NAMESPACE}:{trade_id}:entry"
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="test-run",
        workflow_id="test-cycle",
        idempotency_key=key,
        signal_id="signal-1",
        symbol=symbol,
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        rationale=["test"],
        context={"regime": "trending_up"},
        trade_id=trade_id,
    )
    fill = service.submit(
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        price=entry_price,
        idempotency_key=key,
        trade_id=trade_id,
        position_id=trade_id,
        stop_loss=stop_loss,
        target_price=target_price,
        strategy="trend_following",
        reason="entry",
    )
    assert fill.filled
    lifecycle.mark_open(
        trade_id,
        order_id=fill.order_id,
        position_id=trade_id,
        quantity=quantity,
        fill_price=fill.fill_price,
        entry_charges=fill.entry_charges,
    )
    if journal is not None:
        # run_live_trading.py calls journal.record_trade right after exit_manager
        # registration — TradeFinalizer.finalize's journal.close_trade requires this row
        # to already exist (it updates, never inserts).
        journal.record_trade(
            {
                "symbol": symbol,
                "signal_type": "BUY",
                "strategy": "trend_following",
                "entry_price": fill.fill_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "target_price": target_price,
            },
            workflow_id="test-cycle",
            state={"regime": "trending_up"},
            trade_id=trade_id,
            run_id="test-run",
        )


# ---------------------------------------------------------------------------
# 1. Crash between the wallet fill and lifecycle_store.mark_open
# ---------------------------------------------------------------------------


def test_reconcile_repairs_intent_stuck_open_from_wallet_fill(tmp_path) -> None:
    """Entry filled in the paper engine (wallet position exists) but the process crashed
    before lifecycle_store.mark_open ran — the ledger is still INTENT. reconcile() must
    repair the SAME trade_id from the wallet's own data, not crash on a primary-key
    collision (the ledger's create_intent path was previously reused here) and not mint a
    second identity for the same position."""
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    service = ExecutionService(engine=engine, mode=ExecutionMode.LOCAL_PAPER, idempotency=IdempotencyStore(db_url))
    lifecycle = PaperTradeLifecycleStore(db_url)

    trade_id = lifecycle.new_trade_id()
    key = f"{MOCK_NAMESPACE}:{trade_id}:entry"
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="r",
        workflow_id="w",
        idempotency_key=key,
        signal_id="s",
        symbol="TEST",
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=10,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=110.0,
        rationale=[],
        context={},
        trade_id=trade_id,
    )
    fill = service.submit(
        symbol="TEST",
        side="BUY",
        quantity=10,
        price=100.0,
        idempotency_key=key,
        trade_id=trade_id,
        position_id=trade_id,
        reason="entry",
    )
    assert fill.filled
    # CRASH HERE — mark_open never runs. The ledger still says INTENT; the wallet already
    # holds the position.
    assert lifecycle.get(trade_id)["status"] == "INTENT"
    assert engine.get_positions()[0].position_id == trade_id

    exit_manager = _FakeExitManager()
    report = lifecycle.reconcile(engine.get_positions(), exit_manager)  # no exception

    assert report.repaired_open_from_wallet == 1
    assert report.imported_wallet_positions == 0
    record = lifecycle.get(trade_id)
    assert record["status"] == "OPEN"
    assert record["remaining_quantity"] == 10
    assert record["weighted_entry_price"] == pytest.approx(fill.fill_price)
    assert trade_id in exit_manager.registered
    # No duplicate trade_id was minted — exactly one lifecycle event stream.
    assert [e["event_type"] for e in lifecycle.events(trade_id)] == ["INTENT", "ENTRY_FILL"]


# ---------------------------------------------------------------------------
# 2/3. Orphan INTENT resolved from the durable order ledger
# ---------------------------------------------------------------------------


def test_reconcile_replays_full_close_from_order_ledger_when_intent_orphaned(tmp_path) -> None:
    """The entry AND the full exit both filled in the paper engine, but the lifecycle
    ledger never got past INTENT (crash before the very first lifecycle_store call after
    create_intent — the worst case). reconcile() must use the paper engine's durable order
    table (ground truth) to replay the whole sequence and land on CLOSED with the correct
    net P&L, instead of leaving an unexplained INTENT forever."""
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    service = ExecutionService(engine=engine, mode=ExecutionMode.LOCAL_PAPER, idempotency=IdempotencyStore(db_url))
    lifecycle = PaperTradeLifecycleStore(db_url)

    trade_id = lifecycle.new_trade_id()
    entry_key = f"{MOCK_NAMESPACE}:{trade_id}:entry"
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="r",
        workflow_id="w",
        idempotency_key=entry_key,
        signal_id="s",
        symbol="TEST",
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=10,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=110.0,
        rationale=[],
        context={},
        trade_id=trade_id,
    )
    entry_fill = service.submit(
        symbol="TEST", side="BUY", quantity=10, price=100.0,
        idempotency_key=entry_key, trade_id=trade_id, position_id=trade_id, reason="entry",
    )
    assert entry_fill.filled
    exit_fill = service.submit(
        symbol="TEST", side="SELL", quantity=10, price=110.0,
        idempotency_key=f"{entry_key}:exit", trade_id=trade_id, position_id=trade_id,
        reason="target_hit",
    )
    assert exit_fill.filled
    # CRASH — the ledger never advanced past INTENT despite both fills having happened.
    assert lifecycle.get(trade_id)["status"] == "INTENT"
    assert engine.get_positions() == []  # wallet already fully flat

    exit_manager = _FakeExitManager()
    report = lifecycle.reconcile(
        engine.get_positions(), exit_manager, order_lookup=engine.get_orders_for_trade
    )

    assert report.repaired_from_order_ledger == 1
    assert report.ambiguous == 0
    record = lifecycle.get(trade_id)
    assert record["status"] == "CLOSED"
    assert record["remaining_quantity"] == 0
    assert record["cumulative_pnl"] == pytest.approx(exit_fill.realized_pnl)
    # And it is now correctly picked up by the durable-outbox drain (finalize never ran).
    assert [r["trade_id"] for r in lifecycle.get_unfinalized_closed()] == [trade_id]


def test_reconcile_marks_never_filled_intent_rejected(tmp_path) -> None:
    """An INTENT was recorded but the order was never actually submitted/filled (e.g. crash
    before execution_service.submit_async was even called). No wallet position and no order
    ledger entry exist for it — reconcile() must resolve this deterministically to REJECTED
    rather than leaving a phantom INTENT that looks like a still-pending order forever."""
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    lifecycle = PaperTradeLifecycleStore(db_url)

    trade_id = lifecycle.new_trade_id()
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="r",
        workflow_id="w",
        idempotency_key=f"{MOCK_NAMESPACE}:{trade_id}:entry",
        signal_id="s",
        symbol="TEST",
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=10,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=110.0,
        rationale=[],
        context={},
        trade_id=trade_id,
    )

    exit_manager = _FakeExitManager()
    report = lifecycle.reconcile([], exit_manager, order_lookup=engine.get_orders_for_trade)

    assert report.resolved_never_filled == 1
    assert report.ambiguous == 0
    assert lifecycle.get(trade_id)["status"] == "REJECTED"


def test_reconcile_flags_ambiguous_without_order_lookup(tmp_path) -> None:
    """Without an order_lookup (e.g. a caller that can't reach the paper engine), an
    orphaned INTENT cannot be resolved — it must be surfaced as ambiguous, never silently
    dropped or silently resumed as if it were fine."""
    db_url = _db_url(tmp_path)
    lifecycle = PaperTradeLifecycleStore(db_url)
    trade_id = lifecycle.new_trade_id()
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="r",
        workflow_id="w",
        idempotency_key=f"k:{trade_id}",
        signal_id="s",
        symbol="TEST",
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=10,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=110.0,
        rationale=[],
        context={},
        trade_id=trade_id,
    )

    report = lifecycle.reconcile([], _FakeExitManager())  # no order_lookup

    assert report.ambiguous == 1
    assert report.in_sync is False
    assert lifecycle.get(trade_id)["status"] == "INTENT"  # untouched, not guessed at


# ---------------------------------------------------------------------------
# 4. Partial + final exit -> exactly one lifecycle outcome, correct cumulative net P&L
# ---------------------------------------------------------------------------


def test_partial_and_final_exit_produce_one_outcome_with_correct_cumulative_pnl(tmp_path) -> None:
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    service = ExecutionService(engine=engine, mode=ExecutionMode.LOCAL_PAPER, idempotency=IdempotencyStore(db_url))
    lifecycle = PaperTradeLifecycleStore(db_url)
    journal = TradeJournal(database_url=db_url, namespace=MOCK_NAMESPACE)
    perf_tracker = PerformanceTracker(database_url=db_url, namespace=MOCK_NAMESPACE)
    finalizer = TradeFinalizer(lifecycle_store=lifecycle, perf_tracker=perf_tracker, journal=journal)

    trade_id = lifecycle.new_trade_id()
    _open_trade(lifecycle, engine, service, trade_id=trade_id, quantity=10, entry_price=100.0, journal=journal)

    # Partial exit: 4 shares @ 106.
    partial = service.submit(
        symbol="TEST", side="SELL", quantity=4, price=106.0,
        idempotency_key=f"{trade_id}:exit:1", trade_id=trade_id, position_id=trade_id,
        reason="partial",
    )
    assert partial.filled
    fully_closed = lifecycle.record_exit(
        trade_id, order_id=partial.order_id, quantity=4, exit_price=106.0,
        realized_pnl=partial.realized_pnl, reason="partial",
    )
    assert fully_closed is False
    # Partial exits must NEVER be recorded as a complete trade outcome (the exact C-5
    # defect) — no finalize call happens here in the real loop, and none here either.

    # Final exit: remaining 6 shares @ 112.
    final = service.submit(
        symbol="TEST", side="SELL", quantity=6, price=112.0,
        idempotency_key=f"{trade_id}:exit:2", trade_id=trade_id, position_id=trade_id,
        reason="target_hit",
    )
    assert final.filled
    fully_closed = lifecycle.record_exit(
        trade_id, order_id=final.order_id, quantity=6, exit_price=112.0,
        realized_pnl=final.realized_pnl, reason="target_hit",
    )
    assert fully_closed is True

    record = lifecycle.get(trade_id)
    expected_cumulative = partial.realized_pnl + final.realized_pnl
    assert record["cumulative_pnl"] == pytest.approx(expected_cumulative)

    assert finalizer.finalize(trade_id, exit_price=112.0, exit_reason="target_hit") is True

    # Exactly one performance outcome for the whole lifecycle, with the correct cumulative
    # net P&L — not two (one per leg).
    summary = perf_tracker.get_summary()
    assert summary["total_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(expected_cumulative)

    journaled = journal.get_trade(trade_id)
    assert journaled is not None
    assert journaled["status"] == "closed"
    assert journaled["profit_loss"] == pytest.approx(expected_cumulative)
    assert journaled["entry_quantity"] == 10  # original_quantity, not just the final leg


# ---------------------------------------------------------------------------
# 5. Crash between ledger CLOSED and finalize -> startup replay drains it, exactly once
# ---------------------------------------------------------------------------


def test_finalize_replay_after_crash_between_close_and_fanout_is_idempotent(tmp_path) -> None:
    """The ledger recorded CLOSED (durable), but the process crashed before the journal/
    performance/lesson fan-out ran. On 'restart', TradeFinalizer.replay_unfinalized must
    drive that fan-out exactly once — and a second replay (e.g. a flaky double-start) must
    not double-count the trade into performance stats or double-credit lessons."""
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    service = ExecutionService(engine=engine, mode=ExecutionMode.LOCAL_PAPER, idempotency=IdempotencyStore(db_url))
    lifecycle = PaperTradeLifecycleStore(db_url)
    journal = TradeJournal(database_url=db_url, namespace=MOCK_NAMESPACE)
    perf_tracker = PerformanceTracker(database_url=db_url, namespace=MOCK_NAMESPACE)
    fake_memory_db = _FakeMemoryDB()
    finalizer = TradeFinalizer(
        lifecycle_store=lifecycle,
        perf_tracker=perf_tracker,
        journal=journal,
        enable_learning=True,
        mistake_classifier=_FakeMistakeClassifier(),
        memory_injector=object(),  # never dereferenced when classify() returns None
        memory_db=fake_memory_db,
    )

    trade_id = lifecycle.new_trade_id()
    _open_trade(lifecycle, engine, service, trade_id=trade_id, quantity=5, entry_price=50.0, journal=journal)
    # Seed an "active lesson" the way run_live_trading.py's create_intent call does (there
    # is no public setter post-creation, so reach into the row directly), so we can prove
    # crediting happens exactly once across the crash + replay.
    from src.execution.lifecycle import PaperTradeLifecycleRecord

    session = lifecycle._session()
    try:
        row = session.get(PaperTradeLifecycleRecord, trade_id)
        row.active_lessons_json = '["lesson-1", "lesson-2"]'
        session.commit()
    finally:
        session.close()

    close = service.submit(
        symbol="TEST", side="SELL", quantity=5, price=55.0,
        idempotency_key=f"{trade_id}:exit", trade_id=trade_id, position_id=trade_id,
        reason="target_hit",
    )
    assert close.filled
    fully_closed = lifecycle.record_exit(
        trade_id, order_id=close.order_id, quantity=5, exit_price=55.0,
        realized_pnl=close.realized_pnl, reason="target_hit",
    )
    assert fully_closed is True
    # CRASH — finalize() is never called here, simulating the process dying right after the
    # ledger commit. This is exactly the C-2/C-5 gap: journal/performance/lessons would be
    # lost forever without a replay mechanism.
    assert lifecycle.get(trade_id)["finalized_at"] is None

    pending = lifecycle.get_unfinalized_closed(MOCK_NAMESPACE)
    assert [r["trade_id"] for r in pending] == [trade_id]

    replayed = finalizer.replay_unfinalized(MOCK_NAMESPACE)
    assert [r["trade_id"] for r in replayed] == [trade_id]
    assert lifecycle.get(trade_id)["finalized_at"] is not None
    assert perf_tracker.get_summary()["total_trades"] == 1
    assert journal.get_trade(trade_id)["status"] == "closed"
    assert fake_memory_db.marked == [("lesson-1", True), ("lesson-2", True)]

    # Second replay (nothing left pending) must be a true no-op: no double-count.
    second = finalizer.replay_unfinalized(MOCK_NAMESPACE)
    assert second == []
    assert perf_tracker.get_summary()["total_trades"] == 1
    assert fake_memory_db.marked == [("lesson-1", True), ("lesson-2", True)]

    # And an explicit repeat finalize() call for the same (already-finalized) trade_id is
    # also a safe no-op.
    assert finalizer.finalize(trade_id) is False
    assert perf_tracker.get_summary()["total_trades"] == 1


# ---------------------------------------------------------------------------
# 6. Kill-switch flatten produces the same complete outcome records as a normal close
# ---------------------------------------------------------------------------


def test_kill_switch_reason_produces_same_finalize_shape_as_normal_close(tmp_path) -> None:
    """TradeFinalizer.finalize is reason-agnostic: a trade closed with reason="kill_switch"
    must be journaled, performance-recorded, and lesson-credited exactly like one closed
    with reason="target_hit" — no shortcut that skips the fan-out for an emergency flatten
    (the C-5 finding). scripts/run_live_trading.py routes both normal exits and the
    kill-switch flatten loop through the same _execute_managed_exit -> TradeFinalizer.finalize
    call (see the shared helper there); this test proves that shared call path itself treats
    both reasons identically."""
    db_url = _db_url(tmp_path)
    engine = LocalPaperEngine(initial_balance=1_000_000.0, database_url=db_url, cost_model=CostModel.zero())
    service = ExecutionService(engine=engine, mode=ExecutionMode.LOCAL_PAPER, idempotency=IdempotencyStore(db_url))
    lifecycle = PaperTradeLifecycleStore(db_url)
    journal = TradeJournal(database_url=db_url, namespace=MOCK_NAMESPACE)
    perf_tracker = PerformanceTracker(database_url=db_url, namespace=MOCK_NAMESPACE)
    finalizer = TradeFinalizer(lifecycle_store=lifecycle, perf_tracker=perf_tracker, journal=journal)

    outcomes = {}
    for label, reason in (("normal", "target_hit"), ("emergency", "kill_switch")):
        trade_id = lifecycle.new_trade_id()
        _open_trade(lifecycle, engine, service, trade_id=trade_id, quantity=3, entry_price=200.0, journal=journal)
        close = service.submit(
            symbol="TEST", side="SELL", quantity=3, price=210.0,
            idempotency_key=f"{trade_id}:exit", trade_id=trade_id, position_id=trade_id,
            reason=reason,
        )
        assert close.filled
        fully_closed = lifecycle.record_exit(
            trade_id, order_id=close.order_id, quantity=3, exit_price=210.0,
            realized_pnl=close.realized_pnl, reason=reason,
        )
        assert fully_closed is True
        assert finalizer.finalize(trade_id, exit_price=210.0, exit_reason=reason) is True
        outcomes[label] = (trade_id, close.realized_pnl)

    normal_id, normal_pnl = outcomes["normal"]
    kill_id, kill_pnl = outcomes["emergency"]

    normal_journal = journal.get_trade(normal_id)
    kill_journal = journal.get_trade(kill_id)
    assert normal_journal["status"] == kill_journal["status"] == "closed"
    assert normal_journal["exit_reason"] == "target_hit"
    assert kill_journal["exit_reason"] == "kill_switch"
    assert normal_pnl == pytest.approx(kill_pnl)  # identical fills -> identical P&L

    summary = perf_tracker.get_summary()
    assert summary["total_trades"] == 2  # both credited — kill switch didn't skip perf
    assert lifecycle.get(normal_id)["finalized_at"] is not None
    assert lifecycle.get(kill_id)["finalized_at"] is not None


# ---------------------------------------------------------------------------
# 7. PerformanceTracker.record_trade idempotency (defense in depth for C-5)
# ---------------------------------------------------------------------------


def test_performance_tracker_record_trade_idempotent_by_trade_id(tmp_path) -> None:
    tracker = PerformanceTracker(database_url=_db_url(tmp_path), namespace=MOCK_NAMESPACE)
    tracker.record_trade(
        strategy="trend_following", regime="trending_up", pnl=100.0, pnl_pct=5.0,
        symbol="TEST", trade_id="dup-trade",
    )
    # A replay (e.g. a duplicated finalize call) with the SAME trade_id must be a no-op,
    # even if (implausibly) the pnl argument differs — the first recording wins.
    tracker.record_trade(
        strategy="trend_following", regime="trending_up", pnl=999.0, pnl_pct=99.0,
        symbol="TEST", trade_id="dup-trade",
    )
    summary = tracker.get_summary()
    assert summary["total_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(100.0)


def test_performance_tracker_record_trade_survives_restart_dedup(tmp_path) -> None:
    """A trade_id already committed to Postgres by a prior process (not just this
    in-memory cache) must also be deduped — the fresh-process case, not just the
    same-process replay case."""
    db_url = _db_url(tmp_path)
    first = PerformanceTracker(database_url=db_url, namespace=MOCK_NAMESPACE)
    first.record_trade(
        strategy="momentum", regime="ranging", pnl=50.0, pnl_pct=2.0,
        symbol="TEST", trade_id="restart-trade",
    )

    second = PerformanceTracker(database_url=db_url, namespace=MOCK_NAMESPACE)  # fresh load
    second.record_trade(
        strategy="momentum", regime="ranging", pnl=50.0, pnl_pct=2.0,
        symbol="TEST", trade_id="restart-trade",
    )
    assert second.get_summary()["total_trades"] == 1
