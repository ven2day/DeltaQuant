from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.core.indicators import Timeframe
from src.markets.nse.execution.averaging import CappedAveragingPolicy
from src.markets.nse.execution.lifecycle import MOCK_NAMESPACE, PaperTradeLifecycleStore
from src.markets.nse.execution.paper_engine import LocalPaperEngine
from src.markets.nse.execution.service import ExecutionMode, ExecutionService, IdempotencyStore
from src.markets.nse.market_data.simulated import SimulatedMarketData
from src.markets.nse.risk.costs import CostModel
from src.markets.nse.risk.pretrade import FinalPaperOrder, PaperRiskReservations
from src.markets.nse.strategies.candidate_policy import CandidateAction, evaluate_long_candidate


@pytest.fixture(autouse=True)
def _fake_settings(monkeypatch):
    """LocalPaperEngine/ExecutionService both call the cached get_settings(), which
    requires GROQ_API_KEY (a required field with no default) — this worktree has no
    .env, so without this the test below only passes when some *other* test file happens
    to run first and warm the lru_cache (order-dependent flakiness). Same fix as
    tests/test_config_failclosed.py's ``_base_kwargs``/``_env_file=None`` pattern, applied
    at the get_settings() call sites instead of at direct Settings() construction.
    """
    settings = Settings(groq_api_key="test-key", _env_file=None)
    monkeypatch.setattr("src.markets.nse.execution.paper_engine.get_settings", lambda: settings)
    monkeypatch.setattr("src.markets.nse.execution.service.get_settings", lambda: settings)


def _trend_frame(periods: int, frequency: str, volume_spike: bool = False) -> pd.DataFrame:
    index = pd.date_range("2026-07-01 09:15", periods=periods, freq=frequency)
    close = np.linspace(100.0, 125.0, periods)
    volume = np.full(periods, 1_000.0)
    if volume_spike:
        volume[-1] = 2_000.0
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_coherent_simulator_derives_quotes_and_timeframes_from_one_stream() -> None:
    simulator = SimulatedMarketData(symbols=["TEST"], seed=7, history_bars=3000)
    simulator.tick()
    five = simulator.get_history("TEST", "5m")
    fifteen = simulator.get_history("TEST", "15m")
    quote = simulator.get_quotes(["TEST"])["TEST"]

    assert five is not None and fifteen is not None
    assert quote.last_price == round(float(five.iloc[-1]["close"]), 2)
    assert quote.timestamp == simulator.event_time
    assert len(fifteen) < len(five)


def test_candidate_policy_buy_wait_reject_contract() -> None:
    histories = {
        Timeframe.M5: _trend_frame(240, "5min", volume_spike=True),
        Timeframe.M15: _trend_frame(100, "15min"),
        Timeframe.M30: _trend_frame(100, "30min"),
        Timeframe.H1: _trend_frame(100, "1h"),
        Timeframe.H4: _trend_frame(100, "4h"),
    }
    signal = {
        "symbol": "TEST",
        "signal_type": "BUY",
        "entry_price": 125.0,
        "stop_loss": 121.0,
    }
    buy = evaluate_long_candidate(signal, histories, {"direction": "up", "confidence": 0.8})
    pre_ml = evaluate_long_candidate(signal, histories, None, require_ml=False)
    missing_ml = evaluate_long_candidate(signal, histories, None)
    ml_required = evaluate_long_candidate(signal, histories, None, require_ml=True)
    wait = evaluate_long_candidate(
        signal,
        histories,
        {"direction": "down", "confidence": 0.8},
        require_ml=True,
    )
    reject = evaluate_long_candidate(
        {**signal, "signal_type": "SELL"},
        histories,
        {"direction": "up", "confidence": 0.8},
    )

    assert buy.action == CandidateAction.BUY
    assert pre_ml.action == CandidateAction.BUY
    assert pre_ml.ml_direction == "unknown"
    assert missing_ml.action == CandidateAction.BUY
    assert missing_ml.ml_direction == "unknown"
    assert ml_required.action == CandidateAction.WAIT
    assert buy.target_price == 125.0 * 1.035
    assert wait.action == CandidateAction.WAIT
    assert reject.action == CandidateAction.REJECT


def test_batch_reservation_enforces_zero_size_daily_and_aggregate_limits() -> None:
    reservations = PaperRiskReservations()
    zero = FinalPaperOrder("A", "BUY", 0, 100.0, 98.0, 104.0, "test")
    assert not reservations.evaluate(
        zero,
        positions=[],
        equity=100_000.0,
        entries_today=0,
        daily_entry_cap=6,
        max_positions=5,
        max_position_pct=0.10,
        max_total_exposure_pct=50.0,
        risk_per_trade=0.02,
        max_total_risk=0.10,
    ).approved

    first = FinalPaperOrder("A", "BUY", 100, 100.0, 98.0, 104.0, "test")
    assert reservations.evaluate(
        first,
        positions=[],
        equity=100_000.0,
        entries_today=0,
        daily_entry_cap=6,
        max_positions=5,
        max_position_pct=0.10,
        max_total_exposure_pct=50.0,
        risk_per_trade=0.02,
        max_total_risk=0.10,
    ).approved
    reservations.reserve(first)
    duplicate = FinalPaperOrder("A", "BUY", 1, 100.0, 98.0, 104.0, "test")
    decision = reservations.evaluate(
        duplicate,
        positions=[],
        equity=100_000.0,
        entries_today=0,
        daily_entry_cap=6,
        max_positions=5,
        max_position_pct=0.10,
        max_total_exposure_pct=50.0,
        risk_per_trade=0.02,
        max_total_risk=0.10,
    )
    assert not decision.approved
    assert any("already exists" in reason for reason in decision.reasons)


def test_canonical_lifecycle_tracks_entry_add_and_exit(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path}/paper.db"
    engine = LocalPaperEngine(
        initial_balance=1_000_000.0,
        database_url=database_url,
        cost_model=CostModel.zero(),
    )
    service = ExecutionService(
        engine=engine,
        mode=ExecutionMode.LOCAL_PAPER,
        idempotency=IdempotencyStore(database_url),
    )
    lifecycle = PaperTradeLifecycleStore(database_url)
    trade_id = lifecycle.new_trade_id()
    key = f"{MOCK_NAMESPACE}:test-entry"
    lifecycle.create_intent(
        namespace=MOCK_NAMESPACE,
        run_id="test-run",
        workflow_id="test-cycle",
        idempotency_key=key,
        signal_id="signal-1",
        symbol="TEST",
        side="BUY",
        strategy="trend_following",
        timeframe="5m",
        quantity=10,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=103.5,
        rationale=["test"],
        context={"regime": "trending_up"},
        trade_id=trade_id,
    )
    entry = service.submit(
        symbol="TEST",
        side="BUY",
        quantity=10,
        price=100.0,
        idempotency_key=key,
        trade_id=trade_id,
        position_id=trade_id,
        stop_loss=98.0,
        target_price=103.5,
        strategy="trend_following",
    )
    assert entry.filled
    lifecycle.mark_open(
        trade_id,
        order_id=entry.order_id,
        position_id=trade_id,
        quantity=10,
        fill_price=entry.fill_price,
    )

    exit_fill = service.submit(
        symbol="TEST",
        side="SELL",
        quantity=10,
        price=103.5,
        idempotency_key=f"{key}:exit",
        trade_id=trade_id,
        position_id=trade_id,
        reason="target_hit",
    )
    assert exit_fill.filled
    assert lifecycle.record_exit(
        trade_id,
        order_id=exit_fill.order_id,
        quantity=10,
        exit_price=exit_fill.fill_price,
        realized_pnl=exit_fill.realized_pnl,
        reason="target_hit",
    )
    record = lifecycle.get(trade_id)
    assert record is not None and record["status"] == "CLOSED"
    assert [event["event_type"] for event in lifecycle.events(trade_id)] == [
        "INTENT",
        "ENTRY_FILL",
        "EXIT_FILL",
    ]


def test_averaging_never_exceeds_configured_add_count() -> None:
    policy = CappedAveragingPolicy(enabled=True, max_adds=1, trigger_pct=1.0, add_fraction=0.25)
    first = policy.evaluate(
        original_quantity=20,
        add_count=0,
        weighted_entry_price=100.0,
        current_price=98.5,
        stop_loss=97.0,
        target_price=103.5,
    )
    second = policy.evaluate(
        original_quantity=20,
        add_count=1,
        weighted_entry_price=99.7,
        current_price=98.0,
        stop_loss=97.0,
        target_price=103.5,
    )
    assert first.should_add and first.quantity == 5
    assert not second.should_add
