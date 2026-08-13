from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd

from src.core.indicators import Timeframe
from src.markets.nse.execution.exit_manager import ExitManager
from src.markets.nse.runtime.state import HistoryReadiness, MarketStateStore
from src.memory.injection import MemoryInjector


def _frame(end: str, frequency: str = "5min", bars: int = 60) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=bars, freq=frequency, tz="Asia/Kolkata")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        index=index,
    )


def test_no_new_candle_produces_no_processing_event():
    store = MarketStateStore()
    frames = {("RELIANCE", Timeframe.M5): _frame("2026-08-10 10:20")}
    first, _ = store.plan_events(frames)
    store.mark_processed(first[0], features=object(), feature_version="v1")

    second, metrics = store.plan_events(frames)

    assert second == []
    assert metrics.skipped_unchanged == 1


def test_new_5m_does_not_recalculate_unchanged_higher_timeframes():
    store = MarketStateStore()
    initial = {
        ("RELIANCE", Timeframe.M5): _frame("2026-08-10 10:20", "5min"),
        ("RELIANCE", Timeframe.M15): _frame("2026-08-10 10:15", "15min"),
        ("RELIANCE", Timeframe.M30): _frame("2026-08-10 10:00", "30min"),
        ("RELIANCE", Timeframe.H1): _frame("2026-08-10 10:00", "1h"),
    }
    events, _ = store.plan_events(initial)
    cached = {}
    for event in events:
        feature = object()
        cached[event.timeframe] = feature
        store.mark_processed(event, features=feature, feature_version="v1")

    changed = dict(initial)
    changed[("RELIANCE", Timeframe.M5)] = _frame("2026-08-10 10:25", "5min")
    events, _ = store.plan_events(changed)

    assert [(event.symbol, event.timeframe) for event in events] == [
        ("RELIANCE", Timeframe.M5)
    ]
    context = store.cached_context("RELIANCE")
    assert context[Timeframe.M15] is cached[Timeframe.M15]
    assert context[Timeframe.M30] is cached[Timeframe.M30]
    assert context[Timeframe.H1] is cached[Timeframe.H1]


def test_new_15m_triggers_only_15m_processing():
    store = MarketStateStore()
    frames = {
        ("TCS", Timeframe.M5): _frame("2026-08-10 10:20", "5min"),
        ("TCS", Timeframe.M15): _frame("2026-08-10 10:15", "15min"),
        ("TCS", Timeframe.H1): _frame("2026-08-10 10:00", "1h"),
    }
    events, _ = store.plan_events(frames)
    for event in events:
        store.mark_processed(event)
    frames[("TCS", Timeframe.M15)] = _frame("2026-08-10 10:30", "15min")

    events, _ = store.plan_events(frames)

    assert [event.timeframe for event in events] == [Timeframe.M15]


def test_backfilling_series_cannot_emit_a_signal_event():
    store = MarketStateStore()
    store.set_readiness("TCS", Timeframe.M5, HistoryReadiness.BACKFILLING)

    events, _ = store.plan_events({("TCS", Timeframe.M5): _frame("2026-08-10 10:20")})

    assert events == []
    assert store.timeframe_state("TCS", Timeframe.M5).readiness is HistoryReadiness.BACKFILLING


def test_position_exit_remains_active_when_no_candle_changes(tmp_path):
    state = MarketStateStore()
    frame = _frame("2026-08-10 10:20")
    event = state.plan_events({("RELIANCE", Timeframe.M5): frame})[0][0]
    state.mark_processed(event)
    assert state.plan_events({("RELIANCE", Timeframe.M5): frame})[0] == []

    manager = ExitManager(state_file=tmp_path / "exit-state.json")
    manager.register_position(
        position_id="P1",
        symbol="RELIANCE",
        side="BUY",
        quantity=1,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
    )
    exits = manager.check_exits({"RELIANCE": 94.0})

    assert len(exits) == 1
    assert exits[0][1].should_exit is True


def test_intraday_square_off_forces_exit_after_cutoff(tmp_path):
    manager = ExitManager(
        state_file=tmp_path / "exit-state.json",
        max_hold_minutes=99999,  # isolate: only the square-off rule should fire here
        intraday_square_off_time="15:15",
    )
    manager.register_position(
        position_id="P1",
        symbol="RELIANCE",
        side="BUY",
        quantity=1,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        timeframe="5m",
        entry_time=datetime.now(),
    )
    with patch(
        "src.markets.nse.execution.exit_manager.now_ist",
        return_value=datetime(2026, 8, 13, 15, 20),
    ):
        exits = manager.check_exits({"RELIANCE": 100.0})  # flat price -- no stop/target hit

    assert len(exits) == 1
    _, rule = exits[0]
    assert rule.should_exit is True
    assert rule.exit_type == "time_exit"
    assert "square-off" in rule.reason.lower()


def test_intraday_square_off_does_not_fire_before_cutoff(tmp_path):
    manager = ExitManager(
        state_file=tmp_path / "exit-state.json",
        max_hold_minutes=99999,
        intraday_square_off_time="15:15",
    )
    manager.register_position(
        position_id="P1",
        symbol="RELIANCE",
        side="BUY",
        quantity=1,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        timeframe="5m",
        entry_time=datetime.now(),
    )
    with patch(
        "src.markets.nse.execution.exit_manager.now_ist",
        return_value=datetime(2026, 8, 13, 11, 0),
    ):
        exits = manager.check_exits({"RELIANCE": 100.0})

    assert exits == []


def test_intraday_square_off_does_not_apply_to_swing_timeframes(tmp_path):
    manager = ExitManager(
        state_file=tmp_path / "exit-state.json",
        max_hold_minutes=99999,
        intraday_square_off_time="15:15",
    )
    manager.register_position(
        position_id="P1",
        symbol="RELIANCE",
        side="BUY",
        quantity=1,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        timeframe="1h",
        entry_time=datetime.now(),
    )
    with patch(
        "src.markets.nse.execution.exit_manager.now_ist",
        return_value=datetime(2026, 8, 13, 18, 0),
    ):
        exits = manager.check_exits({"RELIANCE": 100.0})

    assert exits == []


def test_intraday_square_off_can_be_disabled(tmp_path):
    manager = ExitManager(
        state_file=tmp_path / "exit-state.json",
        max_hold_minutes=99999,
        intraday_square_off_enabled=False,
        intraday_square_off_time="15:15",
    )
    manager.register_position(
        position_id="P1",
        symbol="RELIANCE",
        side="BUY",
        quantity=1,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        timeframe="5m",
        entry_time=datetime.now(),
    )
    with patch(
        "src.markets.nse.execution.exit_manager.now_ist",
        return_value=datetime(2026, 8, 13, 18, 0),
    ):
        exits = manager.check_exits({"RELIANCE": 100.0})

    assert exits == []


def test_memory_is_advisory_and_cannot_mutate_validated_strategy_semantics():
    class _Memory:
        def get_top_lessons_for_context(self, **kwargs):
            return [{"lesson_id": "L1", "lesson": "Avoid weak breakouts"}]

        def mark_used(self, lesson_id):
            return None

    state = {
        "regime": "trending_up",
        "active_strategies": ["momentum"],
        "strategy_version": "momentum-v7",
        "stop_loss_pct": 1.0,
        "position_size_pct": 2.0,
        "risk_limit": 0.01,
    }
    before = dict(state)

    update = MemoryInjector(memory_db=_Memory()).inject_lessons(state)

    assert state == before
    assert set(update) == {"memory_lessons"}
