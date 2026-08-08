"""Tests for the persistent signal-history log (src/execution/signal_log.py)."""

from datetime import date, datetime, timedelta

import pandas as pd

from scripts import backfill_signal_history
from src.execution.signal_log import SignalLogger, SignalRecord

# --- SignalRecord.from_signal ---


def test_from_signal_approved_has_no_reason():
    signal = {
        "symbol": "TCS",
        "signal_type": "BUY",
        "entry_price": 3450.5,
        "timeframe": "15m",
        "strategy": "trend_following",
        "confidence": 0.72,
        "timestamp": "2026-08-05T10:00:00",
    }
    record = SignalRecord.from_signal(signal, "approved")

    assert record.symbol == "TCS"
    assert record.side == "BUY"
    assert record.entry_price == 3450.5
    assert record.timeframe == "15m"
    assert record.strategy == "trend_following"
    assert record.confidence == 0.72
    assert record.status == "approved"
    assert record.reason == ""
    assert record.source == "live"


def test_from_signal_preserves_explicit_backfill_metadata():
    record = SignalRecord.from_signal(
        {"symbol": "TCS", "signal_type": "BUY", "entry_price": 3450.5},
        "approved",
        reason="Historical raw signal; validation and risk checks were not replayed.",
        source="backfill",
    )

    assert record.source == "backfill"
    assert record.reason.startswith("Historical raw signal")


def test_replay_signals_marks_records_as_backfill_without_using_future_bars(monkeypatch):
    class FakeSignal:
        def to_dict(self):
            return {
                "symbol": "TCS",
                "signal_type": "BUY",
                "entry_price": 3450.5,
                "timeframe": "1d",
                "strategy": "momentum",
                "confidence": 0.72,
            }

    class FakeEngine:
        def generate_signals(self, indicators):
            return [FakeSignal()]

    seen_prefix_lengths: list[int] = []

    def fake_indicators(prefix, symbol, timeframe):
        seen_prefix_lengths.append(len(prefix))
        return object()

    monkeypatch.setattr(backfill_signal_history, "calculate_indicators", fake_indicators)
    index = pd.date_range("2026-08-01", periods=30, freq="D")
    history = pd.DataFrame(
        {
            "open": range(30),
            "high": range(1, 31),
            "low": range(30),
            "close": range(30),
            "volume": [100] * 30,
        },
        index=index,
    )

    records = backfill_signal_history.replay_signals(
        history,
        "TCS",
        backfill_signal_history.Timeframe.D1,
        date(2026, 8, 29),
        date(2026, 8, 30),
        FakeEngine(),
    )

    assert seen_prefix_lengths == [29, 30]
    assert len(records) == 2
    assert {record.source for record in records} == {"backfill"}
    assert all(record.reason == backfill_signal_history.BACKFILL_REASON for record in records)
    assert all(record.timestamp.endswith("15:30:00") for record in records)


def test_from_signal_rejected_validation_extracts_reasoning():
    signal = {
        "symbol": "INFY",
        "signal_type": "SELL",
        "entry_price": 1500.0,
        "timestamp": "2026-08-05T10:05:00",
        "validation": {"decision": "reject", "reasoning": "Conflicting momentum signals"},
    }
    record = SignalRecord.from_signal(signal, "rejected_validation")

    assert record.reason == "Conflicting momentum signals"


def test_from_signal_rejected_risk_extracts_first_failure_message():
    signal = {
        "symbol": "RELIANCE",
        "signal_type": "BUY",
        "entry_price": 2500.0,
        "timestamp": "2026-08-05T10:10:00",
        "risk_result": {
            "approved": False,
            "failures": [{"message": "Outside trading hours"}, {"message": "Max positions"}],
        },
    }
    record = SignalRecord.from_signal(signal, "rejected_risk")

    assert record.reason == "Outside trading hours"


def test_from_signal_rejected_risk_prioritizes_strategy_admission_over_earlier_failures():
    """The H-8 admission failure (risk_compliance.py check 14, evaluated LAST by list
    order) must be the reported reason when present, even though an earlier check
    like trading-hours or risk-reward also failed first in the list -- otherwise a
    signal that could never be approved on any cycle reads as merely mistimed."""
    signal = {
        "symbol": "AMBER",
        "signal_type": "BUY",
        "entry_price": 433.31,
        "timestamp": "2026-08-09T07:11:00",
        "risk_result": {
            "approved": False,
            "failures": [
                {"rule": "trading_hours", "message": "Outside trading hours (09:15-15:15)"},
                {
                    "rule": "strategy_admission",
                    "message": "Strategy 'trend_following' has no current VALIDATED "
                    "registry artifact for regime 'trending_up' (H-8 strategy admission gate)",
                },
            ],
        },
    }
    record = SignalRecord.from_signal(signal, "rejected_risk")

    assert "strategy admission gate" in record.reason
    assert record.reason.startswith("Strategy 'trend_following'")


def test_from_signal_missing_timestamp_falls_back_to_now():
    signal = {"symbol": "WIPRO", "signal_type": "BUY", "entry_price": 400.0}
    record = SignalRecord.from_signal(signal, "approved")

    # Must not raise, and should produce a parseable ISO timestamp close to now.
    parsed = datetime.fromisoformat(record.timestamp)
    assert abs((datetime.now() - parsed).total_seconds()) < 5


# --- SignalLogger ---


def test_log_then_read_recent_round_trips(tmp_path):
    logger = SignalLogger(database_url=f"sqlite:///{tmp_path}/signals.db")
    record = SignalRecord.from_signal(
        {
            "symbol": "TCS",
            "signal_type": "BUY",
            "entry_price": 3450.5,
            "timeframe": "15m",
            "strategy": "trend_following",
            "confidence": 0.72,
            "timestamp": datetime.now().isoformat(),
        },
        "approved",
    )

    logger.log(record)
    results = logger.read_recent(days=7)

    assert len(results) == 1
    assert results[0]["symbol"] == "TCS"
    assert results[0]["status"] == "approved"
    assert results[0]["source"] == "live"


def test_read_recent_excludes_files_older_than_cutoff(tmp_path):
    logger = SignalLogger(database_url=f"sqlite:///{tmp_path}/signals.db")
    old_day = datetime.now() - timedelta(days=10)
    recent_day = datetime.now() - timedelta(days=1)

    old_record = SignalRecord.from_signal(
        {
            "symbol": "OLD",
            "signal_type": "BUY",
            "entry_price": 1.0,
            "timestamp": old_day.isoformat(),
        },
        "approved",
    )
    recent_record = SignalRecord.from_signal(
        {
            "symbol": "RECENT",
            "signal_type": "BUY",
            "entry_price": 1.0,
            "timestamp": recent_day.isoformat(),
        },
        "approved",
    )
    logger.log(old_record)
    logger.log(recent_record)

    results = logger.read_recent(days=7)

    symbols = [r["symbol"] for r in results]
    assert "RECENT" in symbols
    assert "OLD" not in symbols


def test_read_recent_sorts_newest_first(tmp_path):
    logger = SignalLogger(database_url=f"sqlite:///{tmp_path}/signals.db")
    now = datetime.now()
    for i, symbol in enumerate(["FIRST", "SECOND", "THIRD"]):
        logger.log(
            SignalRecord.from_signal(
                {
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "entry_price": 1.0,
                    "timestamp": (now - timedelta(hours=2 - i)).isoformat(),
                },
                "approved",
            )
        )

    results = logger.read_recent(days=7)

    assert [r["symbol"] for r in results] == ["THIRD", "SECOND", "FIRST"]


def test_read_recent_on_empty_log_dir_returns_empty_list(tmp_path):
    logger = SignalLogger(database_url=f"sqlite:///{tmp_path}/does_not_exist_yet.db")
    assert logger.read_recent(days=7) == []


def test_prune_deletes_only_files_older_than_cutoff(tmp_path):
    logger = SignalLogger(database_url=f"sqlite:///{tmp_path}/signals.db")
    old_day = datetime.now() - timedelta(days=10)
    recent_day = datetime.now() - timedelta(days=1)

    logger.log(
        SignalRecord.from_signal(
            {
                "symbol": "OLD",
                "signal_type": "BUY",
                "entry_price": 1.0,
                "timestamp": old_day.isoformat(),
            },
            "approved",
        )
    )
    logger.log(
        SignalRecord.from_signal(
            {
                "symbol": "RECENT",
                "signal_type": "BUY",
                "entry_price": 1.0,
                "timestamp": recent_day.isoformat(),
            },
            "approved",
        )
    )

    logger.prune(days=7)

    remaining = logger.read_recent(days=365)
    symbols = [r["symbol"] for r in remaining]
    assert symbols == ["RECENT"]
