"""Tests for the IST-calendar-day-scoped daily risk counters (src/risk/daily_state.py)."""

from datetime import datetime

from src.markets.nse.risk import daily_state as daily_state_module
from src.markets.nse.risk.daily_state import DailyRiskStore


def test_get_today_on_fresh_store_returns_zeros(tmp_path):
    store = DailyRiskStore(database_url=f"sqlite:///{tmp_path}/risk.db")

    today = store.get_today()

    assert today == {"trades_count": 0, "profit_loss": 0.0, "max_drawdown": 0.0}


def test_record_trade_increments_count_and_accumulates_pnl(tmp_path):
    store = DailyRiskStore(database_url=f"sqlite:///{tmp_path}/risk.db")

    store.record_trade(100.0)
    store.record_trade(-40.0)

    today = store.get_today()
    assert today["trades_count"] == 2
    assert today["profit_loss"] == 60.0


def test_update_drawdown_keeps_the_max_not_the_latest(tmp_path):
    store = DailyRiskStore(database_url=f"sqlite:///{tmp_path}/risk.db")

    store.update_drawdown(500.0)
    store.update_drawdown(200.0)  # a later, smaller drawdown must not overwrite the peak

    assert store.get_today()["max_drawdown"] == 500.0

    store.update_drawdown(800.0)  # a new, deeper drawdown does update it
    assert store.get_today()["max_drawdown"] == 800.0


def test_state_survives_across_separate_store_instances(tmp_path):
    """The whole point: a restart (a new DailyRiskStore instance) must not reset today's count."""
    database_url = f"sqlite:///{tmp_path}/risk.db"
    store1 = DailyRiskStore(database_url=database_url)
    store1.record_trade(50.0)

    store2 = DailyRiskStore(database_url=database_url)  # simulates a process restart
    assert store2.get_today()["trades_count"] == 1
    assert store2.get_today()["profit_loss"] == 50.0


def test_different_ist_calendar_days_get_independent_rows(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path}/risk.db"
    store = DailyRiskStore(database_url=database_url)

    monkeypatch.setattr(daily_state_module, "now_ist", lambda: datetime(2026, 8, 5, 10, 0, 0))
    store.record_trade(100.0)
    assert store.get_today()["trades_count"] == 1

    # A new day must start fresh — this is exactly what an in-memory counter gets wrong by
    # instead resetting on every restart, not just at the actual day boundary.
    monkeypatch.setattr(daily_state_module, "now_ist", lambda: datetime(2026, 8, 6, 10, 0, 0))
    assert store.get_today() == {"trades_count": 0, "profit_loss": 0.0, "max_drawdown": 0.0}

    monkeypatch.setattr(daily_state_module, "now_ist", lambda: datetime(2026, 8, 5, 14, 0, 0))
    assert store.get_today()["trades_count"] == 1  # yesterday's row is untouched
