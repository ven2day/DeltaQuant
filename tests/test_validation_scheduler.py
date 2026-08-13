import asyncio
import json
import time
from types import SimpleNamespace

from src.backtesting.validation_scheduler import (
    AutomaticValidationScheduler,
    _default_swing_validation_intervals,
)


def _settings(tmp_path):
    return SimpleNamespace(
        strategy_eligibility_registry_dir=str(tmp_path),
        strategy_validation_auto_run=True,
        strategy_validation_refresh_hours=24.0,
    )


def test_scheduler_state_path_under_eligibility_registry_dir(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", lambda: settings)

    scheduler = AutomaticValidationScheduler()

    assert scheduler._state_path.parent == tmp_path
    assert scheduler._state_path.name == "auto_validation_state.json"


def test_scheduler_does_not_repeat_before_refresh_interval(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticValidationScheduler()

    assert scheduler._is_due()

    scheduler._state_path.write_text(
        json.dumps({"last_completed_epoch": time.time()}), encoding="utf-8"
    )

    assert not scheduler._is_due()


def test_maybe_start_noop_when_auto_run_disabled(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.strategy_validation_auto_run = False
    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticValidationScheduler()

    assert scheduler.maybe_start() is False
    assert not scheduler.is_running


def test_default_intervals_derived_from_signal_timeframes(monkeypatch):
    # Must match settings.signal_timeframes (the live SWING scan's actual
    # timeframes) -- a prior hardcoded list omitted "5m" even though it's
    # where most top-ranked live candidates actually fire, silently leaving
    # them permanently UNVALIDATED rather than SHADOW. Catching a mismatch
    # here is cheaper than discovering a missing timeframe live.
    settings = SimpleNamespace(signal_timeframes="5m,15m,30m,1h,4h")
    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", lambda: settings)
    assert _default_swing_validation_intervals() == ("5m", "15m", "30m", "1h", "4h")


def test_default_intervals_falls_back_when_settings_unavailable(monkeypatch):
    def _raise() -> None:
        raise RuntimeError("settings not configured")

    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", _raise)
    assert _default_swing_validation_intervals() == ("15m", "30m", "1h", "4h", "1d")


async def test_force_run_refuses_to_double_start(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.backtesting.validation_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticValidationScheduler()

    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run() -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(scheduler, "_run", _fake_run)

    assert scheduler.force_run() is True
    await started.wait()
    assert scheduler.is_running is True
    assert scheduler.force_run() is False

    release.set()
    await scheduler._task
