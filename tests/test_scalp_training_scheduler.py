import asyncio
import json
import time
from types import SimpleNamespace

from src.agents.scalp_training_scheduler import AutomaticScalpTrainingScheduler


def _settings():
    return SimpleNamespace(
        scalp_training_auto_run=True,
        scalp_training_refresh_hours=24.0,
    )


def test_scheduler_does_not_repeat_before_refresh_interval(tmp_path, monkeypatch):
    settings = _settings()
    monkeypatch.setattr("src.agents.scalp_training_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticScalpTrainingScheduler()
    scheduler._state_path = tmp_path / "auto_scalp_training_state.json"

    assert scheduler._is_due()

    scheduler._state_path.write_text(
        json.dumps({"last_completed_epoch": time.time()}), encoding="utf-8"
    )

    assert not scheduler._is_due()


def test_maybe_start_noop_when_auto_run_disabled(tmp_path, monkeypatch):
    settings = _settings()
    settings.scalp_training_auto_run = False
    monkeypatch.setattr("src.agents.scalp_training_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticScalpTrainingScheduler()
    scheduler._state_path = tmp_path / "auto_scalp_training_state.json"

    assert scheduler.maybe_start() is False
    assert not scheduler.is_running


async def test_force_run_refuses_to_double_start(tmp_path, monkeypatch):
    settings = _settings()
    monkeypatch.setattr("src.agents.scalp_training_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticScalpTrainingScheduler()
    scheduler._state_path = tmp_path / "auto_scalp_training_state.json"

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


async def test_run_writes_error_status_when_training_rejected(tmp_path, monkeypatch):
    settings = _settings()
    monkeypatch.setattr("src.agents.scalp_training_scheduler.get_settings", lambda: settings)
    scheduler = AutomaticScalpTrainingScheduler()
    scheduler._state_path = tmp_path / "auto_scalp_training_state.json"

    def _raise() -> dict:
        raise RuntimeError("training rejected for: {'5m': 'insufficient OOS edge'}")

    monkeypatch.setattr(scheduler, "_train", _raise)

    await scheduler._run()

    payload = json.loads(scheduler._state_path.read_text(encoding="utf-8"))
    assert payload["status"].startswith("error")
    assert "last_completed_epoch" in payload
