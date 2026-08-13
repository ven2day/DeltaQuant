import json
import time
from types import SimpleNamespace

from src.core.indicators import Timeframe
from src.signal_discovery.scheduler import AutomaticDiscoveryScheduler


def _settings(tmp_path):
    return SimpleNamespace(
        signal_discovery_output_dir=str(tmp_path),
        signal_discovery_auto_run=True,
        signal_discovery_refresh_hours=24,
        signal_discovery_timeframes="15m,30m,1h,4h",
    )


def test_scheduler_covers_all_configured_signal_timeframes(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.signal_discovery.scheduler.get_settings", lambda: settings)

    scheduler = AutomaticDiscoveryScheduler()

    assert scheduler._timeframes() == [
        Timeframe.M15,
        Timeframe.M30,
        Timeframe.H1,
        Timeframe.H4,
    ]


def test_scheduler_does_not_repeat_before_refresh_interval(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.signal_discovery.scheduler.get_settings", lambda: settings)
    scheduler = AutomaticDiscoveryScheduler()

    assert scheduler._is_due()

    scheduler._state_path.write_text(
        json.dumps({"last_completed_epoch": time.time()}), encoding="utf-8"
    )

    assert not scheduler._is_due()
