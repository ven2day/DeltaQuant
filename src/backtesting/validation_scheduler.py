"""Automatic, staleness-gated strategy walk-forward validation scheduling.

Mirrors src/signal_discovery/scheduler.py's AutomaticDiscoveryScheduler shape
exactly (settings-gated maybe_start(), _is_due() against an atomic state
file, blocking work run via asyncio.to_thread) so the two schedulers behave
identically from the live loop's point of view.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)

def _default_swing_validation_intervals() -> tuple[str, ...]:
    """Derive from settings.signal_timeframes (the live SWING scan's actual
    timeframes) rather than hardcoding a guessed list -- a hardcoded
    ("15m", "30m", "1h", "4h", "1d") previously omitted "5m" even though it's
    the default's first entry and, in practice, where most top-ranked
    candidates fire (see live activity: nearly every Rank 1 is on 5m), while
    including "1d" which isn't in signal_timeframes at all. Falls back to the
    old hardcoded list only if settings are unavailable at import time."""
    try:
        configured = get_settings().signal_timeframes
    except Exception:
        return ("15m", "30m", "1h", "4h", "1d")
    values = tuple(token.strip() for token in configured.split(",") if token.strip())
    return values or ("15m", "30m", "1h", "4h", "1d")


# The SWING timeframes this repo actually configures strategies against --
# validated together as one batch per run, same as scripts/validate_strategy.py
# is normally run by hand.
SWING_VALIDATION_INTERVALS: tuple[str, ...] = _default_swing_validation_intervals()


class AutomaticValidationScheduler:
    def __init__(self, status_callback: Callable[[str, str], None] | None = None) -> None:
        self.settings = get_settings()
        self.status_callback = status_callback
        self._task: asyncio.Task[None] | None = None
        self._state_path = (
            Path(self.settings.strategy_eligibility_registry_dir) / "auto_validation_state.json"
        )

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def maybe_start(self) -> bool:
        if not self.settings.strategy_validation_auto_run or not self._is_due():
            return False
        return self.force_run()

    def force_run(self) -> bool:
        """Start a validation run immediately, bypassing the staleness check.
        Still refuses to double-start a run that's already in progress."""
        if self.is_running:
            return False
        self._task = asyncio.create_task(self._run())
        return True

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def _is_due(self) -> bool:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            last_completed = float(payload.get("last_completed_epoch", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            last_completed = 0.0
        refresh_seconds = self.settings.strategy_validation_refresh_hours * 3600
        return bool(time.time() - last_completed >= refresh_seconds)

    async def _run(self) -> None:
        self._status(
            f"Automatic strategy validation started for "
            f"{', '.join(SWING_VALIDATION_INTERVALS)}",
            "INFO",
        )
        results: dict[str, str] = {}
        for interval in SWING_VALIDATION_INTERVALS:
            try:
                await asyncio.to_thread(self._validate_one, interval)
                results[interval] = "ok"
                self._status(f"Validation {interval}: completed", "SUCCESS")
            except Exception as exc:
                results[interval] = f"error: {exc}"
                logger.exception("Automatic validation failed for interval=%s", interval)
                self._status(f"Validation {interval} failed: {exc}", "WARNING")

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"last_completed_epoch": time.time(), "intervals": results},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)
        self._status("Automatic strategy validation cycle completed", "INFO")

    @staticmethod
    def _validate_one(interval: str) -> None:
        # Imported lazily (not at module scope) so a plain `import
        # src.backtesting.validation_scheduler` never pulls in the CLI
        # script's argparse/print-heavy module unless a validation run is
        # actually triggered.
        from scripts.validate_strategy import main as run_validation

        run_validation(interval=interval, trade_horizon="SWING")

    def _status(self, message: str, level: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message, level)
        else:
            logger.info("%s: %s", level, message)
