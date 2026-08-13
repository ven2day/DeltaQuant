"""Automatic, staleness-gated scalp ML artifact training scheduling.

Same shape as src/signal_discovery/scheduler.py's AutomaticDiscoveryScheduler
and src/backtesting/validation_scheduler.py's AutomaticValidationScheduler:
settings-gated maybe_start(), _is_due() against an atomic state file, blocking
work run via asyncio.to_thread.

Deliberately does NOT touch NSEModelRegistry/ModelRegistry (the DB-backed
"active approved model" registry served at GET /api/models/status) -- per
this codebase's existing governance pattern (walk-forward validation grants
paper approval only; promoting a model to the one actually used for live
inference stays an explicit, operator-controlled action, same as
LIVE_APPROVED for strategies). This scheduler only produces trained
candidate artifacts via PredictionArtifactRegistry; it never promotes them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


class AutomaticScalpTrainingScheduler:
    def __init__(self, status_callback: Callable[[str, str], None] | None = None) -> None:
        self.settings = get_settings()
        self.status_callback = status_callback
        self._task: asyncio.Task[None] | None = None
        self._state_path = Path("data/prediction_models") / "auto_scalp_training_state.json"

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def maybe_start(self) -> bool:
        if not self.settings.scalp_training_auto_run or not self._is_due():
            return False
        return self.force_run()

    def force_run(self) -> bool:
        """Start a training run immediately, bypassing the staleness check.
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
        refresh_seconds = self.settings.scalp_training_refresh_hours * 3600
        return bool(time.time() - last_completed >= refresh_seconds)

    async def _run(self) -> None:
        self._status("Automatic scalp ML training started", "INFO")
        status = "ok"
        try:
            report = await asyncio.to_thread(self._train)
            self._status(
                f"Scalp training completed: {json.dumps(report.get('timeframes', {}))}",
                "SUCCESS",
            )
        except Exception as exc:
            status = f"error: {exc}"
            logger.exception("Automatic scalp training failed")
            self._status(f"Scalp training failed: {exc}", "WARNING")

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"last_completed_epoch": time.time(), "status": status},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)
        self._status("Automatic scalp ML training cycle completed", "INFO")

    @staticmethod
    def _train() -> dict[str, Any]:
        # Imported lazily so a plain `import
        # src.agents.scalp_training_scheduler` never pulls in the training
        # script's heavier dependencies unless a run is actually triggered.
        from scripts.train_scalp_prediction_artifacts import run_training

        output_path = Path("tmp/scalp_prediction_training.json")
        exit_code = run_training(output=output_path)
        report: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
        if exit_code != 0:
            rejected = {
                timeframe: entry.get("reason")
                for timeframe, entry in report.get("timeframes", {}).items()
                if entry.get("status") == "REJECTED"
            }
            raise RuntimeError(f"training rejected for: {rejected}")
        return report

    def _status(self, message: str, level: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message, level)
        else:
            logger.info("%s: %s", level, message)
