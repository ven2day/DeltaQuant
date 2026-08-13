"""Automatic, budget-gated multi-timeframe signal-discovery scheduling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.core.indicators import Timeframe
from src.markets.nse.market_data.history_manager import HistoryManager

from .operators import build_ohlcv_panel
from .workflow import SignalDiscoveryWorkflow

logger = logging.getLogger(__name__)


class AutomaticDiscoveryScheduler:
    def __init__(self, status_callback: Callable[[str, str], None] | None = None) -> None:
        self.settings = get_settings()
        self.status_callback = status_callback
        self._task: asyncio.Task | None = None
        self._state_path = Path(self.settings.signal_discovery_output_dir) / "auto_run_state.json"

    def maybe_start(self, history_manager: HistoryManager, symbols: list[str]) -> bool:
        if not self.settings.signal_discovery_auto_run or not self._is_due():
            return False
        return self.force_run(history_manager, symbols)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def force_run(self, history_manager: HistoryManager, symbols: list[str]) -> bool:
        """Start a discovery run immediately, bypassing the staleness check.
        Still refuses to double-start a run that's already in progress."""
        if self.is_running:
            return False
        self._task = asyncio.create_task(self._run(history_manager, list(symbols)))
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
        refresh_seconds = self.settings.signal_discovery_refresh_hours * 3600
        return time.time() - last_completed >= refresh_seconds

    async def _run(self, history_manager: HistoryManager, symbols: list[str]) -> None:
        timeframes = self._timeframes()
        self._status(
            f"Automatic signal discovery started for {len(symbols)} symbols on "
            f"{', '.join(timeframe.value for timeframe in timeframes)}",
            "INFO",
        )
        results: dict[str, str] = {}
        for timeframe in timeframes:
            try:
                panel = await self._build_panel(history_manager, symbols, timeframe)
                usable = panel["Close"].shape[1]
                if usable < self.settings.signal_discovery_min_cross_section:
                    raise ValueError(
                        f"only {usable} usable symbols; need "
                        f"{self.settings.signal_discovery_min_cross_section}"
                    )
                request = (
                    f"{self.settings.signal_discovery_request}; horizon and operators must "
                    f"be appropriate for settled {timeframe.value} candles"
                )
                workflow = SignalDiscoveryWorkflow.from_settings(timeframe=timeframe.value)
                result = await workflow.run(request, panel)
                results[timeframe.value] = result.status
                self._status(
                    f"Discovery {timeframe.value}: {result.status}"
                    + (f" ({result.selected_signal.name})" if result.selected_signal else ""),
                    "SUCCESS" if result.status == "accepted" else "WARNING",
                )
            except Exception as exc:
                results[timeframe.value] = f"error: {exc}"
                logger.exception("Automatic discovery failed for %s", timeframe.value)
                self._status(f"Discovery {timeframe.value} failed: {exc}", "WARNING")

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "last_completed_epoch": time.time(),
                    "timeframes": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)
        self._status("Automatic signal discovery cycle completed", "INFO")

    async def _build_panel(
        self,
        history_manager: HistoryManager,
        symbols: list[str],
        timeframe: Timeframe,
    ):
        histories = {}
        for symbol in symbols:
            frame = await asyncio.to_thread(
                history_manager.get_multi_timeframe_history,
                symbol,
                timeframe,
                500,
            )
            if frame is None or len(frame) < self.settings.signal_discovery_min_periods:
                continue
            if self.settings.signals_exclude_forming_bar and len(frame) > 1:
                frame = frame.iloc[:-1]
            histories[symbol] = frame
        return build_ohlcv_panel(histories)

    def _timeframes(self) -> list[Timeframe]:
        values = []
        for token in self.settings.signal_discovery_timeframes.split(","):
            token = token.strip()
            if token:
                values.append(Timeframe(token))
        return values

    def _status(self, message: str, level: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message, level)
        else:
            logger.info("%s: %s", level, message)
