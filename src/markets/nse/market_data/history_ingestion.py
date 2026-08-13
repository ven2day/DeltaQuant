"""Resumable Dhan-to-database historical candle ingestion."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.config import get_settings
from src.core.candles import CandleStore
from src.core.indicators import Timeframe
from src.markets.nse.market_data.historical_feed import HistoricalFeed
from src.markets.nse.market_data.history_manager import normalize_ohlcv
from src.markets.nse.runtime.state import HistoryReadiness
from src.markets.nse.sessions.market_time import is_market_hours

logger = logging.getLogger(__name__)

_NATIVE_INTERVAL = {
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "60m",
    Timeframe.D1: "1d",
}


def parse_ingestion_timeframes(raw: str) -> list[Timeframe]:
    by_value = {timeframe.value.lower(): timeframe for timeframe in _NATIVE_INTERVAL}
    parsed: list[Timeframe] = []
    for token in raw.split(","):
        timeframe = by_value.get(token.strip().lower())
        if timeframe is not None and timeframe not in parsed:
            parsed.append(timeframe)
    return parsed or [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1]


@dataclass
class IngestionSummary:
    symbols_requested: int
    timeframes_requested: int
    series_completed: int = 0
    series_failed: int = 0
    rows_upserted: int = 0
    failures: list[str] = field(default_factory=list)


class HistoricalIngestionJob:
    """Backfill missing coverage and then refresh only a small overlapping tail."""

    def __init__(
        self,
        store: CandleStore,
        feed: HistoricalFeed,
        *,
        lookback_days: int = 730,
        overlap_days: int = 3,
        progress: Callable[[str, str], None] | None = None,
        readiness_callback: (
            Callable[[str, Timeframe, HistoryReadiness], None] | None
        ) = None,
    ) -> None:
        self.store = store
        self.feed = feed
        self.lookback_days = max(30, lookback_days)
        self.overlap_days = max(1, overlap_days)
        self.progress = progress
        self.readiness_callback = readiness_callback
        self.series_readiness: dict[tuple[str, Timeframe], HistoryReadiness] = {}

    def _set_readiness(
        self, symbol: str, timeframe: Timeframe, readiness: HistoryReadiness
    ) -> None:
        self.series_readiness[(symbol, timeframe)] = readiness
        if self.readiness_callback is not None:
            self.readiness_callback(symbol, timeframe, readiness)

    def _log(self, message: str, level: str = "INFO") -> None:
        getattr(logger, level.lower(), logger.info)(message)
        if self.progress is not None:
            self.progress(message, level)

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").to_pydatetime()

    def _period_for(self, symbol: str, timeframe: Timeframe) -> str:
        coverage = self.store.coverage(symbol, timeframe.value)
        now = datetime.now(UTC)
        desired_start = now - timedelta(days=self.lookback_days)
        first = self._as_utc(coverage.first_candle)
        last = self._as_utc(coverage.last_candle)

        # A fresh/partial database gets the full requested horizon. Once its first
        # candle reaches the horizon, every later run downloads only an overlapping
        # tail so revised broker candles are corrected by the idempotent upsert.
        if first is None or first > desired_start + timedelta(days=7):
            return f"{self.lookback_days}d"
        if last is None:
            return f"{self.lookback_days}d"
        refresh_days = max(self.overlap_days, (now - last).days + self.overlap_days)
        return f"{min(self.lookback_days, refresh_days)}d"

    def sync_series(self, symbol: str, timeframe: Timeframe) -> int:
        interval = _NATIVE_INTERVAL[timeframe]
        period = self._period_for(symbol, timeframe)
        frame = normalize_ohlcv(self.feed.get_historical(symbol, period=period, interval=interval))
        if frame is None or frame.empty:
            raise RuntimeError(f"Dhan returned no {timeframe.value} candles")
        return self.store.upsert_frame(symbol, timeframe.value, frame, source="dhan")

    def run(
        self,
        symbols: list[str],
        timeframes: list[Timeframe],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> IngestionSummary:
        unique_symbols = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol))
        summary = IngestionSummary(len(unique_symbols), len(timeframes))
        total = len(unique_symbols) * len(timeframes)
        completed = 0
        self._log(
            f"Historical ingestion started: {len(unique_symbols)} symbols x "
            f"{len(timeframes)} timeframes, {self.lookback_days}-day target"
        )
        for symbol in unique_symbols:
            for timeframe in timeframes:
                if should_stop is not None and should_stop():
                    self._log("Historical ingestion stopped before completion", "WARNING")
                    return summary
                try:
                    self._set_readiness(symbol, timeframe, HistoryReadiness.BACKFILLING)
                    rows = self.sync_series(symbol, timeframe)
                    summary.series_completed += 1
                    summary.rows_upserted += rows
                    coverage = self.store.coverage(symbol, timeframe.value)
                    readiness = (
                        HistoryReadiness.READY
                        if coverage.rows >= 50
                        else HistoryReadiness.INSUFFICIENT_DATA
                    )
                    self._set_readiness(symbol, timeframe, readiness)
                except Exception as exc:
                    self._set_readiness(symbol, timeframe, HistoryReadiness.FAILED)
                    summary.series_failed += 1
                    failure = f"{symbol}[{timeframe.value}]: {exc}"
                    summary.failures.append(failure)
                    logger.warning("Historical ingestion failed for %s", failure, exc_info=True)
                completed += 1
                if completed % 25 == 0 or completed == total:
                    self._log(
                        f"Historical ingestion progress: {completed}/{total} series, "
                        f"{summary.rows_upserted:,} rows upserted, "
                        f"{summary.series_failed} failed"
                    )
        self._log(
            f"Historical ingestion complete: {summary.series_completed}/{total} series, "
            f"{summary.rows_upserted:,} rows upserted, {summary.series_failed} failed",
            "SUCCESS" if not summary.series_failed else "WARNING",
        )
        return summary


class HistoricalIngestionScheduler:
    """Non-blocking background refresh loop used by the live runtime."""

    def __init__(
        self,
        job: HistoricalIngestionJob,
        *,
        refresh_hours: float,
        allow_during_market: bool = False,
    ) -> None:
        self.job = job
        self.refresh_seconds = max(15 * 60, int(refresh_hours * 3600))
        self.allow_during_market = allow_during_market
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False

    def start(self, symbols: list[str], timeframes: list[Timeframe]) -> bool:
        if self._task is not None and not self._task.done():
            return False
        self._stop_requested = False
        self._task = asyncio.create_task(self._run(list(symbols), list(timeframes)))
        return True

    async def _run(self, symbols: list[str], timeframes: list[Timeframe]) -> None:
        while not self._stop_requested:
            if self.allow_during_market or not is_market_hours():
                await asyncio.to_thread(
                    self.job.run,
                    symbols,
                    timeframes,
                    should_stop=lambda: self._stop_requested,
                )
            else:
                self.job._log(
                    "Historical ingestion deferred until NSE is closed to protect live scans"
                )
            try:
                await asyncio.wait_for(self._wait_until_stopped(), timeout=self.refresh_seconds)
            except TimeoutError:
                continue

    async def _wait_until_stopped(self) -> None:
        while not self._stop_requested:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._stop_requested = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except TimeoutError:
                self._task.cancel()
            except asyncio.CancelledError:
                pass


def build_default_ingestion_job(
    store: CandleStore,
    feed: HistoricalFeed,
    progress: Callable[[str, str], None] | None = None,
) -> HistoricalIngestionJob:
    settings = get_settings()
    return HistoricalIngestionJob(
        store,
        feed,
        lookback_days=settings.market_history_lookback_days,
        overlap_days=settings.market_history_overlap_days,
        progress=progress,
    )
