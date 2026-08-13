"""Resumable OANDA history backfill and deterministic coverage reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.markets.forex.runtime.cycle import candles_to_frame
from src.markets.forex.sessions import ForexTradingCalendar

_INTERVALS: dict[str, timedelta] = {
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


@dataclass(frozen=True)
class ForexHistoryCoverage:
    instrument: str
    timeframe: str
    first_candle: str | None
    last_candle: str | None
    row_count: int
    expected_intervals: int
    missing_intervals: int
    coverage_percentage: float
    duplicate_timestamps: int
    incomplete_candles: int
    timezone_errors: int
    alignment_errors: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OandaHistoricalBackfillService:
    """Fetch settled candles in bounded provider windows and upsert idempotently."""

    def __init__(self, provider: Any, candle_repository: Any) -> None:
        self.provider = provider
        self.repository = candle_repository
        self.calendar = ForexTradingCalendar()

    @staticmethod
    def _require_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Historical backfill timestamps must be timezone-aware")
        return value.astimezone(UTC)

    async def backfill(
        self,
        instrument: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> ForexHistoryCoverage:
        interval = _INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported Forex timeframe: {timeframe}")
        cursor = self._require_utc(start)
        stop = min(self._require_utc(end), datetime.now(UTC))
        if cursor >= stop:
            raise ValueError("Historical backfill start must be before end")

        # OANDA caps a response at 5,000 candles. Overlap-free bounded windows plus
        # the schema unique key make retries and resumed runs idempotent.
        chunk_span = interval * 4_900
        while cursor < stop:
            chunk_end = min(cursor + chunk_span, stop)
            candles = await self.provider.get_candles(
                instrument,
                timeframe,
                start=cursor,
                end=chunk_end,
                complete_only=True,
            )
            if candles:
                frame = candles_to_frame(candles)
                self.repository.upsert_frame(
                    instrument,
                    timeframe,
                    frame,
                    source="oanda_backfill",
                    market="FOREX",
                    provider="OANDA",
                    volume_type="OANDA_TICK_COUNT",
                )
            cursor = chunk_end
        frame = self.repository.load_frame(instrument, timeframe, start=start, end=stop)
        return self.coverage(instrument, timeframe, frame)

    def coverage(
        self, instrument: str, timeframe: str, frame: pd.DataFrame
    ) -> ForexHistoryCoverage:
        interval = _INTERVALS[timeframe]
        if frame.empty:
            return ForexHistoryCoverage(
                instrument, timeframe, None, None, 0, 0, 0, 0.0, 0, 0, 0, 0
            )
        index = pd.DatetimeIndex(frame.index)
        duplicates = int(index.duplicated().sum())
        timezone_errors = int(index.tz is None)
        utc_index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        ordered = pd.DatetimeIndex(sorted(set(utc_index)))
        first = ordered[0].to_pydatetime()
        last = ordered[-1].to_pydatetime()
        expected = pd.date_range(first, last, freq=pd.Timedelta(interval), tz="UTC")
        expected_open = pd.DatetimeIndex(
            [timestamp for timestamp in expected if self.calendar.is_open(timestamp.to_pydatetime())]
        )
        missing = expected_open.difference(ordered)
        epoch_seconds = pd.Series(
            [int(timestamp.timestamp()) for timestamp in ordered], dtype="int64"
        )
        interval_seconds = int(interval.total_seconds())
        alignment_errors = int((epoch_seconds % interval_seconds != 0).sum())
        expected_count = len(expected_open)
        coverage = (
            100.0 * max(0, expected_count - len(missing)) / expected_count
            if expected_count
            else 100.0
        )
        incomplete = int((frame["complete"] == False).sum()) if "complete" in frame else 0  # noqa: E712
        return ForexHistoryCoverage(
            instrument=instrument,
            timeframe=timeframe,
            first_candle=first.isoformat(),
            last_candle=last.isoformat(),
            row_count=len(ordered),
            expected_intervals=expected_count,
            missing_intervals=len(missing),
            coverage_percentage=coverage,
            duplicate_timestamps=duplicates,
            incomplete_candles=incomplete,
            timezone_errors=timezone_errors,
            alignment_errors=alignment_errors,
        )


__all__ = ["ForexHistoryCoverage", "OandaHistoricalBackfillService"]
