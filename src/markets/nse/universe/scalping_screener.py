"""NSE intraday swing-frequency screener.

This is market/universe diagnostics, not the strategy engine and not an execution
gate.  It is owned by NSE because its session grouping and volume/data provider
semantics are exchange-specific.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from src.core.indicators import Timeframe
from src.markets.nse.market_data.historical_feed import HistoricalDataFeed, HistoricalFeed
from src.markets.nse.market_data.history_manager import fetch_timeframe_history

logger = logging.getLogger(__name__)

MIN_DAYS_FOR_STABLE_READ = 3
MIN_BARS_PER_DAY = 6
DEFAULT_TIMEFRAMES: tuple[Timeframe, ...] = (Timeframe.M15, Timeframe.M30, Timeframe.H1)
RANKING_TIMEFRAME = Timeframe.M15
CONTINUOUS_SCORING_TIMEFRAMES = frozenset({Timeframe.H1})
DEFAULT_THRESHOLD_PCT = 0.5


@dataclass
class TimeframeSwingStats:
    avg_swings_per_day: float
    avg_swing_size: float
    days_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_swings_per_day": self.avg_swings_per_day,
            "avg_swing_size": self.avg_swing_size,
            "days_analyzed": self.days_analyzed,
        }


@dataclass
class ScalpingCandidate:
    symbol: str
    last_price: float
    avg_daily_range: float
    timeframes: dict[str, TimeframeSwingStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "avg_daily_range": self.avg_daily_range,
            "timeframes": {name: value.to_dict() for name, value in self.timeframes.items()},
        }


def compute_zigzag_swings(closes: pd.Series, threshold: float) -> list[float]:
    values = closes.to_numpy() if isinstance(closes, pd.Series) else list(closes)
    if len(values) < 2:
        return []
    swings: list[float] = []
    pivot = values[0]
    extreme = values[0]
    direction = 0
    for price in values[1:]:
        if direction == 0:
            if price - pivot >= threshold:
                direction, extreme = 1, price
            elif pivot - price >= threshold:
                direction, extreme = -1, price
        elif direction == 1:
            if price > extreme:
                extreme = price
            elif extreme - price >= threshold:
                swings.append(extreme - pivot)
                pivot, extreme, direction = extreme, price, -1
        elif price < extreme:
            extreme = price
        elif price - extreme >= threshold:
            swings.append(pivot - extreme)
            pivot, extreme, direction = extreme, price, 1
    return swings


def _last_n_days(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    dates = pd.DatetimeIndex(frame.index).date
    unique_days = sorted(set(dates))
    if len(unique_days) <= lookback_days:
        return frame
    return cast(pd.DataFrame, frame.loc[dates >= unique_days[-lookback_days]])


def _score_timeframe(
    frame: pd.DataFrame,
    threshold_rs: float,
    lookback_days: int,
    timeframe: Timeframe,
) -> TimeframeSwingStats | None:
    frame = _last_n_days(frame, lookback_days)
    if frame.empty:
        return None
    if timeframe in CONTINUOUS_SCORING_TIMEFRAMES:
        days = len(set(pd.DatetimeIndex(frame.index).date))
        if days < MIN_DAYS_FOR_STABLE_READ:
            return None
        swings = compute_zigzag_swings(frame["close"], threshold_rs)
        return TimeframeSwingStats(
            avg_swings_per_day=round(len(swings) / days, 2),
            avg_swing_size=round(sum(swings) / len(swings), 2) if swings else 0.0,
            days_analyzed=days,
        )
    all_swings: list[float] = []
    daily_counts: list[int] = []
    for _, daily in frame.groupby(pd.DatetimeIndex(frame.index).date):
        if len(daily) < MIN_BARS_PER_DAY:
            continue
        swings = compute_zigzag_swings(daily["close"], threshold_rs)
        daily_counts.append(len(swings))
        all_swings.extend(swings)
    if len(daily_counts) < MIN_DAYS_FOR_STABLE_READ:
        return None
    return TimeframeSwingStats(
        avg_swings_per_day=round(sum(daily_counts) / len(daily_counts), 2),
        avg_swing_size=round(sum(all_swings) / len(all_swings), 2) if all_swings else 0.0,
        days_analyzed=len(daily_counts),
    )


def compute_scalping_candidates(
    feed: HistoricalFeed,
    symbols: list[str],
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    timeframes: tuple[Timeframe, ...] = DEFAULT_TIMEFRAMES,
    lookback_days: int = 7,
    top_n: int = 20,
    min_swings_per_day: float = 2.0,
) -> list[ScalpingCandidate]:
    candidates: list[ScalpingCandidate] = []
    for symbol in symbols:
        if RANKING_TIMEFRAME not in timeframes:
            continue
        try:
            ranking_frame = fetch_timeframe_history(feed, symbol, RANKING_TIMEFRAME)
        except Exception:
            logger.exception("Failed to fetch %s history for %s", RANKING_TIMEFRAME.value, symbol)
            continue
        if ranking_frame is None or ranking_frame.empty or "close" not in ranking_frame.columns:
            continue
        recent = _last_n_days(ranking_frame, lookback_days)
        last_price = float(recent["close"].iloc[-1])
        if last_price <= 0:
            continue
        threshold_rs = last_price * threshold_pct / 100.0
        daily_ranges = [
            float(day["high"].max() - day["low"].min())
            for _, day in recent.groupby(pd.DatetimeIndex(recent.index).date)
            if len(day) >= MIN_BARS_PER_DAY
        ]
        per_timeframe: dict[str, TimeframeSwingStats] = {}
        ranking_stats: TimeframeSwingStats | None = None
        for timeframe in timeframes:
            current_frame: pd.DataFrame | None
            if timeframe == RANKING_TIMEFRAME:
                current_frame = ranking_frame
            else:
                try:
                    current_frame = fetch_timeframe_history(feed, symbol, timeframe)
                except Exception:
                    logger.exception("Failed to fetch %s history for %s", timeframe.value, symbol)
                    continue
            if (
                current_frame is None
                or current_frame.empty
                or "close" not in current_frame.columns
            ):
                continue
            stats = _score_timeframe(current_frame, threshold_rs, lookback_days, timeframe)
            if stats is not None:
                per_timeframe[timeframe.value] = stats
                if timeframe == RANKING_TIMEFRAME:
                    ranking_stats = stats
        if ranking_stats is None or ranking_stats.avg_swings_per_day < min_swings_per_day:
            continue
        candidates.append(
            ScalpingCandidate(
                symbol=symbol,
                last_price=last_price,
                avg_daily_range=(
                    round(sum(daily_ranges) / len(daily_ranges), 2) if daily_ranges else 0.0
                ),
                timeframes=per_timeframe,
            )
        )
    candidates.sort(
        key=lambda item: item.timeframes[RANKING_TIMEFRAME.value].avg_swings_per_day,
        reverse=True,
    )
    return candidates[:top_n]


class ScalpingScreenerTracker:
    def __init__(
        self,
        symbols: list[str],
        refresh_seconds: int,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        lookback_days: int = 7,
        top_n: int = 20,
        on_status: Callable[[str, str], None] | None = None,
        feed: HistoricalFeed | None = None,
        data_source: str = "dhan",
    ) -> None:
        self._feed = feed or HistoricalDataFeed(symbols=symbols)
        self._symbols = symbols
        self.refresh_seconds = refresh_seconds
        self.threshold_pct = threshold_pct
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.candidates: list[dict[str, Any]] = []
        self.scan_status = "pending"
        self.data_source = data_source
        self._on_status = on_status or (lambda message, level: None)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                candidates = await loop.run_in_executor(
                    None,
                    lambda: compute_scalping_candidates(
                        self._feed,
                        self._symbols,
                        threshold_pct=self.threshold_pct,
                        timeframes=DEFAULT_TIMEFRAMES,
                        lookback_days=self.lookback_days,
                        top_n=self.top_n,
                    ),
                )
                self.candidates = [candidate.to_dict() for candidate in candidates]
                self.scan_status = "ready"
                logger.info("Scalping screener refreshed: %d candidates", len(self.candidates))
                self._on_status(
                    f"Scalping screener refreshed: {len(self.candidates)} candidates", "INFO"
                )
            except Exception as exc:
                self.scan_status = "error"
                logger.exception("Scalping screener refresh failed")
                self._on_status(f"Scalping screener refresh failed: {exc}", "WARNING")
            await asyncio.sleep(self.refresh_seconds)


__all__ = [
    "CONTINUOUS_SCORING_TIMEFRAMES",
    "DEFAULT_THRESHOLD_PCT",
    "DEFAULT_TIMEFRAMES",
    "MIN_BARS_PER_DAY",
    "MIN_DAYS_FOR_STABLE_READ",
    "RANKING_TIMEFRAME",
    "ScalpingCandidate",
    "ScalpingScreenerTracker",
    "TimeframeSwingStats",
    "compute_scalping_candidates",
    "compute_zigzag_swings",
]
