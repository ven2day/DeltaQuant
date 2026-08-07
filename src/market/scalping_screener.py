"""Identify NSE symbols that oscillate by a meaningful rupee amount multiple times a
day — useful for scalping, where you need repeated intraday swings, not just a big net
move by close.

Neither yfinance nor DhanHQ expose a "volatility screener" endpoint — this computes one
from raw candles at each of the 15m/30m/1h timeframes using a zigzag swing count: walk
each day's closes at that timeframe, track the running high/low in the current
direction, and count a "swing" every time price reverses by at least ``threshold_rs``
from that extreme. Averaging over several days (not just one) filters out symbols that
only spiked once. Showing all timeframes side by side tells you *how fast* a symbol's
moves happen — a stock hitting its swing threshold within a 15-minute candle is a very
different scalp than one that takes an hour to do the same. (4h was dropped: it needed
a 730-day fetch per symbol — by far the heaviest call in the scan — for a coarse signal
scalpers don't act on anyway.)
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.market.historical_feed import HistoricalDataFeed, HistoricalFeed
from src.market.history_manager import fetch_timeframe_history
from src.market.indicators import Timeframe

logger = logging.getLogger(__name__)

# Below this many usable trading days, a symbol's average is too noisy to trust.
MIN_DAYS_FOR_STABLE_READ = 3
# Below this many bars, a session is too thin (holiday half-day, data gap) to score.
MIN_BARS_PER_DAY = 6

# Timeframes shown side by side, finest first (most directly relevant to scalping).
DEFAULT_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
)
# The screener ranks by this timeframe's swing frequency — the finest granularity is the
# most directly actionable signal for a scalper deciding what to watch right now.
RANKING_TIMEFRAME = Timeframe.M15

# An NSE session is ~6h15m: 1h gives only ~6 bars/day, nowhere near enough for a
# *per-day* zigzag to ever complete (a completed swing needs at least 3 points: a
# start, a reversal extreme, and a confirming move back). This is instead scored on the
# continuous close series across the whole lookback window — including across the
# overnight gap, which is a real, tradeable price move at this granularity, not a bug —
# and reported as swings-per-day averaged over the window.
CONTINUOUS_SCORING_TIMEFRAMES = frozenset({Timeframe.H1})

# A swing "counts" once price reverses by this % of the symbol's own price — not a flat
# rupee amount, since that would only ever flag expensive stocks (see module docstring).
DEFAULT_THRESHOLD_PCT = 0.5


@dataclass
class TimeframeSwingStats:
    """One symbol's oscillation profile at a single candle timeframe."""

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
    """One symbol's oscillation profile across every scanned timeframe."""

    symbol: str
    last_price: float
    avg_daily_range: float
    timeframes: dict[str, TimeframeSwingStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "avg_daily_range": self.avg_daily_range,
            "timeframes": {tf: stats.to_dict() for tf, stats in self.timeframes.items()},
        }


def compute_zigzag_swings(closes: pd.Series, threshold: float) -> list[float]:
    """Return the magnitude (always positive, in price units) of each completed swing.

    A swing completes when price reverses by ``threshold`` or more from the running
    extreme in the current direction — standard zigzag-indicator logic. Small back-and-
    forth noise below the threshold is ignored entirely, which is the point: this counts
    *tradeable* moves, not every tick.
    """
    values = closes.to_numpy() if isinstance(closes, pd.Series) else list(closes)
    if len(values) < 2:
        return []

    swings: list[float] = []
    pivot = values[0]
    extreme = values[0]
    direction = 0  # 0 = undetermined yet, 1 = extreme is a running high, -1 = running low

    for price in values[1:]:
        if direction == 0:
            if price - pivot >= threshold:
                direction = 1
                extreme = price
            elif pivot - price >= threshold:
                direction = -1
                extreme = price
        elif direction == 1:
            if price > extreme:
                extreme = price
            elif extreme - price >= threshold:
                swings.append(extreme - pivot)
                pivot, extreme, direction = extreme, price, -1
        else:  # direction == -1
            if price < extreme:
                extreme = price
            elif price - extreme >= threshold:
                swings.append(pivot - extreme)
                pivot, extreme, direction = extreme, price, 1

    return swings


def _last_n_days(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Restrict to the most recent ``lookback_days`` calendar days present in the data.

    fetch_timeframe_history() always returns whatever range Yahoo allows for that
    interval (~60 days for 15m/30m, ~730 days for 1h/4h) — this trims that down to a
    recent, comparable window across all four timeframes.
    """
    unique_days = sorted({d for d in df.index.date})
    if len(unique_days) <= lookback_days:
        return df
    cutoff = unique_days[-lookback_days]
    return df[df.index.date >= cutoff]


def _score_timeframe(
    df: pd.DataFrame, threshold_rs: float, lookback_days: int, timeframe: Timeframe
) -> TimeframeSwingStats | None:
    """Score one already-fetched OHLCV frame; None if there isn't enough usable data."""
    df = _last_n_days(df, lookback_days)
    if df.empty:
        return None

    if timeframe in CONTINUOUS_SCORING_TIMEFRAMES:
        days_covered = len({d for d in df.index.date})
        if days_covered < MIN_DAYS_FOR_STABLE_READ:
            return None
        swings = compute_zigzag_swings(df["close"], threshold_rs)
        return TimeframeSwingStats(
            avg_swings_per_day=round(len(swings) / days_covered, 2),
            avg_swing_size=round(sum(swings) / len(swings), 2) if swings else 0.0,
            days_analyzed=days_covered,
        )

    all_swings: list[float] = []
    daily_swing_counts: list[int] = []
    for _, day_df in df.groupby(df.index.date):
        if len(day_df) < MIN_BARS_PER_DAY:
            continue
        swings = compute_zigzag_swings(day_df["close"], threshold_rs)
        daily_swing_counts.append(len(swings))
        all_swings.extend(swings)

    if len(daily_swing_counts) < MIN_DAYS_FOR_STABLE_READ:
        return None

    return TimeframeSwingStats(
        avg_swings_per_day=round(sum(daily_swing_counts) / len(daily_swing_counts), 2),
        avg_swing_size=round(sum(all_swings) / len(all_swings), 2) if all_swings else 0.0,
        days_analyzed=len(daily_swing_counts),
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
    """Rank symbols by average zigzag swing count/day at ``RANKING_TIMEFRAME``.

    The swing threshold is **percentage-of-price**, not a flat rupee amount: a fixed
    ₹15 bar is a 5% move on a ₹300 stock but only 0.1% on a ₹14,000 one, so a flat
    threshold structurally favors expensive stocks and leaves cheaper price tiers
    empty. Each symbol's own threshold (in rupees) is derived from its current price
    at ``RANKING_TIMEFRAME`` — fetched once per symbol and reused for every timeframe,
    since it only needs to be roughly right, not bar-exact.

    Each symbol is scored independently at every timeframe in ``timeframes``, each
    restricted to the same last ``lookback_days`` calendar days, so the result shows not
    just *whether* a symbol moves enough, but *how fast* — a symbol that only shows up
    at 4h is a slow grind; one that shows up at 15m moves fast enough to actually scalp.
    A symbol needs a qualifying read (>= min_swings_per_day, >= MIN_DAYS_FOR_STABLE_READ
    usable sessions) at RANKING_TIMEFRAME to be included at all.
    """
    candidates: list[ScalpingCandidate] = []

    for symbol in symbols:
        if RANKING_TIMEFRAME not in timeframes:
            continue

        try:
            ranking_df = fetch_timeframe_history(feed, symbol, RANKING_TIMEFRAME)
        except Exception:
            logger.exception("Failed to fetch %s history for %s", RANKING_TIMEFRAME.value, symbol)
            continue
        if ranking_df is None or ranking_df.empty or "close" not in ranking_df.columns:
            continue

        recent_ranking = _last_n_days(ranking_df, lookback_days)
        last_price = float(recent_ranking["close"].iloc[-1])
        if last_price <= 0:
            continue
        threshold_rs = last_price * threshold_pct / 100.0
        daily_ranges = [
            float(day_df["high"].max() - day_df["low"].min())
            for _, day_df in recent_ranking.groupby(recent_ranking.index.date)
            if len(day_df) >= MIN_BARS_PER_DAY
        ]

        per_timeframe: dict[str, TimeframeSwingStats] = {}
        ranking_stats: TimeframeSwingStats | None = None

        for timeframe in timeframes:
            if timeframe == RANKING_TIMEFRAME:
                df = ranking_df
            else:
                try:
                    df = fetch_timeframe_history(feed, symbol, timeframe)
                except Exception:
                    logger.exception("Failed to fetch %s history for %s", timeframe.value, symbol)
                    continue
            if df is None or df.empty or "close" not in df.columns:
                continue

            stats = _score_timeframe(df, threshold_rs, lookback_days, timeframe)
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
                avg_daily_range=round(sum(daily_ranges) / len(daily_ranges), 2)
                if daily_ranges
                else 0.0,
                timeframes=per_timeframe,
            )
        )

    candidates.sort(
        key=lambda c: c.timeframes[RANKING_TIMEFRAME.value].avg_swings_per_day, reverse=True
    )
    return candidates[:top_n]


class ScalpingScreenerTracker:
    """Background task: rescans the universe on its own timer and caches the result.

    A full rescan fetches multiple days of candles at 4 timeframes per symbol — much
    heavier and slower than the sector-movers scan (one snapshot quote per symbol) — so
    this deliberately refreshes on a much longer interval, since the underlying
    multi-day pattern doesn't change meaningfully within a session anyway.
    """

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
        # Optional sink for the dashboard/web-UI activity log (e.g. TradingStats.log_activity).
        # Kept as a plain callback rather than importing the dashboard here, so this module
        # stays usable (and testable) without it.
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
                self.candidates = [c.to_dict() for c in candidates]
                self.scan_status = "ready"
                logger.info("Scalping screener refreshed: %d candidates", len(self.candidates))
                self._on_status(
                    f"Scalping screener refreshed: {len(self.candidates)} candidates", "INFO"
                )
            except Exception as e:
                self.scan_status = "error"
                logger.exception("Scalping screener refresh failed")
                self._on_status(f"Scalping screener refresh failed: {e}", "WARNING")
            await asyncio.sleep(self.refresh_seconds)
