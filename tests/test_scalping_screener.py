"""Tests for the zigzag-based scalping screener (src/markets/nse/universe/scalping_screener.py)."""

import asyncio
import contextlib
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.indicators import Timeframe
from src.markets.nse.universe.scalping_screener import (
    ScalpingScreenerTracker,
    compute_scalping_candidates,
    compute_zigzag_swings,
)

# --- compute_zigzag_swings ---


def test_empty_or_single_price_has_no_swings():
    assert compute_zigzag_swings(pd.Series([], dtype=float), 15.0) == []
    assert compute_zigzag_swings(pd.Series([100.0]), 15.0) == []


def test_noise_below_threshold_produces_no_swings():
    # Wobbles by at most 5 either way — never reaches the 15 threshold.
    prices = pd.Series([100.0, 103.0, 98.0, 101.0, 97.0, 100.0])
    assert compute_zigzag_swings(prices, 15.0) == []


def test_monotonic_move_with_no_reversal_produces_no_completed_swing():
    # Goes up 30 and never reverses — one leg forming, but never "completes".
    prices = pd.Series([100.0, 110.0, 120.0, 130.0])
    assert compute_zigzag_swings(prices, 15.0) == []


def test_single_up_then_down_swing_is_counted():
    # 100 -> 120 (+20, crosses threshold, becomes the up-leg) -> 100 (-20, completes it).
    prices = pd.Series([100.0, 110.0, 120.0, 110.0, 100.0])
    swings = compute_zigzag_swings(prices, 15.0)
    assert swings == [20.0]


def test_multiple_swings_in_one_day_are_all_counted():
    # 2000 -> 2020 (up 20) -> 2000 (down 20) -> 2018 (up 18) -> 2000 (down 18)
    prices = pd.Series([2000.0, 2010.0, 2020.0, 2010.0, 2000.0, 2010.0, 2018.0, 2005.0, 2000.0])
    swings = compute_zigzag_swings(prices, 15.0)
    assert swings == [20.0, 20.0, 18.0]


def test_threshold_is_a_hard_cutoff():
    # A 14-point move must not register when the threshold is 15.
    prices = pd.Series([100.0, 114.0, 100.0])
    assert compute_zigzag_swings(prices, 15.0) == []
    # A move of exactly 15 does complete the up-leg; there's no 4th point here to also
    # complete the down-leg back, so exactly one swing is recorded.
    prices_15 = pd.Series([100.0, 115.0, 100.0])
    assert compute_zigzag_swings(prices_15, 15.0) == [15.0]


# --- compute_scalping_candidates (multi-timeframe) ---


def _intraday_index(day: str, n_bars: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 09:15", periods=n_bars, freq=freq)


class _FakeFeed:
    """Stands in for HistoricalDataFeed.get_historical(); scalping_screener calls it through
    fetch_timeframe_history(), which maps timeframe -> (interval, period) automatically."""

    def __init__(self, frames_by_symbol: dict[str, dict[str, pd.DataFrame]]):
        # frames_by_symbol[symbol][yf_interval] -> full multi-day DataFrame
        self._frames = frames_by_symbol

    def get_historical(self, symbol, period=None, interval=None):
        return self._frames.get(symbol, {}).get(interval)


def _volatile_frame(days: list[str], n_bars: int, freq: str) -> pd.DataFrame:
    """2000<->2020 repeating every day: three completed 20-point zigzag swings/day."""
    closes = [2000.0, 2010.0, 2020.0, 2010.0, 2000.0, 2010.0, 2020.0, 2010.0, 2000.0]
    closes = (closes * ((n_bars // len(closes)) + 1))[:n_bars]
    frames = []
    for day in days:
        index = _intraday_index(day, n_bars, freq)
        frames.append(
            pd.DataFrame(
                {
                    "Open": closes,
                    "High": [c + 2 for c in closes],
                    "Low": [c - 2 for c in closes],
                    "Close": closes,
                    "Volume": [1000] * n_bars,
                },
                index=index,
            )
        )
    return pd.concat(frames)


def _flat_frame(days: list[str], n_bars: int, freq: str) -> pd.DataFrame:
    """Barely moves — never reaches a 15-point swing."""
    base = [2000.0, 2002.0, 1999.0, 2001.0, 2000.0, 2003.0, 1998.0, 2000.0, 2001.0]
    closes = (base * ((n_bars // len(base)) + 1))[:n_bars]
    frames = []
    for day in days:
        index = _intraday_index(day, n_bars, freq)
        frames.append(
            pd.DataFrame(
                {
                    "Open": closes,
                    "High": [c + 1 for c in closes],
                    "Low": [c - 1 for c in closes],
                    "Close": closes,
                    "Volume": [1000] * n_bars,
                },
                index=index,
            )
        )
    return pd.concat(frames)


_DAYS = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]


def test_volatile_symbol_is_scored_at_every_requested_timeframe():
    # fetch_timeframe_history() maps M15 -> get_historical(interval="15m") and
    # M30 -> get_historical(interval="30m") directly; H1 routes through "60m" and isn't
    # exercised here — the point of this test is that each requested timeframe gets its
    # own independent score in the result, which two is enough to prove.
    feed = _FakeFeed(
        {
            "TCS": {
                "15m": _volatile_frame(_DAYS, 9, "15min"),
                "30m": _volatile_frame(_DAYS, 9, "30min"),
            }
        }
    )

    candidates = compute_scalping_candidates(
        feed, ["TCS"], threshold_pct=0.75, timeframes=(Timeframe.M15, Timeframe.M30)
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.symbol == "TCS"
    assert set(candidate.timeframes.keys()) == {"15m", "30m"}
    assert candidate.timeframes["15m"].avg_swings_per_day == 3.0
    assert candidate.timeframes["15m"].avg_swing_size == pytest.approx(20.0, abs=0.01)
    assert candidate.timeframes["30m"].avg_swings_per_day == 3.0


def test_h1_scores_on_the_continuous_series_not_per_day():
    # Only 2 bars/day — an NSE session (~6h15m) produces ~6-7 one-hour candles, right at
    # the edge of MIN_BARS_PER_DAY (6); this fixture deliberately goes below it so
    # per-day grouping would exclude every single day (this is exactly the bug found
    # live). Continuous scoring must still produce a result by walking the whole series
    # at once.
    from src.markets.nse.universe.scalping_screener import _score_timeframe

    # Chain across days: 2000 -> 2020 -> 2000 -> 2020 -> ... completes a swing every 2 days.
    closes = [2000.0, 2020.0] * len(_DAYS)
    frames = []
    for day in _DAYS:
        index = _intraday_index(day, 2, "240min")
        frames.append(
            pd.DataFrame(
                {
                    "open": [0, 0],
                    "high": [0, 0],
                    "low": [0, 0],
                    "close": [0.0, 0.0],
                    "volume": [1, 1],
                },
                index=index,
            )
        )
    df = pd.concat(frames)
    df["close"] = closes

    stats = _score_timeframe(df, threshold_rs=15.0, lookback_days=7, timeframe=Timeframe.H1)

    assert stats is not None  # would be None/excluded under the old per-day-only logic
    assert stats.days_analyzed == len(_DAYS)
    assert stats.avg_swings_per_day > 0


def test_flat_symbol_is_excluded():
    feed = _FakeFeed({"BORING": {"15m": _flat_frame(_DAYS, 9, "15min")}})

    candidates = compute_scalping_candidates(
        feed, ["BORING"], threshold_pct=0.75, timeframes=(Timeframe.M15,)
    )

    assert candidates == []


def test_threshold_scales_with_symbol_price():
    # A cheap stock oscillating by only Rs.2 (0.67% of its Rs.300 price) would never
    # register a single swing under a flat Rs.15 threshold — but a %-based threshold
    # judges it by its own price, not an absolute rupee bar set by expensive stocks.
    base = 300.0
    swing = 2.0
    closes = ([base, base + swing] * 5)[:9]
    frames = []
    for day in _DAYS:
        index = _intraday_index(day, 9, "15min")
        frames.append(
            pd.DataFrame(
                {
                    "Open": closes,
                    "High": [c + 0.5 for c in closes],
                    "Low": [c - 0.5 for c in closes],
                    "Close": closes,
                    "Volume": [1000] * 9,
                },
                index=index,
            )
        )
    df = pd.concat(frames)
    feed = _FakeFeed({"CHEAP": {"15m": df}})

    # 0.5% of Rs.300 = Rs.1.50 — the Rs.2 swings clear it.
    qualifies = compute_scalping_candidates(
        feed, ["CHEAP"], threshold_pct=0.5, timeframes=(Timeframe.M15,)
    )
    assert len(qualifies) == 1
    assert qualifies[0].symbol == "CHEAP"

    # 5% of Rs.300 = Rs.15 — the same Rs.2 swings never clear that bar, mirroring what
    # the old flat-Rs.15 threshold would have done to every sub-Rs.300 stock.
    excluded = compute_scalping_candidates(
        feed, ["CHEAP"], threshold_pct=5.0, timeframes=(Timeframe.M15,)
    )
    assert excluded == []


def test_too_few_trading_days_is_excluded():
    days = ["2026-08-01", "2026-08-02"]  # below MIN_DAYS_FOR_STABLE_READ (3)
    feed = _FakeFeed({"TCS": {"15m": _volatile_frame(days, 9, "15min")}})

    candidates = compute_scalping_candidates(
        feed, ["TCS"], threshold_pct=0.75, timeframes=(Timeframe.M15,)
    )

    assert candidates == []


def test_missing_symbol_data_is_skipped_not_crashed():
    feed = _FakeFeed({})  # no data for anything

    candidates = compute_scalping_candidates(
        feed, ["UNKNOWN"], threshold_pct=0.75, timeframes=(Timeframe.M15,)
    )

    assert candidates == []


def test_results_are_ranked_by_ranking_timeframe_and_capped_at_top_n():
    feed = _FakeFeed(
        {
            "A": {"15m": _volatile_frame(_DAYS, 9, "15min")},
            "B": {"15m": _volatile_frame(_DAYS, 9, "15min")},
        }
    )

    candidates = compute_scalping_candidates(
        feed, ["A", "B"], threshold_pct=0.75, timeframes=(Timeframe.M15,), top_n=1
    )

    assert len(candidates) == 1


def test_lookback_days_restricts_to_recent_days_only():
    # 6 days of volatile data; lookback_days=3 should only see the most recent 3.
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]
    feed = _FakeFeed({"TCS": {"15m": _volatile_frame(days, 9, "15min")}})

    candidates = compute_scalping_candidates(
        feed, ["TCS"], threshold_pct=0.75, timeframes=(Timeframe.M15,), lookback_days=3
    )

    assert len(candidates) == 1
    assert candidates[0].timeframes["15m"].days_analyzed == 3


# --- ScalpingScreenerTracker.run() status reporting ---


async def _run_one_cycle(tracker: ScalpingScreenerTracker) -> None:
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_tracker_reports_success_via_on_status():
    calls: list[tuple[str, str]] = []
    tracker = ScalpingScreenerTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        on_status=lambda msg, level: calls.append((msg, level)),
    )
    with patch("src.markets.nse.universe.scalping_screener.compute_scalping_candidates", return_value=[]):
        await _run_one_cycle(tracker)

    assert calls
    assert calls[0][1] == "INFO"


async def test_tracker_reports_failure_via_on_status():
    calls: list[tuple[str, str]] = []
    tracker = ScalpingScreenerTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        on_status=lambda msg, level: calls.append((msg, level)),
    )
    with patch(
        "src.markets.nse.universe.scalping_screener.compute_scalping_candidates",
        side_effect=RuntimeError("boom"),
    ):
        await _run_one_cycle(tracker)

    assert calls
    assert calls[0][1] == "WARNING"
    assert "boom" in calls[0][0]


async def test_tracker_accepts_simulated_historical_feed():
    feed = _FakeFeed({})
    tracker = ScalpingScreenerTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        feed=feed,
        data_source="simulated",
    )
    with patch("src.markets.nse.universe.scalping_screener.compute_scalping_candidates", return_value=[]) as scan:
        await _run_one_cycle(tracker)

    assert scan.call_args.args[0] is feed
    assert tracker.scan_status == "ready"
    assert tracker.data_source == "simulated"
