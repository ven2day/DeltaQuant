"""
Small, dependency-free OHLCV geometry helpers shared by scalp-horizon modules
(``entry_quality.py``). Kept separate from ``candidate_policy.py`` (whose existing
``_ema``/``_session_vwap`` compute the same concepts for the swing candidate path)
so this module has no import relationship with -- and therefore cannot accidentally
change the behavior of -- the existing, already-relied-upon swing evaluation path.
"""

from __future__ import annotations

import pandas as pd


def ema(frame: pd.DataFrame, span: int) -> float:
    """Exponential moving average of ``close`` over the whole frame, latest value."""
    return float(frame["close"].ewm(span=span, adjust=False).mean().iloc[-1])


def session_vwap(frame: pd.DataFrame, bars: int = 75) -> float:
    """Volume-weighted average (typical) price over the most recent ``bars`` rows.

    Falls back to the latest close when volume is entirely zero/negative (illiquid
    or synthetic data), never divides by zero.
    """
    recent = frame.iloc[-bars:]
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    volume = recent["volume"].clip(lower=0)
    total = float(volume.sum())
    if total <= 0:
        return float(recent["close"].iloc[-1])
    return float((typical * volume).sum() / total)


def atr_pct(frame: pd.DataFrame, entry: float, lookback: int = 14) -> float:
    """Average True Range over the last ``lookback`` bars, as a percent of ``entry``."""
    if entry <= 0 or len(frame) < lookback + 1:
        return 0.0
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.iloc[-lookback:].mean() / entry * 100)


def wick_ratios(open_: float, high: float, low: float, close: float) -> tuple[float, float]:
    """(upper_wick_ratio, lower_wick_ratio) of one candle, each as a fraction of the
    candle's total high-low range. Both 0.0 for a zero-range (flat) candle."""
    total_range = high - low
    if total_range <= 0:
        return 0.0, 0.0
    body_top = max(open_, close)
    body_bottom = min(open_, close)
    upper_wick = high - body_top
    lower_wick = body_bottom - low
    return upper_wick / total_range, lower_wick / total_range


def zigzag_pivots(closes: pd.Series, threshold: float) -> list[tuple[int, float]]:
    """Return each completed zigzag pivot as ``(index, price)`` -- the actual
    turning-point level, not just the swing magnitude.

    Same threshold-reversal algorithm as
    ``src.market.scalping_screener.compute_zigzag_swings`` (a swing completes when
    price reverses by ``threshold`` or more from the running extreme), but that
    function only returns swing *magnitudes* for oscillation-frequency scoring, which
    can't answer "where is the nearest support/resistance price level" -- this
    sibling function exists specifically to answer that question, kept here rather
    than changing ``compute_zigzag_swings``'s return shape (which the scalping
    screener already depends on).
    """
    values = closes.to_numpy() if isinstance(closes, pd.Series) else list(closes)
    if len(values) < 2:
        return []

    pivots: list[tuple[int, float]] = []
    pivot_idx = 0
    pivot_price = values[0]
    extreme_idx = 0
    extreme_price = values[0]
    direction = 0  # 0 = undetermined, 1 = running high, -1 = running low

    for i, price in enumerate(values[1:], start=1):
        if direction == 0:
            if price - pivot_price >= threshold:
                direction = 1
                extreme_idx, extreme_price = i, price
            elif pivot_price - price >= threshold:
                direction = -1
                extreme_idx, extreme_price = i, price
        elif direction == 1:
            if price > extreme_price:
                extreme_idx, extreme_price = i, price
            elif extreme_price - price >= threshold:
                pivots.append((extreme_idx, extreme_price))
                pivot_idx, pivot_price = extreme_idx, extreme_price
                extreme_idx, extreme_price, direction = i, price, -1
        else:  # direction == -1
            if price < extreme_price:
                extreme_idx, extreme_price = i, price
            elif price - extreme_price >= threshold:
                pivots.append((extreme_idx, extreme_price))
                pivot_idx, pivot_price = extreme_idx, extreme_price
                extreme_idx, extreme_price, direction = i, price, 1

    del pivot_idx, pivot_price  # tracked for algorithm symmetry with compute_zigzag_swings only
    return pivots


def nearest_support_resistance(
    frame: pd.DataFrame, current_price: float, threshold_pct: float, lookback_bars: int
) -> tuple[float | None, float | None]:
    """(nearest_support, nearest_resistance): the closest zigzag pivot price below
    and above ``current_price`` within the last ``lookback_bars`` closes.

    ``threshold_pct`` is a percent of ``current_price`` (not a flat price amount),
    matching ``scalping_screener.py``'s percentage-threshold convention so cheap and
    expensive symbols are held to a comparably-sized bar.
    """
    if current_price <= 0 or len(frame) < 2:
        return None, None
    recent = frame["close"].iloc[-lookback_bars:]
    threshold = current_price * threshold_pct / 100.0
    pivots = zigzag_pivots(recent, threshold)
    if not pivots:
        return None, None

    supports = [price for _, price in pivots if price < current_price]
    resistances = [price for _, price in pivots if price > current_price]
    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None
    return nearest_support, nearest_resistance
