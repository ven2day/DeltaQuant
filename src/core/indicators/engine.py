"""
Technical Indicators Module

Computes various technical indicators for trading signal generation.
Uses the 'ta' library for standard indicators with custom extensions.
"""

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Increment whenever the shared indicator definitions or their default parameters
# change.  It is part of the cache identity so an old feature vector can never be
# reused under new semantics.
FEATURE_SET_VERSION = "technical_features_v4_pivot_sr_zones"


def _safe_float(value: Any) -> float | None:
    """
    Coerce a value to float, returning None for NaN/inf.

    The `ta` library emits NaN during the warm-up window (and on gaps). Passing NaN
    downstream silently distorts signal logic (`nan < 50` is always False) and produces
    invalid JSON ("NaN") in the agent context. Converting to None makes "not yet
    computed" explicit; the signal strategies already guard `if ind.x is None`.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


class Timeframe(Enum):
    """Candle timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculation."""

    # Moving Averages
    sma_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = field(default_factory=lambda: [9, 20, 21, 40, 50, 55, 200])

    # Momentum
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # Trend
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14

    # Volatility
    atr_period: int = 14
    bb_period: int = 20
    bb_std: int = 2

    # Shared production-strategy features.  These parameters are part of the
    # IndicatorCache key; changing one therefore cannot reuse an older snapshot.
    donchian_period: int = 20
    volume_period: int = 20
    momentum_periods: tuple[int, ...] = (5, 10, 20)
    supertrend_period: int = 14
    supertrend_multiplier: float = 3.0
    opening_range_minutes: int = 30

    # Additional trend/momentum indicators (EMA-CCI, EMA-Heiken-Ashi-RSI, EMA-PSAR
    # strategies)
    cci_period: int = 14
    psar_step: float = 0.02
    psar_max_step: float = 0.2

    # Volume Profile (Point of Control / Value Area) for the volume_profile_breakout
    # strategy. A price histogram over the lookback window, weighted by each bar's
    # volume -- an approximation (bar-level, not tick-level) since Dhan supplies OHLCV
    # bars, not individual trades.
    volume_profile_period: int = 60
    volume_profile_bins: int = 24
    volume_profile_value_area_pct: float = 0.70

    # High-volume pivot support/resistance zones (pivot_volume_sr_zones strategy).
    # Matches the source Pine Script's own defaults; note it declares an "ATR Length"
    # input of 100 but its actual zone-height calculation calls ta.atr(200) directly,
    # ignoring that input -- pivot_sr_atr_length reproduces the real (200) behavior.
    pivot_sr_length: int = 40
    pivot_sr_volume_avg_length: int = 20
    pivot_sr_volume_multiplier: float = 1.2
    pivot_sr_atr_length: int = 200

@dataclass
class IndicatorResult:
    """Result of indicator calculations for a symbol."""

    symbol: str
    timeframe: Timeframe

    # Price data
    open: float
    high: float
    low: float
    close: float
    volume: int
    previous_open: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    previous_close: float | None = None

    # Moving Averages
    sma: dict[int, float] = field(default_factory=dict)  # period -> value
    ema: dict[int, float] = field(default_factory=dict)
    ema_previous: dict[int, float] = field(default_factory=dict)
    ema_slope_atr: dict[int, float] = field(default_factory=dict)

    # Momentum
    rsi: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None

    # Trend
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    macd_previous: float | None = None
    macd_signal_previous: float | None = None
    macd_histogram_previous: float | None = None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None

    # Volatility
    atr: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_percent: float | None = None
    bb_bandwidth: float | None = None
    bb_previous_upper: float | None = None
    bb_previous_middle: float | None = None
    bb_previous_lower: float | None = None
    bb_previous_percent: float | None = None

    # VWAP
    vwap: float | None = None
    session_vwap: float | None = None
    vwap_distance: float | None = None
    vwap_distance_atr: float | None = None
    vwap_zscore: float | None = None

    # Breakout, volume, momentum, and session features shared by the ten
    # production strategy evaluators. Donchian levels explicitly exclude the
    # current candle.
    donchian_previous_high: float | None = None
    donchian_previous_low: float | None = None
    average_volume: float | None = None
    relative_volume: float | None = None
    volume_zscore: float | None = None
    returns: dict[int, float] = field(default_factory=dict)
    roc: float | None = None
    atr_normalized_momentum: float | None = None
    rolling_high: float | None = None
    rolling_low: float | None = None
    supertrend: float | None = None
    supertrend_bullish: bool | None = None
    supertrend_previous: float | None = None
    supertrend_previous_bullish: bool | None = None
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    opening_range_end: str | None = None
    opening_range_complete: bool = False
    candle_body_pct: float | None = None
    upper_wick_pct: float | None = None
    lower_wick_pct: float | None = None
    settled_candle_timestamp: str = ""

    # Commodity Channel Index (EMA-CCI strategy)
    cci: float | None = None

    # Parabolic SAR (EMA-Parabolic-SAR strategy). `psar_bullish=True` means the dot is
    # below price (uptrend); False means above price (downtrend).
    psar: float | None = None
    psar_bullish: bool | None = None

    # Heiken Ashi candle color, current bar and the one before it (EMA-Heiken-Ashi-RSI
    # strategy needs the flip from the previous bar, not just the current color).
    ha_bullish: bool | None = None
    ha_prev_bullish: bool | None = None

    # Volume Profile: Point of Control (highest-volume price bucket over the lookback
    # window) and the Value Area bounds around it (volume_profile_breakout strategy).
    volume_poc: float | None = None
    volume_value_area_high: float | None = None
    volume_value_area_low: float | None = None
    poc_distance_atr: float | None = None

    # High-volume pivot support/resistance zones (pivot_volume_sr_zones strategy). The
    # most recent volume-confirmed structural pivot zone of each kind, and whether price
    # has already closed through it at some point before the current bar -- both "top"/
    # "bottom" pairs are ordinary price levels (top > bottom for both zone kinds).
    pivot_res_zone_top: float | None = None
    pivot_res_zone_bottom: float | None = None
    pivot_res_zone_broken: bool | None = None
    pivot_sup_zone_top: float | None = None
    pivot_sup_zone_bottom: float | None = None
    pivot_sup_zone_broken: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "price": {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "previous_open": self.previous_open,
                "previous_high": self.previous_high,
                "previous_low": self.previous_low,
                "previous_close": self.previous_close,
            },
            "moving_averages": {
                "sma": self.sma,
                "ema": self.ema,
                "ema_previous": self.ema_previous,
                "ema_slope_atr": self.ema_slope_atr,
            },
            "momentum": {
                "rsi": self.rsi,
                "stoch_k": self.stoch_k,
                "stoch_d": self.stoch_d,
            },
            "trend": {
                "macd": self.macd,
                "macd_signal": self.macd_signal,
                "macd_histogram": self.macd_histogram,
                "macd_previous": self.macd_previous,
                "macd_signal_previous": self.macd_signal_previous,
                "macd_histogram_previous": self.macd_histogram_previous,
                "adx": self.adx,
                "plus_di": self.plus_di,
                "minus_di": self.minus_di,
            },
            "volatility": {
                "atr": self.atr,
                "bb_upper": self.bb_upper,
                "bb_middle": self.bb_middle,
                "bb_lower": self.bb_lower,
                "bb_percent": self.bb_percent,
                "bb_bandwidth": self.bb_bandwidth,
                "bb_previous_upper": self.bb_previous_upper,
                "bb_previous_middle": self.bb_previous_middle,
                "bb_previous_lower": self.bb_previous_lower,
                "bb_previous_percent": self.bb_previous_percent,
            },
            "vwap": self.vwap,
            "vwap_features": {
                "session": self.session_vwap,
                "distance": self.vwap_distance,
                "distance_atr": self.vwap_distance_atr,
                "zscore": self.vwap_zscore,
            },
            "breakout": {
                "donchian_previous_high": self.donchian_previous_high,
                "donchian_previous_low": self.donchian_previous_low,
                "opening_range_high": self.opening_range_high,
                "opening_range_low": self.opening_range_low,
                "opening_range_end": self.opening_range_end,
                "opening_range_complete": self.opening_range_complete,
            },
            "volume_features": {
                "average": self.average_volume,
                "relative": self.relative_volume,
                "zscore": self.volume_zscore,
            },
            "returns": self.returns,
            "roc": self.roc,
            "atr_normalized_momentum": self.atr_normalized_momentum,
            "supertrend": self.supertrend,
            "supertrend_bullish": self.supertrend_bullish,
            "supertrend_previous": self.supertrend_previous,
            "supertrend_previous_bullish": self.supertrend_previous_bullish,
            "settled_candle_timestamp": self.settled_candle_timestamp,
            "cci": self.cci,
            "psar": self.psar,
            "psar_bullish": self.psar_bullish,
            "ha_bullish": self.ha_bullish,
            "ha_prev_bullish": self.ha_prev_bullish,
        }


def _heiken_ashi_bullish_flags(df: pd.DataFrame) -> tuple[bool | None, bool | None]:
    """Return (current_bullish, previous_bullish) Heiken Ashi candle color.

    HA_close = (O+H+L+C)/4; HA_open = midpoint of the *previous* HA candle's
    open/close (first bar: midpoint of its own O/C) -- each HA bar depends on the one
    before it, so this is computed with a running loop rather than vectorized. Only the
    last two bars are needed by the EMA-Heiken-Ashi-RSI strategy (to detect a color
    flip), so nothing beyond that is retained.
    """
    if len(df) < 2:
        return None, None

    recent = df.iloc[-30:]
    open_values = recent["open"].to_numpy(dtype=float, copy=False)
    high_values = recent["high"].to_numpy(dtype=float, copy=False)
    low_values = recent["low"].to_numpy(dtype=float, copy=False)
    close_values = recent["close"].to_numpy(dtype=float, copy=False)

    ha_open_prev: float | None = None
    ha_close_prev: float | None = None
    bullish_flags: list[bool] = []
    for o, h, low, c in zip(open_values, high_values, low_values, close_values, strict=True):
        ha_close = (o + h + low + c) / 4
        if ha_open_prev is None or ha_close_prev is None:
            ha_open = (o + c) / 2
        else:
            ha_open = (ha_open_prev + ha_close_prev) / 2
        bullish_flags.append(ha_close > ha_open)
        ha_open_prev, ha_close_prev = ha_open, ha_close

    if len(bullish_flags) < 2:
        return bullish_flags[-1], None
    return bullish_flags[-1], bullish_flags[-2]


def _ema_series(values: np.ndarray, period: int, *, alpha: float | None = None) -> np.ndarray:
    """Match pandas ``ewm(adjust=False, min_periods=period)`` for finite price data.

    The trading loop only consumes the latest value, but MACD also needs the intermediate
    series.  A small NumPy loop avoids constructing several pandas/``ta`` objects for every
    symbol and timeframe while retaining the same recursive EMA definition.
    """
    result = np.full(len(values), np.nan, dtype=float)
    smoothing = alpha if alpha is not None else 2.0 / (period + 1.0)
    current: float | None = None
    observations = 0
    for index, raw_value in enumerate(values):
        if not np.isfinite(raw_value):
            continue
        value = float(raw_value)
        current = value if current is None else smoothing * value + (1.0 - smoothing) * current
        observations += 1
        if observations >= period:
            result[index] = current
    return result


def _ema_series_many(values: np.ndarray, periods: set[int]) -> dict[int, np.ndarray]:
    """Calculate several price EMAs in one pass over the bars."""
    ordered = np.asarray(sorted(periods), dtype=int)
    smoothing = 2.0 / (ordered.astype(float) + 1.0)
    current = np.full(len(ordered), float(values[0]), dtype=float)
    matrix = np.full((len(ordered), len(values)), np.nan, dtype=float)
    for index, raw_value in enumerate(values):
        if index > 0:
            current = smoothing * float(raw_value) + (1.0 - smoothing) * current
        ready = ordered <= index + 1
        matrix[ready, index] = current[ready]
    return {int(period): matrix[index] for index, period in enumerate(ordered)}


def _latest_rsi(close: np.ndarray, period: int) -> float | None:
    if len(close) < period:
        return None
    average_up = 0.0
    average_down = 0.0
    alpha = 1.0 / period
    for difference in np.diff(close):
        upward = max(float(difference), 0.0)
        downward = max(float(-difference), 0.0)
        average_up = alpha * upward + (1.0 - alpha) * average_up
        average_down = alpha * downward + (1.0 - alpha) * average_down
    if average_down == 0:
        return 100.0
    return _safe_float(100.0 - 100.0 / (1.0 + average_up / average_down))


def _latest_stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, smooth: int
) -> tuple[float | None, float | None]:
    if len(close) < window:
        return None, None
    k_values: list[float] = []
    first = max(window - 1, len(close) - smooth)
    for index in range(first, len(close)):
        start = index - window + 1
        lowest = float(np.min(low[start : index + 1]))
        highest = float(np.max(high[start : index + 1]))
        denominator = highest - lowest
        k_values.append(
            math.nan if denominator == 0 else 100.0 * (float(close[index]) - lowest) / denominator
        )
    current = _safe_float(k_values[-1])
    signal = _safe_float(np.mean(k_values[-smooth:])) if len(k_values) >= smooth else None
    return current, signal


def _latest_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
    bins: int,
    value_area_pct: float,
) -> tuple[float | None, float | None, float | None]:
    """Point of Control and Value Area over the trailing ``window`` bars.

    Approximates a volume profile from OHLCV bars (no tick data available): each bar's
    volume is assigned to the price bucket containing its typical price
    ((H+L+C)/3), a common bar-level approximation. POC is the highest-volume bucket;
    the Value Area expands outward from POC, alternating to whichever neighbor bucket
    holds more volume, until ``value_area_pct`` of total volume is covered.
    """
    if len(close) < window or bins < 2:
        return None, None, None
    h = high[-window:]
    lo = low[-window:]
    c = close[-window:]
    v = volume[-window:]
    price_low = float(np.min(lo))
    price_high = float(np.max(h))
    if price_high <= price_low:
        return None, None, None
    typical = (h + lo + c) / 3.0
    bucket_width = (price_high - price_low) / bins
    bucket_idx = np.clip(
        ((typical - price_low) / bucket_width).astype(int), 0, bins - 1
    )
    bucket_volume = np.zeros(bins, dtype=float)
    for idx, vol in zip(bucket_idx, v, strict=True):
        bucket_volume[idx] += float(vol)
    total_volume = float(np.sum(bucket_volume))
    if total_volume <= 0.0:
        return None, None, None
    poc_idx = int(np.argmax(bucket_volume))
    poc_price = price_low + (poc_idx + 0.5) * bucket_width

    covered = bucket_volume[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    target = value_area_pct * total_volume
    while covered < target and (lo_idx > 0 or hi_idx < bins - 1):
        next_lo_volume = bucket_volume[lo_idx - 1] if lo_idx > 0 else -1.0
        next_hi_volume = bucket_volume[hi_idx + 1] if hi_idx < bins - 1 else -1.0
        if next_hi_volume >= next_lo_volume:
            hi_idx += 1
            covered += bucket_volume[hi_idx]
        else:
            lo_idx -= 1
            covered += bucket_volume[lo_idx]
    value_area_low = price_low + lo_idx * bucket_width
    value_area_high = price_low + (hi_idx + 1) * bucket_width
    return _safe_float(poc_price), _safe_float(value_area_high), _safe_float(value_area_low)


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous_close = np.concatenate(([np.nan], close[:-1]))
    values = np.nanmax(
        np.vstack((high - low, np.abs(high - previous_close), np.abs(low - previous_close))),
        axis=0,
    )
    return np.asarray(values, dtype=float)


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Wilder-smoothed ATR at every bar (not just the latest) -- needed by
    _latest_pivot_sr_zones, which sizes each zone using the ATR value AT the pivot bar,
    not the ATR now. Same recurrence as _latest_atr, just retaining every step."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < window:
        return result
    true_range = _true_range(high, low, close)
    current = float(np.mean(true_range[:window]))
    result[window - 1] = current
    for idx in range(window, n):
        current = (current * (window - 1) + float(true_range[idx])) / window
        result[idx] = current
    return result


def _rolling_pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """True at index i if high[i] is the strict max of the (left, right) bars around it.

    Mirrors Pine's ta.pivothigh(high, left, right): a peak is only usable once `right`
    bars past it exist, which this full-array computation naturally respects -- index i
    is only ever read by a caller once the array extends to at least i+right.
    """
    n = len(high)
    result = np.zeros(n, dtype=bool)
    if left < 1 or right < 1:
        return result
    for i in range(left, n - right):
        if high[i] > np.max(high[i - left : i]) and high[i] > np.max(high[i + 1 : i + right + 1]):
            result[i] = True
    return result


def _rolling_pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """True at index i if low[i] is the strict min of the (left, right) bars around it."""
    n = len(low)
    result = np.zeros(n, dtype=bool)
    if left < 1 or right < 1:
        return result
    for i in range(left, n - right):
        if low[i] < np.min(low[i - left : i]) and low[i] < np.min(low[i + 1 : i + right + 1]):
            result[i] = True
    return result


def _latest_pivot_sr_zones(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pivot_len: int,
    vol_avg_len: int,
    vol_mult: float,
    atr_len: int,
) -> tuple[float | None, float | None, bool | None, float | None, float | None, bool | None]:
    """Most recent volume-confirmed structural resistance/support zone, and whether each
    has already been closed through at some point before the current (last) bar.

    Adapted from a user-supplied TradingView Pine Script ("High Volume Pivot Support &
    Resistance Zones [BigBeluga]"). This reproduces its zone-formation and break-state
    logic (the actual decision inputs), not its box-drawing/CVD-delta visualization,
    which has no bearing on a trading decision. A pivot only starts a new zone if its
    bar's volume exceeded vol_mult times its own trailing vol_avg_len-bar average --
    low-volume swing points are ignored entirely, same as the source script. Returns
    None-filled values (a strategy consuming this must treat that as "no zone yet") when
    the window is too short or no volume-confirmed pivot exists in it.
    """
    n = len(close)
    if n < pivot_len * 2 + max(vol_avg_len, atr_len) + 2:
        return None, None, None, None, None, None

    avg_vol = pd.Series(volume).rolling(vol_avg_len).mean().to_numpy()
    is_high_volume = volume > avg_vol * vol_mult

    atr = _atr_series(high, low, close, atr_len)
    pivot_high = _rolling_pivot_high(high, pivot_len, pivot_len)
    pivot_low = _rolling_pivot_low(low, pivot_len, pivot_len)

    # "Now" is the last available bar. Only a zone CONFIRMED strictly before now is used --
    # a bar that itself just confirmed a brand-new pivot resets the zone and produces no
    # reaction that same bar in the source script (the trigger checks live in its
    # `else` branch, never alongside a fresh pivot).
    now = n - 1

    res_top = res_bottom = None
    res_broken: bool | None = None
    for peak in range(now - 1 - pivot_len, pivot_len - 1, -1):
        confirm_idx = peak + pivot_len
        if confirm_idx > now - 1:
            continue
        if not (pivot_high[peak] and is_high_volume[peak] and not np.isnan(atr[peak])):
            continue
        body_top = max(open_[peak], close[peak])
        res_bottom = body_top
        res_top = body_top + atr[peak]
        res_broken = bool(np.any(close[confirm_idx:now] > res_top))
        break

    sup_top = sup_bottom = None
    sup_broken: bool | None = None
    for trough in range(now - 1 - pivot_len, pivot_len - 1, -1):
        confirm_idx = trough + pivot_len
        if confirm_idx > now - 1:
            continue
        if not (pivot_low[trough] and is_high_volume[trough] and not np.isnan(atr[trough])):
            continue
        body_bottom = min(open_[trough], close[trough])
        sup_top = body_bottom
        sup_bottom = body_bottom - atr[trough]
        sup_broken = bool(np.any(close[confirm_idx:now] < sup_bottom))
        break

    return (
        _safe_float(res_top),
        _safe_float(res_bottom),
        res_broken,
        _safe_float(sup_top),
        _safe_float(sup_bottom),
        sup_broken,
    )


def _latest_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> float | None:
    if len(close) < window:
        return None
    true_range = _true_range(high, low, close)
    current = float(np.mean(true_range[:window]))
    for value in true_range[window:]:
        current = (current * (window - 1) + float(value)) / window
    return _safe_float(current)


def _latest_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> tuple[float | None, float | None, float | None]:
    """Return ADX/+DI/-DI using the same Wilder recurrence as ``ta.ADXIndicator``."""
    size = len(close) - (window - 1)
    if size <= window:
        return None, None, None

    true_range = _true_range(high, low, close)
    trs = np.zeros(size, dtype=float)
    trs[0] = float(np.sum(true_range[1 : window + 1]))

    up = np.concatenate(([np.nan], np.diff(high)))
    down = np.concatenate(([np.nan], -np.diff(low)))
    positive = np.where((up > down) & (up > 0), up, 0.0)
    negative = np.where((down > up) & (down > 0), down, 0.0)
    positive[0] = np.nan
    negative[0] = np.nan

    dip = np.zeros(size, dtype=float)
    din = np.zeros(size, dtype=float)
    dip[0] = float(np.nansum(positive[: window + 1]))
    din[0] = float(np.nansum(negative[: window + 1]))

    for index in range(1, size - 1):
        source = window + index
        trs[index] = trs[index - 1] - trs[index - 1] / window + true_range[source]
        dip[index] = dip[index - 1] - dip[index - 1] / window + positive[source]
        din[index] = din[index - 1] - din[index - 1] / window + negative[source]

    plus = np.divide(100.0 * dip, trs, out=np.zeros_like(dip), where=trs != 0)
    minus = np.divide(100.0 * din, trs, out=np.zeros_like(din), where=trs != 0)
    denominator = plus + minus
    directional = np.divide(
        100.0 * np.abs(plus - minus),
        denominator,
        out=np.zeros_like(plus),
        where=denominator != 0,
    )

    adx = np.zeros(size, dtype=float)
    adx[window] = float(np.mean(directional[:window]))
    for index in range(window + 1, size):
        adx[index] = (adx[index - 1] * (window - 1) + directional[index - 1]) / window

    latest_smoothed = size - 2
    return (
        _safe_float(adx[-1]),
        _safe_float(plus[latest_smoothed]),
        _safe_float(minus[latest_smoothed]),
    )


def _latest_psar(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    step: float,
    max_step: float,
) -> float | None:
    """Compute the latest Parabolic SAR without pandas scalar indexing.

    This is the same recurrence used by ``ta.PSARIndicator``. Its pandas ``iloc`` loop
    accounted for roughly two thirds of a 250-symbol cold scan in profiling.
    """
    if len(close) < 2:
        return None
    psar = close.copy()
    up_trend = True
    acceleration = step
    up_high = float(high[0])
    down_low = float(low[0])

    for index in range(2, len(close)):
        reversal = False
        max_high = float(high[index])
        min_low = float(low[index])
        if up_trend:
            psar[index] = psar[index - 1] + acceleration * (up_high - psar[index - 1])
            if min_low < psar[index]:
                reversal = True
                psar[index] = up_high
                down_low = min_low
                acceleration = step
            else:
                if max_high > up_high:
                    up_high = max_high
                    acceleration = min(acceleration + step, max_step)
                if low[index - 2] < psar[index]:
                    psar[index] = low[index - 2]
                elif low[index - 1] < psar[index]:
                    psar[index] = low[index - 1]
        else:
            psar[index] = psar[index - 1] - acceleration * (psar[index - 1] - down_low)
            if max_high > psar[index]:
                reversal = True
                psar[index] = down_low
                up_high = max_high
                acceleration = step
            else:
                if min_low < down_low:
                    down_low = min_low
                    acceleration = min(acceleration + step, max_step)
                if high[index - 2] > psar[index]:
                    psar[index] = high[index - 2]
                elif high[index - 1] > psar[index]:
                    psar[index] = high[index - 1]
        up_trend = up_trend != reversal
    return _safe_float(psar[-1])


def _wilder_atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Return a Wilder ATR series using only information available at each bar."""

    result = np.full(len(close), np.nan, dtype=float)
    if len(close) < period:
        return result
    true_range = _true_range(high, low, close)
    current = float(np.mean(true_range[:period]))
    result[period - 1] = current
    for index in range(period, len(close)):
        current = (current * (period - 1) + float(true_range[index])) / period
        result[index] = current
    return result


def _supertrend_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate leakage-free SuperTrend values and bullish/bearish state.

    Final bands are recursively clamped using the previous completed bar, matching
    the standard SuperTrend definition. No future bar is read.
    """

    size = len(close)
    values = np.full(size, np.nan, dtype=float)
    bullish = np.zeros(size, dtype=bool)
    atr = _wilder_atr_series(high, low, close, period)
    midpoint = (high + low) / 2.0
    upper = midpoint + multiplier * atr
    lower = midpoint - multiplier * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    first = period - 1
    if size <= first or not np.isfinite(atr[first]):
        return values, bullish
    bullish[first] = bool(close[first] >= midpoint[first])
    values[first] = final_lower[first] if bullish[first] else final_upper[first]
    for index in range(first + 1, size):
        if not np.isfinite(atr[index]):
            continue
        if upper[index] < final_upper[index - 1] or close[index - 1] > final_upper[index - 1]:
            final_upper[index] = upper[index]
        else:
            final_upper[index] = final_upper[index - 1]
        if lower[index] > final_lower[index - 1] or close[index - 1] < final_lower[index - 1]:
            final_lower[index] = lower[index]
        else:
            final_lower[index] = final_lower[index - 1]

        previous_bullish = bool(bullish[index - 1])
        if previous_bullish:
            bullish[index] = bool(close[index] >= final_lower[index])
        else:
            bullish[index] = bool(close[index] > final_upper[index])
        values[index] = final_lower[index] if bullish[index] else final_upper[index]
    return values, bullish


def _index_in_ist(index: pd.Index) -> pd.DatetimeIndex | None:
    """Normalize candle timestamps for NSE session features.

    Naive timestamps in DeltaQuant's historical frames represent exchange-local time.
    Aware timestamps are converted to Asia/Kolkata. Non-datetime test fixtures simply
    abstain from session features instead of guessing a session.
    """

    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    except (TypeError, ValueError):
        return None
    if timestamps.tz is None:
        return timestamps.tz_localize("Asia/Kolkata")
    return timestamps.tz_convert("Asia/Kolkata")


def _session_features(
    df: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: float | None,
    opening_range_minutes: int,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    str | None,
    bool,
]:
    """Return session VWAP/deviation and completed NSE opening-range state."""

    timestamps = _index_in_ist(df.index)
    if timestamps is None or len(timestamps) == 0:
        return None, None, None, None, None, None, None, False
    latest_day = timestamps[-1].date()
    session_mask = np.asarray([stamp.date() == latest_day for stamp in timestamps], dtype=bool)
    session_positions = np.flatnonzero(session_mask)
    if len(session_positions) == 0:
        return None, None, None, None, None, None, None, False

    session_high = high[session_positions]
    session_low = low[session_positions]
    session_close = close[session_positions]
    session_volume = volume[session_positions]
    typical = (session_high + session_low + session_close) / 3.0
    cumulative_volume = np.cumsum(session_volume)
    cumulative_value = np.cumsum(typical * session_volume)
    valid = cumulative_volume > 0
    vwap_series = np.full(len(session_positions), np.nan, dtype=float)
    vwap_series[valid] = cumulative_value[valid] / cumulative_volume[valid]
    vwap = _safe_float(vwap_series[-1])
    distance = _safe_float(session_close[-1] - vwap) if vwap is not None else None
    distance_atr = (
        _safe_float(distance / float(atr))
        if distance is not None and atr is not None and atr != 0.0
        else None
    )
    deviations = session_close - vwap_series
    finite_deviations = deviations[np.isfinite(deviations)]
    vwap_zscore = None
    if len(finite_deviations) >= 5:
        deviation_std = float(np.std(finite_deviations, ddof=0))
        if deviation_std > 0:
            vwap_zscore = _safe_float(
                (finite_deviations[-1] - float(np.mean(finite_deviations))) / deviation_std
            )

    session_open = pd.Timestamp(latest_day, tz="Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
    opening_end = session_open + pd.Timedelta(minutes=opening_range_minutes)
    range_positions = [
        position
        for position in session_positions
        if session_open <= timestamps[position] < opening_end
    ]
    opening_high = _safe_float(np.max(high[range_positions])) if range_positions else None
    opening_low = _safe_float(np.min(low[range_positions])) if range_positions else None
    complete = bool(range_positions and timestamps[-1] >= opening_end)
    return (
        vwap,
        distance,
        distance_atr,
        vwap_zscore,
        opening_high,
        opening_low,
        opening_end.isoformat(),
        complete,
    )


def calculate_indicators(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe = Timeframe.M5,
    config: IndicatorConfig | None = None,
) -> IndicatorResult:
    """
    Calculate all technical indicators for a symbol.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        symbol: Trading symbol
        timeframe: Candle timeframe
        config: Indicator configuration (uses defaults if None)

    Returns:
        IndicatorResult with all calculated indicators
    """
    if config is None:
        config = IndicatorConfig()

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Materialize each input column once. The live scanner needs only the latest
    # indicator snapshot; repeatedly constructing full pandas Series through the
    # ``ta`` wrappers made a changed 250-symbol generation take minutes.
    open_values = df["open"].to_numpy(dtype=float, copy=False)
    high_values = df["high"].to_numpy(dtype=float, copy=False)
    low_values = df["low"].to_numpy(dtype=float, copy=False)
    close_values = df["close"].to_numpy(dtype=float, copy=False)
    volume_values = df["volume"].to_numpy(dtype=float, copy=False)

    # Get latest values
    latest = df.iloc[-1]

    # Calculate Moving Averages (skip NaN/warm-up values so membership guards hold)
    sma_values: dict[int, float] = {}
    for period in config.sma_periods:
        if len(df) >= period:
            value = _safe_float(np.mean(close_values[-period:]))
            if value is not None:
                sma_values[period] = value

    ema_values: dict[int, float] = {}
    ema_previous: dict[int, float] = {}
    ema_slope_atr: dict[int, float] = {}
    required_ema_periods = set(config.ema_periods) | {config.macd_fast, config.macd_slow}
    ema_series_by_period = _ema_series_many(close_values, required_ema_periods)
    for period in config.ema_periods:
        if len(df) >= period:
            value = _safe_float(ema_series_by_period[period][-1])
            if value is not None:
                ema_values[period] = value
            if len(df) > period:
                previous = _safe_float(ema_series_by_period[period][-2])
                if previous is not None:
                    ema_previous[period] = previous

    # Calculate RSI
    rsi = _latest_rsi(close_values, config.rsi_period)

    # Calculate Stochastic
    stoch_k, stoch_d = _latest_stochastic(
        high_values,
        low_values,
        close_values,
        config.stoch_k_period,
        config.stoch_d_period,
    )

    # Calculate MACD
    macd_val, macd_signal_val, macd_hist = None, None, None
    macd_previous, macd_signal_previous, macd_hist_previous = None, None, None
    if len(df) >= config.macd_slow:
        fast = ema_series_by_period[config.macd_fast]
        slow = ema_series_by_period[config.macd_slow]
        macd_series = fast - slow
        signal_series = _ema_series(macd_series, config.macd_signal)
        macd_val = _safe_float(macd_series[-1])
        macd_signal_val = _safe_float(signal_series[-1])
        if macd_val is not None and macd_signal_val is not None:
            macd_hist = _safe_float(macd_val - macd_signal_val)
        macd_previous = _safe_float(macd_series[-2]) if len(macd_series) > 1 else None
        macd_signal_previous = _safe_float(signal_series[-2]) if len(signal_series) > 1 else None
        if macd_previous is not None and macd_signal_previous is not None:
            macd_hist_previous = _safe_float(macd_previous - macd_signal_previous)

    # Calculate ADX
    adx_val, plus_di, minus_di = _latest_adx(
        high_values,
        low_values,
        close_values,
        config.adx_period,
    )

    # Calculate ATR
    atr = _latest_atr(
        high_values,
        low_values,
        close_values,
        config.atr_period,
    )
    if atr is not None and atr != 0.0:
        for period, value in ema_values.items():
            if period not in ema_previous:
                continue
            slope = _safe_float((value - ema_previous[period]) / atr)
            if slope is not None:
                ema_slope_atr[period] = slope

    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_percent = None, None, None, None
    bb_bandwidth = None
    bb_previous_upper, bb_previous_middle, bb_previous_lower, bb_previous_percent = (
        None,
        None,
        None,
        None,
    )
    if len(df) >= config.bb_period:
        bb_window = close_values[-config.bb_period :]
        bb_middle = _safe_float(np.mean(bb_window))
        deviation = _safe_float(np.std(bb_window, ddof=0))
        if bb_middle is not None and deviation is not None:
            bb_upper = bb_middle + config.bb_std * deviation
            bb_lower = bb_middle - config.bb_std * deviation
            width = bb_upper - bb_lower
            bb_percent = (
                _safe_float((float(close_values[-1]) - bb_lower) / width) if width != 0 else None
            )
            bb_bandwidth = _safe_float(width / bb_middle) if bb_middle != 0 else None
    if len(df) > config.bb_period:
        previous_window = close_values[-config.bb_period - 1 : -1]
        bb_previous_middle = _safe_float(np.mean(previous_window))
        previous_deviation = _safe_float(np.std(previous_window, ddof=0))
        if bb_previous_middle is not None and previous_deviation is not None:
            bb_previous_upper = bb_previous_middle + config.bb_std * previous_deviation
            bb_previous_lower = bb_previous_middle - config.bb_std * previous_deviation
            previous_width = bb_previous_upper - bb_previous_lower
            if previous_width != 0:
                bb_previous_percent = _safe_float(
                    (float(close_values[-2]) - bb_previous_lower) / previous_width
                )

    # Volume, Donchian and momentum features. The current candle is deliberately
    # excluded from Donchian bounds; including it makes a close breakout impossible
    # and is a common self-reference bug.
    average_volume, relative_volume, volume_zscore = None, None, None
    if len(df) >= config.volume_period:
        volume_window = volume_values[-config.volume_period :]
        average_volume = _safe_float(np.mean(volume_window))
        if average_volume is not None and average_volume != 0.0:
            relative_volume = _safe_float(float(volume_values[-1]) / average_volume)
        volume_std = float(np.std(volume_window, ddof=0))
        if volume_std > 0:
            volume_zscore = _safe_float(
                (float(volume_values[-1]) - float(np.mean(volume_window))) / volume_std
            )

    # Exclude the current (possibly still-forming) bar from the profile window --
    # otherwise a breakout bar's own volume helps define the very level it's "breaking",
    # the same self-reference bug Donchian bounds above deliberately avoid.
    volume_poc, volume_value_area_high, volume_value_area_low = _latest_volume_profile(
        high_values[:-1],
        low_values[:-1],
        close_values[:-1],
        volume_values[:-1],
        config.volume_profile_period,
        config.volume_profile_bins,
        config.volume_profile_value_area_pct,
    )
    poc_distance_atr = None
    if volume_poc is not None and atr is not None and atr != 0.0:
        poc_distance_atr = _safe_float((close_values[-1] - volume_poc) / atr)

    (
        pivot_res_zone_top,
        pivot_res_zone_bottom,
        pivot_res_zone_broken,
        pivot_sup_zone_top,
        pivot_sup_zone_bottom,
        pivot_sup_zone_broken,
    ) = _latest_pivot_sr_zones(
        open_values,
        high_values,
        low_values,
        close_values,
        volume_values,
        config.pivot_sr_length,
        config.pivot_sr_volume_avg_length,
        config.pivot_sr_volume_multiplier,
        config.pivot_sr_atr_length,
    )

    donchian_previous_high, donchian_previous_low = None, None
    if len(df) > config.donchian_period:
        donchian_previous_high = _safe_float(np.max(high_values[-config.donchian_period - 1 : -1]))
        donchian_previous_low = _safe_float(np.min(low_values[-config.donchian_period - 1 : -1]))

    returns: dict[int, float] = {}
    for period in config.momentum_periods:
        if len(df) > period and close_values[-period - 1] != 0:
            value = _safe_float(close_values[-1] / close_values[-period - 1] - 1.0)
            if value is not None:
                returns[period] = value
    longest_momentum = max(config.momentum_periods, default=1)
    roc = returns.get(longest_momentum)
    atr_normalized_momentum = None
    if len(df) > longest_momentum and atr is not None and atr != 0.0:
        atr_normalized_momentum = _safe_float(
            (close_values[-1] - close_values[-longest_momentum - 1]) / atr
        )
    rolling_high = (
        _safe_float(np.max(high_values[-config.donchian_period :]))
        if len(df) >= config.donchian_period
        else None
    )
    rolling_low = (
        _safe_float(np.min(low_values[-config.donchian_period :]))
        if len(df) >= config.donchian_period
        else None
    )

    supertrend_series, supertrend_direction = _supertrend_series(
        high_values,
        low_values,
        close_values,
        config.supertrend_period,
        config.supertrend_multiplier,
    )
    supertrend = _safe_float(supertrend_series[-1])
    supertrend_previous = _safe_float(supertrend_series[-2]) if len(supertrend_series) > 1 else None
    supertrend_bullish = bool(supertrend_direction[-1]) if supertrend is not None else None
    supertrend_previous_bullish = (
        bool(supertrend_direction[-2]) if supertrend_previous is not None else None
    )

    (
        session_vwap,
        vwap_distance,
        vwap_distance_atr,
        vwap_zscore,
        opening_range_high,
        opening_range_low,
        opening_range_end,
        opening_range_complete,
    ) = _session_features(
        df,
        high_values,
        low_values,
        close_values,
        volume_values,
        atr,
        config.opening_range_minutes,
    )
    # Preserve the legacy 14-bar rolling VWAP field for existing consumers. The
    # production intraday strategies use ``session_vwap`` exclusively.
    vwap = None
    if len(df) >= 14:
        typical = (high_values[-14:] + low_values[-14:] + close_values[-14:]) / 3.0
        recent_volume = volume_values[-14:]
        total_volume = float(np.sum(recent_volume))
        if total_volume != 0:
            vwap = _safe_float(np.sum(typical * recent_volume) / total_volume)

    # Commodity Channel Index (EMA-CCI strategy)
    cci = None
    if len(df) >= config.cci_period:
        typical = (high_values + low_values + close_values) / 3.0
        recent_typical = typical[-config.cci_period :]
        typical_mean = float(np.mean(recent_typical))
        mean_deviation = float(np.mean(np.abs(recent_typical - typical_mean)))
        if mean_deviation != 0:
            cci = _safe_float((float(recent_typical[-1]) - typical_mean) / (0.015 * mean_deviation))

    # Calculate Parabolic SAR (EMA-Parabolic-SAR strategy)
    psar = _latest_psar(
        high_values,
        low_values,
        close_values,
        config.psar_step,
        config.psar_max_step,
    )
    psar_bullish = psar < float(latest["close"]) if psar is not None else None

    # Heiken Ashi candle color, current + previous bar (EMA-Heiken-Ashi-RSI strategy)
    ha_bullish, ha_prev_bullish = _heiken_ashi_bullish_flags(df)

    candle_range = float(latest["high"] - latest["low"])
    candle_body_pct = (
        _safe_float(abs(float(latest["close"] - latest["open"])) / candle_range)
        if candle_range > 0
        else None
    )
    upper_wick_pct = (
        _safe_float(
            (float(latest["high"]) - max(float(latest["open"]), float(latest["close"])))
            / candle_range
        )
        if candle_range > 0
        else None
    )
    lower_wick_pct = (
        _safe_float(
            (min(float(latest["open"]), float(latest["close"])) - float(latest["low"]))
            / candle_range
        )
        if candle_range > 0
        else None
    )

    return IndicatorResult(
        symbol=symbol,
        timeframe=timeframe,
        open=float(latest["open"]),
        high=float(latest["high"]),
        low=float(latest["low"]),
        close=float(latest["close"]),
        volume=int(latest["volume"]),
        previous_open=_safe_float(open_values[-2]) if len(df) > 1 else None,
        previous_high=_safe_float(high_values[-2]) if len(df) > 1 else None,
        previous_low=_safe_float(low_values[-2]) if len(df) > 1 else None,
        previous_close=_safe_float(close_values[-2]) if len(df) > 1 else None,
        sma=sma_values,
        ema=ema_values,
        ema_previous=ema_previous,
        ema_slope_atr=ema_slope_atr,
        rsi=rsi,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        macd=macd_val,
        macd_signal=macd_signal_val,
        macd_histogram=macd_hist,
        macd_previous=macd_previous,
        macd_signal_previous=macd_signal_previous,
        macd_histogram_previous=macd_hist_previous,
        adx=adx_val,
        plus_di=plus_di,
        minus_di=minus_di,
        atr=atr,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_percent=bb_percent,
        bb_bandwidth=bb_bandwidth,
        bb_previous_upper=bb_previous_upper,
        bb_previous_middle=bb_previous_middle,
        bb_previous_lower=bb_previous_lower,
        bb_previous_percent=bb_previous_percent,
        vwap=vwap,
        session_vwap=session_vwap,
        vwap_distance=vwap_distance,
        vwap_distance_atr=vwap_distance_atr,
        vwap_zscore=vwap_zscore,
        donchian_previous_high=donchian_previous_high,
        donchian_previous_low=donchian_previous_low,
        average_volume=average_volume,
        relative_volume=relative_volume,
        volume_zscore=volume_zscore,
        returns=returns,
        roc=roc,
        atr_normalized_momentum=atr_normalized_momentum,
        rolling_high=rolling_high,
        rolling_low=rolling_low,
        supertrend=supertrend,
        supertrend_bullish=supertrend_bullish,
        supertrend_previous=supertrend_previous,
        supertrend_previous_bullish=supertrend_previous_bullish,
        opening_range_high=opening_range_high,
        opening_range_low=opening_range_low,
        opening_range_end=opening_range_end,
        opening_range_complete=opening_range_complete,
        candle_body_pct=candle_body_pct,
        upper_wick_pct=upper_wick_pct,
        lower_wick_pct=lower_wick_pct,
        settled_candle_timestamp=str(df.index[-1]),
        cci=cci,
        psar=psar,
        psar_bullish=psar_bullish,
        ha_bullish=ha_bullish,
        ha_prev_bullish=ha_prev_bullish,
        volume_poc=volume_poc,
        volume_value_area_high=volume_value_area_high,
        volume_value_area_low=volume_value_area_low,
        poc_distance_atr=poc_distance_atr,
        pivot_res_zone_top=pivot_res_zone_top,
        pivot_res_zone_bottom=pivot_res_zone_bottom,
        pivot_res_zone_broken=pivot_res_zone_broken,
        pivot_sup_zone_top=pivot_sup_zone_top,
        pivot_sup_zone_bottom=pivot_sup_zone_bottom,
        pivot_sup_zone_broken=pivot_sup_zone_broken,
    )


class IndicatorCache:
    """
    Memoizes indicator results so they are recomputed only when the data changes.

    Keyed by (symbol, timeframe, settled timestamp, feature version and config), so a hit
    requires identical inputs — when computing on settled bars the key is stable for the
    whole trading day (compute once instead of every cycle); a forming bar whose close
    moves busts the key correctly. Assumes the default IndicatorConfig.
    """

    # A 250-symbol universe scanned over five timeframes needs 1,250 live keys.
    # The old 256-entry FIFO cache thrashed: by the time the next cycle reached the
    # end of the universe, the entries it wanted had already been evicted. Keep
    # enough room for several full-universe generations (forming + settled bars).
    _MAX_ENTRIES = 4096

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[Any, ...], IndicatorResult] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        config: IndicatorConfig | None = None,
        market: str = "NSE",
    ) -> tuple[Any, ...]:
        effective_config = config or IndicatorConfig()
        config_key = tuple(
            (name, repr(value)) for name, value in sorted(vars(effective_config).items())
        )
        market_key = market.strip().upper()
        if len(df) == 0:
            return (market_key, symbol, timeframe.value, None, FEATURE_SET_VERSION, config_key)
        last_close = _safe_float(df["close"].iloc[-1])
        return (
            market_key,
            symbol,
            timeframe.value,
            str(df.index[-1]),
            FEATURE_SET_VERSION,
            config_key,
            round(last_close, 6) if last_close is not None else None,
        )

    def get_or_compute(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe = Timeframe.D1,
        config: IndicatorConfig | None = None,
        *,
        market: str = "NSE",
    ) -> IndicatorResult:
        key = self._key(df, symbol, timeframe, config, market)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return cached

        # Indicator calculation is CPU work. Do it outside the lock so bounded
        # full-universe workers can calculate different symbols concurrently.
        result = calculate_indicators(df, symbol, timeframe, config)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return existing
            self.misses += 1
            self._cache[key] = result
            if len(self._cache) > self._MAX_ENTRIES:
                self._cache.popitem(last=False)
        return result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


_indicator_cache: IndicatorCache | None = None


def get_indicator_cache() -> IndicatorCache:
    """Get or create the shared IndicatorCache."""
    global _indicator_cache
    if _indicator_cache is None:
        _indicator_cache = IndicatorCache()
    return _indicator_cache


def reset_indicator_cache() -> None:
    """Reset the shared IndicatorCache (test isolation)."""
    global _indicator_cache
    _indicator_cache = None


def aggregate_candles(
    ticks: list[dict[str, Any]],
    timeframe: Timeframe,
) -> pd.DataFrame:
    """
    Aggregate tick data into OHLCV candles.

    Args:
        ticks: List of tick dictionaries with 'price', 'volume', 'timestamp'
        timeframe: Target candle timeframe

    Returns:
        DataFrame with OHLCV columns
    """
    if not ticks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Map timeframe to pandas resample string
    resample_map = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
        Timeframe.D1: "1D",
    }

    resampled = df.resample(resample_map[timeframe]).agg(
        {
            "price": ["first", "max", "min", "last"],
            "volume": "sum",
        }
    )

    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled.dropna(inplace=True)

    return resampled
