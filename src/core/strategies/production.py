"""Leakage-safe evaluators for DeltaQuant's production technical strategies.

This module performs no indicator calculation and has no execution authority. Every
evaluator consumes one shared ``IndicatorResult`` (plus optional cross-sectional context)
and returns either one normalized technical decision or ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.core.features.context import MarketRelativeFeatures
from src.core.indicators import IndicatorResult
from src.core.strategies.config import ProductionStrategyConfig


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quality_above(value: float, threshold: float) -> float:
    """Continuous quality with 0.5 at the required gate and 1 at twice it."""

    if threshold <= 0:
        return 1.0
    return _clip(0.5 + 0.5 * (value - threshold) / threshold)


def _weighted_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    present = {name: _clip(value) for name, value in components.items() if name in weights}
    denominator = sum(float(weights[name]) for name in present)
    if denominator <= 0:
        return 0.0
    return round(
        sum(present[name] * float(weights[name]) for name in present) / denominator,
        6,
    )


@dataclass(frozen=True)
class TechnicalStrategyDecision:
    direction: str
    technical_score: float
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]
    score_components: Mapping[str, float]
    supporting_metrics: Mapping[str, float | int | str | bool]


def _decision(
    strategy_name: str,
    direction: str,
    config: ProductionStrategyConfig,
    components: Mapping[str, float],
    reason_codes: tuple[str, ...],
    reasons: tuple[str, ...],
    metrics: Mapping[str, float | int | str | bool],
) -> TechnicalStrategyDecision:
    normalized = {name: round(_clip(value), 6) for name, value in components.items()}
    return TechnicalStrategyDecision(
        direction=direction,
        technical_score=_weighted_score(normalized, config.weights(strategy_name)),
        reason_codes=reason_codes,
        reasons=reasons,
        score_components=normalized,
        supporting_metrics=metrics,
    )


def _ema_adx_trend(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "ema_adx_trend"
    required = (ind.adx, ind.plus_di, ind.minus_di, ind.atr)
    if any(value is None for value in required) or not ind.ema or not ind.ema_slope_atr:
        return None
    if 20 not in ind.ema or 50 not in ind.ema or 20 not in ind.ema_slope_atr:
        return None
    assert ind.adx is not None
    assert ind.plus_di is not None
    assert ind.minus_di is not None
    assert ind.atr is not None
    p = config.parameters[name]
    ema20, ema50 = ind.ema[20], ind.ema[50]
    slope = ind.ema_slope_atr[20]
    extension = abs(ind.close - ema20) / float(ind.atr)
    rvol = ind.relative_volume if ind.relative_volume is not None else 1.0
    bullish = (
        ema20 > ema50
        and ind.close > ema20
        and slope >= p["min_ema_slope_atr"]
        and ind.adx >= p["adx_min"]
        and ind.plus_di > ind.minus_di
        and extension <= p["max_extension_atr"]
        and rvol >= p["min_relative_volume"]
    )
    bearish = (
        ema20 < ema50
        and ind.close < ema20
        and slope <= -p["min_ema_slope_atr"]
        and ind.adx >= p["adx_min"]
        and ind.minus_di > ind.plus_di
        and extension <= p["max_extension_atr"]
        and rvol >= p["min_relative_volume"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    components = {
        "trend_quality": _quality_above(ind.adx, p["adx_min"]),
        "directional_quality": _clip(abs(ind.plus_di - ind.minus_di) / max(ind.adx, 1.0)),
        "entry_quality": _clip(1.0 - extension / p["max_extension_atr"]),
        "volume_confirmation": _quality_above(rvol, p["min_relative_volume"]),
    }
    return _decision(
        name,
        direction,
        config,
        components,
        ("EMA_TREND_CONFIRMED", "ADX_TREND_CONFIRMED", "DIRECTIONAL_INDEX_CONFIRMED"),
        (
            f"EMA20/EMA50 structure confirms {direction.lower()} trend",
            f"ADX {ind.adx:.2f} and directional index confirm trend quality",
            f"EMA20 extension is {extension:.2f} ATR",
        ),
        {
            "ema20": ema20,
            "ema50": ema50,
            "ema20_slope_atr": slope,
            "adx": ind.adx,
            "relative_volume": rvol,
            "extension_atr": extension,
        },
    )


def _donchian_breakout(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "donchian_breakout"
    if (
        ind.donchian_previous_high is None
        or ind.donchian_previous_low is None
        or ind.previous_close is None
        or ind.atr in (None, 0.0)
        or ind.relative_volume is None
        or ind.adx is None
    ):
        return None
    assert ind.donchian_previous_high is not None
    assert ind.donchian_previous_low is not None
    assert ind.previous_close is not None
    assert ind.atr is not None
    assert ind.relative_volume is not None
    assert ind.adx is not None
    p = config.parameters[name]
    up_extension = (ind.close - ind.donchian_previous_high) / ind.atr
    down_extension = (ind.donchian_previous_low - ind.close) / ind.atr
    bullish = (
        ind.close > ind.donchian_previous_high
        and ind.previous_close <= ind.donchian_previous_high
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= up_extension <= p["max_breakout_extension_atr"]
    )
    bearish = (
        ind.close < ind.donchian_previous_low
        and ind.previous_close >= ind.donchian_previous_low
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= down_extension <= p["max_breakout_extension_atr"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    extension = up_extension if bullish else down_extension
    level = ind.donchian_previous_high if bullish else ind.donchian_previous_low
    return _decision(
        name,
        direction,
        config,
        {
            "breakout_quality": _clip(1.0 - extension / p["max_breakout_extension_atr"]),
            "volume_confirmation": _quality_above(ind.relative_volume, p["min_relative_volume"]),
            "trend_quality": _quality_above(ind.adx, p["min_adx"]),
            "entry_quality": _clip(ind.candle_body_pct or 0.0),
        },
        ("DONCHIAN_BREAKOUT", "RELATIVE_VOLUME_CONFIRMED"),
        (
            f"Close crossed the previous completed-bar Donchian {direction.lower()} level",
            f"Relative volume {ind.relative_volume:.2f} confirms participation",
        ),
        {
            "donchian_level": level,
            "relative_volume": ind.relative_volume,
            "adx": ind.adx,
            "extension_atr": extension,
        },
    )


def _time_series_momentum(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "time_series_momentum"
    if ind.roc is None or ind.atr_normalized_momentum is None or ind.adx is None or not ind.ema:
        return None
    if 20 not in ind.ema or 50 not in ind.ema:
        return None
    assert ind.roc is not None
    assert ind.atr_normalized_momentum is not None
    assert ind.adx is not None
    p = config.parameters[name]
    bullish = (
        ind.roc >= p["min_roc"]
        and ind.atr_normalized_momentum >= p["min_atr_normalized_momentum"]
        and ind.ema[20] > ind.ema[50]
        and ind.close > ind.ema[20]
        and ind.adx >= p["min_adx"]
    )
    bearish = (
        ind.roc <= -p["min_roc"]
        and ind.atr_normalized_momentum <= -p["min_atr_normalized_momentum"]
        and ind.ema[20] < ind.ema[50]
        and ind.close < ind.ema[20]
        and ind.adx >= p["min_adx"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    return _decision(
        name,
        direction,
        config,
        {
            "momentum_quality": _clip(
                abs(ind.atr_normalized_momentum) / (2 * p["min_atr_normalized_momentum"])
            ),
            "trend_quality": _quality_above(ind.adx, p["min_adx"]),
            "entry_quality": _clip(abs(ind.roc) / (2 * p["min_roc"])),
        },
        ("TIME_SERIES_MOMENTUM", "EMA_TREND_CONFIRMED"),
        (
            f"Medium-term return and ATR-normalized momentum confirm {direction.lower()}",
            "EMA20/50 structure agrees with momentum",
        ),
        {"roc": ind.roc, "atr_normalized_momentum": ind.atr_normalized_momentum, "adx": ind.adx},
    )


def _trend_pullback(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "trend_pullback"
    if (
        ind.adx is None
        or ind.rsi is None
        or ind.atr in (None, 0.0)
        or ind.previous_close is None
        or ind.previous_low is None
        or ind.previous_high is None
        or not ind.ema
        or not ind.ema_previous
        or not ind.ema_slope_atr
        or 20 not in ind.ema
        or 50 not in ind.ema
        or 20 not in ind.ema_previous
        or 20 not in ind.ema_slope_atr
    ):
        return None
    assert ind.adx is not None
    assert ind.rsi is not None
    assert ind.atr is not None
    assert ind.previous_close is not None
    assert ind.previous_low is not None
    assert ind.previous_high is not None
    p = config.parameters[name]
    ema20, ema50, previous_ema20 = ind.ema[20], ind.ema[50], ind.ema_previous[20]
    slope = ind.ema_slope_atr[20]
    previous_buy_distance = abs(ind.previous_low - previous_ema20) / ind.atr
    previous_sell_distance = abs(ind.previous_high - previous_ema20) / ind.atr
    rvol = ind.relative_volume if ind.relative_volume is not None else 1.0
    bullish = (
        ema20 > ema50
        and slope > 0
        and ind.adx >= p["adx_min"]
        and previous_buy_distance <= p["pullback_distance_atr"]
        and ind.close > ema20
        and ind.close > ind.open
        and ind.close > ind.previous_close
        and (ind.candle_body_pct or 0.0) >= p["recovery_body_min"]
        and p["rsi_buy_min"] <= ind.rsi <= p["rsi_buy_max"]
        and rvol >= p["min_relative_volume"]
    )
    bearish = (
        ema20 < ema50
        and slope < 0
        and ind.adx >= p["adx_min"]
        and previous_sell_distance <= p["pullback_distance_atr"]
        and ind.close < ema20
        and ind.close < ind.open
        and ind.close < ind.previous_close
        and (ind.candle_body_pct or 0.0) >= p["recovery_body_min"]
        and p["rsi_sell_min"] <= ind.rsi <= p["rsi_sell_max"]
        and rvol >= p["min_relative_volume"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    pullback_distance = previous_buy_distance if bullish else previous_sell_distance
    return _decision(
        name,
        direction,
        config,
        {
            "trend_quality": _quality_above(ind.adx, p["adx_min"]),
            "pullback_quality": _clip(1.0 - pullback_distance / p["pullback_distance_atr"]),
            "recovery_quality": _clip(ind.candle_body_pct or 0.0),
            "volume_confirmation": _quality_above(rvol, p["min_relative_volume"]),
        },
        ("EMA_TREND_CONFIRMED", "TREND_PULLBACK", "COMPLETED_CANDLE_RECOVERY"),
        (
            "Previous completed candle retraced toward EMA20",
            f"Current completed candle confirms {direction.lower()} resumption",
        ),
        {
            "pullback_distance_atr": pullback_distance,
            "ema20_slope_atr": slope,
            "rsi": ind.rsi,
            "adx": ind.adx,
            "relative_volume": rvol,
        },
    )


def _supertrend_adx_ema(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "supertrend_adx_ema"
    if (
        ind.supertrend_bullish is None
        or ind.adx is None
        or ind.plus_di is None
        or ind.minus_di is None
        or ind.atr in (None, 0.0)
        or not ind.ema
        or 50 not in ind.ema
    ):
        return None
    assert ind.supertrend_bullish is not None
    assert ind.adx is not None
    assert ind.plus_di is not None
    assert ind.minus_di is not None
    assert ind.atr is not None
    p = config.parameters[name]
    extension = abs(ind.close - ind.ema[50]) / ind.atr
    bullish = (
        ind.supertrend_bullish
        and ind.close > ind.ema[50]
        and ind.adx >= p["adx_min"]
        and ind.plus_di > ind.minus_di
        and extension <= p["max_extension_atr"]
    )
    bearish = (
        not ind.supertrend_bullish
        and ind.close < ind.ema[50]
        and ind.adx >= p["adx_min"]
        and ind.minus_di > ind.plus_di
        and extension <= p["max_extension_atr"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    return _decision(
        name,
        direction,
        config,
        {
            "supertrend_quality": 1.0,
            "trend_quality": _quality_above(ind.adx, p["adx_min"]),
            "directional_quality": _clip(abs(ind.plus_di - ind.minus_di) / max(ind.adx, 1.0)),
            "entry_quality": _clip(1.0 - extension / p["max_extension_atr"]),
        },
        ("SUPERTREND_CONFIRMED", "EMA_TREND_CONFIRMED", "ADX_TREND_CONFIRMED"),
        (
            f"SuperTrend state confirms {direction.lower()}",
            "EMA50 and directional index independently agree",
        ),
        {
            "supertrend": ind.supertrend or 0.0,
            "ema50": ind.ema[50],
            "adx": ind.adx,
            "extension_atr": extension,
        },
    )


def _macd_trend_continuation(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "macd_trend_continuation"
    values = (
        ind.macd,
        ind.macd_signal,
        ind.macd_histogram,
        ind.macd_previous,
        ind.macd_signal_previous,
        ind.macd_histogram_previous,
        ind.adx,
        ind.rsi,
    )
    if any(value is None for value in values) or not ind.ema or 50 not in ind.ema:
        return None
    assert ind.macd is not None
    assert ind.macd_signal is not None
    assert ind.macd_histogram is not None
    assert ind.macd_previous is not None
    assert ind.macd_signal_previous is not None
    assert ind.macd_histogram_previous is not None
    assert ind.adx is not None
    assert ind.rsi is not None
    p = config.parameters[name]
    bullish_momentum = ind.macd > ind.macd_signal and (
        ind.macd_previous <= ind.macd_signal_previous
        or ind.macd_histogram > ind.macd_histogram_previous
    )
    bearish_momentum = ind.macd < ind.macd_signal and (
        ind.macd_previous >= ind.macd_signal_previous
        or ind.macd_histogram < ind.macd_histogram_previous
    )
    bullish = (
        ind.close > ind.ema[50]
        and bullish_momentum
        and ind.adx >= p["adx_min"]
        and ind.rsi <= p["rsi_buy_max"]
    )
    bearish = (
        ind.close < ind.ema[50]
        and bearish_momentum
        and ind.adx >= p["adx_min"]
        and ind.rsi >= p["rsi_sell_min"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    acceleration = abs(ind.macd_histogram - ind.macd_histogram_previous)
    scale = max(abs(ind.macd_histogram_previous), 1e-9)
    return _decision(
        name,
        direction,
        config,
        {
            "macd_quality": _clip(0.5 + 0.5 * acceleration / scale),
            "trend_quality": _quality_above(ind.adx, p["adx_min"]),
            "entry_quality": _clip(1.0 - abs(ind.rsi - 50.0) / 50.0),
        },
        ("MACD_ACCELERATION", "EMA_TREND_CONFIRMED"),
        (
            f"MACD cross/acceleration confirms {direction.lower()} continuation",
            "EMA50 trend structure agrees; RSI is not at the configured stretch limit",
        ),
        {
            "macd": ind.macd,
            "macd_signal": ind.macd_signal,
            "macd_histogram": ind.macd_histogram,
            "previous_histogram": ind.macd_histogram_previous,
            "adx": ind.adx,
            "rsi": ind.rsi,
        },
    )


def _bollinger_rsi_mean_reversion(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "bollinger_rsi_mean_reversion"
    values = (
        ind.bb_lower,
        ind.bb_upper,
        ind.bb_previous_lower,
        ind.bb_previous_upper,
        ind.previous_close,
        ind.rsi,
        ind.adx,
        ind.bb_bandwidth,
    )
    if any(value is None for value in values):
        return None
    previous_close = ind.previous_close
    previous_lower = ind.bb_previous_lower
    previous_upper = ind.bb_previous_upper
    assert previous_close is not None
    assert previous_lower is not None
    assert previous_upper is not None
    assert ind.bb_lower is not None
    assert ind.bb_upper is not None
    assert ind.rsi is not None
    assert ind.adx is not None
    assert ind.bb_bandwidth is not None
    p = config.parameters[name]
    ranging = ind.adx <= p["adx_max"] and ind.bb_bandwidth >= p["min_bandwidth"]
    bullish = (
        ranging
        and previous_close <= previous_lower
        and ind.close > ind.bb_lower
        and ind.close > ind.open
        and ind.rsi <= p["rsi_buy_max"]
    )
    bearish = (
        ranging
        and previous_close >= previous_upper
        and ind.close < ind.bb_upper
        and ind.close < ind.open
        and ind.rsi >= p["rsi_sell_min"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    oscillator = (
        (p["rsi_buy_max"] - ind.rsi) / p["rsi_buy_max"]
        if bullish
        else (ind.rsi - p["rsi_sell_min"]) / (100.0 - p["rsi_sell_min"])
    )
    return _decision(
        name,
        direction,
        config,
        {
            "reentry_quality": 1.0,
            "oscillator_quality": _clip(0.5 + oscillator),
            "regime_quality": _clip(1.0 - ind.adx / p["adx_max"]),
            "entry_quality": _clip(ind.candle_body_pct or 0.0),
        },
        ("BOLLINGER_REENTRY", "RSI_REVERSAL_CONTEXT", "RANGING_REGIME_CONFIRMED"),
        (
            "Previous candle reached/exceeded the outer band",
            f"Current completed candle re-entered the band toward {direction.lower()} mean reversion",
        ),
        {
            "rsi": ind.rsi,
            "adx": ind.adx,
            "bb_percent": ind.bb_percent or 0.0,
            "bb_bandwidth": ind.bb_bandwidth,
        },
    )


def _vwap_mean_reversion(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "vwap_mean_reversion"
    if (
        ind.session_vwap is None
        or ind.vwap_distance_atr is None
        or ind.vwap_zscore is None
        or ind.rsi is None
        or ind.adx is None
        or ind.previous_close is None
    ):
        return None
    assert ind.session_vwap is not None
    assert ind.vwap_distance_atr is not None
    assert ind.vwap_zscore is not None
    assert ind.rsi is not None
    assert ind.adx is not None
    assert ind.previous_close is not None
    p = config.parameters[name]
    non_trending = ind.adx <= p["adx_max"]
    bullish = (
        non_trending
        and ind.vwap_distance_atr <= -p["min_deviation_atr"]
        and ind.vwap_zscore <= -p["min_abs_zscore"]
        and ind.rsi <= p["rsi_buy_max"]
        and ind.close > ind.open
        and ind.close > ind.previous_close
    )
    bearish = (
        non_trending
        and ind.vwap_distance_atr >= p["min_deviation_atr"]
        and ind.vwap_zscore >= p["min_abs_zscore"]
        and ind.rsi >= p["rsi_sell_min"]
        and ind.close < ind.open
        and ind.close < ind.previous_close
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    rvol = ind.relative_volume if ind.relative_volume is not None else 1.0
    return _decision(
        name,
        direction,
        config,
        {
            "deviation_quality": _quality_above(abs(ind.vwap_distance_atr), p["min_deviation_atr"]),
            "reversal_quality": _clip(ind.candle_body_pct or 0.0),
            "regime_quality": _clip(1.0 - ind.adx / p["adx_max"]),
            "volume_confirmation": _clip(rvol / 2.0),
        },
        ("VWAP_DEVIATION", "COMPLETED_CANDLE_RECOVERY"),
        (
            f"Price is {abs(ind.vwap_distance_atr):.2f} ATR from session VWAP",
            f"Completed candle confirms pressure reversal toward {direction.lower()} mean reversion",
        ),
        {
            "session_vwap": ind.session_vwap,
            "vwap_distance_atr": ind.vwap_distance_atr,
            "vwap_zscore": ind.vwap_zscore,
            "rsi": ind.rsi,
            "adx": ind.adx,
        },
    )


def _opening_range_breakout(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    name = "opening_range_breakout"
    if (
        not ind.opening_range_complete
        or ind.opening_range_high is None
        or ind.opening_range_low is None
        or ind.previous_close is None
        or ind.session_vwap is None
        or ind.relative_volume is None
        or ind.atr in (None, 0.0)
        or ind.adx is None
    ):
        return None
    assert ind.opening_range_high is not None
    assert ind.opening_range_low is not None
    assert ind.previous_close is not None
    assert ind.session_vwap is not None
    assert ind.relative_volume is not None
    assert ind.atr is not None
    assert ind.adx is not None
    p = config.parameters[name]
    up_extension = (ind.close - ind.opening_range_high) / ind.atr
    down_extension = (ind.opening_range_low - ind.close) / ind.atr
    bullish = (
        ind.close > ind.opening_range_high
        and ind.previous_close <= ind.opening_range_high
        and ind.close > ind.session_vwap
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= up_extension <= p["max_breakout_extension_atr"]
    )
    bearish = (
        ind.close < ind.opening_range_low
        and ind.previous_close >= ind.opening_range_low
        and ind.close < ind.session_vwap
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= down_extension <= p["max_breakout_extension_atr"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    extension = up_extension if bullish else down_extension
    return _decision(
        name,
        direction,
        config,
        {
            "breakout_quality": _clip(1.0 - extension / p["max_breakout_extension_atr"]),
            "volume_confirmation": _quality_above(ind.relative_volume, p["min_relative_volume"]),
            "vwap_confirmation": 1.0,
            "entry_quality": _clip(ind.candle_body_pct or 0.0),
        },
        ("OPENING_RANGE_BREAKOUT", "RELATIVE_VOLUME_CONFIRMED", "VWAP_CONFIRMATION"),
        (
            f"Completed candle closed outside the fully formed NSE opening range ({direction})",
            "Relative volume and session VWAP confirm breakout quality",
        ),
        {
            "opening_range_high": ind.opening_range_high,
            "opening_range_low": ind.opening_range_low,
            "opening_range_end": ind.opening_range_end or "",
            "relative_volume": ind.relative_volume,
            "session_vwap": ind.session_vwap,
            "extension_atr": extension,
        },
    )


def _volume_profile_breakout(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    """Break of the prior Value Area (volume-at-price), confirmed by relative volume.

    Unlike every other evaluator in this module, the entry level here is derived from
    where volume actually traded, not from price alone -- see IndicatorConfig's
    volume_profile_* fields and engine.py's ``_latest_volume_profile``.
    """
    name = "volume_profile_breakout"
    if (
        ind.volume_value_area_high is None
        or ind.volume_value_area_low is None
        or ind.volume_poc is None
        or ind.previous_close is None
        or ind.atr in (None, 0.0)
        or ind.relative_volume is None
        or ind.adx is None
    ):
        return None
    assert ind.volume_value_area_high is not None
    assert ind.volume_value_area_low is not None
    assert ind.previous_close is not None
    assert ind.atr is not None
    assert ind.relative_volume is not None
    assert ind.adx is not None
    p = config.parameters[name]
    up_extension = (ind.close - ind.volume_value_area_high) / ind.atr
    down_extension = (ind.volume_value_area_low - ind.close) / ind.atr
    bullish = (
        ind.close > ind.volume_value_area_high
        and ind.previous_close <= ind.volume_value_area_high
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= up_extension <= p["max_breakout_extension_atr"]
    )
    bearish = (
        ind.close < ind.volume_value_area_low
        and ind.previous_close >= ind.volume_value_area_low
        and ind.relative_volume >= p["min_relative_volume"]
        and ind.adx >= p["min_adx"]
        and 0 <= down_extension <= p["max_breakout_extension_atr"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    extension = up_extension if bullish else down_extension
    level = ind.volume_value_area_high if bullish else ind.volume_value_area_low
    return _decision(
        name,
        direction,
        config,
        {
            "breakout_quality": _clip(1.0 - extension / p["max_breakout_extension_atr"]),
            "volume_confirmation": _quality_above(ind.relative_volume, p["min_relative_volume"]),
            "trend_quality": _quality_above(ind.adx, p["min_adx"]),
            "entry_quality": _clip(ind.candle_body_pct or 0.0),
        },
        ("VOLUME_PROFILE_BREAKOUT", "VALUE_AREA_EXIT_VOLUME_CONFIRMED"),
        (
            f"Close broke {'above' if bullish else 'below'} the prior Value Area "
            f"(POC {ind.volume_poc:.2f})",
            f"Relative volume {ind.relative_volume:.2f} confirms participation",
        ),
        {
            "value_area_level": level,
            "volume_poc": ind.volume_poc,
            "relative_volume": ind.relative_volume,
            "adx": ind.adx,
            "extension_atr": extension,
        },
    )


def _ema_trend_confluence(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    """EMA20/50 trend + RSI/MACD + volume + candle-quality confluence, entering on a
    breakout, pullback, or reversal setup.

    Adapted from a user-supplied TradingView Pine Script swing-trading indicator
    ("DeltaTrading"). Two things from the source script are deliberately NOT ported
    here: its Nifty/sector daily trend filter and covariance-based beta-vs-benchmark
    gate (both need cross-symbol/multi-timeframe data this single-symbol evaluator
    doesn't have), and its position-sizing/add-on/trailing-stop/T1-T2-target state
    machine (PositionSizer and ExitManager already own all of that, live-tested this
    session -- reimplementing it here would just create a second, conflicting copy).
    Long-only, matching the source script's own bias (it has no short-entry logic).
    """
    name = "ema_trend_confluence"
    required = (
        ind.previous_close,
        ind.rsi,
        ind.macd,
        ind.macd_signal,
        ind.macd_histogram,
        ind.macd_histogram_previous,
        ind.relative_volume,
        ind.candle_body_pct,
        ind.upper_wick_pct,
        ind.donchian_previous_high,
        ind.donchian_previous_low,
    )
    if any(value is None for value in required) or ind.atr in (None, 0.0):
        return None
    if not ind.ema or 20 not in ind.ema or 50 not in ind.ema or 20 not in ind.ema_previous:
        return None
    assert ind.previous_close is not None
    assert ind.rsi is not None
    assert ind.macd is not None
    assert ind.macd_signal is not None
    assert ind.macd_histogram is not None
    assert ind.macd_histogram_previous is not None
    assert ind.relative_volume is not None
    assert ind.candle_body_pct is not None
    assert ind.upper_wick_pct is not None
    assert ind.donchian_previous_high is not None
    assert ind.donchian_previous_low is not None
    assert ind.atr is not None

    p = config.parameters[name]
    ema20, ema50 = ind.ema[20], ind.ema[50]
    ema20_prev = ind.ema_previous[20]

    stock_trend_ok = ind.close > ema50 and ema20 >= ema50
    stock_recovery_ok = ind.close > ema20 and ema20 > ema20_prev
    rsi_ok = 50.0 < ind.rsi < 75.0
    macd_ok = ind.macd > ind.macd_signal and ind.macd_histogram > ind.macd_histogram_previous

    candle_range = ind.high - ind.low
    close_near_high = ((ind.high - ind.close) / candle_range * 100 <= 30) if candle_range > 0 else False
    strong_green_candle = ind.close > ind.open and close_near_high and ind.candle_body_pct >= 0.35
    big_upper_wick = ind.upper_wick_pct >= 0.40 and ind.upper_wick_pct > ind.candle_body_pct

    one_bar_move_pct = (
        (ind.close - ind.previous_close) / ind.previous_close * 100 if ind.previous_close else 0.0
    )
    not_huge_rally = one_bar_move_pct <= p["max_one_bar_move_pct"]

    breakout_entry = (
        ind.close > ind.donchian_previous_high
        and ind.previous_close <= ind.donchian_previous_high
        and ind.relative_volume > p["volume_multiplier"]
        and strong_green_candle
    )
    pullback_entry = (
        stock_trend_ok
        and ind.low <= ema20 * (1 + p["near_ema_pct"] / 100)
        and ind.close > ema20
        and ind.close > ind.open
        and ind.rsi > 45.0
        and ind.macd > ind.macd_signal
    )
    reversal_entry = (
        ind.close > ema20
        and ind.previous_close < ema20_prev
        and ind.rsi > 50.0
        and ind.macd_histogram > ind.macd_histogram_previous
        and ind.close > ind.open
    )
    entry_setup = breakout_entry or pullback_entry or reversal_entry

    planned_sl = max(ind.donchian_previous_low, ind.close - ind.atr * p["atr_sl_multiplier"])
    planned_risk = ind.close - planned_sl
    planned_risk_pct = (planned_risk / ind.close * 100) if ind.close else 0.0
    valid_sl = planned_risk > 0 and planned_risk_pct <= p["max_stop_loss_pct"]

    # Volume/candle-strength requirements live inside breakout_entry's own definition
    # above (matching the source script, where breakoutEntry alone requires
    # volumeHigh/strongGreenCandle); rsi/macd/wick/rally/stop are the global quality
    # gates the source applies across all three entry types via its score components.
    bullish = (
        entry_setup
        and rsi_ok
        and macd_ok
        and not big_upper_wick
        and not_huge_rally
        and valid_sl
    )
    if not bullish:
        return None

    setup_type = "breakout" if breakout_entry else "pullback" if pullback_entry else "reversal"
    ema_gap_pct = (ema20 - ema50) / ema50 * 100 if ema50 else 0.0
    trend_score = _quality_above(ema_gap_pct, 0.5)
    if stock_trend_ok and stock_recovery_ok:
        trend_score = _clip(trend_score + 0.2)
    return _decision(
        name,
        "BUY",
        config,
        {
            "trend_quality": trend_score,
            "momentum_quality": _quality_above(ind.rsi, 50.0),
            "volume_confirmation": _quality_above(ind.relative_volume, p["volume_multiplier"]),
            "entry_quality": _clip(ind.candle_body_pct),
        },
        ("EMA_TREND_CONFLUENCE", setup_type.upper() + "_SETUP"),
        (
            f"EMA20/50 trend confluence with RSI {ind.rsi:.1f} and confirming MACD",
            f"{setup_type.capitalize()} setup with valid stop ({planned_risk_pct:.2f}% risk)",
        ),
        {
            "setup_type": setup_type,
            "planned_stop": planned_sl,
            "planned_risk_pct": planned_risk_pct,
            "relative_volume": ind.relative_volume,
            "rsi": ind.rsi,
        },
    )


def _pivot_volume_sr_zones(
    ind: IndicatorResult, config: ProductionStrategyConfig
) -> TechnicalStrategyDecision | None:
    """High-volume-confirmed pivot support/resistance zone reactions.

    Adapted from a user-supplied TradingView Pine Script ("High Volume Pivot Support &
    Resistance Zones [BigBeluga]"). Six distinct reactions map to a direction:
      BUY:  resistance breakout, a hold/bounce off support, or a hold on an already-
            broken resistance zone retested from below (now acting as support).
      SELL: support breakdown, a rejection at resistance, or a rejection on an already-
            broken support zone retested from above (now acting as resistance).
    The source script has no alertcondition() or strategy logic at all -- it only plots
    shapes; this evaluator is this session's own translation of those six plotted
    events into a direction, not a conversion of an existing rule. Zone box drawing and
    the CVD-delta visualization are display-only and not ported.
    """
    name = "pivot_volume_sr_zones"
    if ind.previous_high is None or ind.previous_low is None:
        return None
    assert ind.previous_high is not None
    assert ind.previous_low is not None

    direction = ""
    reason_code = ""
    reason_text = ""
    zone_level = 0.0

    if ind.pivot_res_zone_top is not None and ind.pivot_res_zone_bottom is not None:
        r_top, r_bottom = ind.pivot_res_zone_top, ind.pivot_res_zone_bottom
        if ind.pivot_res_zone_broken:
            if ind.previous_low <= r_top and ind.low > r_top:
                direction = "BUY"
                reason_code = "FLIPPED_RESISTANCE_RETEST_HOLD"
                reason_text = f"Retested broken resistance (now support) at {r_top:.2f} and held"
                zone_level = r_top
        else:
            if ind.close > r_top:
                direction = "BUY"
                reason_code = "RESISTANCE_BREAKOUT"
                reason_text = f"Close broke above high-volume resistance zone at {r_top:.2f}"
                zone_level = r_top
            elif ind.previous_high >= r_bottom and ind.high < r_bottom:
                direction = "SELL"
                reason_code = "RESISTANCE_HOLD_REJECTION"
                reason_text = f"Rejected from resistance zone {r_bottom:.2f}-{r_top:.2f}"
                zone_level = r_bottom

    if not direction and ind.pivot_sup_zone_top is not None and ind.pivot_sup_zone_bottom is not None:
        s_top, s_bottom = ind.pivot_sup_zone_top, ind.pivot_sup_zone_bottom
        if ind.pivot_sup_zone_broken:
            if ind.previous_high >= s_bottom and ind.high < s_bottom:
                direction = "SELL"
                reason_code = "FLIPPED_SUPPORT_RETEST_REJECTION"
                reason_text = f"Retested broken support (now resistance) at {s_bottom:.2f}, rejected"
                zone_level = s_bottom
        else:
            if ind.close < s_bottom:
                direction = "SELL"
                reason_code = "SUPPORT_BREAKDOWN"
                reason_text = f"Close broke below high-volume support zone at {s_bottom:.2f}"
                zone_level = s_bottom
            elif ind.previous_low <= s_top and ind.low > s_top:
                direction = "BUY"
                reason_code = "SUPPORT_HOLD_BOUNCE"
                reason_text = f"Bounced off support zone {s_bottom:.2f}-{s_top:.2f}"
                zone_level = s_top

    if not direction:
        return None

    rvol = ind.relative_volume if ind.relative_volume is not None else 1.0
    adx = ind.adx if ind.adx is not None else 20.0
    body_pct = ind.candle_body_pct if ind.candle_body_pct is not None else 0.5
    is_breakout_type = reason_code in ("RESISTANCE_BREAKOUT", "SUPPORT_BREAKDOWN")
    return _decision(
        name,
        direction,
        config,
        {
            "zone_quality": 0.7,  # zone formation itself already required a volume-validated pivot
            "trend_quality": _quality_above(adx, 20.0),
            "volume_confirmation": _quality_above(rvol, 1.2 if is_breakout_type else 1.0),
            "entry_quality": _clip(body_pct),
        },
        ("PIVOT_VOLUME_SR_ZONE", reason_code),
        (
            reason_text,
            f"Zone level {zone_level:.2f}, relative volume {rvol:.2f}",
        ),
        {
            "reaction_type": reason_code,
            "zone_level": zone_level,
            "relative_volume": rvol,
        },
    )


def _relative_strength_momentum(
    ind: IndicatorResult,
    config: ProductionStrategyConfig,
    market_relative: MarketRelativeFeatures | None,
) -> TechnicalStrategyDecision | None:
    name = "relative_strength_momentum"
    if (
        market_relative is None
        or market_relative.timestamp != ind.settled_candle_timestamp
        or ind.adx is None
        or not ind.ema
        or 20 not in ind.ema
        or 50 not in ind.ema
    ):
        return None
    assert ind.adx is not None
    p = config.parameters[name]
    rvol = ind.relative_volume if ind.relative_volume is not None else 1.0
    bullish = (
        market_relative.excess_return >= p["min_excess_return"]
        and market_relative.cross_sectional_percentile >= p["buy_percentile"]
        and ind.ema[20] > ind.ema[50]
        and ind.close > ind.ema[20]
        and ind.adx >= p["min_adx"]
        and rvol >= p["min_relative_volume"]
    )
    bearish = (
        market_relative.excess_return <= -p["min_excess_return"]
        and market_relative.cross_sectional_percentile <= p["sell_percentile"]
        and ind.ema[20] < ind.ema[50]
        and ind.close < ind.ema[20]
        and ind.adx >= p["min_adx"]
        and rvol >= p["min_relative_volume"]
    )
    if not bullish and not bearish:
        return None
    direction = "BUY" if bullish else "SELL"
    percentile_quality = (
        market_relative.cross_sectional_percentile
        if bullish
        else 1.0 - market_relative.cross_sectional_percentile
    )
    return _decision(
        name,
        direction,
        config,
        {
            "relative_strength": _clip(percentile_quality),
            "trend_quality": _quality_above(ind.adx, p["min_adx"]),
            "volume_confirmation": _quality_above(rvol, p["min_relative_volume"]),
            "entry_quality": _quality_above(
                abs(market_relative.excess_return), p["min_excess_return"]
            ),
        },
        (
            "MARKET_RELATIVE_STRENGTH",
            "RELATIVE_STRENGTH_TOP_DECILE" if bullish else "RELATIVE_STRENGTH_BOTTOM_DECILE",
            "EMA_TREND_CONFIRMED",
        ),
        (
            f"Symbol materially {'outperforms' if bullish else 'underperforms'} the aligned benchmark context",
            f"Cross-sectional percentile {market_relative.cross_sectional_percentile:.3f} and EMA trend agree",
        ),
        {**market_relative.to_dict(), "adx": ind.adx, "relative_volume": rvol},
    )


def evaluate_production_strategy(
    strategy_name: str,
    indicators: IndicatorResult,
    config: ProductionStrategyConfig,
    market_relative: MarketRelativeFeatures | None = None,
) -> TechnicalStrategyDecision | None:
    """Dispatch one configured evaluator without recalculating any indicator."""

    evaluators = {
        "ema_adx_trend": _ema_adx_trend,
        "donchian_breakout": _donchian_breakout,
        "time_series_momentum": _time_series_momentum,
        "trend_pullback": _trend_pullback,
        "supertrend_adx_ema": _supertrend_adx_ema,
        "macd_trend_continuation": _macd_trend_continuation,
        "bollinger_rsi_mean_reversion": _bollinger_rsi_mean_reversion,
        "vwap_mean_reversion": _vwap_mean_reversion,
        "opening_range_breakout": _opening_range_breakout,
        "volume_profile_breakout": _volume_profile_breakout,
        "ema_trend_confluence": _ema_trend_confluence,
        "pivot_volume_sr_zones": _pivot_volume_sr_zones,
    }
    if strategy_name == "relative_strength_momentum":
        return _relative_strength_momentum(indicators, config, market_relative)
    evaluator = evaluators.get(strategy_name)
    return evaluator(indicators, config) if evaluator is not None else None
