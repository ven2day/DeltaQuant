"""Configuration for the production technical-strategy family.

The defaults are conservative starting parameters, not validation claims. Deployments
may override the timeframe matrix, thresholds, and score weights with JSON settings;
the effective configuration is versioned and included in strategy artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PRODUCTION_STRATEGY_NAMES: tuple[str, ...] = (
    "ema_adx_trend",
    "donchian_breakout",
    "time_series_momentum",
    "trend_pullback",
    "supertrend_adx_ema",
    "macd_trend_continuation",
    "bollinger_rsi_mean_reversion",
    "vwap_mean_reversion",
    "opening_range_breakout",
    "relative_strength_momentum",
    "volume_profile_breakout",
    "ema_trend_confluence",
    "pivot_volume_sr_zones",
)

# Deprecated compatibility name.  The authoritative policy is owned by
# ``src.markets.forex.strategies``; core never applies this market policy.
FOREX_INITIAL_DISABLED: tuple[str, ...] = (
    "vwap_mean_reversion",
    "opening_range_breakout",
    "relative_strength_momentum",
)

DEFAULT_TIMEFRAME_MAP: dict[str, tuple[str, ...]] = {
    "ema_adx_trend": ("15m", "30m", "1h", "4h"),
    "donchian_breakout": ("15m", "30m", "1h", "4h"),
    "time_series_momentum": ("1h", "4h"),
    "trend_pullback": ("15m", "30m", "1h", "4h"),
    "supertrend_adx_ema": ("15m", "30m", "1h", "4h"),
    "macd_trend_continuation": ("30m", "1h", "4h"),
    "bollinger_rsi_mean_reversion": ("15m", "30m", "1h"),
    "vwap_mean_reversion": ("15m", "30m"),
    "opening_range_breakout": ("15m", "30m"),
    "relative_strength_momentum": ("1h", "4h"),
    # 60-bar lookback needs a reasonably sized window; skip 5m (too noisy/short-horizon)
    # and 1d (90-day validation period leaves too few daily bars to test meaningfully).
    "volume_profile_breakout": ("15m", "30m", "1h", "4h"),
    # Source script targets 1-week-to-1-month swing trades (see its Beta comment); 5m/
    # 15m added anyway on request to get a real verdict instead of an assumption --
    # the walk-forward gate is the actual judge, not this comment.
    "ema_trend_confluence": ("5m", "15m", "30m", "1h", "4h", "1d"),
    # Pivot confirmation needs 40 bars clear on both sides (source script default) plus
    # a 200-bar ATR window -- needs a genuinely long history per bar, so no 5m/15m here.
    "pivot_volume_sr_zones": ("1h", "4h", "1d"),
}

# Every numeric gate used by a production evaluator is visible here. These are starting
# configurations and remain UNVALIDATED/SHADOW until offline walk-forward validation.
DEFAULT_PARAMETERS: dict[str, dict[str, float]] = {
    "ema_adx_trend": {
        "adx_min": 22.0,
        "max_extension_atr": 2.0,
        "min_ema_slope_atr": 0.015,
        "min_relative_volume": 0.8,
    },
    "donchian_breakout": {
        "min_relative_volume": 1.2,
        "min_adx": 15.0,
        "max_breakout_extension_atr": 1.0,
    },
    "time_series_momentum": {
        "min_adx": 18.0,
        "min_roc": 0.01,
        "min_atr_normalized_momentum": 1.0,
    },
    "trend_pullback": {
        "adx_min": 22.0,
        "pullback_distance_atr": 0.65,
        "recovery_body_min": 0.25,
        "rsi_buy_min": 42.0,
        "rsi_buy_max": 65.0,
        "rsi_sell_min": 35.0,
        "rsi_sell_max": 58.0,
        "min_relative_volume": 0.7,
    },
    "supertrend_adx_ema": {
        "adx_min": 22.0,
        "max_extension_atr": 2.25,
    },
    "macd_trend_continuation": {
        "adx_min": 20.0,
        "rsi_buy_max": 72.0,
        "rsi_sell_min": 28.0,
        "require_cross_or_acceleration": 1.0,
    },
    "bollinger_rsi_mean_reversion": {
        "adx_max": 23.0,
        "rsi_buy_max": 38.0,
        "rsi_sell_min": 62.0,
        "min_bandwidth": 0.005,
    },
    "vwap_mean_reversion": {
        "adx_max": 24.0,
        "min_deviation_atr": 0.8,
        "min_abs_zscore": 1.0,
        "rsi_buy_max": 45.0,
        "rsi_sell_min": 55.0,
    },
    "opening_range_breakout": {
        "min_relative_volume": 1.2,
        "min_adx": 15.0,
        "max_breakout_extension_atr": 1.0,
    },
    "relative_strength_momentum": {
        "min_adx": 18.0,
        "buy_percentile": 0.80,
        "sell_percentile": 0.20,
        "min_excess_return": 0.003,
        "min_relative_volume": 0.8,
    },
    "volume_profile_breakout": {
        "min_relative_volume": 1.3,
        "min_adx": 15.0,
        "max_breakout_extension_atr": 2.0,
    },
    "ema_trend_confluence": {
        "volume_multiplier": 1.15,
        "near_ema_pct": 2.0,
        "max_one_bar_move_pct": 5.0,
        "atr_sl_multiplier": 1.5,
        "max_stop_loss_pct": 7.0,
    },
    # No strategy-level parameters -- pivot length/volume threshold/ATR window live on
    # IndicatorConfig (shared, computed once) since they define the zone itself, not a
    # decision threshold this evaluator applies on top of it.
    "pivot_volume_sr_zones": {},
}

# Components are strategy-specific evidence, each normalized to [0, 1]. The result is
# a technical score, not a calibrated probability.
DEFAULT_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "ema_adx_trend": {
        "trend_quality": 0.40,
        "directional_quality": 0.25,
        "entry_quality": 0.25,
        "volume_confirmation": 0.10,
    },
    "donchian_breakout": {
        "breakout_quality": 0.35,
        "volume_confirmation": 0.35,
        "trend_quality": 0.15,
        "entry_quality": 0.15,
    },
    "time_series_momentum": {
        "momentum_quality": 0.45,
        "trend_quality": 0.35,
        "entry_quality": 0.20,
    },
    "trend_pullback": {
        "trend_quality": 0.30,
        "pullback_quality": 0.30,
        "recovery_quality": 0.30,
        "volume_confirmation": 0.10,
    },
    "supertrend_adx_ema": {
        "supertrend_quality": 0.35,
        "trend_quality": 0.35,
        "directional_quality": 0.20,
        "entry_quality": 0.10,
    },
    "macd_trend_continuation": {"macd_quality": 0.40, "trend_quality": 0.35, "entry_quality": 0.25},
    "bollinger_rsi_mean_reversion": {
        "reentry_quality": 0.40,
        "oscillator_quality": 0.25,
        "regime_quality": 0.25,
        "entry_quality": 0.10,
    },
    "vwap_mean_reversion": {
        "deviation_quality": 0.35,
        "reversal_quality": 0.30,
        "regime_quality": 0.20,
        "volume_confirmation": 0.15,
    },
    "opening_range_breakout": {
        "breakout_quality": 0.35,
        "volume_confirmation": 0.30,
        "vwap_confirmation": 0.20,
        "entry_quality": 0.15,
    },
    "relative_strength_momentum": {
        "relative_strength": 0.40,
        "trend_quality": 0.30,
        "volume_confirmation": 0.15,
        "entry_quality": 0.15,
    },
    "volume_profile_breakout": {
        "breakout_quality": 0.35,
        "volume_confirmation": 0.35,
        "trend_quality": 0.15,
        "entry_quality": 0.15,
    },
    "ema_trend_confluence": {
        "trend_quality": 0.30,
        "momentum_quality": 0.30,
        "volume_confirmation": 0.25,
        "entry_quality": 0.15,
    },
    "pivot_volume_sr_zones": {
        "zone_quality": 0.35,
        "volume_confirmation": 0.30,
        "trend_quality": 0.20,
        "entry_quality": 0.15,
    },
}

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "ema_adx_trend": "EMA20/50 trend structure with ADX and directional-index confirmation",
    "donchian_breakout": "previous-bar Donchian breakout confirmed by relative volume",
    "time_series_momentum": "medium-horizon price momentum above volatility noise",
    "trend_pullback": "completed-candle recovery after an EMA20 pullback in an established trend",
    "supertrend_adx_ema": "SuperTrend state confirmed by EMA50, ADX, and directional index",
    "macd_trend_continuation": "MACD re-acceleration inside an EMA50-confirmed trend",
    "bollinger_rsi_mean_reversion": "Bollinger re-entry in a low-ADX range, with RSI confirmation",
    "vwap_mean_reversion": "session-VWAP deviation and completed-candle reversal in a non-trend",
    "opening_range_breakout": "NSE session opening-range close breakout with volume and VWAP",
    "relative_strength_momentum": "timestamp-aligned market-relative and cross-sectional momentum",
    "volume_profile_breakout": "break of the prior Value Area (volume-at-price), volume-confirmed",
    "ema_trend_confluence": "EMA20/50 + RSI/MACD + volume/candle-quality breakout, pullback, or reversal",
    "pivot_volume_sr_zones": "high-volume-confirmed pivot support/resistance zone breakout, breakdown, or hold",
}


def _json_object(raw: str, setting_name: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{setting_name} must be a JSON object")
    return parsed


@dataclass(frozen=True)
class ProductionStrategyConfig:
    """Validated effective matrix and parameters for strategy evaluation."""

    enabled: bool
    version: str
    timeframe_map: Mapping[str, tuple[str, ...]]
    parameters: Mapping[str, Mapping[str, float]]
    score_weights: Mapping[str, Mapping[str, float]]
    benchmark_symbol: str
    benchmark_fallback: str
    ml_policies: Mapping[str, str]
    market: str = "NSE"

    @classmethod
    def defaults(cls) -> ProductionStrategyConfig:
        return cls(
            enabled=True,
            version="production_rules_v1",
            timeframe_map=dict(DEFAULT_TIMEFRAME_MAP),
            parameters={name: dict(values) for name, values in DEFAULT_PARAMETERS.items()},
            score_weights={name: dict(values) for name, values in DEFAULT_SCORE_WEIGHTS.items()},
            benchmark_symbol="",
            benchmark_fallback="cross_sectional_median",
            ml_policies={},
            market="NSE",
        )

    @classmethod
    def from_settings(cls, settings: Any) -> ProductionStrategyConfig:
        timeframe_map = dict(DEFAULT_TIMEFRAME_MAP)
        for name, values in _json_object(
            settings.production_strategy_timeframes_json,
            "PRODUCTION_STRATEGY_TIMEFRAMES_JSON",
        ).items():
            if name not in PRODUCTION_STRATEGY_NAMES or not isinstance(values, list):
                raise ValueError(f"Invalid strategy timeframe mapping for {name!r}")
            timeframe_map[name] = tuple(str(value) for value in values)

        market = str(getattr(settings, "market", "NSE")).upper()

        parameters = {name: dict(values) for name, values in DEFAULT_PARAMETERS.items()}
        for name, values in _json_object(
            settings.production_strategy_parameters_json,
            "PRODUCTION_STRATEGY_PARAMETERS_JSON",
        ).items():
            if name not in PRODUCTION_STRATEGY_NAMES or not isinstance(values, dict):
                raise ValueError(f"Invalid strategy parameter mapping for {name!r}")
            parameters[name].update({str(key): float(value) for key, value in values.items()})

        score_weights = {name: dict(values) for name, values in DEFAULT_SCORE_WEIGHTS.items()}
        for name, values in _json_object(
            settings.production_strategy_score_weights_json,
            "PRODUCTION_STRATEGY_SCORE_WEIGHTS_JSON",
        ).items():
            if name not in PRODUCTION_STRATEGY_NAMES or not isinstance(values, dict):
                raise ValueError(f"Invalid strategy weight mapping for {name!r}")
            score_weights[name].update({str(key): float(value) for key, value in values.items()})
        for name, weights in score_weights.items():
            if not weights or any(value < 0 for value in weights.values()):
                raise ValueError(f"Score weights for {name} must be non-negative")
            if sum(weights.values()) <= 0:
                raise ValueError(f"Score weights for {name} must have positive total")

        return cls(
            enabled=bool(settings.production_strategies_enabled),
            version=str(settings.production_strategy_version),
            timeframe_map=timeframe_map,
            parameters=parameters,
            score_weights=score_weights,
            benchmark_symbol=str(settings.strategy_benchmark_symbol).upper().strip(),
            benchmark_fallback=str(settings.strategy_benchmark_fallback),
            ml_policies={},
            market=market,
        )

    def eligible(self, strategy_name: str, timeframe: str) -> bool:
        return self.enabled and timeframe in self.timeframe_map.get(strategy_name, ())

    def parameter(self, strategy_name: str, name: str) -> float:
        return float(self.parameters[strategy_name][name])

    def weights(self, strategy_name: str) -> Mapping[str, float]:
        return self.score_weights[strategy_name]

    def ml_policy(self, strategy_name: str, timeframe: str) -> str:
        key = f"{strategy_name.strip().lower()}:{timeframe.strip().lower()}"
        return str(self.ml_policies.get(key, "ML_OPTIONAL")).upper()
