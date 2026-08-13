"""Build the Forex strategy matrix from ``config/forex/strategies.yaml``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.core.strategies.config import ProductionStrategyConfig
from src.markets.forex.config import load_forex_config

FOREX_INITIAL_DISABLED: tuple[str, ...] = (
    "vwap_mean_reversion",
    "opening_range_breakout",
    "relative_strength_momentum",
)


def build_strategy_config(settings: Any) -> ProductionStrategyConfig:
    base = ProductionStrategyConfig.from_settings(settings)
    configured = load_forex_config().strategies.get("strategies", {})
    matrix = dict(base.timeframe_map)
    ml_policies = dict(base.ml_policies)
    for name, value in configured.items():
        if not isinstance(value, dict):
            raise ValueError(f"Forex strategy configuration for {name!r} must be a mapping")
        matrix[str(name)] = (
            tuple(str(item) for item in value.get("timeframes", ()))
            if bool(value.get("enabled", False))
            else ()
        )
        for timeframe in matrix[str(name)]:
            ml_policies[f"{str(name).lower()}:{timeframe.lower()}"] = str(
                value.get("ml_policy", "ML_OPTIONAL")
            ).upper()
    for name in FOREX_INITIAL_DISABLED:
        if matrix.get(name):
            raise ValueError(f"Forex-sensitive strategy {name!r} cannot be enabled before validation")
    return replace(base, timeframe_map=matrix, ml_policies=ml_policies, market="FOREX")
