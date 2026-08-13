"""Build the NSE strategy matrix from ``config/nse/strategies.yaml``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.core.strategies.config import ProductionStrategyConfig
from src.markets.nse.config import load_nse_config


def build_strategy_config(settings: Any) -> ProductionStrategyConfig:
    base = ProductionStrategyConfig.from_settings(settings)
    configured = load_nse_config().strategies.get("strategies", {})
    matrix = dict(base.timeframe_map)
    ml_policies = dict(base.ml_policies)
    for name, value in configured.items():
        if not isinstance(value, dict):
            raise ValueError(f"NSE strategy configuration for {name!r} must be a mapping")
        matrix[str(name)] = (
            tuple(str(item) for item in value.get("timeframes", ()))
            if bool(value.get("enabled", False))
            else ()
        )
        for timeframe in matrix[str(name)]:
            ml_policies[f"{str(name).lower()}:{timeframe.lower()}"] = str(
                value.get("ml_policy", "ML_OPTIONAL")
            ).upper()
    return replace(base, timeframe_map=matrix, ml_policies=ml_policies, market="NSE")
