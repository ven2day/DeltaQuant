"""Shared feature contracts without eager aggregation imports."""

from __future__ import annotations

from typing import Any

from src.core.features.context import (
    MarketRelativeFeatures,
    build_historical_market_relative_context,
    build_market_relative_context,
)


def __getattr__(name: str) -> Any:
    if name == "FeatureSnapshot":
        from src.core.aggregation.consolidation import FeatureSnapshot

        return FeatureSnapshot
    raise AttributeError(name)


__all__ = [
    "FeatureSnapshot",
    "MarketRelativeFeatures",
    "build_historical_market_relative_context",
    "build_market_relative_context",
]
