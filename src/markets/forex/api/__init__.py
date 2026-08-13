"""Forex-owned dashboard read model."""

from collections.abc import Callable
from typing import Any

from src.markets.api_registry import MarketApiView


def forex_api_view(
    *,
    status: Callable[[], dict[str, Any]],
    signals: Callable[[int], list[dict[str, Any]]],
    positions: Callable[[], list[dict[str, Any]]],
    candidates: Callable[[], list[dict[str, Any]]] = lambda: [],
    strategies: Callable[[], list[dict[str, Any]]] = lambda: [],
    models: Callable[[], list[dict[str, Any]]] = lambda: [],
) -> MarketApiView:
    return MarketApiView(
        market="FOREX",
        status=status,
        signals=signals,
        positions=positions,
        candidates=candidates,
        strategies=strategies,
        models=models,
    )


__all__ = ["forex_api_view"]
