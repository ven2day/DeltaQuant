"""NSE-owned dashboard read model."""

from collections.abc import Callable
from typing import Any

from src.markets.api_registry import MarketApiView


def nse_api_view(
    *,
    status: Callable[[], dict[str, Any]],
    signals: Callable[[int], list[dict[str, Any]]],
    positions: Callable[[], list[dict[str, Any]]],
    strategies: Callable[[], list[dict[str, Any]]] = lambda: [],
    models: Callable[[], list[dict[str, Any]]] = lambda: [],
) -> MarketApiView:
    return MarketApiView(
        market="NSE",
        status=status,
        signals=signals,
        positions=positions,
        strategies=strategies,
        models=models,
    )


__all__ = ["nse_api_view"]
