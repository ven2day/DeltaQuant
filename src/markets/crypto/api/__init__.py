"""Crypto-owned dashboard read model."""

from collections.abc import Callable
from typing import Any

from src.markets.api_registry import MarketApiView


def crypto_api_view(
    *,
    status: Callable[[], dict[str, Any]],
    signals: Callable[[int], list[dict[str, Any]]],
    positions: Callable[[], list[dict[str, Any]]],
) -> MarketApiView:
    return MarketApiView(market="CRYPTO", status=status, signals=signals, positions=positions)


__all__ = ["crypto_api_view"]
