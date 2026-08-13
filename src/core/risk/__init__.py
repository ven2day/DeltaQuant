"""Generic market-risk contracts; implementations live under ``src.markets``."""

from __future__ import annotations

from typing import Any, Protocol

from src.core.namespaces import normalize_market


class MarketRiskAdapter(Protocol):
    def evaluate(self, candidate: dict[str, Any], portfolio: dict[str, Any]) -> Any: ...


def market_kill_switch_active(settings: Any, market: str) -> bool:
    """Global switch wins; otherwise only the selected market switch applies."""
    normalized = normalize_market(market).value.lower()
    return bool(
        getattr(settings, "global_kill_switch", False)
        or getattr(settings, f"{normalized}_kill_switch", False)
    )


__all__ = ["MarketRiskAdapter", "market_kill_switch_active"]
