"""Single market-selection seam for provider construction."""

from __future__ import annotations

from typing import Any

from src.core.models import Market
from src.markets.forex.market_data.provider_factory import (
    create_forex_market_data_provider,
    parse_forex_symbols,
)

__all__ = ["create_market_data_provider", "parse_forex_symbols"]


def create_market_data_provider(settings: Any) -> Any | None:
    """Build the selected non-NSE adapter.

    NSE intentionally returns ``None`` because the established ``MarketDataManager``
    remains the Dhan adapter during the compatibility phase. The runtime has exactly
    one market-selection branch; strategy modules contain none.
    """

    market = Market.parse(getattr(settings, "market", "NSE"))
    if market is Market.NSE:
        return None
    if market is Market.CRYPTO:
        raise NotImplementedError("CRYPTO provider is not implemented")
    return create_forex_market_data_provider(settings)
