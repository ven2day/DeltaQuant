"""NSE-owned strategy eligibility registry factory."""

from typing import Any

from src.backtesting.strategy_eligibility import (
    StrategyEligibilityRegistry,
    eligibility_registry_from_settings,
)


def create_registry(settings: Any) -> StrategyEligibilityRegistry:
    if str(settings.market).upper() != "NSE":
        raise ValueError("NSE registry requires MARKET=NSE")
    return eligibility_registry_from_settings(settings)


__all__ = ["create_registry"]

