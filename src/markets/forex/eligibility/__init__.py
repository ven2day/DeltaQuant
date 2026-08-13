"""Forex-owned strategy eligibility registry factory."""

from typing import Any

from src.backtesting.strategy_eligibility import (
    StrategyEligibilityRegistry,
    eligibility_registry_from_settings,
)
from src.markets.forex.eligibility.governance import demote_superseded_forex_approvals
from src.markets.forex.eligibility.regime import ForexRegimeClassifier


def create_registry(settings: Any) -> StrategyEligibilityRegistry:
    if str(settings.market).upper() != "FOREX":
        raise ValueError("Forex registry requires MARKET=FOREX")
    return eligibility_registry_from_settings(settings)


__all__ = [
    "ForexRegimeClassifier",
    "create_registry",
    "demote_superseded_forex_approvals",
]
