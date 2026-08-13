"""Forex configuration boundary."""

from src.core.configuration import MarketConfigBundle, load_market_config
from src.markets.forex.config.settings import load_forex_settings


def load_forex_config() -> MarketConfigBundle:
    return load_market_config("FOREX")


__all__ = ["load_forex_config", "load_forex_settings"]
