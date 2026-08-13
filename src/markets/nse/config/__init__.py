"""NSE configuration boundary."""

from src.core.configuration import MarketConfigBundle, load_market_config
from src.markets.nse.config.settings import load_nse_settings


def load_nse_config() -> MarketConfigBundle:
    return load_market_config("NSE")


__all__ = ["load_nse_config", "load_nse_settings"]
