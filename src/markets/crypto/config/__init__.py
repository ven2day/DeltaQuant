"""Crypto configuration boundary."""

from src.core.configuration import MarketConfigBundle, load_market_config


def load_crypto_config() -> MarketConfigBundle:
    return load_market_config("CRYPTO")


__all__ = ["load_crypto_config"]

