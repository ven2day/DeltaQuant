"""Crypto 24/7 session semantics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CryptoTradingCalendar:
    def is_open(self) -> bool:
        return True


__all__ = ["CryptoTradingCalendar"]

