"""Crypto repository factory."""

from typing import Any

from src.core.persistence import SchemaBoundCandleRepository


def bind_candle_repository(store: Any) -> SchemaBoundCandleRepository:
    return SchemaBoundCandleRepository(store, market="CRYPTO", provider="UNCONFIGURED")


__all__ = ["bind_candle_repository"]

