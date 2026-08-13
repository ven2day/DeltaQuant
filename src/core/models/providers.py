"""Provider protocols at the broker-neutral market boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from src.core.models.market import CanonicalCandle, CanonicalQuote, Instrument, ProviderHealth


@runtime_checkable
class InstrumentProvider(Protocol):
    async def list_instruments(self) -> list[Instrument]: ...


@runtime_checkable
class MarketDataProvider(InstrumentProvider, Protocol):
    async def validate_connectivity(self) -> ProviderHealth: ...

    async def get_candles(
        self,
        instrument: str,
        timeframe: str,
        *,
        count: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        complete_only: bool = True,
    ) -> list[CanonicalCandle]: ...

    async def get_prices(self, instruments: Sequence[str]) -> dict[str, CanonicalQuote]: ...

    def stream_prices(self, instruments: Sequence[str]) -> AsyncIterator[CanonicalQuote]: ...

    async def close(self) -> None: ...


@runtime_checkable
class BrokerExecutionProvider(Protocol):
    @property
    def execution_enabled(self) -> bool: ...

    async def account_summary(self) -> dict[str, Any]: ...

    async def open_positions(self) -> list[dict[str, Any]]: ...

    async def pending_orders(self) -> list[dict[str, Any]]: ...

    async def submit_order(self, intent: Any) -> dict[str, Any]: ...

    async def close_position(self, instrument: str, units: float | None = None) -> dict[str, Any]: ...
