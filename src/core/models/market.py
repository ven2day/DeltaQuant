"""Broker-neutral market-domain models used by the DeltaQuant core.

The strategy layer consumes :class:`FeatureSnapshot` and never imports this module's
provider implementations.  These models are the normalization boundary between a
broker feed (Dhan, OANDA, or a future crypto venue) and the shared runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Market(StrEnum):
    NSE = "NSE"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"

    @classmethod
    def parse(cls, value: Market | str) -> Market:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().upper())


class MarketProvider(StrEnum):
    DHAN = "DHAN"
    OANDA = "OANDA"
    SIMULATED = "SIMULATED"
    FUTURE = "FUTURE"


class VolumeType(StrEnum):
    EXCHANGE_UNITS = "EXCHANGE_UNITS"
    OANDA_TICK_COUNT = "OANDA_TICK_COUNT"
    UNKNOWN = "UNKNOWN"


class PriceSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Instrument:
    market: Market
    symbol: str
    broker_symbol: str
    provider: MarketProvider
    base_currency: str = ""
    quote_currency: str = ""
    display_precision: int = 2
    pip_location: int = 0
    trade_units_precision: int = 0
    minimum_trade_size: float = 1.0
    maximum_order_units: float | None = None
    margin_rate: float | None = None
    status: str = "tradeable"
    tradeable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def pip_size(self) -> float:
        return 10.0**self.pip_location


@dataclass(frozen=True)
class CanonicalCandle:
    symbol: str
    market: Market
    provider: MarketProvider
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool
    source: str
    volume_type: VolumeType = VolumeType.UNKNOWN

    def __post_init__(self) -> None:
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            raise ValueError("Canonical candle timestamps must be timezone-aware")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("Candle high is inconsistent with OHLC values")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Candle low is inconsistent with OHLC values")

    @property
    def timestamp_utc(self) -> datetime:
        return self.timestamp.astimezone(UTC)


@dataclass(frozen=True)
class CanonicalQuote:
    symbol: str
    market: Market
    provider: MarketProvider
    timestamp: datetime
    bid: float
    ask: float
    tradeable: bool
    source: str
    status: str = "tradeable"
    liquidity_bid: float | None = None
    liquidity_ask: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Canonical quote timestamps must be timezone-aware")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Bid and ask must be positive")
        if self.ask < self.bid:
            raise ValueError("Ask cannot be below bid")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def spread_pips(self, instrument: Instrument) -> float:
        return self.spread / instrument.pip_size

    def executable_price(self, side: PriceSide | str) -> float:
        normalized = PriceSide(str(side).upper())
        return self.ask if normalized is PriceSide.BUY else self.bid

    def age_seconds(self, now: datetime | None = None) -> float:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("Quote-age reference timestamp must be timezone-aware")
        return max(0.0, (current.astimezone(UTC) - self.timestamp.astimezone(UTC)).total_seconds())


@dataclass(frozen=True)
class ProviderHealth:
    market: Market
    provider: MarketProvider
    healthy: bool
    market_data: str
    execution: str
    environment: str = ""
    pricing_stream: str = "disabled"
    latest_price_age_seconds: float | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "provider": self.provider.value,
            "healthy": self.healthy,
            "market_data": self.market_data,
            "execution": self.execution,
            "environment": self.environment,
            "pricing_stream": self.pricing_stream,
            "latest_price_age_seconds": self.latest_price_age_seconds,
            "message": self.message,
        }
