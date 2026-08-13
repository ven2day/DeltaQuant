"""OANDA construction owned by the Forex worker domain."""

from __future__ import annotations

from typing import Any

from src.markets.forex.broker.oanda import OandaMarketDataProvider, OandaV20Client


def parse_forex_symbols(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip().upper() for item in raw.split(",") if item.strip()))


def create_forex_market_data_provider(settings: Any) -> OandaMarketDataProvider:
    if str(getattr(settings, "market", "")).upper() != "FOREX":
        raise ValueError("Forex provider factory requires MARKET=FOREX")
    client = OandaV20Client(
        environment=settings.oanda_environment,
        account_id=settings.oanda_account_id,
        access_token=settings.oanda_access_token,
        rest_url_override=settings.oanda_rest_url,
        stream_url_override=settings.oanda_stream_url,
        timeout_seconds=settings.oanda_request_timeout_seconds,
        max_read_retries=settings.oanda_max_read_retries,
    )
    return OandaMarketDataProvider(
        client,
        enabled_symbols=parse_forex_symbols(settings.forex_symbols),
        stale_seconds=settings.oanda_stream_stale_seconds,
    )


__all__ = ["create_forex_market_data_provider", "parse_forex_symbols"]
