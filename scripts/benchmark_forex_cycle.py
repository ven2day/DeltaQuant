"""Reproducible shared-core Forex settled-candle benchmark (no broker orders)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_eligibility import StrategyEligibilityRegistry
from src.config.settings import Settings
from src.core.candidates import SignalEngine
from src.core.models import (
    CanonicalCandle,
    CanonicalQuote,
    Instrument,
    Market,
    MarketProvider,
    VolumeType,
)
from src.markets.forex.risk import ForexRiskLimits
from src.markets.forex.runtime.cycle import ForexSettledCandleCycle
from src.markets.forex.strategies import build_strategy_config

SYMBOLS = (
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "USD_CHF",
    "NZD_USD",
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
)
TIMEFRAMES = ("15m", "30m", "1h", "4h")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


class FixtureForexProvider:
    """Deterministic OANDA-shaped data; never opens a network or execution client."""

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbols = symbols
        self.instruments = [self._instrument(symbol) for symbol in symbols]
        self.frames = {
            (symbol, timeframe): self._candles(symbol, timeframe, symbol_index)
            for symbol_index, symbol in enumerate(symbols)
            for timeframe in TIMEFRAMES
        }

    @staticmethod
    def _instrument(symbol: str) -> Instrument:
        base, quote = symbol.split("_")
        return Instrument(
            market=Market.FOREX,
            symbol=symbol,
            broker_symbol=symbol,
            provider=MarketProvider.OANDA,
            base_currency=base,
            quote_currency=quote,
            display_precision=3 if quote == "JPY" else 5,
            pip_location=-2 if quote == "JPY" else -4,
            trade_units_precision=0,
            minimum_trade_size=1,
            maximum_order_units=1_000_000,
        )

    @staticmethod
    def _candles(symbol: str, timeframe: str, symbol_index: int) -> list[CanonicalCandle]:
        minutes = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}[timeframe]
        start = datetime(2026, 1, 1, tzinfo=UTC)
        is_jpy = symbol.endswith("JPY")
        base_price = 145.0 if is_jpy else 1.05 + symbol_index * 0.01
        step = 0.01 if is_jpy else 0.0001
        output: list[CanonicalCandle] = []
        for index in range(260):
            trend = step * index * (1 if symbol_index % 2 == 0 else -0.35)
            oscillation = math.sin(index / 6) * step * 3
            close = base_price + trend + oscillation
            open_price = close - math.sin(index / 4) * step
            high = max(open_price, close) + step * 2
            low = min(open_price, close) - step * 2
            output.append(
                CanonicalCandle(
                    symbol=symbol,
                    market=Market.FOREX,
                    provider=MarketProvider.OANDA,
                    timeframe=timeframe,
                    timestamp=start + timedelta(minutes=minutes * index),
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=100 + (index % 30) * 5,
                    complete=True,
                    source="oanda_benchmark_fixture",
                    volume_type=VolumeType.OANDA_TICK_COUNT,
                )
            )
        return output

    async def list_instruments(self) -> list[Instrument]:
        return self.instruments

    async def get_candles(
        self, symbol: str, timeframe: str, **kwargs: Any
    ) -> list[CanonicalCandle]:
        count = int(kwargs.get("count", 260))
        return self.frames[(symbol, timeframe)][-count:]

    async def get_prices(self, symbols: list[str]) -> dict[str, CanonicalQuote]:
        now = datetime.now(UTC)
        output: dict[str, CanonicalQuote] = {}
        for symbol in symbols:
            last = self.frames[(symbol, "15m")][-1].close
            pip = 0.01 if symbol.endswith("JPY") else 0.0001
            output[symbol] = CanonicalQuote(
                symbol=symbol,
                market=Market.FOREX,
                provider=MarketProvider.OANDA,
                timestamp=now,
                bid=last - pip * 0.6,
                ask=last + pip * 0.6,
                tradeable=True,
                source="oanda_benchmark_fixture",
            )
        return output


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="benchmark-only",
        market="FOREX",
        forex_enabled=True,
        forex_environment="practice",
        oanda_environment="practice",
        oanda_account_id="fixture-account",
        oanda_access_token="fixture-token",
        trading_mode="paper",
        execution_mode="local_paper",
        allow_live_orders=False,
        enable_web_ui=False,
    )


async def benchmark(runs: int, symbol_count: int) -> dict[str, Any]:
    settings = _settings()
    config = build_strategy_config(settings)
    provider = FixtureForexProvider(SYMBOLS[:symbol_count])
    durations: list[float] = []
    last_metrics: dict[str, Any] = {}
    no_change: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="deltaquant-forex-registry-") as directory:
        for _ in range(runs):
            cycle = ForexSettledCandleCycle(
                provider=provider,  # type: ignore[arg-type]
                signal_engine=SignalEngine(production_config=config),
                strategy_config=config,
                eligibility_registry=StrategyEligibilityRegistry(directory),
                risk_limits=ForexRiskLimits(),
                settings=settings,
            )
            result = await cycle.run(timeframes=TIMEFRAMES)
            durations.append(result.metrics.total_duration_ms)
            last_metrics = result.metrics.to_dict()
            no_change = (await cycle.run(timeframes=TIMEFRAMES)).metrics.to_dict()
    return {
        "mode": "DETERMINISTIC_OANDA_SHAPED_FIXTURE",
        "orders_sent": 0,
        "runs": runs,
        "symbols": symbol_count,
        "timeframes": list(TIMEFRAMES),
        "p50_ms": median(durations),
        "p95_ms": _percentile(durations, 0.95),
        "max_ms": max(durations),
        "event_metrics": last_metrics,
        "no_change_metrics": no_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--symbols", type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.runs, args.symbols)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
