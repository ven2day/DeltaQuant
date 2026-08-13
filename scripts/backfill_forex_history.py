"""Backfill normalized OANDA Practice candles into ``forex.market_candles``.

This script performs market-data reads only. It refuses OANDA LIVE and every broker
execution gate, and its output contains no credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.core.candles import CandleStore
from src.markets.forex.market_data import (
    OandaHistoricalBackfillService,
    create_forex_market_data_provider,
)
from src.markets.forex.persistence import bind_candle_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--symbols", default="", help="Comma-separated subset")
    parser.add_argument("--timeframes", default="15m,30m,1h,4h")
    parser.add_argument("--report", default="run/forex_history_coverage.json")
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Read the existing Forex schema without making OANDA candle requests",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    settings = get_settings()
    if settings.market != "FOREX" or settings.oanda_environment != "practice":
        raise RuntimeError("Forex history backfill accepts OANDA Practice only")
    if settings.allow_live_orders or settings.forex_execution_enabled:
        raise RuntimeError("History backfill requires all Forex execution gates disabled")
    provider = create_forex_market_data_provider(settings)
    store = CandleStore(
        settings.market_history_database_url or settings.database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
        schema=str(settings.forex_db_schema),
    )
    repository = bind_candle_repository(store)
    service = OandaHistoricalBackfillService(provider, repository)
    try:
        health = await provider.validate_connectivity()
        if not health.healthy:
            raise RuntimeError(f"OANDA Practice market data unhealthy: {health.message}")
        available = {item.symbol for item in await provider.list_instruments()}
        configured = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        symbols = configured or sorted(available)
        unknown = sorted(set(symbols).difference(available))
        if unknown:
            raise ValueError(f"Configured instruments unavailable: {','.join(unknown)}")
        timeframes = tuple(item.strip().lower() for item in args.timeframes.split(",") if item.strip())
        end = datetime.now(UTC)
        start = end - timedelta(days=max(1, int(args.days)))
        coverage: list[dict[str, object]] = []
        for symbol in symbols:
            for timeframe in timeframes:
                if args.coverage_only:
                    frame = repository.load_frame(
                        symbol, timeframe, start=start, end=end, complete_only=True
                    )
                    item = service.coverage(symbol, timeframe, frame)
                else:
                    item = await service.backfill(
                        symbol, timeframe, start=start, end=end
                    )
                coverage.append(item.to_dict())
        result: dict[str, object] = {
            "market": "FOREX",
            "provider": "OANDA",
            "environment": "PRACTICE",
            "execution": "DISABLED",
            "schema": "forex",
            "requested_days": int(args.days),
            "instruments": len(symbols),
            "timeframes": list(timeframes),
            "coverage": coverage,
        }
        report = ROOT / str(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    finally:
        await provider.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run(_parser().parse_args())), indent=2, sort_keys=True))
