"""Launch the OANDA Forex worker independently from NSE."""

from __future__ import annotations

import asyncio

from src.config import get_settings
from src.core.utils.logging_config import configure_runtime_logging
from src.markets.forex.config import load_forex_settings
from src.markets.forex.runtime.engine import run_forex_market_worker


async def run() -> None:
    scoped = load_forex_settings()
    if scoped.forex_execution_enabled:
        raise RuntimeError("OANDA broker execution is disabled; use local PAPER decisions only")
    settings = get_settings()
    if str(settings.market).upper() != "FOREX":
        raise RuntimeError("Forex worker requires MARKET=FOREX and a single Forex env profile")
    configure_runtime_logging(
        None,
        market="FOREX",
        provider="OANDA",
        runtime_id=str(getattr(settings, "runtime_id", "forex")),
    )
    await run_forex_market_worker(settings)


def main() -> None:
    asyncio.run(run())
