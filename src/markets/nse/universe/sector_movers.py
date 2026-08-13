"""
Sector-wide top gainers/losers.

Scans a wide stock universe (not just the symbols actively picked for trading)
and groups movers by sector for dashboard display. Runs on its own multi-minute
timer, independent of the fast trading-signal loop.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from src.markets.nse.broker.dhan.quotes import QuotesFeed
from src.markets.nse.universe.sectors import get_stock_sector

logger = logging.getLogger(__name__)


class SectorQuote(Protocol):
    """Quote fields needed by the sector scanner, independent of data source."""

    symbol: str
    last_price: float
    change_percent: float


def _serialize(quote: SectorQuote) -> dict[str, Any]:
    return {
        "symbol": quote.symbol,
        "last_price": quote.last_price,
        "change_percent": quote.change_percent,
    }


def compute_sector_movers(
    quotes: Mapping[str, SectorQuote], top_n: int = 5
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group quotes by sector, split into top-N gainers/losers per sector.

    Mirrors the existing gainer/loser convention (change_percent > 0 sorted
    descending / < 0 sorted ascending, capped at top_n) already used by
    MarketDataManager.get_top_movers(), just applied per sector instead of
    globally. Sectors with no movers (nothing quoted, or everything flat) are
    omitted so the caller never has to special-case an empty section.
    """
    by_sector: dict[str, list[SectorQuote]] = defaultdict(list)
    for symbol, quote in quotes.items():
        by_sector[get_stock_sector(symbol)].append(quote)

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sector, sector_quotes in by_sector.items():
        gainers = sorted(
            (q for q in sector_quotes if q.change_percent > 0),
            key=lambda q: q.change_percent,
            reverse=True,
        )[:top_n]
        losers = sorted(
            (q for q in sector_quotes if q.change_percent < 0),
            key=lambda q: q.change_percent,
        )[:top_n]
        if not gainers and not losers:
            continue
        result[sector] = {
            "gainers": [_serialize(q) for q in gainers],
            "losers": [_serialize(q) for q in losers],
        }
    return result


class SectorMoversTracker:
    """Periodically refreshes sector-wide movers in a thread executor.

    ``QuotesFeed.fetch_quotes()`` is synchronous (a handful of batched DhanHQ
    calls) — scanning a wide universe can take real wall-clock seconds, so it must never run
    directly on the asyncio event loop (the same class of bug as a blocking
    ``time.sleep()`` in an async function: it would stall the whole trading
    loop and the WebSocket listener for the duration of the scan). ``run()``
    is meant to be started once via ``asyncio.create_task`` and left running
    for the process lifetime, on its own timer decoupled from the fast
    trading-signal loop.
    """

    def __init__(
        self,
        symbols: list[str],
        refresh_seconds: int,
        on_status: Callable[[str, str], None] | None = None,
        fetch_quotes: Callable[[], Mapping[str, SectorQuote]] | None = None,
        data_source: str = "dhan",
    ) -> None:
        self._feed = None if fetch_quotes is not None else QuotesFeed(symbols=symbols)
        self._fetch_quotes_override = fetch_quotes
        self.refresh_seconds = refresh_seconds
        self.sector_movers: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.scan_status = "pending"
        self.data_source = data_source
        # Optional sink for the dashboard/web-UI activity log (e.g. TradingStats.log_activity).
        # Kept as a plain callback rather than importing the dashboard here, so this module
        # stays usable (and testable) without it.
        self._on_status = on_status or (lambda message, level: None)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                if self._fetch_quotes_override is not None:
                    fetch_quotes = self._fetch_quotes_override
                elif self._feed is not None:
                    fetch_quotes = self._feed.fetch_quotes
                else:  # Defensive: constructor always configures one provider.
                    raise RuntimeError("Sector mover quote provider is unavailable")
                quotes = await loop.run_in_executor(None, fetch_quotes)
                self.sector_movers = compute_sector_movers(quotes)
                self.scan_status = "ready"
                logger.info("Sector movers refreshed: %d sectors", len(self.sector_movers))
                self._on_status(
                    f"Sector movers refreshed: {len(self.sector_movers)} sectors", "INFO"
                )
            except Exception as e:
                self.scan_status = "error"
                logger.exception("Sector movers refresh failed")
                self._on_status(f"Sector movers refresh failed: {e}", "WARNING")
            await asyncio.sleep(self.refresh_seconds)
