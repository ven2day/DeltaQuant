"""
Unified Market Data Manager

Automatically selects between WebSocket (live) and Simulated data
based on market hours and connection availability.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.config import get_settings
from src.market.dhan_quotes_feed import QuotesFeed
from src.market.simulated_data import SimulatedMarketData
from src.market.websocket_feed import (
    NSE_WATCHLIST,
    DhanWebSocketFeed,
    FeedRequestCode,
    QuoteData,
)
from src.utils.market_time import MARKET_CLOSE, MARKET_OPEN, is_market_hours, now_ist

logger = logging.getLogger(__name__)


# NSE Market Hours (IST) — re-exported from utils.market_time for backwards compat.
__all__ = ["MARKET_OPEN", "MARKET_CLOSE", "is_market_open", "MarketQuote", "MarketDataManager"]


def is_market_open() -> bool:
    """Check if the NSE market is currently open (evaluated in IST, not host-local time)."""
    return is_market_hours()


@dataclass
class MarketQuote:
    """Unified market quote structure."""

    symbol: str
    last_price: float
    open: float
    high: float
    low: float
    close: float  # Previous close
    change: float
    change_percent: float
    volume: int
    is_live: bool  # True if from WebSocket, False if simulated
    timestamp: datetime = field(default_factory=now_ist)

    @property
    def is_bullish(self) -> bool:
        return self.change_percent > 0

    @property
    def age_seconds(self) -> float:
        """Seconds since this quote was recorded (IST clock)."""
        now = now_ist()
        ts = self.timestamp
        if ts.tzinfo is None:
            now = now.replace(tzinfo=None)
        return max(0.0, (now - ts).total_seconds())

    def is_stale(self, max_age_seconds: float) -> bool:
        """True if the quote is older than max_age_seconds (<= 0 disables the check)."""
        if max_age_seconds <= 0:
            return False
        return self.age_seconds > max_age_seconds

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "is_live": self.is_live,
            "timestamp": self.timestamp.isoformat(),
        }


class MarketDataManager:
    """
    Unified market data manager.

    Automatically uses:
    - WebSocket feed during market hours
    - Simulated data after hours or on connection failure
    """

    def __init__(
        self,
        on_quote: Callable[[MarketQuote], None] | None = None,
        symbols: list[str] | None = None,
    ):
        self.on_quote = on_quote
        self.symbols = symbols or list(NSE_WATCHLIST.keys())
        self.settings = get_settings()

        self.websocket_feed: DhanWebSocketFeed | None = None
        self.rest_quotes_feed: QuotesFeed | None = None
        simulated_seed = getattr(self.settings, "simulated_seed", 20260807)
        simulated_history_bars = getattr(
            self.settings, "simulated_history_5m_bars", 3000
        )
        self.simulated_data = SimulatedMarketData(
            symbols=self.symbols,
            seed=simulated_seed if isinstance(simulated_seed, int) else 20260807,
            history_bars=(
                simulated_history_bars
                if isinstance(simulated_history_bars, int)
                else 3000
            ),
        )

        self.is_live = False
        self.data_source = "simulated"  # "dhan" or "simulated"
        self.quotes: dict[str, MarketQuote] = {}
        self.running = False

    def _on_websocket_quote(self, quote: QuoteData):
        """Handle incoming WebSocket quote."""
        market_quote = MarketQuote(
            symbol=quote.symbol,
            last_price=quote.last_price,
            open=quote.open,
            high=quote.high,
            low=quote.low,
            close=quote.prev_close,
            change=quote.change,
            change_percent=quote.change_percent,
            volume=quote.volume,
            is_live=True,
        )

        self.quotes[quote.symbol] = market_quote

        if self.on_quote:
            self.on_quote(market_quote)

    def _on_websocket_error(self, error: Exception):
        """Handle WebSocket error - fallback to simulated."""
        logger.warning(f"WebSocket error: {error}")
        logger.info("Falling back to simulated data")
        self.is_live = False

    def _poll_dhan_rest_quotes(self) -> bool:
        """Refresh quotes from DhanHQ's batched REST endpoint."""
        if self.rest_quotes_feed is None:
            self.rest_quotes_feed = QuotesFeed(symbols=self.symbols)

        quotes = self.rest_quotes_feed.fetch_quotes()
        if not quotes:
            return False

        for quote in quotes.values():
            market_quote = MarketQuote(
                symbol=quote.symbol,
                last_price=quote.last_price,
                open=quote.open,
                high=quote.high,
                low=quote.low,
                close=quote.close,
                change=quote.change,
                change_percent=quote.change_percent,
                volume=quote.volume,
                is_live=True,
            )
            self.quotes[market_quote.symbol] = market_quote
            if self.on_quote:
                self.on_quote(market_quote)
        return True

    async def start(self) -> bool:
        """
        Start the market data manager.

        Returns:
            True if using live Dhan data, False if simulated
        """
        self.running = True

        # An explicitly isolated synthetic-history session is a coherent simulation,
        # regardless of whether NSE happens to be open. Never mix its candles with Dhan
        # quotes or probe a live data endpoint.
        if (
            self.settings.enable_synthetic_history
            and not self.settings.enable_dhan_historical_data
            and not self.settings.enable_dhan_quotes
        ):
            self.is_live = False
            self.data_source = "simulated"
            self._load_simulated_quotes()
            return False

        # Check configured data source
        data_source = self.settings.market_data_source

        # DhanHQ WebSocket (real-time, requires account)
        if data_source == "dhan" and is_market_open():
            # Check if DhanHQ credentials are available
            if self.settings.dhan_client_id and self.settings.dhan_access_token:
                logger.info("Market is OPEN - attempting DhanHQ WebSocket connection")

                self.websocket_feed = DhanWebSocketFeed(
                    on_quote=self._on_websocket_quote,
                    on_error=self._on_websocket_error,
                )

                connected = await self.websocket_feed.connect()

                if connected:
                    await self.websocket_feed.subscribe_nse_stocks(
                        symbols=self.symbols,
                        mode=FeedRequestCode.SUBSCRIBE_QUOTE,
                    )
                    self.is_live = True
                    self.data_source = "dhan"
                    logger.info(f"Live WebSocket feed active for {len(self.symbols)} stocks")
                    return True
                else:
                    logger.warning("WebSocket connection failed - using simulated data")
            else:
                logger.info("DhanHQ WebSocket credentials unavailable; falling back to REST quotes")

        if data_source == "dhan" and self.settings.enable_dhan_quotes:
            logger.info("Using DhanHQ REST quote polling for %d stocks", len(self.symbols))
            if self._poll_dhan_rest_quotes():
                self.data_source = "dhan_rest"
                return True
            logger.warning("DhanHQ quote API cooling down after historical backfill; retrying in 60s")
            await asyncio.sleep(60)
            if self._poll_dhan_rest_quotes():
                self.data_source = "dhan_rest"
                return True
            raise RuntimeError("DhanHQ REST quote polling failed after cooldown")
        else:
            logger.info("Market data source not configured - using simulated data")

        # Fallback: Simulated data
        self.is_live = False
        self.data_source = "simulated"
        self._load_simulated_quotes()
        return False

    def _load_simulated_quotes(self):
        """Load simulated quotes into cache."""
        sim_quotes = self.simulated_data.get_quotes(self.symbols)

        for symbol, sq in sim_quotes.items():
            self.quotes[symbol] = MarketQuote(
                symbol=sq.symbol,
                last_price=float(sq.last_price),
                open=float(sq.open),
                high=float(sq.high),
                low=float(sq.low),
                close=float(sq.close),
                change=float(sq.change),
                change_percent=float(sq.change_percent),
                volume=int(sq.volume),
                is_live=False,
                timestamp=sq.timestamp,
            )

            if self.on_quote:
                self.on_quote(self.quotes[symbol])

    def refresh_simulated(self) -> None:
        """Refresh simulated quotes with new random movements."""
        if not self.is_live:
            self.simulated_data.tick()
            self._load_simulated_quotes()

    def refresh(self) -> None:
        """
        Pull the latest quotes for the active source — call once per trading cycle.

        - simulated: advance the random walk (new movement each cycle).
        - dhan websocket: pushes via callback, so this is a no-op.
        """
        if self.data_source == "dhan_rest":
            self._poll_dhan_rest_quotes()
        elif self.data_source == "simulated":
            self.refresh_simulated()

    async def listen(self):
        """Listen for market data updates."""
        if self.is_live and self.websocket_feed:
            await self.websocket_feed.listen()

    async def stop(self):
        """Stop the market data manager."""
        self.running = False

        if self.websocket_feed:
            await self.websocket_feed.disconnect()

    def get_quote(self, symbol: str) -> MarketQuote | None:
        """Get the latest quote for a symbol."""
        return self.quotes.get(symbol)

    def get_all_quotes(self) -> dict[str, MarketQuote]:
        """Get all current quotes."""
        return self.quotes.copy()

    def get_trading_candidates(self, min_change: float = 0.5) -> list[MarketQuote]:
        """Get stocks with significant movement."""
        candidates = [q for q in self.quotes.values() if abs(q.change_percent) >= min_change]
        candidates.sort(key=lambda q: abs(q.change_percent), reverse=True)
        return candidates

    def get_top_movers(self, n: int = 5) -> tuple[list[MarketQuote], list[MarketQuote]]:
        """Get top gainers and losers."""
        sorted_quotes = sorted(
            self.quotes.values(),
            key=lambda q: q.change_percent,
            reverse=True,
        )

        gainers = [q for q in sorted_quotes if q.change_percent > 0][:n]
        losers = [q for q in reversed(sorted_quotes) if q.change_percent < 0][:n]

        return gainers, losers


async def test_market_manager():
    """Test the market data manager."""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("[MARKET] ₹DeltaQuant - Market Data Manager Test")
    print("=" * 60)

    # Check market status
    if is_market_open():
        print("\n[STATUS] Market is OPEN - will try live data")
    else:
        print("\n[STATUS] Market is CLOSED - will use simulated data")

    quotes_received = []

    def on_quote(quote: MarketQuote):
        quotes_received.append(quote)
        mode = "LIVE" if quote.is_live else "SIM"
        trend = "UP" if quote.is_bullish else "DOWN"
        print(
            f"  [{mode}] {quote.symbol:<12} Rs.{quote.last_price:>10,.2f}  {quote.change_percent:>+6.2f}% [{trend}]"
        )

    manager = MarketDataManager(
        on_quote=on_quote,
        symbols=["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"],
    )

    print("\n[START] Starting market data manager...")
    is_live = await manager.start()

    print(f"\n[MODE] Using {'LIVE WebSocket' if is_live else 'SIMULATED'} data")

    if is_live:
        print("\n[LISTEN] Listening for live data (10 seconds)...")
        try:
            await asyncio.wait_for(manager.listen(), timeout=10)
        except TimeoutError:
            pass
    else:
        print("\n[QUOTES] Current simulated quotes:")
        for _ in range(3):
            await asyncio.sleep(2)
            manager.refresh_simulated()

    print(f"\n[RESULT] Received {len(quotes_received)} quote updates")

    # Show trading candidates
    print("\n[CANDIDATES] Trading Candidates:")
    for c in manager.get_trading_candidates()[:5]:
        direction = "BUY" if c.is_bullish else "SELL"
        print(f"  {c.symbol}: {c.change_percent:+.2f}% -> {direction}")

    await manager.stop()
    print("\n[DONE] Test complete")


if __name__ == "__main__":
    asyncio.run(test_market_manager())
