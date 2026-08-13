"""Canonical Dhan imports for the NSE domain.

All internal callers import this domain directly; the former Dhan compatibility
modules were removed during the physical-boundary cleanup.
New market-domain code must import Dhan through this package.
"""

from src.markets.nse.broker.dhan.auth import get_dhan_client, get_valid_access_token
from src.markets.nse.broker.dhan.historical import DhanHistoricalFeed
from src.markets.nse.broker.dhan.instruments import fetch_security_id_map
from src.markets.nse.broker.dhan.quotes import DhanQuotesFeed, Quote, QuotesFeed
from src.markets.nse.broker.dhan.websocket import DhanWebSocketFeed

__all__ = [
    "DhanHistoricalFeed",
    "DhanQuotesFeed",
    "DhanWebSocketFeed",
    "Quote",
    "QuotesFeed",
    "fetch_security_id_map",
    "get_dhan_client",
    "get_valid_access_token",
]
