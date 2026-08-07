"""DhanHQ instrument-master resolver: NSE equity symbol -> Dhan security ID.

DhanHQ publishes a public CSV listing every tradable instrument (equities,
F&O, currency, debt, etc.) across all exchanges. This module downloads it
once, filters to plain NSE cash-equity rows, and builds a symbol -> security
ID map — replacing a hand-maintained ~20-symbol dict that silently dropped
any other symbol from live Dhan quotes/subscriptions. This is what lets a
custom `symbols.csv` universe (see stock_discovery.py) get live Dhan data for
*any* symbol in it, not just the ones someone happened to hardcode.

Schema verified directly against the live file (column names are not fully
documented by DhanHQ): a plain NSE equity row looks like
    NSE,E,2885,EQUITY,0,RELIANCE,1.0,Reliance Industries,,,,10.0000,NA,ES,EQ,...
i.e. SEM_EXM_EXCH_ID=NSE, SEM_SEGMENT=E, SEM_INSTRUMENT_NAME=EQUITY,
SEM_SERIES=EQ identify the row as a cash-equity listing (as opposed to F&O,
currency derivatives, SME, or debt instruments, which share the same
numeric-ID space and would otherwise collide).
"""

import csv
import io
import logging

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)

# Default mirrors settings.dhan_instrument_master_url — kept as a literal fallback
# so this module still works standalone (e.g. in tests) without a Settings object.
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
REQUEST_TIMEOUT_SECONDS = 30

# Last-resort fallback so market-data/order code can still start (with a
# reduced watchlist) if the instrument master is ever unreachable — better
# than a hard startup failure. Not meant to be hand-maintained further; the
# live fetch is the actual source of truth.
FALLBACK_SECURITY_IDS: dict[str, str] = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "HDFCBANK": "1333",
    "INFY": "1594",
    "ICICIBANK": "4963",
    "HINDUNILVR": "1394",
    "SBIN": "3045",
    "BHARTIARTL": "10604",
    "ITC": "1660",
    "KOTAKBANK": "1922",
    "LT": "11483",
    "AXISBANK": "5900",
    "ASIANPAINT": "236",
    "MARUTI": "10999",
    "TITAN": "3506",
    "BAJFINANCE": "317",
    "WIPRO": "3787",
    "ULTRACEMCO": "11532",
    "NESTLEIND": "17963",
    "TECHM": "13538",
}

_cached_map: dict[str, str] | None = None


def _fetch_full_map() -> dict[str, str]:
    """Download and parse the instrument master into a full symbol -> ID map.

    Raises on any network/parsing failure; callers handle the fallback.
    """
    url = get_settings().dhan_instrument_master_url or DHAN_SCRIP_MASTER_URL
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    resolved: dict[str, str] = {}
    for row in reader:
        if (
            row.get("SEM_EXM_EXCH_ID") == "NSE"
            and row.get("SEM_SEGMENT") == "E"
            and row.get("SEM_INSTRUMENT_NAME") == "EQUITY"
            and row.get("SEM_SERIES") == "EQ"
        ):
            symbol = (row.get("SEM_TRADING_SYMBOL") or "").strip()
            security_id = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
            if symbol and security_id:
                resolved[symbol] = security_id
    return resolved


def fetch_security_id_map(symbols: list[str] | None = None) -> dict[str, str]:
    """Return the NSE-equity symbol -> Dhan security ID map.

    Fetches DhanHQ's instrument master on first call and caches it for the
    rest of the process (matching the "load once at startup" treatment the
    discovery universe CSV gets) — call this once, not per-symbol. If
    `symbols` is given, only those are returned (missing ones are simply
    absent from the result, not an error); `None` returns the full map.

    On any network or parsing failure, logs a warning and falls back to
    `FALLBACK_SECURITY_IDS` (filtered to `symbols` if given) — an unreachable
    instrument list must never stop a trading session from starting.
    """
    global _cached_map

    if _cached_map is None:
        try:
            resolved = _fetch_full_map()
            if not resolved:
                logger.warning(
                    "DhanHQ instrument master returned no NSE equity rows; "
                    "using fallback watchlist"
                )
                _cached_map = dict(FALLBACK_SECURITY_IDS)
            else:
                _cached_map = resolved
                logger.info("Resolved %d NSE equity security IDs from DhanHQ", len(resolved))
        except Exception:
            logger.warning(
                "Failed to fetch DhanHQ instrument master; using fallback watchlist",
                exc_info=True,
            )
            _cached_map = dict(FALLBACK_SECURITY_IDS)

    if symbols is None:
        return dict(_cached_map)
    return {s: _cached_map[s] for s in symbols if s in _cached_map}
