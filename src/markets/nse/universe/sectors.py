"""
NSE stock -> sector mapping.

Single source of truth for sector classification, used by the risk engine
(sector exposure limits) and by market-data features (sector-wise movers).
Best-effort labels for dashboard/risk-grouping purposes, not an authoritative
GICS classification.

If `settings.stock_universe_csv_path` is set and that CSV has an optional
'sector' column, those labels override STOCK_SECTORS for matching symbols —
so a custom symbol list can carry its own sector labels in the same file
instead of requiring a second edit here for every new symbol.
"""

import csv
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)

STOCK_SECTORS = {
    # Banking
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "KOTAKBANK": "Banking",
    "AXISBANK": "Banking",
    "INDUSINDBK": "Banking",
    # IT
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "LTIM": "IT",
    "PERSISTENT": "IT",
    "COFORGE": "IT",
    "MPHASIS": "IT",
    "AFFLE": "IT",
    # Pharma / Healthcare
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare",
    # Auto
    "TATAMOTORS": "Auto",
    "MARUTI": "Auto",
    "M&M": "Auto",
    "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    # Energy / Power
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "BPCL": "Energy",
    "IOC": "Energy",
    "NTPC": "Power",
    "POWERGRID": "Power",
    # Metals
    "TATASTEEL": "Metals",
    "HINDALCO": "Metals",
    "JSWSTEEL": "Metals",
    "COALINDIA": "Metals",
    # FMCG
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG",
    "JUBLFOOD": "FMCG",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Infrastructure
    "ADANIENT": "Infrastructure",
    "LT": "Infrastructure",
    "ADANIPORTS": "Infrastructure",
    # Financial Services / Insurance
    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",
    "HDFC": "Financial Services",
    "SHRIRAMFIN": "Financial Services",
    "HDFCLIFE": "Insurance",
    "SBILIFE": "Insurance",
    "PAYTM": "Fintech",
    "POLICYBZR": "Fintech",
    # Cement / Building materials
    "GRASIM": "Cement",
    "ULTRACEMCO": "Cement",
    "ASTRAL": "Building Materials",
    # Consumer goods / durables
    "ASIANPAINT": "Consumer Goods",
    "TITAN": "Consumer Goods",
    "BERGEPAINT": "Consumer Goods",
    "PIDILITIND": "Chemicals",
    "VOLTAS": "Consumer Durables",
    "HAVELLS": "Consumer Durables",
    "CROMPTON": "Consumer Durables",
    "DIXON": "Consumer Durables",
    "POLYCAB": "Electricals",
    # Consumer internet / new-age
    "ZOMATO": "Consumer Internet",
    "NYKAA": "Consumer Internet",
    "DELHIVERY": "Logistics",
    "IRCTC": "Travel & Tourism",
    "TRENT": "Retail",
}


def _load_sector_overrides_from_csv(path: str) -> dict[str, str]:
    """Read an optional 'symbol'/'sector' column pair from the universe CSV.

    Returns {} (no overrides) if the file is missing, unreadable, or has no
    'sector' column at all — a bad/absent column must never break sector lookup,
    it should just fall back to STOCK_SECTORS as if the CSV didn't exist.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}
            symbol_col = next(
                (name for name in reader.fieldnames if name.strip().lower() == "symbol"), None
            )
            sector_col = next(
                (name for name in reader.fieldnames if name.strip().lower() == "sector"), None
            )
            if symbol_col is None or sector_col is None:
                return {}
            overrides: dict[str, str] = {}
            for row in reader:
                symbol = (row.get(symbol_col) or "").strip().upper()
                sector = (row.get(sector_col) or "").strip()
                if symbol and sector:
                    overrides[symbol] = sector
            return overrides
    except OSError:
        logger.warning("Could not read sector overrides from %s", path, exc_info=True)
        return {}


# Keyed by CSV path so distinct paths (e.g. across tests) never collide, and a
# path only ever gets parsed once per process — matches the "load once at
# startup" treatment the same CSV gets in stock_discovery.py.
_sector_override_cache: dict[str, dict[str, str]] = {}


def _csv_sector_overrides() -> dict[str, str]:
    path = get_settings().stock_universe_csv_path
    if not path:
        return {}
    if path not in _sector_override_cache:
        _sector_override_cache[path] = _load_sector_overrides_from_csv(path)
    return _sector_override_cache[path]


def get_stock_sector(symbol: str) -> str:
    """Get sector for a stock symbol."""
    # Normalize symbol (remove exchange suffix, case-insensitively)
    clean_symbol = symbol.upper().replace(".NS", "")
    overrides = _csv_sector_overrides()
    if clean_symbol in overrides:
        return overrides[clean_symbol]
    return STOCK_SECTORS.get(clean_symbol, "Unknown")
