"""DhanHQ historical-candle feed — OHLCV data via DhanHQ's charts endpoints.

Exposes the ``(symbol, period, interval) -> DataFrame | None`` contract this
codebase's HistoricalFeed Protocol (historical_feed.py) expects — capitalized
Open/High/Low/Close/Volume columns, DatetimeIndex — so it flows through the
same ``normalize_ohlcv()``/``_resample_ohlcv()`` pipeline in history_manager.py
unchanged.

Two real API constraints (verified against DhanHQ's docs and the installed
``dhanhq`` SDK source, not guessed) shape this:

- DhanHQ's intraday intervals are only 1/5/15/25/60 minutes — no native 30m,
  which every timeframe-driven feature here needs. Built by resampling the
  native 15m interval, the same technique already used elsewhere in this
  codebase for H4-from-H1.
- Each ``/charts/intraday`` or ``/charts/historical`` call is capped at 90
  days of data per DhanHQ's docs. Every period this codebase actually
  requests (60d/730d/3mo) gets trimmed to a much shorter lookback downstream
  anyway (see scalping_screener.py's ``lookback_days``), so this clamps to 90
  days rather than paginating across multiple calls for a case that doesn't
  occur in practice.

Timestamps: DhanHQ returns Unix epoch seconds (UTC); this codebase's OHLCV
convention is tz-aware Asia/Kolkata timestamps. Converting explicitly to
Asia/Kolkata here is required, not cosmetic — day-grouping logic downstream
(e.g. the scalping screener's per-day zigzag scoring) buckets by calendar day
off this index, and a UTC vs IST mismatch would silently shift bars across day
boundaries.

Rate limiting: confirmed live that DhanHQ enforces a single account-wide cap
of 5 requests/second across ALL Data APIs combined (quote, ohlc, historical,
intraday) — not a separate budget per endpoint. A scalping-screener scan
calling get_historical() for ~275 symbols with zero throttling triggered
DH-904 Rate_Limit errors on nearly every call (each one returning None and
getting logged, so nothing crashed — but it meant DhanHQ was effectively
unusable under the unthrottled load). Every call here goes through the
shared get_dhan_data_api_limiter() (src/utils/rate_limiter.py) instead of a
local delay, since the limit is account-wide: a local-only throttle would
still be exceeded by other concurrent Dhan callers (e.g. the sector-movers
quote scan) drawing from the same account.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.config import get_settings
from src.market.dhan_auth import get_dhan_client, get_valid_access_token
from src.market.dhan_instruments import fetch_security_id_map
from src.utils.rate_limiter import get_dhan_data_api_limiter

logger = logging.getLogger(__name__)

MAX_REQUEST_DAYS = 90
EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT_TYPE = "EQUITY"
IST = "Asia/Kolkata"

# Period strings this codebase actually passes, mapped to a lookback
# in calendar days (clamped to MAX_REQUEST_DAYS below — see module docstring).
_PERIOD_DAYS: dict[str, int] = {
    "5d": 5,
    "10d": 10,
    "1mo": 30,
    "60d": 60,
    "3mo": 90,
    "6mo": 180,
    "730d": 730,
}

# Interval string -> (Dhan minute interval, resample rule if not native).
_INTRADAY_INTERVAL_MAP: dict[str, tuple[int, str | None]] = {
    "15m": (15, None),
    "30m": (15, "30min"),
    "60m": (60, None),
}


def _period_to_days(period: str) -> int:
    return min(_PERIOD_DAYS.get(period, MAX_REQUEST_DAYS), MAX_REQUEST_DAYS)


def _parse_ohlcv_response(data: dict[str, Any] | None) -> pd.DataFrame | None:
    """Parse DhanHQ's parallel-array response into a standard OHLCV DataFrame."""
    if not data or "timestamp" not in data:
        return None
    try:
        index = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST)
        df = pd.DataFrame(
            {
                "Open": data["open"],
                "High": data["high"],
                "Low": data["low"],
                "Close": data["close"],
                "Volume": data.get("volume", [0] * len(data["timestamp"])),
            },
            index=index,
        )
        return df if not df.empty else None
    except (KeyError, ValueError, TypeError):
        logger.warning("Unexpected DhanHQ historical response shape", exc_info=True)
        return None


class DhanHistoricalFeed:
    """Fetches OHLCV history from DhanHQ, matching the HistoricalFeed Protocol's contract."""

    def __init__(self, symbols: list[str] | None = None):
        settings = get_settings()
        self._client = get_dhan_client(settings.dhan_client_id, get_valid_access_token())
        # Pre-resolve for the given universe up front (one instrument-master fetch),
        # falling back to a lazy per-symbol resolve for anything requested later.
        self._security_ids: dict[str, str] = fetch_security_id_map(symbols) if symbols else {}

    def _security_id(self, symbol: str) -> str | None:
        if symbol in self._security_ids:
            return self._security_ids[symbol]
        resolved = fetch_security_id_map([symbol]).get(symbol)
        if resolved:
            self._security_ids[symbol] = resolved
        return resolved

    def _intraday_response(
        self, security_id: str, from_date: datetime, to_date: datetime, interval: int
    ) -> dict[str, Any]:
        """Fetch Dhan intraday candles, backing off on its account-wide rate limit."""
        response: dict[str, Any] = {}
        for attempt in range(3):
            get_dhan_data_api_limiter().acquire_sync()
            response = self._client.intraday_minute_data(
                security_id=security_id,
                exchange_segment=EXCHANGE_SEGMENT,
                instrument_type=INSTRUMENT_TYPE,
                from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval,
            )
            remarks = response.get("remarks") or {}
            if response.get("status") == "success" or remarks.get("error_code") != "DH-904":
                return response
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        return response

    def get_historical(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """Get historical OHLCV data for a symbol."""
        security_id = self._security_id(symbol)
        if security_id is None:
            logger.warning("No DhanHQ security ID for %s; cannot fetch historical data", symbol)
            return None

        days = _period_to_days(period)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        resample_rule: str | None = None

        try:
            if interval == "1d":
                # DhanHQ's daily endpoint rejects valid security IDs such as
                # HDFCBANK with DH-905. Its 60-minute endpoint succeeds for the
                # same IDs, so build genuine daily candles from that source.
                resample_rule = "1D"
                response = self._intraday_response(security_id, from_date, to_date, interval=60)
            else:
                mapping = _INTRADAY_INTERVAL_MAP.get(interval)
                if mapping is None:
                    raise ValueError(f"Unsupported interval for DhanHistoricalFeed: {interval}")
                dhan_interval, resample_rule = mapping
                response = self._intraday_response(
                    security_id, from_date, to_date, interval=dhan_interval
                )

            if response.get("status") != "success":
                logger.warning(
                    "DhanHQ historical fetch failed for %s (%s): %s",
                    symbol,
                    interval,
                    response.get("remarks"),
                )
                return None

            df = _parse_ohlcv_response(response.get("data"))
            if df is None:
                return None

            if resample_rule is not None:
                df = df.resample(resample_rule).agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                ).dropna(subset=["Open", "High", "Low", "Close"])

            return df
        except Exception:
            logger.warning(
                "Error fetching DhanHQ historical data for %s (%s)", symbol, interval, exc_info=True
            )
            return None
