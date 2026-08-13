"""Pure execution preflight checks that must run after fresh or cached AI review."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.markets.nse.sessions.market_time import IST, now_ist


@dataclass(frozen=True)
class QuoteFreshnessDecision:
    approved: bool
    reason: str = ""


def executable_market_quote_is_fresh(
    quote: Any,
    *,
    present_in_latest_batch: bool,
    max_age_seconds: float,
    session_date: date,
    now: datetime | None = None,
    future_tolerance_seconds: float = 5.0,
) -> QuoteFreshnessDecision:
    """Validate one MARKET_PAPER candidate against its exchange timestamp.

    Receipt time is intentionally not a substitute: a broker can successfully return
    yesterday's or an illiquid instrument's old LTP in a brand-new HTTP response.
    """
    if not present_in_latest_batch:
        return QuoteFreshnessDecision(False, "symbol absent from latest quote batch")
    exchange_timestamp = getattr(quote, "exchange_timestamp", None)
    if not isinstance(exchange_timestamp, datetime):
        return QuoteFreshnessDecision(False, "missing or invalid exchange timestamp")
    if exchange_timestamp.tzinfo is None:
        exchange_timestamp = exchange_timestamp.replace(tzinfo=IST)
    else:
        exchange_timestamp = exchange_timestamp.astimezone(IST)
    if exchange_timestamp.date() != session_date:
        return QuoteFreshnessDecision(False, "quote is not from the current NSE session")
    current = now or now_ist()
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    else:
        current = current.astimezone(IST)
    age_seconds = (current - exchange_timestamp).total_seconds()
    if age_seconds < -future_tolerance_seconds:
        return QuoteFreshnessDecision(False, "exchange timestamp is in the future")
    if max_age_seconds > 0 and age_seconds > max_age_seconds:
        return QuoteFreshnessDecision(False, f"quote is {age_seconds:.0f}s old")
    return QuoteFreshnessDecision(True)


def reviewed_price_is_fresh(
    reviewed_entry: float,
    current_price: float,
    *,
    max_move_fraction: float = 0.001,
) -> bool:
    """Return whether the executable quote still matches the reviewed opportunity."""
    if not math.isfinite(reviewed_entry) or not math.isfinite(current_price):
        return False
    if reviewed_entry <= 0 or current_price <= 0:
        return False
    return abs(current_price - reviewed_entry) / reviewed_entry <= max_move_fraction
