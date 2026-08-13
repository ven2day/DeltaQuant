"""Timezone-safe NSE market-hour helpers owned by the NSE domain."""

from datetime import datetime, time, timedelta, timezone

from src.markets.nse.sessions.calendar import NseSessionStatus, get_nse_calendar

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_market_hours(now: datetime | None = None) -> bool:
    return market_session_status(now) is NseSessionStatus.OPEN


def market_session_status(now: datetime | None = None) -> NseSessionStatus:
    return get_nse_calendar().status(now or now_ist())


def is_trading_window(start: str, end: str, now: datetime | time | str | None = None) -> bool:
    if now is None:
        now = now_ist()
    start_time = time.fromisoformat(start)
    end_time = time.fromisoformat(end)
    if isinstance(now, datetime):
        if not get_nse_calendar().is_trading_day(now.astimezone(IST).date()):
            return False
        current_time = now.astimezone(IST).time().replace(tzinfo=None)
    elif isinstance(now, str):
        current_time = time.fromisoformat(now)
    else:
        current_time = now
    return start_time <= current_time <= end_time
