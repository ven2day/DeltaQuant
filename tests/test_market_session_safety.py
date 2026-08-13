from datetime import datetime, timedelta

from src.markets.nse.execution.preflight import executable_market_quote_is_fresh
from src.markets.nse.market_data.manager import MarketQuote
from src.markets.nse.sessions.calendar import NseSessionStatus, get_nse_calendar
from src.markets.nse.sessions.market_time import IST, is_market_hours


def _quote(exchange_time: datetime | None) -> MarketQuote:
    return MarketQuote(
        "RELIANCE",
        100,
        100,
        101,
        99,
        98,
        2,
        2,
        1000,
        True,
        exchange_timestamp=exchange_time,
        batch_id="batch",
    )


def test_nse_calendar_holiday_weekend_open_and_unknown_year():
    calendar = get_nse_calendar()
    assert calendar.status(datetime(2026, 1, 15, 10, 0, tzinfo=IST)) is NseSessionStatus.HOLIDAY
    assert calendar.status(datetime(2026, 8, 15, 10, 0, tzinfo=IST)) is NseSessionStatus.WEEKEND
    assert calendar.status(datetime(2026, 8, 10, 10, 0, tzinfo=IST)) is NseSessionStatus.OPEN
    assert calendar.status(datetime(2027, 8, 10, 10, 0, tzinfo=IST)) is NseSessionStatus.UNKNOWN
    assert not is_market_hours(datetime(2027, 8, 10, 10, 0, tzinfo=IST))


def test_market_quote_requires_current_batch_current_session_exchange_time():
    now = datetime(2026, 8, 10, 10, 0, tzinfo=IST)
    current = _quote(now - timedelta(seconds=10))
    assert executable_market_quote_is_fresh(
        current,
        present_in_latest_batch=True,
        max_age_seconds=30,
        session_date=now.date(),
        now=now,
    ).approved
    assert not executable_market_quote_is_fresh(
        current,
        present_in_latest_batch=False,
        max_age_seconds=30,
        session_date=now.date(),
        now=now,
    ).approved
    assert not executable_market_quote_is_fresh(
        _quote(now - timedelta(days=1)),
        present_in_latest_batch=True,
        max_age_seconds=30,
        session_date=now.date(),
        now=now,
    ).approved
    assert not executable_market_quote_is_fresh(
        _quote(None),
        present_in_latest_batch=True,
        max_age_seconds=30,
        session_date=now.date(),
        now=now,
    ).approved
