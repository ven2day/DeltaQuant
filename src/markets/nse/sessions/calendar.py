"""Fail-closed NSE cash-market session calendar owned by the NSE domain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path

_IST = timezone(timedelta(hours=5, minutes=30), name="IST")


class NseSessionStatus(str, Enum):
    UNKNOWN = "unknown"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    OPEN = "open"
    POST_MARKET = "post_market"


@dataclass(frozen=True)
class NseCalendar:
    holidays_by_year: dict[int, frozenset[date]]

    @classmethod
    def from_file(cls, path: Path | None = None) -> NseCalendar:
        calendar_path = path or Path(__file__).with_name("holidays.json")
        payload = json.loads(calendar_path.read_text(encoding="utf-8"))
        years = {
            int(year): frozenset(date.fromisoformat(value) for value in values)
            for year, values in dict(payload.get("years", {})).items()
        }
        return cls(years)

    def is_known_year(self, year: int) -> bool:
        return year in self.holidays_by_year

    def is_trading_day(self, value: date) -> bool:
        if not self.is_known_year(value.year):
            return False
        return value.weekday() < 5 and value not in self.holidays_by_year[value.year]

    def status(self, value: datetime) -> NseSessionStatus:
        current = value.replace(tzinfo=_IST) if value.tzinfo is None else value.astimezone(_IST)
        current_date = current.date()
        if not self.is_known_year(current_date.year):
            return NseSessionStatus.UNKNOWN
        if current_date.weekday() >= 5:
            return NseSessionStatus.WEEKEND
        if current_date in self.holidays_by_year[current_date.year]:
            return NseSessionStatus.HOLIDAY
        current_time = current.time().replace(tzinfo=None)
        if current_time < time(9, 0):
            return NseSessionStatus.CLOSED
        if current_time < time(9, 15):
            return NseSessionStatus.PRE_OPEN
        if current_time < time(15, 30):
            return NseSessionStatus.OPEN
        return NseSessionStatus.POST_MARKET


_CALENDAR: NseCalendar | None = None


def get_nse_calendar() -> NseCalendar:
    global _CALENDAR
    if _CALENDAR is None:
        _CALENDAR = NseCalendar.from_file()
    return _CALENDAR
