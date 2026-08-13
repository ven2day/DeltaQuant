"""UTC-first Forex session classification with London/New York DST support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo


class ForexSession(StrEnum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"
    ROLLOVER = "ROLLOVER"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class ForexTradingCalendar:
    """Classify the 24x5 market without using server-local time."""

    asia_zone: ZoneInfo = ZoneInfo("Asia/Tokyo")
    london_zone: ZoneInfo = ZoneInfo("Europe/London")
    new_york_zone: ZoneInfo = ZoneInfo("America/New_York")

    @staticmethod
    def _aware_utc(at: datetime | None) -> datetime:
        value = at or datetime.now(UTC)
        if value.tzinfo is None:
            raise ValueError("Forex calendar timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def is_open(self, at: datetime | None = None) -> bool:
        now = self._aware_utc(at)
        ny = now.astimezone(self.new_york_zone)
        weekday = ny.weekday()
        if weekday == 5:
            return False
        if weekday == 6:
            return ny.time() >= time(17, 0)
        if weekday == 4 and ny.time() >= time(17, 0):
            return False
        return True

    def classify(self, at: datetime | None = None) -> ForexSession:
        now = self._aware_utc(at)
        if not self.is_open(now):
            return ForexSession.CLOSED
        ny = now.astimezone(self.new_york_zone)
        if time(16, 55) <= ny.time() < time(17, 10):
            return ForexSession.ROLLOVER

        tokyo = now.astimezone(self.asia_zone)
        london = now.astimezone(self.london_zone)
        asia_open = time(9, 0) <= tokyo.time() < time(17, 0)
        london_open = time(8, 0) <= london.time() < time(17, 0)
        new_york_open = time(8, 0) <= ny.time() < time(17, 0)
        if london_open and new_york_open:
            return ForexSession.OVERLAP
        if london_open:
            return ForexSession.LONDON
        if new_york_open:
            return ForexSession.NEW_YORK
        if asia_open:
            return ForexSession.ASIA
        # The 24x5 market can be open between named liquid sessions. ASIA is the
        # least misleading operational bucket; CLOSED is reserved for actual closure.
        return ForexSession.ASIA
