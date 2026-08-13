"""Persistent, IST-calendar-day-scoped risk counters.

Backs the risk engine's daily trade limit and daily loss limit. A plain in-memory
counter silently resets on every process restart, which defeats the point of a *daily*
cap — restarting the bot a few times in one day would let each restart grant another
full quota. This store survives restarts and only rolls over at IST midnight.

Drawdown tracking (``src.markets.nse.risk.guards.DrawdownTracker``) intentionally stays
session-scoped and reseeded from live equity on each restart (see the seeding comment
in ``src/markets/nse/runtime/live.py``) — that behavior is unrelated to daily limits and is
preserved. ``update_drawdown`` here only mirrors its current value into Postgres for
visibility, it does not change how the tracker itself behaves.
"""

import logging
from datetime import date

from sqlalchemy import Column, Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Session

from src.db.base import Base, get_session
from src.markets.nse.sessions.market_time import now_ist

logger = logging.getLogger(__name__)


class DailyRiskRecord(Base):
    """One row per IST calendar day of trading activity."""

    __tablename__ = "daily_risk_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(40), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    trades_count = Column(Integer, default=0, nullable=False)
    exits_count = Column(Integer, default=0, nullable=False)
    profit_loss = Column(Float, default=0.0, nullable=False)
    max_drawdown = Column(Float, default=0.0, nullable=False)

    __table_args__ = (
        UniqueConstraint("namespace", "trade_date", name="uq_daily_risk_state_namespace_date"),
    )


class DailyRiskStore:
    """Reads/writes today's (IST) risk counters from Postgres — no local fallback."""

    def __init__(
        self,
        database_url: str | None = None,
        namespace: str = "paper_market_data",
    ) -> None:
        self._database_url = database_url
        self.namespace = namespace

    def _session(self) -> Session:
        return get_session(self._database_url)

    def _today(self) -> date:
        return now_ist().date()

    def _get_or_create(self, session: Session, trade_date: date) -> DailyRiskRecord:
        record = (
            session.query(DailyRiskRecord)
            .filter_by(namespace=self.namespace, trade_date=trade_date)
            .first()
        )
        if record is None:
            record = DailyRiskRecord(
                namespace=self.namespace,
                trade_date=trade_date,
                trades_count=0,
                exits_count=0,
                profit_loss=0.0,
                max_drawdown=0.0,
            )
            session.add(record)
            session.commit()
        return record

    def get_today(self) -> dict[str, float | int]:
        """Return the stable public daily-risk contract, creating today's row if new."""
        session = self._session()
        try:
            record = self._get_or_create(session, self._today())
            return {
                "trades_count": int(record.trades_count),
                "profit_loss": float(record.profit_loss),
                "max_drawdown": float(record.max_drawdown),
            }
        finally:
            session.close()

    def record_entry(self) -> None:
        """Reserve and commit one new paper entry against today's hard entry cap."""
        session = self._session()
        try:
            record = self._get_or_create(session, self._today())
            record.trades_count = int(record.trades_count) + 1
            session.commit()
            logger.debug("Daily risk state: %d entries", record.trades_count)
        finally:
            session.close()

    def record_exit_pnl(self, pnl: float) -> None:
        """Accumulate one exit leg's realized P&L without consuming a new-entry slot."""
        session = self._session()
        try:
            record = self._get_or_create(session, self._today())
            record.exits_count = int(record.exits_count) + 1
            record.profit_loss = float(record.profit_loss) + pnl
            session.commit()
            logger.debug(
                "Daily risk state: %d entries, %d exits, P&L Rs.%.2f",
                record.trades_count,
                record.exits_count,
                record.profit_loss,
            )
        finally:
            session.close()

    def record_trade(self, pnl: float) -> None:
        """Backward-compatible completed-trade helper used by older callers/tests."""
        self.record_entry()
        self.record_exit_pnl(pnl)

    def update_drawdown(self, max_drawdown: float) -> None:
        """Mirror the session's current max drawdown into today's record, for visibility only."""
        session = self._session()
        try:
            record = self._get_or_create(session, self._today())
            record.max_drawdown = max(float(record.max_drawdown), max_drawdown)  # type: ignore[arg-type]
            session.commit()
        finally:
            session.close()
