"""Append-only history of every signal the pipeline considers.

Distinct from ``TradeJournal`` (which records actual fills): this captures every signal
at its final pipeline disposition — approved, rejected by ``signal_validation``, or
rejected by ``risk_compliance`` — so the web UI can show a signal history even for
signals that never became trades. Persists to Postgres (no local/file fallback); the
``source`` column ("live" vs "backfill") distinguishes real pipeline decisions from
historical replays instead of using separate storage locations for each.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.orm import Session

from src.db.base import Base, get_session

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """One signal's final disposition, ready to serialize for the web UI."""

    timestamp: str
    symbol: str
    side: str
    entry_price: float
    timeframe: str
    strategy: str
    confidence: float
    status: str
    reason: str = ""
    source: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_signal(
        cls,
        signal: dict[str, Any],
        status: str,
        *,
        reason: str | None = None,
        source: str = "live",
    ) -> "SignalRecord":
        """Build a record from a pipeline signal dict (``TradingSignal.to_dict()`` shape,
        possibly with ``validation``/``risk_result`` merged in by later nodes)."""
        derived_reason = ""
        if status == "rejected_validation":
            validation = signal.get("validation") or {}
            derived_reason = validation.get("reasoning") or signal.get("rejection_reason") or ""
        elif status == "rejected_risk":
            failures = signal.get("risk_result", {}).get("failures", [])
            derived_reason = failures[0].get("message", "") if failures else ""

        return cls(
            timestamp=signal.get("timestamp") or datetime.now().isoformat(),
            symbol=signal.get("symbol", ""),
            side=signal.get("signal_type", ""),
            entry_price=float(signal.get("entry_price") or 0.0),
            timeframe=signal.get("timeframe", ""),
            strategy=signal.get("strategy", ""),
            confidence=float(signal.get("confidence") or 0.0),
            status=status,
            reason=reason if reason is not None else derived_reason,
            source=source,
        )


class SignalHistoryRecord(Base):
    """One row per logged signal. ``timestamp`` is kept as an ISO-8601 string — lexical
    ordering on that format matches chronological ordering, so range filters/sorts work
    the same as on a real datetime column without a timezone round-trip."""

    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(40), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(30), nullable=False, index=True)
    reason = Column(Text, default="", nullable=False)
    source = Column(String(20), default="live", nullable=False, index=True)


class SignalLogger:
    """Appends signal records to Postgres and reads back recent history."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url

    def _session(self) -> Session:
        return get_session(self._database_url)

    def log(self, record: SignalRecord) -> None:
        """Insert one record. Never raises — a logging failure must not break a trading cycle."""
        session = self._session()
        try:
            session.add(
                SignalHistoryRecord(
                    timestamp=record.timestamp,
                    symbol=record.symbol,
                    side=record.side,
                    entry_price=record.entry_price,
                    timeframe=record.timeframe,
                    strategy=record.strategy,
                    confidence=record.confidence,
                    status=record.status,
                    reason=record.reason,
                    source=record.source,
                )
            )
            session.commit()
        except Exception:
            logger.exception("Failed to append signal log record")
            session.rollback()
        finally:
            session.close()

    def read_recent(self, days: int = 7) -> list[dict[str, Any]]:
        """Return signal records from the last ``days`` days (live + backfill), newest first."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        session = self._session()
        try:
            rows = (
                session.query(SignalHistoryRecord)
                .filter(SignalHistoryRecord.timestamp >= cutoff)
                .order_by(SignalHistoryRecord.timestamp.desc())
                .all()
            )
            return [
                {
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "side": row.side,
                    "entry_price": row.entry_price,
                    "timeframe": row.timeframe,
                    "strategy": row.strategy,
                    "confidence": row.confidence,
                    "status": row.status,
                    "reason": row.reason,
                    "source": row.source,
                }
                for row in rows
            ]
        except Exception:
            logger.exception("Failed to read signal history from Postgres")
            return []
        finally:
            session.close()

    def prune(self, days: int = 7) -> None:
        """Delete records older than ``days`` — call periodically to bound table growth."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        session = self._session()
        try:
            session.query(SignalHistoryRecord).filter(
                SignalHistoryRecord.timestamp < cutoff
            ).delete(synchronize_session=False)
            session.commit()
        except Exception:
            logger.exception("Failed to prune old signal history records")
            session.rollback()
        finally:
            session.close()
