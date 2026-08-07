"""Shared Postgres persistence layer for live trading state.

Unlike ``TradeJournal``/``AgentMemoryDB`` (which fall back to a non-persistent in-memory
SQLite database if ``DATABASE_URL`` is unreachable, so the agent pipeline can keep running
without a durable audit trail), the stores built on this module hold state the system
cannot correctly operate without — the paper wallet, daily risk limits, and signal
history. There is intentionally no local-file or in-memory fallback here: a reachable
Postgres database is a hard requirement, and connection failure raises loudly at startup
instead of silently losing state.
"""

from src.db.base import Base, get_engine, get_session

__all__ = ["Base", "get_engine", "get_session"]
