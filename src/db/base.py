"""Shared SQLAlchemy engine/session factory for the Postgres-backed stores.

One engine (one connection pool) is shared across every store in this package rather
than each opening its own — important for a serverless Postgres provider like Neon,
which caps concurrent connections per project.
"""

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import get_settings

logger = logging.getLogger(__name__)

# create_all() only creates missing TABLES — it never adds a missing COLUMN to a table
# that already exists. This project has no migration tool (Alembic etc.), so each column
# added to a model after its table's first release needs a one-time, idempotent ADD
# COLUMN here. Best-effort: the column may already exist (fresh DB, or already migrated)
# or the DB role may lack ALTER privileges, and neither should block startup.
_COLUMN_MIGRATIONS = (
    ("paper_orders", "reason", "reason VARCHAR(20)"),
    ("paper_orders", "trade_id", "trade_id VARCHAR(60)"),
    ("paper_orders", "position_id", "position_id VARCHAR(60)"),
    ("paper_orders", "idempotency_key", "idempotency_key VARCHAR(200)"),
    ("paper_orders", "realized_pnl", "realized_pnl DOUBLE PRECISION DEFAULT 0"),
    ("daily_risk_state", "exits_count", "exits_count INTEGER DEFAULT 0"),
    (
        "strategy_performance_records",
        "namespace",
        "namespace VARCHAR(40) DEFAULT 'paper_market_data'",
    ),
    ("strategy_performance_records", "trade_id", "trade_id VARCHAR(60) DEFAULT ''"),
    ("paper_trade_lifecycles", "finalized_at", "finalized_at TIMESTAMP WITH TIME ZONE"),
    ("paper_trade_lifecycles", "mae", "mae DOUBLE PRECISION DEFAULT 0"),
    ("paper_trade_lifecycles", "mfe", "mfe DOUBLE PRECISION DEFAULT 0"),
)


def _apply_column_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    known_columns: dict[str, set[str]] = {}
    for table_name, column_name, definition in _COLUMN_MIGRATIONS:
        try:
            columns = known_columns.setdefault(
                table_name,
                {column["name"] for column in inspector.get_columns(table_name)},
            )
            if column_name in columns:
                continue
            statement = f"ALTER TABLE {table_name} ADD COLUMN {definition}"
            with engine.begin() as conn:
                conn.execute(text(statement))
            columns.add(column_name)
        except Exception:
            logger.debug(
                "Column migration skipped (already applied or unsupported): %s.%s",
                table_name,
                column_name,
            )

Base = declarative_base()

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
# Engines for explicit (non-default) database_url overrides — e.g. sqlite:///:memory:
# in tests — keyed by URL so repeated calls with the SAME URL share one engine/pool.
# Without this cache, an in-memory sqlite URL would get a brand-new, empty database on
# every single call (each create_engine() to ":memory:" is a wholly separate database),
# so nothing would ever appear to persist across two calls even within one test.
_override_engines: dict[str, Engine] = {}
_override_session_factories: dict[str, sessionmaker[Session]] = {}


def get_engine(database_url: str | None = None) -> Engine:
    """Return the shared engine, creating it (and every registered table) on first use.

    Pass an explicit ``database_url`` (e.g. ``sqlite:///:memory:`` in tests) to get an
    independent, isolated engine instead of the shared default — cached per URL so
    repeated calls with the same override URL reuse the same underlying database.
    """
    global _engine
    if database_url is not None:
        engine = _override_engines.get(database_url)
        if engine is None:
            engine = create_engine(database_url, pool_pre_ping=True)
            Base.metadata.create_all(engine)
            _apply_column_migrations(engine)
            _override_engines[database_url] = engine
        return engine

    if _engine is None:
        settings = get_settings()
        # pool_pre_ping: Neon (and most serverless Postgres) silently drops idle
        # connections; without this, the first query after a lull fails with
        # "server closed the connection unexpectedly" instead of transparently
        # reconnecting. pool_recycle: proactively refresh connections older than
        # this many seconds — a self-hosted Postgres reached over the public
        # internet (e.g. a VPS) can have a connection dropped by an intermediate
        # NAT/firewall's idle-connection timeout well before Postgres itself would
        # close it, so pre_ping alone (which only catches an already-dead
        # connection at checkout) isn't enough.
        _engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=280)
        Base.metadata.create_all(_engine)
        _apply_column_migrations(_engine)
        logger.info("Postgres persistence engine initialized: %s", _engine.url)
    return _engine


def get_session(database_url: str | None = None) -> Session:
    """Return a new session bound to the shared engine (or a cached isolated one, for tests)."""
    global _session_factory
    if database_url is not None:
        factory = _override_session_factories.get(database_url)
        if factory is None:
            factory = sessionmaker(bind=get_engine(database_url))
            _override_session_factories[database_url] = factory
        return factory()

    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()
