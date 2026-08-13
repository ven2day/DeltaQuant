"""Durable OHLCV storage for research, ML, and live signal generation.

The schema is ordinary PostgreSQL/SQLAlchemy first, so it remains portable in tests
and on hosts where the TimescaleDB extension is unavailable.  When TimescaleDB is
installed, :class:`CandleStore` converts ``market_candles`` into a hypertable without
changing the repository API.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    String,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "25m": 25,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 24 * 60,
}


class MarketHistoryBase(DeclarativeBase):
    """Isolated metadata so candle DDL never leaks into the trading-state database."""


class MarketCandle(MarketHistoryBase):
    """One normalized candle; the composite key makes ingestion idempotent."""

    __tablename__ = "market_candles"

    market: Mapped[str] = mapped_column(String(16), primary_key=True, default="NSE")
    provider: Mapped[str] = mapped_column(String(24), primary_key=True, default="DHAN")
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(8), primary_key=True)
    candle_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String(24), nullable=False, default="dhan")
    volume_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="EXCHANGE_UNITS"
    )
    is_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


Index(
    "ix_market_candles_symbol_timeframe_time",
    MarketCandle.market,
    MarketCandle.provider,
    MarketCandle.symbol,
    MarketCandle.timeframe,
    MarketCandle.candle_time.desc(),
)


@dataclass(frozen=True)
class CandleCoverage:
    symbol: str
    timeframe: str
    first_candle: datetime | None
    last_candle: datetime | None
    rows: int


def _utc_datetime(value: Any, *, naive_timezone: str = "Asia/Kolkata") -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        # Broker frames use exchange-local wall time. Treat other naive inputs the
        # same way so tests, CSV imports, and Dhan data have one deterministic contract.
        timestamp = timestamp.tz_localize(naive_timezone)
    return timestamp.tz_convert("UTC").to_pydatetime()


def _frame_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]
    if not set(OHLCV_COLUMNS).issubset(normalized.columns):
        missing = sorted(set(OHLCV_COLUMNS) - set(normalized.columns))
        raise ValueError(f"OHLCV frame is missing columns: {', '.join(missing)}")
    normalized = normalized[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=list(OHLCV_COLUMNS)).sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


class CandleStore:
    """Read/write repository backed by PostgreSQL, TimescaleDB, or SQLite."""

    def __init__(
        self,
        database_url: str,
        *,
        enable_timescale: bool = True,
        require_timescale: bool = False,
        engine: Engine | None = None,
        schema: str | None = None,
    ) -> None:
        self.database_url = database_url
        base_engine = engine or create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=280,
        )
        normalized_schema = (schema or "").strip().lower()
        if normalized_schema and not re.fullmatch(r"[a-z][a-z0-9_]*", normalized_schema):
            raise ValueError(f"Unsafe market-history schema: {schema!r}")
        self.schema = normalized_schema if base_engine.dialect.name == "postgresql" else ""
        if self.schema:
            with base_engine.begin() as connection:
                connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
            self.engine = base_engine.execution_options(
                schema_translate_map={None: self.schema}
            )
        else:
            self.engine = base_engine
        self._session_factory = sessionmaker(bind=self.engine)
        MarketCandle.__table__.create(self.engine, checkfirst=True)
        self._ensure_market_identity_columns()
        self.timescale_enabled = False
        if self.engine.dialect.name == "postgresql" and enable_timescale:
            self.timescale_enabled = self._enable_hypertable()
        if require_timescale and not self.timescale_enabled:
            raise RuntimeError(
                "MARKET_HISTORY_REQUIRE_TIMESCALE=true but the TimescaleDB extension "
                "is not installed or market_candles could not be converted"
            )

    def _ensure_market_identity_columns(self) -> None:
        """Upgrade the legacy NSE-only table before any ORM query uses new columns."""

        if self.engine.dialect.name != "postgresql":
            return
        table = f'"{self.schema}".market_candles' if self.schema else "market_candles"
        statements = (
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS market VARCHAR(16) DEFAULT 'NSE'",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS provider VARCHAR(24) DEFAULT 'DHAN'",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS volume_type VARCHAR(32) DEFAULT 'EXCHANGE_UNITS'",
            f"UPDATE {table} SET market='NSE' WHERE market IS NULL OR market=''",
            f"UPDATE {table} SET provider=UPPER(COALESCE(NULLIF(source,''),'DHAN')) WHERE provider IS NULL OR provider=''",
            f"UPDATE {table} SET volume_type='EXCHANGE_UNITS' WHERE volume_type IS NULL OR volume_type=''",
            f"ALTER TABLE {table} ALTER COLUMN market SET NOT NULL",
            f"ALTER TABLE {table} ALTER COLUMN provider SET NOT NULL",
            f"ALTER TABLE {table} ALTER COLUMN volume_type SET NOT NULL",
        )
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            # PostgreSQL creates an index in the table's schema; the index name in
            # CREATE INDEX itself must not be separately schema-qualified.
            index = "uq_market_candles_market_provider_grain"
            connection.execute(text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {table}"
                "(market, provider, symbol, timeframe, candle_time)"
            ))

    def _enable_hypertable(self) -> bool:
        try:
            with self.engine.connect() as connection:
                installed = bool(
                    connection.execute(
                        text(
                            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='timescaledb')"
                        )
                    ).scalar()
                )
            if not installed:
                logger.info(
                    "TimescaleDB extension not present; market_candles uses regular PostgreSQL"
                )
                return False

            # TimescaleDB 2.20+ uses by_range(); older supported releases use the
            # positional time-column API. Keep both so local installations can upgrade
            # independently of this application.
            hypertable = f"{self.schema}.market_candles" if self.schema else "market_candles"
            statements = (
                f"SELECT create_hypertable('{hypertable}', by_range('candle_time'), "
                "if_not_exists => TRUE, migrate_data => TRUE)",
                f"SELECT create_hypertable('{hypertable}', 'candle_time', "
                "if_not_exists => TRUE, migrate_data => TRUE)",
            )
            for statement in statements:
                try:
                    with self.engine.begin() as connection:
                        connection.execute(text(statement))
                    logger.info("TimescaleDB hypertable ready: %s", hypertable)
                    return True
                except Exception:
                    logger.debug("Timescale hypertable syntax not supported", exc_info=True)
        except Exception:
            logger.warning(
                "Could not initialize TimescaleDB; continuing with regular PostgreSQL",
                exc_info=True,
            )
        return False

    def upsert_frame(
        self,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        *,
        source: str = "dhan",
        market: str = "NSE",
        provider: str = "DHAN",
        volume_type: str = "EXCHANGE_UNITS",
        chunk_size: int = 1000,
    ) -> int:
        """Idempotently insert/update a normalized OHLCV frame."""
        normalized = _frame_for_storage(frame)
        if normalized.empty:
            return 0

        symbol = symbol.strip().upper()
        timeframe = timeframe.strip().lower()
        duration = timedelta(minutes=_TIMEFRAME_MINUTES.get(timeframe, 0))
        now = datetime.now(UTC)
        ingested_at = now
        records: list[dict[str, Any]] = []
        for index, row in normalized.iterrows():
            candle_time = _utc_datetime(
                index,
                naive_timezone="UTC" if market.strip().upper() == "FOREX" else "Asia/Kolkata",
            )
            records.append(
                {
                    "market": market.strip().upper(),
                    "provider": provider.strip().upper(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candle_time": candle_time,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": max(0, int(row["volume"])),
                    "source": source,
                    "volume_type": volume_type.strip().upper(),
                    "is_complete": not duration or candle_time + duration <= now,
                    "ingested_at": ingested_at,
                }
            )

        for offset in range(0, len(records), max(1, chunk_size)):
            self._upsert_records(records[offset : offset + chunk_size])
        return len(records)

    def _upsert_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            with Session(self.engine) as session, session.begin():
                for record in records:
                    session.merge(MarketCandle(**record))
            return

        statement = insert(MarketCandle).values(records)
        statement = statement.on_conflict_do_update(
            index_elements=["market", "provider", "symbol", "timeframe", "candle_time"],
            set_={
                "open": statement.excluded.open,
                "high": statement.excluded.high,
                "low": statement.excluded.low,
                "close": statement.excluded.close,
                "volume": statement.excluded.volume,
                "source": statement.excluded.source,
                "volume_type": statement.excluded.volume_type,
                "is_complete": statement.excluded.is_complete,
                "ingested_at": statement.excluded.ingested_at,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)

    def load_frame(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        complete_only: bool = False,
        market: str = "NSE",
        provider: str = "DHAN",
    ) -> pd.DataFrame | None:
        """Load candles in ascending exchange time, optionally limiting newest rows."""
        query = select(
            MarketCandle.candle_time,
            MarketCandle.open,
            MarketCandle.high,
            MarketCandle.low,
            MarketCandle.close,
            MarketCandle.volume,
        ).where(
            MarketCandle.market == market.strip().upper(),
            MarketCandle.provider == provider.strip().upper(),
            MarketCandle.symbol == symbol.strip().upper(),
            MarketCandle.timeframe == timeframe.strip().lower(),
        )
        if start is not None:
            query = query.where(MarketCandle.candle_time >= _utc_datetime(start, naive_timezone="UTC" if market.upper() == "FOREX" else "Asia/Kolkata"))
        if end is not None:
            query = query.where(MarketCandle.candle_time <= _utc_datetime(end, naive_timezone="UTC" if market.upper() == "FOREX" else "Asia/Kolkata"))
        if complete_only:
            query = query.where(MarketCandle.is_complete.is_(True))
        query = query.order_by(MarketCandle.candle_time.desc())
        if bars is not None:
            query = query.limit(max(1, bars))

        with self._session_factory() as session:
            rows = list(session.execute(query).all())
        if not rows:
            return None
        rows.reverse()
        output_timezone = "UTC" if market.strip().upper() == "FOREX" else "Asia/Kolkata"
        index = pd.to_datetime([row.candle_time for row in rows], utc=True).tz_convert(output_timezone)
        frame = pd.DataFrame(
            {
                "open": [float(row.open) for row in rows],
                "high": [float(row.high) for row in rows],
                "low": [float(row.low) for row in rows],
                "close": [float(row.close) for row in rows],
                "volume": [int(row.volume) for row in rows],
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    def coverage(
        self,
        symbol: str,
        timeframe: str,
        *,
        market: str = "NSE",
        provider: str = "DHAN",
    ) -> CandleCoverage:
        query = select(
            func.min(MarketCandle.candle_time),
            func.max(MarketCandle.candle_time),
            func.count(),
        ).where(
            MarketCandle.market == market.strip().upper(),
            MarketCandle.provider == provider.strip().upper(),
            MarketCandle.symbol == symbol.strip().upper(),
            MarketCandle.timeframe == timeframe.strip().lower(),
        )
        with self._session_factory() as session:
            first_candle, last_candle, rows = session.execute(query).one()
        return CandleCoverage(
            symbol=symbol.strip().upper(),
            timeframe=timeframe.strip().lower(),
            first_candle=first_candle,
            last_candle=last_candle,
            rows=int(rows or 0),
        )

    def load_analysis_frame(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int | None = None,
        complete_only: bool = True,
        market: str = "NSE",
        provider: str = "DHAN",
    ) -> pd.DataFrame:
        """Load a native analysis frame without applying market-specific transforms.

        Exchange/session-aware derivation belongs to the market repository binding.
        This shared store deliberately has no NSE, Forex, Dhan, or OANDA dependency.
        """

        frame = self.load_frame(
            symbol,
            timeframe,
            bars=bars,
            complete_only=complete_only,
            market=market,
            provider=provider,
        )
        if frame is None:
            return pd.DataFrame()
        return frame
