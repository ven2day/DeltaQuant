"""Back up legacy market tables and apply logical PostgreSQL schema isolation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEGACY_TABLES = (
    "market_candles",
    "signal_history",
    "decision_logs",
    "paper_positions",
    "paper_orders",
)


def _database_url(explicit: str, market: str) -> tuple[str, tuple[str, ...]]:
    """Resolve the same market-selected dotenv chain used by the application."""

    from src.config.env_loader import resolve_env_files

    selected_environment = dict(os.environ)
    selected_environment["MARKET"] = market.strip().upper()
    files = resolve_env_files(ROOT, selected_environment)
    values: dict[str, str] = {}
    for path in files:
        values.update(
            {
                str(key): str(value)
                for key, value in dotenv_values(path).items()
                if value is not None
            }
        )
    values.update(os.environ)
    value = explicit or values.get("MARKET_HISTORY_DATABASE_URL", "") or values.get(
        "DATABASE_URL", ""
    )
    if not value:
        raise SystemExit(
            "DATABASE_URL or MARKET_HISTORY_DATABASE_URL is required via the selected "
            "market environment, process environment, or --database-url"
        )
    return value, files


def _row_counts(engine: Engine) -> dict[str, int]:
    inspector = inspect(engine)
    present = set(inspector.get_table_names(schema="public"))
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for table in LEGACY_TABLES:
            if table in present:
                counts[table] = int(
                    connection.execute(text(f'SELECT count(*) FROM public."{table}"')).scalar_one()
                )
    return counts


def _backup(engine: Engine) -> str:
    suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    schema = f"migration_backup_{suffix}"
    if not re.fullmatch(r"[a-z0-9_]+", schema):
        raise RuntimeError("Unsafe generated backup schema")
    present = set(inspect(engine).get_table_names(schema="public"))
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        for table in LEGACY_TABLES:
            if table in present:
                connection.execute(
                    text(
                        f'CREATE TABLE "{schema}"."{table}" AS '
                        f'SELECT * FROM public."{table}"'
                    )
                )
    return schema


def _migration_plan(engine: Engine) -> dict[str, object]:
    inspector = inspect(engine)
    public_tables = set(inspector.get_table_names(schema="public"))
    schemas = set(inspector.get_schema_names())
    market_counts: dict[str, dict[str, int]] = {
        name: {} for name in ("nse", "forex", "crypto")
    }
    with engine.connect() as connection:
        if "market_candles" in public_tables:
            columns = {
                item["name"]
                for item in inspector.get_columns("market_candles", schema="public")
            }
            if "market" in columns:
                rows = connection.execute(
                    text(
                        "SELECT upper(market) AS market, count(*) AS rows "
                        "FROM public.market_candles GROUP BY upper(market)"
                    )
                )
                for row in rows:
                    schema = str(row.market).lower()
                    if schema in market_counts:
                        market_counts[schema]["market_candles"] = int(row.rows)
    return {
        "schemas_to_create": [
            name for name in ("common", "nse", "forex", "crypto") if name not in schemas
        ],
        "legacy_public_tables": sorted(public_tables.intersection(LEGACY_TABLES)),
        "market_rows_to_copy": market_counts,
        "legacy_tables_retained": True,
        "backup_required": True,
        "timescale_hypertables_planned": [
            "nse.market_candles",
            "forex.market_candles",
            "crypto.market_candles",
        ],
        "migration_sql": "migrations/001_market_schema_isolation.sql",
    }


def _verification(engine: Engine) -> dict[str, object]:
    inspector = inspect(engine)
    expected_tables = {
        "market_candles",
        "signals",
        "decisions",
        "positions",
        "orders",
        "strategy_registry",
        "ml_predictions",
    }
    schemas: dict[str, object] = {}
    with engine.connect() as connection:
        hypertables = {
            f"{row.hypertable_schema}.{row.hypertable_name}"
            for row in connection.execute(
                text(
                    "SELECT hypertable_schema, hypertable_name "
                    "FROM timescaledb_information.hypertables"
                )
            )
        }
        for schema in ("nse", "forex", "crypto"):
            tables = set(inspector.get_table_names(schema=schema))
            indexes = {
                str(item["name"])
                for table in tables
                for item in inspector.get_indexes(table, schema=schema)
                if item.get("name")
            }
            counts: dict[str, int] = {}
            for table in sorted(expected_tables.intersection(tables)):
                counts[table] = int(
                    connection.execute(
                        text(f'SELECT count(*) FROM "{schema}"."{table}"')
                    ).scalar_one()
                )
            schemas[schema] = {
                "tables_present": sorted(tables),
                "missing_tables": sorted(expected_tables - tables),
                "indexes": sorted(indexes),
                "market_candles_hypertable": f"{schema}.market_candles" in hypertables,
                "row_counts": counts,
                "schema_usage": bool(
                    connection.execute(
                        text("SELECT has_schema_privilege(current_user, :schema, 'USAGE')"),
                        {"schema": schema},
                    ).scalar_one()
                ),
            }
    schemas["common"] = {
        "tables_present": sorted(inspector.get_table_names(schema="common")),
        "schema_usage": True,
    }
    return schemas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="")
    parser.add_argument("--market", default=os.environ.get("MARKET", "NSE"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    database_url, env_files = _database_url(args.database_url, args.market)
    endpoint = make_url(database_url)
    engine = create_engine(database_url, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        raise SystemExit("Market schema migration requires PostgreSQL/TimescaleDB")
    before = _row_counts(engine)
    print(
        json.dumps(
            {
                "connection": {
                    "host": endpoint.host,
                    "port": endpoint.port or 5432,
                    "database": endpoint.database,
                    "username": endpoint.username,
                    "password_present": endpoint.password is not None,
                    "active_env_sources": [
                        str(Path(path).relative_to(ROOT)) for path in env_files
                    ],
                },
                "legacy_row_counts": before,
                "plan": _migration_plan(engine),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.apply:
        print("Dry run only. Re-run with --apply to create a backup and migrate.")
        return 0
    backup_schema = _backup(engine)
    sql = (ROOT / "migrations" / "001_market_schema_isolation.sql").read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute(sql)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    print(f"Migration applied. Recoverable backup schema: {backup_schema}")
    print(
        json.dumps(
            {
                "legacy_public_tables_retained": _row_counts(engine),
                "verification": _verification(engine),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
