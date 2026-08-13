"""Dry-run/apply the non-destructive common model-registry migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_market_schemas import _database_url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="")
    parser.add_argument("--market", choices=("NSE", "FOREX"), default="NSE")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    database_url, env_files = _database_url(args.database_url, args.market)
    endpoint = make_url(database_url)
    engine = create_engine(database_url, pool_pre_ping=True)
    before = "model_registry" in inspect(engine).get_table_names(schema="common")
    report: dict[str, object] = {
        "connection": {
            "host": endpoint.host,
            "port": endpoint.port or 5432,
            "database": endpoint.database,
            "username": endpoint.username,
            "password_present": endpoint.password is not None,
            "active_env_sources": [str(Path(path).relative_to(ROOT)) for path in env_files],
        },
        "migration": "migrations/003_common_model_registry.sql",
        "destructive_actions": 0,
        "table_present_before": before,
        "mode": "apply" if args.apply else "dry_run",
    }
    if args.apply:
        sql = (ROOT / "migrations/003_common_model_registry.sql").read_text(encoding="utf-8")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.exec_driver_sql(sql)
        inspector = inspect(engine)
        indexes = inspector.get_indexes("model_registry", schema="common")
        report["table_present_after"] = "model_registry" in inspector.get_table_names(
            schema="common"
        )
        report["indexes"] = sorted(str(item["name"]) for item in indexes)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
