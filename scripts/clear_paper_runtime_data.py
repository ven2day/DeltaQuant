#!/usr/bin/env python3
"""Transactionally clear MARKET_PAPER and MOCK runtime records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.base import get_engine
from src.markets.nse.execution.runtime_mode import RuntimeExecutionMode

RUNTIME_TABLES = (
    "paper_trade_events",
    "paper_trade_lifecycles",
    "decision_logs",
    "trade_journal",
    "strategy_performance_records",
    "agent_memory",
    "signal_history",
    "daily_risk_state",
    "idempotency_keys",
    "paper_orders",
    "paper_positions",
    "paper_wallet",
)
NAMESPACES = tuple(mode.namespace for mode in RuntimeExecutionMode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete; without this flag the command is a read-only preview.",
    )
    args = parser.parse_args()
    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    eligible = []
    for table in RUNTIME_TABLES:
        if table not in existing_tables:
            continue
        columns = {column["name"] for column in inspector.get_columns(table)}
        if "namespace" in columns:
            eligible.append(table)

    placeholders = ", ".join(f":namespace_{index}" for index in range(len(NAMESPACES)))
    params = {f"namespace_{index}": value for index, value in enumerate(NAMESPACES)}
    total = 0
    with engine.begin() as connection:
        for table in eligible:
            count = int(
                connection.execute(
                    text(f'SELECT COUNT(*) FROM "{table}" WHERE namespace IN ({placeholders})'),
                    params,
                ).scalar_one()
            )
            total += count
            print(f"{table}: {count}")
            if args.confirm and count:
                connection.execute(
                    text(f'DELETE FROM "{table}" WHERE namespace IN ({placeholders})'),
                    params,
                )
    action = "deleted" if args.confirm else "would delete"
    print(f"{action}: {total} rows across {len(eligible)} namespace-aware tables")


if __name__ == "__main__":
    main()
