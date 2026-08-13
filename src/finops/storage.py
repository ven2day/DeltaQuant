"""Durable, cross-process FinOps event storage.

Only structured attribution and token/cost counters are stored. Prompts, responses,
credentials and broker configuration never enter this database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FinOpsEventStore:
    """Small WAL-mode SQLite store shared safely by independent market workers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_usage_events (
                    call_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    date_ist TEXT NOT NULL,
                    market TEXT NOT NULL,
                    call_reason TEXT NOT NULL,
                    component TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    call_count INTEGER NOT NULL DEFAULT 1,
                    trade_horizon TEXT NOT NULL,
                    runtime_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS ix_llm_usage_date_market_reason_component_model
                ON llm_usage_events(date_ist, market, call_reason, component, model);
                CREATE INDEX IF NOT EXISTS ix_llm_usage_session
                ON llm_usage_events(session_id, timestamp);
                CREATE INDEX IF NOT EXISTS ix_llm_usage_cycle
                ON llm_usage_events(cycle_id, timestamp);
                """
            )

    def upsert(self, payload: dict[str, Any]) -> None:
        columns = (
            "call_id",
            "timestamp",
            "date_ist",
            "market",
            "call_reason",
            "component",
            "agent",
            "model",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "call_count",
            "trade_horizon",
            "runtime_id",
            "session_id",
            "cycle_id",
            "candidate_id",
            "symbol",
            "timeframe",
            "trace_id",
            "cache_hit",
            "metadata_json",
        )
        values = [payload.get(column, "") for column in columns]
        values[-1] = json.dumps(payload.get("metadata", {}), sort_keys=True, separators=(",", ":"))
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column != "call_id")
        statement = (
            f"INSERT INTO llm_usage_events ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(call_id) DO UPDATE SET {updates}"
        )
        for attempt in range(4):
            try:
                with self._connect() as connection:
                    connection.execute(statement, values)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 3:
                    raise
                time.sleep(0.025 * (2**attempt))

    def records(
        self,
        *,
        date_ist: str | None = None,
        market: str | None = None,
        session_id: str | None = None,
        cycle_id: str | None = None,
        since: str | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("date_ist", date_ist),
            ("market", market),
            ("session_id", session_id),
            ("cycle_id", cycle_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if since:
            clauses.append("timestamp >= ?")
            parameters.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(1, min(int(limit), 100_000))
        query = f"SELECT * FROM llm_usage_events{where} ORDER BY timestamp ASC LIMIT ?"
        parameters.append(safe_limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["cache_hit"] = bool(item.get("cache_hit"))
            try:
                item["metadata"] = json.loads(str(item.pop("metadata_json", "{}")))
            except json.JSONDecodeError:
                item["metadata"] = {}
            output.append(item)
        return output
