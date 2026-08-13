"""Small secret-safe read models shared between independent workers and the API."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from src.config.secrets import redact_secrets
from src.core.namespaces import normalize_market
from src.markets.api_registry import MarketApiView

logger = logging.getLogger(__name__)


class MarketSnapshotStore:
    """Atomic per-market operational snapshots; never execution authority."""

    def __init__(
        self,
        root: str | Path = "run",
        *,
        min_write_seconds: float = 2.0,
        stale_after_seconds: float = 180.0,
        writer_runtime_id: str | None = None,
        writable: bool = True,
        replace_attempts: int = 5,
        retry_base_seconds: float = 0.05,
    ) -> None:
        self.root = Path(root)
        self.min_write_seconds = max(0.0, min_write_seconds)
        self.stale_after_seconds = max(1.0, stale_after_seconds)
        self.writer_runtime_id = str(writer_runtime_id or "").strip()
        self.writable = writable
        self.replace_attempts = max(1, replace_attempts)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self._last_write: dict[str, float] = {}
        self._write_failures: dict[str, int] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._lock = threading.RLock()

    def _path(self, market: str) -> Path:
        return Path(
            self.root / normalize_market(market).value.lower() / "market_snapshot.json"
        )

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_base_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay + random.uniform(0.0, delay * 0.1))

    @contextmanager
    def _publication_lock(self, target: Path) -> Iterator[None]:
        """Serialize writers across processes without leaving a stale-owner lock."""

        lock_path = target.with_suffix(".writer.lock")
        handle = lock_path.open("a+b")
        acquired = False
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            for attempt in range(self.replace_attempts):
                try:
                    self._lock_file(handle)
                    acquired = True
                    break
                except (OSError, PermissionError):
                    if attempt + 1 >= self.replace_attempts:
                        raise
                    self._backoff(attempt)
            if not acquired:
                raise PermissionError(f"Could not acquire snapshot writer lock for {target.name}")
            yield
        finally:
            if acquired:
                self._unlock_file(handle)
            handle.close()

    @staticmethod
    def _lock_file(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl_module: Any = __import__("fcntl")
            fcntl_module.flock(
                handle.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB
            )

    @staticmethod
    def _unlock_file(handle: BinaryIO) -> None:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl_module: Any = __import__("fcntl")
                fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_UN)
        except OSError:
            logger.warning("Snapshot writer lock release failed", exc_info=True)

    def metrics(self, market: str) -> dict[str, int]:
        market_name = normalize_market(market).value
        with self._lock:
            return {
                "write_failures": self._write_failures.get(market_name, 0),
                "consecutive_failures": self._consecutive_failures.get(market_name, 0),
            }

    def publish(
        self,
        market: str,
        *,
        status: dict[str, Any],
        signals: list[dict[str, Any]],
        positions: list[dict[str, Any]],
        technical_candidates: list[dict[str, Any]] | None = None,
        strategies: list[dict[str, Any]] | None = None,
        models: list[dict[str, Any]] | None = None,
        dashboard_state: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        market_name = normalize_market(market).value
        if not self.writable:
            raise RuntimeError("Read-only MarketSnapshotStore cannot publish snapshots")
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_write.get(market_name, 0.0) < self.min_write_seconds:
                return False
            generated_at = datetime.now(UTC).isoformat()
            runtime_id = self.writer_runtime_id or str(status.get("runtime_id", "")).strip()
            payload = redact_secrets(
                {
                    "market": market_name,
                    "generated_at": generated_at,
                    "writer_runtime_id": runtime_id,
                    "health_state": "HEALTHY",
                    "status": {
                        **status,
                        "market": market_name,
                        "generated_at": generated_at,
                        "writer_runtime_id": runtime_id,
                        "snapshot_publication": "HEALTHY",
                        "snapshot_write_failures": self._write_failures.get(market_name, 0),
                    },
                    "signals": [{**item, "market": market_name} for item in signals],
                    "technical_candidates": [
                        {**item, "market": market_name}
                        for item in (technical_candidates or [])
                    ],
                    "positions": [{**item, "market": market_name} for item in positions],
                    "strategies": [
                        {**item, "market": market_name} for item in (strategies or [])
                    ],
                    "models": [{**item, "market": market_name} for item in (models or [])],
                    # Read-only UI state for the independently running API.  It is
                    # deliberately stored beside, but never consumed by, trading
                    # or execution code.  Secret redaction applies recursively.
                    "dashboard_state": dict(dashboard_state or {}),
                }
            )
            target = self._path(market_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with self._publication_lock(target):
                    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                    with temporary.open("xb") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    for attempt in range(self.replace_attempts):
                        try:
                            os.replace(temporary, target)
                            break
                        except PermissionError:
                            if attempt + 1 >= self.replace_attempts:
                                raise
                            self._backoff(attempt)
                self._last_write[market_name] = now
                self._consecutive_failures[market_name] = 0
                return True
            except OSError as exc:
                self._write_failures[market_name] = self._write_failures.get(market_name, 0) + 1
                self._consecutive_failures[market_name] = (
                    self._consecutive_failures.get(market_name, 0) + 1
                )
                logger.error(
                    "Market snapshot publication degraded: market=%s error=%s failures=%d",
                    market_name,
                    type(exc).__name__,
                    self._write_failures[market_name],
                    exc_info=True,
                )
                return False
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Unable to remove abandoned snapshot temp file", exc_info=True)

    def _unavailable(self, market_name: str, health_state: str = "UNAVAILABLE") -> dict[str, Any]:
        return {
            "market": market_name,
            "generated_at": None,
            "age_seconds": None,
            "writer_runtime_id": "",
            "health_state": health_state,
            "status": {
                "market": market_name,
                "status": health_state,
                "health_state": health_state,
                "snapshot_publication": health_state,
                "dashboard_connectivity": "DISCONNECTED",
            },
            "signals": [],
            "technical_candidates": [],
            "positions": [],
            "strategies": [],
            "models": [],
            "dashboard_state": {},
        }

    def read(self, market: str) -> dict[str, Any]:
        market_name = normalize_market(market).value
        target = self._path(market_name)
        if not target.exists():
            return self._unavailable(market_name)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Market snapshot read failed: market=%s", market_name, exc_info=True)
            return self._unavailable(market_name, "UNHEALTHY")
        if str(raw.get("market", "")).upper() != market_name:
            raise ValueError("Market snapshot identity mismatch")
        generated_at = str(raw.get("generated_at", "") or "")
        try:
            generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=UTC)
            age_seconds = max(0.0, (datetime.now(UTC) - generated.astimezone(UTC)).total_seconds())
        except ValueError:
            age_seconds = float("inf")
        stale = age_seconds > self.stale_after_seconds
        result = dict(raw)
        result["age_seconds"] = age_seconds
        result["health_state"] = "STALE" if stale else "HEALTHY"
        status = dict(result.get("status", {}))
        status.update(
            {
                "market": market_name,
                "generated_at": generated_at or None,
                "age_seconds": age_seconds,
                "writer_runtime_id": str(result.get("writer_runtime_id", "")),
                "health_state": "STALE" if stale else "HEALTHY",
                "snapshot_publication": "STALE" if stale else "HEALTHY",
                "dashboard_connectivity": "DISCONNECTED" if stale else "CONNECTED",
            }
        )
        if stale:
            status["runtime_last_known_status"] = status.get("status", "UNKNOWN")
            status["status"] = "STALE"
        result["status"] = status
        return result

    def api_view(self, market: str) -> MarketApiView:
        market_name = normalize_market(market).value
        return MarketApiView(
            market=market_name,
            status=lambda: dict(self.read(market_name)["status"]),
            signals=lambda days: list(self.read(market_name)["signals"]),
            positions=lambda: list(self.read(market_name)["positions"]),
            candidates=lambda: list(
                self.read(market_name).get("technical_candidates", [])
            ),
            strategies=lambda: list(self.read(market_name).get("strategies", [])),
            models=lambda: list(self.read(market_name).get("models", [])),
        )
