"""Bounded process logging for the long-running trading service."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from src.core.namespaces import MarketNamespace


class _RuntimeContextFilter(logging.Filter):
    def __init__(self, *, market: str, provider: str, runtime_id: str) -> None:
        super().__init__()
        self.market = market.strip().upper()
        self.provider = provider.strip().upper()
        self.runtime_id = runtime_id.strip()

    def filter(self, record: logging.LogRecord) -> bool:
        record.market = self.market
        record.provider = self.provider
        record.runtime_id = self.runtime_id
        return True


def configure_runtime_logging(
    path: str | Path | None = None,
    *,
    market: str = "NSE",
    provider: str = "DHAN",
    runtime_id: str = "backend",
    max_bytes: int = 15 * 1024 * 1024,
    backup_count: int = 5,
) -> Path:
    """Route application and dashboard activity logs to one rotating file."""
    namespace = MarketNamespace(market, provider, runtime_id)
    log_path = Path(path) if path is not None else namespace.log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s market=%(market)s provider=%(provider)s "
            "runtime=%(runtime_id)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(
        _RuntimeContextFilter(market=market, provider=provider, runtime_id=runtime_id)
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # dashboard.stats installs a non-propagating console logger at import time. Once
    # service logging is configured, fold it into the same bounded process log.
    activity = logging.getLogger("deltaquant.activity")
    for existing in list(activity.handlers):
        activity.removeHandler(existing)
        existing.close()
    activity.propagate = True
    logging.captureWarnings(True)
    return log_path.resolve()
