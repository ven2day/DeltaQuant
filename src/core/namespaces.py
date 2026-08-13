"""Canonical namespaces for caches, events, schemas, artifacts, and logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.core.models import Market

_PART = re.compile(r"^[a-zA-Z0-9_.:-]+$")


def normalize_market(value: Market | str) -> Market:
    return value if isinstance(value, Market) else Market(str(value).strip().upper())


def _safe_part(value: object) -> str:
    part = str(value).strip().lower()
    if not part or not _PART.fullmatch(part):
        raise ValueError(f"Unsafe namespace component: {value!r}")
    return part


@dataclass(frozen=True)
class MarketNamespace:
    """One market's collision-proof operational namespace."""

    market: Market | str
    provider: str
    runtime_id: str

    @property
    def market_value(self) -> str:
        return normalize_market(self.market).value.lower()

    @property
    def schema(self) -> str:
        return self.market_value

    def cache_key(self, category: str, *parts: object) -> str:
        values = [self.market_value, _safe_part(category), *(_safe_part(part) for part in parts)]
        return ":".join(values)

    def event_topic(self, topic: str) -> str:
        return f"{self.market_value}.{_safe_part(topic)}"

    def artifact_root(self, root: str | Path = "artifacts") -> Path:
        return Path(root) / self.market_value

    def log_path(self, root: str | Path = "logs") -> Path:
        return Path(root) / self.market_value / f"{_safe_part(self.runtime_id)}.log"
