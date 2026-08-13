"""Market-owned YAML configuration bundles with no credential handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.models import Market
from src.core.namespaces import normalize_market


@dataclass(frozen=True)
class MarketConfigBundle:
    market: Market
    root: Path
    strategies: dict[str, Any]
    risk: dict[str, Any]
    universe: dict[str, Any]
    runtime: dict[str, Any]
    sessions: dict[str, Any]


def _read_yaml(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing market configuration: {path}")
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Market configuration must be a mapping: {path}")
    return value


def load_market_config(
    market: Market | str,
    *,
    root: str | Path | None = None,
) -> MarketConfigBundle:
    selected = normalize_market(market)
    project = Path(__file__).resolve().parents[2]
    base = Path(root) if root is not None else project / "config"
    market_root = base / selected.value.lower()
    return MarketConfigBundle(
        market=selected,
        root=market_root,
        strategies=_read_yaml(market_root / "strategies.yaml"),
        risk=_read_yaml(market_root / "risk.yaml"),
        universe=_read_yaml(market_root / "universe.yaml"),
        runtime=_read_yaml(market_root / "runtime.yaml"),
        sessions=_read_yaml(market_root / "sessions.yaml", required=False),
    )
