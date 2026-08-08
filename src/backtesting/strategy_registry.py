"""
Strategy admission registry (H-8, DeltaQuant-Quant-Risk-Review.md).

``walk_forward.edge_verdict()`` is a solid offline go/no-go, but before this module it was
consulted only by ``scripts/validate_strategy.py`` -- an operator had to remember to run it,
read the printed verdict, and manually decide whether to trust the live agents. Nothing stopped
a strategy from being selected and traded live-paper with no current validation artifact at all.

This module turns that offline verdict into a durable, versioned, fail-closed runtime gate:

- ``build_strategy_version`` wraps ``edge_verdict()`` -- it is the ONLY function that decides
  VALIDATED/NOT VALIDATED; this module never re-derives or loosens that criteria.
- ``StrategyVersion`` is an immutable record of one such verdict: owner, parameters, approved
  universe/regimes, dataset id, validation date, and an expiry after which it must be re-earned.
- ``StrategyRegistry`` persists versions to disk (atomic write, same pattern as
  ``signal_discovery.store.DiscoveredSignalStore``) and answers the one question that matters at
  runtime: is this strategy name currently admitted to trade? ``is_admitted`` / ``filter_admitted``
  fail closed -- an unknown strategy, a NOT VALIDATED verdict, or an expired version are all
  treated identically: not admitted. There is no "warn but allow" path.

Populate the registry by running ``scripts/validate_strategy.py``, which now registers a
version per named strategy (momentum/mean_reversion/breakout/trend_following) in addition to
printing the universe-level report.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.backtesting.walk_forward import edge_verdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyVersion:
    """An immutable, versioned validation artifact for one named strategy.

    Sourced entirely from ``edge_verdict()``'s own output -- never a parallel validation
    mechanism. ``status`` is re-derived on every read (never a cached flag that could go
    stale), so a version quietly crossing its expiry is reflected immediately without any
    background job.
    """

    strategy_name: str
    version: str
    owner: str
    parameters: dict[str, Any]
    approved_universe: tuple[str, ...]
    approved_regimes: tuple[str, ...]
    dataset_id: str
    validated_at: str  # ISO-8601 timestamp of the walk-forward run
    expires_at: str  # ISO-8601 timestamp; status flips to EXPIRED after this
    verdict: str  # "VALIDATED" | "NOT VALIDATED" -- edge_verdict()'s own verdict string
    reasons: tuple[str, ...]
    oos_trades: int
    oos_expectancy: float
    oos_return_pct: float
    fold_consistency: float

    @property
    def status(self) -> str:
        """Current admission status: VALIDATED, NOT_VALIDATED, or EXPIRED."""
        if self.verdict != "VALIDATED":
            return "NOT_VALIDATED"
        if datetime.now(UTC) >= datetime.fromisoformat(self.expires_at):
            return "EXPIRED"
        return "VALIDATED"

    def is_admissible(self, *, regime: str | None = None, symbol: str | None = None) -> bool:
        """Fail-closed admissibility check for one candidate trade context."""
        if self.status != "VALIDATED":
            return False
        if regime is not None and self.approved_regimes and regime not in self.approved_regimes:
            return False
        if symbol is not None and self.approved_universe and symbol not in self.approved_universe:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "version": self.version,
            "owner": self.owner,
            "parameters": self.parameters,
            "approved_universe": list(self.approved_universe),
            "approved_regimes": list(self.approved_regimes),
            "dataset_id": self.dataset_id,
            "validated_at": self.validated_at,
            "expires_at": self.expires_at,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "oos_trades": self.oos_trades,
            "oos_expectancy": self.oos_expectancy,
            "oos_return_pct": self.oos_return_pct,
            "fold_consistency": self.fold_consistency,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyVersion:
        return cls(
            strategy_name=str(d["strategy_name"]),
            version=str(d["version"]),
            owner=str(d.get("owner", "unknown")),
            parameters=dict(d.get("parameters", {})),
            approved_universe=tuple(d.get("approved_universe", ())),
            approved_regimes=tuple(d.get("approved_regimes", ())),
            dataset_id=str(d.get("dataset_id", "")),
            validated_at=str(d["validated_at"]),
            expires_at=str(d["expires_at"]),
            verdict=str(d["verdict"]),
            reasons=tuple(d.get("reasons", ())),
            oos_trades=int(d.get("oos_trades", 0)),
            oos_expectancy=float(d.get("oos_expectancy", 0.0)),
            oos_return_pct=float(d.get("oos_return_pct", 0.0)),
            fold_consistency=float(d.get("fold_consistency", 0.0)),
        )


def build_strategy_version(
    strategy_name: str,
    *,
    owner: str,
    parameters: dict[str, Any] | None = None,
    approved_universe: list[str] | None = None,
    approved_regimes: list[str] | None = None,
    dataset_id: str,
    oos_trades: int,
    oos_expectancy: float,
    oos_return_pct: float,
    fold_consistency: float,
    validity_days: int = 30,
    version: str | None = None,
    now: datetime | None = None,
) -> StrategyVersion:
    """Build a ``StrategyVersion`` whose verdict comes directly from ``edge_verdict()``.

    This is the single entry point that produces a verdict for the registry; it never
    re-derives VALIDATED/NOT VALIDATED criteria of its own.
    """
    verdict = edge_verdict(oos_trades, oos_expectancy, oos_return_pct, fold_consistency)
    now = now or datetime.now(UTC)
    return StrategyVersion(
        strategy_name=strategy_name,
        version=version or now.strftime("%Y%m%d%H%M%S"),
        owner=owner,
        parameters=dict(parameters or {}),
        approved_universe=tuple(approved_universe or ()),
        approved_regimes=tuple(approved_regimes or ()),
        dataset_id=dataset_id,
        validated_at=now.isoformat(),
        expires_at=(now + timedelta(days=validity_days)).isoformat(),
        verdict=str(verdict["verdict"]),
        reasons=tuple(verdict["reasons"]),
        oos_trades=oos_trades,
        oos_expectancy=oos_expectancy,
        oos_return_pct=oos_return_pct,
        fold_consistency=fold_consistency,
    )


class StrategyRegistry:
    """Durable store of ``StrategyVersion`` artifacts and the runtime admission gate.

    Versions are immutable once written (a re-validation writes a NEW version file rather
    than mutating an old one -- ``register`` uses an atomic temp-file-then-replace write,
    same pattern as ``signal_discovery.store.DiscoveredSignalStore``). ``is_admitted`` /
    ``filter_admitted`` are fail-closed: a strategy with no registry entry at all, a
    NOT VALIDATED verdict, or an expired version are all "not admitted" -- there is no
    warning-only path an operator can silently rely on instead.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def register(self, version: StrategyVersion) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", version.strategy_name.lower()).strip("_")
        path = self.directory / f"{slug or 'strategy'}__{version.version}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(version.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def load_all(self) -> list[StrategyVersion]:
        if not self.directory.exists():
            return []
        versions: list[StrategyVersion] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                versions.append(StrategyVersion.from_dict(payload))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping unreadable strategy registry entry {path}: {e}")
        return versions

    def latest(self, strategy_name: str) -> StrategyVersion | None:
        """Most-recently-validated on-file version for this strategy, or None."""
        candidates = [v for v in self.load_all() if v.strategy_name == strategy_name]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.validated_at)

    def is_admitted(
        self, strategy_name: str, *, regime: str | None = None, symbol: str | None = None
    ) -> bool:
        """Fail-closed: True only if a current, non-expired VALIDATED version exists."""
        version = self.latest(strategy_name)
        if version is None:
            return False
        return version.is_admissible(regime=regime, symbol=symbol)

    def filter_admitted(
        self, strategy_names: list[str], *, regime: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Split ``strategy_names`` into (admitted, stripped) -- fail closed on each name."""
        admitted: list[str] = []
        stripped: list[str] = []
        for name in strategy_names:
            if self.is_admitted(name, regime=regime):
                admitted.append(name)
            else:
                stripped.append(name)
        return admitted, stripped
