"""Versioned strategy eligibility and runtime regime policy.

Offline validation records are addressed only by strategy name, candle timeframe,
and model version.  Trade horizon remains useful decision metadata, but it is not an
artifact key.  Regime is evaluated against the selected record at runtime, so changing
market regime never loads a different model artifact.

This module is deliberately policy-only.  It cannot place orders, change position
sizes, or bypass the deterministic risk engine.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.backtesting.strategy_registry import StrategyRegistry, StrategyVersion
from src.config import get_settings

logger = logging.getLogger(__name__)


class EligibilityStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    SHADOW = "SHADOW"
    PAPER_APPROVED = "PAPER_APPROVED"
    LIVE_APPROVED = "LIVE_APPROVED"
    DISABLED = "DISABLED"


class RegimePolicy(str, Enum):
    ALLOW = "ALLOW"
    REDUCE_CONFIDENCE = "REDUCE_CONFIDENCE"
    BLOCK = "BLOCK"


class EligibilityEnvironment(str, Enum):
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    SIMULATED = "SIMULATED"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


def _normalise_regime(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").strip().lower()).strip("_")


def _regime_keys(value: str | None) -> tuple[str, ...]:
    """Return compatible labels without assigning any policy values."""

    exact = _normalise_regime(value)
    keys = [exact]
    if exact in {"trending_up", "trending_down"}:
        keys.append("trending")
    if exact in {"volatile", "high_volatility"}:
        keys.extend(item for item in ("volatile", "high_volatility") if item != exact)
    return tuple(dict.fromkeys(keys))


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class RegimePerformance:
    policy: RegimePolicy = RegimePolicy.ALLOW
    profit_factor: float | None = None
    trade_count: int = 0
    confidence_multiplier: float = 1.0
    win_rate: float | None = None
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        if self.trade_count < 0:
            raise ValueError("Regime trade_count cannot be negative")
        if not 0.0 <= self.confidence_multiplier <= 1.0:
            raise ValueError("Regime confidence_multiplier must be in [0, 1]")
        if self.policy is RegimePolicy.REDUCE_CONFIDENCE and not (
            0.0 < self.confidence_multiplier < 1.0
        ):
            raise ValueError(
                "REDUCE_CONFIDENCE requires a multiplier strictly between zero and one"
            )
        if self.policy is RegimePolicy.BLOCK and self.confidence_multiplier != 0.0:
            raise ValueError("BLOCK requires confidence_multiplier=0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "profit_factor": self.profit_factor,
            "trade_count": self.trade_count,
            "confidence_multiplier": self.confidence_multiplier,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RegimePerformance:
        return cls(
            policy=RegimePolicy(str(value.get("policy", RegimePolicy.ALLOW.value)).upper()),
            profit_factor=_finite_optional(value.get("profit_factor")),
            trade_count=int(value.get("trade_count", 0)),
            confidence_multiplier=float(value.get("confidence_multiplier", 1.0)),
            win_rate=_finite_optional(value.get("win_rate")),
            max_drawdown=_finite_optional(value.get("max_drawdown")),
        )


@dataclass(frozen=True)
class StrategyEligibility:
    strategy_name: str
    timeframe: str
    model_version: str
    validation_status: EligibilityStatus
    validated_at: str
    validation_window: dict[str, Any]
    oos_trade_count: int
    oos_profit_factor: float | None
    oos_max_drawdown: float | None
    oos_win_rate: float | None
    minimum_model_confidence: float
    minimum_strategy_confidence: float = 0.0
    allowed_regimes: tuple[str, ...] = ()
    disabled_regimes: tuple[str, ...] = ()
    status_reason: str = ""
    artifact_reference: str | None = None
    requires_ml: bool = False
    regime_performance: dict[str, RegimePerformance] = field(default_factory=dict)
    validation_metadata: dict[str, Any] = field(default_factory=dict)
    market: str = "NSE"

    def __post_init__(self) -> None:
        if not self.strategy_name.strip():
            raise ValueError("strategy_name is required")
        if not self.timeframe.strip() or self.timeframe == "*":
            raise ValueError("timeframe must be an exact candle interval")
        if not self.model_version.strip():
            raise ValueError("model_version is required")
        if self.oos_trade_count < 0:
            raise ValueError("oos_trade_count cannot be negative")
        if not 0.0 <= self.minimum_model_confidence <= 1.0:
            raise ValueError("minimum_model_confidence must be in [0, 1]")
        if not 0.0 <= self.minimum_strategy_confidence <= 1.0:
            raise ValueError("minimum_strategy_confidence must be in [0, 1]")

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.market.strip().upper(),
            self.strategy_name.strip().lower(),
            self.timeframe.strip().lower(),
            self.model_version.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.strip().upper(),
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "model_version": self.model_version,
            "validation_status": self.validation_status.value,
            "validated_at": self.validated_at,
            "validation_window": self.validation_window,
            "oos_trade_count": self.oos_trade_count,
            "oos_profit_factor": self.oos_profit_factor,
            "oos_max_drawdown": self.oos_max_drawdown,
            "oos_win_rate": self.oos_win_rate,
            "minimum_model_confidence": self.minimum_model_confidence,
            "minimum_strategy_confidence": self.minimum_strategy_confidence,
            "allowed_regimes": list(self.allowed_regimes),
            "disabled_regimes": list(self.disabled_regimes),
            "status_reason": self.status_reason,
            "artifact_reference": self.artifact_reference,
            "requires_ml": self.requires_ml,
            "regime_performance": {
                name: performance.to_dict()
                for name, performance in sorted(self.regime_performance.items())
            },
            "validation_metadata": self.validation_metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StrategyEligibility:
        regime_performance = {
            _normalise_regime(name): RegimePerformance.from_dict(dict(performance))
            for name, performance in dict(value.get("regime_performance", {})).items()
        }
        return cls(
            strategy_name=str(value["strategy_name"]),
            timeframe=str(value["timeframe"]).lower(),
            model_version=str(value["model_version"]),
            validation_status=EligibilityStatus(str(value["validation_status"]).upper()),
            validated_at=str(value.get("validated_at", "")),
            validation_window=dict(value.get("validation_window", {})),
            oos_trade_count=int(value.get("oos_trade_count", 0)),
            oos_profit_factor=_finite_optional(value.get("oos_profit_factor")),
            oos_max_drawdown=_finite_optional(value.get("oos_max_drawdown")),
            oos_win_rate=_finite_optional(value.get("oos_win_rate")),
            minimum_model_confidence=float(value.get("minimum_model_confidence", 0.0)),
            minimum_strategy_confidence=float(
                value.get("minimum_strategy_confidence", 0.0)
            ),
            allowed_regimes=tuple(
                _normalise_regime(item) for item in value.get("allowed_regimes", ())
            ),
            disabled_regimes=tuple(
                _normalise_regime(item) for item in value.get("disabled_regimes", ())
            ),
            status_reason=str(value.get("status_reason", "")),
            artifact_reference=(
                str(value["artifact_reference"])
                if value.get("artifact_reference") is not None
                else None
            ),
            requires_ml=bool(value.get("requires_ml", False)),
            regime_performance=regime_performance,
            validation_metadata=dict(value.get("validation_metadata", {})),
            market=str(value.get("market", "NSE")).upper(),
        )


@dataclass(frozen=True)
class EligibilityDecision:
    strategy_name: str
    timeframe: str
    model_version: str
    validation_status: EligibilityStatus
    environment: EligibilityEnvironment
    current_regime: str
    regime_policy: RegimePolicy
    original_confidence: float
    adjusted_confidence: float
    confidence_multiplier: float
    registry_allowed: bool
    pipeline_allowed: bool
    qwen_allowed: bool
    execution_allowed: bool
    shadow_only: bool
    record_only: bool
    reason_code: str
    message: str
    eligibility: StrategyEligibility | None = field(compare=False, repr=False)
    market: str = "NSE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "strategy": self.strategy_name,
            "timeframe": self.timeframe,
            "model_version": self.model_version,
            "validation_status": self.validation_status.value,
            "eligibility_environment": self.environment.value,
            "current_regime": self.current_regime,
            "regime_policy": self.regime_policy.value,
            "original_confidence": self.original_confidence,
            "regime_adjusted_confidence": self.adjusted_confidence,
            "regime_confidence_multiplier": self.confidence_multiplier,
            "registry_decision": "ALLOW" if self.pipeline_allowed else "BLOCK",
            "registry_allowed": self.registry_allowed,
            "pipeline_allowed": self.pipeline_allowed,
            "qwen_allowed": self.qwen_allowed,
            "execution_allowed": self.execution_allowed,
            "shadow_only": self.shadow_only,
            "record_only": self.record_only,
            "eligibility_reason": self.reason_code,
            "eligibility_message": self.message,
        }


def resolve_eligibility_environment(
    runtime_mode: str,
    *,
    requested_execution_mode: str = "local_paper",
    trading_mode: str = "paper",
) -> EligibilityEnvironment:
    runtime = str(runtime_mode).strip().lower()
    requested = str(requested_execution_mode).strip().lower()
    trading = str(trading_mode).strip().lower()
    if runtime == "research":
        return EligibilityEnvironment.RESEARCH
    if runtime == "backtest":
        return EligibilityEnvironment.BACKTEST
    if trading == "live" or requested == "live" or runtime == "live":
        return EligibilityEnvironment.LIVE
    if requested in {"shadow", "dhan_paper"} or runtime == "shadow":
        return EligibilityEnvironment.SHADOW
    if runtime in {"mock", "simulated", "simulation"}:
        return EligibilityEnvironment.SIMULATED
    return EligibilityEnvironment.PAPER


class StrategyEligibilityRegistry:
    """Durable registry keyed by market + strategy + timeframe + model version."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._cache_lock = threading.RLock()
        self._cached_records: list[StrategyEligibility] | None = None
        self._cache_expires_monotonic = 0.0

    def register(self, eligibility: StrategyEligibility) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)

        def slug(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

        path = self.directory / (
            f"{slug(eligibility.market)}__{slug(eligibility.strategy_name)}"
            f"__{slug(eligibility.timeframe)}"
            f"__{slug(eligibility.model_version)}.json"
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(eligibility.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
        with self._cache_lock:
            self._cached_records = None
            self._cache_expires_monotonic = 0.0
        return path

    def load_all(self) -> list[StrategyEligibility]:
        with self._cache_lock:
            if (
                self._cached_records is not None
                and time.monotonic() < self._cache_expires_monotonic
            ):
                return list(self._cached_records)
        if not self.directory.exists():
            return []
        records: list[StrategyEligibility] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                records.append(
                    StrategyEligibility.from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable strategy eligibility record %s: %s", path, exc)
        with self._cache_lock:
            self._cached_records = list(records)
            self._cache_expires_monotonic = time.monotonic() + 5.0
        return records

    def get(
        self,
        strategy_name: str,
        timeframe: str,
        model_version: str,
        *,
        market: str = "NSE",
    ) -> StrategyEligibility | None:
        identity = (
            market.strip().upper(),
            strategy_name.strip().lower(),
            timeframe.strip().lower(),
            model_version.strip(),
        )
        matching = [item for item in self.load_all() if item.identity == identity]
        if not matching:
            return None
        return max(matching, key=lambda item: item.validated_at)

    def latest(
        self, strategy_name: str, timeframe: str, *, market: str = "NSE"
    ) -> StrategyEligibility | None:
        """Select a model version without consulting horizon or regime."""

        strategy = strategy_name.strip().lower()
        candle_timeframe = timeframe.strip().lower()
        matching = [
            item
            for item in self.load_all()
            if item.market.strip().upper() == market.strip().upper()
            and item.strategy_name.strip().lower() == strategy
            and item.timeframe.strip().lower() == candle_timeframe
        ]
        if not matching:
            return None
        return max(matching, key=lambda item: item.validated_at)

    def select_for_environment(
        self,
        strategy_name: str,
        timeframe: str,
        environment: EligibilityEnvironment | str,
        *,
        market: str = "NSE",
    ) -> StrategyEligibility | None:
        """Select the latest approved champion appropriate to the runtime.

        A newer UNVALIDATED/SHADOW challenger must not hide an older approved
        record.  This is selection only: no status is changed and the returned
        record is still evaluated by its exact market/strategy/timeframe/version
        identity.  A latest DISABLED record remains authoritative and cannot be
        bypassed by falling back to an older approval.
        """

        env = (
            environment
            if isinstance(environment, EligibilityEnvironment)
            else EligibilityEnvironment(str(environment).upper())
        )
        strategy = strategy_name.strip().lower()
        candle_timeframe = timeframe.strip().lower()
        matching = [
            item
            for item in self.load_all()
            if item.market.strip().upper() == market.strip().upper()
            and item.strategy_name.strip().lower() == strategy
            and item.timeframe.strip().lower() == candle_timeframe
        ]
        if not matching:
            return None
        latest = max(matching, key=lambda item: item.validated_at)
        if latest.validation_status is EligibilityStatus.DISABLED:
            return latest
        approved_statuses: set[EligibilityStatus]
        if env is EligibilityEnvironment.LIVE:
            approved_statuses = {EligibilityStatus.LIVE_APPROVED}
        elif env is EligibilityEnvironment.PAPER:
            approved_statuses = {
                EligibilityStatus.PAPER_APPROVED,
                EligibilityStatus.LIVE_APPROVED,
            }
        else:
            return latest
        approved = [item for item in matching if item.validation_status in approved_statuses]
        return max(approved, key=lambda item: item.validated_at) if approved else latest

    def evaluate(
        self,
        *,
        strategy_name: str,
        timeframe: str,
        model_version: str = "",
        environment: EligibilityEnvironment | str,
        current_regime: str | None,
        confidence: float,
        action: str = "BUY",
        market: str = "NSE",
    ) -> EligibilityDecision:
        env = (
            environment
            if isinstance(environment, EligibilityEnvironment)
            else EligibilityEnvironment(str(environment).upper())
        )
        selected = (
            self.get(strategy_name, timeframe, model_version, market=market)
            if model_version
            else self.select_for_environment(
                strategy_name,
                timeframe,
                env,
                market=market,
            )
        )
        model_mismatch = bool(
            model_version
            and selected is None
            and self.latest(strategy_name, timeframe, market=market) is not None
        )
        status = selected.validation_status if selected else EligibilityStatus.UNVALIDATED
        selected_model = selected.model_version if selected else model_version
        original = max(0.0, min(1.0, float(confidence)))
        regime = _normalise_regime(current_regime)

        status_allowed, execution_allowed, record_only, status_reason = _status_policy(
            status,
            env,
            selected is not None,
            paper_gate_enabled=get_settings().strategy_eligibility_paper_gate_enabled,
        )
        if model_mismatch:
            status_reason = "MODEL_VERSION_MISMATCH"
            if env not in {
                EligibilityEnvironment.RESEARCH,
                EligibilityEnvironment.BACKTEST,
                EligibilityEnvironment.SIMULATED,
                EligibilityEnvironment.SHADOW,
            }:
                status_allowed = False
                execution_allowed = False
                record_only = True
        if status is EligibilityStatus.DISABLED:
            status_allowed = False
            execution_allowed = False
            record_only = True
            status_reason = "STRATEGY_DISABLED"

        regime_policy = RegimePolicy.ALLOW
        multiplier = 1.0
        regime_reason = ""
        if selected is not None:
            regime_policy, multiplier, regime_reason = _evaluate_regime(selected, regime)
        adjusted = original * multiplier
        confidence_ok = bool(
            selected is None or adjusted >= selected.minimum_strategy_confidence
        )
        pipeline_allowed = status_allowed and regime_policy is not RegimePolicy.BLOCK and confidence_ok

        reason_code = status_reason
        if status_allowed and regime_policy is RegimePolicy.BLOCK:
            reason_code = regime_reason or "REGIME_BLOCKED"
        elif status_allowed and not confidence_ok:
            reason_code = "REGIME_ADJUSTED_CONFIDENCE_BELOW_MINIMUM"
        elif pipeline_allowed and regime_policy is RegimePolicy.REDUCE_CONFIDENCE:
            reason_code = "REGIME_CONFIDENCE_REDUCED"
        elif pipeline_allowed and (
            status_reason.startswith("ELIGIBILITY_REGISTRY_MISSING")
            or status_reason == "MODEL_VERSION_MISMATCH"
        ):
            reason_code = status_reason
        elif pipeline_allowed:
            reason_code = "STRATEGY_ELIGIBLE"

        verb = str(action or "BUY").upper()
        if reason_code == "STRATEGY_NOT_LIVE_APPROVED":
            message = f"{verb} blocked: strategy not LIVE_APPROVED"
        elif reason_code == "STRATEGY_NOT_PAPER_APPROVED":
            message = f"{verb} recorded only: strategy not PAPER_APPROVED"
        elif regime_policy is RegimePolicy.BLOCK:
            message = f"{verb} blocked: current regime disabled for this strategy"
        elif reason_code == "REGIME_CONFIDENCE_REDUCED":
            message = f"{verb} confidence reduced by {regime}-regime policy"
        elif reason_code == "REGIME_ADJUSTED_CONFIDENCE_BELOW_MINIMUM":
            message = f"{verb} blocked: regime-adjusted confidence is below strategy minimum"
        elif pipeline_allowed and reason_code.startswith("ELIGIBILITY_REGISTRY_MISSING"):
            message = f"{verb} allowed for analysis despite missing strategy eligibility record"
        elif pipeline_allowed and reason_code == "MODEL_VERSION_MISMATCH":
            message = f"{verb} allowed for analysis despite model-version mismatch"
        elif pipeline_allowed:
            message = f"{verb} allowed by strategy registry; proceeding to ML/risk"
        else:
            message = f"{verb} blocked by strategy eligibility: {reason_code}"

        qwen_allowed = pipeline_allowed and execution_allowed
        return EligibilityDecision(
            strategy_name=strategy_name,
            timeframe=timeframe,
            model_version=selected_model,
            validation_status=status,
            environment=env,
            current_regime=regime,
            regime_policy=regime_policy,
            original_confidence=original,
            adjusted_confidence=adjusted,
            confidence_multiplier=multiplier,
            registry_allowed=status_allowed,
            pipeline_allowed=pipeline_allowed,
            qwen_allowed=qwen_allowed,
            execution_allowed=execution_allowed and pipeline_allowed,
            shadow_only=not (execution_allowed and pipeline_allowed),
            record_only=record_only or not execution_allowed,
            reason_code=reason_code,
            message=message,
            eligibility=selected,
            market=market.strip().upper(),
        )

    @staticmethod
    def ml_qualified(
        decision: EligibilityDecision, ml_confidence: float | None
    ) -> tuple[bool, str]:
        eligibility = decision.eligibility
        if eligibility is None or not eligibility.requires_ml:
            return True, "ML_NOT_REQUIRED"
        if not eligibility.artifact_reference:
            return False, "ML_ARTIFACT_MISSING"
        if ml_confidence is None:
            return False, "ML_ABSTAIN"
        if float(ml_confidence) < eligibility.minimum_model_confidence:
            return False, "ML_BELOW_STRATEGY_MINIMUM"
        return True, "ML_QUALIFIED"


def _status_policy(
    status: EligibilityStatus,
    environment: EligibilityEnvironment,
    record_exists: bool,
    *,
    paper_gate_enabled: bool = True,
) -> tuple[bool, bool, bool, str]:
    if environment in {EligibilityEnvironment.RESEARCH, EligibilityEnvironment.BACKTEST}:
        return True, False, True, (
            "ELIGIBILITY_REGISTRY_MISSING_RESEARCH"
            if not record_exists
            else "STRATEGY_ELIGIBLE_FOR_RESEARCH"
        )
    if environment in {EligibilityEnvironment.SIMULATED, EligibilityEnvironment.SHADOW}:
        return True, False, True, (
            "ELIGIBILITY_REGISTRY_MISSING_SHADOW"
            if not record_exists
            else "STRATEGY_ELIGIBLE_FOR_SHADOW"
        )
    if environment is EligibilityEnvironment.PAPER:
        # STRATEGY_ELIGIBILITY_PAPER_GATE_ENABLED=false lets every strategy trade paper
        # capital regardless of walk-forward/Bonferroni verdict -- an explicit, operator-
        # chosen opt-out of the statistical admission gate for paper trading only. This
        # never touches the LIVE branch below: LIVE_APPROVED is still required to reach a
        # real broker route no matter what this flag is set to.
        if not paper_gate_enabled:
            return True, True, False, (
                "ELIGIBILITY_REGISTRY_MISSING_GATE_DISABLED"
                if not record_exists
                else "STRATEGY_ELIGIBLE_GATE_DISABLED"
            )
        if not record_exists:
            return False, False, True, "ELIGIBILITY_REGISTRY_MISSING"
        allowed = status in {
            EligibilityStatus.PAPER_APPROVED,
            EligibilityStatus.LIVE_APPROVED,
        }
        return allowed, allowed, not allowed, (
            "STRATEGY_ELIGIBLE" if allowed else "STRATEGY_NOT_PAPER_APPROVED"
        )
    if not record_exists:
        return False, False, True, "ELIGIBILITY_REGISTRY_MISSING"
    allowed = status is EligibilityStatus.LIVE_APPROVED
    return allowed, allowed, not allowed, (
        "STRATEGY_ELIGIBLE" if allowed else "STRATEGY_NOT_LIVE_APPROVED"
    )


def _evaluate_regime(
    eligibility: StrategyEligibility, current_regime: str
) -> tuple[RegimePolicy, float, str]:
    if current_regime in {"", "unknown", "none"}:
        return RegimePolicy.ALLOW, 1.0, ""
    keys = _regime_keys(current_regime)
    disabled = {_normalise_regime(item) for item in eligibility.disabled_regimes}
    if any(key in disabled for key in keys):
        return RegimePolicy.BLOCK, 0.0, "REGIME_DISABLED"

    for key in keys:
        performance = eligibility.regime_performance.get(key)
        if performance is not None:
            return performance.policy, performance.confidence_multiplier, (
                "REGIME_BLOCKED" if performance.policy is RegimePolicy.BLOCK else ""
            )

    allowed = {_normalise_regime(item) for item in eligibility.allowed_regimes}
    if allowed and not any(key in allowed for key in keys):
        return RegimePolicy.BLOCK, 0.0, "REGIME_NOT_ALLOWED"
    return RegimePolicy.ALLOW, 1.0, ""


@dataclass(frozen=True)
class MigrationReport:
    created: int
    skipped: int
    conflicts: int
    records: tuple[StrategyEligibility, ...] = field(compare=False, repr=False)


def migrate_legacy_strategy_registry(
    legacy_registry: StrategyRegistry,
    eligibility_registry: StrategyEligibilityRegistry,
    *,
    runtime_timeframes: Iterable[str],
    market: str = "NSE",
) -> MigrationReport:
    """Collapse legacy strict-grain records without granting live approval.

    Unpinned legacy records are expanded to the explicitly supplied runtime
    timeframes.  Conflicting horizon/regime records are downgraded to SHADOW unless
    they share a non-empty safe regime intersection; no migration path emits
    LIVE_APPROVED.
    """

    timeframes = tuple(dict.fromkeys(str(item).lower() for item in runtime_timeframes if item))
    grouped: dict[tuple[str, str, str], list[StrategyVersion]] = {}
    for legacy in legacy_registry.load_all():
        effective = (legacy.timeframe.lower(),) if legacy.timeframe else timeframes
        for timeframe in effective:
            grouped.setdefault((legacy.strategy_name, timeframe, legacy.version), []).append(legacy)

    created = 0
    skipped = 0
    conflicts = 0
    migrated: list[StrategyEligibility] = []
    for (strategy, timeframe, model_version), group in sorted(grouped.items()):
        existing = eligibility_registry.get(
            strategy, timeframe, model_version, market=market
        )
        if existing is not None:
            existing_metadata = existing.validation_metadata
            is_old_migration = (
                existing_metadata.get("migration_source") == "legacy_strategy_registry"
                and int(existing_metadata.get("migration_version", 0)) < 2
            )
            if not is_old_migration:
                skipped += 1
                continue

        statuses = {item.status for item in group}
        horizons = {item.trade_horizon for item in group}
        regime_sets = {tuple(sorted(item.approved_regimes)) for item in group}
        conflict_reasons: list[str] = []
        if len(statuses) > 1:
            conflict_reasons.append("legacy validation statuses disagree")
        if len(horizons) > 1:
            conflict_reasons.append("legacy trade horizons disagree")
        if len(regime_sets) > 1:
            conflict_reasons.append("legacy regime approvals disagree")

        non_empty_regimes = [set(item.approved_regimes) for item in group if item.approved_regimes]
        safe_regimes = set.intersection(*non_empty_regimes) if non_empty_regimes else set()
        if non_empty_regimes and not safe_regimes:
            conflict_reasons.append("legacy regimes have no safe common intersection")

        all_validated = statuses == {"VALIDATED"}
        if all_validated and not conflict_reasons:
            status = EligibilityStatus.PAPER_APPROVED
        elif all_validated and safe_regimes:
            status = EligibilityStatus.PAPER_APPROVED
        elif any(item.verdict == "VALIDATED" for item in group):
            status = EligibilityStatus.SHADOW
        else:
            status = EligibilityStatus.UNVALIDATED

        if conflict_reasons:
            conflicts += 1
        if conflict_reasons:
            status_reason = "Migrated conservatively: " + "; ".join(conflict_reasons)
        elif status is EligibilityStatus.PAPER_APPROVED:
            status_reason = (
                "Migrated from legacy validation as paper-only; "
                "live approval requires explicit review"
            )
        elif status is EligibilityStatus.SHADOW:
            status_reason = "Legacy evidence retained as shadow; not approved for execution"
        else:
            status_reason = "Legacy validation did not pass; retained as unvalidated evidence"

        source = max(group, key=lambda item: item.validated_at)
        parameters = source.parameters or {}
        minimum_confidence = float(
            parameters.get(
                "minimum_model_confidence",
                parameters.get("min_ml_confidence", 0.0),
            )
        )
        requires_ml = bool(parameters.get("requires_ml", parameters.get("ml_required", False)))
        artifact_reference = parameters.get("model_artifact") or parameters.get(
            "artifact_reference"
        )
        regime_performance = {
            _normalise_regime(regime): RegimePerformance(policy=RegimePolicy.ALLOW)
            for regime in safe_regimes
        }
        record = StrategyEligibility(
            strategy_name=strategy,
            timeframe=timeframe,
            model_version=model_version,
            validation_status=status,
            validated_at=max(item.validated_at for item in group),
            validation_window={
                "dataset_ids": sorted({item.dataset_id for item in group}),
                "legacy_expiry": max(item.expires_at for item in group),
                "parameters": parameters,
            },
            oos_trade_count=min(item.oos_trades for item in group),
            oos_profit_factor=None,
            oos_max_drawdown=None,
            oos_win_rate=None,
            minimum_model_confidence=max(0.0, min(1.0, minimum_confidence)),
            minimum_strategy_confidence=max(
                0.0,
                min(1.0, float(parameters.get("minimum_strategy_confidence", 0.0))),
            ),
            allowed_regimes=tuple(sorted(safe_regimes)),
            disabled_regimes=(),
            status_reason=status_reason,
            artifact_reference=str(artifact_reference) if artifact_reference else None,
            requires_ml=requires_ml,
            regime_performance=regime_performance,
            validation_metadata={
                "migration_source": "legacy_strategy_registry",
                "migration_version": 2,
                "legacy_horizons": sorted(horizons),
                "legacy_statuses": sorted(statuses),
                "legacy_approved_universe": sorted(
                    {symbol for item in group for symbol in item.approved_universe}
                ),
                "legacy_oos_expectancy": min(item.oos_expectancy for item in group),
                "legacy_oos_return_pct": min(item.oos_return_pct for item in group),
                "legacy_fold_consistency": min(item.fold_consistency for item in group),
            },
            market=market.strip().upper(),
        )
        eligibility_registry.register(record)
        migrated.append(record)
        created += 1
    return MigrationReport(created, skipped, conflicts, tuple(migrated))


def eligibility_registry_from_settings(
    settings: Any,
    *,
    runtime_timeframes: Iterable[str] = ("5m", "15m", "30m", "1h", "4h", "1d"),
) -> StrategyEligibilityRegistry:
    market = str(getattr(settings, "market", "NSE")).strip().lower()
    legacy_dir = Path(
        str(getattr(settings, "strategy_registry_dir", "data/nse/strategy_registry"))
    )
    legacy_parts = list(legacy_dir.parts)
    lowered_legacy_parts = tuple(part.lower() for part in legacy_parts)
    if market in {"nse", "forex"} and market not in lowered_legacy_parts:
        for index, part in enumerate(legacy_parts):
            if part.lower() in {"nse", "forex"}:
                legacy_parts[index] = market
                legacy_dir = Path(*legacy_parts)
                break
    configured = getattr(settings, "strategy_eligibility_registry_dir", None)
    target_dir = Path(str(configured)) if configured else legacy_dir.parent / "strategy_eligibility"
    lowered_parts = tuple(part.lower() for part in target_dir.parts)
    if market in lowered_parts:
        pass
    elif market in {"nse", "forex"} and any(
        owner in lowered_parts for owner in {"nse", "forex"} - {market}
    ):
        parts = list(target_dir.parts)
        for index, part in enumerate(parts):
            if part.lower() in {"nse", "forex"}:
                parts[index] = market
                break
        target_dir = Path(*parts)
    elif target_dir.name.lower() != market:
        target_dir = target_dir / market
    registry = StrategyEligibilityRegistry(target_dir)
    migrate_legacy_strategy_registry(
        StrategyRegistry(legacy_dir),
        registry,
        runtime_timeframes=runtime_timeframes,
        market=market,
    )
    return registry


def eligibility_registry_rows(
    registry: StrategyEligibilityRegistry,
) -> list[dict[str, Any]]:
    return [
        {
            "strategy": item.strategy_name,
            "timeframe": item.timeframe,
            "model_version": item.model_version,
            "validation_status": item.validation_status.value,
            "validated_at": item.validated_at,
            "allowed_regimes": list(item.allowed_regimes) or ["ALL"],
            "disabled_regimes": list(item.disabled_regimes),
            "minimum_model_confidence": item.minimum_model_confidence,
            "minimum_strategy_confidence": item.minimum_strategy_confidence,
            "oos_trade_count": item.oos_trade_count,
            "oos_profit_factor": item.oos_profit_factor,
            "oos_max_drawdown": item.oos_max_drawdown,
            "oos_win_rate": item.oos_win_rate,
            "status_reason": item.status_reason,
        }
        for item in sorted(registry.load_all(), key=lambda row: row.identity)
    ]
