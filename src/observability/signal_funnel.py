"""Reconciled, execution-neutral observability for the live signal funnel.

This module never decides whether a trade may proceed.  It translates decisions
already made by strategy admission, candidate policy, ML, Qwen, validation, and
risk into stable reason codes and per-cycle counters for LangFuse/the dashboard.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.backtesting.strategy_eligibility import (
    EligibilityDecision,
    StrategyEligibilityRegistry,
    eligibility_registry_rows,
)


class RejectionReason(str, Enum):
    """Canonical diagnostic reasons; these are not trading-policy settings."""

    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    NO_SETTLED_CANDLE = "NO_SETTLED_CANDLE"
    STALE_QUOTE = "STALE_QUOTE"
    INVALID_QUOTE = "INVALID_QUOTE"
    CONSOLIDATED_DUPLICATE = "CONSOLIDATED_DUPLICATE"
    NO_TECHNICAL_SETUP = "NO_TECHNICAL_SETUP"
    ELIGIBILITY_REGISTRY_MISSING = "ELIGIBILITY_REGISTRY_MISSING"
    STRATEGY_UNVALIDATED = "STRATEGY_UNVALIDATED"
    STRATEGY_SHADOW_ONLY = "STRATEGY_SHADOW_ONLY"
    STRATEGY_NOT_PAPER_APPROVED = "STRATEGY_NOT_PAPER_APPROVED"
    STRATEGY_NOT_LIVE_APPROVED = "STRATEGY_NOT_LIVE_APPROVED"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"
    REGIME_BLOCKED = "REGIME_BLOCKED"
    REGIME_CONFIDENCE_BELOW_MINIMUM = "REGIME_ADJUSTED_CONFIDENCE_BELOW_MINIMUM"
    REGIME_CONFIDENCE_REDUCED = "REGIME_CONFIDENCE_REDUCED"
    ML_ARTIFACT_MISSING = "ML_ARTIFACT_MISSING"
    ML_ABSTAIN = "ML_ABSTAIN"
    ML_DISABLED = "ML_DISABLED"
    ML_BELOW_THRESHOLD = "ML_BELOW_THRESHOLD"
    LOW_ENTRY_QUALITY = "LOW_ENTRY_QUALITY"
    LOW_RELATIVE_VOLUME = "LOW_RELATIVE_VOLUME"
    BAD_RR = "BAD_RR"
    REGIME_INCOMPATIBLE = "REGIME_INCOMPATIBLE"
    MTF_MISSING_DATA = "MTF_MISSING_DATA"
    MTF_CONFLICT = "MTF_CONFLICT"
    WAIT_PULLBACK = "WAIT_PULLBACK"
    WAIT_BREAKOUT = "WAIT_BREAKOUT"
    LONG_ONLY = "LONG_ONLY"
    INVALID_LEVELS = "INVALID_LEVELS"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    CAPITAL_CONSTRAINT = "CAPITAL_CONSTRAINT"
    RISK_LIMIT = "RISK_LIMIT"
    QWEN_CONTEXT_REJECT = "QWEN_CONTEXT_REJECT"
    QWEN_DEFERRED = "QWEN_DEFERRED"
    SIGNAL_VALIDATION_REJECT = "SIGNAL_VALIDATION_REJECT"
    EXECUTION_REJECT = "EXECUTION_REJECT"


FUNNEL_KEYS = (
    "symbols_requested",
    "symbols_ready",
    "symbols_scanned",
    "strategy_evaluations",
    "raw_signals",
    "technical_setups",
    "deterministic_qualified",
    "deterministic_rejected",
    "registry_eligible",
    "registry_shadow_only",
    "registry_blocked",
    "registry_unvalidated",
    "regime_blocked",
    "confidence_reduced",
    "ml_candidates",
    "ml_inference",
    "ml_abstain",
    "ml_pass",
    "qwen_required",
    "qwen_skipped_clear",
    "qwen_cache_hits",
    "qwen_external_calls",
    "qwen_deferred",
    "qwen_rejected",
    "signal_validation_pass",
    "signal_validation_reject",
    "risk_approved",
    "risk_blocked",
    "final_buy",
    "final_hold",
    "final_wait",
    "final_reject",
)


def empty_funnel(*, cycle_id: int = 0, trade_horizon: str = "COMBINED") -> dict[str, Any]:
    result: dict[str, Any] = {key: 0 for key in FUNNEL_KEYS}
    result.update(
        {
            "cycle_id": cycle_id,
            "trade_horizon": trade_horizon,
            "reconciled": True,
            "reconciliation_errors": [],
        }
    )
    return result


def eligibility_rejection_reason(decision: EligibilityDecision) -> str | None:
    """Return the stable runtime reason for a blocked eligibility decision."""

    return None if decision.pipeline_allowed else decision.reason_code


def scalp_status_reason(status: Any) -> str | None:
    """Map an existing ScalpSymbolStatus to one stable primary reason."""

    stage = str(getattr(status, "stage", "") or "").upper()
    if stage == "DATA_UNAVAILABLE":
        return RejectionReason.INSUFFICIENT_HISTORY.value
    if stage == "NO_BUY_TRIGGER":
        return RejectionReason.NO_TECHNICAL_SETUP.value
    if stage == "MTF_MISSING_DATA":
        return RejectionReason.MTF_MISSING_DATA.value
    if stage in {"MTF_NOT_CONFIRMED", "MTF_CONFLICT", "NO_PRIMARY_SETUP"}:
        return RejectionReason.MTF_CONFLICT.value
    if stage == "ML_NOT_CONFIRMED":
        return RejectionReason.ML_ABSTAIN.value
    if stage == "REGIME_BLOCKED":
        return RejectionReason.REGIME_INCOMPATIBLE.value
    if stage == "ENTRY_NOT_READY":
        entry_quality = getattr(status, "entry_quality", None)
        quality_status = str(getattr(entry_quality, "status", "") or "").upper()
        if quality_status == "WAIT_PULLBACK":
            return RejectionReason.WAIT_PULLBACK.value
        if quality_status == "WAIT_BREAKOUT":
            return RejectionReason.WAIT_BREAKOUT.value
        return RejectionReason.LOW_ENTRY_QUALITY.value
    return None


@dataclass
class FunnelAudit:
    """Mutable per-cycle accumulator with bounded aggregated drill-down."""

    cycle_id: int
    trade_horizon: str = "COMBINED"
    counts: dict[str, Any] = field(init=False)
    _rejections: Counter[tuple[str, str]] = field(default_factory=Counter)
    _details: Counter[tuple[str, str, str, str, str]] = field(default_factory=Counter)
    _strategies: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def __post_init__(self) -> None:
        self.counts = empty_funnel(cycle_id=self.cycle_id, trade_horizon=self.trade_horizon)

    @staticmethod
    def grain(strategy: str, timeframe: str, trade_horizon: str) -> str:
        return f"{strategy}|{timeframe}|{trade_horizon}"

    def set(self, key: str, value: int) -> None:
        if key not in FUNNEL_KEYS:
            raise KeyError(f"Unknown funnel counter: {key}")
        self.counts[key] = max(0, int(value))

    def increment(self, key: str, amount: int = 1) -> None:
        self.set(key, int(self.counts.get(key, 0)) + amount)

    def strategy_increment(
        self,
        strategy: str,
        timeframe: str,
        trade_horizon: str,
        metric: str,
        amount: int = 1,
    ) -> None:
        self._strategies[self.grain(strategy, timeframe, trade_horizon)][metric] += amount

    def reject(
        self,
        stage: str,
        reason: str,
        *,
        strategy: str = "",
        timeframe: str = "",
        trade_horizon: str = "",
        symbol: str = "",
        amount: int = 1,
    ) -> None:
        normalized_stage = str(stage).upper()
        normalized_reason = str(reason).upper()
        self._rejections[(normalized_stage, normalized_reason)] += amount
        self._details[
            (
                normalized_stage,
                normalized_reason,
                strategy or "*",
                timeframe or "*",
                symbol or "*",
            )
        ] += amount
        if strategy:
            horizon = trade_horizon or self.trade_horizon
            self.strategy_increment(strategy, timeframe or "*", horizon, "rejected", amount)

    def reconciliation(self) -> list[str]:
        """Return accounting errors for the stages whose units partition exactly."""

        errors: list[str] = []
        registry_total = int(self.counts["registry_eligible"]) + int(
            self.counts["registry_blocked"]
        )
        setups = int(self.counts["technical_setups"])
        if setups and registry_total != setups:
            errors.append(
                f"technical_setups={setups} != registry_eligible+blocked={registry_total}"
            )
        deterministic_total = int(self.counts["deterministic_qualified"]) + int(
            self.counts["deterministic_rejected"]
        )
        admitted = int(self.counts["registry_eligible"])
        if admitted and deterministic_total != admitted:
            errors.append(
                "registry_eligible="
                f"{admitted} != deterministic_qualified+rejected={deterministic_total}"
            )
        validation_total = int(self.counts["signal_validation_pass"]) + int(
            self.counts["signal_validation_reject"]
        )
        qwen_outcomes = (
            int(self.counts["qwen_external_calls"])
            + int(self.counts["qwen_cache_hits"])
            + int(self.counts["qwen_skipped_clear"])
            + int(self.counts["qwen_deferred"])
        )
        if validation_total and validation_total > qwen_outcomes:
            errors.append(
                f"validation outcomes={validation_total} exceed context outcomes={qwen_outcomes}"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        errors = self.reconciliation()
        result = dict(self.counts)
        result["reconciled"] = not errors
        result["reconciliation_errors"] = errors
        return result

    def rejection_rows(self) -> list[dict[str, Any]]:
        totals_by_stage: Counter[str] = Counter()
        for (stage, _reason), count in self._rejections.items():
            totals_by_stage[stage] += count
        return [
            {
                "stage": stage,
                "reason": reason,
                "count": count,
                "percentage": (
                    100.0 * count / totals_by_stage[stage] if totals_by_stage[stage] else 0.0
                ),
            }
            for (stage, reason), count in self._rejections.most_common()
        ]

    def rejection_drilldown(self, *, limit: int = 500) -> list[dict[str, Any]]:
        return [
            {
                "stage": stage,
                "reason": reason,
                "strategy": strategy,
                "timeframe": timeframe,
                "symbol": symbol,
                "count": count,
            }
            for (stage, reason, strategy, timeframe, symbol), count in self._details.most_common(
                limit
            )
        ]

    def strategy_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for grain, metrics in sorted(self._strategies.items()):
            strategy, timeframe, horizon = grain.split("|", 2)
            rows.append(
                {
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "trade_horizon": horizon,
                    **{key: int(value) for key, value in metrics.items()},
                }
            )
        return rows


def strategy_eligibility_rows(
    registry: StrategyEligibilityRegistry,
) -> list[dict[str, Any]]:
    """Dashboard-safe records from the active eligibility registry."""

    return eligibility_registry_rows(registry)
