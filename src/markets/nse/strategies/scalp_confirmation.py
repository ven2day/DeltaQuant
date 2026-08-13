"""
Deterministic multi-timeframe scalp confirmation (req 10).

Consumes a symbol's ``assessment_matrix.py`` output directly and checks whether
enough of the configured timeframe roles agree before a scalp candidate is allowed
to proceed:

    5m  -> execution/setup
    15m -> primary setup/confirmation
    30m -> directional confirmation
    1h  -> context
    4h  -> optional macro filter (skippable via settings.scalp_macro_filter_enabled)

Every threshold (which timeframes participate, how many must align, whether the 4h
leg counts at all) is a named ``settings.scalp_*`` field -- none of it is hardcoded
here. A timeframe that's simply missing from the matrix (no data, not even
attempted) is treated as NOT confirming, same as an explicit REJECT/WAIT cell --
absence is never silently counted as alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings
from src.core.indicators import Timeframe

from .assessment_matrix import TimeframeAssessment

# Timeframe -> (settings flag enabling this role, human label). 4h is the only
# optional role; every other role is always evaluated when its data is present.
_ROLE_TIMEFRAMES: tuple[tuple[Timeframe, str], ...] = (
    (Timeframe.M5, "execution"),
    (Timeframe.M15, "primary"),
    (Timeframe.M30, "directional"),
    (Timeframe.H1, "context"),
    (Timeframe.H4, "macro"),
)


@dataclass(frozen=True)
class ScalpConfirmationResult:
    execution_ok: bool
    primary_ok: bool
    directional_ok: bool
    context_ok: bool
    macro_ok: bool | None  # None when scalp_macro_filter_enabled=False (not evaluated)
    aligned_count: int
    required: int
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_ok": self.execution_ok,
            "primary_ok": self.primary_ok,
            "directional_ok": self.directional_ok,
            "context_ok": self.context_ok,
            "macro_ok": self.macro_ok,
            "aligned_count": self.aligned_count,
            "required": self.required,
            "passed": self.passed,
            "reasons": self.reasons,
        }


def _role_ok(matrix: dict[Timeframe, TimeframeAssessment], timeframe: Timeframe) -> bool:
    """True only when this timeframe has a confirming ('BUY') cell. A timeframe
    absent from ``matrix`` entirely (no data was even fetched/scanned) is NOT ok --
    fails closed exactly like an explicit REJECT/WAIT cell, never treated as
    "assume aligned"."""
    assessment = matrix.get(timeframe)
    return assessment is not None and assessment.decision == "BUY"


def confirm_multi_timeframe(matrix: dict[Timeframe, TimeframeAssessment]) -> ScalpConfirmationResult:
    settings = get_settings()

    execution_ok = _role_ok(matrix, Timeframe.M5)
    primary_ok = _role_ok(matrix, Timeframe.M15)
    directional_ok = _role_ok(matrix, Timeframe.M30)
    context_ok = _role_ok(matrix, Timeframe.H1)
    macro_ok: bool | None = (
        _role_ok(matrix, Timeframe.H4) if settings.scalp_macro_filter_enabled else None
    )

    eligible = [execution_ok, primary_ok, directional_ok, context_ok]
    if macro_ok is not None:
        eligible.append(macro_ok)
    aligned_count = sum(1 for ok in eligible if ok)
    required = settings.scalp_required_mtf_alignment
    passed = aligned_count >= required

    reasons = [
        f"5m execution={'ok' if execution_ok else 'not confirmed'}",
        f"15m primary={'ok' if primary_ok else 'not confirmed'}",
        f"30m directional={'ok' if directional_ok else 'not confirmed'}",
        f"1h context={'ok' if context_ok else 'not confirmed'}",
    ]
    if macro_ok is not None:
        reasons.append(f"4h macro={'ok' if macro_ok else 'not confirmed'}")
    else:
        reasons.append("4h macro filter disabled, not evaluated")
    reasons.append(f"{aligned_count}/{len(eligible)} timeframes aligned (need {required})")

    return ScalpConfirmationResult(
        execution_ok=execution_ok,
        primary_ok=primary_ok,
        directional_ok=directional_ok,
        context_ok=context_ok,
        macro_ok=macro_ok,
        aligned_count=aligned_count,
        required=required,
        passed=passed,
        reasons=reasons,
    )
