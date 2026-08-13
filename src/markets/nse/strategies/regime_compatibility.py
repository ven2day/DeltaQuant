"""
Deterministic regime-compatibility rules (req 7: skip obviously incompatible
strategy/regime combinations before spending an LLM call on them).

``REGIME_STRATEGY_COMPATIBILITY`` is the single tunable table -- the same domain
knowledge already encoded informally in ``strategy_selection.py``'s
``_fallback_strategy_selection`` regime->strategies map, made explicit and reusable
here so both the pre-LLM assessment matrix (``assessment_matrix.py``) and the scalp
opportunity filter (wired in a later stage) read one source of truth instead of two
copies drifting apart.

IMPORTANT -- this table is descriptive legacy context only. Runtime eligibility,
confidence reduction, and blocking come from StrategyEligibilityRegistry records.
This module cannot admit a candidate and is not consulted by deterministic risk.

Being permissive on an unrecognized regime label is deliberate: false negatives here
(letting a maybe-incompatible signal reach the LLM) cost a small amount of LLM spend;
false positives (blocking a genuinely compatible strategy) would silently suppress
real opportunities with zero recourse. Versioned runtime regime policy is downstream.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scalp_opportunity import ScalpOpportunity

# Mirrors src/agents/strategy_selection.py's _fallback_strategy_selection regime->
# strategies map (the same underlying domain knowledge -- "which strategies suit
# which market regime" -- already encoded there for the LLM-unavailable fallback path).
# Edit this table directly to tune; no code change needed elsewhere.
REGIME_STRATEGY_COMPATIBILITY: dict[str, set[str]] = {
    "trending_up": {
        "momentum",
        "trend_following",
        "ema_heiken_ashi_rsi",
        "ema_psar",
        "ema_cci",
    },
    "trending_down": {
        "trend_following",
        "ema_heiken_ashi_rsi",
        "ema_psar",
        "ema_cci",
    },
    "ranging": {"mean_reversion"},
    "volatile": {"breakout"},
}


def is_regime_compatible(strategy_name: str, regime: str) -> tuple[bool, str]:
    """Whether ``strategy_name`` is a typical fit for ``regime``.

    Returns ``(compatible, reason)``. An unrecognized regime label (e.g. "unknown")
    is treated as compatible with everything -- see module docstring.
    """
    compatible_strategies = REGIME_STRATEGY_COMPATIBILITY.get(regime)
    if compatible_strategies is None:
        return True, f"regime '{regime}' is not in the compatibility table; no filtering applied"
    if strategy_name in compatible_strategies:
        return True, f"'{strategy_name}' is a typical fit for regime '{regime}'"
    return False, f"'{strategy_name}' is not a typical fit for regime '{regime}'"


def filter_regime_compatible(
    opportunities: list[ScalpOpportunity], regime: str
) -> tuple[list[ScalpOpportunity], list[ScalpOpportunity]]:
    """Partition ``opportunities`` into ``(compatible, rejected)`` before any LLM
    review (req 7), using each opportunity's already-computed ``regime_compatible``
    flag -- set when the opportunity was assembled from its assessment matrix (each
    cell's own ``regime_compatible`` already comes from ``is_regime_compatible`` via
    ``assessment_matrix.py``, so this function doesn't re-derive anything, just
    partitions on it). Rejected opportunities get an explanatory reason appended
    (opportunities are frozen dataclasses, so this returns a new instance rather
    than mutating).

    LEGACY ADVISORY PARTITION ONLY -- see module docstring. It is never a substitute
    for StrategyEligibilityRegistry or deterministic risk.
    """
    compatible: list[ScalpOpportunity] = []
    rejected: list[ScalpOpportunity] = []
    for opportunity in opportunities:
        if opportunity.regime_compatible:
            compatible.append(opportunity)
        else:
            rejected.append(
                dataclasses.replace(
                    opportunity,
                    reason=[*opportunity.reason, f"not a typical fit for regime '{regime}'"],
                )
            )
    return compatible, rejected
