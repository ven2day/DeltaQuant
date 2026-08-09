"""
Deterministic regime-compatibility rules (req 7: skip obviously incompatible
strategy/regime combinations before spending an LLM call on them).

``REGIME_STRATEGY_COMPATIBILITY`` is the single tunable table -- the same domain
knowledge already encoded informally in ``strategy_selection.py``'s
``_fallback_strategy_selection`` regime->strategies map, made explicit and reusable
here so both the pre-LLM assessment matrix (``assessment_matrix.py``) and the scalp
opportunity filter (wired in a later stage) read one source of truth instead of two
copies drifting apart.

IMPORTANT -- what this module is NOT: it is a cost/efficiency filter only, applied
before the LLM review stage to avoid spending a Groq call on a combination that's
obviously a poor fit. It has no side channel into ``active_strategies`` or the H-8
registry and must never be treated as an admission decision. A regime-compatible
signal still has to independently clear ``strategy_selection``'s H-8 gate and
``risk_compliance``'s checks (including the H-8 backstop) exactly like every other
candidate -- see src/backtesting/strategy_registry.py. Do not "optimize" this module
into doing that job; it structurally cannot, since it never touches the registry.

Being permissive on an unrecognized regime label is deliberate: false negatives here
(letting a maybe-incompatible signal reach the LLM) cost a small amount of LLM spend;
false positives (blocking a genuinely compatible strategy) would silently suppress
real opportunities with zero recourse. The real safety gate is H-8, downstream.
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

    COST FILTER ONLY -- see module docstring. Never a substitute for H-8: a
    compatible opportunity here still independently has to clear
    ``strategy_selection``'s admission gate and every ``risk_compliance`` check.
    This function has no import/reference to ``StrategyRegistry`` at all, so it
    cannot be "extended" into doing that job by accident.
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
