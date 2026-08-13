"""
Canonical symbol x timeframe assessment matrix (req 3).

For a given symbol, produces one consolidated BUY/WAIT/REJECT assessment per
timeframe -- score, strategy consensus, ML probability, and regime compatibility --
instead of leaving a human to eyeball a pile of raw per-strategy signal counts.

Pure function, no I/O: it consumes already-computed indicators/consolidated signals/
predictions and returns a plain dict. It reuses ``infer_signal_regime``/
``_ml_probability`` from ``signal_ranking.py`` (the same evidence the swing ranker
already uses) rather than re-deriving regime/ML-agreement logic a second time, and
``is_regime_compatible`` from ``regime_compatibility.py`` for the compatibility flag.

The ``decision`` field here is descriptive only -- a display/funnel-visibility label,
never an admission decision. A cell reading "BUY" still has to independently clear
strategy eligibility and every risk_compliance check before it can become a real trade.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from src.agents.prediction import PredictionSignal
from src.config import get_settings
from src.core.aggregation import ConsolidatedSignal
from src.core.candidates import SignalType
from src.core.indicators import IndicatorResult, Timeframe

from .regime_compatibility import is_regime_compatible
from .signal_ranking import _ml_probability, infer_signal_regime

Decision = Literal["BUY", "WAIT", "REJECT"]


@dataclass(frozen=True)
class TimeframeAssessment:
    """One symbol's consolidated assessment at one timeframe."""

    timeframe: Timeframe
    decision: Decision
    score: float
    strategy_consensus: int
    ml_probability: float | None
    regime_compatible: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe.value,
            "decision": self.decision,
            "score": self.score,
            "strategy_consensus": self.strategy_consensus,
            "ml_probability": self.ml_probability,
            "regime_compatible": self.regime_compatible,
            "reasons": self.reasons,
        }


def _assess_one_timeframe(
    timeframe: Timeframe,
    candidates: list[ConsolidatedSignal],
    predictions: Mapping[tuple[str, str], PredictionSignal],
    cycle_regime: str,
    *,
    reject_score: float,
    buy_score: float,
) -> TimeframeAssessment:
    if not candidates:
        return TimeframeAssessment(
            timeframe=timeframe,
            decision="REJECT",
            score=0.0,
            strategy_consensus=0,
            ml_probability=None,
            regime_compatible=False,
            reasons=[f"no strategy signal generated at {timeframe.value}"],
        )

    # One cell per symbol/timeframe (req 3): pick the strongest-evidenced direction
    # when more than one consolidated signal exists at this timeframe (e.g. a BUY
    # from one strategy group and a SELL from another).
    best = max(candidates, key=lambda c: c.blended_confidence)

    # Infer THIS timeframe's own regime from its own trend/ADX evidence, rather than
    # reusing one regime label collapsed across the whole cycle -- market_regime_node
    # only ever classifies off a single "primary" timeframe's indicators per symbol
    # today, so a 5m cell and a 4h cell can legitimately disagree (e.g. 5m ranging
    # inside a 4h uptrend) and this matrix is exactly where that distinction should
    # become visible instead of being silently flattened away.
    local_regime = infer_signal_regime(best.representative_signal)

    prediction = predictions.get((best.symbol, timeframe.value))
    ml_probability = _ml_probability(best.representative_signal, prediction)
    compatible, compat_reason = is_regime_compatible(
        best.representative_signal.strategy.value, local_regime
    )

    score = best.blended_confidence
    if ml_probability is not None:
        score = 0.6 * best.blended_confidence + 0.4 * ml_probability
    # Compatibility remains descriptive. The versioned eligibility record owns any
    # ALLOW / REDUCE_CONFIDENCE / BLOCK action and supplies the actual multiplier.

    reasons = [
        f"{best.agreement_count} strateg{'y' if best.agreement_count == 1 else 'ies'} "
        f"agree on {best.signal_type.value} "
        f"({', '.join(s.value for s in best.contributing_strategies)})",
        compat_reason,
    ]
    if local_regime != cycle_regime:
        reasons.append(
            f"this timeframe's own regime ('{local_regime}') differs from the "
            f"cycle-wide regime ('{cycle_regime}')"
        )
    if ml_probability is not None:
        reasons.append(f"ML agreement probability {ml_probability:.0%}")

    # This matrix only ever labels a cell "BUY" (never "SELL") -- matching the rest
    # of the pipeline's long-only convention (settings.long_only; see
    # src/markets/nse/runtime/live.py's swing scan, which drops SELL signals before they
    # even reach ranking when long-only is on). A SELL-dominant cell is therefore
    # capped at REJECT here regardless of score -- it must never be mislabeled "BUY"
    # just because it scored well in the opposite direction.
    if best.signal_type != SignalType.BUY:
        decision: Decision = "REJECT"
        reasons.append(
            f"strongest signal at {timeframe.value} is {best.signal_type.value}, "
            "not evaluated as a long-only entry"
        )
    elif score < reject_score:
        decision = "REJECT"
    elif score >= buy_score and compatible:
        decision = "BUY"
    else:
        decision = "WAIT"
        if score >= buy_score and not compatible:
            reasons.append("score clears the bar but regime compatibility does not")

    return TimeframeAssessment(
        timeframe=timeframe,
        decision=decision,
        score=score,
        strategy_consensus=best.agreement_count,
        ml_probability=ml_probability,
        regime_compatible=compatible,
        reasons=reasons,
    )


def build_assessment_matrix(
    symbol: str,
    consolidated_by_tf: Mapping[Timeframe, list[ConsolidatedSignal]],
    predictions: Mapping[tuple[str, str], PredictionSignal],
    regime: str,
    *,
    indicators_by_tf: Mapping[Timeframe, IndicatorResult] | None = None,
) -> dict[Timeframe, TimeframeAssessment]:
    """Build one ``TimeframeAssessment`` per timeframe in ``consolidated_by_tf``.

    ``regime`` is the cycle-wide regime label (from ``market_regime_node``). Each
    cell's own ``regime_compatible`` flag is actually computed against that
    timeframe's OWN locally-inferred regime (see ``_assess_one_timeframe``), not this
    one collapsed label -- ``regime`` is used only to flag when a timeframe's local
    regime diverges from the cycle-wide one, which is itself useful signal.

    ``indicators_by_tf`` is accepted but currently unused by the scoring itself
    (all the evidence it would add -- regime, ADX/DI -- already flows through each
    signal's own ``indicators`` dict via ``infer_signal_regime``); it's kept as an
    explicit parameter so a caller's full per-symbol indicator set is available to
    this function without a signature change if a future scoring refinement needs it
    directly (e.g. a raw ATR/volume check not carried on any individual signal).
    """
    del indicators_by_tf  # reserved, see docstring
    settings = get_settings()
    matrix: dict[Timeframe, TimeframeAssessment] = {}
    for timeframe, candidates in consolidated_by_tf.items():
        matrix[timeframe] = _assess_one_timeframe(
            timeframe,
            candidates,
            predictions,
            regime,
            reject_score=settings.scalp_matrix_reject_score,
            buy_score=settings.scalp_min_confidence,
        )
    return matrix
