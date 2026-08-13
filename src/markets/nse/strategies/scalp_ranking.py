"""
Scalp-horizon ranker/policy (req 2, 9) -- a DIFFERENT weighted formula from
``signal_ranking.rank_signals()``, not a horizon parameter bolted onto it, because
scalp and swing candidates should be judged on different evidence: entry timing
quality and multi-timeframe alignment matter far more for a 5m/15m trade than they
do for a multi-hour swing position, while a single strategy's long-run historical
win rate matters less (scalp sample sizes are smaller and regimes shift faster).

Every weight is a named ``settings.scalp_ranking_weight_*`` field
(src/config/settings.py), validated at startup to sum to ~1.0 -- no unexplained
constant in the formula below.

Historical scalp expectancy is looked up under a ``"scalp::<strategy>"`` namespaced
key (``_scalp_performance_key``) rather than a bare strategy name, so scalp and swing
track records never collide in ``PerformanceTracker`` -- and, since this repo has no
schema migration tooling (``Base.metadata.create_all()`` never alters an existing
table, see src/db/base.py), a free-text namespaced key is the correct fix, not a new
column. Cold start (0 samples) uses the identical Beta(2,2) smoothing
``signal_ranking.rank_signals()`` already uses, so an unproven scalp strategy reads
as a neutral 50%, never as "0% accurate" or silently "good enough".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import get_settings

from .scalp_confirmation import ScalpConfirmationResult
from .scalp_opportunity import ScalpOpportunity
from .signal_ranking import PerformanceTrackerLike

# Neutral prior for ML probability when no prediction is available -- "no
# information", not a penalty. Matches signal_ranking.py's treatment of an
# abstained/missing prediction (never counted as evidence against a signal).
_NEUTRAL_ML_PROBABILITY = 0.5

_ENTRY_QUALITY_STATUS_SCORE: dict[str, float] = {
    "ENTER_NOW": 1.0,
    "WAIT_BREAKOUT": 0.5,
    "WAIT_PULLBACK": 0.5,
    "REJECT": 0.0,
}


def scalp_performance_key(strategy: str) -> str:
    """Namespaced PerformanceTracker lookup key so scalp and swing track records for
    the same strategy name never collide -- see module docstring."""
    return f"scalp::{strategy}"


@dataclass(frozen=True)
class ScalpRankingWeights:
    entry_quality: float
    mtf_alignment: float
    volume_liquidity: float
    regime: float
    historical_expectancy: float
    ml_probability: float

    @classmethod
    def from_settings(cls) -> ScalpRankingWeights:
        settings = get_settings()
        return cls(
            entry_quality=settings.scalp_ranking_weight_entry_quality,
            mtf_alignment=settings.scalp_ranking_weight_mtf_alignment,
            volume_liquidity=settings.scalp_ranking_weight_volume_liquidity,
            regime=settings.scalp_ranking_weight_regime,
            historical_expectancy=settings.scalp_ranking_weight_historical_expectancy,
            ml_probability=settings.scalp_ranking_weight_ml_probability,
        )


@dataclass(frozen=True)
class RankedScalpOpportunity:
    """A ScalpOpportunity plus the weighted evidence used to order it -- every
    component score is independently visible, not just the final blend, matching
    ``signal_ranking.RankedSignal``'s "show your work" precedent."""

    opportunity: ScalpOpportunity
    rank_score: float
    entry_quality_score: float
    mtf_alignment_score: float
    volume_liquidity_score: float
    regime_score: float
    historical_win_probability: float
    historical_sample_size: int
    ml_probability_score: float


def _entry_quality_score(opportunity: ScalpOpportunity) -> float:
    if opportunity.entry_quality is None:
        return 0.0
    return _ENTRY_QUALITY_STATUS_SCORE.get(opportunity.entry_quality.status, 0.0)


def _mtf_alignment_score(confirmation: ScalpConfirmationResult | None) -> float:
    if confirmation is None or confirmation.required <= 0:
        return 0.0
    return min(1.0, confirmation.aligned_count / confirmation.required)


def _volume_liquidity_score(opportunity: ScalpOpportunity, min_ratio: float) -> float:
    if opportunity.entry_quality is None:
        return 0.0
    # Full credit at 2x the configured minimum; never negative, never unbounded.
    ceiling = max(min_ratio, 0.01) * 2.0
    return min(1.0, opportunity.entry_quality.relative_volume / ceiling)


def _historical_win_probability(
    strategy: str,
    regime: str,
    performance_tracker: PerformanceTrackerLike,
    lookback_days: int,
) -> tuple[float, int]:
    perf = performance_tracker.get_strategy_performance(
        scalp_performance_key(strategy), regime, lookback_days
    )
    sample_size = perf.total_trades
    # Same Beta(2,2) smoothing as signal_ranking.rank_signals() -- 0 samples reads
    # as a neutral 50%, never "unproven-but-fine" nor "0% accurate".
    probability = (perf.winning_trades + 2) / (sample_size + 4)
    return probability, sample_size


def rank_scalp_opportunities(
    opportunities: list[ScalpOpportunity],
    performance_tracker: PerformanceTrackerLike,
    regime: str,
    *,
    weights: ScalpRankingWeights | None = None,
    lookback_days: int = 30,
) -> list[RankedScalpOpportunity]:
    """Score and sort ``opportunities`` strongest-first using the scalp-specific
    weighted formula (see module docstring). Pure function: no I/O beyond the
    injected ``performance_tracker``.
    """
    weights = weights or ScalpRankingWeights.from_settings()
    settings = get_settings()

    ranked: list[RankedScalpOpportunity] = []
    for opportunity in opportunities:
        entry_quality_score = _entry_quality_score(opportunity)
        mtf_alignment_score = _mtf_alignment_score(opportunity.mtf_confirmation)
        volume_liquidity_score = _volume_liquidity_score(
            opportunity, settings.scalp_min_volume_ratio
        )
        regime_score = 1.0 if opportunity.regime_compatible else 0.0
        historical_win_probability, historical_sample_size = _historical_win_probability(
            opportunity.primary_strategy, regime, performance_tracker, lookback_days
        )
        ml_probability_score = (
            opportunity.ml_probability
            if opportunity.ml_probability is not None
            else _NEUTRAL_ML_PROBABILITY
        )

        rank_score = (
            weights.entry_quality * entry_quality_score
            + weights.mtf_alignment * mtf_alignment_score
            + weights.volume_liquidity * volume_liquidity_score
            + weights.regime * regime_score
            + weights.historical_expectancy * historical_win_probability
            + weights.ml_probability * ml_probability_score
        )

        ranked.append(
            RankedScalpOpportunity(
                opportunity=opportunity,
                rank_score=rank_score,
                entry_quality_score=entry_quality_score,
                mtf_alignment_score=mtf_alignment_score,
                volume_liquidity_score=volume_liquidity_score,
                regime_score=regime_score,
                historical_win_probability=historical_win_probability,
                historical_sample_size=historical_sample_size,
                ml_probability_score=ml_probability_score,
            )
        )

    ranked.sort(key=lambda r: r.rank_score, reverse=True)
    return ranked
