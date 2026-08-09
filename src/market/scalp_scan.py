"""
One cycle's scalp-horizon scan (Stage 10): raw signal generation across the full
confirmation-timeframe set -> consolidation -> assessment matrix -> multi-timeframe
confirmation -> entry-quality evaluation -> ScalpOpportunity assembly -> regime
pre-filter -> ranking.

VISIBILITY ONLY. This module never invokes the LangGraph agent pipeline and never
calls ExecutionService -- it has no import of either, so it structurally cannot
place a trade. H-8 admission and execution are wired in later stages (see
CLAUDE.md "Scalp horizon").

Extracted out of scripts/run_live_trading.py's ``_run_cycle`` closure specifically
so this logic is unit-testable with plain async stub functions instead of mocking
the entire live-session object graph (market_manager, history_manager, dashboard,
execution_service, ...) that closure depends on.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pandas as pd

from .assessment_matrix import build_assessment_matrix
from .entry_quality import evaluate_entry_quality
from .indicators import IndicatorResult, Timeframe
from .regime_compatibility import filter_regime_compatible
from .scalp_confirmation import confirm_multi_timeframe
from .scalp_opportunity import ScalpOpportunity
from .scalp_ranking import rank_scalp_opportunities
from .signal_consolidation import ConsolidatedSignal, consolidate_signals
from .signals import SignalEngine

if TYPE_CHECKING:
    from src.config.settings import Settings

    from .signal_ranking import PerformanceTrackerLike

FUNNEL_KEYS = (
    "raw_triggers",
    "consolidated",
    "mtf_candidates",
    "entry_quality_passed",
    "regime_compatible",
    "h8_admitted",
    "sent_to_ai",
    "ai_approved",
    "execution_accepted",
)


def empty_funnel() -> dict[str, int]:
    return dict.fromkeys(FUNNEL_KEYS, 0)


async def run_scalp_scan(
    scan_symbols: list[str],
    *,
    settings: Settings,
    scalp_signal_engine: SignalEngine,
    scalp_confirmation_timeframes: list[Timeframe],
    scalp_origin_timeframes: list[Timeframe],
    scalp_cycle_regime: str,
    performance_tracker: PerformanceTrackerLike,
    fetch_indicators: Callable[[str, Timeframe], Awaitable[IndicatorResult | None]],
    fetch_5m_frame: Callable[[str], Awaitable[pd.DataFrame | None]],
) -> tuple[list[ScalpOpportunity], dict[str, int]]:
    """Returns ``(ranked_opportunities, funnel_counters)``.

    ``fetch_indicators``/``fetch_5m_frame`` are injected so this function has no
    direct dependency on ``HistoryManager``/``asyncio.to_thread`` -- the live loop
    wires them to the real history manager; tests wire them to simple stubs.
    """
    funnel = empty_funnel()
    opportunities: list[ScalpOpportunity] = []

    for symbol in scan_symbols:
        indicators_by_tf: dict[Timeframe, IndicatorResult] = {}
        for tf in scalp_confirmation_timeframes:
            ind = await fetch_indicators(symbol, tf)
            if ind is not None:
                indicators_by_tf[tf] = ind
        if not indicators_by_tf:
            continue

        symbol_signals = []
        for ind in indicators_by_tf.values():
            symbol_signals.extend(scalp_signal_engine.generate_signals(ind))
        if settings.long_only:
            symbol_signals = [s for s in symbol_signals if s.signal_type.value == "BUY"]
        if not symbol_signals:
            continue
        funnel["raw_triggers"] += len(symbol_signals)

        consolidated = consolidate_signals(symbol_signals)
        funnel["consolidated"] += len(consolidated)
        consolidated_by_tf: dict[Timeframe, list[ConsolidatedSignal]] = defaultdict(list)
        for c in consolidated:
            consolidated_by_tf[c.timeframe].append(c)

        matrix = build_assessment_matrix(
            symbol,
            consolidated_by_tf,
            {},  # ML predictions not run for the scalp path yet (Stage 10 scope)
            scalp_cycle_regime,
        )

        confirmation = confirm_multi_timeframe(matrix)
        if not confirmation.passed:
            continue
        funnel["mtf_candidates"] += 1

        origin_tf = next(
            (
                tf
                for tf in scalp_origin_timeframes
                if matrix.get(tf) is not None and matrix[tf].decision == "BUY"
            ),
            None,
        )
        if origin_tf is None:
            continue
        origin_candidate = max(consolidated_by_tf[origin_tf], key=lambda c: c.blended_confidence)
        representative_signal = origin_candidate.representative_signal

        frame_5m = await fetch_5m_frame(symbol)
        entry_quality = evaluate_entry_quality(
            symbol,
            "BUY",
            frame_5m,
            indicators_by_tf.get(Timeframe.M5),
            representative_signal.stop_loss,
            representative_signal.target_price,
        )
        if entry_quality.status == "REJECT":
            continue
        funnel["entry_quality_passed"] += 1

        opportunities.append(
            ScalpOpportunity(
                symbol=symbol,
                direction="BUY",
                timeframe_states={tf.value: a for tf, a in matrix.items()},
                primary_strategy=representative_signal.strategy.value,
                primary_timeframe=origin_tf.value,
                entry_quality=entry_quality,
                mtf_confirmation=confirmation,
                regime_compatible=matrix[origin_tf].regime_compatible,
                ml_probability=None,
                final_decision=entry_quality.status,
                reason=list(entry_quality.reasons),
                entry_price=representative_signal.entry_price,
                preferred_entry_low=entry_quality.preferred_entry_low,
                preferred_entry_high=entry_quality.preferred_entry_high,
                stop_loss=representative_signal.stop_loss,
                target_price=representative_signal.target_price,
                # Simplified pass-through pending the scalp ML ensemble (not wired
                # in this visibility-only stage); the ranker below independently
                # scores each opportunity on richer evidence than this one field.
                expected_r=representative_signal.risk_reward_ratio,
            )
        )

    compatible, _rejected_for_regime = filter_regime_compatible(
        opportunities, scalp_cycle_regime
    )
    funnel["regime_compatible"] = len(compatible)

    ranked = rank_scalp_opportunities(compatible, performance_tracker, scalp_cycle_regime)
    ranked = ranked[: settings.scalp_max_active_symbols]
    final_opportunities = [dataclasses.replace(r.opportunity, score=r.rank_score) for r in ranked]

    return final_opportunities, funnel
