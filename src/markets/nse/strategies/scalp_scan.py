"""
One cycle's scalp-horizon scan (Stage 10): raw signal generation across the full
confirmation-timeframe set -> consolidation -> assessment matrix -> multi-timeframe
confirmation -> entry-quality evaluation -> ScalpOpportunity assembly -> regime
pre-filter -> ranking.

VISIBILITY ONLY. This module never invokes the LangGraph agent pipeline and never
calls ExecutionService -- it has no import of either, so it structurally cannot
place a trade. Strategy eligibility and execution are wired in later stages (see
CLAUDE.md "Scalp horizon").

Extracted out of src/markets/nse/runtime/live.py's ``_run_cycle`` closure specifically
so this logic is unit-testable with plain async stub functions instead of mocking
the entire live-session object graph (market_manager, history_manager, dashboard,
execution_service, ...) that closure depends on.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from src.core.aggregation import ConsolidatedSignal, consolidate_signals
from src.core.candidates import SignalEngine
from src.core.indicators import IndicatorResult, Timeframe

from .assessment_matrix import TimeframeAssessment, build_assessment_matrix
from .entry_quality import EntryQualityResult, evaluate_entry_quality
from .scalp_confirmation import ScalpConfirmationResult, confirm_multi_timeframe
from .scalp_opportunity import ScalpOpportunity
from .scalp_ranking import rank_scalp_opportunities

if TYPE_CHECKING:
    from src.agents.prediction import PredictionSignal
    from src.config.settings import Settings

    from .signal_ranking import PerformanceTrackerLike

FUNNEL_KEYS = (
    "raw_triggers",
    "consolidated",
    "mtf_candidates",
    "mtf_confirmed",
    "mtf_conflict",
    "mtf_missing_data",
    "entry_quality_passed",
    "regime_compatible",
    "registry_eligible",
    "regime_blocked",
    "confidence_reduced",
    "sent_to_ai",
    "ai_approved",
    "execution_accepted",
    # Provider-neutral optimization aliases used by the combined SWING/SCALP cost
    # summary. Existing UI keys above remain unchanged for compatibility.
    "raw_candidates",
    "consolidated_candidates",
    "mtf_pass",
    "entry_pass",
    "technical_setups",
    "ml_candidates",
    "ml_qualified",
    "ml_scored",
    "regime_pass",
    "registry_pass",
    "pre_llm_risk_pass",
    "qwen_candidates",
    "qwen_calls",
    "qwen_cache_hits",
    "qwen_cache_misses",
    "qwen_approved",
)


def empty_funnel() -> dict[str, int]:
    return dict.fromkeys(FUNNEL_KEYS, 0)


ScalpUniverseDecision = Literal["BUY", "NO_BUY", "NO_DATA"]


@dataclasses.dataclass(frozen=True)
class ScalpSymbolStatus:
    """Deterministic scalp-signal result for one symbol in the configured universe.

    ``BUY`` means the symbol cleared the technical scalp scan at the latest completed
    candle. It does not mean an order was approved: ranking, strategy eligibility, the agent
    review, risk checks, quote freshness, and execution remain separate downstream
    controls. ``NO_BUY`` preserves the first failed stage and its reasons instead of
    silently dropping the symbol. ``NO_DATA`` is reserved for symbols with no usable
    indicator history at any configured scalp timeframe.
    """

    symbol: str
    decision: ScalpUniverseDecision
    stage: str
    reasons: list[str] = dataclasses.field(default_factory=list)
    available_timeframes: list[str] = dataclasses.field(default_factory=list)
    missing_timeframes: list[str] = dataclasses.field(default_factory=list)
    raw_buy_signals: int = 0
    timeframe_states: dict[str, TimeframeAssessment] = dataclasses.field(default_factory=dict)
    primary_strategy: str = ""
    primary_timeframe: str = ""
    entry_quality: EntryQualityResult | None = None
    mtf_confirmation: ScalpConfirmationResult | None = None
    regime_compatible: bool | None = None
    score: float = 0.0
    selected_for_review: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    expected_r: float = 0.0
    ml_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "stage": self.stage,
            "reasons": self.reasons,
            "available_timeframes": self.available_timeframes,
            "missing_timeframes": self.missing_timeframes,
            "raw_buy_signals": self.raw_buy_signals,
            "timeframe_states": {
                tf: assessment.to_dict() for tf, assessment in self.timeframe_states.items()
            },
            "primary_strategy": self.primary_strategy,
            "primary_timeframe": self.primary_timeframe,
            "entry_quality": self.entry_quality.to_dict() if self.entry_quality else None,
            "mtf_confirmation": (
                self.mtf_confirmation.to_dict() if self.mtf_confirmation else None
            ),
            "regime_compatible": self.regime_compatible,
            "score": self.score,
            "selected_for_review": self.selected_for_review,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "expected_r": self.expected_r,
            "ml_probability": self.ml_probability,
        }


@dataclasses.dataclass(frozen=True)
class ScalpScanResult:
    opportunities: list[ScalpOpportunity]
    funnel: dict[str, int]
    symbol_statuses: list[ScalpSymbolStatus]
    predictions: dict[tuple[str, str], PredictionSignal] = dataclasses.field(default_factory=dict)


def _probability_up(prediction: PredictionSignal | None) -> float | None:
    if prediction is None or prediction.abstained or prediction.direction == "flat":
        return None
    if prediction.direction == "up":
        return float(prediction.confidence)
    if prediction.direction == "down":
        return 1.0 - float(prediction.confidence)
    return None


async def run_scalp_scan_detailed(
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
    fetch_prediction: (
        Callable[[str, Timeframe, str], Awaitable[PredictionSignal | None]] | None
    ) = None,
) -> ScalpScanResult:
    """Return ranked opportunities, funnel counters, and one result per symbol.

    ``fetch_indicators``/``fetch_5m_frame`` are injected so this function has no
    direct dependency on ``HistoryManager``/``asyncio.to_thread`` -- the live loop
    wires them to the real history manager; tests wire them to simple stubs.
    """
    funnel = empty_funnel()
    opportunities: list[ScalpOpportunity] = []
    statuses_by_symbol: dict[str, ScalpSymbolStatus] = {}
    predictions: dict[tuple[str, str], PredictionSignal] = {}

    # De-duplicate without changing universe order. The dashboard contract is exactly
    # one status row per configured symbol, even if an upstream watchlist contains a
    # duplicate.
    for symbol in dict.fromkeys(scan_symbols):
        indicators_by_tf: dict[Timeframe, IndicatorResult] = {}
        for tf in scalp_confirmation_timeframes:
            ind = await fetch_indicators(symbol, tf)
            if ind is not None:
                indicators_by_tf[tf] = ind

        available_timeframes = [
            tf.value for tf in scalp_confirmation_timeframes if tf in indicators_by_tf
        ]
        missing_timeframes = [
            tf.value for tf in scalp_confirmation_timeframes if tf not in indicators_by_tf
        ]
        if not indicators_by_tf:
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_DATA",
                stage="DATA_UNAVAILABLE",
                reasons=["No usable indicator history was available for any scalp timeframe."],
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
            )
            continue

        symbol_signals = []
        for ind in indicators_by_tf.values():
            symbol_signals.extend(scalp_signal_engine.generate_signals(ind))
        if settings.long_only:
            symbol_signals = [s for s in symbol_signals if s.signal_type.value == "BUY"]
        if not symbol_signals:
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_BUY",
                stage="NO_BUY_TRIGGER",
                reasons=["No configured scalp strategy produced a BUY trigger."],
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
            )
            continue
        funnel["raw_triggers"] += len(symbol_signals)
        funnel["raw_candidates"] += len(symbol_signals)

        consolidated = consolidate_signals(symbol_signals)
        funnel["consolidated"] += len(consolidated)
        funnel["consolidated_candidates"] += len(consolidated)
        consolidated_by_tf: dict[Timeframe, list[ConsolidatedSignal]] = defaultdict(list)
        for c in consolidated:
            consolidated_by_tf[c.timeframe].append(c)

        funnel["technical_setups"] += 1
        # First establish that the symbol is a real deterministic setup.  ML must
        # never be invoked merely because one of seven strategies emitted a raw
        # trigger.  The final matrix is rebuilt with ML evidence below.
        technical_matrix = build_assessment_matrix(
            symbol,
            consolidated_by_tf,
            {},
            scalp_cycle_regime,
        )

        technical_confirmation = confirm_multi_timeframe(technical_matrix)
        if not technical_confirmation.passed:
            required_context = {
                Timeframe.M5,
                Timeframe.M15,
                Timeframe.M30,
                Timeframe.H1,
            }
            if settings.scalp_macro_filter_enabled:
                required_context.add(Timeframe.H4)
            required_context.intersection_update(scalp_confirmation_timeframes)
            missing_required = sorted(
                (tf.value for tf in required_context if tf not in indicators_by_tf),
                key=lambda value: scalp_confirmation_timeframes.index(Timeframe(value)),
            )
            missing_context = bool(missing_required)
            failure_stage = "MTF_MISSING_DATA" if missing_context else "MTF_CONFLICT"
            funnel["mtf_missing_data" if missing_context else "mtf_conflict"] += 1
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_BUY",
                stage=failure_stage,
                reasons=(
                    [
                        "Required settled multi-timeframe history is unavailable: "
                        + ", ".join(missing_required),
                        *technical_confirmation.reasons,
                    ]
                    if missing_context
                    else list(technical_confirmation.reasons)
                ),
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
                raw_buy_signals=len(symbol_signals),
                timeframe_states={
                    tf.value: assessment for tf, assessment in technical_matrix.items()
                },
                mtf_confirmation=technical_confirmation,
            )
            continue
        funnel["mtf_candidates"] += 1
        funnel["mtf_confirmed"] += 1
        funnel["mtf_pass"] += 1

        origin_tf = next(
            (
                tf
                for tf in scalp_origin_timeframes
                if technical_matrix.get(tf) is not None
                and technical_matrix[tf].decision == "BUY"
            ),
            None,
        )
        if origin_tf is None:
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_BUY",
                stage="NO_PRIMARY_SETUP",
                reasons=["No configured scalp-origin timeframe has a BUY assessment."],
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
                raw_buy_signals=len(symbol_signals),
                timeframe_states={
                    tf.value: assessment for tf, assessment in technical_matrix.items()
                },
                mtf_confirmation=technical_confirmation,
            )
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
        # WAIT_PULLBACK/WAIT_BREAKOUT are explicitly not actionable. Sending them to
        # Qwen cannot make the current entry valid and previously wasted a full graph
        # review before deterministic execution rejected/staled the opportunity.
        if entry_quality.status != "ENTER_NOW":
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_BUY",
                stage="ENTRY_NOT_READY",
                reasons=[
                    f"Entry timing is {entry_quality.status.replace('_', ' ').lower()}.",
                    *entry_quality.reasons,
                ],
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
                raw_buy_signals=len(symbol_signals),
                timeframe_states={
                    tf.value: assessment for tf, assessment in technical_matrix.items()
                },
                primary_strategy=representative_signal.strategy.value,
                primary_timeframe=origin_tf.value,
                entry_quality=entry_quality,
                mtf_confirmation=technical_confirmation,
                regime_compatible=technical_matrix[origin_tf].regime_compatible,
                entry_price=representative_signal.entry_price,
                stop_loss=representative_signal.stop_loss,
                target_price=representative_signal.target_price,
                expected_r=representative_signal.risk_reward_ratio,
                ml_probability=None,
            )
            continue
        funnel["entry_quality_passed"] += 1
        funnel["entry_pass"] += 1
        if fetch_prediction is not None:
            funnel["ml_candidates"] += 1

        # Only an MTF-confirmed, actionable entry becomes an ML candidate.  Fetch
        # the exact strategy/timeframe artifact for each material cell, then rebuild
        # the descriptive matrix with that evidence.
        if fetch_prediction is not None:
            for timeframe, timeframe_candidates in consolidated_by_tf.items():
                strongest = max(timeframe_candidates, key=lambda c: c.blended_confidence)
                strategy_version = strongest.representative_signal.strategy.value
                prediction = await fetch_prediction(symbol, timeframe, strategy_version)
                if prediction is not None:
                    predictions[(symbol, timeframe.value)] = prediction
                    if _probability_up(prediction) is not None:
                        funnel["ml_scored"] += 1

        matrix = build_assessment_matrix(
            symbol,
            consolidated_by_tf,
            predictions,
            scalp_cycle_regime,
        )
        confirmation = confirm_multi_timeframe(matrix)
        if not confirmation.passed:
            statuses_by_symbol[symbol] = ScalpSymbolStatus(
                symbol=symbol,
                decision="NO_BUY",
                stage="ML_NOT_CONFIRMED",
                reasons=list(confirmation.reasons),
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
                raw_buy_signals=len(symbol_signals),
                timeframe_states={tf.value: assessment for tf, assessment in matrix.items()},
                primary_strategy=representative_signal.strategy.value,
                primary_timeframe=origin_tf.value,
                entry_quality=entry_quality,
                mtf_confirmation=confirmation,
                regime_compatible=matrix[origin_tf].regime_compatible,
                entry_price=representative_signal.entry_price,
                stop_loss=representative_signal.stop_loss,
                target_price=representative_signal.target_price,
                expected_r=representative_signal.risk_reward_ratio,
                ml_probability=_probability_up(
                    predictions.get((symbol, origin_tf.value))
                ),
            )
            continue
        if fetch_prediction is not None:
            funnel["ml_qualified"] += 1
        ml_probability = _probability_up(predictions.get((symbol, origin_tf.value)))
        signal_candle_timestamp = ""
        if frame_5m is not None and len(frame_5m) > 0:
            last_index = frame_5m.index[-1]
            signal_candle_timestamp = (
                str(last_index.isoformat()) if hasattr(last_index, "isoformat") else str(last_index)
            )

        opportunity = ScalpOpportunity(
            symbol=symbol,
            direction="BUY",
            timeframe_states={tf.value: a for tf, a in matrix.items()},
            primary_strategy=representative_signal.strategy.value,
            primary_timeframe=origin_tf.value,
            entry_quality=entry_quality,
            mtf_confirmation=confirmation,
            regime_compatible=matrix[origin_tf].regime_compatible,
            ml_probability=ml_probability,
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
            signal_candle_timestamp=signal_candle_timestamp,
        )
        opportunities.append(opportunity)
        regime_compatible = opportunity.regime_compatible
        statuses_by_symbol[symbol] = ScalpSymbolStatus(
            symbol=symbol,
            decision="BUY",
            stage="BUY_SETUP",
            reasons=(
                [
                    "Technical scalp BUY setup is present now.",
                    *entry_quality.reasons,
                ]
                if regime_compatible
                else [
                    "Legacy regime compatibility is advisory; the eligibility registry "
                    f"will decide policy for '{scalp_cycle_regime}'.",
                    *entry_quality.reasons,
                ]
            ),
            available_timeframes=available_timeframes,
            missing_timeframes=missing_timeframes,
            raw_buy_signals=len(symbol_signals),
            timeframe_states={tf.value: assessment for tf, assessment in matrix.items()},
            primary_strategy=representative_signal.strategy.value,
            primary_timeframe=origin_tf.value,
            entry_quality=entry_quality,
            mtf_confirmation=confirmation,
            regime_compatible=regime_compatible,
            entry_price=representative_signal.entry_price,
            stop_loss=representative_signal.stop_loss,
            target_price=representative_signal.target_price,
            expected_r=representative_signal.risk_reward_ratio,
            ml_probability=ml_probability,
        )

    compatible = opportunities
    funnel["regime_compatible"] = len(compatible)
    funnel["regime_pass"] = len(compatible)

    ranked = rank_scalp_opportunities(compatible, performance_tracker, scalp_cycle_regime)
    # Keep every quality-qualified opportunity.  Ranking determines review order;
    # deterministic portfolio constraints decide which orders can actually coexist.
    selected = ranked
    final_opportunities = [dataclasses.replace(r.opportunity, score=r.rank_score) for r in ranked]
    rank_score_by_symbol = {
        ranked_item.opportunity.symbol: ranked_item.rank_score for ranked_item in ranked
    }
    selected_symbols = {ranked_item.opportunity.symbol for ranked_item in selected}

    for symbol, status in list(statuses_by_symbol.items()):
        if status.decision == "BUY":
            statuses_by_symbol[symbol] = dataclasses.replace(
                status,
                score=rank_score_by_symbol.get(symbol, 0.0),
                selected_for_review=symbol in selected_symbols,
            )

    decision_order = {"BUY": 0, "NO_BUY": 1, "NO_DATA": 2}
    symbol_statuses = sorted(
        statuses_by_symbol.values(),
        key=lambda status: (decision_order[status.decision], -status.score, status.symbol),
    )

    return ScalpScanResult(
        opportunities=final_opportunities,
        funnel=funnel,
        symbol_statuses=symbol_statuses,
        predictions=predictions,
    )


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
    fetch_prediction: (
        Callable[[str, Timeframe, str], Awaitable[PredictionSignal | None]] | None
    ) = None,
) -> tuple[list[ScalpOpportunity], dict[str, int]]:
    """Compatibility wrapper returning the original two-value scan result."""
    result = await run_scalp_scan_detailed(
        scan_symbols,
        settings=settings,
        scalp_signal_engine=scalp_signal_engine,
        scalp_confirmation_timeframes=scalp_confirmation_timeframes,
        scalp_origin_timeframes=scalp_origin_timeframes,
        scalp_cycle_regime=scalp_cycle_regime,
        performance_tracker=performance_tracker,
        fetch_indicators=fetch_indicators,
        fetch_5m_frame=fetch_5m_frame,
        fetch_prediction=fetch_prediction,
    )

    return result.opportunities, result.funnel
