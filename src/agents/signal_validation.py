"""
Signal Validation Agent Module

Validates raw trading signals using context, memory lessons, and LLM reasoning.
Acts as a filter between raw signals and risk management.

Features:
- Rate limiting to prevent API throttling
- Circuit breaker for resilience
- Structured error handling
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.candidate_cache import (
    CandidateVerdict,
    build_candidate_fingerprint,
    get_candidate_verdict_cache,
)
from src.agents.llm_factory import (
    current_provider,
    get_llm_circuit_breaker,
    get_llm_limiter,
    invoke_with_fallback,
    model_for_tier,
    primary_and_fallback_models,
    review_model_tier,
)
from src.config import get_settings
from src.core.qwen import (
    ContextHandling,
    ContextReviewState,
    QwenReviewPolicy,
    parse_context_reviews,
    serialize_context_packets,
)
from src.core.utils.circuit_breaker import CircuitBreakerOpenError
from src.core.utils.formatting import plain_english_fallback_cause
from src.finops import LLMCallReason, get_cost_tracker, record_llm_response
from src.observability.tracing import record_candidate_review

from .state import TradingState

logger = logging.getLogger(__name__)


VALIDATION_SYSTEM_PROMPT = """You are a Signal Validation Agent for an automated trading system.

Your role is to evaluate raw trading signals and decide whether to approve or reject them.
You are the quality control layer before trades go to risk management.

For each signal, consider:
1. Does the signal align with the current market regime?
2. Is the signal from an active strategy (selected by Strategy Agent)?
3. Does the signal have sufficient confidence and risk-reward ratio?
4. Are there memory lessons warning against similar trades?
5. Is the timing appropriate (not chasing, not against major trend)?

You will receive:
- Current market regime and confidence
- Active strategies list
- Raw signals with their details
- Relevant memory lessons

For EACH signal, respond with JSON:
{
    "validations": [
        {
            "signal_id": "the signal ID",
            "decision": "approve" or "reject",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation",
            "modifications": {
                "stop_loss": optional new value,
                "target_price": optional new value,
                "position_size_pct": optional new value
            }
        }
    ]
}

Use ALL the enrichment provided: the ML prediction consensus and news/market-mood are
independent evidence. Reject or reduce a signal that contradicts a high-confidence ML
prediction or strongly opposing sentiment; raise confidence when they confirm the signal.
Give a calibrated confidence (genuine probability the trade works), not a default value.

Be selective - it's better to miss a trade than take a bad one.
Quality over quantity."""


COMPACT_VALIDATION_SYSTEM_PROMPT = """You are DeltaQuant's contextual risk reviewer.
Python has already validated indicators, MTF alignment, entry quality, expected-R, strategy
eligibility, and portfolio risk. Do not recalculate or modify stops, targets, size, exposure,
or indicators.
Review only material event/news contradictions and exceptional market context.
Return JSON only: {"context_reviews":[{"signal_id":"...",
"state":"SUPPORTIVE|NEUTRAL|CAUTION|MATERIAL_CONFLICT","confidence":0.0,
"supporting_factors":["CODE"],"risk_factors":["CODE"],
"unresolved_conflicts":["CODE"],
"handling":"CONTINUE|CONTINUE_WITH_CAUTION|REQUIRE_REVIEW|CONTEXT_REJECT"}]}.
Include exactly one independent result for every supplied signal. Never return BUY/SELL,
position size, stop, target, or an order instruction. Keep codes concise."""


def signal_validation_node(state: TradingState) -> dict[str, Any]:
    """
    LangGraph node for signal validation.

    Validates raw signals and filters out low-quality opportunities.
    Uses rate limiting and circuit breaker for resilience.

    Args:
        state: Current trading state with signals and context

    Returns:
        State updates with validated and rejected signals
    """
    logger.info("Running Signal Validation Agent...")

    settings = get_settings()
    eligibility_rejected = [
        dict(signal)
        for signal in state.get("signals", [])
        if isinstance(signal, dict)
        and (
            signal.get("registry_decision") == "BLOCK"
            or signal.get("registry_qwen_allowed") is False
        )
    ]
    reviewable = [
        dict(signal)
        for signal in state.get("signals", [])
        if not isinstance(signal, dict)
        or (
            signal.get("registry_decision") != "BLOCK"
            and signal.get("registry_qwen_allowed") is not False
        )
    ]
    working_state = state.copy()
    working_state["signals"] = reviewable
    if not reviewable:
        return {
            "validated_signals": [],
            "rejected_signals": eligibility_rejected,
        }
    state = working_state

    if not settings.enable_llm_agents:
        result = _fallback_signal_validation(
            state, state.get("signals", []), "LLM agents disabled via settings"
        )
        result["rejected_signals"] = [
            *eligibility_rejected,
            *result.get("rejected_signals", []),
        ]
        return result

    if getattr(settings, "llm_cost_optimization_enabled", False) is True:
        result = _optimized_signal_validation(state)
        result["rejected_signals"] = [
            *eligibility_rejected,
            *result.get("rejected_signals", []),
        ]
        return result

    rate_limiter = get_llm_limiter()
    circuit_breaker = get_llm_circuit_breaker()

    try:
        signals = state.get("signals", [])

        if not signals:
            logger.info("No signals to validate")
            return {
                "validated_signals": [],
                "rejected_signals": [],
            }

        regime = state.get("regime", "unknown")
        regime_confidence = state.get("regime_confidence", 0.0)
        active_strategies = state.get("active_strategies", [])
        memory_lessons = state.get("memory_lessons", [])

        # Filter relevant lessons
        timing_lessons = [
            lesson
            for lesson in memory_lessons
            if lesson.get("category") in ["poor_timing", "signal_quality"]
        ]

        # Build context (enriched with prediction + sentiment from support agents)
        context = _build_validation_context_enriched(
            signals, regime, regime_confidence, active_strategies, timing_lessons, state
        )

        messages = [
            SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        # Check circuit breaker
        if not circuit_breaker.is_available:
            raise CircuitBreakerOpenError(
                f"{current_provider()}_api", circuit_breaker.recovery_time
            )

        # Apply rate limiting
        if settings.enable_rate_limiting:
            rate_limiter.acquire_sync()

        model_used = primary_and_fallback_models()[0]

        def _record_model_used(name: str) -> None:
            nonlocal model_used
            model_used = name

        response = invoke_with_fallback(
            messages,
            circuit_breaker=circuit_breaker,
            max_tokens=2048,
            on_model_selected=_record_model_used,
        )
        record_llm_response(
            "signal_validation",
            response,
            model=model_used,
            trade_horizon=str(state.get("trade_horizon", "SWING")),
            purpose="candidate_signal_validation_legacy",
            cycle_id=str(state.get("workflow_id", "")),
            candidate_id=",".join(str(signal.get("signal_id", "")) for signal in signals),
            symbol=",".join(sorted({str(signal.get("symbol", "")) for signal in signals})),
            market=str(state.get("market", "NSE")),
            call_reason=LLMCallReason.CANDIDATE_REVIEW,
            component="signal_validation",
            timeframe=",".join(sorted({str(signal.get("timeframe", "")) for signal in signals})),
        )
        result = _parse_validation_response(response.content, signals)

        validated = result["validated"]
        rejected = result["rejected"]

        logger.info(f"Validated: {len(validated)}, Rejected: {len(rejected)}")

        return {
            "validated_signals": validated,
            "rejected_signals": [*eligibility_rejected, *rejected],
            "messages": [response],
        }

    except CircuitBreakerOpenError as e:
        logger.warning(f"Circuit breaker open, applying conservative validation: {e}")
        result = _fallback_signal_validation(state, signals, str(e))
        result["rejected_signals"] = [
            *eligibility_rejected,
            *result.get("rejected_signals", []),
        ]
        return result

    except Exception as e:
        logger.error(f"Signal Validation Agent error: {e}")
        result = _fallback_signal_validation(state, state.get("signals", []), str(e))
        result["rejected_signals"] = [
            *eligibility_rejected,
            *result.get("rejected_signals", []),
        ]
        return result


def _compact_validation_context(state: TradingState, signals: list[dict[str, Any]]) -> str:
    """Serialize only fields the contextual reviewer is allowed to use."""
    compact_signals: list[dict[str, Any]] = []
    for signal in signals:
        ranking = signal.get("ranking") or signal.get("indicators", {}).get("ranking", {})
        if not isinstance(ranking, dict):
            ranking = {}
        compact_signals.append(
            {
                "signal_id": signal.get("signal_id"),
                "symbol": signal.get("symbol"),
                "horizon": signal.get("trade_horizon", state.get("trade_horizon", "SWING")),
                "timeframe": signal.get("timeframe"),
                "direction": signal.get("signal_type"),
                "strategy": signal.get("strategy"),
                "entry_state": signal.get(
                    "entry_state", signal.get("candidate_action", "ENTER_NOW")
                ),
                "entry_score": signal.get("entry_quality_score", signal.get("confidence")),
                "ml_probability": signal.get(
                    "ml_probability", ranking.get("ml_direction_probability")
                ),
                "relative_volume": signal.get("volume_ratio"),
                "expected_r": signal.get(
                    "expected_r",
                    ranking.get("expected_r_multiple", signal.get("risk_reward_ratio")),
                ),
                "eligibility_status": signal.get("eligibility_status", "UNVALIDATED"),
                "eligibility_model_version": signal.get("model_version", ""),
                "regime_policy": signal.get("regime_policy", "ALLOW"),
                "timeframes": signal.get("timeframe_states", {}),
            }
        )
    shared = state.get("shared_market_context") or {}
    payload = {
        "regime": state.get("regime", "unknown"),
        "regime_confidence": state.get("regime_confidence", 0.0),
        "event_risk": shared.get("major_market_event_risk", "NONE"),
        "market_sentiment": shared.get(
            "market_sentiment", (state.get("news_sentiment") or {}).get("avg_sentiment")
        ),
        "market_context_version": shared.get("context_version", ""),
        "candidates": compact_signals,
        "memory_lessons": [
            {
                "category": lesson.get("category"),
                "severity": lesson.get("severity"),
                "description": str(lesson.get("description", ""))[:160],
            }
            for lesson in state.get("memory_lessons", [])[:3]
            if isinstance(lesson, dict)
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _optimized_signal_validation(state: TradingState) -> dict[str, Any]:
    """Selectively review only context-dependent candidates with Qwen."""

    settings = get_settings()
    signals = list(state.get("signals", []))
    if not signals:
        return {"validated_signals": [], "rejected_signals": []}

    horizon = str(state.get("trade_horizon", "SWING")).upper()
    shared = state.get("shared_market_context") or {}
    context_version = str(shared.get("context_version", ""))
    ttl = (
        settings.llm_cache_ttl_scalp_seconds
        if horizon == "SCALP"
        else settings.llm_cache_ttl_swing_seconds
    )
    policy = QwenReviewPolicy(
        regime_uncertain_below=getattr(settings, "qwen_regime_uncertain_below", 0.60),
        ml_conflict_below=getattr(settings, "qwen_ml_conflict_below", 0.45),
    )
    cache = get_candidate_verdict_cache()
    active_strategies = set(state.get("active_strategies", []))
    completed: dict[str, dict[str, Any]] = {}
    review_queue: list[tuple[int, dict[str, Any]]] = []

    for original in signals:
        signal = dict(original)
        signal_id = str(signal.get("signal_id", ""))
        fingerprint = build_candidate_fingerprint(
            signal,
            regime=str(state.get("regime", "unknown")),
            market_context_version=context_version,
            symbol_context_version=str(signal.get("symbol_context_version", "")),
            material_price_move_bps=settings.llm_cache_material_price_move_bps,
            ml_probability_step=settings.llm_cache_ml_probability_step,
            expected_r_step=settings.llm_cache_expected_r_step,
            entry_score_step=settings.llm_cache_entry_score_step,
        )
        signal["candidate_fingerprint"] = fingerprint
        supporting = set(signal.get("supporting_strategies") or [signal.get("strategy", "")])
        if not supporting.intersection(active_strategies):
            signal["validation"] = {
                "decision": "reject",
                "context_state": ContextReviewState.MATERIAL_CONFLICT.value,
                "handling": ContextHandling.CONTEXT_REJECT.value,
                "confidence": 1.0,
                "reason_codes": ["STRATEGY_NOT_ACTIVE"],
                "risk_flags": ["STRATEGY_SELECTION_MISMATCH"],
                "cache_hit": False,
                "model_used": "deterministic",
            }
            completed[signal_id] = signal
            continue

        review_decision = policy.assess(
            signal,
            regime_confidence=float(state.get("regime_confidence", 0.0) or 0.0),
            shared_context=shared,
        )
        signal["qwen_review_required"] = review_decision.required
        signal["qwen_review_reason_codes"] = list(review_decision.reason_codes)
        if not review_decision.required:
            signal["validation"] = {
                "decision": "approve",
                "context_state": ContextReviewState.NEUTRAL.value,
                "handling": ContextHandling.CONTINUE.value,
                "confidence": 1.0,
                "reason_codes": ["CLEAR_DETERMINISTIC_CONTEXT"],
                "risk_flags": [],
                "reasoning": "No material contextual ambiguity required Qwen review",
                "cache_hit": False,
                "qwen_skipped_clear": True,
                "model_used": "deterministic_policy",
            }
            completed[signal_id] = signal
            continue
        review_queue.append((review_decision.priority, signal))

    review_queue.sort(key=lambda item: (-item[0], str(item[1].get("signal_id", ""))))
    capacity = getattr(settings, "qwen_max_reviews_per_event", 20)
    cache_hits: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for _priority, signal in review_queue:
        verdict = cache.get(str(signal.get("candidate_fingerprint", ""))) if ttl > 0 else None
        if verdict is not None:
            signal["validation"] = verdict.validation_dict(cache_hit=True)
            completed[str(signal.get("signal_id", ""))] = signal
            cache_hits.append(signal)
        elif len(misses) < capacity:
            misses.append(signal)
        else:
            deferred.append(signal)
    for signal in deferred:
        signal["validation"] = {
            "decision": "deferred",
            "context_state": ContextReviewState.CAUTION.value,
            "handling": ContextHandling.REQUIRE_REVIEW.value,
            "confidence": 0.0,
            "reason_codes": ["CONTEXT_REVIEW_DEFERRED"],
            "risk_flags": [],
            "reasoning": "Context review capacity reached; opportunity preserved as deferred",
            "cache_hit": False,
            "model_used": "capacity_policy",
        }
        completed[str(signal.get("signal_id", ""))] = signal

    errors: list[str] = []
    model_used = "cache"
    usage = None
    qwen_latency_ms = 0.0
    qwen_called = False
    if misses:
        if get_cost_tracker().is_over_hard_budget(horizon):
            errors.append(f"FinOps hard budget reached for {horizon}")
        else:
            limiter = get_llm_limiter()
            breaker = get_llm_circuit_breaker()
            try:
                from time import perf_counter

                if not breaker.is_available:
                    raise CircuitBreakerOpenError(
                        f"{current_provider()}_api", breaker.recovery_time
                    )
                if settings.enable_rate_limiting:
                    limiter.acquire_sync()
                model_used = model_for_tier("REVIEW")

                def _record_model(name: str) -> None:
                    nonlocal model_used
                    model_used = name

                prompt = serialize_context_packets(
                    misses,
                    state,
                    max_input_tokens=getattr(
                        settings, "qwen_max_input_tokens_per_review", 900
                    )
                    * len(misses),
                )
                started = perf_counter()
                qwen_called = True
                response = invoke_with_fallback(
                    [
                        SystemMessage(content=COMPACT_VALIDATION_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ],
                    circuit_breaker=breaker,
                    max_tokens=min(
                        settings.qwen_max_output_tokens_review * len(misses),
                        4096,
                    ),
                    models_to_try=(model_used, model_for_tier("FAST")),
                    trade_horizon=horizon,
                    on_model_selected=_record_model,
                )
                qwen_latency_ms = (perf_counter() - started) * 1000.0
                usage = record_llm_response(
                    "context_review",
                    response,
                    model=model_used,
                    trade_horizon=horizon,
                    purpose="candidate_context_review",
                    cycle_id=str(state.get("workflow_id", "")),
                    candidate_id=",".join(
                        str(signal.get("signal_id", "")) for signal in misses
                    ),
                    symbol=",".join(
                        sorted({str(signal.get("symbol", "")) for signal in misses})
                    ),
                    market=str(state.get("market", "NSE")),
                    call_reason=LLMCallReason.CANDIDATE_REVIEW,
                    component="context_review",
                    timeframe=",".join(
                        sorted({str(signal.get("timeframe", "")) for signal in misses})
                    ),
                    candidate_metadata={
                        "supporting_strategies": sorted(
                            {
                                strategy
                                for signal in misses
                                for strategy in signal.get("supporting_strategies", [])
                            }
                        )
                    },
                )
                reviews = parse_context_reviews(response.content, misses)
                for signal in misses:
                    signal_id = str(signal.get("signal_id", ""))
                    review = reviews[signal_id]
                    signal["validation"] = review.validation_dict(
                        cache_hit=False, model_used=model_used
                    )
                    completed[signal_id] = signal
                    if ttl > 0:
                        cache.set(
                            CandidateVerdict.create(
                                fingerprint=str(signal.get("candidate_fingerprint", "")),
                                decision=review.downstream_decision,
                                confidence=review.confidence,
                                reason_codes=list(review.supporting_factors),
                                risk_flags=list(review.risk_factors),
                                ttl_seconds=ttl,
                                model_used=model_used,
                                market_context_version=context_version,
                                symbol_context_version=str(
                                    signal.get("symbol_context_version", "")
                                ),
                                context_state=review.state.value,
                                handling=review.handling.value,
                                unresolved_conflicts=list(review.unresolved_conflicts),
                            )
                        )
            except Exception as exc:
                errors.append(str(exc))

    # Missing/malformed/failed Qwen reviews never become an approval.
    for signal in misses:
        signal_id = str(signal.get("signal_id", ""))
        if signal_id in completed:
            continue
        signal["validation"] = {
            "decision": "reject",
            "context_state": ContextReviewState.CAUTION.value,
            "handling": ContextHandling.REQUIRE_REVIEW.value,
            "confidence": 0.0,
            "reason_codes": ["CONTEXT_REVIEW_FAILED"],
            "risk_flags": ["UNRESOLVED_CONTEXT"],
            "reasoning": "Qwen context review failed or was unavailable; fail closed",
            "cache_hit": False,
            "model_used": model_used,
        }
        completed[signal_id] = signal

    ordered = [completed.get(str(signal.get("signal_id", ""))) for signal in signals]
    complete = [signal for signal in ordered if signal is not None]
    for signal in complete:
        validation = signal.get("validation", {})
        record_candidate_review(
            {
                "symbol": signal.get("symbol"),
                "trade_horizon": horizon,
                "timeframe": signal.get("timeframe"),
                "strategy": signal.get("strategy"),
                "model_version": signal.get("model_version", ""),
                "validation_status": signal.get("eligibility_status", ""),
                "current_regime": signal.get("current_regime", ""),
                "regime_policy": signal.get("regime_policy", ""),
                "original_confidence": signal.get("original_confidence"),
                "regime_adjusted_confidence": signal.get("regime_adjusted_confidence"),
                "ml_confidence": signal.get("ml_probability"),
                "registry_decision": signal.get("registry_decision", ""),
                "risk_decision": "PENDING",
                "final_outcome": "PENDING_DETERMINISTIC_RISK",
                "rejection_reason": signal.get("rejection_reason", ""),
                "candidate_fingerprint": signal.get("candidate_fingerprint"),
                "qwen_required": bool(signal.get("qwen_review_required", False)),
                "cache_hit": bool(validation.get("cache_hit", False)),
                "model": validation.get("model_used", model_used),
                "context_version": context_version,
                "decision": validation.get("decision", "reject"),
                "execution_result": "PENDING_DETERMINISTIC_RISK",
            }
        )
    validated = [
        signal
        for signal in complete
        if str(signal.get("validation", {}).get("decision", "reject")).lower() == "approve"
    ]
    rejected = [signal for signal in complete if signal not in validated]
    metrics = dict(state.get("llm_optimization_metrics", {}))
    input_tokens = int(usage.input_tokens) if usage is not None else 0
    output_tokens = int(usage.output_tokens) if usage is not None else 0
    tokens_by_strategy: dict[str, int] = {}
    tokens_by_timeframe: dict[str, int] = {}
    tokens_by_candidate: dict[str, int] = {}
    total_tokens = input_tokens + output_tokens
    if misses and total_tokens:
        base, remainder = divmod(total_tokens, len(misses))
        for index, signal in enumerate(misses):
            allocated = base + (1 if index < remainder else 0)
            strategy = str(signal.get("strategy", "unknown"))
            timeframe = str(signal.get("timeframe", "unknown"))
            signal_id = str(signal.get("signal_id", index))
            tokens_by_strategy[strategy] = tokens_by_strategy.get(strategy, 0) + allocated
            tokens_by_timeframe[timeframe] = tokens_by_timeframe.get(timeframe, 0) + allocated
            tokens_by_candidate[signal_id] = allocated
    metrics.update(
        {
            "qwen_candidates_considered": len(signals),
            "qwen_reviews_requested": len(review_queue),
            "qwen_candidates": len(review_queue),
            "qwen_calls": 1 if qwen_called else 0,
            "qwen_cache_hits": len(cache_hits),
            "qwen_cache_misses": len(misses),
            "qwen_skipped_clear": len(signals) - len(review_queue),
            "qwen_reviews_deferred": len(deferred),
            "qwen_input_tokens": input_tokens,
            "qwen_output_tokens": output_tokens,
            "qwen_total_tokens": total_tokens,
            "qwen_tokens_by_strategy": tokens_by_strategy,
            "qwen_tokens_by_timeframe": tokens_by_timeframe,
            "qwen_tokens_by_candidate": tokens_by_candidate,
            "qwen_latency_ms": qwen_latency_ms,
            "qwen_approved": len(validated),
        }
    )
    result: dict[str, Any] = {
        "validated_signals": validated,
        "rejected_signals": rejected,
        "llm_optimization_metrics": metrics,
    }
    if errors:
        result["errors"] = list(state.get("errors", [])) + errors
    return result


def _batch_all_signal_validation_legacy(state: TradingState) -> dict[str, Any]:
    """Previous optimized implementation retained for replay comparison only."""
    settings = get_settings()
    signals = list(state.get("signals", []))
    if not signals:
        return {"validated_signals": [], "rejected_signals": []}

    horizon = str(state.get("trade_horizon", "SWING")).upper()
    shared = state.get("shared_market_context") or {}
    context_version = str(shared.get("context_version", ""))
    ttl = (
        settings.llm_cache_ttl_scalp_seconds
        if horizon == "SCALP"
        else settings.llm_cache_ttl_swing_seconds
    )
    cache = get_candidate_verdict_cache()
    cached_results: dict[str, dict[str, Any]] = {}
    deterministic_results: dict[str, dict[str, Any]] = {}
    misses: list[dict[str, Any]] = []
    active_strategies = set(state.get("active_strategies", []))

    for original in signals:
        signal = dict(original)
        symbol_context_version = str(signal.get("symbol_context_version", ""))
        fingerprint = build_candidate_fingerprint(
            signal,
            regime=str(state.get("regime", "unknown")),
            market_context_version=context_version,
            symbol_context_version=symbol_context_version,
            material_price_move_bps=settings.llm_cache_material_price_move_bps,
            ml_probability_step=settings.llm_cache_ml_probability_step,
            expected_r_step=settings.llm_cache_expected_r_step,
            entry_score_step=settings.llm_cache_entry_score_step,
        )
        signal["candidate_fingerprint"] = fingerprint
        if str(signal.get("strategy", "")) not in active_strategies:
            signal["validation"] = {
                "decision": "reject",
                "confidence": 1.0,
                "reason_codes": ["STRATEGY_NOT_ACTIVE"],
                "risk_flags": ["STRATEGY_SELECTION_MISMATCH"],
                "reasoning": "STRATEGY_NOT_ACTIVE",
                "cache_hit": False,
                "model_used": "deterministic",
            }
            deterministic_results[str(signal.get("signal_id", fingerprint))] = signal
            continue
        verdict = cache.get(fingerprint) if ttl > 0 else None
        if verdict is None:
            misses.append(signal)
            continue
        signal["validation"] = verdict.validation_dict(cache_hit=True)
        cached_results[str(signal.get("signal_id", fingerprint))] = signal

    fresh_validated: list[dict[str, Any]] = []
    fresh_rejected: list[dict[str, Any]] = []
    model_used = "cache"
    if misses:
        if get_cost_tracker().is_over_hard_budget(horizon):
            fallback = _fallback_signal_validation(
                state, misses, f"FinOps hard budget reached for {horizon}"
            )
            fresh_validated = fallback["validated_signals"]
            fresh_rejected = fallback["rejected_signals"]
            errors = fallback.get("errors", [])
        else:
            limiter = get_llm_limiter()
            breaker = get_llm_circuit_breaker()
            try:
                if not breaker.is_available:
                    raise CircuitBreakerOpenError(
                        f"{current_provider()}_api", breaker.recovery_time
                    )
                if settings.enable_rate_limiting:
                    limiter.acquire_sync()
                tier = review_model_tier(misses, shared)
                selected = model_for_tier(tier)
                fast_fallback = model_for_tier("FAST")
                max_tokens = (
                    settings.qwen_max_output_tokens_review
                    if tier == "REVIEW"
                    else settings.qwen_max_output_tokens_fast
                )
                model_used = selected

                def _record_model(name: str) -> None:
                    nonlocal model_used
                    model_used = name

                response = invoke_with_fallback(
                    [
                        SystemMessage(content=COMPACT_VALIDATION_SYSTEM_PROMPT),
                        HumanMessage(content=_compact_validation_context(state, misses)),
                    ],
                    circuit_breaker=breaker,
                    max_tokens=max_tokens,
                    models_to_try=(selected, fast_fallback),
                    trade_horizon=horizon,
                    on_model_selected=_record_model,
                )
                record_llm_response(
                    "signal_validation",
                    response,
                    model=model_used,
                    trade_horizon=horizon,
                    purpose="candidate_signal_validation",
                    cycle_id=str(state.get("workflow_id", "")),
                    candidate_id=",".join(
                        str(signal.get("signal_id", "")) for signal in misses
                    ),
                    symbol=",".join(
                        sorted({str(signal.get("symbol", "")) for signal in misses})
                    ),
                    market=str(state.get("market", "NSE")),
                    call_reason=LLMCallReason.CANDIDATE_REVIEW,
                    component="signal_validation",
                    timeframe=",".join(
                        sorted({str(signal.get("timeframe", "")) for signal in misses})
                    ),
                )
                parsed = _parse_validation_response(
                    response.content, misses, allow_modifications=False
                )
                fresh_validated = parsed["validated"]
                fresh_rejected = parsed["rejected"]
                errors = []
            except Exception as exc:
                fallback = _fallback_signal_validation(state, misses, str(exc))
                fresh_validated = fallback["validated_signals"]
                fresh_rejected = fallback["rejected_signals"]
                errors = fallback.get("errors", [])

        if ttl > 0 and not errors:
            for signal in [*fresh_validated, *fresh_rejected]:
                validation = signal.get("validation", {})
                decision = str(validation.get("decision", "reject")).lower()
                reason_codes = validation.get("reason_codes") or [
                    "CONTEXT_APPROVED" if decision == "approve" else "CONTEXT_REJECTED"
                ]
                risk_flags = validation.get("risk_flags") or []
                verdict = CandidateVerdict.create(
                    fingerprint=str(signal.get("candidate_fingerprint", "")),
                    decision=decision,
                    confidence=float(validation.get("confidence", 0.0) or 0.0),
                    reason_codes=list(reason_codes),
                    risk_flags=list(risk_flags),
                    ttl_seconds=ttl,
                    model_used=model_used,
                    market_context_version=context_version,
                    symbol_context_version=str(signal.get("symbol_context_version", "")),
                )
                cache.set(verdict)
    else:
        errors = []

    fresh_by_id = {
        str(signal.get("signal_id", signal.get("candidate_fingerprint", ""))): signal
        for signal in [*fresh_validated, *fresh_rejected]
    }
    ordered = [
        deterministic_results.get(str(signal.get("signal_id", "")))
        or cached_results.get(str(signal.get("signal_id", "")))
        or fresh_by_id.get(str(signal.get("signal_id", "")))
        for signal in signals
    ]
    complete = [signal for signal in ordered if signal is not None]
    for signal in complete:
        validation = signal.get("validation", {})
        record_candidate_review(
            {
                "symbol": signal.get("symbol"),
                "trade_horizon": horizon,
                "timeframe": signal.get("timeframe"),
                "strategy": signal.get("strategy"),
                "model_version": signal.get("model_version", ""),
                "validation_status": signal.get("eligibility_status", ""),
                "current_regime": signal.get("current_regime", ""),
                "regime_policy": signal.get("regime_policy", ""),
                "original_confidence": signal.get("original_confidence"),
                "regime_adjusted_confidence": signal.get("regime_adjusted_confidence"),
                "ml_confidence": signal.get("ml_probability"),
                "registry_decision": signal.get("registry_decision", ""),
                "risk_decision": "PENDING",
                "final_outcome": "PENDING_DETERMINISTIC_RISK",
                "rejection_reason": signal.get("rejection_reason", ""),
                "candidate_fingerprint": signal.get("candidate_fingerprint"),
                "cache_hit": bool(validation.get("cache_hit", False)),
                "model": validation.get("model_used", model_used),
                "context_version": context_version,
                "decision": validation.get("decision", "reject"),
                "execution_result": "PENDING_DETERMINISTIC_RISK",
            }
        )
    validated = [
        signal
        for signal in complete
        if str(signal.get("validation", {}).get("decision", "reject")).lower() == "approve"
    ]
    rejected = [signal for signal in complete if signal not in validated]
    metrics = dict(state.get("llm_optimization_metrics", {}))
    metrics.update(
        {
            "qwen_candidates": len(signals) - len(deterministic_results),
            "qwen_calls": 1 if misses and not errors else 0,
            "qwen_cache_hits": len(cached_results),
            "qwen_cache_misses": len(misses),
            "dedup_skip_count": len(cached_results),
            "qwen_approved": len(validated),
            "deterministic_rejection_count": int(metrics.get("deterministic_rejection_count", 0))
            + len(deterministic_results),
        }
    )
    result: dict[str, Any] = {
        "validated_signals": validated,
        "rejected_signals": rejected,
        "llm_optimization_metrics": metrics,
    }
    if errors:
        result["errors"] = errors
    return result


def _fallback_signal_validation(
    state: TradingState,
    signals: list[dict[str, Any]],
    error_msg: str,
) -> dict[str, Any]:
    """
    Fallback signal validation using simple rule-based logic instead of the LLM.

    Used when the LLM is unavailable. Only passes signals that are from an active
    strategy, with confidence/risk-reward at or above the horizon-appropriate bar --
    a much cruder bar than the LLM's actual reasoning, so this should be rare and
    short-lived, not a steady state. Every rejection explains both WHY the AI
    reviewer was skipped and WHICH specific check the signal failed, in plain
    English, instead of the previous generic "Fallback: Failed rule-based validation."

    Thresholds are horizon-aware: a SCALP-tagged signal (``signal["trade_horizon"]``,
    falling back to the cycle's ``state["trade_horizon"]``) is checked against
    ``settings.scalp_min_confidence``/``scalp_min_rr`` instead of the swing 0.6/1.5.
    Both scalp defaults ship numerically EQUAL to the swing hardcoded bar (see
    settings.py) -- this only makes the bar horizon-selectable, it never silently
    lowers it.
    """
    active_strategies = state.get("active_strategies", [])
    cause = plain_english_fallback_cause(error_msg)
    settings = get_settings()

    validated = []
    rejected = []

    for signal in signals:
        strategy = signal.get("strategy", "")
        confidence = signal.get("confidence", 0)
        rr_ratio = signal.get("risk_reward_ratio", 0)
        trade_horizon = signal.get("trade_horizon", state.get("trade_horizon", "SWING"))
        if trade_horizon == "SCALP":
            min_confidence = settings.scalp_min_confidence
            min_rr = settings.scalp_min_rr
        else:
            min_confidence = 0.6
            min_rr = 1.5

        is_valid = (
            strategy in active_strategies and confidence >= min_confidence and rr_ratio >= min_rr
        )

        if is_valid:
            signal["validation"] = {
                "confidence": 0.5,
                "reasoning": (
                    f"AI review skipped because {cause}. Backup check passed it: "
                    f"strategy '{strategy}' is active, confidence {confidence:.0%}, "
                    f"risk-reward {rr_ratio:.2f}."
                ),
            }
            validated.append(signal)
        else:
            if strategy not in active_strategies:
                specific = (
                    f"strategy '{strategy}' isn't one of today's approved strategies "
                    f"({', '.join(active_strategies) or 'none admitted'})"
                )
            elif confidence < min_confidence:
                specific = (
                    f"confidence {confidence:.0%} is below the {min_confidence:.0%} "
                    "backup-check minimum"
                )
            else:
                specific = (
                    f"risk-reward {rr_ratio:.2f} is below the {min_rr:.2f} backup-check minimum"
                )
            signal["rejection_reason"] = (
                f"AI review skipped because {cause}, so a simplified backup check ran "
                f"instead — it rejected this signal because {specific}."
            )
            rejected.append(signal)

    logger.info(f"Fallback validation: {len(validated)} passed, {len(rejected)} rejected")

    return {
        "validated_signals": validated,
        "rejected_signals": rejected,
        "errors": state.get("errors", []) + [f"Validation Agent fallback: {error_msg}"],
    }


def _build_validation_context(
    signals: list[dict[str, Any]],
    regime: str,
    regime_confidence: float,
    active_strategies: list[str],
    lessons: list[dict[str, Any]],
) -> str:
    """Build context for signal validation."""

    context_parts = [
        "## Current Context\n",
        f"- Market Regime: **{regime}** (confidence: {regime_confidence:.2f})",
        f"- Active Strategies: {', '.join(active_strategies) or 'None'}",
    ]

    # Add signals
    context_parts.append(f"\n## Signals to Validate ({len(signals)} total)\n")

    for signal in signals:
        context_parts.append(f"\n### Signal: {signal.get('signal_id', 'N/A')}")
        context_parts.append(f"- Symbol: {signal.get('symbol', 'N/A')}")
        context_parts.append(f"- Type: {signal.get('signal_type', 'N/A')}")
        context_parts.append(f"- Strategy: {signal.get('strategy', 'N/A')}")
        context_parts.append(f"- Strength: {signal.get('strength', 'N/A')}")
        context_parts.append(f"- Confidence: {signal.get('confidence', 0):.2f}")
        context_parts.append(f"- Entry: {signal.get('entry_price', 0):.2f}")
        context_parts.append(f"- Stop Loss: {signal.get('stop_loss', 0):.2f}")
        context_parts.append(f"- Target: {signal.get('target_price', 0):.2f}")
        context_parts.append(f"- R:R Ratio: {signal.get('risk_reward_ratio', 0):.2f}")
        context_parts.append(f"- Position Size: {signal.get('position_size_pct', 0):.1f}%")

        if signal.get("reasons"):
            context_parts.append(f"- Reasons: {'; '.join(signal['reasons'][:3])}")

    # Add lessons
    if lessons:
        context_parts.append("\n## Past Lessons (Consider Carefully)\n")
        for lesson in lessons[:5]:
            context_parts.append(
                f"- [{lesson.get('severity', 'N/A')}] {lesson.get('description', 'N/A')}"
            )

    return "\n".join(context_parts)


def _build_validation_context_enriched(
    signals: list[dict[str, Any]],
    regime: str,
    regime_confidence: float,
    active_strategies: list[str],
    lessons: list[dict[str, Any]],
    state: dict[str, Any],
) -> str:
    """Build enriched validation context with prediction and sentiment data."""

    base = _build_validation_context(signals, regime, regime_confidence, active_strategies, lessons)

    enrichment = []

    # Add prediction consensus for signal symbols
    prediction_signals = state.get("prediction_signals", [])
    if prediction_signals:
        signal_symbols = {s.get("symbol") for s in signals}
        relevant_preds = [
            p
            for p in prediction_signals
            # An abstained prediction (H-7) has no usable direction; excluded rather
            # than surfaced as a "confirm/contradict" vote the LLM can't trust.
            if isinstance(p, dict) and p.get("symbol") in signal_symbols and not p.get("abstained")
        ]
        if relevant_preds:
            enrichment.append("\n## ML Prediction Consensus (use to confirm/contradict signals)\n")
            for pred in relevant_preds:
                enrichment.append(
                    f"- {pred.get('symbol')}: ML predicts **{pred.get('direction', 'N/A')}** "
                    f"(confidence: {pred.get('confidence', 0):.0%})"
                )
            enrichment.append(
                "\n> If a signal contradicts a high-confidence ML prediction, "
                "consider rejecting or reducing position size."
            )

    # Add news sentiment for signal symbols.
    # Contract: news_sentiment is a dict {"avg_sentiment": float} (see state.py).
    news_sentiment = state.get("news_sentiment") or {}
    avg_sentiment = (
        news_sentiment.get("avg_sentiment") if isinstance(news_sentiment, dict) else None
    )
    if avg_sentiment is not None:
        enrichment.append(f"\n## News Sentiment: {avg_sentiment:.2f} (-1 bearish to +1 bullish)")
        if abs(avg_sentiment) > 0.5:
            direction = "bullish" if avg_sentiment > 0 else "bearish"
            enrichment.append(f"> Strong {direction} news sentiment — factor this into validation.")

    # Add market mood.
    # Contract: market_mood is SentimentSignal.to_dict() (see state.py).
    market_mood = state.get("market_mood") or {}
    if isinstance(market_mood, dict) and market_mood.get("mood_index") is not None:
        mood_label = market_mood.get("mood_label", "neutral")
        enrichment.append(
            f"\n## Market Mood: {mood_label} ({float(market_mood['mood_index']):.0f}/100)"
        )

    if enrichment:
        return base + "\n" + "\n".join(enrichment)
    return base


def _parse_validation_response(
    content: str,
    original_signals: list[dict[str, Any]],
    *,
    allow_modifications: bool = True,
) -> dict[str, Any]:
    """Parse validation response and categorize signals."""

    validated = []
    rejected = []

    try:
        content = content.strip()

        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "{" in content and not content.startswith("{"):
            # Some models prepend prose/a heading before a bare JSON object with no
            # code fence (see the same fix in market_regime.py's _parse_regime_response
            # -- observed from Qwen). Take the JSON object itself rather than failing
            # to parse the leading prose as JSON.
            start = content.find("{")
            end = content.rfind("}")
            if end > start:
                content = content[start : end + 1]

        result = json.loads(content)
        validations = result.get("validations", [])

        # Create lookup for original signals
        signal_lookup = {s.get("signal_id"): s for s in original_signals}

        for validation in validations:
            signal_id = validation.get("signal_id")
            decision = str(validation.get("decision", "reject")).lower()

            if signal_id in signal_lookup:
                signal = signal_lookup[signal_id].copy()
                validation["decision"] = decision
                validation.setdefault("reason_codes", [])
                validation.setdefault("risk_flags", [])
                signal["validation"] = validation

                # Apply modifications if approved
                if decision == "approve":
                    mods = validation.get("modifications", {}) if allow_modifications else {}
                    if mods.get("stop_loss"):
                        signal["stop_loss"] = mods["stop_loss"]
                    if mods.get("target_price"):
                        signal["target_price"] = mods["target_price"]
                    if mods.get("position_size_pct"):
                        signal["position_size_pct"] = mods["position_size_pct"]
                    validated.append(signal)
                else:
                    rejected.append(signal)

        # Any signals not in response are rejected
        processed_ids = {v.get("signal_id") for v in validations}
        for signal in original_signals:
            if signal.get("signal_id") not in processed_ids:
                signal["validation"] = {
                    "decision": "reject",
                    "reasoning": (
                        "The AI reviewer's response didn't include a decision for this "
                        "signal (only some of the candidates it was sent) -- rejected "
                        "rather than assumed approved."
                    ),
                }
                rejected.append(signal)

    except json.JSONDecodeError as e:
        # Log the raw content for debugging, but never show it to the user as
        # "reasoning" -- a truncated dump of the model's own output reads as
        # corrupted data, not an explanation.
        logger.warning(f"Failed to parse validation response: {e}. Raw content: {content[:200]}")
        for signal in original_signals:
            signal["validation"] = {
                "decision": "reject",
                "reasoning": (
                    "The AI reviewer's response could not be read (malformed response) "
                    "-- rejected rather than assumed approved."
                ),
            }
            rejected.append(signal)

    return {"validated": validated, "rejected": rejected}
