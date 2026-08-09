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

from src.agents.llm_factory import (
    current_provider,
    get_llm_circuit_breaker,
    get_llm_limiter,
    invoke_with_fallback,
    primary_and_fallback_models,
)
from src.config import get_settings
from src.finops import record_llm_response
from src.utils.circuit_breaker import CircuitBreakerOpenError
from src.utils.formatting import plain_english_fallback_cause

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

    if not settings.enable_llm_agents:
        return _fallback_signal_validation(
            state, state.get("signals", []), "LLM agents disabled via settings"
        )

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
            raise CircuitBreakerOpenError(f"{current_provider()}_api", circuit_breaker.recovery_time)

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
        record_llm_response("signal_validation", response, model=model_used)
        result = _parse_validation_response(response.content, signals)

        validated = result["validated"]
        rejected = result["rejected"]

        logger.info(f"Validated: {len(validated)}, Rejected: {len(rejected)}")

        return {
            "validated_signals": validated,
            "rejected_signals": rejected,
            "messages": [response],
        }

    except CircuitBreakerOpenError as e:
        logger.warning(f"Circuit breaker open, applying conservative validation: {e}")
        return _fallback_signal_validation(state, signals, str(e))

    except Exception as e:
        logger.error(f"Signal Validation Agent error: {e}")
        return _fallback_signal_validation(state, state.get("signals", []), str(e))


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
                    f"risk-reward {rr_ratio:.2f} is below the {min_rr:.2f} "
                    "backup-check minimum"
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
            decision = validation.get("decision", "reject")

            if signal_id in signal_lookup:
                signal = signal_lookup[signal_id].copy()
                signal["validation"] = validation

                # Apply modifications if approved
                if decision == "approve":
                    mods = validation.get("modifications", {})
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
