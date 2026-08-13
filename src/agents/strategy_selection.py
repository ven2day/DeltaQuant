"""
Strategy Selection Agent Module

Selects active trading strategies based on the current market regime
and historical performance data.

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
from src.core.candidates import StrategyType
from src.core.utils.circuit_breaker import CircuitBreakerOpenError
from src.core.utils.formatting import plain_english_fallback_cause
from src.finops import LLMCallReason, record_llm_response

from .state import TradingState

logger = logging.getLogger(__name__)

# Every strategy name the LLM is allowed to select, derived from the same enum the
# signal engine and the eligibility registry use -- one source of truth instead of a
# hardcoded list that could silently drift out of sync when a strategy is added.
_VALID_STRATEGY_NAMES = [member.value for member in StrategyType]


def _retain_eligible_candidate_strategies(state: TradingState, strategies: list[str]) -> list[str]:
    """Retain requested strategies represented by pre-qualified candidates.

    Strategy eligibility is evaluated before this node using the exact candidate
    timeframe and model version. Repeating a coarser lookup here would create policy
    drift, so this node only intersects Qwen's selection with those prior decisions.
    """

    eligible_names: set[str] = set()
    for signal in state.get("signals", []):
        if not isinstance(signal, dict):
            continue
        if signal.get("registry_decision") != "ALLOW":
            continue
        eligible_names.add(str(signal.get("strategy", "")))
        eligible_names.update(str(item) for item in signal.get("supporting_strategies", []))
    retained = [name for name in strategies if name in eligible_names]
    stripped = [name for name in strategies if name not in eligible_names]
    if stripped:
        logger.info(
            "Strategy selection retained prior eligibility decisions; skipped %s",
            stripped,
        )
    return retained


STRATEGY_SYSTEM_PROMPT = """You are a Strategy Selection Agent for an automated trading system.

Your role is to select which trading strategies should be active based on:
1. The current market regime (trending_up, trending_down, ranging, volatile)
2. Historical performance of each strategy in similar conditions
3. Memory lessons from past strategy selection mistakes

Available strategies:
- momentum: Best in trending markets with strong directional moves
- mean_reversion: Best in ranging markets with clear support/resistance
- breakout: Best when volatility is low and a breakout is anticipated
- trend_following: Best in established trends with high ADX
- ema_heiken_ashi_rsi: Best in established trends; explicitly sits out sideways markets
- ema_psar: Best in trending markets with sustained directional moves; avoid in choppy/sideways conditions
- ema_cci: Best when a long-term (EMA200) trend has strong momentum confirmation
- ema_adx_trend: EMA20/50 trend confirmed by ADX and directional index
- donchian_breakout: Previous-bar Donchian breakout with relative-volume confirmation
- time_series_momentum: Medium-horizon return momentum above ATR noise
- trend_pullback: Completed-candle recovery near EMA20 inside an established trend
- supertrend_adx_ema: SuperTrend confirmed independently by EMA50, ADX, and DI
- macd_trend_continuation: MACD re-acceleration inside an EMA50 trend
- bollinger_rsi_mean_reversion: Confirmed Bollinger re-entry in low-ADX conditions
- vwap_mean_reversion: Reversal from a statistically material session-VWAP deviation
- opening_range_breakout: Completed NSE opening-range breakout with volume and VWAP
- relative_strength_momentum: Timestamp-aligned market-relative and cross-sectional momentum

Consider:
- Avoid strategies that historically underperformed in the current regime
- Retain every prior-eligible strategy materially supported by the supplied candidate evidence
- Weight memory lessons heavily - past mistakes should inform current decisions
- Lean on the REAL historical win-rates provided (not assumptions); prefer strategies with a
  proven edge in this regime, and be cautious when win-rates are only prior estimates

Respond with JSON:
{
    "active_strategies": ["list", "of", "strategy", "names"],
    "reasoning": "Explanation of selection with specific reference to regime and lessons",
    "strategy_notes": {
        "strategy_name": "why selected or rejected"
    }
}"""

COMPACT_STRATEGY_SYSTEM_PROMPT = """Select every DeltaQuant strategy materially supported by the supplied eligible candidates.
Use only names in available_strategies. Prefer proven regime performance and heed concise
memory warnings. Return JSON only: {"active_strategies":["name"],"reasoning":"one sentence"}.
Strategy eligibility is enforced by Python before and after this response; never claim to
override it."""


def strategy_selection_node(state: TradingState) -> dict[str, Any]:
    """
    LangGraph node for strategy selection.

    Selects which strategies should be active based on regime and memory.
    Uses rate limiting and circuit breaker for resilience.

    Args:
        state: Current trading state with regime and memory lessons

    Returns:
        State updates with active strategies
    """
    logger.info("Running Strategy Selection Agent...")

    working_state = state.copy()
    eligibility_rejected: list[dict[str, Any]] = list(state.get("pre_llm_rejected", []))
    metrics = state.get("llm_optimization_metrics", {})
    already_checked = bool(metrics.get("pre_llm_risk_pass")) or (
        bool(state.get("signals"))
        and all(
            isinstance(signal, dict)
            and signal.get("registry_decision") == "ALLOW"
            and signal.get("registry_qwen_allowed") is True
            for signal in state.get("signals", [])
        )
    )
    if not already_checked:
        from src.agents.pre_llm import filter_pre_llm_candidates

        gate = filter_pre_llm_candidates(working_state)
        working_state["signals"] = gate.passed
        eligibility_rejected.extend(gate.rejected)
        working_state["pre_llm_rejected"] = eligibility_rejected
        working_state["llm_optimization_metrics"] = {
            **dict(metrics),
            **gate.stage_counts,
            "eligibility_rejection_counts": gate.rejection_counts,
        }
        if not gate.passed:
            return {
                "signals": [],
                "active_strategies": [],
                "strategy_reasoning": (
                    "No candidate passed deterministic strategy eligibility and pre-Qwen risk."
                ),
                "pre_llm_rejected": eligibility_rejected,
                "risk_rejected": eligibility_rejected,
                "llm_optimization_metrics": working_state["llm_optimization_metrics"],
            }

    state = working_state

    settings = get_settings()

    if not settings.enable_llm_agents:
        return _fallback_strategy_selection(
            state, state.get("regime", "unknown"), "LLM agents disabled via settings"
        )

    if getattr(settings, "llm_cost_optimization_enabled", False) is True:
        return _optimized_strategy_selection(state)

    rate_limiter = get_llm_limiter()
    circuit_breaker = get_llm_circuit_breaker()

    try:
        regime = state.get("regime", "unknown")
        regime_confidence = state.get("regime_confidence", 0.0)
        memory_lessons = state.get("memory_lessons", [])
        daily_stats = state.get("daily_stats", {})

        # Filter relevant lessons
        strategy_lessons = [
            lesson
            for lesson in memory_lessons
            if lesson.get("category") in ["strategy_mismatch", "overtrading"]
        ]

        # Build context
        data_namespace = state.get("data_namespace", "paper_market_data")
        context = _build_strategy_context(
            regime, regime_confidence, strategy_lessons, daily_stats, data_namespace
        )

        messages = [
            SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
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
            max_tokens=1024,
            trade_horizon=str(state.get("trade_horizon", "SWING")),
            on_model_selected=_record_model_used,
        )
        record_llm_response(
            "strategy_selection",
            response,
            model=model_used,
            trade_horizon=str(state.get("trade_horizon", "SWING")),
            purpose="candidate_strategy_context",
            cycle_id=str(state.get("workflow_id", "")),
            market=str(state.get("market", "NSE")),
            call_reason=LLMCallReason.SUPPORT_AGENT,
            component="strategy_selection",
        )
        result = _parse_strategy_response(response.content)

        admitted = _retain_eligible_candidate_strategies(state, result["active_strategies"])
        logger.info(f"Selected strategies: {result['active_strategies']} -> admitted: {admitted}")

        return {
            "signals": state.get("signals", []),
            "active_strategies": admitted,
            "strategy_reasoning": result["reasoning"],
            "messages": [response],
            "pre_llm_rejected": eligibility_rejected,
        }

    except CircuitBreakerOpenError as e:
        logger.warning(f"Circuit breaker open: {e}")
        return _fallback_strategy_selection(state, regime, str(e))

    except Exception as e:
        logger.error(f"Strategy Selection Agent error: {e}")
        return _fallback_strategy_selection(state, state.get("regime", "unknown"), str(e))


def _compact_strategy_payload(state: TradingState) -> dict[str, Any]:
    """Build a small, deterministic strategy-selection payload."""
    regime = str(state.get("regime", "unknown"))
    namespace = str(state.get("data_namespace", "paper_market_data"))
    performance: dict[str, dict[str, Any]] = {}
    try:
        from src.memory.performance_tracker import get_performance_tracker

        tracker = get_performance_tracker(namespace)
        for name in _VALID_STRATEGY_NAMES:
            item = tracker.get_strategy_performance(name, regime)
            performance[name] = {
                "win_rate": round(float(item.win_rate), 3),
                "trades": int(item.total_trades),
            }
    except Exception as exc:
        logger.debug("Compact strategy performance unavailable: %s", exc)

    lessons = [
        {
            "category": lesson.get("category"),
            "severity": lesson.get("severity"),
            "description": str(lesson.get("description", ""))[:120],
        }
        for lesson in state.get("memory_lessons", [])[:3]
        if isinstance(lesson, dict)
        and lesson.get("category") in {"strategy_mismatch", "overtrading"}
    ]
    daily = state.get("daily_stats", {})
    return {
        "regime": regime,
        "regime_confidence": round(float(state.get("regime_confidence", 0.0) or 0.0), 2),
        "trade_horizon": str(state.get("trade_horizon", "SWING")).upper(),
        "available_strategies": _VALID_STRATEGY_NAMES,
        "candidate_strategies": sorted(
            {str(signal.get("strategy", "")) for signal in state.get("signals", [])} - {""}
        ),
        "performance": performance,
        "daily": {
            "trades": int(daily.get("trades_count", 0) or 0),
            "pnl_sign": 1
            if float(daily.get("profit_loss", 0.0) or 0.0) > 0
            else -1
            if float(daily.get("profit_loss", 0.0) or 0.0) < 0
            else 0,
        },
        "lessons": lessons,
    }


def _optimized_strategy_selection(state: TradingState) -> dict[str, Any]:
    """Admit every exact-grain strategy represented by this candidate batch.

    Strategy selection is not a pre-evaluation Top-K gate.  The signal engine has
    already evaluated every strategy against the shared feature snapshot; this node
    preserves those hypotheses and reuses prior eligibility decisions. Contextual Qwen review,
    when needed, belongs in ``signal_validation_node`` where conflicts/news are visible.
    """
    requested: list[str] = []
    for signal in state.get("signals", []):
        if not isinstance(signal, dict):
            continue
        names = signal.get("supporting_strategies") or [signal.get("strategy")]
        for name in names:
            normalized = str(name or "")
            if normalized and normalized not in requested:
                requested.append(normalized)
    admitted = _retain_eligible_candidate_strategies(state, requested)
    metrics = dict(state.get("llm_optimization_metrics", {}))
    metrics.update(
        {
            "strategies_requested": len(requested),
            "strategies_admitted": len(admitted),
            "strategy_selection_qwen_calls": 0,
        }
    )
    return {
        "signals": state.get("signals", []),
        "active_strategies": admitted,
        "strategy_reasoning": (
            "All candidate-producing strategies with prior eligibility retained without "
            "preselecting one strategy."
        ),
        "llm_optimization_metrics": metrics,
        "pre_llm_rejected": state.get("pre_llm_rejected", []),
    }


def _fallback_strategy_selection(
    state: TradingState, regime: str, error_msg: str
) -> dict[str, Any]:
    """
    Fallback strategy selection based on regime.

    Used when LLM is unavailable.
    """
    # Default strategies per regime
    regime_strategies = {
        "trending_up": [
            "momentum",
            "trend_following",
            "ema_heiken_ashi_rsi",
            "ema_psar",
            "ema_cci",
        ],
        "trending_down": ["trend_following", "ema_heiken_ashi_rsi", "ema_psar", "ema_cci"],
        "ranging": ["mean_reversion"],
        "volatile": ["breakout"],
    }

    strategies = regime_strategies.get(regime, ["trend_following"])
    admitted = _retain_eligible_candidate_strategies(state, strategies)

    logger.info(f"Using fallback strategies for {regime}: {strategies} -> admitted: {admitted}")

    return {
        "signals": state.get("signals", []),
        "active_strategies": admitted,
        "strategy_reasoning": (
            f"AI review skipped because {plain_english_fallback_cause(error_msg)}. "
            f"Backup rule selected the default strategy set for '{regime}'."
        ),
        "errors": state.get("errors", []) + [f"Strategy Agent fallback: {error_msg}"],
        "pre_llm_rejected": state.get("pre_llm_rejected", []),
    }


def _build_strategy_context(
    regime: str,
    regime_confidence: float,
    lessons: list[dict[str, Any]],
    daily_stats: dict[str, Any],
    data_namespace: str = "paper_market_data",
) -> str:
    """Build context for strategy selection."""

    context_parts = [
        "## Current Market Regime\n",
        f"- Regime: **{regime}**",
        f"- Confidence: {regime_confidence:.2f}",
        "\n## Today's Trading Stats\n",
        f"- Trades executed: {daily_stats.get('trades_count', 0)}",
        f"- P&L: {daily_stats.get('profit_loss', 0):.2f}",
    ]

    # Real strategy performance from tracker (replaces hardcoded data)
    context_parts.append("\n## Historical Strategy Performance by Regime\n")
    try:
        from src.memory.performance_tracker import get_performance_tracker

        # H-9: must query the CURRENT run's namespace (mock vs live-data-paper), not the
        # tracker's bare default -- otherwise a mock/simulated session's strategy-selection
        # prompt is built from unrelated live-data (or vice versa) history.
        tracker = get_performance_tracker(data_namespace)
        performance_data = tracker.get_all_strategy_performance(regime)
        for strategy, win_rate in sorted(
            performance_data.items(), key=lambda x: x[1], reverse=True
        ):
            perf = tracker.get_strategy_performance(strategy, regime)
            if perf.total_trades > 0:
                context_parts.append(
                    f"- {strategy}: {win_rate:.0%} win rate ({perf.total_trades} trades)"
                )
            else:
                context_parts.append(f"- {strategy}: {win_rate:.0%} win rate (prior estimate)")
    except Exception as e:
        logger.warning(f"Performance tracker unavailable, using defaults: {e}")
        default_data = {
            "trending_up": {
                "momentum": 0.60,
                "trend_following": 0.58,
                "breakout": 0.45,
                "mean_reversion": 0.40,
            },
            "trending_down": {
                "momentum": 0.55,
                "trend_following": 0.52,
                "breakout": 0.40,
                "mean_reversion": 0.42,
            },
            "ranging": {
                "mean_reversion": 0.58,
                "breakout": 0.48,
                "momentum": 0.38,
                "trend_following": 0.35,
            },
            "volatile": {
                "breakout": 0.45,
                "momentum": 0.40,
                "mean_reversion": 0.38,
                "trend_following": 0.35,
            },
        }
        if regime in default_data:
            for strategy, win_rate in default_data[regime].items():
                context_parts.append(f"- {strategy}: {win_rate:.0%} win rate (default)")

    # Add lessons
    if lessons:
        context_parts.append("\n## Past Mistakes to Avoid\n")
        for lesson in lessons[:3]:
            context_parts.append(
                f"- [{lesson.get('severity', 'N/A')}] {lesson.get('description', 'N/A')}"
            )

    return "\n".join(context_parts)


def _parse_strategy_response(content: str) -> dict[str, Any]:
    """Parse strategy selection response."""

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

        result = json.loads(content)

        # Validate strategies
        result["active_strategies"] = [
            s for s in result.get("active_strategies", []) if s in _VALID_STRATEGY_NAMES
        ]

        if not result["active_strategies"]:
            result["active_strategies"] = ["trend_following"]

        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse strategy response: {e}")
        return {
            "active_strategies": ["trend_following"],
            "reasoning": f"Parse error, defaulting to trend_following: {content[:200]}",
        }
