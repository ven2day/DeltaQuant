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
from src.backtesting.strategy_registry import StrategyRegistry
from src.config import get_settings
from src.finops import record_llm_response
from src.market.signals import StrategyType
from src.utils.circuit_breaker import CircuitBreakerOpenError
from src.utils.formatting import plain_english_fallback_cause

from .state import TradingState

logger = logging.getLogger(__name__)

# Every strategy name the LLM is allowed to select, derived from the same enum the
# signal engine and the H-8 admission gate use -- one source of truth instead of a
# hardcoded list that could silently drift out of sync when a strategy is added.
_VALID_STRATEGY_NAMES = [member.value for member in StrategyType]


def _gate_active_strategies(
    strategies: list[str], regime: str, *, trade_horizon: str = "SWING"
) -> list[str]:
    """Strip any strategy without a current, non-expired VALIDATED registry artifact
    for this exact trade_horizon (H-8, DeltaQuant-Quant-Risk-Review.md).

    Fail closed: an unknown/unvalidated/expired/wrong-horizon strategy is silently
    *not* removed with just a warning -- it never reaches ``active_strategies`` at
    all, so it cannot survive downstream into a validated signal or an approved
    trade. Every strip is still logged loudly (not swallowed) so an operator can see
    why nothing traded. ``trade_horizon`` is matched exactly against the registry
    artifact -- a SWING-validated artifact can never admit a SCALP request, so
    promoting scalping never silently inherits swing's (daily-bar) validation.
    """
    settings = get_settings()
    registry_dir = getattr(settings, "strategy_registry_dir", "data/strategy_registry")
    registry = StrategyRegistry(str(registry_dir))
    admitted, stripped = registry.filter_admitted(
        strategies, regime=regime, trade_horizon=trade_horizon
    )
    if stripped:
        logger.warning(
            "Strategy admission gate (H-8): stripped %s from active strategies for regime "
            "'%s' / trade_horizon '%s' -- no current VALIDATED artifact in %s. Run "
            "`uv run python scripts/validate_strategy.py` to (re)validate.",
            stripped,
            regime,
            trade_horizon,
            registry_dir,
        )
    return admitted


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

Consider:
- Avoid strategies that historically underperformed in the current regime
- Don't be too aggressive - selecting 1-2 strategies is often better than all 4
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

    settings = get_settings()

    if not settings.enable_llm_agents:
        return _fallback_strategy_selection(
            state, state.get("regime", "unknown"), "LLM agents disabled via settings"
        )

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
            max_tokens=1024,
            on_model_selected=_record_model_used,
        )
        record_llm_response("strategy_selection", response, model=model_used)
        result = _parse_strategy_response(response.content)

        trade_horizon = state.get("trade_horizon", "SWING")
        admitted = _gate_active_strategies(
            result["active_strategies"], regime, trade_horizon=trade_horizon
        )
        logger.info(f"Selected strategies: {result['active_strategies']} -> admitted: {admitted}")

        return {
            "active_strategies": admitted,
            "strategy_reasoning": result["reasoning"],
            "messages": [response],
        }

    except CircuitBreakerOpenError as e:
        logger.warning(f"Circuit breaker open: {e}")
        return _fallback_strategy_selection(state, regime, str(e))

    except Exception as e:
        logger.error(f"Strategy Selection Agent error: {e}")
        return _fallback_strategy_selection(state, state.get("regime", "unknown"), str(e))


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

    trade_horizon = state.get("trade_horizon", "SWING")
    strategies = regime_strategies.get(regime, ["trend_following"])
    admitted = _gate_active_strategies(strategies, regime, trade_horizon=trade_horizon)

    logger.info(f"Using fallback strategies for {regime}: {strategies} -> admitted: {admitted}")

    return {
        "active_strategies": admitted,
        "strategy_reasoning": (
            f"AI review skipped because {plain_english_fallback_cause(error_msg)}. "
            f"Backup rule selected the default strategy set for '{regime}'."
        ),
        "errors": state.get("errors", []) + [f"Strategy Agent fallback: {error_msg}"],
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
