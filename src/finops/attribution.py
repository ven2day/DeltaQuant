"""Canonical, market-aware attribution values for every LLM invocation."""

from __future__ import annotations

from enum import Enum


class LLMMarket(str, Enum):
    NSE = "NSE"
    FOREX = "FOREX"
    COMMON = "COMMON"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"


class LLMCallReason(str, Enum):
    CANDIDATE_REVIEW = "CANDIDATE_REVIEW"
    MARKET_CONTEXT = "MARKET_CONTEXT"
    REGIME_CONTEXT = "REGIME_CONTEXT"
    NEWS_ANALYSIS = "NEWS_ANALYSIS"
    SENTIMENT_ANALYSIS = "SENTIMENT_ANALYSIS"
    SUPPORT_AGENT = "SUPPORT_AGENT"
    BACKGROUND_ANALYSIS = "BACKGROUND_ANALYSIS"
    HEALTH_CHECK = "HEALTH_CHECK"
    MANUAL_TEST = "MANUAL_TEST"
    OTHER = "OTHER"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"


def normalize_llm_market(value: str | LLMMarket | None) -> LLMMarket:
    if isinstance(value, LLMMarket):
        return value
    normalized = str(value or "").strip().upper()
    try:
        return LLMMarket(normalized)
    except ValueError:
        return LLMMarket.UNKNOWN_LEGACY


def normalize_call_reason(value: str | LLMCallReason | None) -> LLMCallReason:
    if isinstance(value, LLMCallReason):
        return value
    normalized = str(value or "").strip().upper()
    try:
        return LLMCallReason(normalized)
    except ValueError:
        return LLMCallReason.UNKNOWN_LEGACY


CONTEXT_SUPPORT_REASONS = frozenset(
    {
        LLMCallReason.MARKET_CONTEXT.value,
        LLMCallReason.REGIME_CONTEXT.value,
        LLMCallReason.NEWS_ANALYSIS.value,
        LLMCallReason.SENTIMENT_ANALYSIS.value,
        LLMCallReason.SUPPORT_AGENT.value,
    }
)
