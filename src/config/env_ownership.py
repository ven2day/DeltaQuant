"""Classify legacy dotenv keys for one-time profile migration."""

from __future__ import annotations

from typing import Literal

EnvOwner = Literal["common", "nse", "forex", "crypto", "ignore"]

_COMMON_PREFIXES = (
    "APP_",
    "DASHSCOPE_",
    "DEEPSEEK_",
    "FINOPS_",
    "GEMINI_",
    "GOOGLE_",
    "GROQ_",
    "LANGFUSE_",
    "LLM_",
    "LOG_",
    "QWEN_",
    "REDIS_",
    "TELEGRAM_",
    "TIMESCALE_",
    "WEB_UI_",
)
_COMMON_EXACT = {
    "DAILY_COST_BUDGET_USD",
    "DAILY_TOKEN_BUDGET",
    "DATABASE_URL",
    "ENABLE_LEARNING",
    "ENABLE_LLM_AGENTS",
    "ENABLE_NEWS_ANALYSIS",
    "GLOBAL_KILL_SWITCH",
    "MARKET_HISTORY_DATABASE_URL",
    "ML_MODEL_ROOT",
    "STRATEGY_CONFIG_ROOT",
    "TRADING_MODE",
}


def env_owner(key: str) -> EnvOwner:
    normalized = key.strip().upper()
    if not normalized or normalized == "MARKET":
        return "ignore"
    if normalized.startswith(("OANDA_", "FOREX_")):
        return "forex"
    if normalized.startswith("CRYPTO_"):
        return "crypto"
    if normalized in _COMMON_EXACT or normalized.startswith(_COMMON_PREFIXES):
        return "common"
    return "nse"


__all__ = ["EnvOwner", "env_owner"]
