"""Provider-agnostic chat-model factory for every LLM-backed agent node.

DeltaQuant's agent nodes (market_regime, strategy_selection, signal_validation,
news_analyst) were each hardcoded to `ChatGroq`. This module lets
`settings.llm_provider` select Groq, Gemini, or DeepSeek once, for every agent, without
duplicating the primary/fallback-model resilience pattern documented in CLAUDE.md's
"LLM agent conventions":

    - acquire the shared rate limiter and circuit breaker for the *configured provider*
      before calling the LLM (see get_llm_limiter/get_llm_circuit_breaker below);
    - try the primary model, then the fallback model, on rate-limit errors;
    - on any failure, the caller returns its deterministic `_fallback_*` result --
      this module does not change that contract, it only changes which provider the
      "try the model" step talks to.

DeepSeek has no dedicated LangChain integration package; it documents OpenAI-API
compatibility (https://api-docs.deepseek.com), so `langchain_openai.ChatOpenAI` pointed
at DeepSeek's base_url is the standard way to reach it -- not a workaround.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_settings
from src.utils.circuit_breaker import CircuitBreaker, get_llm_provider_circuit_breaker
from src.utils.rate_limiter import RateLimiter, get_llm_provider_limiter

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def current_provider() -> str:
    """The currently configured LLM provider ("groq"/"gemini"/"deepseek")."""
    return get_settings().llm_provider


def primary_and_fallback_models() -> tuple[str, str]:
    """The (primary, fallback) model names for the currently configured provider."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_model_primary, settings.gemini_model_fallback
    if settings.llm_provider == "deepseek":
        return settings.deepseek_model_primary, settings.deepseek_model_fallback
    return settings.groq_model_primary, settings.groq_model_fallback


def get_llm_limiter() -> RateLimiter:
    """The shared rate limiter for the currently configured provider."""
    return get_llm_provider_limiter(current_provider())


def get_llm_circuit_breaker() -> CircuitBreaker:
    """The shared circuit breaker for the currently configured provider."""
    return get_llm_provider_circuit_breaker(current_provider())


def create_chat_model(
    model_name: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 1024,
) -> BaseChatModel:
    """Build a chat model for the currently configured provider.

    ``model_name`` is passed explicitly (rather than re-read from settings) so a
    caller's primary/fallback retry loop can request either model without this
    function needing to know which one it is -- see e.g.
    market_regime.py's ``models_to_try`` loop.
    """
    settings = get_settings()
    provider = settings.llm_provider
    resolved_temperature = settings.groq_temperature if temperature is None else temperature

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key.get_secret_value(),
            model=model_name,
            temperature=resolved_temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=model_name,
            temperature=resolved_temperature,
            max_tokens=max_tokens,
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=model_name,
        temperature=resolved_temperature,
        max_tokens=max_tokens,
    )
