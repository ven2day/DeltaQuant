"""Provider-agnostic chat-model factory for every LLM-backed agent node.

DeltaQuant's agent nodes (market_regime, strategy_selection, signal_validation,
news_analyst) were each hardcoded to `ChatGroq`. This module lets
`settings.llm_provider` select Groq, Gemini, DeepSeek, or Qwen once, for every agent,
without duplicating the primary/fallback-model resilience pattern documented in
CLAUDE.md's "LLM agent conventions":

    - acquire the shared rate limiter and circuit breaker for the *configured provider*
      before calling the LLM (see get_llm_limiter/get_llm_circuit_breaker below);
    - try the primary model, then the fallback model, on rate-limit errors;
    - on any failure, the caller returns its deterministic `_fallback_*` result --
      this module does not change that contract, it only changes which provider the
      "try the model" step talks to.

DeepSeek and Qwen have no dedicated LangChain integration package; both document
OpenAI-API compatibility (https://api-docs.deepseek.com,
https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope),
so `langchain_openai.ChatOpenAI` pointed at each provider's base_url is the standard way
to reach them -- not a workaround.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.config import get_settings
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    get_llm_provider_circuit_breaker,
)
from src.utils.errors import RateLimitError
from src.utils.rate_limiter import RateLimiter, get_llm_provider_limiter

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def current_provider() -> str:
    """The currently configured LLM provider ("groq"/"gemini"/"deepseek"/"qwen")."""
    return get_settings().llm_provider


def primary_and_fallback_models() -> tuple[str, str]:
    """The (primary, fallback) model names for the currently configured provider."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_model_primary, settings.gemini_model_fallback
    if settings.llm_provider == "deepseek":
        return settings.deepseek_model_primary, settings.deepseek_model_fallback
    if settings.llm_provider == "qwen":
        return settings.qwen_model_primary, settings.qwen_model_fallback
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

    if provider == "qwen":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            model=model_name,
            temperature=resolved_temperature,
            max_tokens=max_tokens,
            # Qwen3.7 is a "thinking" model by default: it spends completion tokens on
            # hidden chain-of-thought before the visible answer (confirmed live --
            # 222 of 232 completion tokens were reasoning_tokens on a trivial prompt).
            # None of this codebase's agents consume that reasoning content; they only
            # read .content expecting the final JSON. On real prompts, reasoning alone
            # was exhausting the whole max_tokens budget, leaving .content empty and
            # failing JSON parsing every time. Disabling it makes Qwen behave like every
            # other provider here: max_tokens buys only the answer.
            extra_body={"enable_thinking": False},
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=model_name,
        temperature=resolved_temperature,
        max_tokens=max_tokens,
    )


def invoke_with_fallback(
    messages: list[BaseMessage],
    *,
    circuit_breaker: CircuitBreaker,
    max_tokens: int = 1024,
    temperature: float | None = None,
    on_model_selected: Callable[[str], None] | None = None,
) -> Any:
    """Try the primary model, then the fallback model, on rate-limit (429) errors.

    Every LLM node is supposed to have this resilience per CLAUDE.md's "LLM agent
    conventions" -- market_regime.py originally implemented it inline, but
    strategy_selection.py and signal_validation.py never did, so once the primary
    model's daily token quota was exhausted, those two nodes failed straight to their
    crude rule-based fallback for the rest of the day even though the fallback *model*
    (a different model, hence a different Groq daily-quota bucket) was still working
    fine for market_regime and news_analyst. Centralizing the retry loop here (instead
    of a third hand-copied version) makes that impossible to omit by accident again.

    Raises the last error if every model fails for a non-rate-limit reason, or
    ``CircuitBreakerOpenError`` immediately if the breaker is already open.

    A 429/rate-limit response is deliberately never recorded as a circuit-breaker
    failure. It proves the provider *is* reachable -- only this specific model's quota
    is exhausted -- so counting it the same as a genuine outage (timeout, connection
    error, 5xx) would let a primary-model quota exhaustion trip the shared
    provider-level breaker and then block the very fallback-model attempt that's
    supposed to route around it. (Confirmed in production: the primary model's daily
    TPD limit ran out, three consecutive 429s tripped the "groq_api" breaker, and every
    node -- including ones that would have succeeded on the fallback model -- started
    failing closed to crude fallback logic instead of trying it.) The breaker still
    protects against genuine outages: a non-rate-limit failure is recorded normally.
    """
    settings = get_settings()
    models_to_try = list(primary_and_fallback_models())
    last_error: Exception | None = None

    if not circuit_breaker.is_available:
        raise CircuitBreakerOpenError(circuit_breaker.name, circuit_breaker.recovery_time)

    for model_name in models_to_try:
        try:
            agent = create_chat_model(model_name, temperature=temperature, max_tokens=max_tokens)
            response = agent.invoke(messages)
        except Exception as model_error:
            last_error = model_error
            error_str = str(model_error).lower()
            if "rate_limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit on {model_name}, trying fallback...")
                if settings.enable_rate_limiting:
                    time.sleep(2)
            else:
                logger.error(f"LLM error on {model_name}: {model_error}")
                circuit_breaker._record_failure(model_error)
            continue
        else:
            circuit_breaker._record_success()
            if on_model_selected is not None:
                on_model_selected(model_name)
            return response

    if last_error:
        raise last_error
    raise RateLimitError(current_provider(), retry_after=60.0)
