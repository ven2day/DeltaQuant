"""Tests for src/agents/llm_factory.py: provider selection, model resolution, and the
fail-closed config validation gating Gemini/DeepSeek on their API keys.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agents.llm_factory import (
    create_chat_model,
    current_provider,
    get_llm_circuit_breaker,
    get_llm_limiter,
    invoke_with_fallback,
    primary_and_fallback_models,
)
from src.config.settings import Settings
from src.core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState


def _base_kwargs(**overrides):
    # _env_file=None: isolate from the repo's real .env (see the config-failclosed
    # lesson elsewhere in this test suite) -- these tests assert behavior driven by
    # explicit kwargs/class defaults, not whatever the real .env currently contains.
    kwargs = dict(groq_api_key="x", _env_file=None)
    kwargs.update(overrides)
    return kwargs


class TestProviderModelResolution:
    @patch("src.agents.llm_factory.get_settings")
    def test_defaults_to_groq_models(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(
            llm_provider="groq",
            groq_model_primary="groq-primary",
            groq_model_fallback="groq-fallback",
        )
        assert primary_and_fallback_models() == ("groq-primary", "groq-fallback")
        assert current_provider() == "groq"

    @patch("src.agents.llm_factory.get_settings")
    def test_gemini_models(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(
            llm_provider="gemini",
            gemini_model_primary="gemini-primary",
            gemini_model_fallback="gemini-fallback",
        )
        assert primary_and_fallback_models() == ("gemini-primary", "gemini-fallback")
        assert current_provider() == "gemini"

    @patch("src.agents.llm_factory.get_settings")
    def test_deepseek_models(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(
            llm_provider="deepseek",
            deepseek_model_primary="deepseek-primary",
            deepseek_model_fallback="deepseek-fallback",
        )
        assert primary_and_fallback_models() == ("deepseek-primary", "deepseek-fallback")
        assert current_provider() == "deepseek"

    @patch("src.agents.llm_factory.get_settings")
    def test_qwen_models(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(
            llm_provider="qwen",
            qwen_model_primary="qwen-primary",
            qwen_model_fallback="qwen-fallback",
        )
        assert primary_and_fallback_models() == ("qwen-primary", "qwen-fallback")
        assert current_provider() == "qwen"


class TestCreateChatModel:
    @patch("src.agents.llm_factory.get_settings")
    def test_groq_provider_builds_chatgroq(self, mock_get_settings):
        settings = MagicMock(llm_provider="groq", groq_temperature=0.1)
        mock_get_settings.return_value = settings

        with patch("langchain_groq.ChatGroq") as mock_cls:
            create_chat_model("some-model", max_tokens=512)
            # api_key is passed as the SecretStr object itself (not
            # .get_secret_value()) -- ChatGroq's api_key field expects SecretStr.
            mock_cls.assert_called_once_with(
                api_key=settings.groq_api_key,
                model_name="some-model",
                temperature=0.1,
                max_tokens=512,
            )

    @patch("src.agents.llm_factory.get_settings")
    def test_gemini_provider_builds_chat_google_generative_ai(self, mock_get_settings):
        settings = MagicMock(llm_provider="gemini", groq_temperature=0.1)
        settings.google_api_key.get_secret_value.return_value = "gemini-key"
        mock_get_settings.return_value = settings

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
            create_chat_model("gemini-2.0-flash", max_tokens=512)
            mock_cls.assert_called_once_with(
                google_api_key="gemini-key",
                model="gemini-2.0-flash",
                temperature=0.1,
                max_output_tokens=512,
            )

    @patch("src.agents.llm_factory.get_settings")
    def test_deepseek_provider_builds_chatopenai_with_deepseek_base_url(self, mock_get_settings):
        settings = MagicMock(llm_provider="deepseek", groq_temperature=0.1)
        mock_get_settings.return_value = settings

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            create_chat_model("deepseek-chat", max_tokens=512)
            # api_key is passed as the SecretStr object itself (not
            # .get_secret_value()) -- ChatOpenAI's api_key field expects SecretStr.
            mock_cls.assert_called_once_with(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=512,
            )

    @patch("src.agents.llm_factory.get_settings")
    def test_qwen_provider_builds_chatopenai_with_qwen_base_url(self, mock_get_settings):
        settings = MagicMock(
            llm_provider="qwen",
            groq_temperature=0.1,
            qwen_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        mock_get_settings.return_value = settings

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            create_chat_model("qwen3.7-plus", max_tokens=512)
            mock_cls.assert_called_once_with(
                api_key=settings.dashscope_api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                model="qwen3.7-plus",
                temperature=0.1,
                max_tokens=512,
                extra_body={"enable_thinking": False},
            )

    @patch("src.agents.llm_factory.get_settings")
    def test_qwen_provider_disables_thinking_mode(self, mock_get_settings):
        """Regression guard: qwen3.7-plus spends completion tokens on hidden
        chain-of-thought by default, which was exhausting max_tokens before any
        visible answer -- every regime/strategy/validation call returned empty
        content and failed JSON parsing. Confirmed live against the real API."""
        settings = MagicMock(llm_provider="qwen", groq_temperature=0.1)
        mock_get_settings.return_value = settings

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            create_chat_model("qwen3.7-plus", max_tokens=512)
            _args, kwargs = mock_cls.call_args
            assert kwargs["extra_body"] == {"enable_thinking": False}

    @patch("src.agents.llm_factory.get_settings")
    def test_explicit_temperature_overrides_default(self, mock_get_settings):
        settings = MagicMock(llm_provider="groq", groq_temperature=0.1)
        settings.groq_api_key.get_secret_value.return_value = "groq-key"
        mock_get_settings.return_value = settings

        with patch("langchain_groq.ChatGroq") as mock_cls:
            create_chat_model("some-model", temperature=0.9, max_tokens=512)
            _args, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.9


class TestProviderLimiterAndCircuitBreaker:
    @patch("src.agents.llm_factory.get_settings")
    def test_groq_limiter_is_the_singleton_groq_limiter(self, mock_get_settings):
        from src.core.utils.rate_limiter import get_groq_limiter

        mock_get_settings.return_value = MagicMock(llm_provider="groq")
        assert get_llm_limiter() is get_groq_limiter()

    @patch("src.agents.llm_factory.get_settings")
    def test_gemini_and_deepseek_limiters_are_distinct(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(llm_provider="gemini")
        gemini_limiter = get_llm_limiter()

        mock_get_settings.return_value = MagicMock(llm_provider="deepseek")
        deepseek_limiter = get_llm_limiter()

        assert gemini_limiter is not deepseek_limiter

    @patch("src.agents.llm_factory.get_settings")
    def test_qwen_limiter_is_distinct_from_other_providers(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(llm_provider="groq")
        groq_limiter = get_llm_limiter()

        mock_get_settings.return_value = MagicMock(llm_provider="qwen")
        qwen_limiter = get_llm_limiter()

        assert qwen_limiter is not groq_limiter

    @patch("src.agents.llm_factory.get_settings")
    def test_circuit_breakers_are_named_per_provider(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(llm_provider="gemini")
        cb = get_llm_circuit_breaker()
        assert cb.name == "gemini_api"

    @patch("src.agents.llm_factory.get_settings")
    def test_qwen_circuit_breaker_is_named_per_provider(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(llm_provider="qwen")
        cb = get_llm_circuit_breaker()
        assert cb.name == "qwen_api"


class TestFailClosedProviderConfig:
    def test_groq_provider_needs_no_extra_keys(self):
        s = Settings(**_base_kwargs())
        assert s.llm_provider == "groq"

    def test_gemini_provider_without_key_fails_startup(self):
        with pytest.raises(ValidationError, match="LLM_PROVIDER=gemini requires GOOGLE_API_KEY"):
            Settings(**_base_kwargs(llm_provider="gemini"))

    def test_gemini_provider_with_key_constructs(self):
        s = Settings(**_base_kwargs(llm_provider="gemini", google_api_key="g-key"))
        assert s.llm_provider == "gemini"

    def test_deepseek_provider_without_key_fails_startup(self):
        with pytest.raises(
            ValidationError, match="LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY"
        ):
            Settings(**_base_kwargs(llm_provider="deepseek"))

    def test_deepseek_provider_with_key_constructs(self):
        s = Settings(**_base_kwargs(llm_provider="deepseek", deepseek_api_key="d-key"))
        assert s.llm_provider == "deepseek"

    def test_qwen_provider_without_key_fails_startup(self):
        with pytest.raises(
            ValidationError, match="LLM_PROVIDER=qwen requires DASHSCOPE_API_KEY"
        ):
            Settings(**_base_kwargs(llm_provider="qwen"))

    def test_qwen_provider_with_key_constructs(self):
        s = Settings(**_base_kwargs(llm_provider="qwen", dashscope_api_key="q-key"))
        assert s.llm_provider == "qwen"


class TestInvokeWithFallback:
    """invoke_with_fallback (used by market_regime, strategy_selection, and
    signal_validation) is the shared retry-on-429 path documented in CLAUDE.md's "LLM
    agent conventions". These tests guard the production incident it was built to fix:
    the primary model's Groq daily token quota ran out, three consecutive 429s tripped
    the shared "groq_api" circuit breaker, and from that point every node -- including
    ones that would have succeeded on the still-healthy fallback model -- failed closed
    to crude rule-based logic instead of ever trying it.
    """

    @patch("src.agents.llm_factory.get_settings")
    def test_fallback_model_used_on_primary_rate_limit(self, mock_get_settings, monkeypatch):
        mock_get_settings.return_value = MagicMock(
            llm_provider="groq",
            groq_model_primary="primary-model",
            groq_model_fallback="fallback-model",
            enable_rate_limiting=False,
        )
        ok_agent = MagicMock()
        ok_agent.invoke.return_value = "OK"
        monkeypatch.setattr(
            "src.agents.llm_factory.create_chat_model",
            MagicMock(side_effect=[Exception("Error 429: rate_limit_exceeded"), ok_agent]),
        )
        breaker = CircuitBreaker(name="test_groq", failure_threshold=2, recovery_time=30.0)
        used_models: list[str] = []

        response = invoke_with_fallback(
            [], circuit_breaker=breaker, on_model_selected=used_models.append
        )

        assert response == "OK"
        assert used_models == ["fallback-model"]

    @patch("src.agents.llm_factory.get_settings")
    def test_repeated_rate_limits_never_trip_the_breaker(self, mock_get_settings, monkeypatch):
        """Many consecutive 429s (well past failure_threshold) must never open the
        breaker -- a 429 proves the provider is reachable; only genuine outages should
        count against it. This is the exact scenario that broke production: a daily
        quota exhausted on the primary model fires a 429 on every single cycle."""
        mock_get_settings.return_value = MagicMock(
            llm_provider="groq",
            groq_model_primary="primary-model",
            groq_model_fallback="fallback-model",
            enable_rate_limiting=False,
        )
        monkeypatch.setattr(
            "src.agents.llm_factory.create_chat_model",
            MagicMock(side_effect=Exception("Error 429: rate_limit_exceeded")),
        )
        breaker = CircuitBreaker(name="test_groq", failure_threshold=2, recovery_time=30.0)

        for _ in range(10):
            with pytest.raises(Exception, match="429"):
                invoke_with_fallback([], circuit_breaker=breaker)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available

    @patch("src.agents.llm_factory.get_settings")
    def test_genuine_failure_still_trips_the_breaker(self, mock_get_settings, monkeypatch):
        """A real outage (not a 429) on both models must still count -- the breaker's
        actual job (protecting against a genuinely down provider) is preserved."""
        mock_get_settings.return_value = MagicMock(
            llm_provider="groq",
            groq_model_primary="primary-model",
            groq_model_fallback="fallback-model",
            enable_rate_limiting=False,
        )
        monkeypatch.setattr(
            "src.agents.llm_factory.create_chat_model",
            MagicMock(side_effect=Exception("Connection refused")),
        )
        breaker = CircuitBreaker(name="test_groq", failure_threshold=2, recovery_time=30.0)

        with pytest.raises(Exception, match="Connection refused"):
            invoke_with_fallback([], circuit_breaker=breaker)

        # Both the primary and fallback attempts failed genuinely -> 2 failures ->
        # threshold reached.
        assert breaker.state == CircuitState.OPEN

    @patch("src.agents.llm_factory.get_settings")
    def test_already_open_breaker_is_not_retried(self, mock_get_settings, monkeypatch):
        mock_get_settings.return_value = MagicMock(
            llm_provider="groq",
            groq_model_primary="primary-model",
            groq_model_fallback="fallback-model",
        )
        mock_create = MagicMock()
        monkeypatch.setattr("src.agents.llm_factory.create_chat_model", mock_create)
        breaker = CircuitBreaker(name="test_groq", failure_threshold=1, recovery_time=30.0)
        breaker._transition_to(CircuitState.OPEN)
        breaker._last_failure_time = time.time()

        with pytest.raises(CircuitBreakerOpenError):
            invoke_with_fallback([], circuit_breaker=breaker)

        mock_create.assert_not_called()
