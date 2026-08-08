"""Tests for src/agents/llm_factory.py: provider selection, model resolution, and the
fail-closed config validation gating Gemini/DeepSeek on their API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agents.llm_factory import (
    create_chat_model,
    current_provider,
    get_llm_circuit_breaker,
    get_llm_limiter,
    primary_and_fallback_models,
)
from src.config.settings import Settings


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
        from src.utils.rate_limiter import get_groq_limiter

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
    def test_circuit_breakers_are_named_per_provider(self, mock_get_settings):
        mock_get_settings.return_value = MagicMock(llm_provider="gemini")
        cb = get_llm_circuit_breaker()
        assert cb.name == "gemini_api"


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
