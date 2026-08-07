"""Tests for the fast, config-based service checks in src/api/health.py.

Covers the System Status page's checks: telegram, langfuse, dhan_broker,
news_feed. The pre-existing checks (database/groq/market_data/memory/circuit
breakers/paper_wallet) are exercised implicitly wherever they were already
covered; this file focuses on the newly added ones.
"""

from unittest.mock import MagicMock, patch

from src.api.health import (
    HealthStatus,
    check_dhan_broker,
    check_langfuse,
    check_news_feed,
    check_telegram,
)


def _settings(**overrides):
    settings = MagicMock()
    settings.telegram_enabled = False
    settings.telegram_bot_token = None
    settings.telegram_chat_id = None
    settings.langfuse_tracing_enabled = False
    settings.langfuse_environment = "paper"
    settings.langfuse_host = "https://cloud.langfuse.com"
    settings.dhan_client_id = None
    settings.dhan_access_token = None
    settings.dhan_pin = None
    settings.dhan_totp_secret = None
    settings.market_data_source = "dhan"
    settings.execution_mode = "local_paper"
    settings.enable_news_analysis = False
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# --- check_telegram ---


async def test_check_telegram_healthy_when_fully_configured():
    settings = _settings(
        telegram_enabled=True, telegram_bot_token="token", telegram_chat_id="chat"
    )
    with patch("src.api.health.get_settings", return_value=settings):
        result = await check_telegram()

    assert result.status == HealthStatus.HEALTHY


async def test_check_telegram_degraded_but_not_unhealthy_when_unconfigured():
    with patch("src.api.health.get_settings", return_value=_settings()):
        result = await check_telegram()

    # Optional integration — must never report UNHEALTHY for being unset.
    assert result.status == HealthStatus.DEGRADED


# --- check_langfuse ---


async def test_check_langfuse_healthy_when_tracing_enabled():
    with patch("src.api.health.get_settings", return_value=_settings(langfuse_tracing_enabled=True)):
        result = await check_langfuse()

    assert result.status == HealthStatus.HEALTHY
    assert result.details["environment"] == "paper"


async def test_check_langfuse_degraded_when_tracing_disabled():
    with patch("src.api.health.get_settings", return_value=_settings()):
        result = await check_langfuse()

    assert result.status == HealthStatus.DEGRADED


# --- check_dhan_broker ---


async def test_check_dhan_broker_healthy_when_static_token_configured():
    settings = _settings(dhan_client_id="id", dhan_access_token="token")
    with patch("src.api.health.get_settings", return_value=settings):
        result = await check_dhan_broker()

    assert result.status == HealthStatus.HEALTHY


async def test_check_dhan_broker_healthy_when_auto_login_credentials_configured():
    # No static dhan_access_token — this is the auto-login setup (PIN+TOTP), which
    # deliberately leaves it unset since get_valid_access_token() generates one on
    # demand. Must not be reported as missing credentials.
    settings = _settings(dhan_client_id="id", dhan_pin="1234", dhan_totp_secret="SECRET")
    with patch("src.api.health.get_settings", return_value=settings):
        result = await check_dhan_broker()

    assert result.status == HealthStatus.HEALTHY


async def test_check_dhan_broker_unhealthy_when_only_pin_set_without_totp_secret():
    settings = _settings(dhan_client_id="id", dhan_pin="1234", execution_mode="live")
    with patch("src.api.health.get_settings", return_value=settings):
        result = await check_dhan_broker()

    assert result.status == HealthStatus.UNHEALTHY


async def test_check_dhan_broker_degraded_when_optional_and_unconfigured():
    """local_paper never needs Dhan — missing creds must not fail health (market data
    just falls back to simulated, per MarketDataManager)."""
    with patch("src.api.health.get_settings", return_value=_settings()):
        result = await check_dhan_broker()

    assert result.status == HealthStatus.DEGRADED


async def test_check_dhan_broker_unhealthy_when_required_by_execution_mode():
    settings = _settings(execution_mode="live")
    with patch("src.api.health.get_settings", return_value=settings):
        result = await check_dhan_broker()

    assert result.status == HealthStatus.UNHEALTHY


# --- check_news_feed ---


async def test_check_news_feed_healthy_when_enabled():
    with patch("src.api.health.get_settings", return_value=_settings(enable_news_analysis=True)):
        result = await check_news_feed()

    assert result.status == HealthStatus.HEALTHY


async def test_check_news_feed_degraded_when_disabled():
    with patch("src.api.health.get_settings", return_value=_settings()):
        result = await check_news_feed()

    assert result.status == HealthStatus.DEGRADED
