"""Tests for src/market/dhan_historical_feed.py — DhanHQ candle fetching.

Response shapes here mirror DhanHQ's documented /charts/intraday and
/charts/historical schema: an object of parallel arrays keyed by
open/high/low/close/volume/timestamp (epoch seconds, UTC).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.market.dhan_historical_feed import (
    DhanHistoricalFeed,
    _parse_ohlcv_response,
    _period_to_days,
)


@pytest.fixture(autouse=True)
def _no_real_rate_limiting():
    # Tests must never touch the real global DhanHQ rate-limiter singleton — it's
    # shared process-wide state (see src/utils/rate_limiter.py), and depleting its
    # token bucket across many tests causes genuine sleeps (observed: a 13-test file
    # went from <1s to 4+s before this fixture existed).
    with patch("src.market.dhan_historical_feed.get_dhan_data_api_limiter") as mock_limiter:
        yield mock_limiter


def _mock_settings():
    settings = MagicMock()
    settings.dhan_client_id = "client-id"
    return settings


def _feed(security_ids=None):
    with (
        patch("src.market.dhan_historical_feed.get_settings", return_value=_mock_settings()),
        patch("src.market.dhan_historical_feed.get_valid_access_token", return_value="token"),
        patch("src.market.dhan_historical_feed.get_dhan_client"),
        patch(
            "src.market.dhan_historical_feed.fetch_security_id_map",
            return_value=security_ids or {"RELIANCE": "2885"},
        ),
    ):
        return DhanHistoricalFeed(symbols=list((security_ids or {"RELIANCE": "2885"}).keys()))


# 2026-01-01 00:00:00 UTC = 2026-01-01 05:30:00 IST
_EPOCH_2026_01_01_UTC = 1767225600


def _sample_response(n: int = 4, start_epoch: int = _EPOCH_2026_01_01_UTC, step: int = 900):
    timestamps = [start_epoch + i * step for i in range(n)]
    return {
        "status": "success",
        "data": {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000] * n,
            "timestamp": timestamps,
        },
    }


# --- _period_to_days ---


def test_period_to_days_known_values():
    assert _period_to_days("60d") == 60
    assert _period_to_days("3mo") == 90


def test_period_to_days_clamps_to_max_request_days():
    assert _period_to_days("730d") == 90


def test_period_to_days_unknown_period_defaults_to_max():
    assert _period_to_days("whatever") == 90


# --- _parse_ohlcv_response ---


def test_parse_ohlcv_response_converts_epoch_to_ist():
    df = _parse_ohlcv_response(_sample_response(n=2)["data"])

    assert df is not None
    assert str(df.index.tz) == "Asia/Kolkata"
    # 2026-01-01 00:00:00 UTC == 2026-01-01 05:30:00 IST
    assert df.index[0].hour == 5
    assert df.index[0].minute == 30
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_parse_ohlcv_response_none_on_missing_data():
    assert _parse_ohlcv_response(None) is None
    assert _parse_ohlcv_response({}) is None


def test_parse_ohlcv_response_none_on_malformed_data():
    assert _parse_ohlcv_response({"timestamp": [1, 2], "open": [1.0]}) is None


# --- DhanHistoricalFeed.get_historical ---


def test_get_historical_daily_resamples_real_hourly_data():
    feed = _feed()
    feed._client.intraday_minute_data.return_value = _sample_response(n=3)

    df = feed.get_historical("RELIANCE", period="3mo", interval="1d")

    assert df is not None
    assert len(df) == 1
    _, kwargs = feed._client.intraday_minute_data.call_args
    assert kwargs["interval"] == 60


def test_get_historical_15m_uses_intraday_native_interval():
    feed = _feed()
    feed._client.intraday_minute_data.return_value = _sample_response(n=5)

    df = feed.get_historical("RELIANCE", period="60d", interval="15m")

    assert df is not None
    assert len(df) == 5
    _, kwargs = feed._client.intraday_minute_data.call_args
    assert kwargs["interval"] == 15


def test_get_historical_30m_resamples_from_native_15m():
    feed = _feed()
    # 4 bars of 15m spanning exactly one 30m window each pair -> 2 resampled bars.
    feed._client.intraday_minute_data.return_value = _sample_response(n=4, step=900)

    df = feed.get_historical("RELIANCE", period="60d", interval="30m")

    assert df is not None
    assert len(df) == 2
    _, kwargs = feed._client.intraday_minute_data.call_args
    assert kwargs["interval"] == 15  # requests the native interval, resamples locally


def test_get_historical_returns_none_on_dhan_failure_status():
    feed = _feed()
    feed._client.intraday_minute_data.return_value = {
        "status": "failure",
        "remarks": "DH-901 invalid token",
    }

    assert feed.get_historical("RELIANCE", period="60d", interval="15m") is None


def test_get_historical_returns_none_on_unresolved_symbol():
    feed = _feed(security_ids={})

    assert feed.get_historical("UNKNOWN", period="60d", interval="15m") is None


def test_get_historical_returns_none_on_unsupported_interval():
    feed = _feed()

    assert feed.get_historical("RELIANCE", period="60d", interval="5m") is None


def test_get_historical_returns_none_on_client_exception():
    feed = _feed()
    feed._client.intraday_minute_data.side_effect = Exception("network error")

    assert feed.get_historical("RELIANCE", period="60d", interval="15m") is None
