"""Tests for HistoryManager's multi-timeframe intraday history support."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.market.history_manager import HistoryManager
from src.market.indicators import Timeframe


def _ohlcv(n: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(), periods=n, freq=freq)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


@pytest.fixture
def hm():
    return HistoryManager(symbols=["TEST"])


def test_d1_delegates_to_get_history(hm):
    df = _ohlcv(60, "D")
    df.columns = [c.lower() for c in df.columns]
    hm._history["TEST"] = df

    result = hm.get_multi_timeframe_history("TEST", Timeframe.D1, bars=10)

    assert result is not None
    assert len(result) == 10


def test_intraday_timeframe_fetches_and_caches(hm):
    fetched = _ohlcv(100, "15min")
    with patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch:
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50)
        assert result is not None
        assert len(result) == 50
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        mock_fetch.assert_called_once_with("TEST", period="60d", interval="15m")

        # Second call within the TTL window must not re-fetch.
        hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        assert mock_fetch.call_count == 1


def test_intraday_refetches_after_ttl_expires(hm):
    fetched = _ohlcv(50, "15min")
    with patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch:
        hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        assert mock_fetch.call_count == 1

        # Force the cached fetch to look stale.
        key = ("TEST", Timeframe.M15)
        hm._intraday_last_fetch[key] = datetime.now() - timedelta(hours=1)

        hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        assert mock_fetch.call_count == 2


def test_m30_resamples_from_m15_without_a_second_fetch(hm):
    fetched = _ohlcv(200, "15min")
    with patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch:
        m15 = hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        m30 = hm.get_multi_timeframe_history("TEST", Timeframe.M30)

    assert m15 is not None and m30 is not None
    assert len(m30) < len(m15)
    assert mock_fetch.call_count == 1


def test_intraday_fetch_failure_returns_none_when_nothing_cached(hm):
    with patch.object(hm._feed, "get_historical", return_value=None):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.H1)
        assert result is None


def test_intraday_fetch_failure_falls_back_to_cache(hm):
    fetched = _ohlcv(50, "60min")
    with patch.object(hm._feed, "get_historical", return_value=fetched):
        first = hm.get_multi_timeframe_history("TEST", Timeframe.H1)
        assert first is not None

    key = ("TEST", Timeframe.H1)
    hm._intraday_last_fetch[key] = datetime.now() - timedelta(hours=1)
    with patch.object(hm._feed, "get_historical", side_effect=Exception("network down")):
        second = hm.get_multi_timeframe_history("TEST", Timeframe.H1)
        assert second is not None
        assert len(second) == len(fetched)


# --- Dynamic real/simulated switching (closed-market pipeline testing) ---


@pytest.fixture
def hm_with_sim():
    sim = MagicMock()
    sim.get_history.return_value = _ohlcv(50, "15min")
    manager = HistoryManager(symbols=["TEST"], simulated_stream=sim)
    return manager, sim


def test_uses_simulated_stream_when_market_closed(hm_with_sim):
    hm, sim = hm_with_sim
    with (
        patch("src.market.history_manager.is_market_hours", return_value=False),
        patch.object(hm._feed, "get_historical") as mock_real_fetch,
    ):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15, bars=10)

    assert result is not None
    sim.get_history.assert_called_once_with("TEST", "15m", 10)
    mock_real_fetch.assert_not_called()


def test_uses_real_feed_when_market_open_even_with_simulated_stream_configured(hm_with_sim):
    hm, sim = hm_with_sim
    fetched = _ohlcv(50, "15min")
    with (
        patch("src.market.history_manager.is_market_hours", return_value=True),
        patch.object(hm._feed, "get_historical", return_value=fetched) as mock_real_fetch,
    ):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15, bars=10)

    assert result is not None
    mock_real_fetch.assert_called_once()
    sim.get_history.assert_not_called()


def test_daily_timeframe_also_switches_to_simulated_when_closed(hm_with_sim):
    hm, sim = hm_with_sim
    sim.get_history.return_value = _ohlcv(30, "D")
    with patch("src.market.history_manager.is_market_hours", return_value=False):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.D1, bars=5)

    assert result is not None
    sim.get_history.assert_called_once_with("TEST", "1d", 5)


def test_none_simulated_stream_is_unaffected_by_market_hours(hm):
    """A HistoryManager with no simulated_stream configured at all (the default,
    e.g. every existing test in this file) must behave identically regardless of
    real market hours -- confirms this feature is purely additive."""
    fetched = _ohlcv(50, "15min")
    with (
        patch("src.market.history_manager.is_market_hours", return_value=False),
        patch.object(hm._feed, "get_historical", return_value=fetched) as mock_real_fetch,
    ):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15)

    assert result is not None
    mock_real_fetch.assert_called_once()


def test_h4_resamples_from_h1(hm):
    fetched = _ohlcv(200, "60min")
    with patch.object(hm._feed, "get_historical", return_value=fetched):
        h1 = hm.get_multi_timeframe_history("TEST", Timeframe.H1)
        h4 = hm.get_multi_timeframe_history("TEST", Timeframe.H4)

    assert h1 is not None and h4 is not None
    assert len(h4) < len(h1)
    assert list(h4.columns) == ["open", "high", "low", "close", "volume"]


def test_h4_returns_none_without_h1_data(hm):
    with patch.object(hm._feed, "get_historical", return_value=None):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.H4)
        assert result is None


def test_bars_trims_result(hm):
    fetched = _ohlcv(100, "15min")
    with patch.object(hm._feed, "get_historical", return_value=fetched):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15, bars=20)
        assert len(result) == 20
