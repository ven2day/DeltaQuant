"""Tests for HistoryManager's multi-timeframe intraday history support."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.indicators import Timeframe
from src.markets.nse.market_data.history_manager import HistoryManager


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
    with (
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=False),
        patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch,
    ):
        hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        assert mock_fetch.call_count == 1

        # Force the cached fetch to look stale.
        key = ("TEST", Timeframe.M15)
        hm._intraday_last_fetch[key] = datetime.now() - timedelta(hours=1)

        hm.get_multi_timeframe_history("TEST", Timeframe.M15)
        assert mock_fetch.call_count == 2


def test_m5_intraday_timeframe_fetches_and_caches(hm):
    """M5 is a native, directly-fetched timeframe (not resampled from anything) --
    same fetch/cache contract as M15, just its own native interval string."""
    fetched = _ohlcv(100, "5min")
    with patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch:
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M5, bars=50)
        assert result is not None
        assert len(result) == 50
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        mock_fetch.assert_called_once_with("TEST", period="60d", interval="5m")

        # Second call within the TTL window must not re-fetch.
        hm.get_multi_timeframe_history("TEST", Timeframe.M5)
        assert mock_fetch.call_count == 1


def test_m5_refetches_after_ttl_expires(hm):
    fetched = _ohlcv(50, "5min")
    with (
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=False),
        patch.object(hm._feed, "get_historical", return_value=fetched) as mock_fetch,
    ):
        hm.get_multi_timeframe_history("TEST", Timeframe.M5)
        assert mock_fetch.call_count == 1

        # Force the cached fetch to look stale.
        key = ("TEST", Timeframe.M5)
        hm._intraday_last_fetch[key] = datetime.now() - timedelta(hours=1)

        hm.get_multi_timeframe_history("TEST", Timeframe.M5)
        assert mock_fetch.call_count == 2


def test_m5_has_an_explicit_refresh_ttl_not_the_implicit_default():
    """Stage 10 scans M5 for every symbol every cycle (not just candidate-eval as
    before), so the TTL should be a deliberate, named value rather than silently
    inherited from INTRADAY_REFRESH_SECONDS.get(tf, 300)'s fallback."""
    from src.markets.nse.market_data.history_manager import INTRADAY_REFRESH_SECONDS

    assert Timeframe.M5 in INTRADAY_REFRESH_SECONDS
    assert INTRADAY_REFRESH_SECONDS[Timeframe.M5] == 5 * 60


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
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=False),
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
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=True),
        patch.object(hm._feed, "get_historical", return_value=fetched) as mock_real_fetch,
    ):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.M15, bars=10)

    assert result is not None
    mock_real_fetch.assert_called_once()
    sim.get_history.assert_not_called()


def test_daily_timeframe_also_switches_to_simulated_when_closed(hm_with_sim):
    hm, sim = hm_with_sim
    sim.get_history.return_value = _ohlcv(30, "D")
    with patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=False):
        result = hm.get_multi_timeframe_history("TEST", Timeframe.D1, bars=5)

    assert result is not None
    sim.get_history.assert_called_once_with("TEST", "1d", 5)


def test_none_simulated_stream_is_unaffected_by_market_hours(hm):
    """A HistoryManager with no simulated_stream configured at all (the default,
    e.g. every existing test in this file) must behave identically regardless of
    real market hours -- confirms this feature is purely additive."""
    fetched = _ohlcv(50, "15min")
    with (
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=False),
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


def test_short_live_tail_merges_persistent_prefix_and_prefers_live_duplicate():
    store = MagicMock()
    stored = _ohlcv(80, "15min")
    stored.columns = [column.lower() for column in stored.columns]
    live = stored.iloc[-3:].copy()
    live.iloc[-1, live.columns.get_loc("close")] = 999.0
    store.load_frame.return_value = stored
    manager = HistoryManager(symbols=["TEST"], candle_store=store)
    manager._intraday_history[("TEST", Timeframe.M15)] = live

    with patch.object(manager._feed, "get_historical") as broker_fetch:
        result = manager.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50)

    assert result is not None and len(result) == 50
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.iloc[-1]["close"] == 999.0
    store.load_frame.assert_called_once_with(
        "TEST", "15m", bars=50, complete_only=True
    )
    broker_fetch.assert_not_called()


def test_timeframe_readiness_is_independent_and_does_not_refetch_sufficient_5m():
    store = MagicMock()
    five = _ohlcv(60, "5min")
    fifteen = _ohlcv(60, "15min")
    five.columns = [column.lower() for column in five.columns]
    fifteen.columns = [column.lower() for column in fifteen.columns]
    store.load_frame.return_value = fifteen
    manager = HistoryManager(symbols=["TEST"], candle_store=store)
    manager._intraday_history[("TEST", Timeframe.M5)] = five

    with (
        patch("src.markets.nse.market_data.history_manager.is_market_hours", return_value=True),
        patch.object(manager._feed, "get_historical") as broker_fetch,
    ):
        assert manager.get_multi_timeframe_history("TEST", Timeframe.M5, bars=50) is not None
        assert manager.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50) is not None

    store.load_frame.assert_called_once_with(
        "TEST", "15m", bars=50, complete_only=True
    )
    broker_fetch.assert_not_called()


def test_broker_history_is_used_only_when_persistent_coverage_remains_insufficient():
    store = MagicMock()
    stored = _ohlcv(10, "15min")
    fetched = _ohlcv(80, "15min")
    stored.columns = [column.lower() for column in stored.columns]
    store.load_frame.return_value = stored
    manager = HistoryManager(symbols=["TEST"], candle_store=store)

    with patch.object(manager._feed, "get_historical", return_value=fetched) as broker_fetch:
        result = manager.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50)

    assert result is not None and len(result) == 50
    broker_fetch.assert_called_once_with("TEST", period="60d", interval="15m")


def test_insufficient_history_fetch_failure_is_throttled_between_scheduler_wakes():
    store = MagicMock()
    stored = _ohlcv(10, "15min")
    stored.columns = [column.lower() for column in stored.columns]
    store.load_frame.return_value = stored
    manager = HistoryManager(symbols=["TEST"], candle_store=store)

    with patch.object(manager._feed, "get_historical", return_value=None) as broker_fetch:
        first = manager.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50)
        second = manager.get_multi_timeframe_history("TEST", Timeframe.M15, bars=50)

    assert first is not None and second is not None
    assert broker_fetch.call_count == 1
    assert store.load_frame.call_count == 1


def test_derived_timeframes_request_sufficient_native_persistent_history():
    def session_frame(days: int, *, periods: int, frequency: str) -> pd.DataFrame:
        timestamps = [
            timestamp
            for day in pd.bdate_range("2026-06-01", periods=days)
            for timestamp in pd.date_range(
                day + pd.Timedelta(hours=9, minutes=15),
                periods=periods,
                freq=frequency,
            )
        ]
        values = list(range(len(timestamps)))
        return pd.DataFrame(
            {
                "open": [100.0 + value for value in values],
                "high": [101.0 + value for value in values],
                "low": [99.0 + value for value in values],
                "close": [100.5 + value for value in values],
                "volume": [1000 + value for value in values],
            },
            index=pd.DatetimeIndex(timestamps),
        )

    store = MagicMock()
    m15 = session_frame(12, periods=25, frequency="15min")
    h1 = session_frame(30, periods=7, frequency="1h")
    store.load_frame.side_effect = lambda symbol, timeframe, **kwargs: (
        m15.tail(kwargs["bars"]) if timeframe == "15m" else h1.tail(kwargs["bars"])
    )
    manager = HistoryManager(symbols=["TEST"], candle_store=store)

    with patch.object(manager._feed, "get_historical") as broker_fetch:
        m30 = manager.get_multi_timeframe_history("TEST", Timeframe.M30, bars=40)
        h4 = manager.get_multi_timeframe_history("TEST", Timeframe.H4, bars=40)

    assert m30 is not None and len(m30) == 40
    assert h4 is not None and len(h4) == 40
    requested = {(call.args[1], call.kwargs["bars"]) for call in store.load_frame.call_args_list}
    assert ("15m", 80) in requested
    assert ("1h", 160) in requested
    broker_fetch.assert_not_called()


def test_get_settled_history_excludes_current_forming_candle():
    manager = HistoryManager(symbols=["TEST"])
    frame = _ohlcv(2, "5min")
    frame.columns = [column.lower() for column in frame.columns]
    frame.index = pd.DatetimeIndex(["2026-08-10 10:15", "2026-08-10 10:20"])
    manager._intraday_history[("TEST", Timeframe.M5)] = frame

    with patch.object(manager._feed, "get_historical", return_value=None):
        settled = manager.get_settled_history(
            "TEST",
            Timeframe.M5,
            bars=20,
            as_of=pd.Timestamp("2026-08-10 10:22", tz="Asia/Kolkata"),
        )

    assert settled is not None
    assert list(settled.index.strftime("%H:%M")) == ["10:15"]


def test_live_quotes_keep_native_intraday_caches_fresh_without_refetch(hm):
    """Once bootstrapped, WebSocket quotes maintain all native candle caches."""
    timestamp = datetime(2026, 8, 10, 10, 1)
    for timeframe, freq in (
        (Timeframe.M5, "5min"),
        (Timeframe.M15, "15min"),
        (Timeframe.H1, "1h"),
    ):
        frame = _ohlcv(60, freq)
        frame.columns = [column.lower() for column in frame.columns]
        # Native candles are exchange-session aligned. At 10:01 the active hourly
        # candle opened at 09:15, not pandas' midnight-anchored 10:00 bucket.
        end = (
            datetime(2026, 8, 10, 9, 15)
            if timeframe is Timeframe.H1
            else datetime(2026, 8, 10, 10, 0)
        )
        frame.index = pd.date_range(end=end, periods=60, freq=freq)
        hm._intraday_history[("TEST", timeframe)] = frame
        hm._intraday_last_fetch[("TEST", timeframe)] = datetime.now() - timedelta(hours=2)

    hm.append_intraday_quote("TEST", 200.0, 10_000, timestamp)
    hm.append_intraday_quote("TEST", 205.0, 10_025, timestamp + timedelta(minutes=1))

    with patch.object(hm._feed, "get_historical") as mock_fetch:
        for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
            result = hm.get_multi_timeframe_history("TEST", timeframe)
            assert result is not None
            assert result.iloc[-1]["close"] == 205.0
        mock_fetch.assert_not_called()


def test_live_quote_starts_a_new_five_minute_candle(hm):
    frame = _ohlcv(60, "5min")
    frame.columns = [column.lower() for column in frame.columns]
    # Pin the final bar to a known bucket so the next quote crosses the boundary.
    end = datetime(2026, 8, 10, 10, 0)
    frame.index = pd.date_range(end=end, periods=60, freq="5min")
    hm._intraday_history[("TEST", Timeframe.M5)] = frame

    hm.append_intraday_quote("TEST", 200.0, 10_000, end)
    hm.append_intraday_quote("TEST", 201.0, 10_012, end + timedelta(minutes=5))

    updated = hm._intraday_history[("TEST", Timeframe.M5)]
    assert updated.index[-1] == pd.Timestamp(end + timedelta(minutes=5))
    assert updated.iloc[-1]["open"] == 201.0
    assert updated.iloc[-1]["volume"] == 12
