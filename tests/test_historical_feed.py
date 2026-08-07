"""Tests for src/market/historical_feed.py — the thin Dhan-only historical facade."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.market.historical_feed import HistoricalDataFeed


def _mock_settings(enable_dhan_historical_data=True):
    settings = MagicMock()
    settings.enable_dhan_historical_data = enable_dhan_historical_data
    return settings


_DF = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]})


def test_returns_dhan_result_when_it_succeeds():
    with (
        patch("src.market.historical_feed.get_settings", return_value=_mock_settings(True)),
        patch("src.market.historical_feed.DhanHistoricalFeed") as mock_dhan_cls,
    ):
        mock_dhan_cls.return_value.get_historical.return_value = _DF
        feed = HistoricalDataFeed(symbols=["RELIANCE"])

        result = feed.get_historical("RELIANCE", period="60d", interval="15m")

    assert result is _DF


def test_returns_none_when_dhan_returns_none():
    with (
        patch("src.market.historical_feed.get_settings", return_value=_mock_settings(True)),
        patch("src.market.historical_feed.DhanHistoricalFeed") as mock_dhan_cls,
    ):
        mock_dhan_cls.return_value.get_historical.return_value = None
        feed = HistoricalDataFeed(symbols=["RELIANCE"])

        result = feed.get_historical("RELIANCE", period="60d", interval="15m")

    assert result is None


def test_skips_dhan_entirely_when_feature_disabled():
    with (
        patch("src.market.historical_feed.get_settings", return_value=_mock_settings(False)),
        patch("src.market.historical_feed.DhanHistoricalFeed") as mock_dhan_cls,
    ):
        feed = HistoricalDataFeed(symbols=["RELIANCE"])
        result = feed.get_historical("RELIANCE")

    mock_dhan_cls.assert_not_called()
    assert result is None
    assert feed.is_available is False


def test_dhan_init_failure_degrades_to_none():
    with (
        patch("src.market.historical_feed.get_settings", return_value=_mock_settings(True)),
        patch(
            "src.market.historical_feed.DhanHistoricalFeed", side_effect=Exception("bad token")
        ),
    ):
        feed = HistoricalDataFeed(symbols=["RELIANCE"])

        result = feed.get_historical("RELIANCE")

    assert result is None
    assert feed._dhan is None
