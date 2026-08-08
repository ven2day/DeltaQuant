"""Tests for src/webui/candles.py: OHLCV DataFrame -> chart-candle conversion."""

import numpy as np
import pandas as pd

from src.webui.candles import dataframe_to_candles


def _frame(rows: dict) -> pd.DataFrame:
    dates = pd.date_range("2026-08-01", periods=len(next(iter(rows.values()))), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=dates)


def test_none_frame_returns_empty_list():
    assert dataframe_to_candles(None) == []


def test_empty_frame_returns_empty_list():
    assert dataframe_to_candles(pd.DataFrame()) == []


def test_converts_a_clean_frame():
    frame = _frame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 2000],
        }
    )
    candles = dataframe_to_candles(frame)
    assert len(candles) == 2
    assert candles[0]["open"] == 100.0
    assert candles[0]["high"] == 102.0
    assert candles[0]["low"] == 99.0
    assert candles[0]["close"] == 101.0
    assert candles[0]["volume"] == 1000.0
    assert isinstance(candles[0]["time"], int)
    assert candles[1]["time"] > candles[0]["time"]


def test_drops_rows_with_nan_ohlc():
    frame = _frame(
        {
            "open": [100.0, np.nan],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 2000],
        }
    )
    candles = dataframe_to_candles(frame)
    assert len(candles) == 1
    assert candles[0]["open"] == 100.0


def test_missing_volume_column_defaults_to_zero():
    frame = _frame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    )
    candles = dataframe_to_candles(frame)
    assert candles[0]["volume"] == 0.0


def test_missing_ohlc_column_drops_row_rather_than_raising():
    frame = _frame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            # no "close" column at all
        }
    )
    assert dataframe_to_candles(frame) == []


def test_time_is_unix_seconds_matching_the_index():
    frame = _frame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000],
        }
    )
    candles = dataframe_to_candles(frame)
    assert candles[0]["time"] == int(frame.index[0].timestamp())
