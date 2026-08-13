from __future__ import annotations

import pandas as pd

from src.core.indicators import Timeframe
from src.markets.nse.market_data.history_manager import _nse_bucket_open, _resample_ohlcv
from src.markets.nse.sessions.market_time import IST


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = list(range(1, len(index) + 1))
    return pd.DataFrame(
        {
            "open": values,
            "high": [value + 1 for value in values],
            "low": [value - 1 for value in values],
            "close": [value + 0.5 for value in values],
            "volume": [10 * value for value in values],
        },
        index=index,
    )


def test_15m_to_30m_is_0915_aligned_with_correct_ohlcv():
    index = pd.date_range("2026-08-10 09:15", periods=4, freq="15min", tz=IST)
    result = _resample_ohlcv(_frame(index), "30min")

    assert list(result.index.strftime("%H:%M")) == ["09:15", "09:45"]
    first = result.iloc[0]
    assert first.open == 1
    assert first.high == 3
    assert first.low == 0
    assert first.close == 2.5
    assert first.volume == 30


def test_1h_to_4h_is_0915_aligned_and_keeps_partial_close_session_bucket():
    index = pd.date_range("2026-08-10 09:15", periods=7, freq="1h", tz=IST)
    result = _resample_ohlcv(_frame(index), "4h")

    assert list(result.index.strftime("%H:%M")) == ["09:15", "13:15"]
    assert result.iloc[0].volume == 100
    assert result.iloc[1].volume == 180


def test_resampling_never_crosses_sessions_or_accepts_weekend_or_known_holiday():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-07 15:15", tz=IST),  # Friday
            pd.Timestamp("2026-08-08 09:15", tz=IST),  # Saturday
            pd.Timestamp("2026-08-10 09:15", tz=IST),  # Monday
            pd.Timestamp("2026-01-26 09:15", tz=IST),  # maintained NSE holiday
        ]
    )
    result = _resample_ohlcv(_frame(index), "30min")

    assert list(result.index.date) == [
        pd.Timestamp("2026-08-07").date(),
        pd.Timestamp("2026-08-10").date(),
    ]
    assert list(result.volume) == [10, 30]


def test_live_hourly_bucket_uses_exchange_open_not_midnight():
    assert _nse_bucket_open(pd.Timestamp("2026-08-10 10:22", tz=IST), Timeframe.H1) == pd.Timestamp(
        "2026-08-10 10:15", tz=IST
    )
