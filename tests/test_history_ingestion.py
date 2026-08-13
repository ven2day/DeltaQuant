from datetime import UTC, datetime, timedelta

import pandas as pd

from src.core.candles import CandleStore
from src.core.indicators import Timeframe
from src.markets.nse.market_data.history_ingestion import (
    HistoricalIngestionJob,
    parse_ingestion_timeframes,
)


class _Feed:
    def __init__(self):
        self.calls = []
        self.now = datetime.now(UTC).replace(microsecond=0)

    def get_historical(self, symbol, period="1mo", interval="1d"):
        self.calls.append((symbol, period, interval))
        index = pd.DatetimeIndex(
            [self.now - timedelta(days=730), self.now - timedelta(days=1)]
        ).tz_convert("Asia/Kolkata")
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=index,
        )


def test_ingestion_backfills_then_switches_to_overlapping_incremental_tail(tmp_path):
    store = CandleStore(f"sqlite:///{tmp_path / 'ingestion.db'}")
    feed = _Feed()
    job = HistoricalIngestionJob(store, feed, lookback_days=730, overlap_days=3)

    first = job.run(["RELIANCE"], [Timeframe.M15])
    second = job.run(["RELIANCE"], [Timeframe.M15])

    assert first.series_completed == 1
    assert second.series_completed == 1
    assert feed.calls[0] == ("RELIANCE", "730d", "15m")
    assert feed.calls[1] == ("RELIANCE", "4d", "15m")
    assert store.coverage("RELIANCE", "15m").rows == 2


def test_ingestion_timeframes_keep_only_native_series():
    assert parse_ingestion_timeframes("5m,30m,15m,4h,1h,1d") == [
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.D1,
    ]
