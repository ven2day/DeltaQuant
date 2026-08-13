from datetime import UTC, datetime, timedelta

import pandas as pd

from src.core.candles import CandleStore
from src.core.indicators import Timeframe
from src.db import Base
from src.markets.nse.market_data.history_manager import HistoryManager
from src.markets.nse.persistence import bind_candle_repository


def _frame(start: datetime, rows: int, frequency: str = "15min") -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    if start_timestamp.tzinfo is None:
        start_timestamp = start_timestamp.tz_localize("Asia/Kolkata")
    else:
        start_timestamp = start_timestamp.tz_convert("Asia/Kolkata")
    index = pd.date_range(start_timestamp, periods=rows, freq=frequency)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000 + i for i in range(rows)],
        },
        index=index,
    )


def test_candle_store_upsert_is_idempotent_and_loads_newest_rows(tmp_path):
    store = CandleStore(f"sqlite:///{tmp_path / 'candles.db'}")
    frame = _frame(datetime(2026, 1, 2, 9, 15), 12)

    assert store.upsert_frame("reliance", "15m", frame) == 12
    revised = frame.copy()
    revised.iloc[-1, revised.columns.get_loc("close")] = 999.0
    assert store.upsert_frame("RELIANCE", "15m", revised) == 12

    coverage = store.coverage("RELIANCE", "15m")
    assert coverage.rows == 12
    loaded = store.load_frame("RELIANCE", "15m", bars=5)
    assert loaded is not None
    assert len(loaded) == 5
    assert loaded.index.is_monotonic_increasing
    assert str(loaded.index.tz) == "Asia/Kolkata"
    assert loaded.iloc[-1]["close"] == 999.0


def test_candle_table_is_not_registered_on_shared_trading_database_metadata():
    assert "market_candles" not in Base.metadata.tables


def test_history_manager_reads_deep_ml_window_from_store_without_api(tmp_path):
    store = CandleStore(f"sqlite:///{tmp_path / 'history.db'}")
    frame = _frame(datetime.now(UTC) - timedelta(days=30), 300)
    store.upsert_frame("TCS", "15m", frame)
    manager = HistoryManager(symbols=["TCS"], candle_store=store)

    loaded = manager.get_multi_timeframe_history("TCS", Timeframe.M15, bars=250)

    assert loaded is not None
    assert len(loaded) == 250
    assert loaded.iloc[-1]["close"] == frame.iloc[-1]["close"]


def test_candle_store_load_analysis_frame_reuses_nse_aligned_derivation(tmp_path):
    store = CandleStore(f"sqlite:///{tmp_path / 'derived.db'}")
    frame = _frame(datetime(2026, 1, 2, 9, 15), 6)
    store.upsert_frame("RELIANCE", "15m", frame)

    derived = bind_candle_repository(store).load_analysis_frame("RELIANCE", "30m", bars=3)

    assert len(derived) == 3
    assert derived.index[0].hour == 9 and derived.index[0].minute == 15
    assert derived.iloc[0]["open"] == frame.iloc[0]["open"]
    assert derived.iloc[0]["close"] == frame.iloc[1]["close"]
    assert derived.iloc[0]["volume"] == frame.iloc[0]["volume"] + frame.iloc[1]["volume"]
