"""Pure OHLCV-DataFrame -> chart-candle conversion for the dashboard's Charts tab.

Split out from the ``_get_candles`` closure in ``scripts/run_live_trading.py`` (which
just supplies the ``HistoryManager`` dependency) so the conversion logic itself --
NaN handling, timestamp conversion -- is unit-testable without needing a running
trading loop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_candles(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert an OHLCV DataFrame (DatetimeIndex; open/high/low/close/volume columns,
    lowercase -- see HistoryManager) into the chart-ready shape:
    ``{time: unix_seconds, open, high, low, close, volume}``.

    Rows with a NaN/missing O/H/L/C are dropped rather than passed through as
    ``null`` -- a candle needs all four to render, and the caller (a chart) has no
    sensible way to display a partial one. ``None``/empty input returns ``[]``.
    """
    if frame is None or frame.empty:
        return []
    candles: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            o, h, lo, c = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        if any(v != v for v in (o, h, lo, c)):  # NaN != NaN; avoids importing math/numpy here
            continue
        candles.append(
            {
                "time": int(index.timestamp()),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
        )
    return candles
