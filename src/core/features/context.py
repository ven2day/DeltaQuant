"""Timestamp-safe market-relative context shared across strategy snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

import pandas as pd

from src.core.indicators import IndicatorResult


@dataclass(frozen=True)
class MarketRelativeFeatures:
    """One aligned cross-sectional observation for one symbol and settled candle."""

    benchmark_return: float
    symbol_return: float
    excess_return: float
    cross_sectional_percentile: float
    cross_sectional_count: int
    timestamp: str
    source: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "benchmark_return": self.benchmark_return,
            "symbol_return": self.symbol_return,
            "excess_return": self.excess_return,
            "cross_sectional_percentile": self.cross_sectional_percentile,
            "cross_sectional_count": self.cross_sectional_count,
            "timestamp": self.timestamp,
            "source": self.source,
        }


def _required_roc(indicator: IndicatorResult) -> float:
    """Return a ROC already screened as available by the context builder."""

    if indicator.roc is None:
        raise ValueError("market-relative context requires a settled ROC value")
    return indicator.roc


def build_market_relative_context(
    indicators_by_symbol: Mapping[str, IndicatorResult],
    *,
    benchmark_symbol: str = "",
    fallback: str = "cross_sectional_median",
) -> dict[str, MarketRelativeFeatures]:
    """Build one rank set from exactly aligned settled-candle snapshots.

    Inputs with missing momentum or a timestamp different from the modal timestamp are
    excluded. This prevents cross-sectional and benchmark lookahead caused by comparing
    a fresh symbol candle with stale/future peers.
    """

    usable = {
        symbol.upper(): indicator
        for symbol, indicator in indicators_by_symbol.items()
        if indicator.roc is not None and indicator.settled_candle_timestamp
    }
    if not usable:
        return {}
    timestamps: dict[str, int] = {}
    for indicator in usable.values():
        timestamps[indicator.settled_candle_timestamp] = (
            timestamps.get(indicator.settled_candle_timestamp, 0) + 1
        )
    aligned_timestamp = max(timestamps, key=lambda value: (timestamps[value], value))
    aligned = {
        symbol: indicator
        for symbol, indicator in usable.items()
        if indicator.settled_candle_timestamp == aligned_timestamp
    }
    if len(aligned) < 3:
        return {}

    benchmark_key = benchmark_symbol.upper().strip()
    if benchmark_key and benchmark_key in aligned:
        benchmark_return = _required_roc(aligned[benchmark_key])
        source = f"benchmark:{benchmark_key}"
    elif fallback == "cross_sectional_median":
        benchmark_return = float(median(_required_roc(indicator) for indicator in aligned.values()))
        source = "cross_sectional_median_proxy"
    else:
        return {}

    ordered = sorted(
        ((_required_roc(indicator), symbol) for symbol, indicator in aligned.items()),
        key=lambda item: (item[0], item[1]),
    )
    count = len(ordered)
    percentiles = {
        symbol: (rank / (count - 1) if count > 1 else 0.5)
        for rank, (_value, symbol) in enumerate(ordered)
    }
    return {
        symbol: MarketRelativeFeatures(
            benchmark_return=benchmark_return,
            symbol_return=_required_roc(indicator),
            excess_return=_required_roc(indicator) - benchmark_return,
            cross_sectional_percentile=percentiles[symbol],
            cross_sectional_count=count,
            timestamp=aligned_timestamp,
            source=source,
        )
        for symbol, indicator in aligned.items()
    }


def build_historical_market_relative_context(
    data_by_symbol: Mapping[str, object],
    *,
    momentum_period: int = 20,
    benchmark_symbol: str = "",
    fallback: str = "cross_sectional_median",
) -> dict[str, dict[str, MarketRelativeFeatures]]:
    """Build timestamp-indexed relative context for offline walk-forward replay.

    Each return uses only the close at that timestamp and a strictly earlier close.
    Ranking groups exact timestamps, so differently aligned histories never leak across
    a bar boundary.
    """

    returns_by_symbol: dict[str, pd.Series] = {}
    for raw_symbol, raw_frame in data_by_symbol.items():
        if not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
            continue
        columns = {str(column).lower(): column for column in raw_frame.columns}
        if "close" not in columns:
            continue
        close = pd.to_numeric(raw_frame[columns["close"]], errors="coerce")
        returns_by_symbol[raw_symbol.upper()] = close / close.shift(momentum_period) - 1.0
    if not returns_by_symbol:
        return {}

    panel = pd.concat(returns_by_symbol, axis=1).sort_index()
    output: dict[str, dict[str, MarketRelativeFeatures]] = {
        symbol: {} for symbol in returns_by_symbol
    }
    benchmark_key = benchmark_symbol.upper().strip()
    for timestamp, row in panel.iterrows():
        valid = row.dropna()
        if len(valid) < 3:
            continue
        if benchmark_key and benchmark_key in valid:
            benchmark_return = float(valid[benchmark_key])
            source = f"benchmark:{benchmark_key}"
        elif fallback == "cross_sectional_median":
            benchmark_return = float(valid.median())
            source = "cross_sectional_median_proxy"
        else:
            continue
        percentiles = valid.rank(method="average", pct=True)
        timestamp_key = str(timestamp)
        for (symbol, value), percentile in zip(
            valid.items(), percentiles.to_numpy(dtype=float, copy=False), strict=True
        ):
            output[str(symbol)][timestamp_key] = MarketRelativeFeatures(
                benchmark_return=benchmark_return,
                symbol_return=float(value),
                excess_return=float(value) - benchmark_return,
                cross_sectional_percentile=float(percentile),
                cross_sectional_count=len(valid),
                timestamp=timestamp_key,
                source=source,
            )
    return output
