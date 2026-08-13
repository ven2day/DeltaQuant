"""State for the settled-candle runtime.

This module is deliberately small: it does not replace ``SignalEngine`` or the
LangGraph desk.  It is the state boundary that lets the existing engines run once
per settled candle instead of once per timer wake-up.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from src.core.indicators import Timeframe


class HistoryReadiness(str, Enum):
    READY = "ready"
    BACKFILLING = "backfilling"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"
    STALE = "stale"


@dataclass(frozen=True)
class SettledCandleEvent:
    symbol: str
    timeframe: Timeframe
    candle_timestamp: pd.Timestamp
    frame: pd.DataFrame


@dataclass
class TimeframeState:
    readiness: HistoryReadiness = HistoryReadiness.INSUFFICIENT_DATA
    latest_settled_candle: pd.Timestamp | None = None
    last_processed_settled_candle: pd.Timestamp | None = None
    frame: pd.DataFrame | None = None
    features: Any | None = None
    feature_version: str = ""
    updated_at: datetime | None = None


@dataclass
class SymbolState:
    latest_quote: Any | None = None
    timeframes: dict[Timeframe, TimeframeState] = field(default_factory=dict)
    latest_ml_prediction: dict[tuple[str, str], Any] = field(default_factory=dict)
    strategy_state: dict[str, Any] = field(default_factory=dict)
    active_candidate: Any | None = None
    open_position_references: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CandleWakeMetrics:
    symbols_considered: int
    timeframe_states_considered: int
    changed_events: int
    skipped_unchanged: int
    skipped_not_ready: int


class MarketStateStore:
    """Thread-safe in-process view of the latest material market state."""

    def __init__(self) -> None:
        self._states: dict[str, SymbolState] = {}
        self._lock = threading.RLock()

    def _symbol(self, symbol: str) -> SymbolState:
        return self._states.setdefault(symbol.upper(), SymbolState())

    def update_quote(self, symbol: str, quote: Any) -> None:
        with self._lock:
            self._symbol(symbol).latest_quote = quote

    def set_readiness(
        self, symbol: str, timeframe: Timeframe, readiness: HistoryReadiness
    ) -> None:
        with self._lock:
            state = self._symbol(symbol).timeframes.setdefault(timeframe, TimeframeState())
            state.readiness = readiness
            state.updated_at = datetime.now()

    def observe_settled_frame(
        self,
        symbol: str,
        timeframe: Timeframe,
        frame: pd.DataFrame | None,
        *,
        minimum_bars: int = 26,
    ) -> SettledCandleEvent | None:
        """Record a settled frame and return an event only for a new candle.

        ``last_processed_settled_candle`` is advanced separately after successful
        processing.  A failed feature/strategy evaluation is therefore retried rather
        than silently acknowledged.
        """
        with self._lock:
            tf_state = self._symbol(symbol).timeframes.setdefault(timeframe, TimeframeState())
            if tf_state.readiness is HistoryReadiness.BACKFILLING:
                tf_state.updated_at = datetime.now()
                return None
            if frame is None or frame.empty or len(frame) < minimum_bars:
                tf_state.readiness = HistoryReadiness.INSUFFICIENT_DATA
                tf_state.updated_at = datetime.now()
                return None
            timestamp = pd.Timestamp(frame.index[-1])
            tf_state.readiness = HistoryReadiness.READY
            tf_state.latest_settled_candle = timestamp
            tf_state.frame = frame
            tf_state.updated_at = datetime.now()
            if tf_state.last_processed_settled_candle == timestamp:
                return None
            return SettledCandleEvent(symbol.upper(), timeframe, timestamp, frame)

    def mark_processed(
        self,
        event: SettledCandleEvent,
        *,
        features: Any | None = None,
        feature_version: str = "",
    ) -> None:
        with self._lock:
            tf_state = self._symbol(event.symbol).timeframes.setdefault(
                event.timeframe, TimeframeState()
            )
            tf_state.last_processed_settled_candle = event.candle_timestamp
            tf_state.latest_settled_candle = event.candle_timestamp
            tf_state.frame = event.frame
            tf_state.features = features
            tf_state.feature_version = feature_version
            tf_state.readiness = HistoryReadiness.READY
            tf_state.updated_at = datetime.now()

    def cached_frame(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame | None:
        with self._lock:
            state = self._states.get(symbol.upper())
            tf_state = state.timeframes.get(timeframe) if state else None
            return tf_state.frame if tf_state else None

    def cached_features(self, symbol: str, timeframe: Timeframe) -> Any | None:
        with self._lock:
            state = self._states.get(symbol.upper())
            tf_state = state.timeframes.get(timeframe) if state else None
            return tf_state.features if tf_state else None

    def timeframe_state(self, symbol: str, timeframe: Timeframe) -> TimeframeState:
        with self._lock:
            state = self._states.get(symbol.upper())
            tf_state = state.timeframes.get(timeframe) if state else None
            if tf_state is None:
                return TimeframeState()
            return TimeframeState(
                readiness=tf_state.readiness,
                latest_settled_candle=tf_state.latest_settled_candle,
                last_processed_settled_candle=tf_state.last_processed_settled_candle,
                frame=tf_state.frame,
                features=tf_state.features,
                feature_version=tf_state.feature_version,
                updated_at=tf_state.updated_at,
            )

    def cached_context(self, symbol: str) -> dict[Timeframe, Any]:
        """Return existing higher/lower timeframe features without recomputation."""
        with self._lock:
            state = self._states.get(symbol.upper())
            if state is None:
                return {}
            return {
                timeframe: tf_state.features
                for timeframe, tf_state in state.timeframes.items()
                if tf_state.readiness is HistoryReadiness.READY and tf_state.features is not None
            }

    def remember_prediction(
        self, symbol: str, timeframe: Timeframe, trade_horizon: str, prediction: Any
    ) -> None:
        with self._lock:
            self._symbol(symbol).latest_ml_prediction[
                (timeframe.value, trade_horizon.upper())
            ] = prediction

    def plan_events(
        self,
        frames: dict[tuple[str, Timeframe], pd.DataFrame | None],
        *,
        minimum_bars: int = 26,
    ) -> tuple[list[SettledCandleEvent], CandleWakeMetrics]:
        events: list[SettledCandleEvent] = []
        skipped_not_ready = 0
        symbols = {symbol.upper() for symbol, _ in frames}
        for (symbol, timeframe), frame in frames.items():
            event = self.observe_settled_frame(
                symbol, timeframe, frame, minimum_bars=minimum_bars
            )
            if event is not None:
                events.append(event)
            elif frame is None or frame.empty or len(frame) < minimum_bars:
                skipped_not_ready += 1
        considered = len(frames)
        return events, CandleWakeMetrics(
            symbols_considered=len(symbols),
            timeframe_states_considered=considered,
            changed_events=len(events),
            skipped_unchanged=max(0, considered - len(events) - skipped_not_ready),
            skipped_not_ready=skipped_not_ready,
        )

    def plan_timestamps(
        self,
        timestamps: dict[tuple[str, Timeframe], pd.Timestamp | None],
    ) -> tuple[list[tuple[str, Timeframe]], CandleWakeMetrics]:
        """Cheap event plan used before loading any historical frame."""
        changed: list[tuple[str, Timeframe]] = []
        not_ready = 0
        with self._lock:
            for (symbol, timeframe), timestamp in timestamps.items():
                tf_state = self._symbol(symbol).timeframes.setdefault(
                    timeframe, TimeframeState()
                )
                if tf_state.readiness is HistoryReadiness.BACKFILLING:
                    not_ready += 1
                    continue
                if timestamp is None:
                    if tf_state.readiness is not HistoryReadiness.FAILED:
                        tf_state.readiness = HistoryReadiness.INSUFFICIENT_DATA
                    not_ready += 1
                    continue
                normalized = pd.Timestamp(timestamp)
                tf_state.latest_settled_candle = normalized
                if tf_state.last_processed_settled_candle != normalized:
                    changed.append((symbol.upper(), timeframe))
        considered = len(timestamps)
        return changed, CandleWakeMetrics(
            symbols_considered=len({symbol.upper() for symbol, _ in timestamps}),
            timeframe_states_considered=considered,
            changed_events=len(changed),
            skipped_unchanged=max(0, considered - len(changed) - not_ready),
            skipped_not_ready=not_ready,
        )
