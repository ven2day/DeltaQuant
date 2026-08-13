"""Deterministic, coherent market simulation for permanent paper trading.

One five-minute event stream is the source of truth for every symbol. Quotes and all
coarser candles are projections of that same stream, so a signal, fill, chart, and exit
can never observe unrelated synthetic price scales. ``tick`` advances simulated event
time by exactly one NSE-style five-minute bar, even when wall-clock markets are closed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.markets.nse.sessions.market_time import now_ist

logger = logging.getLogger(__name__)


# Familiar prices improve the demo for common NSE names. Every other configured symbol
# receives a stable price derived from its symbol, so coverage is never limited to this map.
NSE_BASE_PRICES = {
    "RELIANCE": 2450.0,
    "TCS": 4200.0,
    "HDFCBANK": 1680.0,
    "INFY": 1850.0,
    "ICICIBANK": 1280.0,
    "HINDUNILVR": 2350.0,
    "SBIN": 780.0,
    "BHARTIARTL": 1620.0,
    "ITC": 465.0,
    "KOTAKBANK": 1820.0,
    "LT": 3650.0,
    "AXISBANK": 1150.0,
    "ASIANPAINT": 2280.0,
    "MARUTI": 11200.0,
    "TITAN": 3450.0,
    "BAJFINANCE": 7200.0,
    "WIPRO": 295.0,
    "ULTRACEMCO": 11500.0,
    "NESTLEIND": 2180.0,
    "TECHM": 1680.0,
}

_RESAMPLE_RULES = {
    "5m": None,
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

_HISTORICAL_INTERVAL_TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _previous_business_day(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _session_slots(day: date) -> list[datetime]:
    """Return 75 five-minute bar timestamps from 09:15 through 15:25."""
    start = datetime.combine(day, time(9, 15))
    return [start + timedelta(minutes=5 * index) for index in range(75)]


def _initial_index(bars: int, end_day: date) -> pd.DatetimeIndex:
    sessions: list[datetime] = []
    day = _previous_business_day(end_day)
    while len(sessions) < bars:
        if day.weekday() < 5:
            sessions[0:0] = _session_slots(day)
        day -= timedelta(days=1)
    return pd.DatetimeIndex(sessions[-bars:])


def _next_event_time(previous: datetime) -> datetime:
    if previous.time() < time(15, 25):
        return previous + timedelta(minutes=5)
    day = previous.date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime.combine(day, time(9, 15))


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule, origin="start_day", offset="15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


@dataclass
class SimulatedQuote:
    """Latest quote projected from the current five-minute simulated bar."""

    symbol: str
    security_id: int
    last_price: float
    open: float
    high: float
    low: float
    close: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime = field(default_factory=now_ist)
    sequence: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.change_percent > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "last_price": self.last_price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        }


@dataclass
class _InstrumentState:
    frame: pd.DataFrame
    rng: np.random.Generator
    beta: float
    drift: float


class SimulatedMarketData:
    """Full-universe, deterministic event-time simulator.

    A shared market shock creates realistic cross-symbol co-movement while stable
    symbol-specific random generators retain dispersion. All public data is derived from
    ``_states[symbol].frame``; there is no second quote-only random walk.
    """

    def __init__(
        self,
        volatility: float = 0.0015,
        symbols: list[str] | None = None,
        seed: int = 20260807,
        history_bars: int = 3000,
    ) -> None:
        self.volatility = volatility
        self.seed = seed
        self.symbols = list(dict.fromkeys(symbols or list(NSE_BASE_PRICES)))
        self.history_bars = history_bars
        self.sequence = 0
        self._market_rng = np.random.default_rng(seed)
        self._states: dict[str, _InstrumentState] = {}
        self._build_history()

    @property
    def event_time(self) -> datetime:
        first = next(iter(self._states.values()))
        return first.frame.index[-1].to_pydatetime()

    @property
    def current_prices(self) -> dict[str, float]:
        """Compatibility view backed by the same canonical event stream."""
        return {
            symbol: float(state.frame.iloc[-1]["close"]) for symbol, state in self._states.items()
        }

    def _base_price(self, symbol: str) -> float:
        if symbol in NSE_BASE_PRICES:
            return NSE_BASE_PRICES[symbol]
        # Stable ₹100–₹3,000 price for every configured symbol.
        return 100.0 + (_stable_int(symbol) % 290_001) / 100.0

    def _build_history(self) -> None:
        end_day = _previous_business_day(now_ist().date())
        index = _initial_index(self.history_bars, end_day)
        common_rng = np.random.default_rng(self.seed)
        common = common_rng.normal(0.000015, self.volatility * 0.55, len(index))

        for symbol in self.symbols:
            symbol_seed = (self.seed + _stable_int(symbol)) % (2**32)
            rng = np.random.default_rng(symbol_seed)
            beta = float(rng.uniform(0.75, 1.25))
            drift = float(rng.uniform(-0.000005, 0.000035))
            residual = rng.normal(0.0, self.volatility * 0.75, len(index))
            returns = np.clip(drift + beta * common + residual, -0.035, 0.035)
            closes = self._base_price(symbol) * np.cumprod(1.0 + returns)
            opens = np.concatenate(([closes[0] / (1.0 + returns[0])], closes[:-1]))
            wick = np.maximum(np.abs(returns), 0.0005) + rng.uniform(0.0002, 0.0015, len(index))
            highs = np.maximum(opens, closes) * (1.0 + wick * rng.uniform(0.2, 0.8, len(index)))
            lows = np.minimum(opens, closes) * (1.0 - wick * rng.uniform(0.2, 0.8, len(index)))
            volumes = rng.lognormal(mean=10.7, sigma=0.55, size=len(index)).astype(int)
            frame = pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                },
                index=index,
            )
            self._states[symbol] = _InstrumentState(frame, rng, beta, drift)

        logger.info(
            "Initialized coherent simulated stream: %d symbols x %d five-minute bars",
            len(self._states),
            self.history_bars,
        )

    def tick(self) -> datetime:
        """Advance one simulated five-minute event and return its event timestamp."""
        event_time = _next_event_time(self.event_time)
        market_return = float(self._market_rng.normal(0.000015, self.volatility * 0.55))
        self.sequence += 1

        for state in self._states.values():
            previous_close = float(state.frame.iloc[-1]["close"])
            residual = float(state.rng.normal(0.0, self.volatility * 0.75))
            bar_return = float(
                np.clip(state.drift + state.beta * market_return + residual, -0.035, 0.035)
            )
            close = previous_close * (1.0 + bar_return)
            wick = max(abs(bar_return), 0.0005) + float(state.rng.uniform(0.0002, 0.0015))
            high = max(previous_close, close) * (1.0 + wick * float(state.rng.uniform(0.2, 0.8)))
            low = min(previous_close, close) * (1.0 - wick * float(state.rng.uniform(0.2, 0.8)))
            volume = int(state.rng.lognormal(mean=10.7, sigma=0.55))
            state.frame.loc[pd.Timestamp(event_time)] = [previous_close, high, low, close, volume]
            if len(state.frame) > self.history_bars:
                state.frame = state.frame.iloc[-self.history_bars :]
        return event_time

    def get_history(
        self, symbol: str, timeframe: str = "5m", bars: int | None = None
    ) -> pd.DataFrame | None:
        state = self._states.get(symbol)
        if state is None:
            return None
        value = getattr(timeframe, "value", timeframe)
        rule = _RESAMPLE_RULES.get(str(value))
        if str(value) not in _RESAMPLE_RULES:
            return None
        frame = state.frame.copy() if rule is None else _resample(state.frame, rule)
        if bars is not None:
            frame = frame.iloc[-bars:]
        return frame

    def get_historical(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """Expose the canonical stream through the shared ``HistoricalFeed`` contract.

        ``period`` is intentionally ignored: the simulator owns a fixed-size rolling
        history and callers already trim it to their requested lookback. Supporting the
        feed contract lets informational scanners consume the exact same candles as the
        signal, chart, quote, and paper-fill paths instead of probing Dhan in mock mode.
        """
        del period
        timeframe = _HISTORICAL_INTERVAL_TIMEFRAMES.get(interval)
        if timeframe is None:
            logger.warning("Unsupported simulated historical interval: %s", interval)
            return None
        return self.get_history(symbol, timeframe=timeframe)

    def get_quotes(self, symbols: list[str] | None = None) -> dict[str, SimulatedQuote]:
        requested = symbols or self.symbols
        quotes: dict[str, SimulatedQuote] = {}
        for symbol in requested:
            state = self._states.get(symbol)
            if state is None:
                continue
            bar = state.frame.iloc[-1]
            session = state.frame[state.frame.index.date == state.frame.index[-1].date()]
            previous = state.frame.iloc[-2]
            previous_close = float(previous["close"])
            last_price = float(bar["close"])
            change = last_price - previous_close
            quotes[symbol] = SimulatedQuote(
                symbol=symbol,
                security_id=_stable_int(symbol) % 100_000_000,
                last_price=round(last_price, 2),
                open=round(float(session.iloc[0]["open"]), 2),
                high=round(float(session["high"].max()), 2),
                low=round(float(session["low"].min()), 2),
                close=round(previous_close, 2),
                change=round(change, 2),
                change_percent=round((change / previous_close) * 100, 2),
                volume=int(session["volume"].sum()),
                timestamp=self.event_time,
                sequence=self.sequence,
            )
        return quotes

    def get_trading_candidates(self, min_change: float = 0.5) -> list[SimulatedQuote]:
        candidates = [
            quote for quote in self.get_quotes().values() if abs(quote.change_percent) >= min_change
        ]
        return sorted(candidates, key=lambda quote: abs(quote.change_percent), reverse=True)

    def get_top_movers(self, n: int = 5) -> tuple[list[SimulatedQuote], list[SimulatedQuote]]:
        ordered = sorted(self.get_quotes().values(), key=lambda quote: quote.change_percent)
        return list(reversed(ordered[-n:])), ordered[:n]
