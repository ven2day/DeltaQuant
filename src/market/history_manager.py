"""
Historical Data Manager

Pre-fetches, caches, and maintains rolling OHLCV DataFrames for each symbol.
Enables real indicator calculation from actual price history instead of
fabricating indicators from a single price point.

Features:
- Pre-fetch historical data on startup via DhanHQ
- Append new quotes as intraday candles
- Configurable lookback and cache TTL
- Thread-safe data access
"""

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.market.historical_feed import HistoricalDataFeed, HistoricalFeed
from src.market.indicators import Timeframe
from src.utils.market_time import is_market_hours, now_ist

logger = logging.getLogger(__name__)

# Minimum bars needed for indicator calculation
MIN_BARS_FOR_INDICATORS = 50
DEFAULT_LOOKBACK_PERIOD = "3mo"  # 3 months of daily data
MAX_INTRADAY_BARS = 500  # Max intraday bars to keep in memory

# Feed interval/lookback pairs. HistoryManager fetches M15 and H1 as base frames,
# then derives M30 and H4. The one-off fetch helper can still request native M30.
INTRADAY_YF_PARAMS: dict[Timeframe, tuple[str, str]] = {
    Timeframe.M5: ("5m", "60d"),
    Timeframe.M15: ("15m", "60d"),
    # One-off callers may request native M30; HistoryManager derives it from M15.
    Timeframe.M30: ("30m", "60d"),
    Timeframe.H1: ("60m", "730d"),
}

# Minimum seconds between re-fetches of each intraday timeframe, so a signal cycle that
# runs every few seconds doesn't hammer Yahoo with a fresh request per symbol per cycle.
INTRADAY_REFRESH_SECONDS: dict[Timeframe, int] = {
    Timeframe.M5: 5 * 60,
    Timeframe.M15: 5 * 60,
    Timeframe.M30: 10 * 60,
    Timeframe.H1: 20 * 60,
    Timeframe.H4: 40 * 60,
}

_OHLCV_RESAMPLE_RULE: dict[Timeframe, str] = {
    Timeframe.M30: "30min",
    Timeframe.H4: "4h",
}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV-indexed DataFrame to a coarser candle size."""
    return df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["open", "high", "low", "close"])


def normalize_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize a raw yfinance frame to the lower-case OHLCV contract used throughout."""
    if df is None or df.empty:
        return None
    normalized = df.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(normalized.columns):
        logger.warning("Skipping data with missing OHLCV columns: %s", normalized.columns.tolist())
        return None
    normalized = normalized[required].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=required).sort_index()
    return normalized if not normalized.empty else None


def fetch_timeframe_history(
    feed: HistoricalFeed, symbol: str, timeframe: Timeframe
) -> pd.DataFrame | None:
    """Fetch OHLCV at a specific candle timeframe directly from a feed (no caching/state).

    Shared by anything that needs a one-off multi-timeframe pull (the backfill script,
    the scalping screener) — the same intervals/H4 resampling ``HistoryManager`` itself
    uses, just without its per-symbol TTL cache, which is tuned for a small actively
    traded set re-read every cycle rather than a one-off bulk scan.
    """
    if timeframe == Timeframe.D1:
        return normalize_ohlcv(feed.get_historical(symbol, period="3mo", interval="1d"))

    if timeframe == Timeframe.H4:
        hourly = normalize_ohlcv(feed.get_historical(symbol, period="730d", interval="60m"))
        return _resample_ohlcv(hourly, "4h") if hourly is not None else None

    params = INTRADAY_YF_PARAMS.get(timeframe)
    if params is None:
        raise ValueError(f"Unsupported timeframe: {timeframe.value}")
    interval, period = params
    return normalize_ohlcv(feed.get_historical(symbol, period=period, interval=interval))


class HistoryManager:
    """
    Manages rolling historical DataFrames for each symbol.

    Pre-fetches daily OHLCV data on startup and merges new intraday
    quotes to maintain a continuously updated price history suitable
    for real indicator calculation.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback_period: str = DEFAULT_LOOKBACK_PERIOD,
        allow_synthetic: bool = False,
        simulated_stream: Any | None = None,
    ) -> None:
        """
        Initialize the history manager.

        Args:
            symbols: List of stock symbols to track
            lookback_period: YFinance period string for initial data fetch
            allow_synthetic: Seed demo OHLCV when the configured history feed is unavailable
        """
        self.symbols = symbols or []
        self.lookback_period = lookback_period
        self.allow_synthetic = allow_synthetic
        self._simulated_stream = simulated_stream

        # Thread-safe storage for historical DataFrames
        self._lock = threading.Lock()
        self._history: dict[str, pd.DataFrame] = {}
        self._last_fetch: dict[str, datetime] = {}
        self._feed = HistoricalDataFeed(symbols=self.symbols)
        self._history_cache_dir = Path("data/history_cache")

        # Intraday multi-timeframe cache: keyed by (symbol, timeframe). Separate from
        # `_history` (which holds the daily bars built up from live quotes) because these
        # are fetched directly from Yahoo at their real interval, not aggregated ticks.
        self._intraday_history: dict[tuple[str, Timeframe], pd.DataFrame] = {}
        self._intraday_last_fetch: dict[tuple[str, Timeframe], datetime] = {}

    def prefetch_all(self) -> dict[str, bool]:
        """
        Pre-fetch historical data for all symbols.

        Returns:
            Dict of symbol -> success status
        """
        results = {}
        logger.info(f"Pre-fetching historical data for {len(self.symbols)} symbols...")

        if self.allow_synthetic and self._simulated_stream is not None:
            for symbol in self.symbols:
                self._seed_synthetic(symbol)
                results[symbol] = symbol in self._history
            logger.warning(
                "TESTING ONLY: loaded coherent simulated history for %d symbols",
                sum(1 for loaded in results.values() if loaded),
            )
            return results

        if not self._feed.is_available:
            for symbol in self.symbols:
                if self.allow_synthetic:
                    self._seed_synthetic(symbol)
                    results[symbol] = True
                else:
                    results[symbol] = False
            if self.allow_synthetic:
                logger.warning(
                    "TESTING ONLY: seeded synthetic daily history for %d symbols",
                    len(self.symbols),
                )
            else:
                logger.error(
                    "Pre-fetch skipped: DhanHQ history is unavailable; no synthetic history will be used"
                )
            return results

        for symbol in self.symbols:
            success = self.fetch_history(symbol)
            if not success and self.allow_synthetic:
                self._seed_synthetic(symbol)
                success = True
            results[symbol] = success

        fetched = sum(1 for v in results.values() if v)
        logger.info(
            "Pre-fetch complete: %d/%d symbols loaded with real DhanHQ history",
            fetched,
            len(self.symbols),
        )
        return results

    def _seed_synthetic(self, symbol: str, bars: int = 180) -> None:
        """Seed from the coherent stream, or use the legacy isolated fallback in tests."""
        if self._simulated_stream is not None:
            daily = self._simulated_stream.get_history(symbol, "1d")
            five_minute = self._simulated_stream.get_history(symbol, "5m")
            if daily is None or five_minute is None:
                return
            with self._lock:
                self._history[symbol] = daily
                self._intraday_history[(symbol, Timeframe.M5)] = five_minute
                self._last_fetch[symbol] = datetime.now()
                self._intraday_last_fetch[(symbol, Timeframe.M5)] = datetime.now()
            return

        """Generate a plausible random-walk daily OHLCV history (legacy test fallback)."""
        import numpy as np

        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        base = float(rng.uniform(100, 3000))
        returns = rng.normal(0.0004, 0.015, bars)
        closes = base * np.cumprod(1 + returns)
        dates = pd.date_range(end=now_ist().date(), periods=bars, freq="D")
        df = pd.DataFrame(
            {
                "open": closes * (1 - rng.uniform(0, 0.004, bars)),
                "high": closes * (1 + rng.uniform(0, 0.008, bars)),
                "low": closes * (1 - rng.uniform(0, 0.008, bars)),
                "close": closes,
                "volume": rng.integers(100_000, 5_000_000, bars),
            },
            index=dates,
        )
        with self._lock:
            self._history[symbol] = df
            self._last_fetch[symbol] = datetime.now()

    def fetch_history(self, symbol: str, period: str | None = None) -> bool:
        """
        Fetch historical data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            period: YFinance period string (default: lookback_period)

        Returns:
            True if data was fetched successfully
        """
        period = period or self.lookback_period
        try:
            cached = self._load_cached_history(symbol)
            if cached is not None:
                with self._lock:
                    self._history[symbol] = cached
                    self._last_fetch[symbol] = datetime.now()
                logger.info("Loaded %d cached real bars for %s", len(cached), symbol)
                return True

            df = self._feed.get_historical(symbol, period=period)

            if df is None or df.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return False

            # Normalize column names to lowercase for consistency
            df.columns = [c.lower() for c in df.columns]

            # Ensure required columns exist
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                logger.warning(f"Missing columns for {symbol}: {df.columns.tolist()}")
                return False

            with self._lock:
                self._history[symbol] = df.copy()
                self._last_fetch[symbol] = datetime.now()
            self._save_cached_history(symbol, df)

            logger.info(
                f"Loaded {len(df)} bars for {symbol} "
                f"({df.index[0].date()} to {df.index[-1].date()})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to fetch history for {symbol}: {e}")
            return False

    def _history_cache_path(self, symbol: str) -> Path:
        return self._history_cache_dir / f"{symbol}.pkl"

    def _load_cached_history(self, symbol: str) -> pd.DataFrame | None:
        try:
            df = pd.read_pickle(self._history_cache_path(symbol))
            if len(df) < MIN_BARS_FOR_INDICATORS:
                return None
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                return None
            return df
        except (OSError, ValueError, TypeError, EOFError):
            return None

    def _save_cached_history(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            self._history_cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(self._history_cache_path(symbol))
        except OSError:
            logger.warning("Could not cache real historical data for %s", symbol, exc_info=True)

    def append_quote(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Append a new quote as a candle to the symbol's history.

        If the timestamp matches the last bar's date, the bar is updated
        (intraday aggregation). Otherwise, a new bar is appended.

        Args:
            symbol: Stock symbol
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            timestamp: Quote timestamp (default: now, in IST)
        """
        if timestamp is None:
            timestamp = now_ist()

        with self._lock:
            if symbol not in self._history:
                logger.debug(f"No history for {symbol}, skipping quote append")
                return

            df = self._history[symbol]

            # Check if we should update the last bar or create a new one
            if len(df) > 0:
                last_date = df.index[-1]

                # Same trading day — update the last bar
                if hasattr(last_date, "date") and last_date.date() == timestamp.date():
                    df.at[df.index[-1], "high"] = max(df.iloc[-1]["high"], high)
                    df.at[df.index[-1], "low"] = min(df.iloc[-1]["low"], low)
                    df.at[df.index[-1], "close"] = close
                    # Quote volume is the cumulative daily total, not a per-tick delta —
                    # keep the latest (monotonic) cumulative value instead of summing,
                    # which previously inflated volume without bound each cycle.
                    df.at[df.index[-1], "volume"] = max(float(df.iloc[-1]["volume"]), float(volume))
                    return

            # New day or first bar — append new row
            new_row = pd.DataFrame(
                {
                    "open": [open_price],
                    "high": [high],
                    "low": [low],
                    "close": [close],
                    "volume": [volume],
                },
                index=[pd.Timestamp(timestamp)],
            )

            self._history[symbol] = pd.concat([df, new_row])

            # Trim to max bars to prevent memory growth
            if len(self._history[symbol]) > MAX_INTRADAY_BARS:
                self._history[symbol] = self._history[symbol].iloc[-MAX_INTRADAY_BARS:]

    def get_history(
        self,
        symbol: str,
        bars: int | None = None,
        include_forming: bool = True,
    ) -> pd.DataFrame | None:
        """
        Get historical DataFrame for a symbol.

        Args:
            symbol: Stock symbol
            bars: Number of recent bars to return (None = all)
            include_forming: If False, drop the current (still-forming) bar — i.e. any
                trailing rows dated today (IST). The forming bar's OHLC repaints as live
                quotes arrive, so indicators/signals computed on it look ahead within the
                bar; pass False to compute on settled bars only. Never returns empty
                purely from this trim (falls back to the full history if everything is today).

        Returns:
            DataFrame with OHLCV columns, or None if not available
        """
        with self._lock:
            if symbol not in self._history:
                return None

            df = self._history[symbol].copy()

        if not include_forming and len(df) > 0:
            today = now_ist().date()
            settled = df[[not (hasattr(ts, "date") and ts.date() == today) for ts in df.index]]
            if len(settled) > 0:
                df = settled

        if bars is not None and len(df) > bars:
            df = df.iloc[-bars:]

        return df

    def get_multi_timeframe_history(
        self,
        symbol: str,
        timeframe: Timeframe,
        bars: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Get OHLCV history for a symbol at a specific candle timeframe.

        Daily (D1) is served from the existing quote-built history (`get_history`).
        M15 and H1 are fetched from the configured feed and cached with a TTL.
        M30 and H4 are resampled from those base frames, avoiding duplicate broker
        history requests during full-universe analysis.

        Returns None if the timeframe isn't supported or no data is available yet
        (never raises — a fetch failure just falls back to whatever is already cached).

        Re-checks market hours on every call (not just once at construction): with a
        configured simulated_stream, closed-market cycles compute indicators/signals
        off simulated data instead of a frozen real snapshot -- so the signal-
        generation pipeline is actually exercisable for pipeline testing on a closed
        weekend, the same way MarketDataManager already auto-switches the live quote
        feed. Reverts to real DhanHQ data automatically once the market reopens, no
        restart required. Chart display and already-open positions' pricing/exits are
        unaffected -- they resolve their own lineage separately (see
        run_live_trading.py's _get_candles and manager.py's get_real_quotes()/
        get_simulated_quotes()), not through this method.
        """
        if self._simulated_stream is not None and not is_market_hours():
            return self._simulated_stream.get_history(symbol, timeframe.value, bars)

        if timeframe == Timeframe.D1:
            return self.get_history(symbol, bars=bars)

        if timeframe in _OHLCV_RESAMPLE_RULE:
            base_timeframe = Timeframe.M15 if timeframe == Timeframe.M30 else Timeframe.H1
            base = self.get_multi_timeframe_history(symbol, base_timeframe)
            if base is None or base.empty:
                return None
            df = _resample_ohlcv(base, _OHLCV_RESAMPLE_RULE[timeframe])
        else:
            params = INTRADAY_YF_PARAMS.get(timeframe)
            if params is None:
                logger.warning(f"Unsupported intraday timeframe: {timeframe}")
                return None

            key = (symbol, timeframe)
            with self._lock:
                last = self._intraday_last_fetch.get(key)
            refresh_after = INTRADAY_REFRESH_SECONDS.get(timeframe, 300)
            stale = last is None or (datetime.now() - last).total_seconds() >= refresh_after

            if stale:
                interval, period = params
                try:
                    fetched = self._feed.get_historical(symbol, period=period, interval=interval)
                except Exception as e:
                    logger.warning(f"Intraday fetch failed for {symbol} [{timeframe.value}]: {e}")
                    fetched = None

                if fetched is not None and not fetched.empty:
                    fetched = fetched.copy()
                    fetched.columns = [c.lower() for c in fetched.columns]
                    with self._lock:
                        self._intraday_history[key] = fetched
                        self._intraday_last_fetch[key] = datetime.now()
                elif last is None:
                    # Never successfully fetched — nothing to fall back to.
                    return None

            with self._lock:
                cached = self._intraday_history.get(key)
            if cached is None:
                return None
            df = cached.copy()

        if bars is not None and len(df) > bars:
            df = df.iloc[-bars:]
        return df

    def sync_simulated_stream(self) -> None:
        """Refresh daily and five-minute caches from the authoritative simulated stream."""
        if self._simulated_stream is None:
            return
        for symbol in self.symbols:
            self._seed_synthetic(symbol)

    def has_sufficient_data(self, symbol: str, min_bars: int = MIN_BARS_FOR_INDICATORS) -> bool:
        """
        Check if a symbol has enough data for indicator calculation.

        Args:
            symbol: Stock symbol
            min_bars: Minimum number of bars required

        Returns:
            True if sufficient data is available
        """
        with self._lock:
            if symbol not in self._history:
                return False
            return len(self._history[symbol]) >= min_bars

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with loaded history."""
        with self._lock:
            return [s for s in self._history if len(self._history[s]) >= MIN_BARS_FOR_INDICATORS]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about loaded history data."""
        with self._lock:
            stats = {}
            for symbol, df in self._history.items():
                stats[symbol] = {
                    "bars": len(df),
                    "start": str(df.index[0].date()) if len(df) > 0 else "N/A",
                    "end": str(df.index[-1].date()) if len(df) > 0 else "N/A",
                    "sufficient": len(df) >= MIN_BARS_FOR_INDICATORS,
                }
            return stats

    def refresh_stale(self, max_age_hours: int = 12) -> int:
        """
        Re-fetch data for symbols whose cache is stale.

        Args:
            max_age_hours: Maximum age in hours before refetching

        Returns:
            Number of symbols refreshed
        """
        refreshed = 0
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for symbol in self.symbols:
            last = self._last_fetch.get(symbol)
            if last is None or last < cutoff:
                if self.fetch_history(symbol):
                    refreshed += 1

        return refreshed
