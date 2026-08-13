from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    Strategy,
    Trade,
    _annualization_factor,
)
from src.backtesting.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RSIStrategy,
    SMACrossoverStrategy,
)
from src.core.indicators import Timeframe

# --- BacktestEngine Tests ---


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": np.linspace(100, 200, 100),  # Uptrend
            "Volume": 1000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def engine():
    return BacktestEngine(initial_capital=100000.0)


class SimpleStrategy(Strategy):
    name = "Simple"

    def on_bar(self, row, history):
        # Buy on first bar, sell on last
        if len(history) == 20:  # Start after lookback
            return "BUY"
        if len(history) == 99:
            return "SELL"
        return None


def test_backtest_run(engine, sample_data):
    strategy = SimpleStrategy()
    result = engine.run(strategy, sample_data, "TEST")

    assert isinstance(result, BacktestResult)
    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.total_return > 0
    assert len(result.trades) == 1
    assert result.trades[0].pnl > 0


def test_backtest_no_data(engine):
    with pytest.raises(ValueError):
        engine.run(SimpleStrategy(), pd.DataFrame())


def test_fetch_data(engine):
    with patch("src.backtesting.engine.HistoricalDataFeed") as MockFeed:
        mock_feed = MockFeed.return_value
        mock_feed.get_historical.return_value = pd.DataFrame()

        data = engine.fetch_data("AAPL")
        assert isinstance(data, pd.DataFrame)
        # interval defaults to "1d" -- every pre-existing caller (which never
        # requested an interval) keeps fetching daily bars exactly as before.
        mock_feed.get_historical.assert_called_with("AAPL", period="1y", interval="1d")


def test_fetch_data_passes_through_a_non_default_interval(engine):
    with patch("src.backtesting.engine.HistoricalDataFeed") as MockFeed:
        mock_feed = MockFeed.return_value
        mock_feed.get_historical.return_value = pd.DataFrame()

        engine.fetch_data("RELIANCE", period="60d", interval="5m")

        mock_feed.get_historical.assert_called_with("RELIANCE", period="60d", interval="5m")


def test_calculate_metrics(engine, sample_data):
    # Manually create trades
    trades = [
        Trade(datetime(2024, 1, 1), datetime(2024, 1, 2), "TEST", "LONG", 100, 110, 1, 10, 10),
        Trade(datetime(2024, 1, 3), datetime(2024, 1, 4), "TEST", "LONG", 100, 90, 1, -10, -10),
    ]
    equity_curve = [100, 110, 100]

    result = engine._calculate_metrics("Test", "TEST", sample_data, trades, equity_curve, 100)

    assert result.total_trades == 2
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.win_rate == 50.0
    assert result.max_drawdown > 0


# --- Sharpe annualization (Stage 14: interval-aware, was hardcoded sqrt(252)) ---


def test_annualization_factor_daily_matches_the_original_hardcoded_sqrt_252():
    """The one invariant every pre-existing caller depends on: daily data must
    produce EXACTLY the old sqrt(252) factor, not an approximation of it."""
    assert _annualization_factor("1d") == pytest.approx(np.sqrt(252))


def test_annualization_factor_unrecognized_interval_falls_back_to_daily():
    assert _annualization_factor("bogus") == pytest.approx(np.sqrt(252))
    assert _annualization_factor("") == pytest.approx(np.sqrt(252))


def test_annualization_factor_grows_for_finer_intraday_intervals():
    """More bars/day -> a larger annualization multiplier -- 5m has 3x the bars/day
    of 15m (both divide the 375-minute NSE session), so its factor must be exactly
    sqrt(3) times larger."""
    factor_5m = _annualization_factor("5m")
    factor_15m = _annualization_factor("15m")
    factor_1d = _annualization_factor("1d")

    assert factor_5m > factor_15m > factor_1d
    assert factor_5m == pytest.approx(factor_15m * np.sqrt(3))


def test_calculate_metrics_uses_the_interval_aware_sharpe(engine, sample_data):
    """Regression guard for the actual bug this stage fixes: feeding 5m bars through
    the OLD unconditional sqrt(252) would silently overstate Sharpe. With the same
    trades/equity curve, the reported Sharpe must scale by the annualization ratio
    between two different intervals."""
    trades = [
        Trade(datetime(2024, 1, 1), datetime(2024, 1, 2), "TEST", "LONG", 100, 110, 1, 10, 10),
        Trade(datetime(2024, 1, 3), datetime(2024, 1, 4), "TEST", "LONG", 100, 90, 1, -10, -10),
    ]
    equity_curve = [100.0, 110.0, 95.0, 105.0, 98.0]

    daily_result = engine._calculate_metrics(
        "Test", "TEST", sample_data, trades, equity_curve, 100, interval="1d"
    )
    scalp_result = engine._calculate_metrics(
        "Test", "TEST", sample_data, trades, equity_curve, 100, interval="5m"
    )

    ratio = scalp_result.sharpe_ratio / daily_result.sharpe_ratio
    expected_ratio = _annualization_factor("5m") / _annualization_factor("1d")
    assert ratio == pytest.approx(expected_ratio)


def test_run_threads_interval_through_to_metrics(engine, sample_data):
    strategy = SimpleStrategy()

    daily_result = engine.run(strategy, sample_data, "TEST", interval="1d")
    scalp_result = engine.run(SimpleStrategy(), sample_data, "TEST", interval="5m")

    # Same trades/equity curve (same strategy, same data) -- only the Sharpe
    # annualization should differ between the two calls.
    assert daily_result.total_trades == scalp_result.total_trades
    assert daily_result.sharpe_ratio != scalp_result.sharpe_ratio


# --- RealSignalStrategy timeframe (Stage 14: was hardcoded Timeframe.D1) ---


def test_real_signal_strategy_defaults_to_daily_timeframe():
    from src.backtesting.strategies import RealSignalStrategy

    strategy = RealSignalStrategy(symbol="TEST")
    assert strategy.timeframe == Timeframe.D1


def test_real_signal_strategy_accepts_an_explicit_timeframe():
    from src.backtesting.strategies import RealSignalStrategy

    strategy = RealSignalStrategy(symbol="TEST", timeframe=Timeframe.M5)
    assert strategy.timeframe == Timeframe.M5


def test_real_signal_strategy_passes_its_timeframe_to_calculate_indicators():
    from src.backtesting.strategies import RealSignalStrategy

    dates = pd.date_range("2024-01-01", periods=60, freq="5min")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000},
        index=dates,
    )
    strategy = RealSignalStrategy(symbol="TEST", timeframe=Timeframe.M5, min_bars=10)

    with patch("src.backtesting.strategies.calculate_indicators") as mock_calc:
        mock_calc.return_value = MagicMock(indicators={})
        strategy._engine.generate_signals = lambda *a, **k: []
        strategy.on_bar(history.iloc[-1], history.iloc[:-1])

    assert mock_calc.call_args.kwargs["timeframe"] == Timeframe.M5


def test_real_signal_strategy_reuses_shared_offline_indicator_cache():
    from src.backtesting.strategies import RealSignalStrategy

    dates = pd.date_range("2024-01-01", periods=60, freq="5min")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000},
        index=dates,
    )
    shared_cache = {}
    first = RealSignalStrategy(
        symbol="TEST", timeframe=Timeframe.M5, min_bars=10, indicator_cache=shared_cache
    )
    second = RealSignalStrategy(
        symbol="TEST", timeframe=Timeframe.M5, min_bars=10, indicator_cache=shared_cache
    )

    with patch("src.backtesting.strategies.calculate_indicators") as mock_calc:
        mock_calc.return_value = MagicMock(indicators={})
        first._engine.generate_signals = lambda *a, **k: []
        second._engine.generate_signals = lambda *a, **k: []
        prior = history.iloc[:-1]
        first.on_bar(history.iloc[-1], prior)
        second.on_bar(history.iloc[-1], prior)

    assert mock_calc.call_count == 1
    assert len(shared_cache) == 1


# --- Strategies Tests ---


@pytest.fixture
def strategy_data():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {"Close": np.concatenate([np.linspace(100, 150, 25), np.linspace(150, 100, 25)])},
        index=dates,
    )
    return df


def test_momentum_strategy(strategy_data):
    strat = MomentumStrategy(sma_period=10)

    # Run through data
    for i in range(10, len(strategy_data)):
        row = strategy_data.iloc[i]
        history = strategy_data.iloc[:i]
        strat.on_bar(row, history)
        # Just check it runs without error, logic verification is complex on synthetic data


def test_mean_reversion_strategy(strategy_data):
    strat = MeanReversionStrategy(bb_period=10)
    for i in range(10, len(strategy_data)):
        row = strategy_data.iloc[i]
        history = strategy_data.iloc[:i]
        strat.on_bar(row, history)


def test_sma_crossover_strategy():
    # Use synthetic data that guarantees a crossover
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    # First 25 days flat 100, then jump to 200
    # Fast SMA will react faster than Slow SMA -> Golden Cross
    prices = np.concatenate([np.ones(25) * 100, np.ones(25) * 200])
    df = pd.DataFrame({"Close": prices}, index=dates)

    strat = SMACrossoverStrategy(fast_period=5, slow_period=10)

    signals = []
    for i in range(10, len(df)):
        row = df.iloc[i]
        history = df.iloc[:i]
        sig = strat.on_bar(row, history)
        if sig:
            signals.append(sig)

    assert len(signals) > 0  # Should have at least buy signal


def test_rsi_strategy(strategy_data):
    strat = RSIStrategy(rsi_period=10)
    for i in range(15, len(strategy_data)):
        row = strategy_data.iloc[i]
        history = strategy_data.iloc[:i]
        strat.on_bar(row, history)


def test_strategy_abstract():
    with pytest.raises(NotImplementedError):
        Strategy().on_bar(None, None)
