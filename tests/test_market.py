"""
Tests for the market module.
"""

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import (
    IndicatorResult,
    Timeframe,
    calculate_indicators,
)
from src.market.signals import (
    SignalEngine,
    SignalType,
    StrategyType,
)


class TestIndicators:
    """Tests for the indicators module."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_price = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 10000, n)

        return pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_calculate_indicators_returns_result(self, sample_df):
        """Test that indicators are calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert isinstance(result, IndicatorResult)
        assert result.symbol == "TEST"
        assert result.timeframe == Timeframe.M5

    def test_calculate_indicators_has_rsi(self, sample_df):
        """Test RSI is calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.rsi is not None
        assert 0 <= result.rsi <= 100

    def test_calculate_indicators_has_macd(self, sample_df):
        """Test MACD is calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.macd is not None
        assert result.macd_signal is not None
        assert result.macd_histogram is not None

    def test_calculate_indicators_has_bollinger(self, sample_df):
        """Test Bollinger Bands are calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.bb_upper is not None
        assert result.bb_middle is not None
        assert result.bb_lower is not None
        assert result.bb_upper > result.bb_middle > result.bb_lower

    def test_calculate_indicators_has_adx(self, sample_df):
        """Test ADX is calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.adx is not None
        assert result.plus_di is not None
        assert result.minus_di is not None

    def test_calculate_indicators_has_moving_averages(self, sample_df):
        """Test moving averages are calculated."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.sma is not None
        assert result.ema is not None
        assert 20 in result.sma
        assert 9 in result.ema

    def test_indicator_result_to_dict(self, sample_df):
        """Test IndicatorResult serialization."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)
        data = result.to_dict()

        assert isinstance(data, dict)
        assert "symbol" in data
        assert "momentum" in data
        assert "trend" in data
        assert "volatility" in data

    def test_calculate_indicators_has_cci(self, sample_df):
        """CCI(14) -- used by the EMA-CCI strategy."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.cci is not None

    def test_calculate_indicators_has_psar(self, sample_df):
        """Parabolic SAR + its price-relative direction -- used by the EMA-PSAR strategy."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.psar is not None
        assert result.psar_bullish is not None
        assert result.psar_bullish == (result.psar < result.close)

    def test_calculate_indicators_has_heiken_ashi_flags(self, sample_df):
        """Current + previous Heiken Ashi candle color -- used by the
        EMA-Heiken-Ashi-RSI strategy to detect a color flip."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert result.ha_bullish is not None
        assert result.ha_prev_bullish is not None

    def test_calculate_indicators_ema_includes_20_40_200(self, sample_df):
        """EMA-PSAR needs 20/40; EMA-CCI needs 200 -- added to the default period list
        without disturbing the original 9/21/55."""
        result = calculate_indicators(sample_df, "TEST", Timeframe.M5)

        assert {9, 20, 21, 40, 55}.issubset(result.ema.keys())
        # sample_df only has 100 bars; EMA200 needs >= 200 bars to appear at all.
        assert 200 not in result.ema


class TestSignalEngine:
    """Tests for the signal engine."""

    @pytest.fixture
    def sample_indicators(self):
        """Create sample indicator result."""
        return IndicatorResult(
            symbol="RELIANCE",
            timeframe=Timeframe.M5,
            open=2480.0,
            high=2510.0,
            low=2475.0,
            close=2500.0,
            volume=1000000,
            sma={20: 2450.0, 50: 2400.0, 200: 2300.0},
            ema={9: 2490.0, 21: 2470.0, 55: 2430.0},
            rsi=55.0,
            stoch_k=65.0,
            stoch_d=60.0,
            macd=15.0,
            macd_signal=10.0,
            macd_histogram=5.0,
            adx=30.0,
            plus_di=28.0,
            minus_di=18.0,
            atr=25.0,
            bb_upper=2530.0,
            bb_middle=2480.0,
            bb_lower=2430.0,
            bb_percent=0.7,
            vwap=2485.0,
        )

    def test_signal_engine_generates_signals(self, sample_indicators):
        """Test signal generation."""
        engine = SignalEngine()
        signals = engine.generate_signals(sample_indicators)

        # Should generate at least some signals with bullish indicators
        assert isinstance(signals, list)

    def test_signal_has_required_fields(self, sample_indicators):
        """Test signal contains required fields."""
        engine = SignalEngine()

        # Force a trend following signal with strong indicators
        sample_indicators.adx = 35.0
        sample_indicators.plus_di = 30.0
        sample_indicators.minus_di = 15.0

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.TREND_FOLLOWING]
        )

        if signals:
            signal = signals[0]
            assert signal.signal_id is not None
            assert signal.symbol == "RELIANCE"
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.target_price > 0
            assert 0 <= signal.confidence <= 1

    def test_signal_to_dict(self, sample_indicators):
        """Test signal serialization."""
        engine = SignalEngine()
        sample_indicators.adx = 35.0
        sample_indicators.plus_di = 30.0
        sample_indicators.minus_di = 15.0

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.TREND_FOLLOWING]
        )

        if signals:
            data = signals[0].to_dict()
            assert isinstance(data, dict)
            assert "signal_id" in data
            assert "entry_price" in data
            assert "risk_reward_ratio" in data

    def test_mean_reversion_at_lower_bb(self, sample_indicators):
        """Test mean reversion signal at lower Bollinger Band."""
        engine = SignalEngine()

        # Price at lower BB with oversold RSI
        sample_indicators.close = 2425.0  # Below lower BB
        sample_indicators.rsi = 25.0  # Oversold

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.MEAN_REVERSION]
        )

        # Should generate a BUY signal
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0

    # --- EMA-Heiken-Ashi-RSI (Traderversity.com guide, adapted) ---

    def test_ema_heiken_ashi_rsi_buy_on_bullish_flip_in_uptrend(self, sample_indicators):
        """ADX trending, price > EMA21, RSI > 50, HA just flipped red -> green: BUY."""
        engine = SignalEngine()
        # Fixture defaults already give adx=30 (> threshold 25), close=2500 > ema21=2470,
        # rsi=55 (> 50) -- only the HA flip needs setting.
        sample_indicators.ha_prev_bullish = False
        sample_indicators.ha_bullish = True

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_HEIKEN_ASHI_RSI]
        )

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) == 1
        assert buy_signals[0].strategy == StrategyType.EMA_HEIKEN_ASHI_RSI

    def test_ema_heiken_ashi_rsi_no_trade_in_sideways_market(self, sample_indicators):
        """The source strategy's #1 rule: zero entries when ADX says the market is
        sideways, even if every other condition (RSI, HA flip) lines up."""
        engine = SignalEngine()
        sample_indicators.adx = 15.0  # below adx_trend_threshold (25)
        sample_indicators.ha_prev_bullish = False
        sample_indicators.ha_bullish = True

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_HEIKEN_ASHI_RSI]
        )

        assert signals == []

    def test_ema_heiken_ashi_rsi_no_trade_without_flip(self, sample_indicators):
        """Trend + momentum conditions met, but Heiken Ashi hasn't just flipped
        (already green last bar too) -- no fresh entry trigger."""
        engine = SignalEngine()
        sample_indicators.ha_prev_bullish = True
        sample_indicators.ha_bullish = True

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_HEIKEN_ASHI_RSI]
        )

        assert signals == []

    # --- EMA-Parabolic SAR (Traderversity.com guide, adapted) ---

    def test_ema_psar_buy_on_bullish_cross_and_dot_below_price(self, sample_indicators):
        """20 EMA above 40 EMA + PSAR dot below price: BUY."""
        engine = SignalEngine()
        sample_indicators.ema[20] = 2495.0
        sample_indicators.ema[40] = 2460.0
        sample_indicators.psar = 2480.0  # below close (2500) -> bullish
        sample_indicators.psar_bullish = True

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_PSAR]
        )

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) == 1
        assert buy_signals[0].strategy == StrategyType.EMA_PSAR

    def test_ema_psar_no_signal_when_ema_and_psar_disagree(self, sample_indicators):
        """EMA cross says uptrend but the PSAR dot is still above price (not yet
        confirmed) -- no signal."""
        engine = SignalEngine()
        sample_indicators.ema[20] = 2495.0
        sample_indicators.ema[40] = 2460.0
        sample_indicators.psar = 2520.0  # above close -> bearish
        sample_indicators.psar_bullish = False

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_PSAR]
        )

        assert signals == []

    # --- EMA-CCI (TraderVersity-EMACCI.tpl: 200 EMA + CCI(14), +-100 levels) ---

    def test_ema_cci_buy_above_ema200_with_bullish_momentum_breakout(self, sample_indicators):
        """Price above EMA200 + CCI above +100: BUY."""
        engine = SignalEngine()
        sample_indicators.ema[200] = 2300.0  # close (2500) is above it
        sample_indicators.cci = 120.0

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_CCI]
        )

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) == 1
        assert buy_signals[0].strategy == StrategyType.EMA_CCI

    def test_ema_cci_sell_below_ema200_with_bearish_momentum_breakout(self, sample_indicators):
        """Price below EMA200 + CCI below -100: SELL."""
        engine = SignalEngine()
        sample_indicators.close = 2200.0
        sample_indicators.ema[200] = 2300.0  # close is now below it
        sample_indicators.cci = -120.0

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_CCI]
        )

        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) == 1
        assert sell_signals[0].strategy == StrategyType.EMA_CCI

    def test_ema_cci_no_signal_inside_the_100_band(self, sample_indicators):
        """Price above EMA200 but CCI hasn't broken +100 yet -- no momentum
        confirmation, no signal."""
        engine = SignalEngine()
        sample_indicators.ema[200] = 2300.0
        sample_indicators.cci = 40.0

        signals = engine.generate_signals(
            sample_indicators, active_strategies=[StrategyType.EMA_CCI]
        )

        assert signals == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
