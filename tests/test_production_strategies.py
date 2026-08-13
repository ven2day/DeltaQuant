from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtesting.production_strategy_registration import (
    ensure_production_strategy_placeholders,
)
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    StrategyEligibilityRegistry,
)
from src.core.aggregation import (
    FeatureSnapshot,
    aggregate_strategy_signals,
    evaluate_registered_strategies,
)
from src.core.candidates import SignalEngine, SignalType, StrategyType
from src.core.features import MarketRelativeFeatures, build_market_relative_context
from src.core.indicators import IndicatorConfig, IndicatorResult, Timeframe, calculate_indicators
from src.core.strategies import (
    DEFAULT_TIMEFRAME_MAP,
    ProductionStrategyConfig,
    evaluate_production_strategy,
)


@pytest.fixture
def config() -> ProductionStrategyConfig:
    return ProductionStrategyConfig.defaults()


def _indicator(**overrides) -> IndicatorResult:
    values = {
        "symbol": "RELIANCE",
        "timeframe": Timeframe.M15,
        "open": 104.0,
        "high": 106.0,
        "low": 103.0,
        "close": 105.0,
        "volume": 1500,
        "previous_open": 103.0,
        "previous_high": 105.0,
        "previous_low": 102.5,
        "previous_close": 104.0,
        "ema": {20: 103.0, 50: 100.0, 200: 95.0},
        "ema_previous": {20: 102.5, 50: 99.8, 200: 94.9},
        "ema_slope_atr": {20: 0.25, 50: 0.10, 200: 0.05},
        "rsi": 55.0,
        "macd": 1.0,
        "macd_signal": 0.8,
        "macd_histogram": 0.2,
        "macd_previous": 0.7,
        "macd_signal_previous": 0.75,
        "macd_histogram_previous": -0.05,
        "adx": 30.0,
        "plus_di": 35.0,
        "minus_di": 15.0,
        "atr": 2.0,
        "bb_upper": 107.0,
        "bb_middle": 103.0,
        "bb_lower": 99.0,
        "bb_percent": 0.75,
        "bb_bandwidth": 0.07,
        "bb_previous_upper": 106.5,
        "bb_previous_middle": 102.5,
        "bb_previous_lower": 98.5,
        "bb_previous_percent": 0.7,
        "vwap": 103.0,
        "session_vwap": 103.0,
        "vwap_distance": 2.0,
        "vwap_distance_atr": 1.0,
        "vwap_zscore": 1.3,
        "donchian_previous_high": 104.5,
        "donchian_previous_low": 98.0,
        "average_volume": 1000.0,
        "relative_volume": 1.5,
        "volume_zscore": 1.2,
        "returns": {5: 0.01, 10: 0.02, 20: 0.03},
        "roc": 0.03,
        "atr_normalized_momentum": 2.0,
        "rolling_high": 106.0,
        "rolling_low": 98.0,
        "supertrend": 101.0,
        "supertrend_bullish": True,
        "supertrend_previous": 100.5,
        "supertrend_previous_bullish": True,
        "opening_range_high": 104.5,
        "opening_range_low": 100.0,
        "opening_range_end": "2026-08-11T09:45:00+05:30",
        "opening_range_complete": True,
        "candle_body_pct": 0.5,
        "upper_wick_pct": 0.25,
        "lower_wick_pct": 0.25,
        "settled_candle_timestamp": "2026-08-11 10:15:00+05:30",
    }
    values.update(overrides)
    return IndicatorResult(**values)


def _context(**overrides) -> MarketRelativeFeatures:
    values = {
        "benchmark_return": 0.01,
        "symbol_return": 0.03,
        "excess_return": 0.02,
        "cross_sectional_percentile": 0.9,
        "cross_sectional_count": 272,
        "timestamp": "2026-08-11 10:15:00+05:30",
        "source": "benchmark:NIFTY",
    }
    values.update(overrides)
    return MarketRelativeFeatures(**values)


@pytest.mark.parametrize(
    ("strategy", "indicator", "direction"),
    [
        ("ema_adx_trend", _indicator(), "BUY"),
        (
            "ema_adx_trend",
            _indicator(
                close=95,
                open=96,
                ema={20: 97, 50: 100},
                ema_previous={20: 97.5},
                ema_slope_atr={20: -0.25},
                plus_di=12,
                minus_di=35,
            ),
            "SELL",
        ),
        ("donchian_breakout", _indicator(), "BUY"),
        (
            "donchian_breakout",
            _indicator(
                close=97,
                open=98,
                previous_close=99,
                donchian_previous_high=105,
                donchian_previous_low=98,
                plus_di=12,
                minus_di=30,
            ),
            "SELL",
        ),
        (
            "time_series_momentum",
            replace(_indicator(), timeframe=Timeframe.H1),
            "BUY",
        ),
        (
            "time_series_momentum",
            replace(
                _indicator(
                    close=95, ema={20: 97, 50: 100}, roc=-0.03, atr_normalized_momentum=-2.0
                ),
                timeframe=Timeframe.H1,
            ),
            "SELL",
        ),
        ("supertrend_adx_ema", _indicator(close=104), "BUY"),
        (
            "supertrend_adx_ema",
            _indicator(close=96, ema={50: 100}, supertrend_bullish=False, plus_di=12, minus_di=35),
            "SELL",
        ),
        (
            "macd_trend_continuation",
            replace(_indicator(), timeframe=Timeframe.M30),
            "BUY",
        ),
        (
            "macd_trend_continuation",
            replace(
                _indicator(
                    close=95,
                    open=96,
                    ema={50: 100},
                    macd=-1.0,
                    macd_signal=-0.8,
                    macd_histogram=-0.2,
                    macd_previous=-0.7,
                    macd_signal_previous=-0.75,
                    macd_histogram_previous=0.05,
                ),
                timeframe=Timeframe.M30,
            ),
            "SELL",
        ),
    ],
)
def test_trend_strategy_buy_sell_fixtures(config, strategy, indicator, direction):
    decision = evaluate_production_strategy(strategy, indicator, config)
    assert decision is not None
    assert decision.direction == direction
    assert 0.0 <= decision.technical_score <= 1.0
    assert decision.score_components


def test_ema_adx_no_signal_when_overextended(config):
    indicator = _indicator(close=115.0)
    assert evaluate_production_strategy("ema_adx_trend", indicator, config) is None


def test_time_series_momentum_no_signal_when_below_noise(config):
    indicator = replace(_indicator(roc=0.001, atr_normalized_momentum=0.1), timeframe=Timeframe.H1)
    assert evaluate_production_strategy("time_series_momentum", indicator, config) is None


def test_trend_pullback_requires_completed_recovery(config):
    valid = _indicator(previous_low=102.7, previous_close=102.8, close=104.0, open=103.0, rsi=52.0)
    decision = evaluate_production_strategy("trend_pullback", valid, config)
    assert decision is not None and decision.direction == "BUY"
    no_recovery = replace(valid, close=102.5, open=103.0)
    assert evaluate_production_strategy("trend_pullback", no_recovery, config) is None


def test_bollinger_reentry_not_simple_oversold(config):
    valid = _indicator(
        previous_close=98.0,
        bb_previous_lower=98.5,
        close=99.5,
        open=99.0,
        bb_lower=99.0,
        rsi=35.0,
        adx=18.0,
    )
    decision = evaluate_production_strategy("bollinger_rsi_mean_reversion", valid, config)
    assert decision is not None and decision.direction == "BUY"
    still_outside = replace(valid, close=98.5, open=98.0)
    assert (
        evaluate_production_strategy("bollinger_rsi_mean_reversion", still_outside, config) is None
    )


def test_bollinger_sell_reentry(config):
    indicator = _indicator(
        previous_close=108.0,
        bb_previous_upper=107.0,
        close=106.5,
        open=107.0,
        bb_upper=107.0,
        rsi=67.0,
        adx=18.0,
    )
    decision = evaluate_production_strategy("bollinger_rsi_mean_reversion", indicator, config)
    assert decision is not None and decision.direction == "SELL"


def test_vwap_mean_reversion_buy_and_sell(config):
    buy = _indicator(
        close=99,
        open=98.5,
        previous_close=98.7,
        vwap=102,
        vwap_distance=-3,
        vwap_distance_atr=-1.5,
        vwap_zscore=-1.5,
        rsi=38,
        adx=18,
    )
    sell = _indicator(
        close=106,
        open=106.5,
        previous_close=106.3,
        vwap=103,
        vwap_distance=3,
        vwap_distance_atr=1.5,
        vwap_zscore=1.5,
        rsi=62,
        adx=18,
    )
    assert evaluate_production_strategy("vwap_mean_reversion", buy, config).direction == "BUY"
    assert evaluate_production_strategy("vwap_mean_reversion", sell, config).direction == "SELL"


def test_opening_range_requires_completion_and_first_cross(config):
    valid = _indicator()
    assert evaluate_production_strategy("opening_range_breakout", valid, config).direction == "BUY"
    assert (
        evaluate_production_strategy(
            "opening_range_breakout", replace(valid, opening_range_complete=False), config
        )
        is None
    )
    assert (
        evaluate_production_strategy(
            "opening_range_breakout", replace(valid, previous_close=105), config
        )
        is None
    )


def test_relative_strength_requires_aligned_context(config):
    indicator = replace(_indicator(), timeframe=Timeframe.H1)
    assert (
        evaluate_production_strategy(
            "relative_strength_momentum", indicator, config, _context()
        ).direction
        == "BUY"
    )
    stale = _context(timestamp="2026-08-11 09:15:00+05:30")
    assert (
        evaluate_production_strategy("relative_strength_momentum", indicator, config, stale) is None
    )


def test_relative_strength_sell(config):
    indicator = replace(
        _indicator(close=95, ema={20: 97, 50: 100}, plus_di=12, minus_di=35), timeframe=Timeframe.H1
    )
    context = _context(symbol_return=-0.03, excess_return=-0.04, cross_sectional_percentile=0.05)
    assert (
        evaluate_production_strategy(
            "relative_strength_momentum", indicator, config, context
        ).direction
        == "SELL"
    )


def _ohlcv(index: pd.DatetimeIndex, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.arange(len(closes), dtype=float) + 1000,
        },
        index=index,
    )


def test_donchian_excludes_current_candle():
    index = pd.date_range("2026-08-10 09:15", periods=30, freq="15min", tz="Asia/Kolkata")
    closes = np.linspace(100, 105, 30)
    frame = _ohlcv(index, closes)
    frame.iloc[-1, frame.columns.get_loc("high")] = 130.0
    result = calculate_indicators(frame, "TEST", Timeframe.M15)
    assert result.donchian_previous_high < 130.0
    assert result.rolling_high == 130.0


def test_volume_profile_excludes_current_candle_and_finds_poc():
    # 60 prior bars clustered tightly around 100 (the value area), then one breakout bar
    # far above with much higher volume -- the breakout bar's own volume must not smear
    # the profile it is breaking (same self-reference bug Donchian excludes above).
    index = pd.date_range("2026-08-01 09:15", periods=61, freq="15min", tz="Asia/Kolkata")
    closes = np.concatenate([np.full(60, 100.0) + np.tile([0.0, 0.3, -0.3, 0.1], 15), [110.0]])
    frame = _ohlcv(index, closes)
    frame["volume"] = 1000.0
    frame.iloc[-1, frame.columns.get_loc("volume")] = 5000.0

    result = calculate_indicators(frame, "TEST", Timeframe.M15)

    assert result.volume_poc is not None
    assert 99.0 < result.volume_poc < 101.0
    assert result.volume_value_area_high is not None
    assert result.volume_value_area_high < 102.0
    assert result.close == 110.0
    assert result.close > result.volume_value_area_high


def test_pivot_sr_zones_detects_high_volume_pivot_and_ignores_low_volume_one():
    # 450 flat/noisy bars (default pivot_sr_length=40 needs 40 clear bars each side plus
    # a 200-bar ATR window). One pivot high at index 200 with 5x volume (should form a
    # zone); a second, equally sharp pivot high later at index 350 with ORDINARY volume
    # (should be ignored entirely -- same as the source script's isHighVolume gate).
    n = 450
    closes = 100.0 + np.tile([0.0, 0.2, -0.2, 0.1], n // 4 + 1)[:n]
    closes[200] = closes[160:241].max() + 5.0
    closes[350] = closes[310:391].max() + 5.0
    index = pd.date_range("2026-01-01 09:15", periods=n, freq="15min", tz="Asia/Kolkata")
    frame = _ohlcv(index, closes)
    frame["volume"] = 1000.0
    frame.iloc[200, frame.columns.get_loc("volume")] = 5000.0  # high-volume pivot
    # index 350's volume stays at the 1000.0 default -- an ordinary, unconfirmed pivot.

    result = calculate_indicators(frame, "TEST", Timeframe.M15)

    # The most recent VALID (volume-confirmed) pivot is still the one at 200, not 350,
    # because 350 never cleared the volume threshold.
    assert result.pivot_res_zone_top is not None
    assert result.pivot_res_zone_bottom is not None
    assert result.pivot_res_zone_broken is False


def test_pivot_sr_zones_marks_broken_once_price_closes_through():
    n = 450
    closes = 100.0 + np.tile([0.0, 0.2, -0.2, 0.1], n // 4 + 1)[:n]
    closes[200] = closes[160:241].max() + 5.0
    closes[-3:] = closes[200] + 20.0  # clean close through the zone, late in the series
    index = pd.date_range("2026-01-01 09:15", periods=n, freq="15min", tz="Asia/Kolkata")
    frame = _ohlcv(index, closes)
    frame["volume"] = 1000.0
    frame.iloc[200, frame.columns.get_loc("volume")] = 5000.0

    result = calculate_indicators(frame, "TEST", Timeframe.M15)

    assert result.pivot_res_zone_broken is True


def test_volume_profile_breakout_buy_and_sell(config):
    buy = _indicator(
        close=105,
        previous_close=104.0,
        volume_poc=102.0,
        volume_value_area_high=104.5,
        volume_value_area_low=100.0,
    )
    sell = _indicator(
        close=97,
        previous_close=99,
        volume_poc=102.0,
        volume_value_area_high=103.0,
        volume_value_area_low=98.0,
    )
    assert evaluate_production_strategy("volume_profile_breakout", buy, config).direction == "BUY"
    assert (
        evaluate_production_strategy("volume_profile_breakout", sell, config).direction == "SELL"
    )


def test_volume_profile_breakout_none_when_no_fresh_cross(config):
    # Close was already outside the value area on the previous bar too -- not a fresh
    # breakout, must not keep re-firing every bar while price sits outside the range.
    stale = _indicator(
        close=105,
        previous_close=104.6,
        volume_poc=102.0,
        volume_value_area_high=104.5,
        volume_value_area_low=100.0,
    )
    assert evaluate_production_strategy("volume_profile_breakout", stale, config) is None


def test_volume_profile_breakout_none_when_profile_unavailable(config):
    missing = _indicator(volume_poc=None, volume_value_area_high=None, volume_value_area_low=None)
    assert evaluate_production_strategy("volume_profile_breakout", missing, config) is None


def test_ema_trend_confluence_pullback_entry(config):
    # Defaults already satisfy pullback: trend up, dip near EMA20, green candle, RSI/MACD ok.
    pullback = _indicator()
    decision = evaluate_production_strategy("ema_trend_confluence", pullback, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["setup_type"] == "pullback"


def test_ema_trend_confluence_breakout_entry(config):
    breakout = _indicator(
        close=106.0,
        open=105.0,
        high=106.2,
        low=104.8,
        previous_close=104.0,
        candle_body_pct=0.71,
        upper_wick_pct=0.14,
        donchian_previous_high=104.5,
        donchian_previous_low=98.0,
        relative_volume=1.5,
    )
    decision = evaluate_production_strategy("ema_trend_confluence", breakout, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["setup_type"] == "breakout"


def test_ema_trend_confluence_reversal_entry(config):
    # ema50 set ABOVE close so stock_trend_ok (and therefore pullback_entry) is false --
    # isolates this as a pure reversal, not an accidental pullback match.
    reversal = _indicator(
        close=104.0,
        open=103.0,
        previous_close=102.0,
        ema={20: 103.0, 50: 105.0},
        ema_previous={20: 103.0, 50: 105.3},
        rsi=55.0,
        macd_histogram=0.3,
        macd_histogram_previous=0.1,
    )
    decision = evaluate_production_strategy("ema_trend_confluence", reversal, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["setup_type"] == "reversal"


def test_ema_trend_confluence_rejects_overextended_one_bar_rally(config):
    # entry_setup would otherwise pass (reversal), but a >5% single-bar move is
    # exactly what the source script's notHugeRally guard exists to block.
    huge_rally = _indicator(
        close=110.0,
        open=103.0,
        previous_close=100.0,  # (110-100)/100 = 10% one-bar move
        ema={20: 103.0, 50: 100.0},
        ema_previous={20: 103.0, 50: 99.8},
        rsi=55.0,
        macd_histogram=0.3,
        macd_histogram_previous=0.1,
    )
    assert evaluate_production_strategy("ema_trend_confluence", huge_rally, config) is None


def test_ema_trend_confluence_none_when_no_setup_qualifies(config):
    # Flat/no-trend indicator: none of breakout/pullback/reversal conditions hold.
    flat = _indicator(
        close=100.0,
        open=100.0,
        high=100.5,
        low=99.5,
        previous_close=100.0,
        ema={20: 100.0, 50: 100.0},
        ema_previous={20: 100.0, 50: 100.0},
        rsi=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,
        macd_histogram_previous=0.0,
    )
    assert evaluate_production_strategy("ema_trend_confluence", flat, config) is None


def test_ema_trend_confluence_none_when_fields_missing(config):
    missing = _indicator(relative_volume=None)
    assert evaluate_production_strategy("ema_trend_confluence", missing, config) is None


def test_pivot_sr_resistance_breakout_buy(config):
    ind = _indicator(pivot_res_zone_top=104.0, pivot_res_zone_bottom=102.0, pivot_res_zone_broken=False)
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["reaction_type"] == "RESISTANCE_BREAKOUT"


def test_pivot_sr_resistance_hold_rejection_sell(config):
    # Previous bar poked into the zone's bottom edge; this bar pulled back below it.
    ind = _indicator(
        high=104.0,
        previous_high=105.0,
        pivot_res_zone_top=110.0,
        pivot_res_zone_bottom=104.5,
        pivot_res_zone_broken=False,
    )
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "SELL"
    assert decision.supporting_metrics["reaction_type"] == "RESISTANCE_HOLD_REJECTION"


def test_pivot_sr_flipped_resistance_retest_hold_buy(config):
    # Zone already broken (now acting as support); previous low dipped to the level,
    # this bar's low held above it.
    ind = _indicator(pivot_res_zone_top=102.8, pivot_res_zone_bottom=100.0, pivot_res_zone_broken=True)
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["reaction_type"] == "FLIPPED_RESISTANCE_RETEST_HOLD"


def test_pivot_sr_support_breakdown_sell(config):
    ind = _indicator(pivot_sup_zone_top=108.0, pivot_sup_zone_bottom=106.0, pivot_sup_zone_broken=False)
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "SELL"
    assert decision.supporting_metrics["reaction_type"] == "SUPPORT_BREAKDOWN"


def test_pivot_sr_support_hold_bounce_buy(config):
    ind = _indicator(pivot_sup_zone_top=102.8, pivot_sup_zone_bottom=100.0, pivot_sup_zone_broken=False)
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "BUY"
    assert decision.supporting_metrics["reaction_type"] == "SUPPORT_HOLD_BOUNCE"


def test_pivot_sr_flipped_support_retest_rejection_sell(config):
    ind = _indicator(
        high=104.0,
        previous_high=105.0,
        pivot_sup_zone_top=110.0,
        pivot_sup_zone_bottom=104.5,
        pivot_sup_zone_broken=True,
    )
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.direction == "SELL"
    assert decision.supporting_metrics["reaction_type"] == "FLIPPED_SUPPORT_RETEST_REJECTION"


def test_pivot_sr_none_when_no_zones_available(config):
    ind = _indicator()  # no pivot_* overrides -- both zone pairs default to None
    assert evaluate_production_strategy("pivot_volume_sr_zones", ind, config) is None


def test_pivot_sr_resistance_zone_takes_priority_over_support(config):
    # Both zones present and both would fire -- resistance is checked first (matches
    # the source script's own top-to-bottom else-branch ordering).
    ind = _indicator(
        pivot_res_zone_top=104.0,
        pivot_res_zone_bottom=102.0,
        pivot_res_zone_broken=False,
        pivot_sup_zone_top=102.8,
        pivot_sup_zone_bottom=100.0,
        pivot_sup_zone_broken=False,
    )
    decision = evaluate_production_strategy("pivot_volume_sr_zones", ind, config)
    assert decision.supporting_metrics["reaction_type"] == "RESISTANCE_BREAKOUT"


def test_session_vwap_resets_between_nse_sessions():
    first = pd.date_range("2026-08-10 09:15", periods=4, freq="15min", tz="Asia/Kolkata")
    second = pd.date_range("2026-08-11 09:15", periods=4, freq="15min", tz="Asia/Kolkata")
    frame = pd.concat([_ohlcv(first, np.full(4, 100.0)), _ohlcv(second, np.full(4, 200.0))])
    result = calculate_indicators(frame, "TEST", Timeframe.M15)
    assert result.session_vwap == pytest.approx(200.0)


def test_opening_range_uses_only_opening_candles_and_no_future():
    index = pd.date_range("2026-08-11 09:15", periods=6, freq="15min", tz="Asia/Kolkata")
    frame = _ohlcv(index, np.array([100, 101, 102, 103, 104, 105], dtype=float))
    frame.loc[index[4], "high"] = 150.0
    result = calculate_indicators(
        frame, "TEST", Timeframe.M15, IndicatorConfig(opening_range_minutes=30)
    )
    assert result.opening_range_complete
    assert result.opening_range_high < 150.0
    early = calculate_indicators(
        frame.iloc[:2], "TEST", Timeframe.M15, IndicatorConfig(opening_range_minutes=30)
    )
    assert not early.opening_range_complete


def test_cross_sectional_rank_excludes_stale_timestamps():
    aligned = "2026-08-11 10:15:00+05:30"
    inputs = {
        "A": _indicator(symbol="A", roc=0.03, settled_candle_timestamp=aligned),
        "B": _indicator(symbol="B", roc=0.01, settled_candle_timestamp=aligned),
        "C": _indicator(symbol="C", roc=-0.01, settled_candle_timestamp=aligned),
        "FUTURE": _indicator(
            symbol="FUTURE", roc=0.50, settled_candle_timestamp="2026-08-11 10:30:00+05:30"
        ),
    }
    context = build_market_relative_context(inputs)
    assert set(context) == {"A", "B", "C"}
    assert context["A"].cross_sectional_percentile == 1.0


def test_feature_snapshot_reuses_same_indicator_and_context():
    indicator = _indicator()
    context = _context()
    first = FeatureSnapshot.create(
        indicator, settled_candle_timestamp=context.timestamp, market_relative=context
    )
    second = FeatureSnapshot.create(
        indicator, settled_candle_timestamp=context.timestamp, market_relative=context
    )
    assert first.indicators is indicator
    assert first.snapshot_id == second.snapshot_id


def test_strategy_timeframe_matrix_is_enforced(config):
    engine = SignalEngine(production_config=config)
    for strategy_name, timeframes in DEFAULT_TIMEFRAME_MAP.items():
        strategy = StrategyType(strategy_name)
        # Pick any timeframe genuinely outside this strategy's own allowed set, rather
        # than a fixed M15/M5 guess -- a strategy allowed at every probeable timeframe
        # (e.g. one registered at all six) would otherwise leave nothing to test against.
        disallowed = next(tf for tf in Timeframe if tf.value not in timeframes)
        assert (
            engine.generate_signals(replace(_indicator(), timeframe=disallowed), [strategy]) == []
        )


def test_standard_candidate_contract_and_no_hold(config):
    engine = SignalEngine(production_config=config)
    signals = engine.generate_signals(_indicator(), [StrategyType.EMA_ADX_TREND])
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type in {SignalType.BUY, SignalType.SELL}
    assert signal.reason_codes
    assert signal.supporting_metrics
    assert signal.score_components
    assert not signal.technical_score_is_probability
    assert signal.technical_quality == signal.confidence


def test_new_strategy_grains_are_seeded_shadow_only(tmp_path, config):
    registry = StrategyEligibilityRegistry(tmp_path)
    created = ensure_production_strategy_placeholders(registry, config)

    assert len(created) == sum(len(values) for values in DEFAULT_TIMEFRAME_MAP.values())
    records = registry.load_all()
    assert records
    assert all(record.validation_status is EligibilityStatus.SHADOW for record in records)
    assert all(record.oos_trade_count == 0 for record in records)
    assert all(record.artifact_reference is None for record in records)


def test_multiple_new_buy_strategies_consolidate_without_qwen_or_execution(tmp_path, config):
    registry = StrategyEligibilityRegistry(tmp_path)
    ensure_production_strategy_placeholders(registry, config)
    indicator = _indicator()
    snapshot = FeatureSnapshot.create(
        indicator,
        settled_candle_timestamp=indicator.settled_candle_timestamp,
        market_relative=_context(),
    )
    outputs = evaluate_registered_strategies(
        SignalEngine(production_config=config),
        snapshot,
        registry,
        trade_horizon="SWING",
        regime="unknown",
        execution_mode="local_paper",
        eligibility_environment=EligibilityEnvironment.SIMULATED,
    )
    new_buys = [
        item
        for item in outputs
        if item.signal.strategy.value in DEFAULT_TIMEFRAME_MAP
        and item.signal.signal_type is SignalType.BUY
    ]
    assert len(new_buys) >= 3

    candidates = aggregate_strategy_signals(new_buys)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert len(candidate.supporting_signals) == len(new_buys)
    assert not candidate.execution_allowed
    assert not any(item.signal.registry_qwen_allowed for item in new_buys)
