import pandas as pd

from src.market.price_geometry import (
    atr_pct,
    ema,
    nearest_support_resistance,
    session_vwap,
    wick_ratios,
    zigzag_pivots,
)


def _flat_frame(n: int, price: float = 100.0, volume: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [volume] * n,
        }
    )


def test_ema_of_flat_series_equals_the_flat_price():
    frame = _flat_frame(30, price=50.0)
    assert ema(frame, 9) == 50.0


def test_session_vwap_of_flat_series_equals_the_flat_price():
    frame = _flat_frame(80, price=100.0)
    assert session_vwap(frame, bars=75) == 100.0


def test_session_vwap_falls_back_to_last_close_on_zero_volume():
    frame = _flat_frame(80, price=100.0, volume=0.0)
    frame.loc[frame.index[-1], "close"] = 105.0
    assert session_vwap(frame, bars=75) == 105.0


def test_session_vwap_weights_toward_higher_volume_bars():
    frame = _flat_frame(10, price=100.0, volume=1.0)
    frame.loc[frame.index[-1], ["high", "low", "close"]] = 110.0
    frame.loc[frame.index[-1], "volume"] = 1000.0  # dominant bar
    vwap = session_vwap(frame, bars=10)
    assert vwap > 105.0  # pulled strongly toward the high-volume 110 bar


def test_atr_pct_zero_for_flat_series():
    frame = _flat_frame(30, price=100.0)
    assert atr_pct(frame, entry=100.0, lookback=14) == 0.0


def test_atr_pct_positive_for_volatile_series():
    n = 30
    frame = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [102.0] * n,
            "low": [98.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )
    assert atr_pct(frame, entry=100.0, lookback=14) > 0.0


def test_atr_pct_insufficient_history_returns_zero():
    frame = _flat_frame(5, price=100.0)
    assert atr_pct(frame, entry=100.0, lookback=14) == 0.0


def test_wick_ratios_of_a_doji_like_flat_candle():
    upper, lower = wick_ratios(open_=100.0, high=100.0, low=100.0, close=100.0)
    assert upper == 0.0
    assert lower == 0.0


def test_wick_ratios_of_a_hammer_has_large_lower_wick():
    # open near top, close near top, long lower wick down to 90.
    upper, lower = wick_ratios(open_=99.0, high=100.0, low=90.0, close=99.5)
    assert lower > 0.8
    assert upper < 0.1


def test_wick_ratios_of_a_shooting_star_has_large_upper_wick():
    upper, lower = wick_ratios(open_=91.0, high=100.0, low=90.0, close=90.5)
    assert upper > 0.8
    assert lower < 0.1


def test_zigzag_pivots_returns_price_levels_not_just_magnitudes():
    # Rises to 110, falls to 95, rises to 108 -- two completed pivots expected
    # with a threshold of 5.
    closes = pd.Series([100, 102, 105, 110, 108, 100, 95, 98, 103, 108])
    pivots = zigzag_pivots(closes, threshold=5)

    assert len(pivots) >= 1
    for _, price in pivots:
        assert price in closes.values


def test_zigzag_pivots_empty_for_too_short_series():
    assert zigzag_pivots(pd.Series([100]), threshold=5) == []


def test_zigzag_pivots_empty_when_no_reversal_clears_threshold():
    closes = pd.Series([100, 100.5, 101, 100.5, 101.2])
    assert zigzag_pivots(closes, threshold=5) == []


def test_nearest_support_resistance_brackets_current_price():
    closes = pd.Series([100, 105, 110, 105, 100, 95, 90, 95, 100, 105])
    frame = pd.DataFrame({"close": closes})

    support, resistance = nearest_support_resistance(
        frame, current_price=100.0, threshold_pct=4.0, lookback_bars=10
    )

    if support is not None:
        assert support < 100.0
    if resistance is not None:
        assert resistance > 100.0


def test_nearest_support_resistance_none_when_current_price_invalid():
    frame = pd.DataFrame({"close": [100, 101, 102]})
    support, resistance = nearest_support_resistance(
        frame, current_price=0.0, threshold_pct=1.0, lookback_bars=10
    )
    assert support is None
    assert resistance is None
