"""
EntryQualityEvaluator tests. Fixtures are deliberately synthetic OHLCV series
engineered to land cleanly in each of the four output states, rather than random
data -- each test documents exactly which feature of the fixture is meant to trigger
that state.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.market.entry_quality import MIN_BARS_REQUIRED, evaluate_entry_quality


def _settings_mock(**overrides):
    settings = MagicMock()
    settings.scalp_min_volume_ratio = 1.2
    settings.scalp_vwap_max_distance_pct = 0.5
    settings.scalp_ema9_max_distance_pct = 0.6
    settings.scalp_atr_extension_max_multiple = 1.5
    settings.scalp_wick_ratio_max = 0.6
    settings.scalp_swing_lookback_bars = 60
    settings.scalp_breakout_retest_lookback_bars = 12
    settings.scalp_resistance_min_distance_pct = 0.3
    settings.scalping_swing_threshold_pct = 0.5
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _evaluate(rows, *, direction="BUY", stop_loss=99.5, target_price=101.0, settings=None):
    df = _frame(rows)
    with patch(
        "src.market.entry_quality.get_settings",
        return_value=settings or _settings_mock(),
    ):
        return evaluate_entry_quality("TEST", direction, df, None, stop_loss, target_price)


def test_insufficient_history_rejects_without_crashing():
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(MIN_BARS_REQUIRED - 1)]
    result = _evaluate(rows)

    assert result.status == "REJECT"
    assert "insufficient" in result.reasons[0]


def test_incoherent_stop_target_geometry_rejects():
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(25)]
    # BUY with stop ABOVE entry -- nonsensical geometry, must reject regardless of
    # every other feature.
    result = _evaluate(rows, direction="BUY", stop_loss=101.0, target_price=102.0)

    assert result.status == "REJECT"


def test_wide_rejection_wick_on_trigger_candle_rejects():
    """A long upper wick on the triggering BUY candle (price spiked and was sold
    back down) is a rejection/indecision signature, not a clean trigger."""
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(24)]
    rows.append([100.0, 105.0, 99.9, 100.2, 1500])  # huge upper wick, small body
    result = _evaluate(rows)

    assert result.status == "REJECT"
    assert result.upper_wick_ratio > 0.6
    assert "wick" in result.reasons[0]


def test_low_relative_volume_rejects():
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(24)]
    rows.append([100.0, 100.3, 99.8, 100.1, 300])  # well below the 20-bar average
    result = _evaluate(rows)

    assert result.status == "REJECT"
    assert result.relative_volume < 1.2
    assert "volume" in result.reasons[0]


def test_quiet_consolidation_with_good_volume_enters_now():
    """Price sitting right at VWAP/EMA9, no nearby untested level, no overextension,
    and a volume pickup on the trigger bar -- the textbook clean entry."""
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(24)]
    rows.append([100.0, 100.4, 99.9, 100.2, 1500])
    result = _evaluate(rows)

    assert result.status == "ENTER_NOW"
    assert result.preferred_entry_low < 100.2 < result.preferred_entry_high
    assert abs(result.vwap_distance_pct) < 0.5
    assert result.nearest_swing_resistance is None


def test_extended_price_after_a_sharp_move_waits_for_pullback():
    """A quiet uptrend followed by a sharp jump: price is now several ATRs away
    from VWAP -- entry quality should prefer a pullback, not chase."""
    rows = [
        [100.0 + i * 0.05, 100.1 + i * 0.05, 99.9 + i * 0.05, 100.0 + i * 0.05, 1000]
        for i in range(24)
    ]
    last_close = rows[-1][3]
    rows.append(
        [last_close, last_close + 3.2, last_close - 0.1, last_close + 3.0, 1400]
    )
    result = _evaluate(rows, stop_loss=last_close - 1, target_price=last_close + 6)

    assert result.status == "WAIT_PULLBACK"
    assert result.atr_extension > 1.5
    assert "pullback" in result.reasons[-1]
    # Preferred re-entry range should sit back toward VWAP/EMA9, below current price.
    assert result.preferred_entry_high < last_close + 3.0


def test_consolidating_below_untested_resistance_waits_for_breakout():
    """A confirmed prior swing high (resistance) that price approaches but has not
    yet closed through within the recent lookback window -- the setup is good, but
    entering now means buying right into an untested ceiling."""
    rows: list[list[float]] = []
    for p in [100, 102, 105, 108, 110]:  # ramp up to establish the resistance pivot
        rows.append([p - 0.2, p + 0.2, p - 0.3, p, 1000])
    for p in [109, 107, 106]:  # pull back far enough to CONFIRM 110 as a pivot
        rows.append([p + 0.2, p + 0.3, p - 0.2, p, 1000])
    for p in [107, 108, 109]:  # rise again, approaching but not through 110
        rows.append([p - 0.3, p + 0.3, p - 0.4, p, 1000])
    # Long, wider-range consolidation just under resistance -- dominates the VWAP
    # window (realistic ATR, not an artificially silent tape) and keeps the last
    # ~12 bars (breakout lookback) entirely below 110, so breakout_state stays "none".
    cycle = [109.7, 109.9, 109.75, 109.95, 109.8]
    for i in range(45):
        p = cycle[i % len(cycle)]
        rows.append([p - 0.3, p + 0.3, p - 0.35, p, 1000])
    rows[-1] = [109.8, 109.98, 109.5, 109.9, 1400]  # volume pickup on the last bar

    result = _evaluate(rows, stop_loss=108.5, target_price=111.5)

    assert result.status == "WAIT_BREAKOUT"
    assert result.nearest_swing_resistance == pytest.approx(110.0)
    assert result.breakout_state == "none"
    assert "resistance" in result.reasons[-1]
    assert result.preferred_entry_low == pytest.approx(110.0)


def test_result_is_json_serializable():
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(24)]
    rows.append([100.0, 100.4, 99.9, 100.2, 1500])
    result = _evaluate(rows)

    import json

    json.dumps(result.to_dict())
