from unittest.mock import MagicMock, patch

from src.agents.prediction import PredictionSignal
from src.market.assessment_matrix import build_assessment_matrix
from src.market.indicators import Timeframe
from src.market.signal_consolidation import consolidate_signals
from src.market.signals import SignalStrength, SignalType, StrategyType, TradingSignal


def _signal(
    symbol: str = "RELIANCE",
    *,
    confidence: float = 0.8,
    strategy: StrategyType = StrategyType.MOMENTUM,
    timeframe: Timeframe = Timeframe.M15,
    signal_type: SignalType = SignalType.BUY,
    adx: float = 30.0,
    plus_di: float = 28.0,
    minus_di: float = 12.0,
) -> TradingSignal:
    return TradingSignal(
        signal_id=f"SIG-{symbol}-{strategy.value}-{timeframe.value}",
        symbol=symbol,
        signal_type=signal_type,
        strength=SignalStrength.STRONG,
        strategy=strategy,
        timeframe=timeframe,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=104.0,
        risk_reward_ratio=2.0,
        position_size_pct=5.0,
        confidence=confidence,
        indicators={"trend": {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}},
    )


def _settings_mock(reject_score=0.35, min_confidence=0.6):
    settings = MagicMock()
    settings.scalp_matrix_reject_score = reject_score
    settings.scalp_min_confidence = min_confidence
    return settings


def test_missing_timeframe_data_rejects_with_reason_never_crashes():
    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE",
            {Timeframe.M15: []},  # no candidates at all for this timeframe
            {},
            "trending_up",
        )

    assert matrix[Timeframe.M15].decision == "REJECT"
    assert matrix[Timeframe.M15].strategy_consensus == 0
    assert matrix[Timeframe.M15].ml_probability is None
    assert "no strategy signal generated" in matrix[Timeframe.M15].reasons[0]


def test_strong_agreeing_compatible_signal_produces_buy():
    signals = [
        _signal(strategy=StrategyType.MOMENTUM, confidence=0.9, adx=30, plus_di=28, minus_di=12),
        _signal(
            strategy=StrategyType.TREND_FOLLOWING, confidence=0.85, adx=30, plus_di=28, minus_di=12
        ),
    ]
    consolidated = consolidate_signals(signals)

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE",
            {Timeframe.M15: consolidated},
            {},
            "trending_up",
        )

    cell = matrix[Timeframe.M15]
    assert cell.decision == "BUY"
    assert cell.strategy_consensus == 2
    assert cell.regime_compatible is True
    assert cell.score >= 0.6


def test_weak_signal_is_rejected():
    signals = [_signal(confidence=0.1, adx=10, plus_di=None, minus_di=None)]
    consolidated = consolidate_signals(signals)

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE", {Timeframe.M15: consolidated}, {}, "ranging"
        )

    assert matrix[Timeframe.M15].decision == "REJECT"


def test_regime_incompatible_signal_is_capped_below_buy():
    """mean_reversion is not a typical fit for trending_up (see
    regime_compatibility.py's table) -- even a high raw confidence must not reach BUY
    once the compatibility penalty is applied."""
    signals = [
        _signal(
            strategy=StrategyType.MEAN_REVERSION,
            confidence=0.9,
            adx=30,
            plus_di=28,
            minus_di=12,  # -> locally inferred regime is trending_up
        )
    ]
    consolidated = consolidate_signals(signals)

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE", {Timeframe.M15: consolidated}, {}, "trending_up"
        )

    cell = matrix[Timeframe.M15]
    assert cell.regime_compatible is False
    assert cell.decision != "BUY"


def test_ml_probability_blended_into_score():
    signals = [_signal(confidence=0.7)]
    consolidated = consolidate_signals(signals)
    prediction = PredictionSignal(
        symbol="RELIANCE",
        direction="up",
        confidence=0.9,
        predicted_change_pct=1.0,
        reasoning="test",
    )

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE",
            {Timeframe.M15: consolidated},
            {("RELIANCE", "15m"): prediction},
            "trending_up",
        )

    cell = matrix[Timeframe.M15]
    assert cell.ml_probability == 0.9  # agrees with BUY direction
    assert cell.score == 0.6 * 0.7 + 0.4 * 0.9


def test_multiple_timeframes_assessed_independently():
    m15_signals = consolidate_signals([_signal(timeframe=Timeframe.M15, confidence=0.9)])
    h1_signals = consolidate_signals([_signal(timeframe=Timeframe.H1, confidence=0.1, adx=10)])

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE",
            {Timeframe.M15: m15_signals, Timeframe.H1: h1_signals},
            {},
            "trending_up",
        )

    assert set(matrix.keys()) == {Timeframe.M15, Timeframe.H1}
    assert matrix[Timeframe.M15].decision == "BUY"
    assert matrix[Timeframe.H1].decision == "REJECT"


def test_sell_dominant_cell_is_never_mislabeled_buy():
    """Regression: a strong, regime-compatible SELL signal must not be labeled
    'BUY' just because it scored well in the opposite direction -- this matrix only
    ever calls out long-only entries as 'BUY', matching the rest of the pipeline's
    long-only convention (settings.long_only)."""
    signals = [
        _signal(
            strategy=StrategyType.TREND_FOLLOWING,
            confidence=0.95,
            signal_type=SignalType.SELL,
            adx=30,
            plus_di=12,
            minus_di=28,  # -> locally inferred regime trending_down, matches strategy
        )
    ]
    consolidated = consolidate_signals(signals)

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE", {Timeframe.M15: consolidated}, {}, "trending_down"
        )

    cell = matrix[Timeframe.M15]
    assert cell.decision == "REJECT"
    assert cell.decision != "BUY"
    assert "SELL" in " ".join(cell.reasons)


def test_local_regime_divergence_from_cycle_regime_is_noted():
    # This signal's own ADX/DI implies "ranging" (adx<20), even though the caller
    # passes a cycle-wide regime of "trending_up" -- the divergence must be visible.
    signals = [_signal(strategy=StrategyType.MEAN_REVERSION, confidence=0.9, adx=10)]
    consolidated = consolidate_signals(signals)

    with patch("src.market.assessment_matrix.get_settings", return_value=_settings_mock()):
        matrix = build_assessment_matrix(
            "RELIANCE", {Timeframe.M15: consolidated}, {}, "trending_up"
        )

    reasons_text = " ".join(matrix[Timeframe.M15].reasons)
    assert "differs from the cycle-wide regime" in reasons_text
