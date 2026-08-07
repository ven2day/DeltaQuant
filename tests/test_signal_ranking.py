from dataclasses import dataclass

import pytest

from src.agents.prediction import PredictionSignal
from src.market.indicators import Timeframe
from src.market.signal_ranking import rank_signals, select_diversified_signals
from src.market.signals import (
    SignalStrength,
    SignalType,
    StrategyType,
    TradingSignal,
)


@dataclass
class _Performance:
    total_trades: int
    winning_trades: int


class _Tracker:
    def __init__(self, total: int = 0, winners: int = 0):
        self.performance = _Performance(total, winners)

    def get_strategy_performance(self, strategy, regime, lookback_days=30):
        return self.performance


def _signal(
    symbol: str = "RELIANCE",
    *,
    confidence: float = 0.70,
    risk_reward: float = 2.0,
    strategy: StrategyType = StrategyType.MOMENTUM,
    timeframe: Timeframe = Timeframe.M15,
) -> TradingSignal:
    return TradingSignal(
        signal_id=f"SIG-{symbol}",
        symbol=symbol,
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        strategy=strategy,
        timeframe=timeframe,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=104.0,
        risk_reward_ratio=risk_reward,
        position_size_pct=5.0,
        confidence=confidence,
        indicators={
            "trend": {"adx": 30.0, "plus_di": 28.0, "minus_di": 12.0}
        },
    )


def test_cold_start_is_not_reported_as_measured_accuracy():
    signal = _signal()

    result = rank_signals([signal], {}, _Tracker())[0]

    assert result.historical_win_probability == 0.5
    assert result.historical_sample_size == 0
    assert result.accuracy_status == "cold_start"
    assert signal.indicators["ranking"]["accuracy_status"] == "cold_start"


def test_ml_agreement_improves_rank_and_disagreement_reduces_it():
    agreeing = _signal("RELIANCE")
    disagreeing = _signal("TCS")
    predictions = {
        ("RELIANCE", "15m"): PredictionSignal("RELIANCE", "up", 0.8, 1.0, ""),
        ("TCS", "15m"): PredictionSignal("TCS", "down", 0.8, -1.0, ""),
    }

    ranked = rank_signals([disagreeing, agreeing], predictions, _Tracker())

    assert ranked[0].signal.symbol == "RELIANCE"
    assert ranked[0].estimated_win_probability > ranked[1].estimated_win_probability


def test_historical_accuracy_uses_beta_smoothing():
    result = rank_signals([_signal()], {}, _Tracker(total=20, winners=15))[0]

    assert result.historical_win_probability == pytest.approx(17 / 24)
    assert result.accuracy_status == "empirical"


def test_discovered_signal_tilt_changes_probability():
    baseline = rank_signals([_signal()], {}, _Tracker())[0]
    tilted = rank_signals(
        [_signal()],
        {},
        _Tracker(),
        discovered_signal_tilts={"RELIANCE": 0.08},
    )[0]

    assert tilted.estimated_win_probability == pytest.approx(
        baseline.estimated_win_probability + 0.08
    )
    assert tilted.discovered_signal_tilt == 0.08


def test_discovered_signal_tilt_is_scoped_to_its_timeframe():
    m15 = _signal(timeframe=Timeframe.M15)
    h1 = _signal(timeframe=Timeframe.H1)

    ranked = rank_signals(
        [m15, h1],
        {},
        _Tracker(),
        discovered_signal_tilts={("RELIANCE", "15m"): 0.08},
    )
    by_timeframe = {item.signal.timeframe: item for item in ranked}

    assert by_timeframe[Timeframe.M15].discovered_signal_tilt == 0.08
    assert by_timeframe[Timeframe.H1].discovered_signal_tilt == 0.0


def test_diversified_selection_limits_symbols_and_signals(monkeypatch):
    monkeypatch.setattr(
        "src.market.signal_ranking.get_stock_sector",
        lambda symbol: {"A": "Tech", "B": "Tech", "C": "Bank"}[symbol],
    )
    signals = [_signal("A"), _signal("A"), _signal("B"), _signal("C")]
    ranked = rank_signals(signals, {}, _Tracker())

    selected = select_diversified_signals(
        ranked, max_symbols=2, max_per_sector=1, max_signals_per_symbol=1
    )

    assert [item.signal.symbol for item in selected] == ["A", "C"]
