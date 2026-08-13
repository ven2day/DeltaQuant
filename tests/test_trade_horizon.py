"""
Tests for the TradeHorizon domain type (src/core/candidates/signals.py) and its Stage-1
scaffolding: every pre-existing call site must keep behaving exactly as before now
that a trade_horizon field/setting exists, since scalp support is additive-only.
"""

from src.core.candidates import (
    SignalStrength,
    SignalType,
    StrategyType,
    TradeHorizon,
    TradingSignal,
)
from src.core.indicators import Timeframe


def _signal(**overrides) -> TradingSignal:
    defaults = dict(
        signal_id="SIG-1",
        symbol="RELIANCE",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        strategy=StrategyType.MOMENTUM,
        timeframe=Timeframe.M15,
        entry_price=100.0,
        stop_loss=98.0,
        target_price=104.0,
        risk_reward_ratio=2.0,
        position_size_pct=5.0,
        confidence=0.7,
    )
    defaults.update(overrides)
    return TradingSignal(**defaults)


def test_trading_signal_defaults_to_swing_horizon():
    """Every pre-existing constructor call (none of which pass trade_horizon) must
    keep producing SWING signals -- this is the backward-compatibility guarantee the
    whole scalping feature depends on."""
    signal = _signal()

    assert signal.trade_horizon is TradeHorizon.SWING
    assert signal.to_dict()["trade_horizon"] == "SWING"


def test_trading_signal_accepts_explicit_scalp_horizon():
    signal = _signal(trade_horizon=TradeHorizon.SCALP)

    assert signal.trade_horizon is TradeHorizon.SCALP
    assert signal.to_dict()["trade_horizon"] == "SCALP"


def test_trade_horizon_enum_values():
    assert TradeHorizon.SWING.value == "SWING"
    assert TradeHorizon.SCALP.value == "SCALP"
