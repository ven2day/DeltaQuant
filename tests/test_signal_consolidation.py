from src.market.indicators import Timeframe
from src.market.signal_consolidation import (
    AGREEMENT_CONFIDENCE_CAP,
    AGREEMENT_CONFIDENCE_STEP,
    consolidate_signals,
)
from src.market.signals import (
    SignalStrength,
    SignalType,
    StrategyType,
    TradingSignal,
)


def _signal(
    symbol: str = "RELIANCE",
    *,
    confidence: float = 0.70,
    strategy: StrategyType = StrategyType.MOMENTUM,
    timeframe: Timeframe = Timeframe.M15,
    signal_type: SignalType = SignalType.BUY,
    entry_price: float = 100.0,
) -> TradingSignal:
    return TradingSignal(
        signal_id=f"SIG-{symbol}-{strategy.value}",
        symbol=symbol,
        signal_type=signal_type,
        strength=SignalStrength.STRONG,
        strategy=strategy,
        timeframe=timeframe,
        entry_price=entry_price,
        stop_loss=98.0,
        target_price=104.0,
        risk_reward_ratio=2.0,
        position_size_pct=5.0,
        confidence=confidence,
    )


def test_agreeing_strategies_consolidate_into_one_opportunity():
    signals = [
        _signal(strategy=StrategyType.MOMENTUM, confidence=0.6),
        _signal(strategy=StrategyType.TREND_FOLLOWING, confidence=0.7),
        _signal(strategy=StrategyType.EMA_PSAR, confidence=0.5),
    ]

    result = consolidate_signals(signals)

    assert len(result) == 1
    c = result[0]
    assert c.agreement_count == 3
    assert set(c.contributing_strategies) == {
        StrategyType.MOMENTUM,
        StrategyType.TREND_FOLLOWING,
        StrategyType.EMA_PSAR,
    }


def test_representative_signal_is_the_highest_confidence_contributor():
    weak = _signal(strategy=StrategyType.MOMENTUM, confidence=0.4, entry_price=99.0)
    strong = _signal(strategy=StrategyType.TREND_FOLLOWING, confidence=0.9, entry_price=101.0)

    c = consolidate_signals([weak, strong])[0]

    assert c.representative_signal is strong
    assert c.representative_signal.entry_price == 101.0


def test_blended_confidence_rewards_agreement_but_caps():
    # Single signal: no boost at all (agreement_count=1 -> +0.05*(1-1)=0).
    single = consolidate_signals([_signal(confidence=0.5)])[0]
    assert single.blended_confidence == 0.5

    # Three agreeing signals: representative's own confidence (0.9) plus two steps.
    signals = [
        _signal(strategy=StrategyType.MOMENTUM, confidence=0.9),
        _signal(strategy=StrategyType.TREND_FOLLOWING, confidence=0.3),
        _signal(strategy=StrategyType.EMA_PSAR, confidence=0.2),
    ]
    c = consolidate_signals(signals)[0]
    expected = min(AGREEMENT_CONFIDENCE_CAP, 0.9 + AGREEMENT_CONFIDENCE_STEP * 2)
    assert c.blended_confidence == expected

    # Many agreeing signals must never exceed the cap, however high individual
    # confidences or agreement count are.
    many = [
        _signal(strategy=st, confidence=0.95)
        for st in list(StrategyType)  # all 7 strategies "agree"
    ]
    c_many = consolidate_signals(many)[0]
    assert c_many.blended_confidence <= AGREEMENT_CONFIDENCE_CAP


def test_different_directions_never_merge():
    buy = _signal(signal_type=SignalType.BUY, strategy=StrategyType.MOMENTUM)
    sell = _signal(signal_type=SignalType.SELL, strategy=StrategyType.TREND_FOLLOWING)

    result = consolidate_signals([buy, sell])

    assert len(result) == 2
    directions = {c.signal_type for c in result}
    assert directions == {SignalType.BUY, SignalType.SELL}


def test_different_timeframes_never_merge():
    m15 = _signal(timeframe=Timeframe.M15, strategy=StrategyType.MOMENTUM)
    h1 = _signal(timeframe=Timeframe.H1, strategy=StrategyType.TREND_FOLLOWING)

    result = consolidate_signals([m15, h1])

    assert len(result) == 2
    timeframes = {c.timeframe for c in result}
    assert timeframes == {Timeframe.M15, Timeframe.H1}


def test_different_symbols_never_merge():
    a = _signal(symbol="RELIANCE", strategy=StrategyType.MOMENTUM)
    b = _signal(symbol="TCS", strategy=StrategyType.MOMENTUM)

    result = consolidate_signals([a, b])

    assert len(result) == 2
    assert {c.symbol for c in result} == {"RELIANCE", "TCS"}


def test_hold_signals_are_skipped():
    hold = _signal(signal_type=SignalType.HOLD)

    result = consolidate_signals([hold])

    assert result == []


def test_empty_input_produces_empty_output():
    assert consolidate_signals([]) == []


def test_to_dict_carries_blended_confidence_and_contributors():
    signals = [
        _signal(strategy=StrategyType.MOMENTUM, confidence=0.6),
        _signal(strategy=StrategyType.TREND_FOLLOWING, confidence=0.5),
    ]
    c = consolidate_signals(signals)[0]

    payload = c.to_dict()

    assert payload["confidence"] == c.blended_confidence
    assert set(payload["contributing_strategies"]) == {"momentum", "trend_following"}
    assert payload["agreement_count"] == 2
