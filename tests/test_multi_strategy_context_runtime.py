from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.candidate_cache import reset_candidate_verdict_cache
from src.agents.context_review import (
    ContextHandling,
    ContextReviewState,
    QwenReviewPolicy,
    build_context_packet,
    parse_context_reviews,
    serialize_context_packets,
)
from src.agents.signal_validation import signal_validation_node
from src.agents.state import create_initial_state
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.core.aggregation import (
    FeatureSnapshot,
    aggregate_strategy_signals,
    evaluate_registered_strategies,
)
from src.core.candidates import (
    SignalStrength,
    SignalType,
    StrategyType,
    TradingSignal,
)
from src.core.indicators import IndicatorResult, Timeframe


@pytest.fixture(autouse=True)
def _isolated_qwen_cache():
    reset_candidate_verdict_cache()
    yield
    reset_candidate_verdict_cache()


def _version(strategy: StrategyType, *, validated: bool = True) -> StrategyEligibility:
    return StrategyEligibility(
        strategy_name=strategy.value,
        timeframe="15m",
        model_version=f"{strategy.value}-model-v1",
        validation_status=(
            EligibilityStatus.PAPER_APPROVED if validated else EligibilityStatus.SHADOW
        ),
        validated_at="2026-08-01T00:00:00+00:00",
        validation_window={"dataset_id": "test-dataset"},
        oos_trade_count=100,
        oos_profit_factor=1.4 if validated else 0.8,
        oos_max_drawdown=0.08,
        oos_win_rate=0.58,
        minimum_model_confidence=0.0,
        requires_ml=strategy == StrategyType.BREAKOUT,
        artifact_reference=("breakout-model.joblib" if strategy == StrategyType.BREAKOUT else None),
    )


def _indicator() -> IndicatorResult:
    return IndicatorResult(
        symbol="RELIANCE",
        timeframe=Timeframe.M15,
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=100_000,
        sma={20: 99.0},
        ema={9: 100.0, 20: 99.5, 50: 98.0},
        rsi=61.0,
        macd=1.0,
        macd_signal=0.5,
        macd_histogram=0.5,
        adx=29.0,
        plus_di=30.0,
        minus_di=15.0,
        atr=1.2,
        bb_upper=104.0,
        bb_middle=100.0,
        bb_lower=96.0,
        bb_percent=0.62,
        vwap=100.4,
        cci=110.0,
        psar=99.0,
        psar_bullish=True,
        ha_bullish=True,
        ha_prev_bullish=False,
    )


def _signal(strategy: StrategyType, side: SignalType) -> TradingSignal:
    return TradingSignal(
        signal_id=f"{strategy.value}-{side.value}",
        symbol="RELIANCE",
        signal_type=side,
        strength=SignalStrength.STRONG,
        strategy=strategy,
        timeframe=Timeframe.M15,
        entry_price=101.0,
        stop_loss=99.0,
        target_price=105.0,
        risk_reward_ratio=2.0,
        position_size_pct=1.0,
        confidence=0.8,
        timestamp=datetime(2026, 8, 11, tzinfo=UTC),
    )


class _RecordingEngine:
    def __init__(self, sides: dict[StrategyType, SignalType]):
        self.sides = sides
        self.indicator_ids: list[int] = []

    def generate_signals(self, indicators, active_strategies=None):
        self.indicator_ids.append(id(indicators))
        strategy = active_strategies[0]
        return [_signal(strategy, self.sides.get(strategy, SignalType.BUY))]


def test_all_strategies_share_one_feature_snapshot_and_shadow_isolated(tmp_path):
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(_version(StrategyType.MOMENTUM))
    registry.register(_version(StrategyType.BREAKOUT))
    registry.register(_version(StrategyType.MEAN_REVERSION, validated=False))
    snapshot = FeatureSnapshot.create(
        _indicator(), settled_candle_timestamp="2026-08-11T10:15:00+05:30"
    )
    engine = _RecordingEngine({})

    # Explicit gate-enabled settings, independent of whatever the real environment's
    # STRATEGY_ELIGIBILITY_PAPER_GATE_ENABLED happens to be set to right now -- this
    # test is specifically about SHADOW isolation, not the gate toggle.
    settings = SimpleNamespace(strategy_eligibility_paper_gate_enabled=True)
    with patch("src.backtesting.strategy_eligibility.get_settings", return_value=settings):
        outputs = evaluate_registered_strategies(
            engine,
            snapshot,
            registry,
            trade_horizon="SCALP",
            regime="trending_up",
            execution_mode="local_paper",
            eligibility_environment=EligibilityEnvironment.PAPER,
        )

    assert len(engine.indicator_ids) == len(StrategyType)
    assert set(engine.indicator_ids) == {id(snapshot.indicators)}
    assert {item.strategy_name for item in outputs if item.executable} == {
        "momentum",
        "breakout",
    }
    assert all(item.shadow_only for item in outputs if not item.executable)
    assert next(item for item in outputs if item.strategy_name == "breakout").ml_required
    assert not next(item for item in outputs if item.strategy_name == "momentum").ml_required


def test_aggregation_makes_one_candidate_and_retains_opposition(tmp_path):
    registry = StrategyEligibilityRegistry(tmp_path)
    for strategy in (StrategyType.MOMENTUM, StrategyType.BREAKOUT, StrategyType.MEAN_REVERSION):
        registry.register(_version(strategy))
    snapshot = FeatureSnapshot.create(
        _indicator(), settled_candle_timestamp="2026-08-11T10:15:00+05:30"
    )
    engine = _RecordingEngine({StrategyType.MEAN_REVERSION: SignalType.SELL})
    outputs = evaluate_registered_strategies(
        engine,
        snapshot,
        registry,
        trade_horizon="SCALP",
        regime="trending_up",
        execution_mode="local_paper",
        eligibility_environment=EligibilityEnvironment.PAPER,
    )
    candidates = aggregate_strategy_signals(outputs)

    assert len(candidates) == 1
    assert candidates[0].conflict_level == "HIGH"
    assert candidates[0].conflict_resolution == "UNRESOLVED_BLOCK"
    assert not candidates[0].pipeline_eligible
    assert not candidates[0].execution_allowed
    assert candidates[0].opposing_signals
    payload = candidates[0].to_dict()
    assert payload["supporting_strategies"]
    assert payload["opposing_strategies"] == ["mean_reversion"]


def _settings(*, capacity: int = 20):
    return SimpleNamespace(
        enable_llm_agents=True,
        llm_cost_optimization_enabled=True,
        llm_cache_ttl_scalp_seconds=300,
        llm_cache_ttl_swing_seconds=1800,
        llm_cache_material_price_move_bps=20.0,
        llm_cache_ml_probability_step=0.05,
        llm_cache_expected_r_step=0.1,
        llm_cache_entry_score_step=5.0,
        qwen_regime_uncertain_below=0.6,
        qwen_ml_conflict_below=0.45,
        qwen_max_reviews_per_event=capacity,
        qwen_max_input_tokens_per_review=900,
        qwen_max_output_tokens_review=256,
        enable_rate_limiting=False,
    )


def _candidate(signal_id: str = "REL-1", **overrides):
    value = {
        "signal_id": signal_id,
        "symbol": "RELIANCE",
        "timeframe": "15m",
        "trade_horizon": "SCALP",
        "signal_type": "BUY",
        "strategy": "momentum",
        "strategy_version": "momentum-scalp-v1",
        "feature_snapshot_id": "features-v3-hash",
        "feature_version": "features_v3",
        "signal_candle_timestamp": "2026-08-11T10:15:00+05:30",
        "supporting_strategies": ["momentum", "breakout"],
        "opposing_strategies": [],
        "agreement_level": "MEDIUM",
        "conflict_level": "NONE",
        "technical_quality": 0.81,
        "confidence": 0.81,
        "risk_reward_ratio": 2.0,
        "entry_price": 101.0,
        "stop_loss": 99.0,
        "target_price": 105.0,
        "ml_status": "ABSTAIN",
        "ml_probability": None,
        "execution_mode": "local_paper",
    }
    value.update(overrides)
    return value


def _state(*signals):
    state = create_initial_state(trade_horizon="SCALP", execution_mode="local_paper")
    state["signals"] = list(signals)
    state["active_strategies"] = ["momentum", "breakout"]
    state["regime"] = "trending_up"
    state["regime_confidence"] = 0.9
    state["shared_market_context"] = {
        "context_version": "market-v1",
        "major_market_event_risk": "NONE",
        "market_sentiment": 0.0,
        "major_news_summary": [],
    }
    return state


def _run_validation(state, response=None, *, capacity=20, error=None, usage=None):
    budget = MagicMock()
    budget.is_over_hard_budget.return_value = False
    breaker = MagicMock(is_available=True, recovery_time=1)
    invoke = MagicMock(return_value=response)
    if error is not None:
        invoke.side_effect = error
    with (
        patch(
            "src.agents.signal_validation.get_settings", return_value=_settings(capacity=capacity)
        ),
        patch("src.agents.signal_validation.get_cost_tracker", return_value=budget),
        patch("src.agents.signal_validation.get_llm_circuit_breaker", return_value=breaker),
        patch("src.agents.signal_validation.invoke_with_fallback", invoke),
        patch("src.agents.signal_validation.model_for_tier", return_value="qwen-review"),
        patch("src.agents.signal_validation.record_llm_response", return_value=usage),
        patch("src.agents.signal_validation.record_candidate_review"),
    ):
        result = signal_validation_node(state)
    return result, invoke


def test_clear_candidate_skips_qwen_and_ml_abstain_does_not_block():
    result, invoke = _run_validation(_state(_candidate()))
    invoke.assert_not_called()
    assert len(result["validated_signals"]) == 1
    assert result["llm_optimization_metrics"]["qwen_skipped_clear"] == 1


def test_conflict_invokes_qwen_caches_structured_review_and_counts_tokens():
    reset_candidate_verdict_cache()
    response = MagicMock()
    response.content = (
        '{"context_reviews":[{"signal_id":"REL-1","state":"CAUTION",'
        '"confidence":0.74,"supporting_factors":["TREND_SUPPORT"],'
        '"risk_factors":["STRATEGY_CONFLICT"],"unresolved_conflicts":[],'
        '"handling":"CONTINUE_WITH_CAUTION"}]}'
    )
    usage = SimpleNamespace(input_tokens=120, output_tokens=30)
    candidate = _candidate(opposing_strategies=["mean_reversion"], conflict_level="HIGH")
    first, invoke = _run_validation(_state(candidate), response, usage=usage)
    second, invoke_again = _run_validation(_state(candidate), response, usage=usage)

    assert invoke.call_count == 1
    invoke_again.assert_not_called()
    assert first["llm_optimization_metrics"]["qwen_total_tokens"] == 150
    assert sum(first["llm_optimization_metrics"]["qwen_tokens_by_strategy"].values()) == 150
    assert second["llm_optimization_metrics"]["qwen_cache_hits"] == 1
    assert second["validated_signals"][0]["validation"]["cache_hit"] is True


def test_material_news_requires_review_and_capacity_defers_without_bad_label():
    response = MagicMock()
    response.content = (
        '{"context_reviews":[{"signal_id":"A","state":"NEUTRAL","confidence":0.7,'
        '"supporting_factors":[],"risk_factors":[],"unresolved_conflicts":[],'
        '"handling":"CONTINUE"}]}'
    )
    state = _state(
        _candidate("A", opposing_strategies=["mean_reversion"], conflict_level="HIGH"),
        _candidate("B", opposing_strategies=["mean_reversion"], conflict_level="HIGH"),
    )
    state["shared_market_context"]["major_market_event_risk"] = "HIGH"
    result, _invoke = _run_validation(state, response, capacity=1)

    assert result["llm_optimization_metrics"]["qwen_reviews_requested"] == 2
    assert result["llm_optimization_metrics"]["qwen_reviews_deferred"] == 1
    deferred = next(
        signal
        for signal in result["rejected_signals"]
        if signal["validation"]["decision"] == "deferred"
    )
    assert deferred["validation"]["reason_codes"] == ["CONTEXT_REVIEW_DEFERRED"]


def test_qwen_failure_is_not_approval():
    candidate = _candidate(opposing_strategies=["mean_reversion"], conflict_level="HIGH")
    result, _invoke = _run_validation(_state(candidate), error=RuntimeError("offline"))
    assert result["validated_signals"] == []
    assert result["rejected_signals"][0]["validation"]["handling"] == "REQUIRE_REVIEW"


def test_context_packet_is_compact_and_schema_enforced():
    candidate = _candidate(
        opposing_strategies=["mean_reversion"],
        conflict_level="HIGH",
        raw_candles=[{"open": 1}] * 5000,
        historical_dataframe="must-not-leak",
    )
    state = _state(candidate)
    packet = build_context_packet(candidate, state)
    serialized = serialize_context_packets([candidate], state, max_input_tokens=900)
    assert "raw_candles" not in packet
    assert "must-not-leak" not in serialized

    parsed = parse_context_reviews(
        '{"context_reviews":[{"signal_id":"REL-1","state":"SUPPORTIVE",'
        '"confidence":0.8,"supporting_factors":[],"risk_factors":[],'
        '"unresolved_conflicts":[],"handling":"CONTINUE"}]}',
        [candidate],
    )
    assert parsed["REL-1"].state is ContextReviewState.SUPPORTIVE
    assert parsed["REL-1"].handling is ContextHandling.CONTINUE
    with pytest.raises(ValueError):
        parse_context_reviews('{"context_reviews":[]}', [candidate])


def test_policy_uses_need_for_context_not_rank():
    policy = QwenReviewPolicy()
    assert not policy.assess(
        _candidate(), regime_confidence=0.9, shared_context={"major_market_event_risk": "NONE"}
    ).required
    assert policy.assess(
        _candidate(opposing_strategies=["mean_reversion"], conflict_level="HIGH"),
        regime_confidence=0.9,
        shared_context={"major_market_event_risk": "NONE"},
    ).required
