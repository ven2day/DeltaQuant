from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.risk_compliance import RiskLimits, check_kill_switch, risk_compliance_node
from src.agents.signal_validation import signal_validation_node
from src.agents.state import create_initial_state
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    RegimePerformance,
    RegimePolicy,
    StrategyEligibility,
    StrategyEligibilityRegistry,
    migrate_legacy_strategy_registry,
)
from src.backtesting.strategy_registry import StrategyRegistry, StrategyVersion
from src.core.candidates import SignalEngine, StrategyType
from src.core.indicators import Timeframe


def _record(
    *,
    status: EligibilityStatus,
    strategy: str = "momentum",
    timeframe: str = "15m",
    model_version: str = "model-v7",
    minimum_model_confidence: float = 0.0,
    minimum_strategy_confidence: float = 0.0,
    requires_ml: bool = False,
    regime_performance: dict[str, RegimePerformance] | None = None,
) -> StrategyEligibility:
    return StrategyEligibility(
        strategy_name=strategy,
        timeframe=timeframe,
        model_version=model_version,
        validation_status=status,
        validated_at="2026-08-01T00:00:00+00:00",
        validation_window={"dataset": "fixture", "start": "2024-01-01", "end": "2026-01-01"},
        oos_trade_count=120,
        oos_profit_factor=1.42,
        oos_max_drawdown=0.08,
        oos_win_rate=0.58,
        minimum_model_confidence=minimum_model_confidence,
        minimum_strategy_confidence=minimum_strategy_confidence,
        artifact_reference="model-v7.joblib" if requires_ml else None,
        requires_ml=requires_ml,
        regime_performance=regime_performance or {},
    )


def _registry(tmp_path, record: StrategyEligibility) -> StrategyEligibilityRegistry:
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(record)
    return registry


def _decision(
    registry: StrategyEligibilityRegistry,
    environment: EligibilityEnvironment,
    *,
    regime: str = "trending_up",
    confidence: float = 0.8,
    model_version: str = "model-v7",
):
    return registry.evaluate(
        strategy_name="momentum",
        timeframe="15m",
        model_version=model_version,
        environment=environment,
        current_regime=regime,
        confidence=confidence,
        action="BUY",
    )


def test_paper_runtime_selects_latest_approved_champion_not_newer_unvalidated(
    tmp_path,
) -> None:
    registry = StrategyEligibilityRegistry(tmp_path)
    approved = replace(
        _record(status=EligibilityStatus.PAPER_APPROVED, model_version="approved-v1"),
        validated_at="2026-08-01T00:00:00+00:00",
    )
    challenger = replace(
        _record(status=EligibilityStatus.UNVALIDATED, model_version="challenger-v2"),
        validated_at="2026-08-02T00:00:00+00:00",
    )
    registry.register(approved)
    registry.register(challenger)

    decision = registry.evaluate(
        strategy_name=" MOMENTUM ",
        timeframe="15M",
        model_version="",
        environment=EligibilityEnvironment.PAPER,
        current_regime="trending_up",
        confidence=0.8,
    )

    assert decision.pipeline_allowed
    assert decision.execution_allowed
    assert decision.model_version == "approved-v1"
    assert decision.validation_status is EligibilityStatus.PAPER_APPROVED
    assert registry.get("momentum", "15m", "challenger-v2").validation_status is (
        EligibilityStatus.UNVALIDATED
    )


def test_explicit_wrong_version_still_fails_closed_when_approved_champion_exists(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.PAPER_APPROVED))

    decision = registry.evaluate(
        strategy_name="momentum",
        timeframe="15m",
        model_version="wrong-version",
        environment=EligibilityEnvironment.PAPER,
        current_regime="trending_up",
        confidence=0.8,
    )

    assert not decision.pipeline_allowed
    assert decision.reason_code == "MODEL_VERSION_MISMATCH"


def _risk_state(*, execution_mode: str = "live", daily_pnl: float = 0.0):
    state = create_initial_state(execution_mode=execution_mode)
    state["regime"] = "trending_up"
    state["portfolio"] = {"capital": 100_000.0, "positions": []}
    state["daily_stats"] = {
        "trades_count": 0,
        "profit_loss": daily_pnl,
        "max_drawdown": 0.0,
        "simulation_mode": execution_mode != "live",
    }
    state["validated_signals"] = [
        {
            "symbol": "RELIANCE",
            "signal_type": "BUY",
            "strategy": "momentum",
            "timeframe": "15m",
            "model_version": "model-v7",
            "execution_mode": execution_mode,
            "confidence": 0.8,
            "original_confidence": 0.8,
            "position_size_pct": 1.0,
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "target_price": 104.0,
            "risk_reward_ratio": 2.0,
            "validation": {"decision": "approve", "context_state": "SUPPORTIVE"},
        }
    ]
    return state


def _run_risk(state, registry: StrategyEligibilityRegistry, *, trading_mode: str = "live"):
    settings = SimpleNamespace(execution_mode=trading_mode, trading_mode=trading_mode)
    limits = RiskLimits(force_trading_window=True)
    with (
        patch("src.agents.risk_compliance.get_settings", return_value=settings),
        patch(
            "src.agents.risk_compliance.eligibility_registry_from_settings",
            return_value=registry,
        ),
        patch("src.agents.risk_compliance.RiskLimits.from_settings", return_value=limits),
    ):
        return risk_compliance_node(state)


def test_live_approved_allowed_regime_reaches_risk_compliance(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.LIVE_APPROVED))

    result = _run_risk(_risk_state(), registry)

    assert len(result["approved_trades"]) == 1
    assert result["approved_trades"][0]["registry_decision"] == "ALLOW"


def test_live_strategy_blocked_regime_stops_before_qwen(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        _record(
            status=EligibilityStatus.LIVE_APPROVED,
            regime_performance={
                "high_volatility": RegimePerformance(
                    policy=RegimePolicy.BLOCK, confidence_multiplier=0.0
                )
            },
        ),
    )

    decision = _decision(registry, EligibilityEnvironment.LIVE, regime="high_volatility")

    assert not decision.pipeline_allowed
    assert not decision.qwen_allowed
    assert decision.regime_policy is RegimePolicy.BLOCK


def test_paper_approved_strategy_allowed_in_paper(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.PAPER_APPROVED))
    decision = _decision(registry, EligibilityEnvironment.PAPER)
    assert decision.execution_allowed


def test_paper_approved_strategy_blocked_in_live(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.PAPER_APPROVED))
    decision = _decision(registry, EligibilityEnvironment.LIVE)
    assert not decision.pipeline_allowed
    assert decision.reason_code == "STRATEGY_NOT_LIVE_APPROVED"


def test_paper_gate_disabled_allows_unvalidated_strategy_in_paper(tmp_path) -> None:
    # STRATEGY_ELIGIBILITY_PAPER_GATE_ENABLED=false is an explicit opt-out of the
    # walk-forward/Bonferroni proof requirement for PAPER only -- an UNVALIDATED record
    # (never proven on OOS data) must be allowed to execute once the gate is off.
    registry = _registry(tmp_path, _record(status=EligibilityStatus.UNVALIDATED))
    settings = SimpleNamespace(strategy_eligibility_paper_gate_enabled=False)
    with patch("src.backtesting.strategy_eligibility.get_settings", return_value=settings):
        decision = _decision(registry, EligibilityEnvironment.PAPER)
    assert decision.execution_allowed
    assert not decision.record_only
    # The generic "pipeline_allowed" reason-code collapse (same one every other allowed
    # path funnels through) wins over the specific GATE_DISABLED reason from
    # _status_policy -- execution_allowed/record_only above are the real behavioral proof.
    assert decision.reason_code == "STRATEGY_ELIGIBLE"


def test_paper_gate_disabled_never_relaxes_live(tmp_path) -> None:
    # The gate flag is documented as PAPER-only. Confirm it cannot be used, even by
    # accident, to let an unproven strategy reach a real broker order in LIVE.
    registry = _registry(tmp_path, _record(status=EligibilityStatus.UNVALIDATED))
    settings = SimpleNamespace(strategy_eligibility_paper_gate_enabled=False)
    with patch("src.backtesting.strategy_eligibility.get_settings", return_value=settings):
        decision = _decision(registry, EligibilityEnvironment.LIVE)
    assert not decision.pipeline_allowed
    assert not decision.execution_allowed
    assert decision.reason_code == "STRATEGY_NOT_LIVE_APPROVED"


def test_shadow_strategy_in_simulated_mode_is_evaluated_but_not_executable(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.SHADOW))
    decision = _decision(registry, EligibilityEnvironment.SIMULATED)
    assert decision.pipeline_allowed
    assert decision.record_only
    assert decision.shadow_only
    assert not decision.qwen_allowed
    assert not decision.execution_allowed


def test_missing_registry_record_in_live_fails_closed(tmp_path) -> None:
    decision = _decision(StrategyEligibilityRegistry(tmp_path), EligibilityEnvironment.LIVE)
    assert not decision.pipeline_allowed
    assert decision.reason_code == "ELIGIBILITY_REGISTRY_MISSING"


def test_missing_registry_record_in_research_continues_and_is_recorded(tmp_path) -> None:
    decision = _decision(StrategyEligibilityRegistry(tmp_path), EligibilityEnvironment.RESEARCH)
    assert decision.pipeline_allowed
    assert decision.record_only
    assert not decision.execution_allowed
    assert decision.reason_code == "ELIGIBILITY_REGISTRY_MISSING_RESEARCH"


def test_reduce_confidence_applies_exact_multiplier(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        _record(
            status=EligibilityStatus.LIVE_APPROVED,
            regime_performance={
                "ranging": RegimePerformance(
                    policy=RegimePolicy.REDUCE_CONFIDENCE,
                    confidence_multiplier=0.8,
                )
            },
        ),
    )
    decision = _decision(
        registry, EligibilityEnvironment.LIVE, regime="ranging", confidence=0.78
    )
    assert decision.adjusted_confidence == pytest.approx(0.624)
    assert decision.pipeline_allowed


def test_reduced_confidence_below_strategy_floor_is_rejected(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        _record(
            status=EligibilityStatus.LIVE_APPROVED,
            minimum_strategy_confidence=0.65,
            regime_performance={
                "ranging": RegimePerformance(
                    policy=RegimePolicy.REDUCE_CONFIDENCE,
                    confidence_multiplier=0.8,
                )
            },
        ),
    )
    decision = _decision(
        registry, EligibilityEnvironment.LIVE, regime="ranging", confidence=0.78
    )
    assert not decision.pipeline_allowed
    assert decision.reason_code == "REGIME_ADJUSTED_CONFIDENCE_BELOW_MINIMUM"


def test_ml_threshold_is_applied_after_registry_and_regime(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        _record(
            status=EligibilityStatus.LIVE_APPROVED,
            minimum_model_confidence=0.7,
            requires_ml=True,
        ),
    )
    decision = _decision(registry, EligibilityEnvironment.LIVE)
    assert decision.pipeline_allowed
    assert registry.ml_qualified(decision, 0.69) == (False, "ML_BELOW_STRATEGY_MINIMUM")
    assert registry.ml_qualified(decision, 0.71) == (True, "ML_QUALIFIED")


def test_regime_change_reuses_same_model_artifact(tmp_path) -> None:
    registry = _registry(
        tmp_path,
        _record(
            status=EligibilityStatus.LIVE_APPROVED,
            regime_performance={
                "ranging": RegimePerformance(
                    policy=RegimePolicy.REDUCE_CONFIDENCE,
                    confidence_multiplier=0.75,
                )
            },
        ),
    )
    trend = _decision(registry, EligibilityEnvironment.LIVE, regime="trending_up")
    ranging = _decision(registry, EligibilityEnvironment.LIVE, regime="ranging")
    assert trend.model_version == ranging.model_version == "model-v7"
    assert trend.eligibility is ranging.eligibility


def test_qwen_cannot_override_registry_rejection(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.PAPER_APPROVED))
    decision = _decision(registry, EligibilityEnvironment.LIVE)
    signal = {
        "symbol": "RELIANCE",
        "strategy": "momentum",
        "timeframe": "15m",
        "validation": {"decision": "approve", "context_state": "SUPPORTIVE"},
        **decision.to_dict(),
    }
    state = create_initial_state(execution_mode="live")
    state["signals"] = [signal]
    with (
        patch(
            "src.agents.signal_validation.get_settings",
            return_value=SimpleNamespace(enable_llm_agents=True),
        ),
        patch("src.agents.signal_validation._optimized_signal_validation") as review,
    ):
        result = signal_validation_node(state)
    review.assert_not_called()
    assert result["validated_signals"] == []
    assert result["rejected_signals"] == [signal]


def test_qwen_supportive_cannot_override_risk_rejection(tmp_path) -> None:
    registry = _registry(tmp_path, _record(status=EligibilityStatus.LIVE_APPROVED))
    result = _run_risk(_risk_state(daily_pnl=-10_000.0), registry)
    assert result["approved_trades"] == []
    failures = result["risk_rejected"][0]["risk_result"]["failures"]
    assert any(item["rule"] == "daily_loss_limit" for item in failures)


def test_legacy_conflicts_migrate_without_live_approval(tmp_path) -> None:
    legacy = StrategyRegistry(tmp_path / "legacy")
    now = datetime.now(UTC)
    common = dict(
        strategy_name="momentum",
        version="legacy-v1",
        owner="test",
        parameters={},
        approved_universe=(),
        dataset_id="legacy",
        validated_at=now.isoformat(),
        expires_at=(now + timedelta(days=30)).isoformat(),
        verdict="VALIDATED",
        reasons=(),
        oos_trades=100,
        oos_expectancy=0.2,
        oos_return_pct=8.0,
        fold_consistency=0.8,
        timeframe="15m",
    )
    legacy.register(
        StrategyVersion(
            **common,
            approved_regimes=("trending_up",),
            trade_horizon="SWING",
        )
    )
    legacy.register(
        StrategyVersion(
            **common,
            approved_regimes=("ranging",),
            trade_horizon="SCALP",
        )
    )
    target = StrategyEligibilityRegistry(tmp_path / "eligibility")

    report = migrate_legacy_strategy_registry(
        legacy, target, runtime_timeframes=("15m",)
    )

    assert report.conflicts == 1
    migrated = target.get("momentum", "15m", "legacy-v1")
    assert migrated is not None
    assert migrated.validation_status is EligibilityStatus.SHADOW
    assert migrated.validation_status is not EligibilityStatus.LIVE_APPROVED
    assert "conservatively" in migrated.status_reason


def test_buy_sell_candidate_generation_is_independent_of_registry(tmp_path) -> None:
    indicator = MagicMock()
    indicator.symbol = "RELIANCE"
    indicator.timeframe = Timeframe.M15
    indicator.close = 100.0
    indicator.rsi = 35.0
    indicator.macd_histogram = 1.0
    indicator.plus_di = 30.0
    indicator.minus_di = 10.0
    indicator.ema = {21: 99.0}
    indicator.sma = {}
    indicator.atr = 1.0
    indicator.to_dict.return_value = {}
    engine = SignalEngine()
    before = engine.generate_signals(indicator, [StrategyType.MOMENTUM])

    _registry(tmp_path, _record(status=EligibilityStatus.DISABLED))
    after = engine.generate_signals(indicator, [StrategyType.MOMENTUM])

    assert len(before) == len(after) == 1
    assert before[0].signal_type == after[0].signal_type
    assert before[0].strategy == after[0].strategy
    assert before[0].confidence == after[0].confidence


def test_existing_kill_switch_remains_authoritative() -> None:
    state = create_initial_state()
    state["daily_stats"] = {"profit_loss": -10_001.0, "trades_count": 0}
    assert check_kill_switch(state, RiskLimits(max_daily_loss=10_000.0))
