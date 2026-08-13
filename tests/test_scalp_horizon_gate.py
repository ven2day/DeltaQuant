"""SCALP keeps horizon-specific risk thresholds without horizon-specific eligibility."""

from unittest.mock import patch

from src.agents.risk_compliance import RiskLimits
from src.agents.signal_validation import _fallback_signal_validation
from src.agents.state import create_initial_state
from src.backtesting.strategy_eligibility import (
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.config.settings import Settings


def _paper_eligibility(tmp_path, strategy_name="momentum"):
    record = StrategyEligibility(
        strategy_name=strategy_name,
        timeframe="5m",
        model_version="model-v1",
        validation_status=EligibilityStatus.PAPER_APPROVED,
        validated_at="2026-08-01T00:00:00+00:00",
        validation_window={"dataset": "test-dataset"},
        oos_trade_count=100,
        oos_profit_factor=1.4,
        oos_max_drawdown=0.08,
        oos_win_rate=0.58,
        minimum_model_confidence=0.0,
    )
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(record)
    return registry


def test_same_strategy_timeframe_model_eligibility_serves_scalp_and_swing(tmp_path):
    registry = _paper_eligibility(tmp_path)
    for horizon in ("SCALP", "SWING"):
        decision = registry.evaluate(
            strategy_name="momentum",
            timeframe="5m",
            model_version="model-v1",
            environment="PAPER",
            current_regime="trending_up",
            confidence=0.8,
        )
        assert decision.execution_allowed, horizon


def test_missing_strategy_eligibility_still_fails_closed_in_paper(tmp_path):
    # Explicit gate-enabled settings, independent of whatever the real environment's
    # STRATEGY_ELIGIBILITY_PAPER_GATE_ENABLED happens to be set to right now -- this
    # test is specifically about the fail-closed default, not the toggle.
    settings = _real_settings(strategy_eligibility_paper_gate_enabled=True)
    with patch("src.backtesting.strategy_eligibility.get_settings", return_value=settings):
        decision = StrategyEligibilityRegistry(tmp_path).evaluate(
            strategy_name="momentum",
            timeframe="5m",
            model_version="model-v1",
            environment="PAPER",
            current_regime="trending_up",
            confidence=0.99,
        )
    assert not decision.pipeline_allowed
    assert decision.reason_code == "ELIGIBILITY_REGISTRY_MISSING"


# ---------------------------------------------------------------------------
# RiskLimits.from_settings(trade_horizon=...) -- differs only in max_position_size_pct
# ---------------------------------------------------------------------------


def _real_settings(**overrides) -> Settings:
    # _env_file=None: assert behavior of explicit kwargs/class defaults, not
    # whatever the repo's actual .env happens to contain (see
    # test_config_failclosed.py's _base_kwargs for the same rationale).
    base = dict(groq_api_key="x", _env_file=None)
    base.update(overrides)
    return Settings(**base)


def test_risk_limits_scalp_horizon_differs_only_in_position_size(monkeypatch):
    settings = _real_settings()
    with patch("src.agents.risk_compliance.get_settings", return_value=settings):
        swing_limits = RiskLimits.from_settings(trade_horizon="SWING")
        scalp_limits = RiskLimits.from_settings(trade_horizon="SCALP")

    assert scalp_limits.max_position_size_pct == settings.scalp_max_position_pct * 100
    assert swing_limits.max_position_size_pct == settings.max_position_pct * 100
    assert scalp_limits.max_position_size_pct < swing_limits.max_position_size_pct

    # Every other field must be identical -- horizon only ever touches position size.
    for field_name in (
        "max_total_exposure_pct",
        "max_positions",
        "max_daily_trades",
        "max_daily_loss",
        "no_trading_before",
        "no_trading_after",
        "force_trading_window",
        "force_window_requires_simulation_marker",
        "max_sector_exposure_pct",
        "max_pairwise_correlation",
    ):
        assert getattr(swing_limits, field_name) == getattr(scalp_limits, field_name), field_name


def test_risk_limits_default_horizon_is_swing():
    settings = _real_settings()
    with patch("src.agents.risk_compliance.get_settings", return_value=settings):
        default_limits = RiskLimits.from_settings()
        explicit_swing_limits = RiskLimits.from_settings(trade_horizon="SWING")

    assert default_limits.max_position_size_pct == explicit_swing_limits.max_position_size_pct


# ---------------------------------------------------------------------------
# signal_validation fallback: horizon-selected thresholds, scalp default == swing bar
# ---------------------------------------------------------------------------


def test_fallback_validation_scalp_thresholds_default_equal_to_swing_hardcoded_bar():
    settings = _real_settings()
    state = create_initial_state(trade_horizon="SCALP")
    state["active_strategies"] = ["momentum"]
    signals = [
        {
            "signal_id": "1",
            "strategy": "momentum",
            "confidence": 0.6,  # exactly the swing bar
            "risk_reward_ratio": 1.5,  # exactly the swing bar
            "trade_horizon": "SCALP",
        }
    ]

    with patch("src.agents.signal_validation.get_settings", return_value=settings):
        result = _fallback_signal_validation(state, signals, "test")

    # scalp_min_confidence/scalp_min_rr default equal to 0.6/1.5 -- must still pass.
    assert len(result["validated_signals"]) == 1
    assert result["rejected_signals"] == []


def test_fallback_validation_is_horizon_aware_not_globally_looser():
    """A signal tagged SCALP is checked against scalp_min_rr/scalp_min_confidence;
    a signal tagged SWING (or untagged) in the SAME batch still uses the swing
    hardcoded 0.6/1.5 -- proving this is per-signal horizon selection, not a global
    settings swap that would loosen swing's bar too."""
    settings = _real_settings(scalp_min_rr=1.0, scalp_min_confidence=0.4)
    state = create_initial_state(trade_horizon="SWING")
    state["active_strategies"] = ["momentum"]
    signals = [
        {
            "signal_id": "swing-1",
            "strategy": "momentum",
            "confidence": 0.5,  # below swing's 0.6 bar
            "risk_reward_ratio": 1.5,
            # no trade_horizon key -> falls back to state's SWING
        },
        {
            "signal_id": "scalp-1",
            "strategy": "momentum",
            "confidence": 0.5,  # above the (loosened, for this test) scalp 0.4 bar
            "risk_reward_ratio": 1.5,
            "trade_horizon": "SCALP",
        },
    ]

    with patch("src.agents.signal_validation.get_settings", return_value=settings):
        result = _fallback_signal_validation(state, signals, "test")

    validated_ids = {s["signal_id"] for s in result["validated_signals"]}
    rejected_ids = {s["signal_id"] for s in result["rejected_signals"]}
    assert validated_ids == {"scalp-1"}
    assert rejected_ids == {"swing-1"}


def test_fallback_validation_defaults_untagged_signals_to_swing():
    settings = _real_settings()
    state = create_initial_state()  # trade_horizon defaults to SWING
    state["active_strategies"] = ["momentum"]
    signals = [
        {
            "signal_id": "1",
            "strategy": "momentum",
            "confidence": 0.6,
            "risk_reward_ratio": 1.5,
            # no trade_horizon key at all -- must use state's default (SWING)
        }
    ]

    with patch("src.agents.signal_validation.get_settings", return_value=settings):
        result = _fallback_signal_validation(state, signals, "test")

    assert len(result["validated_signals"]) == 1
