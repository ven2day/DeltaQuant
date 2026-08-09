"""
Stage 11 acceptance tests: the H-8 admission gate and signal_validation/
risk_compliance thresholds are horizon-aware, and specifically verify the two named
risk scenarios from the scalping-feature requirements never happen:

1. H-8 must never be bypassed for scalp -- a SWING-validated artifact (or no
   artifact at all) must not admit a SCALP-tagged signal, regardless of how strong
   its entry quality/ranking evidence is.
2. signal_validation's rr_ratio/confidence fallback bar must be horizon-selectable
   but never silently lowered for scalp -- scalp defaults ship numerically equal to
   the swing hardcoded 1.5/0.6.
"""

from unittest.mock import patch

from src.agents.risk_compliance import RiskLimits, risk_compliance_node
from src.agents.signal_validation import _fallback_signal_validation
from src.agents.state import create_initial_state
from src.backtesting.strategy_registry import StrategyRegistry, build_strategy_version
from src.config.settings import Settings


def _validated_swing_artifact(tmp_path, strategy_name="momentum"):
    """A current, VALIDATED artifact for the SWING horizon only (the default) --
    exactly what scripts/validate_strategy.py has always produced pre-Stage-2."""
    version = build_strategy_version(
        strategy_name,
        owner="test",
        dataset_id="test-dataset",
        oos_trades=100,
        oos_expectancy=2.0,
        oos_return_pct=10.0,
        fold_consistency=0.8,
    )
    StrategyRegistry(tmp_path).register(version)


def test_scalp_signal_blocked_at_h8_with_no_scalp_artifact_regardless_of_quality(tmp_path):
    """The core acceptance test: a scalp signal with excellent entry-quality
    evidence (high confidence, tight risk-reward) must still be rejected at H-8
    check #14 when no SCALP-horizon registry artifact exists -- no amount of local
    evidence can substitute for admission."""
    _validated_swing_artifact(tmp_path, "momentum")  # SWING only, no SCALP artifact

    state = create_initial_state(trade_horizon="SCALP")
    state["regime"] = "trending_up"
    state["validated_signals"] = [
        {
            "symbol": "RELIANCE",
            "strategy": "momentum",
            "timeframe": "5m",
            "trade_horizon": "SCALP",
            "confidence": 0.99,  # deliberately excellent evidence
            "risk_reward_ratio": 5.0,
        }
    ]

    settings_mock = type("S", (), {"strategy_registry_dir": str(tmp_path)})()
    with (
        patch("src.agents.risk_compliance.now_ist") as mock_now,
        patch(
            "src.agents.risk_compliance.RiskLimits.from_settings",
            return_value=RiskLimits(),
        ),
        patch("src.agents.risk_compliance.get_settings", return_value=settings_mock),
    ):
        mock_now.return_value.strftime.return_value = "12:00"
        result = risk_compliance_node(state)

    assert result["approved_trades"] == []
    failures = result["risk_rejected"][0]["risk_result"]["failures"]
    admission_failure = next(f for f in failures if f["rule"] == "strategy_admission")
    assert "trade_horizon 'SCALP'" in admission_failure["message"]


def test_swing_signal_for_same_strategy_still_admitted(tmp_path):
    """The flip side: the exact same registry state must keep admitting the SWING
    signal it always has -- proving Stage 11 didn't collaterally break swing."""
    _validated_swing_artifact(tmp_path, "momentum")

    state = create_initial_state(trade_horizon="SWING")
    state["regime"] = "trending_up"
    state["validated_signals"] = [
        {
            "symbol": "RELIANCE",
            "strategy": "momentum",
            "timeframe": "1h",
            "confidence": 0.8,
            "risk_reward_ratio": 2.0,
        }
    ]

    settings_mock = type("S", (), {"strategy_registry_dir": str(tmp_path)})()
    with (
        patch("src.agents.risk_compliance.now_ist") as mock_now,
        patch(
            "src.agents.risk_compliance.RiskLimits.from_settings",
            return_value=RiskLimits(),
        ),
        patch("src.agents.risk_compliance.get_settings", return_value=settings_mock),
    ):
        mock_now.return_value.strftime.return_value = "12:00"
        result = risk_compliance_node(state)

    assert len(result["approved_trades"]) == 1
    assert result["risk_rejected"] == []


def test_scalp_admitted_once_a_matching_scalp_artifact_exists(tmp_path):
    """Confirms the gate isn't just permanently closed for SCALP -- once a genuine
    SCALP/5m VALIDATED artifact is registered (Stage 14's operational step), a
    matching signal clears H-8 exactly like swing always has."""
    version = build_strategy_version(
        "momentum",
        owner="test",
        dataset_id="test-dataset-5m",
        oos_trades=100,
        oos_expectancy=2.0,
        oos_return_pct=10.0,
        fold_consistency=0.8,
        timeframe="5m",
        trade_horizon="SCALP",
    )
    StrategyRegistry(tmp_path).register(version)

    state = create_initial_state(trade_horizon="SCALP")
    state["regime"] = "trending_up"
    state["validated_signals"] = [
        {
            "symbol": "RELIANCE",
            "strategy": "momentum",
            "timeframe": "5m",
            "trade_horizon": "SCALP",
            "confidence": 0.8,
            "risk_reward_ratio": 2.0,
        }
    ]

    settings_mock = type("S", (), {"strategy_registry_dir": str(tmp_path)})()
    with (
        patch("src.agents.risk_compliance.now_ist") as mock_now,
        patch(
            "src.agents.risk_compliance.RiskLimits.from_settings",
            return_value=RiskLimits(),
        ),
        patch("src.agents.risk_compliance.get_settings", return_value=settings_mock),
    ):
        mock_now.return_value.strftime.return_value = "12:00"
        result = risk_compliance_node(state)

    assert len(result["approved_trades"]) == 1


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
