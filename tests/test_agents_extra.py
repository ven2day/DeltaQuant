from unittest.mock import MagicMock, patch

from src.agents.market_regime import (
    _build_regime_context,
    _parse_regime_response,
    market_regime_node,
)
from src.agents.risk_compliance import (
    RiskLimits,
    check_kill_switch,
    compute_return_correlations,
    risk_compliance_node,
)
from src.agents.signal_validation import (
    _build_validation_context,
    _parse_validation_response,
    signal_validation_node,
)
from src.agents.state import MarketRegime, create_initial_state
from src.agents.strategy_selection import (
    _build_strategy_context,
    _parse_strategy_response,
    strategy_selection_node,
)

# Mock settings
with patch("src.config.get_settings") as mock_get_settings:
    mock_settings = MagicMock()
    mock_settings.groq_api_key.get_secret_value.return_value = "token"
    mock_settings.groq_model_primary = "llama"
    mock_get_settings.return_value = mock_settings

# --- Market Regime Tests ---


def test_build_regime_context():
    indicators = {"A": {"trend": {"adx": 30}}}
    market_data = {"A": {"change_percent": 1.0}}
    lessons = [{"severity": "high", "description": "mistake"}]

    context = _build_regime_context(indicators, market_data, lessons)
    assert "ADX" in context
    assert "mistake" in context


def test_parse_regime_response():
    # Valid
    content = '{"regime": "trending_up", "confidence": 0.8, "reasoning": "trend"}'
    res = _parse_regime_response(content)
    assert res["regime"] == "trending_up"

    # Invalid regime
    content = '{"regime": "invalid", "confidence": 0.8}'
    res = _parse_regime_response(content)
    assert res["regime"] == MarketRegime.UNKNOWN.value

    # Invalid JSON
    res = _parse_regime_response("invalid json")
    assert res["regime"] == MarketRegime.UNKNOWN.value


@patch("src.agents.market_regime.ChatGroq")
@patch("src.agents.market_regime.get_settings")
def test_market_regime_node(mock_settings, mock_llm_cls):
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"regime": "ranging", "confidence": 0.6, "reasoning": "flat"}'
    )
    mock_llm_cls.return_value = mock_llm

    state = create_initial_state()
    result = market_regime_node(state)

    assert result["regime"] == "ranging"
    assert result["regime_confidence"] == 0.6


@patch("src.agents.market_regime.ChatGroq")
@patch("src.agents.market_regime.get_settings")
def test_market_regime_node_skips_llm_when_disabled(mock_settings, mock_llm_cls):
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["market_data"] = {"A": {"change_percent": 1.0}}

    result = market_regime_node(state)

    mock_llm_cls.assert_not_called()
    assert result["regime"] == "trending_up"


@patch("src.agents.market_regime.ChatGroq")
def test_market_regime_node_error(mock_llm_cls):
    mock_llm_cls.side_effect = Exception("API Error")

    state = create_initial_state()
    state["market_data"] = {"A": {"change_percent": 1.0}}  # Positive -> trending_up fallback

    with patch("src.agents.market_regime.get_settings"):
        result = market_regime_node(state)

    assert result["regime"] == "trending_up"
    assert "fallback" in result["regime_reasoning"].lower()


# --- Risk Compliance Tests ---


def test_risk_compliance_node():
    state = create_initial_state()
    state["validated_signals"] = [{"symbol": "A", "confidence": 0.8, "risk_reward_ratio": 2.0}]

    # Mock IST clock to be within trading hours
    with patch("src.agents.risk_compliance.now_ist") as mock_now:
        mock_now.return_value.strftime.return_value = "12:00"

        with patch("src.agents.risk_compliance.RiskLimits.from_settings") as mock_limits:
            limits = RiskLimits()  # Defaults
            mock_limits.return_value = limits

            result = risk_compliance_node(state)

            assert len(result["approved_trades"]) == 1
            assert result["approved_trades"][0]["risk_result"]["approved"] is True


def test_risk_compliance_force_trading_window_bypasses_hours_check():
    # Regression guard for the testing-only override: outside trading hours, a signal
    # is normally blocked by the trading_hours rule; force_trading_window must let it
    # through so the full pipeline can be exercised end-to-end while markets are closed.
    state = create_initial_state()
    state["validated_signals"] = [{"symbol": "A", "confidence": 0.8, "risk_reward_ratio": 2.0}]

    with patch("src.agents.risk_compliance.now_ist") as mock_now:
        mock_now.return_value.strftime.return_value = "20:00"  # well outside 09:15-15:15

        with patch("src.agents.risk_compliance.RiskLimits.from_settings") as mock_limits:
            mock_limits.return_value = RiskLimits()  # force_trading_window=False (default)
            blocked = risk_compliance_node(state)
            assert blocked["approved_trades"] == []

        with patch("src.agents.risk_compliance.RiskLimits.from_settings") as mock_limits:
            mock_limits.return_value = RiskLimits(force_trading_window=True)
            approved = risk_compliance_node(state)
            assert len(approved["approved_trades"]) == 1


def test_risk_compliance_blocking():
    state = create_initial_state()
    # Daily trade limit exceeded
    state["daily_stats"]["trades_count"] = 100
    state["validated_signals"] = [{"symbol": "A"}]

    with patch("src.agents.risk_compliance.RiskLimits.from_settings") as mock_limits:
        limits = RiskLimits(max_daily_trades=50)
        mock_limits.return_value = limits

        result = risk_compliance_node(state)

        assert len(result["approved_trades"]) == 0
        assert len(result["risk_rejected"]) == 1


def test_compute_return_correlations():
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base = pd.Series(np.linspace(100, 140, 40), index=dates)

    # B tracks A almost exactly (highly correlated); C is independent noise.
    rng = np.random.default_rng(42)
    price_histories = {
        "A": base,
        "B": base * 1.01,
        "C": pd.Series(100 + rng.normal(0, 1, 40).cumsum(), index=dates),
        "SHORT": pd.Series([100.0, 101.0]),  # too little history, must be excluded
    }

    correlations = compute_return_correlations(price_histories, min_periods=5)

    assert correlations["A"]["B"] > 0.95
    assert "C" in correlations["A"]  # computed, just not necessarily high
    assert "SHORT" not in correlations


def test_risk_compliance_pairwise_correlation_blocks():
    # A already holds RELIANCE; a new candidate that is 0.95-correlated with it must be
    # REJECTED even though they're in different sectors (sector caps alone would miss
    # this). H-3 (DeltaQuant-Quant-Risk-Review.md): pairwise_correlation was hardened
    # from a warning to a block -- a correlated-cluster cap that can't actually block
    # let the system pyramid the same macro bet under a formally "risk approved" trade.
    state = create_initial_state()
    state["validated_signals"] = [{"symbol": "CANDIDATE", "confidence": 0.8, "risk_reward_ratio": 2.0}]
    state["portfolio"] = {
        "capital": 1_000_000.0,
        "positions": [{"symbol": "RELIANCE", "quantity": 10, "entry_price": 2500}],
        "return_correlations": {"CANDIDATE": {"RELIANCE": 0.95}},
    }

    with patch("src.agents.risk_compliance.now_ist") as mock_now:
        mock_now.return_value.strftime.return_value = "12:00"
        with patch("src.agents.risk_compliance.RiskLimits.from_settings") as mock_limits:
            mock_limits.return_value = RiskLimits(max_pairwise_correlation=0.80)
            result = risk_compliance_node(state)

    assert len(result["approved_trades"]) == 0
    assert len(result["risk_rejected"]) == 1
    rejected_rules = {f["rule"] for f in result["risk_rejected"][0]["risk_result"]["failures"]}
    assert "pairwise_correlation" in rejected_rules


def test_check_kill_switch():
    state = create_initial_state()
    limits = RiskLimits(max_daily_loss=1000)

    # Safe
    state["daily_stats"]["profit_loss"] = -500
    assert check_kill_switch(state, limits) is False

    # Breached
    state["daily_stats"]["profit_loss"] = -1500
    assert check_kill_switch(state, limits) is True


# --- Signal Validation Tests ---


def test_build_validation_context():
    signals = [{"symbol": "A", "confidence": 0.8}]
    context = _build_validation_context(signals, "bull", 0.9, ["mom"], [])
    assert "A" in context
    assert "bull" in context


def test_parse_validation_response():
    signals = [{"signal_id": "1"}, {"signal_id": "2"}]
    content = '{"validations": [{"signal_id": "1", "decision": "approve"}, {"signal_id": "2", "decision": "reject"}]}'

    res = _parse_validation_response(content, signals)
    assert len(res["validated"]) == 1
    assert res["validated"][0]["signal_id"] == "1"
    assert len(res["rejected"]) == 1


@patch("src.agents.signal_validation.ChatGroq")
@patch("src.agents.signal_validation.get_settings")
def test_signal_validation_node(mock_settings, mock_llm_cls):
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"validations": [{"signal_id": "1", "decision": "approve"}]}'
    )
    mock_llm_cls.return_value = mock_llm

    state = create_initial_state()
    state["signals"] = [{"signal_id": "1"}]

    result = signal_validation_node(state)
    assert len(result["validated_signals"]) == 1


@patch("src.agents.signal_validation.ChatGroq")
@patch("src.agents.signal_validation.get_settings")
def test_signal_validation_node_skips_llm_when_disabled(mock_settings, mock_llm_cls):
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["signals"] = [{"signal_id": "1", "strategy": "momentum", "confidence": 0.9, "risk_reward_ratio": 2.0}]
    state["active_strategies"] = ["momentum"]

    result = signal_validation_node(state)

    mock_llm_cls.assert_not_called()
    assert len(result["validated_signals"]) == 1


# --- Strategy Selection Tests ---


def test_build_strategy_context():
    context = _build_strategy_context("bull", 0.9, [], {})
    assert "bull" in context


@patch("src.memory.performance_tracker.get_performance_tracker")
def test_build_strategy_context_queries_given_namespace(mock_get_tracker):
    # H-9 regression: a mock/simulated session's LLM context must be built from ITS OWN
    # namespace, not the tracker's bare "paper_market_data" default -- otherwise a
    # simulated run's strategy-selection prompt is silently populated from unrelated
    # live-data-paper history (or vice versa).
    mock_tracker = MagicMock()
    mock_tracker.get_all_strategy_performance.return_value = {}
    mock_get_tracker.return_value = mock_tracker

    _build_strategy_context("bull", 0.9, [], {}, data_namespace="mock_simulated")

    mock_get_tracker.assert_called_once_with("mock_simulated")


@patch("src.agents.strategy_selection.ChatGroq")
@patch("src.agents.strategy_selection.get_settings")
@patch("src.memory.performance_tracker.get_performance_tracker")
def test_strategy_selection_node_threads_data_namespace_from_state(
    mock_get_tracker, mock_settings, mock_llm_cls
):
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"active_strategies": ["breakout"], "reasoning": "vol"}'
    mock_llm_cls.return_value = mock_llm
    mock_tracker = MagicMock()
    mock_tracker.get_all_strategy_performance.return_value = {}
    mock_get_tracker.return_value = mock_tracker

    state = create_initial_state(data_namespace="mock_simulated")
    strategy_selection_node(state)

    mock_get_tracker.assert_called_once_with("mock_simulated")


def test_parse_strategy_response():
    content = '{"active_strategies": ["momentum"], "reasoning": "trend"}'
    res = _parse_strategy_response(content)
    assert res["active_strategies"] == ["momentum"]

    # Invalid strategy filtered
    content = '{"active_strategies": ["invalid"], "reasoning": "none"}'
    res = _parse_strategy_response(content)
    # Defaults to trend_following if list empty
    assert "trend_following" in res["active_strategies"]


@patch("src.agents.strategy_selection.ChatGroq")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node(mock_settings, mock_llm_cls):
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"active_strategies": ["breakout"], "reasoning": "vol"}'
    mock_llm_cls.return_value = mock_llm

    state = create_initial_state()
    result = strategy_selection_node(state)

    assert result["active_strategies"] == ["breakout"]


@patch("src.agents.strategy_selection.ChatGroq")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node_skips_llm_when_disabled(mock_settings, mock_llm_cls):
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["regime"] = "trending_up"

    result = strategy_selection_node(state)

    mock_llm_cls.assert_not_called()
    assert result["active_strategies"] == ["momentum", "trend_following"]
