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


def _register_validated_strategy(
    directory, strategy_name: str, *, timeframe: str = "15m", trade_horizon: str = "SWING"
) -> None:
    """Seed one paper-approved strategy-eligibility record."""
    from src.backtesting.strategy_eligibility import (
        EligibilityStatus,
        StrategyEligibility,
        StrategyEligibilityRegistry,
    )

    record = StrategyEligibility(
        strategy_name=strategy_name,
        timeframe=timeframe,
        model_version=f"{strategy_name}-v1",
        validation_status=EligibilityStatus.PAPER_APPROVED,
        validated_at="2026-08-01T00:00:00+00:00",
        validation_window={"dataset_id": "test-dataset", "legacy_horizon": trade_horizon},
        oos_trade_count=100,
        oos_profit_factor=1.4,
        oos_max_drawdown=0.08,
        oos_win_rate=0.58,
        minimum_model_confidence=0.0,
    )
    StrategyEligibilityRegistry(directory).register(record)


def _eligible_signal(strategy: str, *, timeframe: str = "15m") -> dict:
    return {
        "symbol": "RELIANCE",
        "signal_type": "BUY",
        "strategy": strategy,
        "supporting_strategies": [strategy],
        "timeframe": timeframe,
        "confidence": 0.8,
        "risk_reward_ratio": 2.0,
        "registry_decision": "ALLOW",
        "registry_qwen_allowed": True,
    }


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


def test_parse_regime_response_extracts_bare_json_after_markdown_heading():
    """Observed from Qwen: a markdown heading before a bare JSON object (no code
    fence) used to fail to parse entirely, dumping the raw truncated response into
    the user-facing "reasoning" field. Must now parse the JSON object itself."""
    content = (
        "## Market Regime Classification\n"
        '{"regime": "ranging", "confidence": 0.5, "reasoning": "low ADX"}'
    )
    res = _parse_regime_response(content)
    assert res["regime"] == "ranging"
    assert res["confidence"] == 0.5
    assert res["reasoning"] == "low ADX"


def test_parse_regime_response_genuine_parse_failure_is_plain_english():
    """When it genuinely can't be parsed, the reasoning shown to the user must not be
    a raw dump of the model's output (reads as corrupted data)."""
    res = _parse_regime_response("not json at all, no braces here")
    assert res["regime"] == MarketRegime.UNKNOWN.value
    assert "could not be read" in res["reasoning"]
    assert "not json at all" not in res["reasoning"]


@patch("src.agents.llm_factory.create_chat_model")
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


@patch("src.agents.market_regime.create_chat_model")
@patch("src.agents.market_regime.get_settings")
def test_market_regime_node_skips_llm_when_disabled(mock_settings, mock_llm_cls):
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["market_data"] = {"A": {"change_percent": 1.0}}

    result = market_regime_node(state)

    mock_llm_cls.assert_not_called()
    assert result["regime"] == "trending_up"


@patch("src.agents.llm_factory.create_chat_model")
def test_market_regime_node_error(mock_llm_cls):
    mock_llm_cls.side_effect = Exception("API Error")

    state = create_initial_state()
    state["market_data"] = {"A": {"change_percent": 1.0}}  # Positive -> trending_up fallback

    with patch("src.agents.market_regime.get_settings"):
        result = market_regime_node(state)

    assert result["regime"] == "trending_up"
    assert "backup rule" in result["regime_reasoning"].lower()


@patch("src.agents.llm_factory.create_chat_model")
def test_market_regime_node_error_reasoning_is_plain_english(mock_llm_cls):
    """Regression guard: the fallback reasoning used to end with a raw dump of the
    exception ("... Error: <raw exception str>"). Must be readable prose instead."""
    mock_llm_cls.side_effect = Exception(
        "Service qwen_api unavailable (circuit breaker open). Retry in 30.0s"
    )
    state = create_initial_state()
    state["market_data"] = {"A": {"change_percent": 1.0}}

    with patch("src.agents.market_regime.get_settings"):
        result = market_regime_node(state)

    assert "AI review skipped because" in result["regime_reasoning"]
    assert "Error: Service qwen_api" not in result["regime_reasoning"]


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
    # this). H-3 (docs/audits/DeltaQuant-Quant-Risk-Review.md): pairwise_correlation was hardened
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


def test_parse_validation_response_extracts_bare_json_after_markdown_heading():
    signals = [{"signal_id": "1"}]
    content = '## Validation\n{"validations": [{"signal_id": "1", "decision": "approve"}]}'
    res = _parse_validation_response(content, signals)
    assert len(res["validated"]) == 1


def test_parse_validation_response_unprocessed_signal_reason_is_plain_english():
    """A signal the LLM's response never mentioned must not say the bare, unexplained
    'Not processed' -- must say why it was rejected anyway (missing from the
    response, not assumed approved)."""
    signals = [{"signal_id": "1"}, {"signal_id": "2"}]
    content = '{"validations": [{"signal_id": "1", "decision": "approve"}]}'

    res = _parse_validation_response(content, signals)

    unprocessed = [s for s in res["rejected"] if s["signal_id"] == "2"]
    assert len(unprocessed) == 1
    reason = unprocessed[0]["validation"]["reasoning"]
    assert reason != "Not processed"
    assert "didn't include a decision" in reason


def test_parse_validation_response_genuine_parse_failure_is_plain_english():
    signals = [{"signal_id": "1"}]

    res = _parse_validation_response("not json at all, no braces here", signals)

    assert len(res["rejected"]) == 1
    reason = res["rejected"][0]["validation"]["reasoning"]
    assert reason != "Parse error"
    assert "could not be read" in reason


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.signal_validation.get_settings")
def test_signal_validation_node_fails_closed_on_qwen_rate_limit(
    mock_settings, mock_llm_cls
):
    """A Qwen failure must not introduce another provider or approve a trade."""
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"

    mock_agent_ok = MagicMock()
    mock_agent_ok.invoke.return_value.content = (
        '{"validations": [{"signal_id": "1", "decision": "approve"}]}'
    )
    mock_llm_cls.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded on tokens per day"),
        mock_agent_ok,
    ]

    state = create_initial_state()
    state["signals"] = [{"signal_id": "1"}]

    result = signal_validation_node(state)

    assert mock_llm_cls.call_count == 1
    assert result["validated_signals"] == []
    assert len(result["rejected_signals"]) == 1


@patch("src.agents.llm_factory.create_chat_model")
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


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.signal_validation.get_settings")
def test_signal_validation_node_skips_llm_when_disabled(mock_settings, mock_llm_cls):
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["signals"] = [{"signal_id": "1", "strategy": "momentum", "confidence": 0.9, "risk_reward_ratio": 2.0}]
    state["active_strategies"] = ["momentum"]

    result = signal_validation_node(state)

    mock_llm_cls.assert_not_called()
    assert len(result["validated_signals"]) == 1


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.signal_validation.get_settings")
def test_fallback_validation_rejection_reason_is_specific_and_plain_english(
    mock_settings, mock_llm_cls
):
    """Regression guard for the generic, unhelpful 'Fallback: Failed rule-based
    validation' message (gave no indication of why the AI reviewer was skipped or
    which check the signal actually failed). The new message must name both."""
    mock_settings.return_value.enable_llm_agents = False
    state = create_initial_state()
    state["active_strategies"] = ["mean_reversion"]
    state["signals"] = [
        {
            "signal_id": "1",
            "strategy": "trend_following",  # not in active_strategies
            "confidence": 0.9,
            "risk_reward_ratio": 2.0,
        }
    ]

    result = signal_validation_node(state)

    assert len(result["rejected_signals"]) == 1
    reason = result["rejected_signals"][0]["rejection_reason"]
    assert "AI review is turned off in settings" in reason
    assert "trend_following" in reason
    assert "mean_reversion" in reason
    assert reason != "Fallback: Failed rule-based validation"


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


@patch("src.agents.llm_factory.create_chat_model")
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
    state["signals"] = [_eligible_signal("breakout")]
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


def test_parse_strategy_response_accepts_ema_strategies():
    """The three Traderversity-derived strategies (ema_heiken_ashi_rsi, ema_psar,
    ema_cci) must pass the same allowlist as the original four -- the LLM can select
    them, but (like every strategy) they still need an environment-appropriate eligibility
    decision to trade, enforced downstream of candidate generation."""
    content = (
        '{"active_strategies": ["ema_heiken_ashi_rsi", "ema_psar", "ema_cci"], '
        '"reasoning": "trending market"}'
    )
    res = _parse_strategy_response(content)
    assert res["active_strategies"] == ["ema_heiken_ashi_rsi", "ema_psar", "ema_cci"]


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node_fails_closed_on_qwen_rate_limit(
    mock_settings, mock_llm_cls, tmp_path
):
    """A Qwen rate limit uses the deterministic fail-closed path, not another LLM."""
    _register_validated_strategy(tmp_path, "breakout")
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_settings.return_value.strategy_registry_dir = str(tmp_path)

    mock_agent_ok = MagicMock()
    mock_agent_ok.invoke.return_value.content = (
        '{"active_strategies": ["breakout"], "reasoning": "vol"}'
    )
    # First create_chat_model() call (primary model) raises a 429; the second
    # (fallback model) succeeds -- invoke_with_fallback must try both, in order.
    mock_llm_cls.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded on tokens per day"),
        mock_agent_ok,
    ]

    state = create_initial_state()
    state["signals"] = [_eligible_signal("breakout")]
    result = strategy_selection_node(state)

    assert mock_llm_cls.call_count == 1
    assert result["active_strategies"] == []
    assert "backup rule" in result["strategy_reasoning"].lower()


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node(mock_settings, mock_llm_cls, tmp_path):
    # Strategy selection can only retain an already-eligible candidate strategy.
    _register_validated_strategy(tmp_path, "breakout")
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_settings.return_value.strategy_registry_dir = str(tmp_path)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"active_strategies": ["breakout"], "reasoning": "vol"}'
    mock_llm_cls.return_value = mock_llm

    state = create_initial_state()
    state["signals"] = [_eligible_signal("breakout")]
    result = strategy_selection_node(state)

    assert result["active_strategies"] == ["breakout"]


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node_skips_llm_when_disabled(mock_settings, mock_llm_cls, tmp_path):
    _register_validated_strategy(tmp_path, "momentum")
    _register_validated_strategy(tmp_path, "trend_following")
    mock_settings.return_value.enable_llm_agents = False
    mock_settings.return_value.strategy_registry_dir = str(tmp_path)
    state = create_initial_state()
    state["regime"] = "trending_up"
    state["signals"] = [
        _eligible_signal("momentum"),
        _eligible_signal("trend_following"),
    ]

    result = strategy_selection_node(state)

    mock_llm_cls.assert_not_called()
    assert result["active_strategies"] == ["momentum", "trend_following"]
    assert "AI review skipped because AI review is turned off in settings" in (
        result["strategy_reasoning"]
    )
    assert "Error:" not in result["strategy_reasoning"]


def test_strategy_selection_node_skips_qwen_without_candidates(tmp_path):
    # No technical candidate means there is nothing to select or send to Qwen.
    with patch("src.agents.strategy_selection.get_settings") as mock_settings:
        mock_settings.return_value.enable_llm_agents = False
        mock_settings.return_value.strategy_registry_dir = str(tmp_path)
        state = create_initial_state()
        state["regime"] = "trending_up"

        result = strategy_selection_node(state)

    assert result["active_strategies"] == []


def test_strategy_selection_does_not_use_horizon_as_eligibility_identity(tmp_path):
    """Horizon metadata no longer selects a separate strategy-admission record."""
    _register_validated_strategy(tmp_path, "momentum")

    with patch("src.agents.strategy_selection.get_settings") as mock_settings:
        mock_settings.return_value.enable_llm_agents = False
        mock_settings.return_value.strategy_registry_dir = str(tmp_path)

        swing_state = create_initial_state(trade_horizon="SWING")
        swing_state["regime"] = "trending_up"
        swing_state["signals"] = [_eligible_signal("momentum")]
        swing_result = strategy_selection_node(swing_state)

        scalp_state = create_initial_state(trade_horizon="SCALP")
        scalp_state["regime"] = "trending_up"
        scalp_state["signals"] = [_eligible_signal("momentum")]
        scalp_result = strategy_selection_node(scalp_state)

    assert "momentum" in swing_result["active_strategies"]
    assert scalp_result["active_strategies"] == ["momentum"]
