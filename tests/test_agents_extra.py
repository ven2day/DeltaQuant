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
    directory, strategy_name: str, *, timeframe: str = "", trade_horizon: str = "SWING"
) -> None:
    """Seed a current VALIDATED H-8 registry artifact for ``strategy_name`` in
    ``directory`` -- test helper for strategy_selection_node's admission gate."""
    from src.backtesting.strategy_registry import StrategyRegistry, build_strategy_version

    version = build_strategy_version(
        strategy_name,
        owner="test",
        dataset_id="test-dataset",
        oos_trades=100,
        oos_expectancy=2.0,
        oos_return_pct=10.0,
        fold_consistency=0.8,
        timeframe=timeframe,
        trade_horizon=trade_horizon,
    )
    StrategyRegistry(directory).register(version)


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
def test_signal_validation_node_falls_back_to_secondary_model_on_rate_limit(
    mock_settings, mock_llm_cls
):
    """Same regression guard as strategy_selection's version, for signal_validation_node."""
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

    assert mock_llm_cls.call_count == 2
    assert len(result["validated_signals"]) == 1
    # Confirms it took the real LLM path, not _fallback_signal_validation.
    assert result["validated_signals"][0]["signal_id"] == "1"


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
    them, but (like every strategy) they still need a current VALIDATED registry entry
    to actually trade, enforced downstream by strategy_selection_node's H-8 gate."""
    content = (
        '{"active_strategies": ["ema_heiken_ashi_rsi", "ema_psar", "ema_cci"], '
        '"reasoning": "trending market"}'
    )
    res = _parse_strategy_response(content)
    assert res["active_strategies"] == ["ema_heiken_ashi_rsi", "ema_psar", "ema_cci"]


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node_falls_back_to_secondary_model_on_rate_limit(
    mock_settings, mock_llm_cls, tmp_path
):
    """Regression guard: strategy_selection_node (and signal_validation_node, tested
    separately below) used to call ONLY the primary model -- discarding
    primary_and_fallback_models()'s second element entirely -- so once Groq's daily
    token quota on the primary model was exhausted, every call failed straight to the
    crude rule-based fallback for the rest of the day, even though the secondary model
    (a different model = a different Groq daily-quota bucket) was still working fine
    for market_regime and news_analyst. Both nodes now go through
    llm_factory.invoke_with_fallback, the same retry-on-429 path market_regime always
    had -- this proves a primary-model 429 no longer skips straight to the fallback
    rule-based validation."""
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
    result = strategy_selection_node(state)

    assert mock_llm_cls.call_count == 2
    assert result["active_strategies"] == ["breakout"]
    # Confirms it took the real LLM path, not _fallback_strategy_selection.
    assert result["strategy_reasoning"] == "vol"


@patch("src.agents.llm_factory.create_chat_model")
@patch("src.agents.strategy_selection.get_settings")
def test_strategy_selection_node(mock_settings, mock_llm_cls, tmp_path):
    # H-8: strategy_selection_node now gates its output through the strategy admission
    # registry (fail closed) -- seed a current VALIDATED "breakout" version so it survives.
    _register_validated_strategy(tmp_path, "breakout")
    mock_settings.return_value.groq_api_key.get_secret_value.return_value = "token"
    mock_settings.return_value.strategy_registry_dir = str(tmp_path)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"active_strategies": ["breakout"], "reasoning": "vol"}'
    mock_llm_cls.return_value = mock_llm

    state = create_initial_state()
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

    result = strategy_selection_node(state)

    mock_llm_cls.assert_not_called()
    assert result["active_strategies"] == ["momentum", "trend_following"]
    assert "AI review skipped because AI review is turned off in settings" in (
        result["strategy_reasoning"]
    )
    assert "Error:" not in result["strategy_reasoning"]


def test_strategy_selection_node_strips_strategies_with_no_registry_entry(tmp_path):
    # H-8 fail-closed: with an empty (freshly created) registry directory, nothing is
    # admitted -- no strategy trades without a current VALIDATED artifact, not even the
    # regime-default fallback.
    with patch("src.agents.strategy_selection.get_settings") as mock_settings:
        mock_settings.return_value.enable_llm_agents = False
        mock_settings.return_value.strategy_registry_dir = str(tmp_path)
        state = create_initial_state()
        state["regime"] = "trending_up"

        result = strategy_selection_node(state)

    assert result["active_strategies"] == []


def test_strategy_selection_node_is_horizon_aware(tmp_path):
    """A strategy validated only on SWING must not be admitted for a SCALP-horizon
    cycle, and a SCALP-horizon-validated strategy must not leak into a SWING cycle --
    proven through the actual node, not just the registry unit.
    """
    _register_validated_strategy(tmp_path, "momentum")  # SWING (default)
    _register_validated_strategy(tmp_path, "breakout", timeframe="5m", trade_horizon="SCALP")

    with patch("src.agents.strategy_selection.get_settings") as mock_settings:
        mock_settings.return_value.enable_llm_agents = False
        mock_settings.return_value.strategy_registry_dir = str(tmp_path)

        swing_state = create_initial_state(trade_horizon="SWING")
        swing_state["regime"] = "trending_up"
        swing_result = strategy_selection_node(swing_state)

        scalp_state = create_initial_state(trade_horizon="SCALP")
        scalp_state["regime"] = "volatile"  # -> fallback default is ["breakout"]
        scalp_result = strategy_selection_node(scalp_state)

    assert "momentum" in swing_result["active_strategies"]
    assert "breakout" not in swing_result["active_strategies"]  # SCALP-only artifact
    assert scalp_result["active_strategies"] == ["breakout"]  # admitted under SCALP
