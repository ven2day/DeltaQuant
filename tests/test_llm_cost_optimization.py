"""Regression tests for Qwen cost controls and post-cache safety invariants."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.candidate_cache import (
    build_candidate_fingerprint,
    reset_candidate_verdict_cache,
)
from src.agents.graph import run_trading_cycle, should_continue_after_regime
from src.agents.risk_compliance import RiskLimits, risk_compliance_node
from src.agents.shared_context import (
    get_shared_market_context_service,
    reset_shared_market_context_service,
)
from src.agents.signal_validation import signal_validation_node
from src.agents.state import create_initial_state
from src.markets.nse.execution.preflight import reviewed_price_is_fresh


def _optimization_settings(registry_dir: str = "data/strategy_registry") -> SimpleNamespace:
    return SimpleNamespace(
        llm_cost_optimization_enabled=True,
        enable_llm_agents=True,
        enable_rate_limiting=False,
        strategy_registry_dir=registry_dir,
        strategy_eligibility_registry_dir=registry_dir,
        llm_cache_ttl_scalp_seconds=300,
        llm_cache_ttl_swing_seconds=1800,
        llm_cache_material_price_move_bps=20.0,
        llm_cache_ml_probability_step=0.05,
        llm_cache_expected_r_step=0.10,
        llm_cache_entry_score_step=5.0,
        qwen_max_output_tokens_fast=128,
        qwen_max_output_tokens_review=256,
        market_context_ttl_seconds=600,
        market_news_ttl_seconds=600,
        sector_context_ttl_seconds=900,
        market_context_breadth_invalidation_step=0.10,
        market_context_volatility_invalidation_step=0.25,
        market_context_sentiment_invalidation_step=0.20,
        strategy_eligibility_paper_gate_enabled=True,
    )


def _shared_context() -> dict[str, object]:
    return {
        "regime": "ranging",
        "regime_confidence": 0.8,
        "regime_reasoning": "bounded",
        "context_version": "market-v1",
        "major_market_event_risk": "NONE",
        "news_sentiment": {"avg_sentiment": 0.0},
        "news_headlines": [],
        "market_mood": {},
        "errors": [],
    }


def _candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "signal_id": "AUBANK-5m",
        "symbol": "AUBANK",
        "trade_horizon": "SCALP",
        "timeframe": "5m",
        "primary_timeframe": "5m",
        "signal_type": "BUY",
        "strategy": "momentum",
        "signal_candle_timestamp": "2026-08-10T09:25:00+05:30",
        "entry_state": "ENTER_NOW",
        "entry_quality_score": 81,
        "mtf_confirmation_passed": True,
        "regime_compatible": True,
        "confidence": 0.82,
        "ml_probability": 0.68,
        "expected_r": 2.1,
        "risk_reward_ratio": 2.1,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "target_price": 104.2,
        "position_size_pct": 1.0,
    }
    candidate.update(overrides)
    return candidate


@pytest.mark.parametrize(
    ("overrides", "expected_rule"),
    [
        ({"entry_state": "WAIT_PULLBACK"}, "entry_quality"),
        ({"mtf_confirmation_passed": False}, "mtf_confirmation"),
        ({}, "strategy_eligibility"),
    ],
)
async def test_deterministic_rejection_skips_graph_and_qwen(tmp_path, overrides, expected_rule):
    settings = _optimization_settings(str(tmp_path))
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    state_signal = _candidate(**overrides)
    budget = MagicMock()
    budget.is_over_hard_budget.return_value = False

    with (
        patch("src.agents.graph.get_settings", return_value=settings),
        patch("src.agents.pre_llm.get_settings", return_value=settings),
        patch("src.backtesting.strategy_eligibility.get_settings", return_value=settings),
        patch("src.agents.pre_llm.get_cost_tracker", return_value=budget),
        patch("src.agents.pre_llm.check_kill_switch", return_value=False),
    ):
        result = await run_trading_cycle(
            graph,
            market_data={},
            indicators={},
            signals=[state_signal],
            trade_horizon="SCALP",
            shared_market_context=_shared_context(),
        )

    graph.ainvoke.assert_not_awaited()
    failures = result["risk_rejected"][0]["risk_result"]["failures"]
    assert failures[0]["rule"] == expected_rule
    assert result["llm_optimization_metrics"]["qwen_calls"] == 0


def test_candidate_fingerprint_material_invalidation_and_horizon_isolation():
    base = _candidate()
    kwargs = {"regime": "ranging", "market_context_version": "market-v1"}
    fingerprint = build_candidate_fingerprint(base, **kwargs)
    assert fingerprint == build_candidate_fingerprint(dict(base), **kwargs)
    assert fingerprint != build_candidate_fingerprint(
        _candidate(signal_candle_timestamp="2026-08-10T09:30:00+05:30"), **kwargs
    )
    assert fingerprint != build_candidate_fingerprint(_candidate(ml_probability=0.80), **kwargs)
    assert fingerprint != build_candidate_fingerprint(_candidate(expected_r=2.4), **kwargs)
    assert fingerprint != build_candidate_fingerprint(_candidate(trade_horizon="SWING"), **kwargs)
    assert fingerprint != build_candidate_fingerprint(
        base, regime="volatile", market_context_version="market-v1"
    )
    assert fingerprint != build_candidate_fingerprint(
        base, regime="ranging", market_context_version="market-v2"
    )


def test_identical_candidate_reuses_qwen_verdict():
    reset_candidate_verdict_cache()
    settings = _optimization_settings()
    state = create_initial_state(trade_horizon="SCALP")
    state["signals"] = [_candidate()]
    state["regime"] = "ranging"
    state["regime_confidence"] = 0.8
    state["active_strategies"] = ["momentum"]
    state["shared_market_context"] = _shared_context()
    response = MagicMock()
    response.content = (
        '{"validations":[{"signal_id":"AUBANK-5m","decision":"approve",'
        '"confidence":0.78,"reason_codes":["NO_EVENT_VETO"],"risk_flags":[]}]}'
    )
    budget = MagicMock()
    budget.is_over_hard_budget.return_value = False
    breaker = MagicMock(is_available=True, recovery_time=1)

    with (
        patch("src.agents.signal_validation.get_settings", return_value=settings),
        patch("src.agents.signal_validation.get_cost_tracker", return_value=budget),
        patch("src.agents.signal_validation.get_llm_circuit_breaker", return_value=breaker),
        patch("src.agents.signal_validation.invoke_with_fallback", return_value=response) as invoke,
        patch("src.agents.signal_validation.model_for_tier", return_value="qwen-fast"),
        patch("src.agents.signal_validation.record_llm_response"),
        patch("src.agents.signal_validation.record_candidate_review"),
    ):
        first = signal_validation_node(state)
        second = signal_validation_node(state)

    assert invoke.call_count == 1
    assert first["llm_optimization_metrics"]["qwen_cache_misses"] == 1
    assert second["llm_optimization_metrics"]["qwen_cache_hits"] == 1
    assert second["validated_signals"][0]["validation"]["cache_hit"] is True


async def test_shared_market_context_generated_once_for_unchanged_inputs():
    reset_shared_market_context_service()
    service = get_shared_market_context_service()
    settings = _optimization_settings()
    news = SimpleNamespace(avg_sentiment=0.1, items=[])
    analyst = MagicMock()
    analyst.get_market_sentiment = AsyncMock(return_value=news)

    with (
        patch("src.agents.shared_context.get_settings", return_value=settings),
        patch("src.agents.shared_context.NewsAnalyst", return_value=analyst),
        patch(
            "src.agents.shared_context.market_regime_node",
            return_value={
                "regime": "ranging",
                "regime_confidence": 0.8,
                "regime_reasoning": "bounded",
            },
        ) as regime,
    ):
        first = await service.get_or_create({"A": {"last_price": 100, "change_percent": 0.1}}, {})
        second = await service.get_or_create({"A": {"last_price": 100, "change_percent": 0.1}}, {})

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert regime.call_count == 1


def test_cached_approval_does_not_bypass_fresh_execution_checks():
    assert reviewed_price_is_fresh(100.0, 100.05) is True
    assert reviewed_price_is_fresh(100.0, 100.2) is False

    state = create_initial_state()
    state["regime_confidence"] = 0.9
    state["daily_stats"] = {"profit_loss": -20_000}
    assert should_continue_after_regime(state) == "end"

    cached = _candidate(strategy="custom")
    cached["validation"] = {
        "decision": "approve",
        "confidence": 0.8,
        "cache_hit": True,
    }
    risk_state = create_initial_state()
    risk_state["validated_signals"] = [cached]
    risk_state["portfolio"] = {
        "capital": 100_000,
        "positions": [{"symbol": "AUBANK", "quantity": 1, "entry_price": 100}],
    }
    risk_state["daily_stats"] = {"trades_count": 0, "profit_loss": 0, "max_drawdown": 0}
    limits = RiskLimits(force_trading_window=True)
    with patch("src.agents.risk_compliance.RiskLimits.from_settings", return_value=limits):
        risk_result = risk_compliance_node(risk_state)
    rules = {
        failure["rule"] for failure in risk_result["risk_rejected"][0]["risk_result"]["failures"]
    }
    assert "duplicate_position" in rules


def test_model_router_uses_deterministic_uncertainty():
    from src.agents.llm_factory import model_for_tier, review_model_tier

    settings = SimpleNamespace(
        llm_provider="qwen",
        qwen_fast_model="qwen-turbo",
        qwen_review_model="qwen-plus",
        qwen_model_primary="qwen-plus",
        qwen_model_fallback="qwen-turbo",
    )
    with patch("src.agents.llm_factory.get_settings", return_value=settings):
        assert model_for_tier("FAST") == "qwen-turbo"
        assert model_for_tier("REVIEW") == "qwen-plus"
        assert (
            review_model_tier(
                [{"confidence": 0.9}], {"regime_confidence": 0.9, "major_market_event_risk": "NONE"}
            )
            == "FAST"
        )
        assert (
            review_model_tier(
                [{"confidence": 0.9}], {"regime_confidence": 0.9, "major_market_event_risk": "HIGH"}
            )
            == "REVIEW"
        )


def test_finops_hard_cap_blocks_model_construction():
    from langchain_core.messages import HumanMessage

    from src.agents.llm_factory import invoke_with_fallback

    settings = SimpleNamespace(llm_cost_optimization_enabled=True)
    tracker = MagicMock()
    tracker.is_over_hard_budget.return_value = True
    breaker = MagicMock(is_available=True, recovery_time=1, name="qwen")
    with (
        patch("src.agents.llm_factory.get_settings", return_value=settings),
        patch("src.finops.get_cost_tracker", return_value=tracker),
        patch("src.agents.llm_factory.create_chat_model") as create,
        pytest.raises(RuntimeError, match="FinOps hard budget"),
    ):
        invoke_with_fallback(
            [HumanMessage(content="fixture")],
            circuit_breaker=breaker,
            models_to_try=("qwen-fast",),
            trade_horizon="SCALP",
        )
    create.assert_not_called()


def test_langfuse_candidate_span_preserves_traceability():
    from src.observability import tracing

    span = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = span
    client = MagicMock()
    client.start_as_current_observation.return_value = context
    metadata = {
        "symbol": "AUBANK",
        "trade_horizon": "SCALP",
        "timeframe": "5m",
        "candidate_fingerprint": "fingerprint-1",
        "cache_hit": True,
        "model": "qwen-fast",
        "context_version": "market-v1",
        "decision": "approve",
        "execution_result": "PENDING_DETERMINISTIC_RISK",
        "secret": "must-not-be-recorded",
    }
    with patch.object(tracing, "_langfuse_client", client):
        tracing.record_candidate_review(metadata)

    recorded = client.start_as_current_observation.call_args.kwargs["metadata"]
    assert recorded["candidate_fingerprint"] == "fingerprint-1"
    assert recorded["cache_hit"] is True
    assert "secret" not in recorded
