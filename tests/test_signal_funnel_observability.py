from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.dashboard.stats import TradingStats
from src.finops.cost_tracker import CostTracker
from src.markets.nse.execution.runtime_mode import RuntimeExecutionMode
from src.markets.nse.runtime.live import wait_for_next_cycle
from src.markets.nse.strategies.candidate_policy import CandidateAction, evaluate_long_candidate
from src.observability.signal_funnel import (
    FunnelAudit,
    RejectionReason,
    eligibility_rejection_reason,
    scalp_status_reason,
    strategy_eligibility_rows,
)
from src.webui.schema import stats_to_dict


def _eligibility(
    *,
    strategy: str = "momentum",
    timeframe: str = "15m",
    status: EligibilityStatus = EligibilityStatus.PAPER_APPROVED,
    validated_offset: int = 0,
) -> StrategyEligibility:
    now = datetime.now(UTC)
    return StrategyEligibility(
        strategy_name=strategy,
        timeframe=timeframe,
        model_version=f"v{validated_offset}",
        validation_status=status,
        validated_at=(now + timedelta(seconds=validated_offset)).isoformat(),
        validation_window={"dataset_id": "fixture"},
        oos_trade_count=40,
        oos_profit_factor=1.3,
        oos_max_drawdown=0.08,
        oos_win_rate=0.57,
        minimum_model_confidence=0.0,
        allowed_regimes=("trending_up",),
    )


def test_rejected_candidate_has_stable_reason_code() -> None:
    decision = evaluate_long_candidate(
        {
            "symbol": "RELIANCE",
            "signal_type": "SELL",
            "entry_price": 100.0,
            "stop_loss": 101.0,
            "target_price": 98.0,
        },
        {},
        None,
        target_pct=1.0,
        min_ml_confidence=0.6,
        min_volume_ratio=1.0,
        required_higher_timeframes=2,
        require_ml=False,
    )

    assert decision.action is CandidateAction.REJECT
    assert decision.reason_codes == (RejectionReason.LONG_ONLY.value,)


def test_missing_mtf_data_and_real_conflict_have_distinct_reason_codes() -> None:
    missing = SimpleNamespace(stage="MTF_MISSING_DATA")
    conflict = SimpleNamespace(stage="MTF_CONFLICT")

    assert scalp_status_reason(missing) == RejectionReason.MTF_MISSING_DATA.value
    assert scalp_status_reason(conflict) == RejectionReason.MTF_CONFLICT.value


def test_funnel_counts_reconcile_and_raw_partition_is_exact() -> None:
    audit = FunnelAudit(cycle_id=17, trade_horizon="SWING")
    audit.set("raw_signals", 3)
    audit.set("technical_setups", 2)
    audit.set("registry_eligible", 1)
    audit.set("registry_blocked", 1)
    audit.set("deterministic_qualified", 1)
    audit.set("deterministic_rejected", 0)
    audit.set("ml_candidates", 1)
    audit.set("ml_abstain", 1)
    audit.reject("CONSOLIDATION", RejectionReason.CONSOLIDATED_DUPLICATE.value)
    audit.reject("STRATEGY_ELIGIBILITY", RejectionReason.STRATEGY_UNVALIDATED.value)

    payload = audit.to_dict()
    reasons = {row["reason"]: row["count"] for row in audit.rejection_rows()}

    assert payload["reconciled"] is True
    assert payload["raw_signals"] == payload["technical_setups"] + reasons[
        RejectionReason.CONSOLIDATED_DUPLICATE.value
    ]
    assert payload["technical_setups"] == payload["registry_eligible"] + payload[
        "registry_blocked"
    ]


def test_funnel_detects_non_reconciling_stage() -> None:
    audit = FunnelAudit(cycle_id=18)
    audit.set("technical_setups", 3)
    audit.set("registry_eligible", 1)
    audit.set("registry_blocked", 1)

    payload = audit.to_dict()

    assert payload["reconciled"] is False
    assert payload["reconciliation_errors"]


def test_eligibility_registry_mismatches_are_reported_distinctly(tmp_path: Path) -> None:
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(_eligibility())

    missing_timeframe = registry.evaluate(
        strategy_name="momentum",
        timeframe="5m",
        model_version="v0",
        environment=EligibilityEnvironment.LIVE,
        current_regime="trending_up",
        confidence=0.8,
    )
    wrong_model = registry.evaluate(
        strategy_name="momentum",
        timeframe="15m",
        model_version="different",
        environment=EligibilityEnvironment.LIVE,
        current_regime="trending_up",
        confidence=0.8,
    )
    assert eligibility_rejection_reason(missing_timeframe) == (
        RejectionReason.ELIGIBILITY_REGISTRY_MISSING.value
    )
    assert eligibility_rejection_reason(wrong_model) == (
        RejectionReason.MODEL_VERSION_MISMATCH.value
    )


def test_registry_rows_expose_latest_status_without_calling_it_executable(tmp_path: Path) -> None:
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(_eligibility(status=EligibilityStatus.SHADOW, validated_offset=1))

    rows = strategy_eligibility_rows(registry)
    assert len(rows) == 1
    assert rows[0]["validation_status"] == EligibilityStatus.SHADOW.value


def test_shadow_and_market_paper_namespaces_are_isolated() -> None:
    assert RuntimeExecutionMode.MOCK.expected_quote_source == "simulated"
    assert RuntimeExecutionMode.MARKET_PAPER.expected_quote_source == "real"
    assert RuntimeExecutionMode.MOCK.namespace != RuntimeExecutionMode.MARKET_PAPER.namespace


def test_qwen_usage_attribution_is_traceable() -> None:
    tracker = CostTracker(pricing={"qwen-test": (0.0, 0.0)})
    tracker.record_usage(
        "context_review",
        "qwen-test",
        120,
        30,
        "SCALP",
        purpose="candidate_context_review",
        cycle_id="cycle-19",
        candidate_id="RELIANCE-15m-1",
        symbol="RELIANCE",
    )

    records = tracker.recent_records()
    assert len(records) == 1
    assert records[0]["total_tokens"] == 150
    assert records[0]["purpose"] == "candidate_context_review"
    assert records[0]["cycle_id"] == "cycle-19"
    assert records[0]["candidate_id"] == "RELIANCE-15m-1"
    assert records[0]["symbol"] == "RELIANCE"
    assert tracker.recent_records(0) == []


def test_dashboard_serializes_backend_funnel_exactly() -> None:
    stats = TradingStats()
    audit = FunnelAudit(cycle_id=20)
    audit.set("raw_signals", 5)
    audit.set("technical_setups", 0)
    audit.reject(
        "STRATEGY_ELIGIBILITY", RejectionReason.STRATEGY_UNVALIDATED.value, amount=5
    )
    stats.signal_funnel = audit.to_dict()
    stats.rejection_reasons = audit.rejection_rows()

    payload = stats_to_dict(stats)

    assert payload["signal_funnel"] == audit.to_dict()
    assert payload["rejection_reasons"] == audit.rejection_rows()


def test_web_labels_separate_execution_data_broker_market_and_dashboard() -> None:
    header = Path("web/components/Header.tsx").read_text(encoding="utf-8")
    pipeline = Path("web/components/PipelinePanel.tsx").read_text(encoding="utf-8")

    for label in ("EXECUTION:", "DATA:", "BROKER ORDERS:", "MARKET:", "DASHBOARD:"):
        assert label in header
    assert "No candidate" not in pipeline
    assert "None admitted" not in pipeline


@pytest.mark.asyncio
async def test_no_change_scheduler_keeps_configured_cadence() -> None:
    dashboard = MagicMock()
    refresh = AsyncMock()
    sleep = AsyncMock()

    with patch("src.markets.nse.runtime.live.asyncio.sleep", sleep):
        await wait_for_next_cycle(3, dashboard, refresh)

    dashboard.set_next_cycle_countdown.assert_called_once_with(3)
    assert sleep.await_count == 3
    assert refresh.await_count == 3
