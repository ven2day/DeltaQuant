from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.models import CanonicalQuote, Market, MarketProvider
from src.markets.forex.runtime.lifecycle import (
    FINAL_DECISION_STAGES,
    ForexDecisionLifecycle,
    ForexDecisionStage,
    candidate_identity,
    classify_instrument_conflicts,
    qwen_review_identity,
    update_candidate_state,
)
from src.markets.forex.runtime.paper_positions import (
    ForexPaperPositionManager,
    create_local_paper_position,
    persist_lifecycle_record,
)
from src.markets.snapshots import MarketSnapshotStore


class _Candidate(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return {
            "market": "FOREX",
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "signal_type": self.direction.value,
            "strategy": "ema_adx_trend",
            "supporting_strategies": ["ema_adx_trend"],
            "strategy_consensus_identity": "ema_adx_trend:production_rules_v1",
            "feature_version": "features-v1",
            "technical_quality": self.technical_quality,
            "conflict_level": "NONE",
            "conflict_resolution": "NO_CONFLICT",
            "entry_price": 1.1,
            "stop_loss": 1.09 if self.direction.value == "BUY" else 1.11,
            "target_price": 1.12 if self.direction.value == "BUY" else 1.08,
            "eligibility_status": "PAPER_APPROVED" if self.execution_allowed else "SHADOW",
            "current_regime": "trending",
            "regime_policy": "ALLOW",
        }


def _candidate(
    symbol: str = "EUR_USD",
    timeframe: str = "15m",
    side: str = "BUY",
    *,
    executable: bool = True,
    score: float = 0.8,
) -> _Candidate:
    eligibility = SimpleNamespace(oos_profit_factor=1.2)
    validated = SimpleNamespace(eligibility=eligibility) if executable else None
    return _Candidate(
        symbol=symbol,
        timeframe=SimpleNamespace(value=timeframe),
        direction=SimpleNamespace(value=side),
        settled_candle_timestamp="2026-08-12T10:15:00+00:00",
        feature_snapshot_id=f"{symbol}:{timeframe}",
        technical_quality=score,
        strongest_validated_signal=validated,
        execution_allowed=executable,
    )


class _MemoryRepository:
    def __init__(self) -> None:
        self.decisions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.intents: set[str] = set()

    def persist_decision(self, **kwargs: Any) -> None:
        self.decisions[str(kwargs["decision_id"])] = dict(kwargs["payload"])

    def persist_paper_fill(self, **kwargs: Any) -> bool:
        intent = str(kwargs["intent_id"])
        if intent in self.intents:
            return False
        self.intents.add(intent)
        self.orders[str(kwargs["order_id"])] = dict(kwargs["order_payload"])
        self.positions[str(kwargs["position_id"])] = dict(kwargs["position_payload"])
        return True

    def load_open_positions(self) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.positions.values()
            if str(item.get("status", "OPEN")) == "OPEN"
        ]

    def load_decision(self, decision_id: str) -> dict[str, Any] | None:
        payload = self.decisions.get(decision_id)
        return dict(payload) if payload is not None else None

    def persist_paper_close(self, **kwargs: Any) -> bool:
        intent = str(kwargs["intent_id"])
        if intent in self.intents:
            return False
        self.intents.add(intent)
        self.orders[str(kwargs["order_id"])] = dict(kwargs["order_payload"])
        self.positions[str(kwargs["position_id"])] = dict(kwargs["position_payload"])
        self.decisions[str(kwargs["decision_id"])] = dict(kwargs["decision_payload"])
        return True

    def final_decisions(self) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.decisions.values()
            if item.get("final_action") in {"PAPER_BUY", "PAPER_SELL"}
        ]


def _approved_record(candidate: _Candidate | None = None) -> ForexDecisionLifecycle:
    record = ForexDecisionLifecycle.from_candidate(candidate or _candidate())  # type: ignore[arg-type]
    record.advance(ForexDecisionStage.ELIGIBILITY_PASSED)
    record.advance(ForexDecisionStage.REGIME_PASSED)
    record.advance(ForexDecisionStage.ENTRY_QUALITY_PASSED)
    record.advance(ForexDecisionStage.ML_SKIPPED)
    record.advance(ForexDecisionStage.QWEN_SKIPPED)
    record.risk_result = "APPROVED"
    record.advance(ForexDecisionStage.RISK_APPROVED)
    return record


def test_technical_candidates_and_final_decisions_are_separate_snapshot_models(tmp_path) -> None:
    store = MarketSnapshotStore(tmp_path, min_write_seconds=0)
    technical = {"candidate_id": "candidate-1", "symbol": "EUR_USD", "side": "BUY"}
    final = {
        "decision_id": "decision-1",
        "symbol": "GBP_USD",
        "side": "SELL",
        "current_stage": "POSITION_OPEN",
    }
    assert store.publish(
        "FOREX",
        status={"status": "HEALTHY"},
        signals=[final],
        technical_candidates=[technical],
        positions=[],
        force=True,
    )
    view = store.api_view("FOREX")
    assert view.scoped_signals(7) == [{**final, "market": "FOREX"}]
    assert view.scoped_candidates() == [{**technical, "market": "FOREX"}]


def test_eligibility_blocked_candidate_never_becomes_final_decision() -> None:
    repository = _MemoryRepository()
    record = ForexDecisionLifecycle.from_candidate(_candidate(executable=False))  # type: ignore[arg-type]
    record.reject("STRATEGY_NOT_PAPER_APPROVED", shadow=True)
    persist_lifecycle_record(repository, record)
    assert repository.final_decisions() == []
    assert repository.decisions[record.decision_id]["rejection_reason"] == (
        "STRATEGY_NOT_PAPER_APPROVED"
    )


def test_risk_approved_decision_creates_persisted_order_and_position() -> None:
    repository = _MemoryRepository()
    record = _approved_record()
    created = create_local_paper_position(
        repository,
        record,
        quantity=10_000,
        position_payload={
            "symbol": "EUR_USD",
            "side": "BUY",
            "entry_price": 1.1,
            "stop_price": 1.09,
            "target_price": 1.12,
        },
    )
    assert created
    assert repository.final_decisions()[0]["current_stage"] == "POSITION_OPEN"
    assert len(repository.orders) == 1
    assert len(repository.load_open_positions()) == 1
    assert {item["stage"] for item in record.stage_history}.issuperset(
        {"FINAL_PAPER_DECISION", "PAPER_ORDER_CREATED", "POSITION_OPEN"}
    )


def test_final_decision_and_position_survive_unrelated_next_scan() -> None:
    repository = _MemoryRepository()
    record = _approved_record()
    assert create_local_paper_position(
        repository,
        record,
        quantity=10_000,
        position_payload={"symbol": "EUR_USD", "side": "BUY", "status": "OPEN"},
    )
    # A no-change scan writes no candidate state and does not touch authoritative rows.
    assert len(repository.final_decisions()) == 1
    assert len(repository.load_open_positions()) == 1


def test_duplicate_candidate_does_not_create_duplicate_position() -> None:
    repository = _MemoryRepository()
    first = _approved_record()
    payload = {"symbol": "EUR_USD", "side": "BUY", "status": "OPEN"}
    assert create_local_paper_position(repository, first, quantity=10_000, position_payload=payload)
    second = _approved_record()
    assert not create_local_paper_position(
        repository, second, quantity=10_000, position_payload=payload
    )
    assert len(repository.orders) == 1
    assert len(repository.positions) == 1


def test_opposing_timeframes_are_deterministic_conflict_and_have_no_primary() -> None:
    buy = _candidate(timeframe="15m", side="BUY")
    sell = _candidate(timeframe="1h", side="SELL")
    resolutions = classify_instrument_conflicts([buy, sell])  # type: ignore[list-item]
    for candidate in (buy, sell):
        resolution = resolutions[candidate_identity(candidate)]  # type: ignore[arg-type]
        assert resolution.conflicted
        assert resolution.primary_candidate_id is None
        assert resolution.buy_timeframes == ("15m",)
        assert resolution.sell_timeframes == ("1h",)


def test_same_direction_timeframes_select_one_validated_primary() -> None:
    weaker = _candidate(timeframe="15m", side="BUY", score=0.70)
    stronger = _candidate(timeframe="1h", side="BUY", score=0.85)
    resolutions = classify_instrument_conflicts([weaker, stronger])  # type: ignore[list-item]
    assert resolutions[candidate_identity(weaker)].primary_candidate_id == candidate_identity(  # type: ignore[arg-type]
        stronger
    )


def test_unchanged_higher_timeframe_candidate_remains_in_conflict_context() -> None:
    state: dict[tuple[str, str], Any] = {}
    higher_sell = _candidate(timeframe="4h", side="SELL")
    update_candidate_state(
        state,
        changed_grains=(("EUR_USD", "4h"),),
        current_candidates=(higher_sell,),  # type: ignore[arg-type]
    )
    lower_buy = _candidate(timeframe="15m", side="BUY")
    context = update_candidate_state(
        state,
        changed_grains=(("EUR_USD", "15m"),),
        current_candidates=(lower_buy,),  # type: ignore[arg-type]
    )
    resolutions = classify_instrument_conflicts(context)
    assert resolutions[candidate_identity(lower_buy)].conflicted  # type: ignore[arg-type]
    assert resolutions[candidate_identity(higher_sell)].conflicted  # type: ignore[arg-type]


def test_settled_timeframe_without_candidate_clears_previous_direction() -> None:
    state: dict[tuple[str, str], Any] = {}
    previous = _candidate(timeframe="4h", side="SELL")
    update_candidate_state(
        state,
        changed_grains=(("EUR_USD", "4h"),),
        current_candidates=(previous,),  # type: ignore[arg-type]
    )
    context = update_candidate_state(
        state,
        changed_grains=(("EUR_USD", "4h"),),
        current_candidates=(),
    )
    assert context == ()


def test_qwen_review_identity_is_stable_and_materially_scoped() -> None:
    candidate = _candidate()
    first = qwen_review_identity(candidate)  # type: ignore[arg-type]
    second = qwen_review_identity(candidate)  # type: ignore[arg-type]
    changed = qwen_review_identity(_candidate(timeframe="1h"))  # type: ignore[arg-type]
    assert first == second
    assert changed != first


class _PriceProvider:
    def __init__(self, quote: CanonicalQuote) -> None:
        self.quote = quote

    async def get_prices(self, symbols: list[str]) -> dict[str, CanonicalQuote]:
        return {symbol: self.quote for symbol in symbols}


@pytest.mark.asyncio
async def test_position_closes_only_after_actual_stop_or_target_event() -> None:
    repository = _MemoryRepository()
    record = _approved_record()
    assert create_local_paper_position(
        repository,
        record,
        quantity=10_000,
        position_payload={
            "symbol": "EUR_USD",
            "side": "BUY",
            "entry_price": 1.1,
            "stop_price": 1.09,
            "target_price": 1.12,
        },
    )
    neutral = CanonicalQuote(
        symbol="EUR_USD",
        market=Market.FOREX,
        provider=MarketProvider.OANDA,
        timestamp=datetime.now(UTC),
        bid=1.10,
        ask=1.1002,
        tradeable=True,
        source="test",
    )
    result = await ForexPaperPositionManager(_PriceProvider(neutral), repository).manage()
    assert result.positions_closed == 0
    assert len(repository.load_open_positions()) == 1

    target = CanonicalQuote(
        symbol="EUR_USD",
        market=Market.FOREX,
        provider=MarketProvider.OANDA,
        timestamp=datetime.now(UTC),
        bid=1.1201,
        ask=1.1203,
        tradeable=True,
        source="test",
    )
    result = await ForexPaperPositionManager(_PriceProvider(target), repository).manage()
    assert result.positions_closed == 1
    assert result.target_exits == 1
    assert repository.load_open_positions() == []
    decision = repository.final_decisions()[0]
    assert decision["current_stage"] == ForexDecisionStage.POSITION_CLOSED.value
    assert decision["current_stage"] in FINAL_DECISION_STAGES


def test_snapshot_failure_cannot_erase_persisted_decision_or_position(tmp_path) -> None:
    repository = _MemoryRepository()
    record = _approved_record()
    assert create_local_paper_position(
        repository,
        record,
        quantity=10_000,
        position_payload={"symbol": "EUR_USD", "side": "BUY", "status": "OPEN"},
    )
    store = MarketSnapshotStore(tmp_path, writable=False)
    with pytest.raises(RuntimeError, match="Read-only"):
        store.publish("FOREX", status={}, signals=[], positions=[], force=True)
    assert len(repository.final_decisions()) == 1
    assert len(repository.load_open_positions()) == 1
