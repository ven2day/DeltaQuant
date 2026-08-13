from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.backtesting.strategy_eligibility import (
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.core.candidates import SignalType
from src.core.candidates.signals import StrategyType
from src.core.indicators import Timeframe, calculate_indicators
from src.core.persistence import SchemaBoundCandleRepository, SchemaBoundTradingRepository
from src.markets.forex.eligibility import (
    ForexRegimeClassifier,
    demote_superseded_forex_approvals,
)
from src.markets.forex.market_data.history import OandaHistoricalBackfillService
from src.markets.forex.ml.validation import (
    build_shared_indicator_history,
    evaluate_strategy_history,
    summarize_validation,
)
from src.markets.forex.persistence import bind_trading_repository


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: Any, parameters: Any = None) -> None:
        self.statements.append(str(statement))


class _PostgresEngine:
    dialect = SimpleNamespace(name="postgresql")

    def __init__(self) -> None:
        self.connection = _Connection()

    @contextmanager
    def begin(self) -> Iterator[_Connection]:
        yield self.connection


class _InsertResult:
    def __init__(self, inserted: bool) -> None:
        self._inserted = inserted

    def first(self) -> tuple[str] | None:
        return ("intent",) if self._inserted else None


class _AtomicConnection(_Connection):
    def __init__(self) -> None:
        super().__init__()
        self.intent_inserted = False

    def execute(self, statement: Any, parameters: Any = None) -> _InsertResult:
        sql = str(statement)
        self.statements.append(sql)
        if "RETURNING intent_id" in sql:
            inserted = not self.intent_inserted
            self.intent_inserted = True
            return _InsertResult(inserted)
        return _InsertResult(False)


class _AtomicPostgresEngine(_PostgresEngine):
    def __init__(self) -> None:
        self.connection = _AtomicConnection()


def test_schema_bound_candle_repository_normalizes_empty_result() -> None:
    delegate = SimpleNamespace(load_frame=lambda *args, **kwargs: None)
    repository = SchemaBoundCandleRepository(delegate, market="FOREX", provider="OANDA")
    result = repository.load_frame("EUR_USD", "15m", complete_only=True)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_superseded_forex_approval_is_demoted_without_deleting_audit_data(tmp_path: Any) -> None:
    registry = StrategyEligibilityRegistry(tmp_path / "registry")

    def approved(strategy: str, version: str) -> StrategyEligibility:
        return StrategyEligibility(
            market="FOREX",
            strategy_name=strategy,
            timeframe="1h",
            model_version=version,
            validation_status=EligibilityStatus.PAPER_APPROVED,
            validated_at="2026-01-01T00:00:00+00:00",
            validation_window={"source": "legacy"},
            oos_trade_count=100,
            oos_profit_factor=1.2,
            oos_max_drawdown=5.0,
            oos_win_rate=52.0,
            minimum_model_confidence=0.0,
            status_reason="LEGACY_APPROVAL",
        )

    legacy = approved("momentum", "legacy-v1")
    current = approved("ema_adx_trend", "production_rules_v1")
    registry.register(legacy)
    registry.register(current)
    demoted = demote_superseded_forex_approvals(
        registry,
        approved_identities={current.identity},
        current_model_version="production_rules_v1",
        validated_at="2026-08-11T20:00:00+00:00",
    )
    assert [item.identity for item in demoted] == [legacy.identity]
    reloaded = {item.identity: item for item in registry.load_all()}
    assert reloaded[legacy.identity].validation_status is EligibilityStatus.SHADOW
    assert reloaded[legacy.identity].oos_trade_count == 100
    assert "SUPERSEDED_BY_FOREX_OOS_VALIDATION" in reloaded[legacy.identity].status_reason
    assert reloaded[current.identity].validation_status is EligibilityStatus.PAPER_APPROVED


def test_forex_trading_repository_writes_only_forex_schema() -> None:
    engine = _PostgresEngine()
    repository = bind_trading_repository(engine)  # type: ignore[arg-type]
    assert isinstance(repository, SchemaBoundTradingRepository)
    repository.persist_signal(
        signal_id="signal-1",
        symbol="EUR_USD",
        timeframe="15m",
        side="BUY",
        strategy="ema_adx_trend",
        payload={"market": "FOREX"},
    )
    repository.persist_decision(
        decision_id="decision-1",
        signal_id="signal-1",
        symbol="EUR_USD",
        final_action="SHADOW_BUY",
        rejection_reason="NOT_PAPER_APPROVED",
        payload={"broker_orders": "OFF"},
    )
    sql = "\n".join(engine.connection.statements)
    assert '"forex"."signals"' in sql
    assert '"forex"."decisions"' in sql
    assert '"nse".' not in sql
    assert "public." not in sql


def test_forex_paper_fill_is_atomic_schema_bound_and_idempotent() -> None:
    engine = _AtomicPostgresEngine()
    repository = bind_trading_repository(engine)  # type: ignore[arg-type]
    arguments = {
        "intent_id": "intent-1",
        "order_id": "order-1",
        "position_id": "position-1",
        "symbol": "EUR_USD",
        "side": "BUY",
        "quantity": 10_000.0,
        "order_payload": {"execution": "PAPER", "broker_orders": "OFF"},
        "position_payload": {"status": "OPEN", "market": "FOREX"},
    }
    assert repository.persist_paper_fill(**arguments)
    assert not repository.persist_paper_fill(**arguments)
    sql = "\n".join(engine.connection.statements)
    assert '"forex"."orders"' in sql
    assert '"forex"."positions"' in sql
    assert '"nse".' not in sql
    assert "ON CONFLICT (intent_id) DO NOTHING" in sql


def test_forex_coverage_is_utc_aligned_and_ignores_closed_weekend() -> None:
    service = OandaHistoricalBackfillService(provider=object(), candle_repository=object())
    index = pd.DatetimeIndex(
        [
            datetime(2026, 8, 10, 0, 0, tzinfo=UTC),
            datetime(2026, 8, 10, 0, 15, tzinfo=UTC),
            datetime(2026, 8, 10, 0, 45, tzinfo=UTC),
            datetime(2026, 8, 10, 1, 0, tzinfo=UTC),
        ]
    )
    frame = pd.DataFrame({"close": [1.0] * len(index)}, index=index)
    coverage = service.coverage("EUR_USD", "15m", frame)
    assert coverage.row_count == 4
    assert coverage.expected_intervals == 5
    assert coverage.missing_intervals == 1
    assert coverage.duplicate_timestamps == 0
    assert coverage.timezone_errors == 0
    assert coverage.alignment_errors == 0
    assert coverage.coverage_percentage == 80.0


def test_forex_validation_enters_next_bar_and_charges_spread_slippage() -> None:
    class _Engine:
        def generate_signals(self, indicator: Any, strategies: list[Any]) -> list[Any]:
            return [
                SimpleNamespace(
                    signal_type=SignalType.BUY,
                    entry_price=1.1000,
                    stop_loss=1.0990,
                    target_price=1.1020,
                )
            ]

    indicator = SimpleNamespace(
        close=1.1000,
        atr=0.0005,
        ema={20: 1.1001, 50: 1.1000},
        adx=30.0,
    )
    index = pd.date_range("2026-08-10T00:00:00Z", periods=4, freq="15min")
    # The signal candle touches the target, but the next bar hits the stop. Correct
    # next-bar execution must record STOP rather than leaking the signal candle.
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1000],
            "high": [1.1030, 1.1005, 1.1005, 1.1005],
            "low": [1.0995, 1.0985, 1.0995, 1.0995],
            "close": [1.1000, 1.0990, 1.1000, 1.1000],
            "volume": [100, 100, 100, 100],
        },
        index=index,
    )
    trades = evaluate_strategy_history(
        frame,
        {0: indicator},  # type: ignore[dict-item]
        instrument="EUR_USD",
        timeframe="15m",
        strategy=StrategyType.EMA_ADX_TREND,
        signal_engine=_Engine(),  # type: ignore[arg-type]
        regime_classifier=ForexRegimeClassifier(),
        pip_size=0.0001,
        spread_pips=1.0,
        slippage_pips_per_side=0.2,
        max_holding_bars=2,
        development_fraction=0.0,
    )
    assert len(trades) == 1
    assert trades[0].entry_time == str(index[1])
    assert trades[0].exit_reason == "STOP"
    assert trades[0].r_multiple < -1.0


def test_forex_validation_feature_window_matches_live_260_bar_semantics() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=500, freq="15min")
    frame = pd.DataFrame(
        {
            "open": [1.0 + index * 0.00001 for index in range(500)],
            "high": [1.001 + index * 0.00001 for index in range(500)],
            "low": [0.999 + index * 0.00001 for index in range(500)],
            "close": [1.0005 + index * 0.00001 for index in range(500)],
            "volume": [100 + index for index in range(500)],
        },
        index=index,
    )
    snapshots = build_shared_indicator_history(frame, "EUR_USD", "15m")
    expected = calculate_indicators(frame.iloc[41:301], "EUR_USD", timeframe=Timeframe.M15)
    actual = snapshots[300]
    assert actual.ema == expected.ema
    assert actual.adx == expected.adx
    assert actual.atr == expected.atr
    assert actual.supertrend == expected.supertrend


def test_forex_validation_does_not_approve_weak_oos_edge() -> None:
    trades = [
        SimpleNamespace(
            oos=True,
            entry_time=f"2026-01-{index + 1:02d}",
            r_multiple=-1.0,
            direction="BUY",
            spread_cost=0.0001,
            slippage_cost=0.00004,
            regime="ranging",
        )
        for index in range(10)
    ]
    metrics = summarize_validation(
        "ema_adx_trend",
        "15m",
        trades,  # type: ignore[arg-type]
        folds=5,
        risk_fraction=0.005,
        eligibility_config={
            "minimum_oos_trades": 30,
            "minimum_fold_consistency": 0.5,
            "require_positive_expectancy": True,
            "require_positive_return": True,
        },
        regime_policy_config={
            "minimum_regime_trades": 20,
            "allow_profit_factor": 1.1,
            "reduce_profit_factor": 1.0,
            "reduce_confidence_multiplier": 0.8,
        },
    )
    assert not metrics.approved
    assert "INSUFFICIENT_OOS_TRADES" in metrics.rejection_reasons
    assert metrics.regime_performance["ranging"].policy.value == "BLOCK"
