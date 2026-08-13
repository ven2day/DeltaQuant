from src.markets.nse.execution.paper_engine import LocalPaperEngine
from src.markets.nse.execution.runtime_mode import RuntimeExecutionMode
from src.markets.nse.execution.service import IdempotencyStore
from src.markets.nse.execution.signal_log import SignalLogger, SignalRecord
from src.markets.nse.risk.costs import CostModel
from src.markets.nse.risk.daily_state import DailyRiskStore


def _record(symbol: str) -> SignalRecord:
    return SignalRecord(
        timestamp="2026-08-10T10:00:00+05:30",
        symbol=symbol,
        side="BUY",
        entry_price=100.0,
        timeframe="15m",
        strategy="momentum",
        confidence=0.7,
        status="approved",
    )


def test_market_and_mock_wallets_positions_and_orders_are_isolated(tmp_path):
    url = f"sqlite:///{tmp_path}/paper.db"
    market = LocalPaperEngine(
        100_000,
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MARKET_PAPER,
    )
    mock = LocalPaperEngine(
        50_000,
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MOCK,
    )

    assert (
        market.place_order("RELIANCE", "BUY", 10, 100, entry_data_source="real").status == "FILLED"
    )
    assert mock.place_order("TCS", "BUY", 5, 200, entry_data_source="simulated").status == "FILLED"

    reloaded_market = LocalPaperEngine(
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MARKET_PAPER,
    )
    reloaded_mock = LocalPaperEngine(
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MOCK,
    )
    assert {p.symbol for p in reloaded_market.get_positions()} == {"RELIANCE"}
    assert {p.symbol for p in reloaded_mock.get_positions()} == {"TCS"}
    assert reloaded_market.get_balance() == 99_000
    assert reloaded_mock.get_balance() == 49_000


def test_quote_source_mismatch_is_rejected_in_both_modes(tmp_path):
    url = f"sqlite:///{tmp_path}/source.db"
    market = LocalPaperEngine(
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MARKET_PAPER,
    )
    mock = LocalPaperEngine(
        database_url=url,
        cost_model=CostModel.zero(),
        execution_mode=RuntimeExecutionMode.MOCK,
    )

    assert (
        market.place_order("A", "BUY", 1, 100, entry_data_source="simulated").status == "REJECTED"
    )
    assert mock.place_order("B", "BUY", 1, 100, entry_data_source="real").status == "REJECTED"
    assert market.get_positions() == []
    assert mock.get_positions() == []


def test_daily_risk_and_signal_history_are_namespaced(tmp_path):
    url = f"sqlite:///{tmp_path}/events.db"
    market_risk = DailyRiskStore(database_url=url, namespace="paper_market_data")
    mock_risk = DailyRiskStore(database_url=url, namespace="mock_simulated")
    market_risk.record_trade(-100)
    mock_risk.record_trade(50)
    assert market_risk.get_today()["profit_loss"] == -100
    assert mock_risk.get_today()["profit_loss"] == 50

    market_log = SignalLogger(database_url=url, namespace="paper_market_data")
    mock_log = SignalLogger(database_url=url, namespace="mock_simulated")
    market_log.log(_record("RELIANCE"))
    mock_log.log(_record("TCS"))
    assert [row["symbol"] for row in market_log.read_recent()] == ["RELIANCE"]
    assert [row["symbol"] for row in mock_log.read_recent()] == ["TCS"]


def test_identical_idempotency_keys_are_isolated_by_namespace(tmp_path):
    url = f"sqlite:///{tmp_path}/idempotency.db"
    market = IdempotencyStore(database_url=url, namespace="paper_market_data")
    mock = IdempotencyStore(database_url=url, namespace="mock_simulated")
    market.record("same-intent", {"mode": "market"})
    mock.record("same-intent", {"mode": "mock"})
    assert market.seen("same-intent") == {"mode": "market"}
    assert mock.seen("same-intent") == {"mode": "mock"}
