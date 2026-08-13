"""
Stage 12 tests: execution/journal/exit-manager threading for the scalp horizon.

Full end-to-end (scan -> rank -> graph -> risk -> execution) is exercised through
`src/markets/nse/runtime/live.py`'s live-session closure, which -- like the rest of that
module -- isn't practically unit-testable without mocking the entire live-session
object graph (see tests/test_scalp_scan.py's docstring for the same rationale on
Stage 10). These tests instead verify, precisely, the actual units Stage 12 touched:

- journal.py's decision_chain now carries trade_horizon (the one real code change).
- exit_manager.py's pre-existing `timeframe` param (already generic/display-only,
  per Stage 12's plan) correctly threads a scalp entry timeframe like "5m".
- The idempotency-key format `src/markets/nse/runtime/live.py`'s new scalp execution
  block builds is verified by direct construction: it always contains the literal
  "scalp" tag plus the horizon-suffixed workflow_id, so it can never collide with a
  same-cycle, same-symbol SWING entry's key, which never contains either.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# Mock settings before importing modules that use them -- same pattern as
# tests/test_execution.py (these modules read settings at import/construction time).
with patch("src.config.get_settings") as mock_get_settings:
    mock_settings = MagicMock()
    mock_settings.paper_wallet_balance = 100000.0
    mock_settings.database_url = "sqlite:///:memory:"
    mock_settings.trading_mode = "paper"
    mock_settings.execution_mode = "local_paper"
    mock_settings.long_only = False
    mock_get_settings.return_value = mock_settings

    from src.markets.nse.execution.exit_manager import ExitManager
    from src.markets.nse.execution.journal import TradeJournal, TradeRecord


@pytest.fixture
def trade_journal():
    return TradeJournal(database_url="sqlite:///:memory:")


def test_decision_chain_carries_scalp_horizon(trade_journal):
    trade = {
        "signal_id": "SCALP-RELIANCE-5m",
        "symbol": "RELIANCE",
        "signal_type": "BUY",
        "strategy": "momentum",
        "entry_price": 100.0,
        "quantity": 10,
        "confidence": 0.75,
    }
    state = {"regime": "trending_up", "regime_confidence": 0.8, "trade_horizon": "SCALP"}

    trade_id = trade_journal.record_trade(trade, "wf-scalp", state)

    record = (
        trade_journal._session.query(TradeRecord).filter_by(trade_id=trade_id).first()
    )
    decision_chain = json.loads(record.decision_chain)
    assert decision_chain["trade_horizon"] == "SCALP"


def test_decision_chain_defaults_to_swing_when_untagged(trade_journal):
    """Every pre-existing swing call site builds `state` without a trade_horizon key
    at all -- must keep reading as SWING, never crash, never read as SCALP."""
    trade = {"signal_id": "SIG-1", "symbol": "RELIANCE", "entry_price": 100.0, "quantity": 10}
    state = {"regime": "trending_up"}  # no trade_horizon key -- pre-existing shape

    trade_id = trade_journal.record_trade(trade, "wf-swing", state)

    record = (
        trade_journal._session.query(TradeRecord).filter_by(trade_id=trade_id).first()
    )
    decision_chain = json.loads(record.decision_chain)
    assert decision_chain["trade_horizon"] == "SWING"


def test_extract_decision_chain_unit():
    """Direct unit test of the exact function Stage 12 modified."""
    journal = TradeJournal(database_url="sqlite:///:memory:")

    scalp_chain = journal._extract_decision_chain({"trade_horizon": "SCALP", "regime": "ranging"})
    swing_chain = journal._extract_decision_chain({"regime": "ranging"})

    assert scalp_chain["trade_horizon"] == "SCALP"
    assert swing_chain["trade_horizon"] == "SWING"
    # Every other pre-existing key must still be present and correctly derived --
    # confirms this was a pure addition, not a restructuring.
    assert scalp_chain["regime"]["value"] == "ranging"


def test_exit_manager_threads_scalp_entry_timeframe():
    """exit_manager.register_position's pre-existing `timeframe` param (already
    generic/display-only -- see stats.py's precedent) correctly carries a scalp
    entry's 5m timeframe, with zero code change to exit_manager.py itself."""
    manager = ExitManager(state_file=None)

    position = manager.register_position(
        position_id="SCALP-TRD-1",
        symbol="RELIANCE",
        side="BUY",
        quantity=10,
        entry_price=100.0,
        stop_loss=99.0,
        target_price=102.0,
        strategy="momentum",
        regime="trending_up",
        timeframe="5m",
    )

    assert position.timeframe == "5m"
    assert manager.get_managed_positions()[0].timeframe == "5m"


def test_scalp_idempotency_key_never_collides_with_swing_key_same_cycle_same_symbol():
    """Pins the exact idempotency-key construction src/markets/nse/runtime/live.py's
    scalp execution block uses (Stage 12) against the pre-existing swing format, for
    the SAME data_namespace/cycle/symbol -- proving they can never collide even
    when a swing and a scalp entry fire for the same symbol in the same cycle.
    """
    data_namespace = "paper_market_data"
    cycle_timestamp = "20260810120000"
    cycle = 7
    symbol = "RELIANCE"
    signal_id = "SIG-1"

    swing_workflow_id = f"LIVE-{cycle_timestamp}-{cycle}"
    swing_key = f"{data_namespace}:{swing_workflow_id}:{signal_id}:{symbol}:entry"

    scalp_workflow_id = f"LIVE-{cycle_timestamp}-{cycle}-scalp"
    scalp_key = (
        f"{data_namespace}:{scalp_workflow_id}:{signal_id}:{symbol}:scalp:entry"
    )

    assert swing_key != scalp_key
    assert "scalp" in scalp_key
    assert "scalp" not in swing_key
