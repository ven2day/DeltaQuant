from dataclasses import dataclass

from src.markets.nse.risk.pretrade import FinalPaperOrder, PaperRiskReservations


@dataclass
class _Position:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float


def _order(loss_at_stop: float) -> FinalPaperOrder:
    return FinalPaperOrder(
        symbol="RELIANCE",
        side="BUY",
        quantity=10,
        entry_price=100,
        stop_loss=100 - loss_at_stop / 10,
        target_price=110,
        strategy="momentum",
    )


def test_realized_loss_reduces_remaining_open_risk_budget():
    risk = PaperRiskReservations()
    decision = risk.evaluate(
        _order(300),
        positions=[],
        equity=100_000,
        entries_today=0,
        daily_entry_cap=5,
        max_positions=5,
        max_position_pct=0.2,
        max_total_exposure_pct=100,
        risk_per_trade=0.01,
        max_total_risk=0.02,
        realized_daily_pnl=-700,
        daily_loss_limit=1_000,
        daily_loss_buffer=0.8,
    )
    assert not decision.approved
    assert any("remaining daily-loss" in reason for reason in decision.reasons)


def test_realized_profit_does_not_expand_daily_loss_budget():
    risk = PaperRiskReservations()
    decision = risk.evaluate(
        _order(850),
        positions=[],
        equity=100_000,
        entries_today=0,
        daily_entry_cap=5,
        max_positions=5,
        max_position_pct=0.2,
        max_total_exposure_pct=100,
        risk_per_trade=0.01,
        max_total_risk=0.02,
        realized_daily_pnl=5_000,
        daily_loss_limit=1_000,
        daily_loss_buffer=0.8,
    )
    assert not decision.approved
