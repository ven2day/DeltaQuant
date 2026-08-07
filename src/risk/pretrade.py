"""Final exact-order risk gate and in-cycle reservation ledger for paper entries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FinalPaperOrder:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    strategy: str

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    @property
    def loss_at_stop(self) -> float:
        return self.quantity * max(0.0, self.entry_price - self.stop_loss)


@dataclass(frozen=True)
class FinalRiskDecision:
    approved: bool
    reasons: list[str]
    projected_exposure_pct: float
    projected_risk_pct: float


class PaperRiskReservations:
    """Reserve approved batch risk so later candidates see earlier approvals immediately."""

    def __init__(self) -> None:
        self._orders: list[FinalPaperOrder] = []

    @property
    def orders(self) -> tuple[FinalPaperOrder, ...]:
        return tuple(self._orders)

    def reserve(self, order: FinalPaperOrder) -> None:
        self._orders.append(order)

    def release(self, order: FinalPaperOrder) -> None:
        if order in self._orders:
            self._orders.remove(order)

    def evaluate(
        self,
        order: FinalPaperOrder,
        *,
        positions: Iterable[Any],
        equity: float,
        entries_today: int,
        daily_entry_cap: int,
        max_positions: int,
        max_position_pct: float,
        max_total_exposure_pct: float,
        risk_per_trade: float,
        max_total_risk: float,
    ) -> FinalRiskDecision:
        reasons: list[str] = []
        positions = list(positions)
        reserved = list(self._orders)
        if order.side != "BUY":
            reasons.append("Permanent long-only policy accepts BUY entries only.")
        if order.quantity <= 0:
            reasons.append("Position sizing returned zero shares; no forced trade is allowed.")
        if order.entry_price <= 0 or not (0 < order.stop_loss < order.entry_price):
            reasons.append("Long entry/stop levels are invalid.")
        if order.target_price <= order.entry_price:
            reasons.append("Long target must be above entry.")
        if equity <= 0:
            reasons.append("Account equity is unavailable.")
        if entries_today + len(reserved) >= daily_entry_cap:
            reasons.append(f"Daily paper-entry cap of {daily_entry_cap} reached or reserved.")
        if len(positions) + len(reserved) >= max_positions:
            reasons.append(f"Maximum concurrent positions ({max_positions}) reached or reserved.")
        existing_symbols = {str(position.symbol) for position in positions}
        existing_symbols.update(item.symbol for item in reserved)
        if order.symbol in existing_symbols:
            reasons.append(f"An open/reserved {order.symbol} position already exists.")

        current_exposure = sum(
            float(position.quantity)
            * float(position.current_price or position.entry_price)
            for position in positions
        )
        current_exposure += sum(item.notional for item in reserved)
        current_risk = sum(
            float(position.quantity)
            * max(0.0, float(position.entry_price) - float(position.stop_loss))
            for position in positions
            if str(position.side) == "BUY"
        )
        current_risk += sum(item.loss_at_stop for item in reserved)
        projected_exposure = current_exposure + order.notional
        projected_risk = current_risk + order.loss_at_stop
        exposure_pct = projected_exposure / equity * 100 if equity > 0 else 100.0
        risk_pct = projected_risk / equity if equity > 0 else 1.0
        position_pct = order.notional / equity if equity > 0 else 1.0
        order_risk_pct = order.loss_at_stop / equity if equity > 0 else 1.0

        if position_pct > max_position_pct:
            reasons.append(
                f"Position notional {position_pct:.1%} exceeds {max_position_pct:.1%}."
            )
        if exposure_pct > max_total_exposure_pct:
            reasons.append(
                f"Projected portfolio exposure {exposure_pct:.1f}% exceeds "
                f"{max_total_exposure_pct:.1f}%."
            )
        if order_risk_pct > risk_per_trade:
            reasons.append(
                f"Loss at stop {order_risk_pct:.2%} exceeds per-trade risk {risk_per_trade:.2%}."
            )
        if risk_pct > max_total_risk:
            reasons.append(
                f"Projected portfolio stop risk {risk_pct:.2%} exceeds {max_total_risk:.2%}."
            )
        return FinalRiskDecision(not reasons, reasons, exposure_pct, risk_pct * 100)
