"""Isolated, explicitly capped averaging experiment for simulated paper trades only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AveragingDecision:
    should_add: bool
    quantity: int
    reason: str


@dataclass(frozen=True)
class CappedAveragingPolicy:
    """Allow a small add only under fixed limits; martingale behavior is impossible."""

    enabled: bool = False
    max_adds: int = 1
    trigger_pct: float = 1.0
    add_fraction: float = 0.25

    def evaluate(
        self,
        *,
        original_quantity: int,
        add_count: int,
        weighted_entry_price: float,
        current_price: float,
        stop_loss: float,
        target_price: float,
    ) -> AveragingDecision:
        if not self.enabled:
            return AveragingDecision(False, 0, "Averaging experiment disabled.")
        if self.max_adds <= 0 or add_count >= self.max_adds:
            return AveragingDecision(False, 0, "Maximum averaging adds reached.")
        if original_quantity <= 0 or weighted_entry_price <= 0 or current_price <= 0:
            return AveragingDecision(False, 0, "Invalid price or quantity for averaging.")
        if not (stop_loss < current_price < weighted_entry_price < target_price):
            return AveragingDecision(
                False,
                0,
                "Add blocked: price must remain above stop and below weighted entry.",
            )
        adverse_pct = (weighted_entry_price - current_price) / weighted_entry_price * 100
        if adverse_pct < self.trigger_pct:
            return AveragingDecision(
                False,
                0,
                f"Adverse move {adverse_pct:.2f}% is below {self.trigger_pct:.2f}% trigger.",
            )
        quantity = int(original_quantity * self.add_fraction)
        if quantity <= 0:
            return AveragingDecision(False, 0, "Configured add fraction rounds to zero shares.")
        return AveragingDecision(
            True,
            quantity,
            f"Capped add {add_count + 1}/{self.max_adds} after {adverse_pct:.2f}% adverse move.",
        )
