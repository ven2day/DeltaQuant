"""Forex-specific spread, slippage, and financing assumptions for paper/backtests."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.models import CanonicalQuote, Instrument


@dataclass(frozen=True)
class ForexCostModel:
    slippage_pips_per_side: float = 0.2
    daily_financing_rate: float = 0.0

    def round_trip_cost_quote(
        self,
        *,
        instrument: Instrument,
        quote: CanonicalQuote,
        units: float,
        holding_days: float = 0.0,
    ) -> float:
        spread_cost = abs(units) * quote.spread
        slippage_cost = abs(units) * instrument.pip_size * self.slippage_pips_per_side * 2.0
        financing = abs(units) * quote.mid * self.daily_financing_rate * max(0.0, holding_days)
        return spread_cost + slippage_cost + financing
