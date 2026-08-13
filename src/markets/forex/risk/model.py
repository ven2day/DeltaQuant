"""Deterministic Forex sizing and pre-trade controls.

This adapter computes broker units and market-specific checks, then the existing
``risk_compliance`` node remains the final portfolio/order authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.core.models import CanonicalQuote, Instrument, PriceSide


@dataclass(frozen=True)
class ForexRiskLimits:
    risk_per_trade: float = 0.005
    max_positions: int = 5
    max_portfolio_exposure: float = 2.0
    max_currency_exposure: float = 1.0
    max_correlated_currency_positions: int = 3
    max_spread_pips: float = 5.0
    max_spread_atr_ratio: float = 0.10
    max_quote_age_seconds: float = 15.0
    daily_loss_limit_fraction: float = 0.02
    max_margin_utilization: float = 0.50

    @classmethod
    def from_settings(cls, settings: Any, *, trade_horizon: str = "SWING") -> ForexRiskLimits:
        """Build limits for a trade horizon.

        SCALP swaps in its own per-trade risk, position cap, and (tighter) spread
        ceiling -- a scalp fill must not tolerate the wider spreads a multi-hour
        swing can absorb. Every other control (exposure, correlation, daily-loss,
        margin) stays identical and deterministic across horizons.
        """
        scalp = str(trade_horizon).upper() == "SCALP"
        return cls(
            risk_per_trade=float(
                settings.forex_scalp_risk_per_trade if scalp else settings.forex_risk_per_trade
            ),
            max_positions=int(
                settings.forex_scalp_max_positions if scalp else settings.forex_max_positions
            ),
            max_portfolio_exposure=float(settings.forex_max_portfolio_exposure),
            max_currency_exposure=float(settings.forex_max_currency_exposure),
            max_correlated_currency_positions=int(settings.forex_max_correlated_currency_positions),
            max_spread_pips=float(
                settings.forex_scalp_max_spread_pips if scalp else settings.forex_max_spread_pips
            ),
            max_spread_atr_ratio=float(settings.forex_max_spread_atr_ratio),
            max_quote_age_seconds=float(settings.oanda_stream_stale_seconds),
            daily_loss_limit_fraction=float(settings.forex_daily_loss_limit_fraction),
            max_margin_utilization=float(settings.forex_max_margin_utilization),
        )


@dataclass(frozen=True)
class ForexSizeResult:
    units: float
    risk_amount_home: float
    stop_distance_pips: float
    pip_value_per_unit_home: float
    entry_price: float
    reason: str


@dataclass(frozen=True)
class ForexRiskDecision:
    approved: bool
    reason_codes: tuple[str, ...]
    executable_price: float
    spread_pips: float
    spread_atr_ratio: float
    details: dict[str, float | int | str] = field(default_factory=dict)


def quote_to_home_rate(
    instrument: Instrument,
    account_home_currency: str,
    conversion_rates: dict[tuple[str, str], float],
) -> float:
    home = account_home_currency.upper()
    quote = instrument.quote_currency.upper()
    if quote == home:
        return 1.0
    direct = conversion_rates.get((quote, home))
    if direct is not None and direct > 0:
        return direct
    inverse = conversion_rates.get((home, quote))
    if inverse is not None and inverse > 0:
        return 1.0 / inverse
    raise ValueError(f"Missing conversion rate from {quote} to account currency {home}")


def size_forex_position(
    *,
    instrument: Instrument,
    quote: CanonicalQuote,
    side: PriceSide | str,
    stop_price: float,
    account_equity: float,
    risk_fraction: float,
    account_home_currency: str,
    conversion_rates: dict[tuple[str, str], float],
) -> ForexSizeResult:
    if account_equity <= 0 or not 0 < risk_fraction <= 0.05:
        raise ValueError("Forex account equity/risk fraction is invalid")
    entry = quote.executable_price(side)
    stop_distance = abs(entry - stop_price)
    if stop_distance <= 0:
        raise ValueError("Forex stop loss must have positive distance from executable entry")
    stop_pips = stop_distance / instrument.pip_size
    conversion = quote_to_home_rate(instrument, account_home_currency, conversion_rates)
    pip_value_per_unit_home = instrument.pip_size * conversion
    risk_amount = account_equity * risk_fraction
    raw_units = risk_amount / (stop_pips * pip_value_per_unit_home)
    precision_factor = 10**instrument.trade_units_precision
    # Counter normal binary-float noise at an exact unit boundary (e.g.
    # 19_999.999999999996) without ever rounding a genuinely fractional unit up.
    units = math.floor(raw_units * precision_factor + 1e-9) / precision_factor
    if instrument.maximum_order_units is not None:
        units = min(units, instrument.maximum_order_units)
    if units < instrument.minimum_trade_size:
        return ForexSizeResult(
            units=0.0,
            risk_amount_home=risk_amount,
            stop_distance_pips=stop_pips,
            pip_value_per_unit_home=pip_value_per_unit_home,
            entry_price=entry,
            reason="BELOW_MINIMUM_TRADE_SIZE",
        )
    return ForexSizeResult(
        units=units,
        risk_amount_home=risk_amount,
        stop_distance_pips=stop_pips,
        pip_value_per_unit_home=pip_value_per_unit_home,
        entry_price=entry,
        reason="SIZED",
    )


def evaluate_forex_pretrade(
    *,
    instrument: Instrument,
    quote: CanonicalQuote,
    side: PriceSide | str,
    atr: float,
    open_positions: int,
    portfolio_exposure_ratio: float,
    currency_exposure_ratios: dict[str, float],
    correlated_currency_positions: int,
    limits: ForexRiskLimits,
    global_kill_switch: bool,
    forex_kill_switch: bool,
    stop_price: float | None = None,
    risk_amount_home: float = 0.0,
    account_equity_home: float = 0.0,
    daily_realized_pnl_home: float = 0.0,
    margin_used_home: float = 0.0,
    required_margin_home: float = 0.0,
    duplicate_position: bool = False,
    require_stop_loss: bool = False,
    now: datetime | None = None,
) -> ForexRiskDecision:
    reasons: list[str] = []
    if global_kill_switch:
        reasons.append("GLOBAL_KILL_SWITCH")
    if forex_kill_switch:
        reasons.append("FOREX_KILL_SWITCH")
    executable_price = quote.executable_price(side)
    normalized_side = PriceSide(side) if isinstance(side, str) else side
    if require_stop_loss and stop_price is None:
        reasons.append("STOP_LOSS_REQUIRED")
    elif stop_price is not None and (
        (normalized_side is PriceSide.BUY and stop_price >= executable_price)
        or (normalized_side is PriceSide.SELL and stop_price <= executable_price)
    ):
        reasons.append("INVALID_STOP_LOSS")
    if duplicate_position:
        reasons.append("DUPLICATE_POSITION")
    if not quote.tradeable:
        reasons.append("INSTRUMENT_NOT_TRADEABLE")
    reference = now or datetime.now(UTC)
    if quote.age_seconds(reference) > limits.max_quote_age_seconds:
        reasons.append("STALE_PRICE")
    spread_pips = quote.spread_pips(instrument)
    spread_atr_ratio = quote.spread / atr if atr > 0 else math.inf
    if spread_pips > limits.max_spread_pips:
        reasons.append("SPREAD_PIPS_LIMIT")
    if spread_atr_ratio > limits.max_spread_atr_ratio:
        reasons.append("SPREAD_ATR_LIMIT")
    if open_positions >= limits.max_positions:
        reasons.append("FOREX_POSITION_LIMIT")
    if portfolio_exposure_ratio >= limits.max_portfolio_exposure:
        reasons.append("FOREX_PORTFOLIO_EXPOSURE")
    if any(
        abs(value) >= limits.max_currency_exposure for value in currency_exposure_ratios.values()
    ):
        reasons.append("CURRENCY_EXPOSURE_LIMIT")
    if correlated_currency_positions >= limits.max_correlated_currency_positions:
        reasons.append("CORRELATED_CURRENCY_EXPOSURE")
    if account_equity_home > 0:
        allowed_risk = account_equity_home * limits.risk_per_trade
        if risk_amount_home <= 0 or risk_amount_home > allowed_risk + 1e-9:
            reasons.append("MONETARY_RISK_LIMIT")
        daily_loss_limit = account_equity_home * limits.daily_loss_limit_fraction
        consumed_daily_loss = abs(min(0.0, daily_realized_pnl_home))
        if consumed_daily_loss + max(0.0, risk_amount_home) > daily_loss_limit:
            reasons.append("DAILY_LOSS_LIMIT")
        if (margin_used_home + max(0.0, required_margin_home)) / account_equity_home > (
            limits.max_margin_utilization
        ):
            reasons.append("MARGIN_UTILIZATION_LIMIT")
    return ForexRiskDecision(
        approved=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        executable_price=executable_price,
        spread_pips=spread_pips,
        spread_atr_ratio=spread_atr_ratio,
        details={
            "open_positions": open_positions,
            "portfolio_exposure_ratio": portfolio_exposure_ratio,
            "quote_age_seconds": quote.age_seconds(reference),
            "risk_amount_home": risk_amount_home,
            "daily_realized_pnl_home": daily_realized_pnl_home,
            "margin_used_home": margin_used_home,
            "required_margin_home": required_margin_home,
        },
    )
