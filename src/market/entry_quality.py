"""
Deterministic ``EntryQualityEvaluator`` for the scalp horizon (req 5).

Given a 5m price/volume history and a candidate stop/target, decides whether RIGHT
NOW is a good moment to actually take the trade -- as opposed to the earlier pipeline
stages, which only establish THAT a symbol/timeframe/direction is a reasonable
opportunity at all. Entry timing and opportunity selection are deliberately separate
concerns: a genuinely good scalp setup can still have a bad entry moment (price
already extended, running into untested resistance, a rejection candle) and this
evaluator is the one place that says so.

Every threshold is a named ``settings.scalp_*`` field (src/config/settings.py) --
none of the numbers in this module are unexplained magic constants; each is read
from config and documented there.

Output is descriptive, not authoritative: like the assessment matrix, an ENTER_NOW
here does not by itself approve a trade. It only reaches execution after H-8
admission and every risk_compliance check pass independently (see
DeltaQuant-Quant-Risk-Review.md H-8) -- this module has no path to bypass either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.config import get_settings

from . import price_geometry
from .indicators import IndicatorResult

Status = Literal["ENTER_NOW", "WAIT_PULLBACK", "WAIT_BREAKOUT", "REJECT"]
BreakoutState = Literal["none", "breaking_out", "retesting"]

MIN_BARS_REQUIRED = 20


@dataclass(frozen=True)
class EntryQualityResult:
    status: Status
    preferred_entry_low: float
    preferred_entry_high: float
    vwap_distance_pct: float
    ema9_distance_pct: float
    atr_extension: float
    nearest_swing_support: float | None
    nearest_swing_resistance: float | None
    breakout_state: BreakoutState
    relative_volume: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    risk_reward: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "preferred_entry_low": self.preferred_entry_low,
            "preferred_entry_high": self.preferred_entry_high,
            "vwap_distance_pct": self.vwap_distance_pct,
            "ema9_distance_pct": self.ema9_distance_pct,
            "atr_extension": self.atr_extension,
            "nearest_swing_support": self.nearest_swing_support,
            "nearest_swing_resistance": self.nearest_swing_resistance,
            "breakout_state": self.breakout_state,
            "relative_volume": self.relative_volume,
            "upper_wick_ratio": self.upper_wick_ratio,
            "lower_wick_ratio": self.lower_wick_ratio,
            "risk_reward": self.risk_reward,
            "reasons": self.reasons,
        }


def _reject(
    reason: str,
    *,
    vwap_distance_pct: float = 0.0,
    ema9_distance_pct: float = 0.0,
    atr_extension: float = 0.0,
    nearest_swing_support: float | None = None,
    nearest_swing_resistance: float | None = None,
    breakout_state: BreakoutState = "none",
    relative_volume: float = 0.0,
    upper_wick_ratio: float = 0.0,
    lower_wick_ratio: float = 0.0,
    risk_reward: float = 0.0,
) -> EntryQualityResult:
    return EntryQualityResult(
        status="REJECT",
        preferred_entry_low=0.0,
        preferred_entry_high=0.0,
        vwap_distance_pct=vwap_distance_pct,
        ema9_distance_pct=ema9_distance_pct,
        atr_extension=atr_extension,
        nearest_swing_support=nearest_swing_support,
        nearest_swing_resistance=nearest_swing_resistance,
        breakout_state=breakout_state,
        relative_volume=relative_volume,
        upper_wick_ratio=upper_wick_ratio,
        lower_wick_ratio=lower_wick_ratio,
        risk_reward=risk_reward,
        reasons=[reason],
    )


def _breakout_state(
    frame: pd.DataFrame,
    direction: Literal["BUY", "SELL"],
    level: float | None,
    current_price: float,
    *,
    retest_distance_pct: float,
    lookback_bars: int,
) -> tuple[BreakoutState, str | None]:
    """Whether ``level`` (resistance for BUY, support for SELL) has recently been
    broken through, and if so, whether price is now retesting it or still extending
    away from it. "none" whenever there's no level to react to, or price never
    crossed it within the lookback window."""
    if level is None or level <= 0 or len(frame) < 2:
        return "none", None

    recent_closes = frame["close"].iloc[-lookback_bars:]
    if direction == "BUY":
        crossed = (recent_closes > level).any()
        beyond_now = current_price > level
    else:
        crossed = (recent_closes < level).any()
        beyond_now = current_price < level

    if not crossed:
        return "none", None

    distance_pct = abs(current_price - level) / level * 100
    if beyond_now and distance_pct <= retest_distance_pct:
        return "retesting", f"price broke through {level:.2f} and is retesting it"
    if beyond_now:
        return "breaking_out", f"price broke through {level:.2f} and is extending"
    return "none", f"{level:.2f} was tested but price fell back"


def evaluate_entry_quality(
    symbol: str,
    direction: Literal["BUY", "SELL"],
    frame_5m: pd.DataFrame | None,
    indicators_5m: IndicatorResult | None,
    stop_loss: float,
    target_price: float,
) -> EntryQualityResult:
    """Evaluate whether right now is a good moment to enter ``symbol`` at its
    current 5m price, given ``stop_loss``/``target_price`` already decided upstream.
    """
    settings = get_settings()

    if frame_5m is None or len(frame_5m) < MIN_BARS_REQUIRED:
        return _reject(f"insufficient 5m history for {symbol} (need >= {MIN_BARS_REQUIRED} bars)")

    current = frame_5m.iloc[-1]
    entry_price = float(current["close"])
    if entry_price <= 0:
        return _reject(f"invalid current price for {symbol}")

    # --- risk_reward: a sanity floor only. The authoritative minimum is
    # settings.scalp_min_rr, enforced independently downstream in signal_validation;
    # this is just "is the level geometry even coherent" (never <= 0).
    if direction == "BUY":
        risk = entry_price - stop_loss
        reward = target_price - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - target_price
    risk_reward = reward / risk if risk > 0 else 0.0
    if risk_reward <= 0:
        return _reject(
            f"stop/target geometry is incoherent for a {direction} "
            f"(entry={entry_price:.2f}, stop={stop_loss:.2f}, target={target_price:.2f})",
            risk_reward=risk_reward,
        )

    # --- relative volume: current bar vs. the prior 20-bar average.
    prior_volume = frame_5m["volume"].iloc[-21:-1]
    avg_volume = float(prior_volume.mean()) if len(prior_volume) > 0 else 0.0
    relative_volume = float(current["volume"]) / avg_volume if avg_volume > 0 else 0.0

    # --- candle wick/rejection.
    upper_wick_ratio, lower_wick_ratio = price_geometry.wick_ratios(
        open_=float(current["open"]),
        high=float(current["high"]),
        low=float(current["low"]),
        close=float(current["close"]),
    )
    rejecting_wick_ratio = upper_wick_ratio if direction == "BUY" else lower_wick_ratio

    # --- VWAP / EMA9 / ATR extension.
    vwap = price_geometry.session_vwap(frame_5m)
    vwap_distance_pct = (entry_price - vwap) / vwap * 100 if vwap > 0 else 0.0
    ema9 = (
        indicators_5m.ema.get(9)
        if indicators_5m is not None and indicators_5m.ema
        else None
    ) or price_geometry.ema(frame_5m, 9)
    ema9_distance_pct = (entry_price - ema9) / ema9 * 100 if ema9 > 0 else 0.0
    atr = (
        indicators_5m.atr
        if indicators_5m is not None and indicators_5m.atr
        else price_geometry.atr_pct(frame_5m, entry_price) / 100 * entry_price
    )
    atr_extension = abs(entry_price - vwap) / atr if atr and atr > 0 else 0.0

    # --- swing support/resistance + breakout/retest state.
    support, resistance = price_geometry.nearest_support_resistance(
        frame_5m,
        entry_price,
        threshold_pct=settings.scalping_swing_threshold_pct,
        lookback_bars=settings.scalp_swing_lookback_bars,
    )
    reactive_level = resistance if direction == "BUY" else support
    breakout_state, breakout_reason = _breakout_state(
        frame_5m,
        direction,
        reactive_level,
        entry_price,
        retest_distance_pct=settings.scalp_resistance_min_distance_pct,
        lookback_bars=settings.scalp_breakout_retest_lookback_bars,
    )

    # --- hard REJECT gates: liquidity and candle rejection.
    if relative_volume < settings.scalp_min_volume_ratio:
        return _reject(
            f"relative volume {relative_volume:.2f}x below minimum "
            f"{settings.scalp_min_volume_ratio:.2f}x",
            vwap_distance_pct=vwap_distance_pct,
            ema9_distance_pct=ema9_distance_pct,
            atr_extension=atr_extension,
            nearest_swing_support=support,
            nearest_swing_resistance=resistance,
            breakout_state=breakout_state,
            relative_volume=relative_volume,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            risk_reward=risk_reward,
        )
    if rejecting_wick_ratio > settings.scalp_wick_ratio_max:
        return _reject(
            f"triggering candle's {'upper' if direction == 'BUY' else 'lower'} wick ratio "
            f"{rejecting_wick_ratio:.2f} exceeds max {settings.scalp_wick_ratio_max:.2f} "
            "(rejection/indecision candle)",
            vwap_distance_pct=vwap_distance_pct,
            ema9_distance_pct=ema9_distance_pct,
            atr_extension=atr_extension,
            nearest_swing_support=support,
            nearest_swing_resistance=resistance,
            breakout_state=breakout_state,
            relative_volume=relative_volume,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            risk_reward=risk_reward,
        )

    reasons = [
        f"relative volume {relative_volume:.2f}x",
        f"VWAP distance {vwap_distance_pct:+.2f}%, EMA9 distance {ema9_distance_pct:+.2f}%",
    ]
    if breakout_reason:
        reasons.append(breakout_reason)

    overextended = atr_extension > settings.scalp_atr_extension_max_multiple or (
        abs(vwap_distance_pct) > settings.scalp_vwap_max_distance_pct
        and abs(ema9_distance_pct) > settings.scalp_ema9_max_distance_pct
    )

    near_unbroken_level = (
        reactive_level is not None
        and breakout_state == "none"
        and abs(entry_price - reactive_level) / reactive_level * 100
        <= settings.scalp_resistance_min_distance_pct
    )

    if overextended and breakout_state != "retesting":
        reasons.append(
            f"price is extended {atr_extension:.2f}x ATR from VWAP -- prefer a pullback"
        )
        status: Status = "WAIT_PULLBACK"
        preferred_low = min(vwap, ema9)
        preferred_high = max(vwap, ema9)
    elif near_unbroken_level:
        assert reactive_level is not None  # guaranteed by near_unbroken_level's own check
        level_kind = "resistance" if direction == "BUY" else "support"
        reasons.append(f"price is approaching untested {level_kind} at {reactive_level:.2f}")
        status = "WAIT_BREAKOUT"
        preferred_low = reactive_level
        preferred_high = reactive_level * (1.001 if direction == "BUY" else 0.999)
    else:
        reasons.append("no overextension or untested level blocking immediate entry")
        status = "ENTER_NOW"
        buffer = max(atr * 0.1, entry_price * 0.0005) if atr else entry_price * 0.0005
        preferred_low = entry_price - buffer
        preferred_high = entry_price + buffer

    return EntryQualityResult(
        status=status,
        preferred_entry_low=round(preferred_low, 4),
        preferred_entry_high=round(preferred_high, 4),
        vwap_distance_pct=vwap_distance_pct,
        ema9_distance_pct=ema9_distance_pct,
        atr_extension=atr_extension,
        nearest_swing_support=support,
        nearest_swing_resistance=resistance,
        breakout_state=breakout_state,
        relative_volume=relative_volume,
        upper_wick_ratio=upper_wick_ratio,
        lower_wick_ratio=lower_wick_ratio,
        risk_reward=risk_reward,
        reasons=reasons,
    )
