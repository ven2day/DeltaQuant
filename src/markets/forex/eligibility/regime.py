"""Deterministic Forex regime classification for runtime eligibility context.

Classification and policy are deliberately separate: this module names the current
market state from a completed FeatureSnapshot; StrategyEligibility records decide
whether that state is ALLOW, REDUCE_CONFIDENCE, or BLOCK.  It never grants approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.indicators import IndicatorResult


@dataclass(frozen=True)
class ForexRegimeClassifier:
    trend_adx_min: float = 25.0
    high_volatility_atr_ratio: float = 0.01
    low_volatility_atr_ratio: float = 0.002

    @classmethod
    def from_settings(cls, settings: Any | None) -> ForexRegimeClassifier:
        return cls(
            trend_adx_min=float(getattr(settings, "forex_regime_trend_adx_min", 25.0)),
            high_volatility_atr_ratio=float(
                getattr(settings, "forex_regime_high_volatility_atr_ratio", 0.01)
            ),
            low_volatility_atr_ratio=float(
                getattr(settings, "forex_regime_low_volatility_atr_ratio", 0.002)
            ),
        )

    def classify(self, indicator: IndicatorResult) -> str:
        close = float(indicator.close)
        atr_ratio = float(indicator.atr or 0.0) / close if close > 0 else 0.0
        if atr_ratio >= self.high_volatility_atr_ratio:
            return "high_volatility"
        if atr_ratio <= self.low_volatility_atr_ratio:
            return "low_volatility"

        ema20 = indicator.ema.get(20)
        ema50 = indicator.ema.get(50)
        adx = float(indicator.adx or 0.0)
        if ema20 is not None and ema50 is not None and adx >= self.trend_adx_min:
            if float(ema20) > float(ema50):
                return "trending_up"
            if float(ema20) < float(ema50):
                return "trending_down"
        return "ranging"


__all__ = ["ForexRegimeClassifier"]
