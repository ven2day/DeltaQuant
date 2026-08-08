"""Local ranking for strategy signals before any LLM review.

The ranker deliberately separates three different concepts:

* technical confidence from deterministic indicator agreement;
* ML direction agreement from the local sklearn ensemble; and
* measured outcomes from closed trades in the same strategy/regime.

Only the last item is historical accuracy.  Small samples are shrunk toward 50%
with a Beta prior so a 1/1 record is never displayed as "100% accurate".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from src.agents.prediction import PredictionSignal
from src.market.sectors import get_stock_sector
from src.market.signals import SignalType, TradingSignal


class StrategyPerformanceLike(Protocol):
    total_trades: int
    winning_trades: int


class PerformanceTrackerLike(Protocol):
    def get_strategy_performance(
        self, strategy: str, regime: str, lookback_days: int = 30
    ) -> StrategyPerformanceLike: ...


@dataclass(frozen=True)
class RankedSignal:
    """A signal plus the evidence used to order it."""

    signal: TradingSignal
    rank_score: float
    estimated_win_probability: float
    expected_r_multiple: float
    technical_confidence: float
    ml_direction_probability: float | None
    historical_win_probability: float
    historical_sample_size: int
    discovered_signal_tilt: float
    regime: str
    accuracy_status: str


def infer_signal_regime(signal: TradingSignal) -> str:
    """Infer a coarse local regime from the signal's indicator snapshot."""

    trend = signal.indicators.get("trend", {})
    adx = trend.get("adx")
    plus_di = trend.get("plus_di")
    minus_di = trend.get("minus_di")

    if adx is not None and float(adx) >= 25:
        if plus_di is not None and minus_di is not None:
            return "trending_up" if float(plus_di) >= float(minus_di) else "trending_down"
        return "trending_up" if signal.signal_type == SignalType.BUY else "trending_down"
    if adx is not None and float(adx) < 20:
        return "ranging"
    return "volatile"


def _ml_probability(signal: TradingSignal, prediction: PredictionSignal | None) -> float | None:
    # An abstained prediction (H-7: insufficient OOS sample or collapsed ensemble
    # weights) carries no usable direction -- treat it exactly like "no prediction",
    # never as a confidence-floor vote.
    if prediction is None or prediction.abstained:
        return None
    agrees = (signal.signal_type == SignalType.BUY and prediction.direction == "up") or (
        signal.signal_type == SignalType.SELL and prediction.direction == "down"
    )
    confidence = min(0.95, max(0.05, float(prediction.confidence)))
    return confidence if agrees else 1.0 - confidence


def rank_signals(
    signals: list[TradingSignal],
    predictions: Mapping[tuple[str, str], PredictionSignal],
    performance_tracker: PerformanceTrackerLike,
    *,
    lookback_days: int = 90,
    discovered_signal_tilts: Mapping[str | tuple[str, str], float] | None = None,
) -> list[RankedSignal]:
    """Score signals locally and return strongest expected-R opportunities first.

    Prediction keys are ``(symbol, timeframe.value)``.  The probability is an
    evidence-weighted estimate, not a promise of future accuracy.
    """

    ranked: list[RankedSignal] = []
    for signal in signals:
        regime = infer_signal_regime(signal)
        perf = performance_tracker.get_strategy_performance(
            signal.strategy.value, regime, lookback_days
        )
        sample_size = int(perf.total_trades)

        # Beta(2, 2) smoothing: four prior observations centered at 50%.
        historical_probability = (int(perf.winning_trades) + 2) / (sample_size + 4)
        ml_probability = _ml_probability(
            signal, predictions.get((signal.symbol, signal.timeframe.value))
        )
        technical = min(0.95, max(0.05, float(signal.confidence)))

        if ml_probability is None:
            technical_weight = 0.70 if sample_size < 5 else 0.55
            history_weight = 1.0 - technical_weight
            estimated_probability = (
                technical_weight * technical + history_weight * historical_probability
            )
        else:
            history_weight = min(0.40, 0.15 + sample_size / 100)
            ml_weight = 0.30
            technical_weight = 1.0 - history_weight - ml_weight
            estimated_probability = (
                technical_weight * technical
                + ml_weight * ml_probability
                + history_weight * historical_probability
            )

        tilt_values = discovered_signal_tilts or {}
        discovered_tilt = float(
            tilt_values.get(
                (signal.symbol, signal.timeframe.value),
                tilt_values.get(signal.symbol, 0.0),
            )
        )
        discovered_tilt = min(0.20, max(-0.20, discovered_tilt))
        estimated_probability = min(
            0.95, max(0.05, estimated_probability + discovered_tilt)
        )
        expected_r = (
            estimated_probability * float(signal.risk_reward_ratio)
            - (1.0 - estimated_probability)
        )
        # Expected R is primary; probability breaks near-ties without overwhelming payoff.
        rank_score = expected_r + (0.10 * estimated_probability)
        accuracy_status = (
            "empirical" if sample_size >= 20 else "limited_history" if sample_size else "cold_start"
        )

        ranking_payload = {
            "rank_score": round(rank_score, 4),
            "estimated_win_probability": round(estimated_probability, 4),
            "expected_r_multiple": round(expected_r, 4),
            "technical_confidence": round(technical, 4),
            "ml_direction_probability": (
                round(ml_probability, 4) if ml_probability is not None else None
            ),
            "historical_win_probability": round(historical_probability, 4),
            "historical_sample_size": sample_size,
            "discovered_signal_tilt": round(discovered_tilt, 4),
            "regime": regime,
            "accuracy_status": accuracy_status,
        }
        signal.indicators["ranking"] = ranking_payload
        ranked.append(
            RankedSignal(
                signal=signal,
                rank_score=rank_score,
                estimated_win_probability=estimated_probability,
                expected_r_multiple=expected_r,
                technical_confidence=technical,
                ml_direction_probability=ml_probability,
                historical_win_probability=historical_probability,
                historical_sample_size=sample_size,
                discovered_signal_tilt=discovered_tilt,
                regime=regime,
                accuracy_status=accuracy_status,
            )
        )

    return sorted(
        ranked,
        key=lambda item: (
            item.rank_score,
            item.estimated_win_probability,
            item.signal.confidence,
        ),
        reverse=True,
    )


def select_diversified_signals(
    ranked: list[RankedSignal],
    *,
    max_symbols: int,
    max_per_sector: int,
    max_signals_per_symbol: int,
) -> list[RankedSignal]:
    """Keep ranked signals for a bounded, sector-diverse set of symbols."""

    if max_symbols <= 0 or max_signals_per_symbol <= 0:
        return []

    selected: list[RankedSignal] = []
    admitted_symbols: set[str] = set()
    symbol_counts: dict[str, int] = {}
    sector_counts: dict[str, int] = {}

    for item in ranked:
        symbol = item.signal.symbol
        if symbol not in admitted_symbols:
            if len(admitted_symbols) >= max_symbols:
                continue
            sector = get_stock_sector(symbol)
            if sector != "Unknown" and sector_counts.get(sector, 0) >= max_per_sector:
                continue
            admitted_symbols.add(symbol)
            if sector != "Unknown":
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        count = symbol_counts.get(symbol, 0)
        if count >= max_signals_per_symbol:
            continue
        selected.append(item)
        symbol_counts[symbol] = count + 1

    return selected
