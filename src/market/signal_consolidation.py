"""
Signal consolidation.

Multiple strategies can independently fire on the same symbol/timeframe/direction in
one scan cycle (e.g. momentum AND trend_following both raise a BUY on RELIANCE@15m).
Today, before this module, each such signal becomes an independent ranked candidate --
so strategy agreement doesn't strengthen a single opportunity, it just multiplies the
count of near-duplicate candidates reaching the LLM review stage.

``consolidate_signals`` groups by (symbol, timeframe, signal_type) and produces one
``ConsolidatedSignal`` per group, carrying every contributing strategy plus a
confidence blend that rewards agreement without fabricating conviction from nothing.
It never merges across a different symbol, timeframe, or direction -- a BUY and a SELL
on the same symbol/timeframe are never combined, regardless of confidence.

Wiring (see scripts/run_live_trading.py) is gated behind
``settings.enable_signal_consolidation`` (default False) precisely so the existing
swing candidate mix is provably unchanged unless explicitly enabled.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .indicators import Timeframe
from .signals import SignalType, StrategyType, TradingSignal

# How much each additional agreeing strategy adds to the representative signal's own
# confidence, capped so agreement alone can never manufacture near-certainty out of
# several weak signals. Tunable, not a magic number buried in the formula below.
AGREEMENT_CONFIDENCE_STEP = 0.05
AGREEMENT_CONFIDENCE_CAP = 0.95


@dataclass(frozen=True)
class ConsolidatedSignal:
    """One strengthened trading opportunity representing 1+ strategies agreeing on
    the same symbol, timeframe, and direction."""

    symbol: str
    timeframe: Timeframe
    signal_type: SignalType
    contributing_strategies: tuple[StrategyType, ...]
    agreement_count: int
    representative_signal: TradingSignal
    blended_confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = self.representative_signal.to_dict()
        payload["confidence"] = self.blended_confidence
        payload["contributing_strategies"] = [s.value for s in self.contributing_strategies]
        payload["agreement_count"] = self.agreement_count
        return payload


def _blend_confidence(base_confidence: float, agreement_count: int) -> float:
    """Boost the representative signal's own confidence by how many strategies
    agree, never treating agreement alone as sufficient to reach near-certainty."""
    boosted = base_confidence + AGREEMENT_CONFIDENCE_STEP * (agreement_count - 1)
    return min(AGREEMENT_CONFIDENCE_CAP, boosted)


def consolidate_signals(signals: list[TradingSignal]) -> list[ConsolidatedSignal]:
    """Group ``signals`` by (symbol, timeframe, signal_type) into one
    ``ConsolidatedSignal`` per group.

    The representative signal for each group is the highest-confidence contributor --
    its entry/stop/target/reasons are what downstream ranking and evaluation actually
    see, on the theory that the best-evidenced individual signal is a safer anchor for
    trade levels than an average across strategies with different sizing conventions.
    HOLD signals (already filtered out by SignalEngine before this point in practice,
    but defensively skipped here too) never consolidate into a tradeable opportunity.
    """
    groups: dict[tuple[str, Timeframe, SignalType], list[TradingSignal]] = defaultdict(list)
    for signal in signals:
        if signal.signal_type == SignalType.HOLD:
            continue
        groups[(signal.symbol, signal.timeframe, signal.signal_type)].append(signal)

    consolidated: list[ConsolidatedSignal] = []
    for (symbol, timeframe, signal_type), group in groups.items():
        representative = max(group, key=lambda s: s.confidence)
        contributing = tuple(dict.fromkeys(s.strategy for s in group))  # de-duplicated, ordered
        agreement_count = len(group)
        consolidated.append(
            ConsolidatedSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                contributing_strategies=contributing,
                agreement_count=agreement_count,
                representative_signal=representative,
                blended_confidence=_blend_confidence(representative.confidence, agreement_count),
            )
        )
    return consolidated
