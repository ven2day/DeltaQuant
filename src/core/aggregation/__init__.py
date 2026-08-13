"""Shared deterministic candidate aggregation."""

from src.core.aggregation.consolidation import (
    AGREEMENT_CONFIDENCE_CAP,
    AGREEMENT_CONFIDENCE_STEP,
    AggregatedCandidate,
    ConsolidatedSignal,
    FeatureSnapshot,
    RegisteredStrategySignal,
    aggregate_strategy_signals,
    consolidate_signals,
    eligible_strategy_types,
    evaluate_registered_strategies,
    strategy_requires_ml,
)

__all__ = [
    "AGREEMENT_CONFIDENCE_CAP",
    "AGREEMENT_CONFIDENCE_STEP",
    "AggregatedCandidate",
    "ConsolidatedSignal",
    "FeatureSnapshot",
    "RegisteredStrategySignal",
    "aggregate_strategy_signals",
    "consolidate_signals",
    "eligible_strategy_types",
    "evaluate_registered_strategies",
    "strategy_requires_ml",
]
