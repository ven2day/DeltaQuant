"""Forex-owned ML artifact factory."""

from pathlib import Path
from typing import Any

from src.core.ml import PredictionArtifactRegistry
from src.markets.forex.ml.registry import ForexModelRegistry
from src.markets.forex.ml.validation import (
    ForexValidationMetrics,
    build_shared_indicator_history,
    evaluate_strategy_history,
    summarize_validation,
)


def create_artifact_registry(settings: Any) -> PredictionArtifactRegistry:
    if str(settings.market).upper() != "FOREX":
        raise ValueError("Forex artifacts require MARKET=FOREX")
    return PredictionArtifactRegistry(Path(str(settings.ml_model_root)))


__all__ = [
    "ForexValidationMetrics",
    "ForexModelRegistry",
    "build_shared_indicator_history",
    "create_artifact_registry",
    "evaluate_strategy_history",
    "summarize_validation",
]
