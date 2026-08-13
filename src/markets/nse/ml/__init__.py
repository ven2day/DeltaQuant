"""NSE-owned ML artifact factory."""

from pathlib import Path
from typing import Any

from src.core.ml import PredictionArtifactRegistry
from src.markets.nse.ml.registry import NSEModelRegistry


def create_artifact_registry(settings: Any) -> PredictionArtifactRegistry:
    if str(settings.market).upper() != "NSE":
        raise ValueError("NSE artifacts require MARKET=NSE")
    return PredictionArtifactRegistry(Path(str(settings.ml_model_root)))


__all__ = ["NSEModelRegistry", "create_artifact_registry"]
