"""Crypto-owned ML artifact factory."""

from pathlib import Path
from typing import Any

from src.core.ml import PredictionArtifactRegistry


def create_artifact_registry(settings: Any) -> PredictionArtifactRegistry:
    if str(settings.market).upper() != "CRYPTO":
        raise ValueError("Crypto artifacts require MARKET=CRYPTO")
    return PredictionArtifactRegistry(Path(str(settings.ml_model_root)))


__all__ = ["create_artifact_registry"]
