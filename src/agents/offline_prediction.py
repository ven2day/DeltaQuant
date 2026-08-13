"""Offline-only training and validation for ``PredictionAgent`` artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.prediction import (
    FEATURE_COLS,
    MIN_LABELED_SAMPLES,
    WEIGHT_COLLAPSE_EPS,
    PredictionAgent,
    StandardScaler,
)
from src.core.ml import (
    ArtifactRejectedError,
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactMetadata,
    PredictionArtifactRegistry,
)


class OfflinePredictionTrainer:
    """The only service allowed to fit the live prediction ensemble."""

    def __init__(
        self,
        registry: PredictionArtifactRegistry,
        agent: PredictionAgent | None = None,
    ) -> None:
        self.registry = registry
        self.agent = agent or PredictionAgent(artifact_registry=registry)
        self.training_count = 0
        self.walk_forward_count = 0

    def train(
        self,
        historical_data: pd.DataFrame | dict[str, Any],
        key: PredictionArtifactKey,
        *,
        artifact_version: str | None = None,
    ) -> PredictionArtifactMetadata:
        if isinstance(historical_data, dict):
            frame_input = pd.DataFrame(historical_data)
        else:
            frame_input = historical_data.copy()
        feature_frame = self.agent._compute_feature_frame(frame_input)
        if feature_frame is None:
            raise ArtifactRejectedError("Insufficient feature history for offline training")
        return self._train_feature_frame(
            feature_frame,
            key,
            artifact_version=artifact_version,
        )

    def train_panel(
        self,
        historical_by_symbol: dict[str, pd.DataFrame],
        key: PredictionArtifactKey,
        *,
        artifact_version: str | None = None,
    ) -> PredictionArtifactMetadata:
        """Train one shared cross-sectional market model without symbol-boundary leakage.

        Features are built independently for every symbol, then the median feature and
        next-bar target are taken at each exchange timestamp.  The resulting chronological
        panel is validated by the existing rolling-origin splitter.  This avoids joining
        the end of one symbol's candles to the start of another symbol's history while
        keeping the live artifact intentionally symbol-agnostic.
        """
        feature_frames: list[pd.DataFrame] = []
        for historical_data in historical_by_symbol.values():
            frame_input = historical_data.rename(columns=lambda value: str(value).title())
            feature_frame = self.agent._compute_feature_frame(frame_input)
            if feature_frame is not None:
                feature_frames.append(feature_frame)
        if not feature_frames:
            raise ArtifactRejectedError("No symbol had sufficient feature history")
        combined = pd.concat(feature_frames)
        combined = combined.sort_index()
        panel = combined.groupby(level=0)[[*FEATURE_COLS, "target"]].median()
        return self._train_feature_frame(
            panel,
            key,
            artifact_version=artifact_version,
        )

    def _train_feature_frame(
        self,
        feature_frame: pd.DataFrame,
        key: PredictionArtifactKey,
        *,
        artifact_version: str | None,
    ) -> PredictionArtifactMetadata:
        """Validate, fit and persist one already-constructed chronological feature set."""
        labeled = feature_frame.dropna(subset=["target"])
        if len(labeled) < MIN_LABELED_SAMPLES:
            raise ArtifactRejectedError(
                f"Only {len(labeled)} labeled samples; need {MIN_LABELED_SAMPLES}"
            )
        x_matrix = labeled[FEATURE_COLS].to_numpy()
        y = labeled["target"].to_numpy()
        factories = self.agent._model_factories()
        validation = self.agent._walk_forward_validate(x_matrix, y, FEATURE_COLS, factories)
        self.walk_forward_count += 1
        if not validation.oos_rows or validation.total_weight < WEIGHT_COLLAPSE_EPS:
            raise ArtifactRejectedError("Offline ensemble failed out-of-sample validation")

        oos_raw = [
            sum(row.preds.get(name, 0.0) * weight for name, weight in validation.weights.items())
            / validation.total_weight
            for row in validation.oos_rows
        ]
        actual_direction = [1 if row.actual > 0 else 0 for row in validation.oos_rows]
        regimes = [row.regime for row in validation.oos_rows]
        calibrator = self.agent._fit_calibrator(oos_raw, actual_direction)
        calibration = self.agent._calibration_error_by_regime(
            oos_raw, actual_direction, regimes, calibrator
        )

        scaler = StandardScaler().fit(x_matrix)
        scaled = scaler.transform(x_matrix)
        models: dict[str, Any] = {}
        for name, factory in factories.items():
            model = factory()
            model.fit(scaled, y)
            models[name] = model
        self.training_count += 1
        metadata = self.registry.metadata(
            key,
            oos_samples=len(validation.oos_rows),
            folds_used=validation.folds_used,
            calibration_by_regime=calibration,
            artifact_version=artifact_version,
        )
        return self.registry.save(
            PredictionArtifact(
                metadata=metadata,
                scaler=scaler,
                models=models,
                weights=validation.weights,
                calibrator=calibrator,
            )
        )
