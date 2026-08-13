"""Versioned, fail-closed artifacts for live prediction inference."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ArtifactRejectedError(RuntimeError):
    """Raised when an artifact is missing, corrupt, unvalidated, or semantically wrong."""


@dataclass(frozen=True)
class PredictionArtifactKey:
    strategy_version: str
    timeframe: str
    trade_horizon: str
    regime: str
    model_version: str
    feature_version: str
    market: str = "NSE"
    instrument_scope: str = "*"

    def normalized(self) -> PredictionArtifactKey:
        return PredictionArtifactKey(
            strategy_version=self.strategy_version.strip(),
            timeframe=self.timeframe.strip().lower(),
            trade_horizon=self.trade_horizon.strip().upper(),
            regime=(self.regime or "all").strip().lower(),
            model_version=self.model_version.strip(),
            feature_version=self.feature_version.strip(),
            market=self.market.strip().upper(),
            instrument_scope=(self.instrument_scope or "*").strip().upper(),
        )

    @property
    def eligibility_identity(self) -> tuple[str, str, str, str, str, str]:
        """Mandatory model identity; horizon/regime remain diagnostic metadata."""

        normalized = self.normalized()
        return (
            normalized.market,
            normalized.instrument_scope,
            normalized.strategy_version,
            normalized.timeframe,
            normalized.model_version,
            normalized.feature_version,
        )

    @property
    def artifact_id(self) -> str:
        value = "__".join(self.eligibility_identity)
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


@dataclass(frozen=True)
class PredictionArtifactMetadata:
    key: PredictionArtifactKey
    artifact_version: str
    validation_status: str
    created_at: str
    oos_samples: int
    folds_used: int
    calibration_by_regime: dict[str, float]
    payload_sha256: str = ""

    @property
    def validated(self) -> bool:
        return self.validation_status.upper() == "VALIDATED"


@dataclass
class PredictionArtifact:
    metadata: PredictionArtifactMetadata
    scaler: Any
    models: dict[str, Any]
    weights: dict[str, float]
    calibrator: Any


class PredictionArtifactRegistry:
    """Local immutable-ish registry used by the live inference-only path.

    Metadata and payload are separate so the admission contract can be inspected
    without unpickling model objects.  Only locally produced, checksum-matching files
    should be placed in this directory.
    """

    def __init__(self, root: str | Path = "data/prediction_models") -> None:
        self.root = Path(root)

    @staticmethod
    def _path_part(value: str) -> str:
        part = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
        if not part:
            raise ArtifactRejectedError("Artifact path identity contains an empty component")
        return part

    def _paths(self, key: PredictionArtifactKey) -> tuple[Path, Path]:
        normalized = key.normalized()
        # Physical market ownership is deliberately visible on disk.  Keeping the
        # market above the strategy/timeframe/model grain prevents a registry rooted
        # at ``artifacts`` from ever resolving an NSE payload while serving FOREX.
        base = (
            self.root
            / self._path_part(normalized.market.lower())
            / self._path_part(normalized.strategy_version)
            / self._path_part(normalized.timeframe)
            / self._path_part(normalized.model_version)
            / normalized.artifact_id
        )
        return base.with_suffix(".json"), base.with_suffix(".pkl")

    def _resolve_paths(self, key: PredictionArtifactKey) -> tuple[Path, Path]:
        """Resolve the new core identity, with a fail-closed legacy-file fallback."""

        direct = self._paths(key)
        if direct[0].exists() and direct[1].exists():
            return direct
        if not self.root.exists():
            return direct
        matching: list[tuple[Path, Path]] = []
        for metadata_path in self.root.glob("*.json"):
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                stored = PredictionArtifactKey(**raw["key"]).normalized()
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            payload_path = metadata_path.with_suffix(".pkl")
            if (
                stored.eligibility_identity == key.normalized().eligibility_identity
                and payload_path.exists()
            ):
                matching.append((metadata_path, payload_path))
        if len(matching) > 1:
            raise ArtifactRejectedError(
                "Conflicting legacy model artifacts share the same "
                "strategy/timeframe/model identity"
            )
        return matching[0] if matching else direct

    def save(self, artifact: PredictionArtifact) -> PredictionArtifactMetadata:
        key = artifact.metadata.key.normalized()
        if not artifact.metadata.validated:
            raise ArtifactRejectedError("Only VALIDATED prediction artifacts may be admitted")
        metadata_path, payload_path = self._paths(key)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists() or payload_path.exists():
            raise FileExistsError(f"Artifact version already exists: {key.artifact_id}")
        payload = pickle.dumps(
            {
                "scaler": artifact.scaler,
                "models": artifact.models,
                "weights": artifact.weights,
                "calibrator": artifact.calibrator,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        checksum = hashlib.sha256(payload).hexdigest()
        metadata = PredictionArtifactMetadata(
            key=key,
            artifact_version=artifact.metadata.artifact_version,
            validation_status=artifact.metadata.validation_status,
            created_at=artifact.metadata.created_at,
            oos_samples=artifact.metadata.oos_samples,
            folds_used=artifact.metadata.folds_used,
            calibration_by_regime=artifact.metadata.calibration_by_regime,
            payload_sha256=checksum,
        )
        payload_path.write_bytes(payload)
        metadata_path.write_text(
            json.dumps(
                {**asdict(metadata), "key": asdict(metadata.key)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return metadata

    def load(self, requested: PredictionArtifactKey) -> PredictionArtifact:
        requested = requested.normalized()
        metadata_path, payload_path = self._resolve_paths(requested)
        if not metadata_path.exists() or not payload_path.exists():
            raise ArtifactRejectedError(f"No validated artifact for {requested.artifact_id}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = PredictionArtifactMetadata(
            key=PredictionArtifactKey(**raw["key"]).normalized(),
            artifact_version=str(raw["artifact_version"]),
            validation_status=str(raw["validation_status"]),
            created_at=str(raw["created_at"]),
            oos_samples=int(raw["oos_samples"]),
            folds_used=int(raw["folds_used"]),
            calibration_by_regime={
                str(name): float(value)
                for name, value in dict(raw.get("calibration_by_regime", {})).items()
            },
            payload_sha256=str(raw.get("payload_sha256", "")),
        )
        if metadata.key.eligibility_identity != requested.eligibility_identity:
            raise ArtifactRejectedError("Artifact semantic key does not match the live request")
        if not metadata.validated:
            raise ArtifactRejectedError("Prediction artifact is not VALIDATED")
        payload = payload_path.read_bytes()
        if not metadata.payload_sha256 or hashlib.sha256(payload).hexdigest() != metadata.payload_sha256:
            raise ArtifactRejectedError("Prediction artifact checksum mismatch")
        values = pickle.loads(payload)  # noqa: S301 - trusted local registry, checksum verified
        if not values.get("models") or not values.get("weights") or values.get("scaler") is None:
            raise ArtifactRejectedError("Prediction artifact payload is incomplete")
        return PredictionArtifact(metadata=metadata, **values)

    @staticmethod
    def metadata(
        key: PredictionArtifactKey,
        *,
        oos_samples: int,
        folds_used: int,
        calibration_by_regime: dict[str, float],
        artifact_version: str | None = None,
    ) -> PredictionArtifactMetadata:
        return PredictionArtifactMetadata(
            key=key.normalized(),
            artifact_version=artifact_version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            validation_status="VALIDATED",
            created_at=datetime.now(UTC).isoformat(),
            oos_samples=oos_samples,
            folds_used=folds_used,
            calibration_by_regime=calibration_by_regime,
        )
