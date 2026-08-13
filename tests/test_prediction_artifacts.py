from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.model_artifacts import (
    ArtifactRejectedError,
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactRegistry,
)
from src.agents.offline_prediction import OfflinePredictionTrainer
from src.agents.prediction import (
    FEATURE_COLS,
    FEATURE_VERSION,
    MODEL_VERSION,
    PredictionAgent,
    _OOSRow,
    _ValidationResult,
)


class _IdentityScaler:
    def transform(self, values):
        return values


class _ConstantModel:
    def __init__(self, value: float = 0.01):
        self.value = value

    def predict(self, values):
        return np.full(len(values), self.value)


def _key(horizon: str = "SWING") -> PredictionArtifactKey:
    return PredictionArtifactKey(
        strategy_version="momentum",
        timeframe="15m",
        trade_horizon=horizon,
        regime="all",
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
    )


def _frame(end: str = "2026-08-10 10:15", rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.005, rows)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(10_000, 100_000, rows),
        },
        index=pd.date_range(end=end, periods=rows, freq="15min"),
    )


def _admit(registry: PredictionArtifactRegistry, key: PredictionArtifactKey) -> None:
    metadata = registry.metadata(
        key,
        oos_samples=50,
        folds_used=5,
        calibration_by_regime={"ranging": 0.2},
        artifact_version="model-v7",
    )
    registry.save(
        PredictionArtifact(
            metadata=metadata,
            scaler=_IdentityScaler(),
            models={"constant": _ConstantModel()},
            weights={"constant": 1.0},
            calibrator=None,
        )
    )


def test_live_path_is_inference_only_and_cached_per_settled_candle(tmp_path, monkeypatch):
    registry = PredictionArtifactRegistry(tmp_path)
    _admit(registry, _key())
    agent = PredictionAgent(artifact_registry=registry)
    monkeypatch.setattr(
        agent,
        "_walk_forward_validate",
        lambda *args, **kwargs: pytest.fail("live path attempted walk-forward validation"),
    )
    monkeypatch.setattr(
        agent,
        "predict",
        lambda *args, **kwargs: pytest.fail("live path attempted training helper"),
    )

    first = agent.predict_cached(
        _frame(),
        "RELIANCE",
        timeframe="15m",
        trade_horizon="SWING",
        strategy_version="momentum",
    )
    second = agent.predict_cached(
        _frame(),
        "RELIANCE",
        timeframe="15m",
        trade_horizon="SWING",
        strategy_version="momentum",
    )
    third = agent.predict_cached(
        _frame("2026-08-10 10:30"),
        "RELIANCE",
        timeframe="15m",
        trade_horizon="SWING",
        strategy_version="momentum",
    )

    assert first.abstained is False and second.abstained is False and third.abstained is False
    metrics = agent.cache_metrics()
    assert metrics.hits == 1
    assert metrics.misses == 2
    assert metrics.inference_count == 2
    assert metrics.training_runs == 0
    assert metrics.walk_forward_runs == 0


def test_horizon_change_reuses_same_validated_model_artifact(tmp_path):
    registry = PredictionArtifactRegistry(tmp_path)
    _admit(registry, _key("SWING"))
    agent = PredictionAgent(artifact_registry=registry)

    result = agent.predict_cached(
        _frame(),
        "RELIANCE",
        timeframe="15m",
        trade_horizon="SCALP",
        strategy_version="momentum",
    )

    assert result.abstained is False
    assert "Inference-only artifact" in result.reasoning
    assert agent.cache_metrics().inference_count == 1


def test_regime_change_does_not_load_another_model_artifact(tmp_path):
    registry = PredictionArtifactRegistry(tmp_path)
    original = _key("SWING")
    _admit(registry, original)
    changed = PredictionArtifactKey(
        strategy_version=original.strategy_version,
        timeframe=original.timeframe,
        trade_horizon="SCALP",
        regime="high_volatility",
        model_version=original.model_version,
        feature_version=original.feature_version,
    )

    loaded = registry.load(changed)

    assert loaded.metadata.key.eligibility_identity == changed.eligibility_identity
    assert len(list(tmp_path.rglob("*.pkl"))) == 1


def test_unvalidated_or_tampered_artifact_is_rejected(tmp_path):
    registry = PredictionArtifactRegistry(tmp_path)
    _admit(registry, _key())
    metadata_path, _ = registry._paths(_key())
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["validation_status"] = "REJECTED"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactRejectedError, match="not VALIDATED"):
        registry.load(_key())


def test_offline_validation_produces_versioned_artifact(tmp_path, monkeypatch):
    registry = PredictionArtifactRegistry(tmp_path)
    trainer = OfflinePredictionTrainer(registry)
    oos_rows = [
        _OOSRow(
            preds={"linear": 0.01, "random_forest": 0.01, "gradient_boost": 0.01},
            actual=0.01 if index % 2 else -0.01,
            regime="ranging",
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        trainer.agent,
        "_walk_forward_validate",
        lambda *args, **kwargs: _ValidationResult(
            weights={"linear": 1.0, "random_forest": 1.0, "gradient_boost": 1.0},
            oos_rows=oos_rows,
            total_weight=3.0,
            folds_used=5,
        ),
    )

    metadata = trainer.train(_frame(rows=120), _key(), artifact_version="offline-v1")
    loaded = registry.load(_key())

    assert metadata.artifact_version == "offline-v1"
    assert loaded.metadata.validated is True
    assert set(loaded.models) == {"linear", "random_forest", "gradient_boost"}
    assert trainer.training_count == 1
    assert trainer.walk_forward_count == 1


def test_panel_training_builds_features_per_symbol_before_combining(tmp_path, monkeypatch):
    trainer = OfflinePredictionTrainer(PredictionArtifactRegistry(tmp_path))
    captured = {}

    def capture(panel, key, *, artifact_version):
        captured["panel"] = panel
        captured["key"] = key
        return "trained"

    monkeypatch.setattr(trainer, "_train_feature_frame", capture)
    first = _frame("2026-08-10 10:15", rows=120)
    second = _frame("2026-08-10 10:15", rows=120) * 1.03

    result = trainer.train_panel({"A": first, "B": second}, _key("SCALP"))

    assert result == "trained"
    assert captured["panel"].index.is_monotonic_increasing
    assert captured["panel"].index.is_unique
    assert list(captured["panel"].columns) == [*FEATURE_COLS, "target"]
