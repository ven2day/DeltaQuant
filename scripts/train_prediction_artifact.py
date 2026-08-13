#!/usr/bin/env python3
"""Offline command for producing a validated live-inference artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.model_artifacts import PredictionArtifactKey, PredictionArtifactRegistry
from src.agents.offline_prediction import OfflinePredictionTrainer
from src.agents.prediction import FEATURE_VERSION, MODEL_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="CSV or parquet OHLCV training data")
    parser.add_argument("--registry", type=Path, default=Path("data/prediction_models"))
    parser.add_argument("--strategy-version", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument(
        "--trade-horizon",
        choices=("SWING", "SCALP"),
        default="SCALP",
        help="Diagnostic training metadata; not part of mandatory model identity",
    )
    parser.add_argument(
        "--regime",
        default="all",
        help="Calibration/report metadata; not part of mandatory model identity",
    )
    parser.add_argument("--artifact-version")
    args = parser.parse_args()

    if args.data.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(args.data)
    else:
        frame = pd.read_csv(args.data, index_col=0, parse_dates=True)
    frame = frame.rename(columns=lambda value: str(value).title())
    key = PredictionArtifactKey(
        strategy_version=args.strategy_version,
        timeframe=args.timeframe,
        trade_horizon=args.trade_horizon,
        regime=args.regime,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
    )
    trainer = OfflinePredictionTrainer(PredictionArtifactRegistry(args.registry))
    metadata = trainer.train(frame, key, artifact_version=args.artifact_version)
    print(json.dumps({**metadata.__dict__, "key": metadata.key.__dict__}, indent=2))


if __name__ == "__main__":
    main()
