#!/usr/bin/env python3
"""Train and admit strategy/timeframe prediction artifacts from local TimescaleDB candles.

The underlying prediction model is intentionally shared across strategies because its
features predict next-bar direction and contain no strategy identity.  It is trained once
per timeframe on a chronological cross-sectional panel, then persisted under the same
strategy/timeframe/model identity used across horizon and runtime-regime metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.model_artifacts import (  # noqa: E402
    ArtifactRejectedError,
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactRegistry,
)
from src.agents.offline_prediction import OfflinePredictionTrainer  # noqa: E402
from src.agents.prediction import FEATURE_VERSION, MODEL_VERSION  # noqa: E402
from src.backtesting.strategy_eligibility import (  # noqa: E402
    EligibilityStatus,
    eligibility_registry_from_settings,
)
from src.config import get_settings  # noqa: E402
from src.core.candidates import StrategyType  # noqa: E402
from src.core.candles import CandleStore  # noqa: E402


def _symbols(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(
            dict.fromkeys(
                row.get("symbol", "").strip().upper()
                for row in csv.DictReader(handle)
                if row.get("symbol", "").strip()
            )
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols-file", type=Path, default=Path("config/nse/symbols.csv")
    )
    parser.add_argument("--timeframes", default="5m,15m")
    parser.add_argument("--bars", type=int, default=6000)
    parser.add_argument("--minimum-bars", type=int, default=1000)
    parser.add_argument("--minimum-symbols", type=int, default=20)
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--registry", type=Path, default=Path("data/prediction_models"))
    parser.add_argument("--artifact-version")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/scalp_prediction_training.json"),
    )
    return parser.parse_args()


def run_training(
    *,
    symbols_file: Path = Path("config/nse/symbols.csv"),
    timeframes: str = "5m,15m",
    bars: int = 6000,
    minimum_bars: int = 1000,
    minimum_symbols: int = 20,
    max_symbols: int = 40,
    registry_path: Path = Path("data/prediction_models"),
    artifact_version: str | None = None,
    output: Path = Path("tmp/scalp_prediction_training.json"),
) -> int:
    """Core training logic, parameterized for reuse (e.g. by a scheduler) --
    separate from main()'s argv parsing so it never touches sys.argv."""
    settings = get_settings()
    store = CandleStore(
        settings.market_history_database_url or settings.database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
    )
    registry = PredictionArtifactRegistry(registry_path)
    strategy_registry = eligibility_registry_from_settings(settings)
    trainer = OfflinePredictionTrainer(registry)
    universe = _symbols(symbols_file)
    parsed_timeframes = list(dict.fromkeys(t.strip().lower() for t in timeframes.split(",") if t))
    base_version = artifact_version or datetime.now(UTC).strftime("scalp-panel-%Y%m%dT%H%M%SZ")
    report: dict[str, object] = {
        "trade_horizon": "SCALP",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "timeframes": {},
    }

    failures = 0
    for timeframe in parsed_timeframes:
        panel = {}
        for symbol in universe:
            frame = store.load_frame(
                symbol,
                timeframe,
                bars=max(minimum_bars, bars),
                complete_only=True,
            )
            if frame is None or len(frame) < minimum_bars:
                continue
            panel[symbol] = frame
            if max_symbols > 0 and len(panel) >= max_symbols:
                break
        if len(panel) < minimum_symbols:
            raise RuntimeError(
                f"{timeframe}: only {len(panel)} symbols have >= {minimum_bars} rows; "
                f"need {minimum_symbols}"
            )

        template_key = PredictionArtifactKey(
            strategy_version="shared_prediction_v1",
            timeframe=timeframe,
            trade_horizon="SCALP",
            regime="all",
            model_version=MODEL_VERSION,
            feature_version=FEATURE_VERSION,
        )
        version = f"{base_version}-{timeframe}"
        try:
            template_metadata = trainer.train_panel(
                panel,
                template_key,
                artifact_version=version,
            )
        except ArtifactRejectedError as exc:
            failures += 1
            report["timeframes"][timeframe] = {
                "status": "REJECTED",
                "reason": str(exc),
                "symbols_used": list(panel),
                "bars_requested_per_symbol": bars,
                "artifact_version": version,
                "exact_artifacts": [],
            }
            continue
        template = registry.load(template_key)
        admitted = [template_key.artifact_id]
        skipped_strategies: dict[str, str] = {}
        for strategy in StrategyType:
            strategy_version = strategy_registry.latest(strategy.value, timeframe)
            if strategy_version is None or strategy_version.validation_status not in {
                EligibilityStatus.PAPER_APPROVED,
                EligibilityStatus.LIVE_APPROVED,
            }:
                skipped_strategies[strategy.value] = (
                    "No paper-approved strategy eligibility record for this timeframe"
                )
                continue
            key = PredictionArtifactKey(
                strategy_version=strategy_version.model_version,
                timeframe=timeframe,
                trade_horizon="SCALP",
                regime="all",
                model_version=MODEL_VERSION,
                feature_version=FEATURE_VERSION,
            )
            metadata = registry.metadata(
                key,
                oos_samples=template_metadata.oos_samples,
                folds_used=template_metadata.folds_used,
                calibration_by_regime=template_metadata.calibration_by_regime,
                artifact_version=version,
            )
            registry.save(
                PredictionArtifact(
                    metadata=metadata,
                    scaler=template.scaler,
                    models=template.models,
                    weights=template.weights,
                    calibrator=template.calibrator,
                )
            )
            # Verify checksum and core semantic identity immediately.
            registry.load(key)
            admitted.append(key.artifact_id)

        report["timeframes"][timeframe] = {
            "status": "VALIDATED",
            "symbols_used": list(panel),
            "bars_requested_per_symbol": bars,
            "artifact_version": version,
            "oos_samples": template_metadata.oos_samples,
            "folds_used": template_metadata.folds_used,
            "calibration_by_regime": template_metadata.calibration_by_regime,
            "exact_artifacts": admitted,
            "skipped_strategies": skipped_strategies,
        }

    report["offline_training_count"] = trainer.training_count
    report["offline_walk_forward_count"] = trainer.walk_forward_count
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 1 if failures else 0


def main() -> int:
    args = _arguments()
    return run_training(
        symbols_file=args.symbols_file,
        timeframes=args.timeframes,
        bars=args.bars,
        minimum_bars=args.minimum_bars,
        minimum_symbols=args.minimum_symbols,
        max_symbols=args.max_symbols,
        registry_path=args.registry,
        artifact_version=args.artifact_version,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
