"""Train validated Forex prediction artifacts offline; never part of live runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.offline_prediction import OfflinePredictionTrainer
from src.agents.prediction import FEATURE_VERSION, MODEL_VERSION
from src.backtesting.strategy_eligibility import EligibilityStatus
from src.config import get_settings
from src.core.candles import CandleStore
from src.core.ml import ArtifactRejectedError, PredictionArtifactKey
from src.markets.forex.eligibility import create_registry
from src.markets.forex.ml import create_artifact_registry
from src.markets.forex.persistence import bind_candle_repository
from src.markets.forex.strategies import build_strategy_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bars", type=int, default=5000)
    parser.add_argument("--report", default="run/forex_ml_training_report.json")
    return parser


def _run(args: argparse.Namespace) -> dict[str, object]:
    settings = get_settings()
    if settings.market != "FOREX" or settings.oanda_environment != "practice":
        raise RuntimeError("Forex ML training accepts OANDA Practice configuration only")
    if settings.forex_execution_enabled or settings.allow_live_orders:
        raise RuntimeError("Forex ML training requires broker execution disabled")
    eligibility = create_registry(settings)
    production_config = build_strategy_config(settings)
    approved = [
        record
        for record in eligibility.load_all()
        if record.market == "FOREX"
        and record.validation_status is EligibilityStatus.PAPER_APPROVED
        and record.model_version == production_config.version
    ]
    approved_timeframes = sorted({record.timeframe for record in approved})
    artifact_registry = create_artifact_registry(settings)
    trainer = OfflinePredictionTrainer(artifact_registry)
    store = CandleStore(
        settings.market_history_database_url or settings.database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
        schema=str(settings.forex_db_schema),
    )
    repository = bind_candle_repository(store)
    symbols = [item.strip().upper() for item in settings.forex_symbols.split(",") if item.strip()]
    results: list[dict[str, object]] = []
    for timeframe in approved_timeframes:
        panel = {}
        for symbol in symbols:
            frame = repository.load_frame(
                symbol, timeframe, bars=max(300, int(args.max_bars)), complete_only=True
            )
            if not frame.empty:
                panel[symbol] = frame
        key = PredictionArtifactKey(
            market="FOREX",
            instrument_scope="*",
            strategy_version=production_config.version,
            timeframe=timeframe,
            trade_horizon="SWING",
            regime="all",
            model_version=MODEL_VERSION,
            feature_version=FEATURE_VERSION,
        )
        try:
            metadata = trainer.train_panel(panel, key)
            results.append(
                {
                    "timeframe": timeframe,
                    "status": "VALIDATED",
                    "artifact_id": key.artifact_id,
                    "artifact_version": metadata.artifact_version,
                    "oos_samples": metadata.oos_samples,
                    "folds_used": metadata.folds_used,
                }
            )
        except (ArtifactRejectedError, FileExistsError) as exc:
            results.append(
                {
                    "timeframe": timeframe,
                    "status": "ABSTAIN",
                    "reason": type(exc).__name__,
                    "artifact_id": key.artifact_id,
                }
            )
    result: dict[str, object] = {
        "market": "FOREX",
        "environment": "PRACTICE",
        "execution": "OFFLINE_TRAINING_ONLY",
        "broker_orders": "OFF",
        "paper_approved_grains": len(approved),
        "training_count": trainer.training_count,
        "walk_forward_count": trainer.walk_forward_count,
        "artifacts": results,
    }
    report_path = ROOT / str(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(_run(_parser().parse_args()), indent=2, sort_keys=True))
