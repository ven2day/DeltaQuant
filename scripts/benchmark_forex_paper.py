"""Benchmark real OANDA Practice data through the local-paper Forex pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.prediction import PredictionAgent
from src.config import get_settings
from src.core.candidates import SignalEngine
from src.core.candles import CandleStore
from src.markets.forex.eligibility import create_registry
from src.markets.forex.market_data import create_forex_market_data_provider
from src.markets.forex.ml import create_artifact_registry
from src.markets.forex.persistence import bind_candle_repository, bind_trading_repository
from src.markets.forex.risk import ForexRiskLimits
from src.markets.forex.runtime.cycle import ForexSettledCandleCycle
from src.markets.forex.strategies import build_strategy_config


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


async def _run(iterations: int) -> dict[str, Any]:
    settings = get_settings()
    if settings.market != "FOREX" or settings.oanda_environment != "practice":
        raise RuntimeError("Benchmark accepts MARKET=FOREX/OANDA Practice only")
    if settings.forex_execution_enabled or settings.allow_live_orders:
        raise RuntimeError("Benchmark requires all OANDA broker-order gates disabled")
    provider = create_forex_market_data_provider(settings)
    try:
        health = await provider.validate_connectivity()
        if not health.healthy:
            raise RuntimeError(health.message)
        instruments = await provider.list_instruments()
        await provider.start_price_stream([item.symbol for item in instruments])
        await provider.wait_for_stream_message(
            float(settings.oanda_stream_startup_timeout_seconds)
        )

        raw_store = CandleStore(
            settings.market_history_database_url or settings.database_url,
            enable_timescale=settings.market_history_enable_timescale,
            require_timescale=settings.market_history_require_timescale,
            schema=str(settings.forex_db_schema),
        )
        candle_repository = bind_candle_repository(raw_store)
        trading_repository = bind_trading_repository(raw_store.engine)
        config = build_strategy_config(settings)
        registry = create_registry(settings)
        prediction = PredictionAgent(artifact_registry=create_artifact_registry(settings))
        runs: list[dict[str, Any]] = []
        cpu_started = process_time()
        wall_started = perf_counter()
        for _ in range(max(1, iterations)):
            cycle = ForexSettledCandleCycle(
                provider=provider,
                signal_engine=SignalEngine(production_config=config),
                strategy_config=config,
                eligibility_registry=registry,
                risk_limits=ForexRiskLimits.from_settings(settings),
                candle_store=candle_repository,
                trading_repository=trading_repository,
                prediction_agent=prediction,
                settings=settings,
            )
            result = await cycle.run()
            runs.append(result.metrics.to_dict())
        wall_seconds = perf_counter() - wall_started
        phase_fields = (
            "market_data_duration_ms",
            "feature_duration_ms",
            "strategy_duration_ms",
            "aggregation_duration_ms",
            "ml_duration_ms",
            "qwen_duration_ms",
            "risk_duration_ms",
            "persistence_duration_ms",
            "total_duration_ms",
        )
        return {
            "market": "FOREX",
            "provider": "OANDA",
            "environment": "PRACTICE",
            "execution": "PAPER",
            "broker_orders": "OFF",
            "iterations": len(runs),
            "stream_state": provider.stream_health_state().value,
            "cpu_seconds": process_time() - cpu_started,
            "wall_seconds": wall_seconds,
            "phases": {
                field: _summary([float(run[field]) for run in runs]) for field in phase_fields
            },
            "runs": runs,
        }
    finally:
        await provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", default="run/forex_paper_benchmark.json")
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    report = asyncio.run(_run(arguments.iterations))
    output = ROOT / arguments.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
