"""Benchmark the shared-feature strategy path over the configured NSE universe.

This script reads settled local Timescale candles and runs in SIMULATED eligibility
mode. It performs no external Qwen call, broker request, ML training, or order action.
New production strategies remain SHADOW unless separately approved offline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.production_strategy_registration import (
    ensure_production_strategy_placeholders,
)
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    eligibility_registry_from_settings,
)
from src.config import get_settings
from src.core.aggregation import (
    FeatureSnapshot,
    aggregate_strategy_signals,
    eligible_strategy_types,
    evaluate_registered_strategies,
)
from src.core.candidates import SignalEngine, SignalType
from src.core.candles import CandleStore
from src.core.features import build_market_relative_context
from src.core.indicators import FEATURE_SET_VERSION, IndicatorCache, Timeframe
from src.core.strategies.config import ProductionStrategyConfig
from src.markets.nse.market_data.history_manager import resample_nse_ohlcv
from src.markets.nse.persistence import bind_candle_repository
from src.markets.nse.strategies import build_strategy_config


def _symbols(path: str) -> list[str]:
    return [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip().lower() != "symbol"
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _run_timeframe(
    timeframe: Timeframe,
    frames: dict[str, Any],
    *,
    runs: int,
    engine: SignalEngine,
    registry: Any,
    production_config: ProductionStrategyConfig,
    memory_profile: bool,
) -> dict[str, Any]:
    run_metrics: list[dict[str, Any]] = []
    for run_index in range(runs):
        # Use a distinct settled candle for each measured cycle so every run is a
        # genuine event rather than an unchanged-candle cache hit.
        shift = runs - run_index - 1
        selected = {
            symbol: frame.iloc[: len(frame) - shift] if shift else frame
            for symbol, frame in frames.items()
            if frame is not None and len(frame) > 210 + shift
        }
        cpu_started = time.process_time()
        total_started = time.perf_counter()
        if memory_profile:
            tracemalloc.start()

        feature_started = time.perf_counter()
        cache = IndicatorCache()
        indicators = {
            symbol: cache.get_or_compute(frame, symbol, timeframe)
            for symbol, frame in selected.items()
        }
        feature_ms = (time.perf_counter() - feature_started) * 1000.0

        context_started = time.perf_counter()
        relative = build_market_relative_context(
            indicators,
            benchmark_symbol=production_config.benchmark_symbol,
            fallback=production_config.benchmark_fallback,
        )
        context_ms = (time.perf_counter() - context_started) * 1000.0

        strategy_started = time.perf_counter()
        registered = []
        evaluations = 0
        for symbol in sorted(indicators):
            indicator = indicators[symbol]
            snapshot = FeatureSnapshot.create(
                indicator,
                settled_candle_timestamp=indicator.settled_candle_timestamp,
                feature_version=FEATURE_SET_VERSION,
                market_relative=relative.get(symbol),
            )
            evaluations += len(eligible_strategy_types(engine, timeframe))
            registered.extend(
                evaluate_registered_strategies(
                    engine,
                    snapshot,
                    registry,
                    trade_horizon="SWING",
                    regime="unknown",
                    execution_mode="local_paper",
                    eligibility_environment=EligibilityEnvironment.SIMULATED,
                )
            )
        strategy_ms = (time.perf_counter() - strategy_started) * 1000.0

        aggregation_started = time.perf_counter()
        candidates = aggregate_strategy_signals(registered)
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0

        # Eligibility placeholders are SHADOW. Therefore ML artifacts are neither
        # required nor invoked, Qwen is deterministically ineligible, and executable
        # risk receives no candidate. The measured zeroes are safety behavior, not
        # omitted simulated approvals.
        ml_started = time.perf_counter()
        ml_evaluated = sum(item.ml_required and item.pipeline_eligible for item in registered)
        ml_ms = (time.perf_counter() - ml_started) * 1000.0
        qwen_started = time.perf_counter()
        qwen_required = sum(
            bool(item.representative_signal.registry_qwen_allowed) for item in candidates
        )
        qwen_ms = (time.perf_counter() - qwen_started) * 1000.0
        risk_started = time.perf_counter()
        risk_candidates = [item for item in candidates if item.execution_allowed]
        risk_approved = 0
        risk_ms = (time.perf_counter() - risk_started) * 1000.0

        if memory_profile:
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            current_bytes, peak_bytes = 0, 0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        cpu_ms = (time.process_time() - cpu_started) * 1000.0
        run_metrics.append(
            {
                "run": run_index + 1,
                "symbols_scanned": len(indicators),
                "feature_snapshots": len(indicators),
                "strategy_evaluations": evaluations,
                "technical_buy": sum(
                    item.signal.signal_type is SignalType.BUY for item in registered
                ),
                "technical_sell": sum(
                    item.signal.signal_type is SignalType.SELL for item in registered
                ),
                "consolidated_candidates": len(candidates),
                "conflicts": sum(item.conflict_level != "NONE" for item in candidates),
                "eligibility_passed": sum(item.pipeline_eligible for item in candidates),
                "regime_blocked": sum(
                    item.eligibility_decision.regime_policy.value == "BLOCK" for item in registered
                ),
                "shadow_candidates": sum(item.shadow_only for item in candidates),
                "ml_evaluated": ml_evaluated,
                "ml_qualified": 0,
                "qwen_required": qwen_required,
                "qwen_external_calls": 0,
                "risk_candidates": len(risk_candidates),
                "risk_approved": risk_approved,
                "final_buy": 0,
                "final_sell": 0,
                "feature_ms": round(feature_ms, 3),
                "cross_sectional_context_ms": round(context_ms, 3),
                "strategy_ms": round(strategy_ms, 3),
                "aggregation_ms": round(aggregation_ms, 3),
                "ml_ms": round(ml_ms, 3),
                "qwen_ms": round(qwen_ms, 3),
                "risk_ms": round(risk_ms, 3),
                "total_ms": round(total_ms, 3),
                "cpu_ms": round(cpu_ms, 3),
                "peak_python_mb": round(peak_bytes / (1024 * 1024), 3),
                "retained_python_mb": round(current_bytes / (1024 * 1024), 3),
            }
        )
    totals = [float(item["total_ms"]) for item in run_metrics]
    return {
        "timeframe": timeframe.value,
        "runs": run_metrics,
        "p50_total_ms": round(statistics.median(totals), 3),
        "p95_total_ms": round(_percentile(totals, 0.95), 3),
        "max_total_ms": round(max(totals), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", default="config/nse/symbols.csv")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--bars", type=int, default=260)
    parser.add_argument("--timeframes", default="15m,30m,1h,4h")
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable tracemalloc (useful for memory, but intentionally excluded from clean timing)",
    )
    args = parser.parse_args()

    settings = get_settings()
    production_config = build_strategy_config(settings)
    timeframes = [Timeframe(value.strip()) for value in args.timeframes.split(",")]
    symbols = _symbols(args.symbols_file)
    raw_store = CandleStore(
        settings.market_history_database_url or settings.database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
        schema=str(getattr(settings, "nse_db_schema", "nse")),
    )
    store = bind_candle_repository(raw_store)
    registry = eligibility_registry_from_settings(
        settings,
        runtime_timeframes=[timeframe.value for timeframe in timeframes],
    )
    created = ensure_production_strategy_placeholders(registry, production_config)
    engine = SignalEngine(production_config=production_config)

    required_bars = args.bars + args.runs
    # Two native Timescale reads per symbol feed all four requested timeframes.
    # Derived candles use the exact shared NSE-session resampler.
    native_15m = {
        symbol: store.load_frame(
            symbol,
            "15m",
            bars=required_bars * 2,
            complete_only=True,
        )
        for symbol in symbols
    }
    native_1h = {
        symbol: store.load_frame(
            symbol,
            "1h",
            bars=required_bars * 4,
            complete_only=True,
        )
        for symbol in symbols
    }
    available_15m = {
        symbol: frame for symbol, frame in native_15m.items() if frame is not None
    }
    available_1h = {symbol: frame for symbol, frame in native_1h.items() if frame is not None}
    frames_by_timeframe = {
        Timeframe.M15: {
            symbol: frame.tail(required_bars) for symbol, frame in available_15m.items()
        },
        Timeframe.M30: {
            symbol: resample_nse_ohlcv(frame, "30min").tail(required_bars)
            for symbol, frame in available_15m.items()
        },
        Timeframe.H1: {
            symbol: frame.tail(required_bars) for symbol, frame in available_1h.items()
        },
        Timeframe.H4: {
            symbol: resample_nse_ohlcv(frame, "4h").tail(required_bars)
            for symbol, frame in available_1h.items()
        },
    }

    results = []
    for timeframe in timeframes:
        frames = frames_by_timeframe[timeframe]
        results.append(
            _run_timeframe(
                timeframe,
                frames,
                runs=args.runs,
                engine=engine,
                registry=registry,
                production_config=production_config,
                memory_profile=args.memory_profile,
            )
        )
    print(
        json.dumps(
            {
                "mode": "SIMULATED_SHADOW_BENCHMARK",
                "broker_orders": False,
                "qwen_external_calls": 0,
                "model_training": 0,
                "symbols_requested": len(symbols),
                "symbols_with_complete_frames": min(
                    sum(not frame.empty for frame in frames.values())
                    for frames in frames_by_timeframe.values()
                ),
                "memory_profile": args.memory_profile,
                "placeholder_grains_created": len(created),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
