#!/usr/bin/env python3
"""Reproducible benchmark for the stateful 272-symbol event runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.candidate_cache import (
    CandidateVerdict,
    CandidateVerdictCache,
    build_candidate_fingerprint,
)
from src.agents.context_review import QwenReviewPolicy
from src.agents.model_artifacts import (
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactRegistry,
)
from src.agents.prediction import FEATURE_VERSION, MODEL_VERSION, PredictionAgent
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    RegimePolicy,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.backtesting.strategy_registry import StrategyRegistry
from src.core.aggregation import (
    FeatureSnapshot,
    aggregate_strategy_signals,
    evaluate_registered_strategies,
)
from src.core.candidates import SignalEngine, StrategyType
from src.core.indicators import FEATURE_SET_VERSION, Timeframe, get_indicator_cache
from src.markets.nse.execution.exit_manager import ExitManager
from src.markets.nse.risk.pretrade import FinalPaperOrder, PaperRiskReservations
from src.markets.nse.runtime.state import MarketStateStore
from src.markets.nse.strategies.candidate_policy import CandidateAction, evaluate_long_candidate
from src.markets.nse.strategies.signal_ranking import select_stage1_candidates, technical_pre_rank


class _IdentityScaler:
    def transform(self, values):
        return values


class _ConstantModel:
    def predict(self, values):
        return np.full(len(values), 0.01)


@dataclass(frozen=True)
class _Performance:
    total_trades: int = 0
    winning_trades: int = 0


class _Tracker:
    def get_strategy_performance(
        self, strategy: str, regime: str, lookback_days: int = 30
    ) -> _Performance:
        return _Performance()


def _frame(seed: int, timeframe: Timeframe, bars: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0008 if seed % 2 == 0 else -0.0004, 0.006, bars)
    close = 100 * np.cumprod(1 + returns)
    spread = rng.uniform(0.001, 0.008, bars)
    frequency = {
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
    }[timeframe]
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(50_000, 2_000_000, bars),
        },
        index=pd.date_range(
            end="2026-08-10 10:00", periods=bars, freq=frequency, tz="Asia/Kolkata"
        ),
    )


def _append(frame: pd.DataFrame, timeframe: Timeframe, cycle: int) -> pd.DataFrame:
    delta = {
        Timeframe.M5: pd.Timedelta(minutes=5),
        Timeframe.M15: pd.Timedelta(minutes=15),
        Timeframe.M30: pd.Timedelta(minutes=30),
        Timeframe.H1: pd.Timedelta(hours=1),
        Timeframe.H4: pd.Timedelta(hours=4),
    }[timeframe]
    previous = frame.iloc[-1]
    close = float(previous.close) * (1.001 + cycle * 0.00001)
    row = pd.DataFrame(
        {
            "open": [float(previous.close)],
            "high": [close * 1.002],
            "low": [close * 0.998],
            "close": [close],
            "volume": [int(previous.volume)],
        },
        index=[frame.index[-1] + delta],
    )
    return pd.concat([frame, row]).iloc[-240:]


def _peak_ram_mb() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        success = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1024 / 1024
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 if sys.platform != "darwin" else 1024 * 1024)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile)) if values else 0.0


def _admit_artifacts(registry: PredictionArtifactRegistry, timeframes: list[Timeframe]) -> None:
    for strategy in StrategyType:
        for timeframe in timeframes:
            key = PredictionArtifactKey(
                strategy_version=f"{strategy.value}-benchmark-v1",
                timeframe=timeframe.value,
                trade_horizon="SCALP",
                regime="all",
                model_version=MODEL_VERSION,
                feature_version=FEATURE_VERSION,
            )
            metadata = registry.metadata(
                key,
                oos_samples=100,
                folds_used=5,
                calibration_by_regime={"ranging": 0.2},
                artifact_version="benchmark-v1",
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


def _admit_strategies(
    registry: StrategyEligibilityRegistry, timeframes: list[Timeframe]
) -> None:
    for strategy in StrategyType:
        for timeframe in timeframes:
            registry.register(
                StrategyEligibility(
                    strategy_name=strategy.value,
                    timeframe=timeframe.value,
                    model_version=f"{strategy.value}-benchmark-v1",
                    validation_status=EligibilityStatus.PAPER_APPROVED,
                    validated_at="2026-08-01T00:00:00+00:00",
                    validation_window={"dataset": "synthetic-performance-fixture"},
                    oos_trade_count=100,
                    oos_profit_factor=1.4,
                    oos_max_drawdown=0.08,
                    oos_win_rate=0.58,
                    minimum_model_confidence=0.0,
                    status_reason="Synthetic benchmark record; never used for trading",
                )
            )


def _process_event_cycle(
    *,
    frames: dict[tuple[str, Timeframe], pd.DataFrame],
    state: MarketStateStore,
    prediction_agent: PredictionAgent,
    strategy_registry: StrategyEligibilityRegistry,
    comparison_legacy_registry: StrategyRegistry,
    comparison_eligibility_registry: StrategyEligibilityRegistry,
    qwen_cache: CandidateVerdictCache,
) -> dict[str, object]:
    started = perf_counter()
    cpu_started = process_time()
    events, wake = state.plan_events(frames)
    feature_started = perf_counter()
    normalized_signals = []
    indicator_cache = get_indicator_cache()
    signal_engine = SignalEngine()
    for event in events:
        feature = indicator_cache.get_or_compute(
            event.frame, event.symbol, timeframe=event.timeframe
        )
        state.mark_processed(event, features=feature, feature_version=FEATURE_SET_VERSION)
        snapshot = FeatureSnapshot.create(
            feature,
            settled_candle_timestamp=event.candle_timestamp,
            feature_version=FEATURE_SET_VERSION,
        )
        normalized_signals.extend(
            evaluate_registered_strategies(
                signal_engine,
                snapshot,
                strategy_registry,
                trade_horizon="SCALP",
                regime="ranging",
                execution_mode="local_paper",
                eligibility_environment=EligibilityEnvironment.SIMULATED,
            )
        )
    feature_seconds = perf_counter() - feature_started

    aggregated = aggregate_strategy_signals(normalized_signals)
    legacy_strict_admitted = 0
    simulated_eligible = 0
    paper_executable = 0
    comparison_regime_blocked = 0
    comparison_confidence_reduced = 0
    for item in normalized_signals:
        signal = item.signal
        legacy_strict_admitted += int(
            comparison_legacy_registry.is_admitted(
                signal.strategy.value,
                regime="ranging",
                symbol=signal.symbol,
                timeframe=signal.timeframe.value,
                trade_horizon="SCALP",
            )
        )
        simulated = comparison_eligibility_registry.evaluate(
            strategy_name=signal.strategy.value,
            timeframe=signal.timeframe.value,
            environment=EligibilityEnvironment.SIMULATED,
            current_regime="ranging",
            confidence=signal.confidence,
        )
        paper = comparison_eligibility_registry.evaluate(
            strategy_name=signal.strategy.value,
            timeframe=signal.timeframe.value,
            environment=EligibilityEnvironment.PAPER,
            current_regime="ranging",
            confidence=signal.confidence,
        )
        simulated_eligible += int(simulated.pipeline_allowed)
        paper_executable += int(paper.execution_allowed)
        comparison_regime_blocked += int(
            simulated.regime_policy is RegimePolicy.BLOCK
        )
        comparison_confidence_reduced += int(simulated.confidence_multiplier < 1.0)
    signals = [
        candidate.apply_lineage()
        for candidate in aggregated
        if candidate.pipeline_eligible and candidate.direction.value == "BUY"
    ]

    scan_started = perf_counter()
    technical_setups = select_stage1_candidates(
        technical_pre_rank(signals, _Tracker()),
        max_candidates=max(1, len(signals)),
        max_per_sector=max(1, len(signals)),
    )
    scan_seconds = perf_counter() - scan_started

    qualification_started = perf_counter()
    histories_by_symbol: dict[str, dict[Timeframe, pd.DataFrame]] = {}
    ml_candidates = []
    for item in technical_setups:
        signal = item.signal
        histories = histories_by_symbol.setdefault(
            signal.symbol,
            {
                timeframe: frames[(signal.symbol, timeframe)]
                for timeframe in (
                    Timeframe.M5,
                    Timeframe.M15,
                    Timeframe.M30,
                    Timeframe.H1,
                    Timeframe.H4,
                )
            },
        )
        decision = evaluate_long_candidate(
            signal.to_dict(), histories, None, require_ml=False
        )
        if decision.action == CandidateAction.BUY:
            ml_candidates.append(item)
    qualification_seconds = perf_counter() - qualification_started

    ml_before = prediction_agent.cache_metrics()
    ml_started = perf_counter()
    predictions = {}
    for item in ml_candidates:
        signal = item.signal
        prediction = prediction_agent.predict_cached(
            frames[(signal.symbol, signal.timeframe)].rename(columns=str.title),
            signal.symbol,
            timeframe=signal.timeframe.value,
            trade_horizon="SCALP",
            strategy_version=signal.strategy_version,
        )
        predictions[(signal.symbol, signal.timeframe.value)] = prediction
    ml_seconds = perf_counter() - ml_started
    ml_after = prediction_agent.cache_metrics()

    ml_qualified = []
    for item in ml_candidates:
        signal = item.signal
        decision = evaluate_long_candidate(
            signal.to_dict(),
            histories_by_symbol[signal.symbol],
            predictions[(signal.symbol, signal.timeframe.value)],
            require_ml=signal.ml_required,
        )
        if decision.action == CandidateAction.BUY:
            ml_qualified.append(item)

    def frequency_key(item) -> str:
        signal = item.signal
        return f"{signal.strategy.value}|{signal.timeframe.value}|SCALP"

    raw_frequency = Counter(
        f"{signal.strategy_name}|{signal.signal.timeframe.value}|SCALP"
        for signal in normalized_signals
    )
    qualified_frequency = Counter(frequency_key(item) for item in ml_candidates)
    ml_qualified_frequency = Counter(frequency_key(item) for item in ml_qualified)

    qwen_started = perf_counter()
    qwen_calls = 0
    qwen_required = 0
    qwen_skipped_clear = 0
    qwen_skipped_shadow = 0
    qwen_deferred = 0
    qwen_hits_before = qwen_cache.hits
    qwen_misses: list[str] = []
    qwen_miss_keys: list[str] = []
    qwen_miss_item_ids: list[int] = []
    qwen_required_frequency: Counter[str] = Counter()
    qwen_reviewed_frequency: Counter[str] = Counter()
    review_policy = QwenReviewPolicy(regime_uncertain_below=0.6, ml_conflict_below=0.45)
    for item in ml_qualified:
        signal = item.signal
        prediction = predictions[(signal.symbol, signal.timeframe.value)]
        payload = signal.to_dict()
        payload.update(
            {
                "signal_candle_timestamp": str(frames[(signal.symbol, signal.timeframe)].index[-1]),
                "strategy_version": signal.strategy_version,
                "ml_status": "VALIDATED" if not prediction.abstained else "ABSTAIN",
                "ml_model_version": prediction.model_version,
                "ml_probability": prediction.confidence,
            }
        )
        if not signal.registry_qwen_allowed:
            qwen_skipped_shadow += 1
            continue
        review = review_policy.assess(
            payload,
            regime_confidence=0.9,
            shared_context={"major_market_event_risk": "NONE"},
        )
        if not review.required:
            qwen_skipped_clear += 1
            continue
        qwen_required += 1
        qwen_required_frequency[frequency_key(item)] += 1
        fingerprint = build_candidate_fingerprint(
            payload,
            regime="ranging",
            market_context_version="benchmark-context-v1",
        )
        if qwen_cache.get(fingerprint) is None:
            qwen_misses.append(fingerprint)
            qwen_miss_keys.append(frequency_key(item))
            qwen_miss_item_ids.append(id(item))
    selected_qwen_misses = qwen_misses[:20]
    selected_qwen_keys = qwen_miss_keys[:20]
    deferred_item_ids = set(qwen_miss_item_ids[20:])
    qwen_deferred = max(0, len(qwen_misses) - len(selected_qwen_misses))
    if selected_qwen_misses:
        # signal_validation_node uses one schema-constrained batch request for the
        # event. This benchmark stubs that network boundary but models call count,
        # cache admission, and capacity exactly.
        qwen_calls = 1
        qwen_reviewed_frequency.update(selected_qwen_keys)
        for fingerprint in selected_qwen_misses:
            qwen_cache.set(
                CandidateVerdict.create(
                    fingerprint=fingerprint,
                    decision="BUY",
                    confidence=0.7,
                    reason_codes=["BENCHMARK_STUB"],
                    risk_flags=[],
                    ttl_seconds=3600,
                    model_used="qwen-benchmark-boundary",
                    market_context_version="benchmark-context-v1",
                    symbol_context_version="",
                )
            )
    qwen_seconds = perf_counter() - qwen_started

    risk_started = perf_counter()
    reservations = PaperRiskReservations()
    risk_decisions = 0
    risk_approved_frequency: Counter[str] = Counter()
    risk_rejected_frequency: Counter[str] = Counter()
    for item in ml_qualified:
        if id(item) in deferred_item_ids:
            continue
        signal = item.signal
        order = FinalPaperOrder(
            symbol=signal.symbol,
            side="BUY",
            quantity=1,
            entry_price=signal.entry_price,
            stop_loss=min(signal.stop_loss, signal.entry_price * 0.99),
            target_price=max(signal.target_price, signal.entry_price * 1.02),
            strategy=signal.strategy.value,
        )
        risk_decision = reservations.evaluate(
            order,
            positions=[],
            equity=1_000_000,
            entries_today=0,
            daily_entry_cap=10_000,
            max_positions=10_000,
            max_position_pct=1.0,
            max_total_exposure_pct=100.0,
            risk_per_trade=0.002,
            max_total_risk=1.0,
            realized_daily_pnl=0,
            daily_loss_limit=100_000,
            daily_loss_buffer=0.8,
        )
        target_counter = (
            risk_approved_frequency if risk_decision.approved else risk_rejected_frequency
        )
        target_counter[frequency_key(item)] += 1
        risk_decisions += 1
    risk_seconds = perf_counter() - risk_started
    total = perf_counter() - started
    cpu_seconds = process_time() - cpu_started
    frequency_keys = sorted(
        set(raw_frequency)
        | set(qualified_frequency)
        | set(ml_qualified_frequency)
        | set(qwen_required_frequency)
        | set(risk_approved_frequency)
        | set(risk_rejected_frequency)
    )
    return {
        "symbols_examined": wake.symbols_considered,
        "changed_events": wake.changed_events,
        "features_updated": len(events),
        "strategy_evaluations": len(events) * len(StrategyType),
        "raw_strategy_signals": len(normalized_signals),
        "eligibility_comparison": {
            "legacy_strict_admitted": legacy_strict_admitted,
            "new_simulated_eligible": simulated_eligible,
            "new_paper_executable": paper_executable,
            "new_regime_blocked": comparison_regime_blocked,
            "new_confidence_reduced": comparison_confidence_reduced,
        },
        "registry_eligible": sum(
            item.eligibility_decision.pipeline_allowed for item in normalized_signals
        ),
        "registry_blocked": sum(
            not item.eligibility_decision.pipeline_allowed for item in normalized_signals
        ),
        "regime_blocked": sum(
            item.eligibility_decision.regime_policy.value == "BLOCK"
            for item in normalized_signals
        ),
        "confidence_reduced": sum(
            item.eligibility_decision.confidence_multiplier < 1.0
            for item in normalized_signals
        ),
        "aggregated_candidates": len(aggregated),
        "technical_setups": len(technical_setups),
        "technical_candidates": len(ml_candidates),
        "ml_candidates": len(ml_candidates),
        "ml_qualified": len(ml_qualified),
        "ml_inference_count": ml_after.inference_count - ml_before.inference_count,
        "ml_cache_hits": ml_after.hits - ml_before.hits,
        "qwen_calls": qwen_calls,
        "qwen_cache_hits": qwen_cache.hits - qwen_hits_before,
        "qwen_candidates": qwen_required,
        "qwen_skipped_clear": qwen_skipped_clear,
        "qwen_skipped_shadow": qwen_skipped_shadow,
        "qwen_deferred": qwen_deferred,
        "qwen_input_tokens": 0,
        "qwen_output_tokens": 0,
        "risk_decisions": risk_decisions,
        "risk_approved": sum(risk_approved_frequency.values()),
        "risk_blocked": sum(risk_rejected_frequency.values()),
        "execution_count": 0,
        "shadow_count": sum(risk_approved_frequency.values()),
        "signal_frequency": {
            key: {
                "raw": raw_frequency[key],
                "qualified": qualified_frequency[key],
                "ml_qualified": ml_qualified_frequency[key],
                "qwen_required": qwen_required_frequency[key],
                "qwen_reviewed": qwen_reviewed_frequency[key],
                "risk_approved": risk_approved_frequency[key],
                "risk_rejected": risk_rejected_frequency[key],
            }
            for key in frequency_keys
        },
        "feature_update_seconds": feature_seconds,
        "technical_scan_seconds": scan_seconds,
        "qualification_seconds": qualification_seconds,
        "ml_seconds": ml_seconds,
        "qwen_seconds": qwen_seconds,
        "risk_seconds": risk_seconds,
        "total_seconds": total,
        "cpu_seconds": cpu_seconds,
        "cpu_percent_of_machine": (
            cpu_seconds / total * 100.0 / max(1, os.cpu_count() or 1) if total else 0.0
        ),
        "peak_ram_mb": _peak_ram_mb(),
    }


def _position_benchmark(iterations: int = 100) -> dict[str, float | int]:
    manager = ExitManager(state_file=None)
    for index in range(20):
        manager.register_position(
            position_id=f"P{index}",
            symbol=f"BENCH{index:03d}",
            side="BUY",
            quantity=1,
            entry_price=100.0,
            stop_loss=95.0,
            target_price=110.0,
        )
    values = []
    for _ in range(iterations):
        started = perf_counter()
        manager.check_exits({f"BENCH{index:03d}": 100.5 for index in range(20)})
        values.append(perf_counter() - started)
    return {
        "iterations": iterations,
        "positions_monitored": 20,
        "p50_ms": median(values) * 1000,
        "p95_ms": _percentile(values, 95) * 1000,
        "max_ms": max(values) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=272)
    parser.add_argument(
        "--symbols-file", type=Path, default=Path("config/nse/symbols.csv")
    )
    parser.add_argument(
        "--comparison-legacy-registry",
        type=Path,
        default=Path("data/nse/strategy_registry"),
    )
    parser.add_argument(
        "--comparison-eligibility-registry",
        type=Path,
        default=Path("data/nse/strategy_eligibility"),
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.symbols_file.exists():
        universe = pd.read_csv(args.symbols_file)
        column = "Symbol" if "Symbol" in universe.columns else universe.columns[0]
        symbols = list(
            dict.fromkeys(
                str(value).strip().upper()
                for value in universe[column].dropna()
                if str(value).strip()
            )
        )[: args.symbols]
        universe_source = str(args.symbols_file.resolve())
    else:
        symbols = [f"BENCH{index:03d}" for index in range(args.symbols)]
        universe_source = "synthetic-symbol-names"
    timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4]
    frames = {
        (symbol, timeframe): _frame(index * 10 + offset, timeframe)
        for index, symbol in enumerate(symbols)
        for offset, timeframe in enumerate(timeframes)
    }

    with tempfile.TemporaryDirectory(prefix="deltaquant-benchmark-") as directory:
        registry = PredictionArtifactRegistry(directory)
        _admit_artifacts(registry, timeframes)
        strategy_registry = StrategyEligibilityRegistry(Path(directory) / "strategies")
        _admit_strategies(strategy_registry, timeframes)
        agent = PredictionAgent(cache_size=max(4096, args.symbols * len(timeframes) * 4), artifact_registry=registry)
        state = MarketStateStore()
        qwen_cache = CandidateVerdictCache()
        comparison_legacy_registry = StrategyRegistry(args.comparison_legacy_registry)
        comparison_eligibility_registry = StrategyEligibilityRegistry(
            args.comparison_eligibility_registry
        )
        # Cold state hydration is outside measured live events.
        initial_events, _ = state.plan_events(frames)
        cache = get_indicator_cache()
        engine = SignalEngine()
        for event in initial_events:
            feature = cache.get_or_compute(event.frame, event.symbol, timeframe=event.timeframe)
            state.mark_processed(event, features=feature, feature_version=FEATURE_SET_VERSION)
            engine.generate_signals(feature)

        no_change = _process_event_cycle(
            frames=frames,
            state=state,
            prediction_agent=agent,
            strategy_registry=strategy_registry,
            comparison_legacy_registry=comparison_legacy_registry,
            comparison_eligibility_registry=comparison_eligibility_registry,
            qwen_cache=qwen_cache,
        )
        scenarios: dict[str, list[dict[str, float | int]]] = {}
        scenario_timeframes = {
            "new_5m": [Timeframe.M5],
            "new_15m": [Timeframe.M15],
            "new_30m": [Timeframe.M30],
            "simultaneous_5m_15m_30m": [Timeframe.M5, Timeframe.M15, Timeframe.M30],
            "new_1h": [Timeframe.H1],
        }
        for name, changed_timeframes in scenario_timeframes.items():
            rows = []
            for cycle in range(args.cycles):
                for symbol in symbols:
                    for timeframe in changed_timeframes:
                        frames[(symbol, timeframe)] = _append(
                            frames[(symbol, timeframe)], timeframe, cycle
                        )
                rows.append(
                    _process_event_cycle(
                        frames=frames,
                        state=state,
                        prediction_agent=agent,
                        strategy_registry=strategy_registry,
                        comparison_legacy_registry=comparison_legacy_registry,
                        comparison_eligibility_registry=comparison_eligibility_registry,
                        qwen_cache=qwen_cache,
                    )
                )
            scenarios[name] = rows

        summary = {}
        all_durations = []
        for name, rows in scenarios.items():
            durations = [float(row["total_seconds"]) for row in rows]
            all_durations.extend(durations)
            summary[name] = {
                "runs": rows,
                "p50_seconds": median(durations),
                "p95_seconds": _percentile(durations, 95),
                "max_seconds": max(durations),
            }
        metrics = agent.cache_metrics()
        report = {
            "symbols": len(symbols),
            "universe_source": universe_source,
            "position_management": _position_benchmark(),
            "no_change": no_change,
            "scenarios": summary,
            "overall_p50_seconds": median(all_durations),
            "overall_p95_seconds": _percentile(all_durations, 95),
            "overall_max_seconds": max(all_durations),
            "live_training_count": metrics.training_runs,
            "live_walk_forward_count": metrics.walk_forward_runs,
            "model_artifact_source": str(Path(directory)),
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "acceptance": {
                "p95_under_limit": _percentile(all_durations, 95) < args.max_seconds,
                "no_change_zero_strategy": no_change["strategy_evaluations"] == 0,
                "no_change_zero_ml": no_change["ml_inference_count"] == 0,
                "no_change_zero_qwen": no_change["qwen_calls"] == 0,
                "live_training_zero": metrics.training_runs == 0,
                "live_walk_forward_zero": metrics.walk_forward_runs == 0,
            },
        }
        rendered = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        if not all(report["acceptance"].values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
