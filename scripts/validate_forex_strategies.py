"""Walk-forward/OOS validation for shared strategies on Forex-only OANDA data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting.strategy_eligibility import (
    EligibilityStatus,
    RegimePolicy,
    StrategyEligibility,
)
from src.config import get_settings
from src.core.candidates import SignalEngine
from src.core.candidates.signals import StrategyType
from src.core.candles import CandleStore
from src.markets.forex.eligibility import (
    ForexRegimeClassifier,
    create_registry,
    demote_superseded_forex_approvals,
)
from src.markets.forex.market_data import create_forex_market_data_provider
from src.markets.forex.ml import (
    build_shared_indicator_history,
    evaluate_strategy_history,
    summarize_validation,
)
from src.markets.forex.persistence import bind_candle_repository, bind_trading_repository
from src.markets.forex.strategies import build_strategy_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--report", default="run/forex_walk_forward_report.json")
    return parser


def _load_policy() -> dict[str, Any]:
    path = ROOT / "config" / "forex" / "validation.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Forex validation configuration must be a mapping")
    return value


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    settings = get_settings()
    if settings.market != "FOREX" or settings.oanda_environment != "practice":
        raise RuntimeError("Forex validation accepts OANDA Practice configuration only")
    if settings.forex_execution_enabled or settings.allow_live_orders:
        raise RuntimeError("Forex validation requires broker execution disabled")
    policy = _load_policy()
    walk = dict(policy["walk_forward"])
    costs = dict(policy["execution_costs"])
    maximum_bars = int(args.max_bars or walk["max_bars_per_instrument_timeframe"])
    provider = create_forex_market_data_provider(settings)
    store = CandleStore(
        settings.market_history_database_url or settings.database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
        schema=str(settings.forex_db_schema),
    )
    candle_repository = bind_candle_repository(store)
    trading_repository = bind_trading_repository(store.engine)
    registry = create_registry(settings)
    production_config = build_strategy_config(settings)
    signal_engine = SignalEngine(production_config=production_config)
    classifier = ForexRegimeClassifier.from_settings(settings)
    trades_by_grain: dict[tuple[str, str], list[Any]] = defaultdict(list)
    dataset_windows: dict[tuple[str, str], list[str]] = defaultdict(list)
    try:
        instruments = await provider.list_instruments()
        quotes = await provider.get_prices([item.symbol for item in instruments])
        for instrument in instruments:
            quote = quotes.get(instrument.symbol)
            observed_spread = quote.spread / instrument.pip_size if quote is not None else 0.0
            spread_pips = max(float(costs["minimum_spread_pips"]), observed_spread)
            for timeframe in ("5m", "15m", "30m", "1h", "4h"):
                strategy_names = [
                    name
                    for name in production_config.timeframe_map
                    if production_config.eligible(name, timeframe)
                ]
                if not strategy_names:
                    continue
                for strategy_name in strategy_names:
                    trades_by_grain.setdefault((strategy_name, timeframe), [])
                    dataset_windows.setdefault((strategy_name, timeframe), [])
                frame = candle_repository.load_frame(
                    instrument.symbol,
                    timeframe,
                    bars=maximum_bars,
                    complete_only=True,
                )
                if frame.empty or len(frame) < 300:
                    continue
                snapshots = build_shared_indicator_history(
                    frame,
                    instrument.symbol,
                    timeframe,
                    max_lookback_bars=int(walk["feature_lookback_bars"]),
                )
                for strategy_name in strategy_names:
                    trades = evaluate_strategy_history(
                        frame,
                        snapshots,
                        instrument=instrument.symbol,
                        timeframe=timeframe,
                        strategy=StrategyType(strategy_name),
                        signal_engine=signal_engine,
                        regime_classifier=classifier,
                        pip_size=instrument.pip_size,
                        spread_pips=spread_pips,
                        slippage_pips_per_side=float(costs["slippage_pips_per_side"]),
                        max_holding_bars=int(walk["max_holding_bars"]),
                        development_fraction=float(walk["development_fraction"]),
                    )
                    trades_by_grain[(strategy_name, timeframe)].extend(trades)
                    dataset_windows[(strategy_name, timeframe)].extend(
                        [str(frame.index[0]), str(frame.index[-1])]
                    )

        reports: list[dict[str, Any]] = []
        approved_identities: set[tuple[str, str, str, str]] = set()
        timestamp = datetime.now(UTC).isoformat()
        for strategy_name, timeframe in sorted(trades_by_grain):
            metrics = summarize_validation(
                strategy_name,
                timeframe,
                trades_by_grain[(strategy_name, timeframe)],
                folds=int(walk["folds"]),
                risk_fraction=float(settings.forex_risk_per_trade),
                eligibility_config=dict(policy["eligibility"]),
                regime_policy_config=dict(policy["regime_policy"]),
            )
            windows = dataset_windows[(strategy_name, timeframe)]
            status = (
                EligibilityStatus.PAPER_APPROVED if metrics.approved else EligibilityStatus.SHADOW
            )
            regime_performance = metrics.regime_performance
            record = StrategyEligibility(
                market="FOREX",
                strategy_name=strategy_name,
                timeframe=timeframe,
                model_version=production_config.version,
                validation_status=status,
                validated_at=timestamp,
                validation_window={
                    "first_candle": min(windows) if windows else "",
                    "last_candle": max(windows) if windows else "",
                    "development_fraction": float(walk["development_fraction"]),
                    "final_oos_only": True,
                    "instruments": len(instruments),
                    "maximum_bars_per_instrument": maximum_bars,
                },
                oos_trade_count=metrics.trade_count,
                oos_profit_factor=metrics.profit_factor,
                oos_max_drawdown=metrics.max_drawdown,
                oos_win_rate=metrics.win_rate,
                minimum_model_confidence=0.0,
                minimum_strategy_confidence=0.0,
                allowed_regimes=tuple(
                    key
                    for key, value in regime_performance.items()
                    if value.policy is not RegimePolicy.BLOCK
                ),
                disabled_regimes=tuple(
                    key
                    for key, value in regime_performance.items()
                    if value.policy is RegimePolicy.BLOCK
                ),
                status_reason=(
                    "OOS_VALIDATION_PASSED"
                    if metrics.approved
                    else ",".join(metrics.rejection_reasons)
                ),
                artifact_reference=None,
                requires_ml=False,
                regime_performance=regime_performance,
                validation_metadata={
                    "parameter_version": production_config.version,
                    "provider": "OANDA",
                    "cost_model": costs,
                    "walk_forward": walk,
                    "random_shuffle": False,
                    "parameter_optimization": "NONE_FIXED_CONFIGURATION",
                    "metrics": metrics.to_dict(),
                },
            )
            registry.register(record)
            trading_repository.persist_strategy_eligibility(
                identity="|".join(record.identity), payload=record.to_dict()
            )
            reports.append(record.to_dict())
            if status is EligibilityStatus.PAPER_APPROVED:
                approved_identities.add(record.identity)
        demoted = demote_superseded_forex_approvals(
            registry,
            approved_identities=approved_identities,
            current_model_version=production_config.version,
            validated_at=timestamp,
            repository=trading_repository,
        )
        result = {
            "market": "FOREX",
            "provider": "OANDA",
            "environment": "PRACTICE",
            "execution": "PAPER_VALIDATION_ONLY",
            "broker_orders": "OFF",
            "reports": reports,
            "paper_approved": sum(
                item["validation_status"] == "PAPER_APPROVED" for item in reports
            ),
            "shadow": sum(item["validation_status"] == "SHADOW" for item in reports),
            "superseded_approvals_demoted": len(demoted),
            "duration_seconds": perf_counter() - started,
        }
        report_path = ROOT / str(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    finally:
        await provider.close()


if __name__ == "__main__":
    import asyncio

    print(json.dumps(asyncio.run(_run(_parser().parse_args())), indent=2, sort_keys=True))
