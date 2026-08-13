"""Forex-only OOS validation using shared indicators and strategy implementations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.strategy_eligibility import RegimePerformance, RegimePolicy
from src.core.candidates import SignalEngine, SignalType
from src.core.candidates.signals import StrategyType
from src.core.indicators import IndicatorResult, Timeframe, calculate_indicators
from src.markets.forex.eligibility import ForexRegimeClassifier


@dataclass(frozen=True)
class ForexValidationTrade:
    instrument: str
    timeframe: str
    strategy: str
    direction: str
    signal_time: str
    entry_time: str
    exit_time: str
    regime: str
    r_multiple: float
    gross_price_move: float
    spread_cost: float
    slippage_cost: float
    exit_reason: str
    oos: bool


@dataclass(frozen=True)
class ForexValidationMetrics:
    strategy: str
    timeframe: str
    trade_count: int
    buy_count: int
    sell_count: int
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    sharpe: float | None
    average_trade: float
    turnover: int
    spread_cost: float
    slippage_cost: float
    fold_consistency: float
    fold_results: tuple[dict[str, Any], ...]
    regime_performance: dict[str, RegimePerformance]
    development_trade_count: int
    approved: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime_performance"] = {
            key: value.to_dict() for key, value in self.regime_performance.items()
        }
        return payload


def build_shared_indicator_history(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    min_bars: int = 200,
    max_lookback_bars: int = 260,
) -> dict[int, IndicatorResult]:
    """Calculate one IndicatorResult per settled bar for reuse by all strategies."""

    normalized = frame.rename(columns=str.lower).sort_index()
    snapshots: dict[int, IndicatorResult] = {}
    for position in range(min_bars - 1, len(normalized) - 1):
        history = normalized.iloc[max(0, position + 1 - max_lookback_bars) : position + 1]
        snapshots[position] = calculate_indicators(history, symbol, timeframe=Timeframe(timeframe))
    return snapshots


def evaluate_strategy_history(
    frame: pd.DataFrame,
    snapshots: dict[int, IndicatorResult],
    *,
    instrument: str,
    timeframe: str,
    strategy: StrategyType,
    signal_engine: SignalEngine,
    regime_classifier: ForexRegimeClassifier,
    pip_size: float,
    spread_pips: float,
    slippage_pips_per_side: float,
    max_holding_bars: int,
    development_fraction: float,
) -> list[ForexValidationTrade]:
    """Replay one shared strategy with next-bar entry and conservative intrabar exits."""

    normalized = frame.rename(columns=str.lower).sort_index()
    split_position = int(len(normalized) * development_fraction)
    trades: list[ForexValidationTrade] = []
    active_until = -1
    for position, indicator in snapshots.items():
        if position <= active_until:
            continue
        generated = signal_engine.generate_signals(indicator, [strategy])
        if not generated:
            continue
        signal = generated[0]
        if signal.signal_type not in {SignalType.BUY, SignalType.SELL}:
            continue
        entry_position = position + 1
        if entry_position >= len(normalized):
            continue
        entry_mid = float(normalized.iloc[entry_position]["open"])
        stop = float(signal.stop_loss)
        target = float(signal.target_price)
        risk_distance = abs(float(signal.entry_price) - stop)
        if risk_distance <= 0:
            continue
        is_buy = signal.signal_type is SignalType.BUY
        if (is_buy and not (stop < entry_mid < target)) or (
            not is_buy and not (target < entry_mid < stop)
        ):
            continue
        end_position = min(len(normalized) - 1, entry_position + max_holding_bars - 1)
        exit_mid = float(normalized.iloc[end_position]["close"])
        exit_reason = "TIME_EXIT"
        actual_exit = end_position
        for future_position in range(entry_position, end_position + 1):
            bar = normalized.iloc[future_position]
            hit_stop = float(bar["low"]) <= stop if is_buy else float(bar["high"]) >= stop
            hit_target = float(bar["high"]) >= target if is_buy else float(bar["low"]) <= target
            # When both are touched in one OHLC candle, assume the adverse path first.
            if hit_stop:
                exit_mid = stop
                exit_reason = "STOP"
                actual_exit = future_position
                break
            if hit_target:
                exit_mid = target
                exit_reason = "TARGET"
                actual_exit = future_position
                break
        gross = (exit_mid - entry_mid) if is_buy else (entry_mid - exit_mid)
        spread_cost = max(0.0, spread_pips) * pip_size
        slippage_cost = max(0.0, slippage_pips_per_side) * pip_size * 2.0
        r_multiple = (gross - spread_cost - slippage_cost) / risk_distance
        trades.append(
            ForexValidationTrade(
                instrument=instrument,
                timeframe=timeframe,
                strategy=strategy.value,
                direction=signal.signal_type.value,
                signal_time=str(normalized.index[position]),
                entry_time=str(normalized.index[entry_position]),
                exit_time=str(normalized.index[actual_exit]),
                regime=regime_classifier.classify(indicator),
                r_multiple=float(r_multiple),
                gross_price_move=float(gross),
                spread_cost=float(spread_cost),
                slippage_cost=float(slippage_cost),
                exit_reason=exit_reason,
                oos=entry_position >= split_position,
            )
        )
        active_until = actual_exit
    return trades


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    if losses == 0:
        # Keep registry/report JSON standards-compliant.  A no-loss sample has an
        # undefined/infinite raw PF, so use one observation-sized pseudo-loss in
        # R-space instead of serialising Infinity or granting an unbounded score.
        return gains * max(1, int(values.size)) if gains > 0 else 0.0
    return gains / losses


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    return abs(float(np.min(equity - peaks)))


def _regime_stats(
    trades: Iterable[ForexValidationTrade], policy_config: dict[str, float]
) -> dict[str, RegimePerformance]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        groups[trade.regime].append(trade.r_multiple)
    output: dict[str, RegimePerformance] = {}
    minimum = int(policy_config["minimum_regime_trades"])
    allow_pf = float(policy_config["allow_profit_factor"])
    reduce_pf = float(policy_config["reduce_profit_factor"])
    multiplier = float(policy_config["reduce_confidence_multiplier"])
    for regime, raw in groups.items():
        values = np.asarray(raw, dtype=float)
        pf = _profit_factor(values)
        win_rate = float((values > 0).mean() * 100.0) if values.size else 0.0
        if len(values) >= minimum and pf >= allow_pf and float(values.mean()) > 0:
            policy, confidence_multiplier = RegimePolicy.ALLOW, 1.0
        elif len(values) >= minimum and pf >= reduce_pf and float(values.mean()) > 0:
            policy, confidence_multiplier = RegimePolicy.REDUCE_CONFIDENCE, multiplier
        else:
            policy, confidence_multiplier = RegimePolicy.BLOCK, 0.0
        output[regime] = RegimePerformance(
            policy=policy,
            profit_factor=pf,
            trade_count=len(values),
            confidence_multiplier=confidence_multiplier,
            win_rate=win_rate,
            max_drawdown=_max_drawdown(values),
        )
    return output


def summarize_validation(
    strategy: str,
    timeframe: str,
    trades: list[ForexValidationTrade],
    *,
    folds: int,
    risk_fraction: float,
    eligibility_config: dict[str, Any],
    regime_policy_config: dict[str, float],
) -> ForexValidationMetrics:
    oos = sorted((trade for trade in trades if trade.oos), key=lambda item: item.entry_time)
    values = np.asarray([trade.r_multiple for trade in oos], dtype=float)
    fold_results: list[dict[str, Any]] = []
    positive_folds = 0
    evaluated_folds = 0
    for fold_index, indices in enumerate(np.array_split(np.arange(len(oos)), max(1, folds))):
        fold_values = values[indices] if len(indices) else np.asarray([], dtype=float)
        if fold_values.size:
            evaluated_folds += 1
            positive_folds += int(float(fold_values.sum()) > 0)
        fold_results.append(
            {
                "fold": fold_index,
                "trades": int(fold_values.size),
                "expectancy_r": float(fold_values.mean()) if fold_values.size else 0.0,
                "return_pct": float(fold_values.sum() * risk_fraction * 100.0),
            }
        )
    consistency = positive_folds / evaluated_folds if evaluated_folds >= 2 else 0.0
    expectancy = float(values.mean()) if values.size else 0.0
    return_pct = float(values.sum() * risk_fraction * 100.0)
    reasons: list[str] = []
    if len(oos) < int(eligibility_config["minimum_oos_trades"]):
        reasons.append("INSUFFICIENT_OOS_TRADES")
    if bool(eligibility_config["require_positive_expectancy"]) and expectancy <= 0:
        reasons.append("NON_POSITIVE_OOS_EXPECTANCY")
    if bool(eligibility_config["require_positive_return"]) and return_pct <= 0:
        reasons.append("NON_POSITIVE_OOS_RETURN")
    if consistency <= float(eligibility_config["minimum_fold_consistency"]):
        reasons.append("INSUFFICIENT_FOLD_STABILITY")
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sharpe = (
        float(values.mean() / standard_deviation * math.sqrt(values.size))
        if standard_deviation > 0
        else None
    )
    return ForexValidationMetrics(
        strategy=strategy,
        timeframe=timeframe,
        trade_count=len(oos),
        buy_count=sum(trade.direction == "BUY" for trade in oos),
        sell_count=sum(trade.direction == "SELL" for trade in oos),
        win_rate=float((values > 0).mean() * 100.0) if values.size else 0.0,
        profit_factor=_profit_factor(values),
        expectancy=expectancy,
        max_drawdown=_max_drawdown(values),
        sharpe=sharpe,
        average_trade=expectancy,
        turnover=len(oos),
        spread_cost=sum(trade.spread_cost for trade in oos),
        slippage_cost=sum(trade.slippage_cost for trade in oos),
        fold_consistency=consistency,
        fold_results=tuple(fold_results),
        regime_performance=_regime_stats(oos, regime_policy_config),
        development_trade_count=sum(not trade.oos for trade in trades),
        approved=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "ForexValidationMetrics",
    "ForexValidationTrade",
    "build_shared_indicator_history",
    "evaluate_strategy_history",
    "summarize_validation",
]
