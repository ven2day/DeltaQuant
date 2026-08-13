"""
Signal consolidation.

Multiple strategies can independently fire on the same symbol/timeframe/direction in
one scan cycle (e.g. momentum AND trend_following both raise a BUY on RELIANCE@15m).
Today, before this module, each such signal becomes an independent ranked candidate --
so strategy agreement doesn't strengthen a single opportunity, it just multiplies the
count of near-duplicate candidates reaching the LLM review stage.

``consolidate_signals`` groups by (symbol, timeframe, signal_type) and produces one
``ConsolidatedSignal`` per group, carrying every contributing strategy plus a
confidence blend that rewards agreement without fabricating conviction from nothing.
It never merges across a different symbol, timeframe, or direction -- a BUY and a SELL
on the same symbol/timeframe are never combined, regardless of confidence.

Wiring (see src/markets/nse/runtime/live.py) is gated behind
``settings.enable_signal_consolidation`` (default False) precisely so the existing
swing candidate mix is provably unchanged unless explicitly enabled.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.backtesting.strategy_eligibility import (
    EligibilityDecision,
    EligibilityEnvironment,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.core.candidates.signals import (
    SignalEngine,
    SignalType,
    StrategyType,
    TradeHorizon,
    TradingSignal,
)
from src.core.features.context import MarketRelativeFeatures
from src.core.indicators import FEATURE_SET_VERSION, IndicatorResult, Timeframe
from src.core.strategies.config import PRODUCTION_STRATEGY_NAMES

# How much each additional agreeing strategy adds to the representative signal's own
# confidence, capped so agreement alone can never manufacture near-certainty out of
# several weak signals. Tunable, not a magic number buried in the formula below.
AGREEMENT_CONFIDENCE_STEP = 0.05
AGREEMENT_CONFIDENCE_CAP = 0.95


@dataclass(frozen=True)
class ConsolidatedSignal:
    """One strengthened trading opportunity representing 1+ strategies agreeing on
    the same symbol, timeframe, and direction."""

    symbol: str
    timeframe: Timeframe
    signal_type: SignalType
    contributing_strategies: tuple[StrategyType, ...]
    agreement_count: int
    representative_signal: TradingSignal
    blended_confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = self.representative_signal.to_dict()
        payload["confidence"] = self.blended_confidence
        payload["contributing_strategies"] = [s.value for s in self.contributing_strategies]
        payload["agreement_count"] = self.agreement_count
        return payload


def _blend_confidence(base_confidence: float, agreement_count: int) -> float:
    """Boost the representative signal's own confidence by how many strategies
    agree, never treating agreement alone as sufficient to reach near-certainty."""
    boosted = base_confidence + AGREEMENT_CONFIDENCE_STEP * (agreement_count - 1)
    return min(AGREEMENT_CONFIDENCE_CAP, boosted)


def consolidate_signals(signals: list[TradingSignal]) -> list[ConsolidatedSignal]:
    """Group ``signals`` by (symbol, timeframe, signal_type) into one
    ``ConsolidatedSignal`` per group.

    The representative signal for each group is the highest-confidence contributor --
    its entry/stop/target/reasons are what downstream ranking and evaluation actually
    see, on the theory that the best-evidenced individual signal is a safer anchor for
    trade levels than an average across strategies with different sizing conventions.
    HOLD signals (already filtered out by SignalEngine before this point in practice,
    but defensively skipped here too) never consolidate into a tradeable opportunity.
    """
    groups: dict[tuple[str, Timeframe, SignalType], list[TradingSignal]] = defaultdict(list)
    for signal in signals:
        if signal.signal_type == SignalType.HOLD:
            continue
        groups[(signal.symbol, signal.timeframe, signal.signal_type)].append(signal)

    consolidated: list[ConsolidatedSignal] = []
    for (symbol, timeframe, signal_type), group in groups.items():
        representative = max(group, key=lambda s: s.confidence)
        contributing = tuple(dict.fromkeys(s.strategy for s in group))  # de-duplicated, ordered
        agreement_count = len(group)
        consolidated.append(
            ConsolidatedSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                contributing_strategies=contributing,
                agreement_count=agreement_count,
                representative_signal=representative,
                blended_confidence=_blend_confidence(representative.confidence, agreement_count),
            )
        )
    return consolidated


# ---------------------------------------------------------------------------
# Stateful multi-strategy runtime models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSnapshot:
    """Identity plus the one shared indicator result used by every strategy.

    ``IndicatorResult`` is deliberately retained by reference.  Constructing a snapshot
    never recalculates an indicator and strategies cannot mistake a copied dataframe for
    a different feature set.
    """

    snapshot_id: str
    symbol: str
    timeframe: Timeframe
    settled_candle_timestamp: str
    feature_version: str
    indicators: IndicatorResult = field(compare=False, repr=False)
    market_relative: MarketRelativeFeatures | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    market: str = "NSE"
    provider: str = "DHAN"
    volume_type: str = "EXCHANGE_UNITS"

    @classmethod
    def create(
        cls,
        indicators: IndicatorResult,
        *,
        settled_candle_timestamp: Any,
        feature_version: str = FEATURE_SET_VERSION,
        market_relative: MarketRelativeFeatures | None = None,
        market: str = "NSE",
        provider: str = "DHAN",
        volume_type: str = "EXCHANGE_UNITS",
    ) -> FeatureSnapshot:
        timestamp = str(settled_candle_timestamp)
        identity = {
            "symbol": indicators.symbol.upper(),
            "timeframe": indicators.timeframe.value,
            "settled_candle_timestamp": timestamp,
            "feature_version": feature_version,
            "market_relative": market_relative.to_dict() if market_relative is not None else None,
            "market": market.upper(),
            "provider": provider.upper(),
            "volume_type": volume_type.upper(),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            snapshot_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            symbol=indicators.symbol,
            timeframe=indicators.timeframe,
            settled_candle_timestamp=timestamp,
            feature_version=feature_version,
            indicators=indicators,
            market_relative=market_relative,
            market=market.upper(),
            provider=provider.upper(),
            volume_type=volume_type.upper(),
        )


def eligible_strategy_types(
    signal_engine: SignalEngine,
    timeframe: Timeframe,
) -> tuple[StrategyType, ...]:
    """Return the configured strategy matrix for one timeframe.

    Existing strategy semantics are preserved. Only the new production family is
    constrained by the configurable timeframe matrix.
    """

    production_config = getattr(signal_engine, "production_config", None)
    return tuple(
        strategy
        for strategy in StrategyType
        if strategy.value not in PRODUCTION_STRATEGY_NAMES
        or production_config is None
        or production_config.eligible(strategy.value, timeframe.value)
    )


def strategy_requires_ml(eligibility: StrategyEligibility | None) -> bool:
    """Return the versioned policy flag; absence never invents an ML dependency."""

    return bool(eligibility and eligibility.requires_ml)


@dataclass(frozen=True)
class RegisteredStrategySignal:
    """One normalized strategy output with explicit eligibility semantics."""

    signal: TradingSignal
    feature_snapshot_id: str
    settled_candle_timestamp: str
    eligibility: StrategyEligibility | None
    eligibility_decision: EligibilityDecision
    validation_status: str
    pipeline_eligible: bool
    executable: bool
    shadow_only: bool
    ml_required: bool
    admission_reason: str | None = None

    @property
    def strategy_name(self) -> str:
        return self.signal.strategy.value

    def to_dict(self) -> dict[str, Any]:
        payload = self.signal.to_dict()
        payload.update(
            {
                "feature_snapshot_id": self.feature_snapshot_id,
                "signal_candle_timestamp": self.settled_candle_timestamp,
                "strategy_version": self.eligibility.model_version
                if self.eligibility is not None
                else "",
                "model_version": self.eligibility_decision.model_version,
                "validation_status": self.validation_status,
                "execution_eligibility": "EXECUTABLE" if self.executable else "SHADOW_ONLY",
                "pipeline_eligible": self.pipeline_eligible,
                "shadow_only": self.shadow_only,
                "ml_required": self.ml_required,
                "admission_reason": self.admission_reason,
                **self.eligibility_decision.to_dict(),
            }
        )
        return payload


def evaluate_registered_strategies(
    signal_engine: SignalEngine,
    snapshot: FeatureSnapshot,
    registry: StrategyEligibilityRegistry,
    *,
    trade_horizon: TradeHorizon | str,
    regime: str | None,
    execution_mode: str,
    eligibility_environment: EligibilityEnvironment | str = EligibilityEnvironment.PAPER,
) -> list[RegisteredStrategySignal]:
    """Evaluate every strategy against the exact same cached feature snapshot.

    Validation controls only whether a generated signal is executable.  Unvalidated,
    expired, and missing versions remain observable as shadow research evidence.
    """

    horizon = (
        trade_horizon.value if isinstance(trade_horizon, TradeHorizon) else str(trade_horizon)
    ).upper()
    outputs: list[RegisteredStrategySignal] = []
    for strategy in eligible_strategy_types(signal_engine, snapshot.timeframe):
        if snapshot.market_relative is None:
            generated = signal_engine.generate_signals(snapshot.indicators, [strategy])
        else:
            generated = signal_engine.generate_signals(
                snapshot.indicators,
                [strategy],
                market_relative=snapshot.market_relative,
            )
        eligibility = registry.select_for_environment(
            strategy.value,
            snapshot.timeframe.value,
            eligibility_environment,
            market=snapshot.market,
        )
        for signal in generated:
            requested_model_version = (
                eligibility.model_version
                if eligibility is not None
                else getattr(getattr(signal_engine, "production_config", None), "version", "")
                if strategy.value in PRODUCTION_STRATEGY_NAMES
                else ""
            )
            decision = registry.evaluate(
                strategy_name=strategy.value,
                timeframe=snapshot.timeframe.value,
                model_version=requested_model_version,
                environment=eligibility_environment,
                current_regime=regime,
                confidence=signal.confidence,
                action=signal.signal_type.value,
                market=snapshot.market,
            )
            signal.market = snapshot.market
            signal.market_data_provider = snapshot.provider
            signal.trade_horizon = TradeHorizon(horizon)
            signal.execution_mode = execution_mode
            signal.strategy_version = decision.model_version
            signal.model_version = decision.model_version
            signal.validation_status = decision.validation_status.value
            signal.eligibility_status = decision.validation_status.value
            signal.eligibility_environment = decision.environment.value
            signal.current_regime = decision.current_regime
            signal.regime_policy = decision.regime_policy.value
            signal.original_confidence = decision.original_confidence
            signal.regime_adjusted_confidence = decision.adjusted_confidence
            signal.confidence = decision.adjusted_confidence
            signal.registry_decision = "ALLOW" if decision.pipeline_allowed else "BLOCK"
            signal.registry_reason = decision.reason_code
            signal.registry_execution_allowed = decision.execution_allowed
            signal.registry_qwen_allowed = decision.qwen_allowed
            signal.shadow_only = decision.shadow_only
            signal.feature_snapshot_id = snapshot.snapshot_id
            signal.feature_version = snapshot.feature_version
            signal.signal_candle_timestamp = snapshot.settled_candle_timestamp
            signal.ml_required = strategy_requires_ml(eligibility)
            outputs.append(
                RegisteredStrategySignal(
                    signal=signal,
                    feature_snapshot_id=snapshot.snapshot_id,
                    settled_candle_timestamp=snapshot.settled_candle_timestamp,
                    eligibility=eligibility,
                    eligibility_decision=decision,
                    validation_status=decision.validation_status.value,
                    pipeline_eligible=decision.pipeline_allowed,
                    executable=decision.execution_allowed,
                    shadow_only=decision.shadow_only,
                    ml_required=strategy_requires_ml(eligibility),
                    admission_reason=(None if decision.pipeline_allowed else decision.reason_code),
                )
            )
    return outputs


@dataclass(frozen=True)
class AggregatedCandidate:
    """One symbol/timeframe/candle candidate retaining support and opposition."""

    symbol: str
    timeframe: Timeframe
    trade_horizon: TradeHorizon
    settled_candle_timestamp: str
    feature_snapshot_id: str
    direction: SignalType
    supporting_signals: tuple[RegisteredStrategySignal, ...]
    opposing_signals: tuple[RegisteredStrategySignal, ...]
    strongest_validated_signal: RegisteredStrategySignal | None
    agreement_level: str
    conflict_level: str
    conflict_resolution: str
    technical_quality: float | None
    pipeline_eligible: bool
    execution_allowed: bool
    shadow_only: bool

    @property
    def representative_signal(self) -> TradingSignal:
        anchor = self.strongest_validated_signal or self.supporting_signals[0]
        return anchor.signal

    def to_dict(self) -> dict[str, Any]:
        anchor = self.strongest_validated_signal or self.supporting_signals[0]
        payload = anchor.to_dict()
        payload.update(
            {
                "signal_type": self.direction.value,
                "supporting_strategies": [item.strategy_name for item in self.supporting_signals],
                "opposing_strategies": [item.strategy_name for item in self.opposing_signals],
                "strategy_consensus_identity": "|".join(
                    sorted(
                        f"{item.strategy_name}:{item.signal.signal_type.value}:"
                        f"{item.eligibility.model_version if item.eligibility else 'unregistered'}"
                        for item in (*self.supporting_signals, *self.opposing_signals)
                    )
                ),
                "agreement_level": self.agreement_level,
                "conflict_level": self.conflict_level,
                "conflict_resolution": self.conflict_resolution,
                "technical_quality": self.technical_quality,
                "pipeline_eligible": self.pipeline_eligible,
                "registry_execution_allowed": self.execution_allowed,
                "shadow_only": self.shadow_only,
                "execution_eligibility": "SHADOW_ONLY" if self.shadow_only else "EXECUTABLE",
            }
        )
        return payload

    def apply_lineage(self) -> TradingSignal:
        """Attach aggregation metadata to the existing representative signal."""

        signal = self.representative_signal
        signal.supporting_strategies = [item.strategy_name for item in self.supporting_signals]
        signal.opposing_strategies = [item.strategy_name for item in self.opposing_signals]
        signal.agreement_level = self.agreement_level
        signal.conflict_level = self.conflict_level
        if self.conflict_resolution == "UNRESOLVED_BLOCK":
            signal.registry_reason = "UNRESOLVED_STRATEGY_CONFLICT"
            signal.registry_execution_allowed = False
            signal.registry_qwen_allowed = False
        signal.strategy_consensus_identity = "|".join(
            sorted(
                f"{item.strategy_name}:{item.signal.signal_type.value}:"
                f"{item.eligibility.model_version if item.eligibility else 'unregistered'}"
                for item in (*self.supporting_signals, *self.opposing_signals)
            )
        )
        signal.technical_quality = self.technical_quality
        signal.shadow_only = self.shadow_only
        return signal


def _evidence_key(item: RegisteredStrategySignal) -> tuple[int, float, float, float]:
    eligibility = item.eligibility
    return (
        1 if item.pipeline_eligible else 0,
        float(eligibility.oos_profit_factor)
        if eligibility is not None and eligibility.oos_profit_factor is not None
        else float("-inf"),
        float(eligibility.oos_trade_count) if eligibility is not None else float("-inf"),
        float(item.signal.confidence),
    )


def aggregate_strategy_signals(
    signals: list[RegisteredStrategySignal],
) -> list[AggregatedCandidate]:
    """Aggregate both directions without vote counting or arbitrary strategy weights.

    The direction anchor is the strongest exact-grain validated artifact (OOS
    expectancy, then fold consistency, then the strategy's own defined confidence).
    Shadow evidence can expose conflict but can never displace validated evidence or
    become executable.
    """

    groups: dict[tuple[str, Timeframe, TradeHorizon, str, str], list[RegisteredStrategySignal]]
    groups = defaultdict(list)
    for item in signals:
        groups[
            (
                item.signal.symbol,
                item.signal.timeframe,
                item.signal.trade_horizon,
                item.settled_candle_timestamp,
                item.feature_snapshot_id,
            )
        ].append(item)

    candidates: list[AggregatedCandidate] = []
    for (symbol, timeframe, horizon, timestamp, snapshot_id), group in groups.items():
        eligible = [item for item in group if item.pipeline_eligible]
        strongest = max(eligible, key=_evidence_key) if eligible else None
        conflict_resolution = "NO_CONFLICT"
        unresolved_conflict = False
        eligible_directions = {item.signal.signal_type for item in eligible}
        if len(eligible_directions) > 1:
            best_by_direction = {
                direction: max(
                    (item for item in eligible if item.signal.signal_type == direction),
                    key=_evidence_key,
                )
                for direction in eligible_directions
            }
            profit_factors = {
                direction: (
                    item.eligibility.oos_profit_factor if item.eligibility is not None else None
                )
                for direction, item in best_by_direction.items()
            }
            finite = {
                direction: float(value)
                for direction, value in profit_factors.items()
                if value is not None
            }
            if len(finite) == len(eligible_directions) and len(set(finite.values())) > 1:
                resolved_direction = max(finite, key=lambda candidate: finite[candidate])
                strongest = best_by_direction[resolved_direction]
                conflict_resolution = "OOS_PROFIT_FACTOR"
            else:
                unresolved_conflict = True
                conflict_resolution = "UNRESOLVED_BLOCK"
        anchor = strongest or max(group, key=_evidence_key)
        direction = anchor.signal.signal_type
        supporting = tuple(item for item in group if item.signal.signal_type == direction)
        opposing = tuple(item for item in group if item.signal.signal_type != direction)
        executable_opposition = any(item.pipeline_eligible for item in opposing)
        conflict = "HIGH" if executable_opposition else "LOW" if opposing else "NONE"
        support_count = sum(item.pipeline_eligible for item in supporting) or len(supporting)
        agreement = "HIGH" if support_count >= 3 else "MEDIUM" if support_count == 2 else "SINGLE"
        candidates.append(
            AggregatedCandidate(
                symbol=symbol,
                timeframe=timeframe,
                trade_horizon=horizon,
                settled_candle_timestamp=timestamp,
                feature_snapshot_id=snapshot_id,
                direction=direction,
                supporting_signals=supporting,
                opposing_signals=opposing,
                strongest_validated_signal=strongest,
                agreement_level=agreement,
                conflict_level=conflict,
                conflict_resolution=conflict_resolution,
                technical_quality=max(
                    (item.signal.confidence for item in supporting), default=None
                ),
                pipeline_eligible=strongest is not None and not unresolved_conflict,
                execution_allowed=bool(
                    strongest and strongest.executable and not unresolved_conflict
                ),
                shadow_only=not bool(
                    strongest and strongest.executable and not unresolved_conflict
                ),
            )
        )
    return candidates
