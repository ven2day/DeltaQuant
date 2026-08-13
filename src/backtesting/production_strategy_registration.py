"""Safe initial registration for newly implemented production strategy grains."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.strategies import ProductionStrategyConfig

from .strategy_eligibility import (
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)


def ensure_production_strategy_placeholders(
    registry: StrategyEligibilityRegistry,
    config: ProductionStrategyConfig,
) -> list[str]:
    """Register missing configured grains as SHADOW, never as executable.

    The placeholder records implementation/config identity only. They contain no OOS
    statistics and are replaced by a separately versioned validation artifact only when
    the offline walk-forward pipeline is run.
    """

    created: list[str] = []
    now = datetime.now(UTC).isoformat()
    for strategy_name, timeframes in config.timeframe_map.items():
        for timeframe in timeframes:
            if registry.latest(strategy_name, timeframe, market=config.market) is not None:
                continue
            registry.register(
                StrategyEligibility(
                    strategy_name=strategy_name,
                    timeframe=timeframe,
                    model_version=config.version,
                    validation_status=EligibilityStatus.SHADOW,
                    validated_at=now,
                    validation_window={
                        "kind": "implementation_placeholder",
                        "validated": False,
                    },
                    oos_trade_count=0,
                    oos_profit_factor=None,
                    oos_max_drawdown=None,
                    oos_win_rate=None,
                    minimum_model_confidence=0.0,
                    minimum_strategy_confidence=0.0,
                    status_reason=(
                        "Implemented with leakage-safe shared features; offline OOS "
                        "validation has not yet granted paper eligibility"
                    ),
                    artifact_reference=None,
                    requires_ml=False,
                    regime_performance={},
                    validation_metadata={
                        "production_strategy_version": config.version,
                        "parameters": dict(config.parameters[strategy_name]),
                        "score_weights": dict(config.score_weights[strategy_name]),
                        "executable": False,
                    },
                    market=config.market,
                )
            )
            created.append(f"{strategy_name}:{timeframe}:{config.version}")
    return created
