"""Authoritative model lifecycle registry and fail-safe live hot reload."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    and_,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine


class ModelStatus(str, Enum):
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    REJECTED = "REJECTED"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


class MLPolicy(str, Enum):
    DISABLED = "ML_DISABLED"
    OPTIONAL = "ML_OPTIONAL"
    REQUIRED = "ML_REQUIRED"


@dataclass(frozen=True)
class ModelGrain:
    market: str
    strategy_name: str
    timeframe: str
    strategy_version: str

    def normalized(self) -> ModelGrain:
        return ModelGrain(
            market=self.market.strip().upper(),
            strategy_name=self.strategy_name.strip().lower(),
            timeframe=self.timeframe.strip().lower(),
            strategy_version=self.strategy_version.strip(),
        )


@dataclass(frozen=True)
class ModelRecord:
    grain: ModelGrain
    model_version: str
    status: ModelStatus
    artifact_location: str
    artifact_checksum: str
    trained_at: datetime | None = None
    training_data_start: datetime | None = None
    training_data_end: datetime | None = None
    training_data_until: datetime | None = None
    validated_at: datetime | None = None
    oos_trade_count: int = 0
    oos_win_rate: float | None = None
    oos_profit_factor: float | None = None
    oos_expectancy: float | None = None
    oos_max_drawdown: float | None = None
    oos_sharpe: float | None = None
    calibration_metrics: dict[str, Any] | None = None
    is_active: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.grain.market,
            "strategy_name": self.grain.strategy_name,
            "timeframe": self.grain.timeframe,
            "strategy_version": self.grain.strategy_version,
            "model_version": self.model_version,
            "status": self.status.value,
            "artifact_location": self.artifact_location,
            "artifact_checksum": self.artifact_checksum,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_data_until": (
                self.training_data_until.isoformat() if self.training_data_until else None
            ),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "oos_trade_count": self.oos_trade_count,
            "oos_win_rate": self.oos_win_rate,
            "oos_profit_factor": self.oos_profit_factor,
            "oos_expectancy": self.oos_expectancy,
            "oos_max_drawdown": self.oos_max_drawdown,
            "oos_sharpe": self.oos_sharpe,
            "calibration_metrics": dict(self.calibration_metrics or {}),
            "is_active": self.is_active,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class MLPolicyDecision:
    policy: MLPolicy
    run_inference: bool
    reject_candidate: bool
    reason: str


def evaluate_ml_policy(policy: MLPolicy, *, active_model_available: bool) -> MLPolicyDecision:
    if policy is MLPolicy.DISABLED:
        return MLPolicyDecision(policy, False, False, "ML_DISABLED")
    if active_model_available:
        return MLPolicyDecision(policy, True, False, "ACTIVE_MODEL_AVAILABLE")
    if policy is MLPolicy.REQUIRED:
        return MLPolicyDecision(policy, False, True, "ACTIVE_APPROVED_MODEL_MISSING")
    return MLPolicyDecision(policy, False, False, "ML_OPTIONAL_ABSTAIN")


class ModelRegistry:
    """Database-backed registry with one ACTIVE champion per exact market grain."""

    def __init__(self, engine: Engine, *, schema: str = "common", create: bool = True) -> None:
        self.engine = engine
        self.schema = schema if engine.dialect.name != "sqlite" else None
        if create and self.schema:
            with engine.begin() as connection:
                connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
        metadata = MetaData(schema=self.schema)
        self.table = Table(
            "model_registry",
            metadata,
            Column("market", String(16), nullable=False),
            Column("strategy_name", String(100), nullable=False),
            Column("timeframe", String(16), nullable=False),
            Column("strategy_version", String(100), nullable=False),
            Column("model_version", String(100), nullable=False),
            Column("status", String(20), nullable=False),
            Column("artifact_location", String(1000), nullable=False),
            Column("artifact_checksum", String(128), nullable=False),
            Column("trained_at", DateTime(timezone=True)),
            Column("training_data_start", DateTime(timezone=True)),
            Column("training_data_end", DateTime(timezone=True)),
            Column("training_data_until", DateTime(timezone=True)),
            Column("validated_at", DateTime(timezone=True)),
            Column("oos_trade_count", Integer, nullable=False, default=0),
            Column("oos_win_rate", Float),
            Column("oos_profit_factor", Float),
            Column("oos_expectancy", Float),
            Column("oos_max_drawdown", Float),
            Column("oos_sharpe", Float),
            Column("calibration_metrics", JSON),
            Column("is_active", Boolean, nullable=False, default=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint(
                "market",
                "strategy_name",
                "timeframe",
                "strategy_version",
                "model_version",
                name="uq_model_registry_identity",
            ),
        )
        if create:
            metadata.create_all(engine)

    def _grain_clause(self, grain: ModelGrain) -> Any:
        value = grain.normalized()
        return and_(
            self.table.c.market == value.market,
            self.table.c.strategy_name == value.strategy_name,
            self.table.c.timeframe == value.timeframe,
            self.table.c.strategy_version == value.strategy_version,
        )

    @staticmethod
    def _assert_artifact_market(record: ModelRecord) -> None:
        path = PurePosixPath(record.artifact_location.replace("\\", "/"))
        segments = {segment.upper() for segment in path.parts}
        if record.grain.market.strip().upper() not in segments:
            raise ValueError("Artifact location is not contained in the requested market namespace")

    def register(self, record: ModelRecord) -> None:
        grain = record.grain.normalized()
        self._assert_artifact_market(ModelRecord(**{**record.__dict__, "grain": grain}))
        if record.status is ModelStatus.ACTIVE or record.is_active:
            raise ValueError("Register as APPROVED, then promote atomically")
        now = datetime.now(UTC)
        values = {
            "market": grain.market,
            "strategy_name": grain.strategy_name,
            "timeframe": grain.timeframe,
            "strategy_version": grain.strategy_version,
            "model_version": record.model_version,
            "status": record.status.value,
            "artifact_location": record.artifact_location,
            "artifact_checksum": record.artifact_checksum,
            "trained_at": record.trained_at,
            "training_data_start": record.training_data_start,
            "training_data_end": record.training_data_end,
            "training_data_until": record.training_data_until,
            "validated_at": record.validated_at,
            "oos_trade_count": record.oos_trade_count,
            "oos_win_rate": record.oos_win_rate,
            "oos_profit_factor": record.oos_profit_factor,
            "oos_expectancy": record.oos_expectancy,
            "oos_max_drawdown": record.oos_max_drawdown,
            "oos_sharpe": record.oos_sharpe,
            "calibration_metrics": record.calibration_metrics or {},
            "is_active": False,
            "created_at": record.created_at or now,
            "updated_at": now,
        }
        with self.engine.begin() as connection:
            connection.execute(insert(self.table).values(**values))

    def set_status(self, grain: ModelGrain, model_version: str, status: ModelStatus) -> None:
        if status is ModelStatus.ACTIVE:
            raise ValueError("Use promote() for ACTIVE status")
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.table)
                .where(self._grain_clause(grain), self.table.c.model_version == model_version)
                .values(status=status.value, is_active=False, updated_at=datetime.now(UTC))
            )
            if result.rowcount != 1:
                raise KeyError("Unknown model registry identity")

    def promote(self, grain: ModelGrain, model_version: str) -> ModelRecord:
        with self.engine.begin() as connection:
            candidate = connection.execute(
                select(self.table).where(
                    self._grain_clause(grain), self.table.c.model_version == model_version
                )
            ).mappings().one_or_none()
            if candidate is None:
                raise KeyError("Unknown model registry identity")
            if candidate["status"] != ModelStatus.APPROVED.value:
                raise ValueError("Only an APPROVED challenger can become ACTIVE")
            now = datetime.now(UTC)
            connection.execute(
                update(self.table)
                .where(self._grain_clause(grain), self.table.c.is_active.is_(True))
                .values(status=ModelStatus.RETIRED.value, is_active=False, updated_at=now)
            )
            connection.execute(
                update(self.table)
                .where(self._grain_clause(grain), self.table.c.model_version == model_version)
                .values(status=ModelStatus.ACTIVE.value, is_active=True, updated_at=now)
            )
        active = self.latest_active(grain)
        if active is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("Atomic model promotion failed")
        return active

    def latest_active(self, grain: ModelGrain) -> ModelRecord | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.table).where(
                    self._grain_clause(grain),
                    self.table.c.status == ModelStatus.ACTIVE.value,
                    self.table.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        return self._record(row) if row is not None else None

    def list_records(self, *, market: str | None = None) -> list[ModelRecord]:
        statement = select(self.table)
        if market is not None:
            statement = statement.where(self.table.c.market == market.strip().upper())
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: Any) -> ModelRecord:
        return ModelRecord(
            grain=ModelGrain(
                market=row["market"],
                strategy_name=row["strategy_name"],
                timeframe=row["timeframe"],
                strategy_version=row["strategy_version"],
            ),
            model_version=row["model_version"],
            status=ModelStatus(row["status"]),
            artifact_location=row["artifact_location"],
            artifact_checksum=row["artifact_checksum"],
            trained_at=row["trained_at"],
            training_data_start=row["training_data_start"],
            training_data_end=row["training_data_end"],
            training_data_until=row["training_data_until"],
            validated_at=row["validated_at"],
            oos_trade_count=row["oos_trade_count"],
            oos_win_rate=row["oos_win_rate"],
            oos_profit_factor=row["oos_profit_factor"],
            oos_expectancy=row["oos_expectancy"],
            oos_max_drawdown=row["oos_max_drawdown"],
            oos_sharpe=row["oos_sharpe"],
            calibration_metrics=dict(row["calibration_metrics"] or {}),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class ModelHotReloader:
    """Cache one champion; a broken replacement never evicts the working model."""

    def __init__(
        self,
        registry: ModelRegistry,
        grain: ModelGrain,
        loader: Callable[[ModelRecord], Any],
    ) -> None:
        self.registry = registry
        self.grain = grain.normalized()
        self.loader = loader
        self.active_record: ModelRecord | None = None
        self.model: Any | None = None
        self.last_error: str = ""

    def refresh(self) -> bool:
        candidate = self.registry.latest_active(self.grain)
        if candidate is None:
            return False
        if self.active_record and candidate.model_version == self.active_record.model_version:
            return False
        try:
            loaded = self.loader(candidate)
        except Exception as exc:  # hot reload must not interrupt live inference
            self.last_error = type(exc).__name__
            return False
        self.model = loaded
        self.active_record = candidate
        self.last_error = ""
        return True


__all__ = [
    "MLPolicy",
    "MLPolicyDecision",
    "ModelGrain",
    "ModelHotReloader",
    "ModelRecord",
    "ModelRegistry",
    "ModelStatus",
    "evaluate_ml_policy",
]
