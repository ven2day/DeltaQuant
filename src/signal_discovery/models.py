"""Portable data contracts for signal discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class DiscoveredSignal:
    name: str
    formula: str
    meaning: str
    category: str = "other"
    data_fields_used: tuple[str, ...] = ()
    operators_used: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiscoveredSignal:
        return cls(
            name=str(value.get("name", "Unnamed Signal")).strip(),
            formula=str(value.get("formula", "")).strip(),
            meaning=str(value.get("meaning", value.get("description", ""))).strip(),
            category=str(value.get("category", "other")).strip().lower(),
            data_fields_used=tuple(str(item) for item in value.get("data_fields_used", [])),
            operators_used=tuple(str(item) for item in value.get("operators_used", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["data_fields_used"] = list(self.data_fields_used)
        value["operators_used"] = list(self.operators_used)
        return value


@dataclass(frozen=True)
class EvaluationMetrics:
    mean_ic: float
    ic_std: float
    ic_ir: float
    t_stat: float
    p_value: float
    num_periods: int
    positive_ic_ratio: float
    average_cross_section_size: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryResult:
    status: str
    request: str
    iteration: int
    total_iterations: int
    selected_signal: DiscoveredSignal | None
    metrics: EvaluationMetrics | None
    signals: list[DiscoveredSignal] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    last_feedback: str = ""
    saved_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "request": self.request,
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "selected_signal": self.selected_signal.to_dict() if self.selected_signal else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "signals": [signal.to_dict() for signal in self.signals],
            "evaluations": self.evaluations,
            "thresholds": self.thresholds,
            "last_feedback": self.last_feedback,
            "saved_path": self.saved_path,
            "created_at": self.created_at,
            "source": {
                "project": "NVIDIA quantitative-signal-discovery-agent adaptation",
                "url": "https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent",
                "license": "Apache-2.0",
            },
        }
