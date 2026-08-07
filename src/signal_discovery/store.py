"""Durable accepted-signal storage with load-time quality gates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .models import DiscoveredSignal, DiscoveryResult, EvaluationMetrics
from .operators import FormulaCompiler


@dataclass(frozen=True)
class AcceptedSignal:
    signal: DiscoveredSignal
    metrics: EvaluationMetrics
    path: Path
    payload: dict[str, Any]


class DiscoveredSignalStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        ic_threshold: float,
        p_value_threshold: float,
    ) -> None:
        self.directory = Path(directory)
        self.ic_threshold = abs(float(ic_threshold))
        self.p_value_threshold = float(p_value_threshold)

    def save_accepted(self, result: DiscoveryResult) -> Path:
        if result.status != "accepted" or result.selected_signal is None or result.metrics is None:
            raise ValueError("only accepted results can enter the live signal store")
        self.directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", result.selected_signal.name.lower()).strip("_")
        digest = sha256(result.selected_signal.formula.encode("utf-8")).hexdigest()[:10]
        path = self.directory / f"{slug or 'signal'}_{digest}.json"
        temporary = path.with_suffix(".json.tmp")
        payload = result.to_dict()
        payload["saved_path"] = str(path)
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return path

    def load_accepted(self) -> list[AcceptedSignal]:
        if not self.directory.exists():
            return []
        accepted: list[AcceptedSignal] = []
        compiler = FormulaCompiler()
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "accepted":
                    continue
                signal_value = payload.get("selected_signal")
                metrics_value = payload.get("metrics")
                if not isinstance(signal_value, dict) or not isinstance(metrics_value, dict):
                    continue
                signal = DiscoveredSignal.from_dict(signal_value)
                compiler.compile(signal.formula)
                metrics = EvaluationMetrics(
                    mean_ic=float(metrics_value["mean_ic"]),
                    ic_std=float(metrics_value["ic_std"]),
                    ic_ir=float(metrics_value["ic_ir"]),
                    t_stat=float(metrics_value["t_stat"]),
                    p_value=float(metrics_value["p_value"]),
                    num_periods=int(metrics_value["num_periods"]),
                    positive_ic_ratio=float(metrics_value["positive_ic_ratio"]),
                    average_cross_section_size=float(
                        metrics_value.get("average_cross_section_size", 0.0)
                    ),
                )
                if abs(metrics.mean_ic) < self.ic_threshold:
                    continue
                if metrics.p_value > self.p_value_threshold:
                    continue
                accepted.append(AcceptedSignal(signal, metrics, path, payload))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return accepted
