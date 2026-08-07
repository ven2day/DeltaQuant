"""Compute bounded live confidence tilts from accepted discovered signals."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .operators import FormulaCompiler
from .store import DiscoveredSignalStore


class DiscoveredSignalScorer:
    def __init__(
        self,
        store: DiscoveredSignalStore,
        *,
        max_probability_tilt: float = 0.08,
        min_cross_section: int = 8,
    ) -> None:
        self.store = store
        self.max_probability_tilt = max(0.0, min(0.20, float(max_probability_tilt)))
        self.min_cross_section = max(3, int(min_cross_section))

    def score_panel(self, panel: Mapping[str, pd.DataFrame]) -> dict[str, float]:
        contributions: dict[str, list[float]] = {}
        compiler = FormulaCompiler()
        for artifact in self.store.load_accepted():
            try:
                factor = compiler.compile(artifact.signal.formula).evaluate(panel)
                latest = self._latest_cross_section(factor)
                if latest is None:
                    continue
                standard_deviation = float(latest.std(ddof=0))
                if not math.isfinite(standard_deviation) or standard_deviation <= 0:
                    continue
                z_scores = (latest - float(latest.mean())) / standard_deviation
                direction = 1.0 if artifact.metrics.mean_ic >= 0 else -1.0
                for symbol, z_score in z_scores.items():
                    if math.isfinite(float(z_score)):
                        # tanh prevents one extreme cross-sectional outlier from dominating.
                        contribution = direction * math.tanh(float(z_score) / 2.0)
                        contributions.setdefault(str(symbol), []).append(contribution)
            except (ValueError, TypeError, KeyError):
                continue

        return {
            symbol: self.max_probability_tilt * float(np.mean(values))
            for symbol, values in contributions.items()
            if values
        }

    def _latest_cross_section(self, factor: pd.DataFrame) -> pd.Series | None:
        for timestamp in reversed(factor.index):
            values = factor.loc[timestamp].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) >= self.min_cross_section and values.nunique() >= 2:
                return values
        return None
