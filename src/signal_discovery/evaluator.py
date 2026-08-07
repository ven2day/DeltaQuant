"""Cross-sectional Rank-IC evaluation for discovered signals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from .models import DiscoveredSignal, EvaluationMetrics
from .operators import FormulaCompiler


@dataclass
class RankICEvaluator:
    forward_periods: int = 5
    min_cross_section: int = 8
    min_periods: int = 30

    def evaluate(
        self,
        signal: DiscoveredSignal,
        panel: Mapping[str, pd.DataFrame],
    ) -> EvaluationMetrics:
        if self.forward_periods < 1:
            raise ValueError("forward_periods must be positive")
        close = panel.get("Close")
        if close is None or close.empty:
            raise ValueError("Close panel is empty")

        factor = FormulaCompiler().compile(signal.formula).evaluate(panel)
        factor, close = factor.align(close, join="inner", axis=0)
        factor, close = factor.align(close, join="inner", axis=1)
        forward_returns = close.shift(-self.forward_periods) / close - 1.0

        correlations: list[float] = []
        cross_section_sizes: list[int] = []
        for timestamp in factor.index:
            values = pd.concat(
                [factor.loc[timestamp].rename("factor"), forward_returns.loc[timestamp].rename("return")],
                axis=1,
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < self.min_cross_section:
                continue
            if values["factor"].nunique() < 2 or values["return"].nunique() < 2:
                continue
            correlation = spearmanr(values["factor"], values["return"]).statistic
            if correlation is not None and math.isfinite(float(correlation)):
                correlations.append(float(correlation))
                cross_section_sizes.append(len(values))

        if len(correlations) < self.min_periods:
            raise ValueError(
                f"only {len(correlations)} valid IC periods; at least {self.min_periods} required"
            )

        ic = np.asarray(correlations, dtype=float)
        mean_ic = float(ic.mean())
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        standard_error = ic_std / math.sqrt(len(ic)) if ic_std else 0.0
        t_stat = mean_ic / standard_error if standard_error else 0.0
        p_value = (
            float(2 * student_t.sf(abs(t_stat), df=len(ic) - 1)) if len(ic) > 1 else 1.0
        )
        return EvaluationMetrics(
            mean_ic=mean_ic,
            ic_std=ic_std,
            ic_ir=mean_ic / ic_std if ic_std else 0.0,
            t_stat=t_stat,
            p_value=p_value,
            num_periods=len(ic),
            positive_ic_ratio=float((ic > 0).mean()),
            average_cross_section_size=float(np.mean(cross_section_sizes)),
        )
