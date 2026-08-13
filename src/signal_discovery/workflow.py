"""Groq-powered signal discovery optimization workflow.

The agent roles follow NVIDIA's quantitative-signal-discovery-agent:
Signal Agent proposes formulas, Code Agent compiles them, and Eval Agent scores
Rank-IC. The local Code Agent is deterministic and allowlisted instead of executing
LLM-produced Python.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm_factory import create_chat_model, model_for_tier, primary_and_fallback_models
from src.config import get_settings
from src.finops import LLMCallReason, get_cost_tracker
from src.finops.cost_tracker import record_llm_response

from .evaluator import RankICEvaluator
from .models import DiscoveredSignal, DiscoveryResult, EvaluationMetrics
from .operators import FormulaCompiler
from .store import DiscoveredSignalStore

logger = logging.getLogger(__name__)


def _balanced_json_objects(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    candidates = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL | re.I)
    candidates.append(text)
    for candidate in candidates:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        except json.JSONDecodeError:
            continue
    return []


class SignalDiscoveryWorkflow:
    def __init__(
        self,
        *,
        llm: Any | None = None,
        num_signals: int = 3,
        max_iterations: int = 3,
        forward_periods: int = 5,
        ic_threshold: float = 0.015,
        p_value_threshold: float = 0.05,
        min_periods: int = 30,
        min_cross_section: int = 8,
        output_directory: str = "data/nse/discovered_signals",
    ) -> None:
        settings = get_settings()
        primary_model, _fallback_model = primary_and_fallback_models()
        self.model_name = (
            model_for_tier("REVIEW")
            if getattr(settings, "llm_cost_optimization_enabled", False) is True
            else primary_model
        )
        max_tokens = (
            settings.qwen_max_output_tokens_discovery
            if getattr(settings, "llm_cost_optimization_enabled", False) is True
            else 1800
        )
        self.llm = llm or create_chat_model(self.model_name, temperature=0.5, max_tokens=max_tokens)
        self.num_signals = max(1, min(10, int(num_signals)))
        self.max_iterations = max(1, min(10, int(max_iterations)))
        self.ic_threshold = abs(float(ic_threshold))
        self.p_value_threshold = float(p_value_threshold)
        self.compiler = FormulaCompiler()
        self.evaluator = RankICEvaluator(forward_periods, min_cross_section, min_periods)
        self.store = DiscoveredSignalStore(
            output_directory,
            ic_threshold=self.ic_threshold,
            p_value_threshold=self.p_value_threshold,
        )

    @classmethod
    def from_settings(
        cls, *, llm: Any | None = None, timeframe: str | None = None
    ) -> SignalDiscoveryWorkflow:
        settings = get_settings()
        output_directory = Path(settings.signal_discovery_output_dir)
        if timeframe:
            output_directory /= timeframe
        return cls(
            llm=llm,
            num_signals=settings.signal_discovery_num_signals,
            max_iterations=settings.signal_discovery_max_iterations,
            forward_periods=settings.signal_discovery_forward_periods,
            ic_threshold=settings.signal_discovery_ic_threshold,
            p_value_threshold=settings.signal_discovery_p_value_threshold,
            min_periods=settings.signal_discovery_min_periods,
            min_cross_section=settings.signal_discovery_min_cross_section,
            output_directory=str(output_directory),
        )

    async def run(
        self,
        request: str,
        panel: Mapping[str, pd.DataFrame],
        *,
        seed_feedback: str = "",
    ) -> DiscoveryResult:
        feedback = seed_feedback
        best_signal: DiscoveredSignal | None = None
        best_metrics: EvaluationMetrics | None = None
        all_evaluations: list[dict[str, Any]] = []
        latest_signals: list[DiscoveredSignal] = []
        # Count of formulas that actually produced a p-value (compile failures don't
        # count — they were never a statistical test). Used for the Bonferroni
        # correction below: an agent that can keep proposing formulas until one
        # clears p_value_threshold by chance is exactly the multiple-comparisons
        # trap this search is prone to, since num_signals * max_iterations formulas
        # get tested against the same panel in pursuit of a single "hit".
        num_tests = 0

        for iteration in range(1, self.max_iterations + 1):
            raw = await self._invoke(
                "signal_discovery_generator", self._signal_prompt(request, feedback)
            )
            latest_signals = [
                DiscoveredSignal.from_dict(item) for item in _balanced_json_objects(raw)
            ]
            iteration_evaluations: list[dict[str, Any]] = []
            valid_signals: list[DiscoveredSignal] = []
            for signal in latest_signals:
                evaluation: dict[str, Any] = {"signal": signal.to_dict()}
                try:
                    compiled = self.compiler.compile(signal.formula)
                    normalized = DiscoveredSignal(
                        name=signal.name,
                        formula=signal.formula,
                        meaning=signal.meaning,
                        category=signal.category,
                        data_fields_used=compiled.fields_used,
                        operators_used=compiled.operators_used,
                    )
                    metrics = self.evaluator.evaluate(normalized, panel)
                    num_tests += 1
                    corrected_p_threshold = self._corrected_p_threshold(num_tests)
                    evaluation["metrics"] = metrics.to_dict()
                    evaluation["bonferroni_corrected_p_value_threshold"] = corrected_p_threshold
                    evaluation["accepted"] = self._accepted(metrics, corrected_p_threshold)
                    valid_signals.append(normalized)
                    if best_metrics is None or abs(metrics.mean_ic) > abs(best_metrics.mean_ic):
                        best_signal, best_metrics = normalized, metrics
                except (ValueError, TypeError, KeyError) as exc:
                    evaluation["error"] = str(exc)
                    evaluation["accepted"] = False
                iteration_evaluations.append(evaluation)
                all_evaluations.append(evaluation)

            accepted = [item for item in iteration_evaluations if item.get("accepted")]
            if accepted:
                winner = max(accepted, key=lambda item: abs(item["metrics"]["mean_ic"]))
                winner_signal = DiscoveredSignal.from_dict(winner["signal"])
                winner_metrics = EvaluationMetrics(**winner["metrics"])
                result = DiscoveryResult(
                    status="accepted",
                    request=request,
                    iteration=iteration,
                    total_iterations=iteration,
                    selected_signal=winner_signal,
                    metrics=winner_metrics,
                    signals=valid_signals,
                    evaluations=all_evaluations,
                    thresholds={
                        "ic_threshold": self.ic_threshold,
                        "p_value_threshold": self.p_value_threshold,
                        "bonferroni_corrected_p_value_threshold": winner[
                            "bonferroni_corrected_p_value_threshold"
                        ],
                    },
                    last_feedback=feedback,
                )
                path = self.store.save_accepted(result)
                result.saved_path = str(path)
                return result

            if iteration < self.max_iterations:
                feedback = await self._optimization_feedback(request, iteration_evaluations)

        return DiscoveryResult(
            status="best_effort" if best_signal else "failed",
            request=request,
            iteration=self.max_iterations,
            total_iterations=self.max_iterations,
            selected_signal=best_signal,
            metrics=best_metrics,
            signals=latest_signals,
            evaluations=all_evaluations,
            thresholds={
                "ic_threshold": self.ic_threshold,
                "p_value_threshold": self.p_value_threshold,
                "bonferroni_corrected_p_value_threshold": self._corrected_p_threshold(num_tests),
            },
            last_feedback=feedback,
        )

    def _corrected_p_threshold(self, num_tests: int) -> float:
        """Bonferroni-corrected significance level for the num_tests-th formula tested
        against this panel in the current run. Dividing by the running test count means
        the bar rises as more candidates get tried, instead of letting sheer search
        volume produce a false "significant" IC at the flat nominal threshold."""
        return self.p_value_threshold / max(1, num_tests)

    def _accepted(self, metrics: EvaluationMetrics, p_value_threshold: float) -> bool:
        return abs(metrics.mean_ic) >= self.ic_threshold and metrics.p_value <= p_value_threshold

    async def _invoke(self, agent: str, prompt: str) -> str:
        if get_cost_tracker().is_over_hard_budget():
            raise RuntimeError("FinOps hard budget reached; signal discovery stopped")
        response = await self.llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only the requested final answer. Do not reveal chain-of-thought. "
                        "Never invent operators or data fields."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        record_llm_response(
            agent,
            response,
            self.model_name,
            "SHARED",
            market="NSE",
            call_reason=LLMCallReason.BACKGROUND_ANALYSIS,
            component=f"signal_discovery_{agent}",
            purpose="offline_signal_discovery",
        )
        content = response.content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def _signal_prompt(self, request: str, feedback: str) -> str:
        feedback_block = f"\nPrevious evaluation feedback:\n{feedback}\n" if feedback else ""
        return f"""You are the Signal Agent for cross-sectional NSE alpha research.
Generate exactly {self.num_signals} distinct formulas for: {request}
Data matrices have rows=settled bars and columns=symbols. Available fields are
Open, High, Low, Close, Volume. Formulas may only use the functions below.

{self.compiler.operator_prompt()}
{feedback_block}
Return a JSON array only. Each object must contain:
name, formula, meaning, category, data_fields_used, operators_used.
Use only past/current data. Do not use future shifts, Python operators, methods, or attributes.
"""

    async def _optimization_feedback(self, request: str, evaluations: list[dict[str, Any]]) -> str:
        compact = json.dumps(evaluations, separators=(",", ":"))[:10000]
        prompt = f"""You are the Eval Agent optimizing cross-sectional signals for: {request}
Acceptance requires abs(mean_ic)>={self.ic_threshold} and p_value<={self.p_value_threshold}.
Here are the latest compiler errors and Rank-IC metrics:
{compact}
Give at most six concise changes for the next Signal Agent iteration. Mention operator and
lookback changes, avoid overfitting, and do not provide Python code.
"""
        return await self._invoke("signal_discovery_optimizer", prompt)
