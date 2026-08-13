"""Deterministic candidate rejection before any candidate-specific Qwen call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.risk_compliance import (
    RiskCheckResult,
    RiskLimits,
    _run_risk_checks,
    check_kill_switch,
)
from src.backtesting.strategy_eligibility import (
    EligibilityDecision,
    StrategyEligibilityRegistry,
    eligibility_registry_from_settings,
    resolve_eligibility_environment,
)
from src.config import get_settings
from src.finops import get_cost_tracker

from .state import TradingState

_ACTIONABLE_ENTRY_STATES = {"", "BUY", "ENTER_NOW", "APPROVE", "APPROVED"}


@dataclass
class PreLLMGateResult:
    passed: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(
        default_factory=lambda: {
            "mtf_pass": 0,
            "entry_pass": 0,
            "regime_pass": 0,
            "registry_pass": 0,
            "regime_confidence_reduced": 0,
            "shadow_evaluated": 0,
            "pre_llm_risk_pass": 0,
        }
    )

    def reject(self, signal: dict[str, Any], check: RiskCheckResult) -> None:
        rejected = dict(signal)
        rejected["risk_result"] = {
            "approved": False,
            "failures": [check.to_dict()],
            "stage": "pre_qwen",
        }
        self.rejected.append(rejected)
        self.rejection_counts[check.rule] = self.rejection_counts.get(check.rule, 0) + 1


def _block(rule: str, message: str) -> RiskCheckResult:
    return RiskCheckResult(False, rule, message, "block")


def _entry_state(signal: dict[str, Any]) -> str:
    return str(
        signal.get(
            "entry_state",
            signal.get("final_decision", signal.get("candidate_action", "")),
        )
    ).upper()


def evaluate_signal_eligibility(
    signal: dict[str, Any],
    *,
    regime: str,
    runtime_mode: str,
    registry: StrategyEligibilityRegistry | None = None,
) -> EligibilityDecision:
    """Evaluate and attach the deterministic registry decision to one candidate."""

    settings = get_settings()
    selected_registry = registry or eligibility_registry_from_settings(settings)
    environment = resolve_eligibility_environment(
        runtime_mode,
        requested_execution_mode=str(getattr(settings, "execution_mode", "local_paper")),
        trading_mode=str(getattr(settings, "trading_mode", "paper")),
    )
    decision = selected_registry.evaluate(
        strategy_name=str(signal.get("strategy", "")),
        timeframe=str(signal.get("timeframe", "")),
        model_version=str(signal.get("model_version", signal.get("strategy_version", ""))),
        environment=environment,
        current_regime=regime,
        confidence=float(signal.get("original_confidence", signal.get("confidence", 0.0)) or 0.0),
        action=str(signal.get("signal_type", "BUY")),
        market=str(signal.get("market", getattr(settings, "market", "NSE"))),
    )
    signal.update(decision.to_dict())
    signal["strategy_version"] = decision.model_version
    signal["model_version"] = decision.model_version
    signal["eligibility_status"] = decision.validation_status.value
    signal["validation_status"] = decision.validation_status.value
    signal["confidence"] = decision.adjusted_confidence
    signal["registry_reason"] = decision.reason_code
    signal["registry_execution_allowed"] = decision.execution_allowed
    signal["registry_qwen_allowed"] = decision.qwen_allowed
    signal["shadow_only"] = decision.shadow_only
    if decision.eligibility is not None:
        signal["ml_required"] = decision.eligibility.requires_ml
    return decision


def filter_pre_llm_candidates(state: TradingState) -> PreLLMGateResult:
    """Filter candidates with existing deterministic authorities.

    The full ``_run_risk_checks`` function is reused so the early gate cannot drift
    from final risk semantics. Final risk still executes after a fresh/cached Qwen
    verdict; this function is only an early rejection optimization.
    """
    settings = get_settings()
    result = PreLLMGateResult()
    signals = list(state.get("signals", []))
    if not signals:
        return result

    horizon = str(state.get("trade_horizon", "SWING")).upper()
    if check_kill_switch(state):
        failure = _block("kill_switch", "Kill switch is active before Qwen review")
        for signal in signals:
            result.reject(signal, failure)
        return result

    if get_cost_tracker().is_over_hard_budget(horizon):
        failure = _block(
            "finops_hard_budget",
            f"Global or {horizon} Qwen hard budget is exhausted",
        )
        for signal in signals:
            result.reject(signal, failure)
        return result

    regime = str(state.get("regime", "unknown"))
    portfolio = state.get("portfolio", {})
    daily_stats = state.get("daily_stats", {})
    limits = RiskLimits.from_settings(trade_horizon=horizon)
    registry = eligibility_registry_from_settings(settings)

    for original in signals:
        signal = dict(original)
        signal.setdefault("trade_horizon", horizon)

        mtf_passed = signal.get("mtf_confirmation_passed")
        if mtf_passed is False:
            result.reject(signal, _block("mtf_confirmation", "MTF confirmation did not pass"))
            continue
        result.stage_counts["mtf_pass"] += 1

        entry_state = _entry_state(signal)
        if entry_state not in _ACTIONABLE_ENTRY_STATES:
            result.reject(
                signal,
                _block(
                    "entry_quality", f"Entry state {entry_state or 'UNKNOWN'} is not actionable"
                ),
            )
            continue
        result.stage_counts["entry_pass"] += 1

        decision = evaluate_signal_eligibility(
            signal,
            regime=regime,
            runtime_mode=str(state.get("execution_mode", "market_paper")),
            registry=registry,
        )
        if not decision.pipeline_allowed:
            result.reject(signal, _block("strategy_eligibility", decision.message))
            continue
        result.stage_counts["regime_pass"] += 1
        result.stage_counts["registry_pass"] += 1
        if decision.confidence_multiplier < 1.0:
            result.stage_counts["regime_confidence_reduced"] += 1

        # Shadow/research candidates are still evaluated against deterministic risk so
        # their would-have-been outcome is reproducible, but they never consume Qwen
        # capacity and never reach executable order creation.
        if not decision.qwen_allowed:
            shadow_checks = _run_risk_checks(
                signal,
                portfolio,
                daily_stats,
                limits,
                regime=regime,
                enforce_strategy_eligibility=False,
            )
            shadow_blocking = [
                check
                for check in shadow_checks
                if not check.passed and check.severity == "block"
            ]
            signal["shadow_evaluation"] = {
                "final": "WOULD_REJECT_BY_RISK" if shadow_blocking else "WOULD_PASS_RISK",
                "risk_failures": [check.to_dict() for check in shadow_blocking],
            }
            result.stage_counts["shadow_evaluated"] += 1
            result.reject(
                signal,
                _block(
                    "shadow_only",
                    "Candidate evaluated in shadow; Qwen and executable order creation skipped",
                ),
            )
            continue

        # Reuse every existing hard portfolio/risk rule. The confidence rule is the
        # only warning and therefore cannot reject here or at final risk.
        checks = _run_risk_checks(signal, portfolio, daily_stats, limits, regime=regime)
        blocking = [check for check in checks if not check.passed and check.severity == "block"]
        if blocking:
            rejected = dict(signal)
            rejected["risk_result"] = {
                "approved": False,
                "failures": [check.to_dict() for check in blocking],
                "stage": "pre_qwen",
            }
            result.rejected.append(rejected)
            for check in blocking:
                result.rejection_counts[check.rule] = result.rejection_counts.get(check.rule, 0) + 1
            continue

        result.passed.append(signal)
        result.stage_counts["pre_llm_risk_pass"] += 1

    return result
