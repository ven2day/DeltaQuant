#!/usr/bin/env python3
"""Small REAL-Qwen contextual soak with no execution/broker-order capability."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.candidate_cache import (  # noqa: E402
    CandidateVerdict,
    CandidateVerdictCache,
    build_candidate_fingerprint,
)
from src.agents.context_review import (  # noqa: E402
    QwenReviewPolicy,
    parse_context_reviews,
    serialize_context_packets,
)
from src.agents.llm_factory import (  # noqa: E402
    current_provider,
    get_llm_circuit_breaker,
    get_llm_limiter,
    invoke_with_fallback,
    model_for_tier,
)
from src.agents.signal_validation import COMPACT_VALIDATION_SYSTEM_PROMPT  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.finops.cost_tracker import record_llm_response  # noqa: E402
from src.markets.nse.risk.pretrade import FinalPaperOrder, PaperRiskReservations  # noqa: E402


def _json_payload(raw: str) -> dict[str, object]:
    """Parse the legacy advisory-soak schema without permitting trade directions.

    The live/runtime review path uses ``ContextReview``.  This narrow helper remains
    for compatibility with the original soak contract and deliberately accepts only
    non-executable HOLD/REVIEW advice.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Qwen advisory response must be a JSON object")
    if payload.get("decision") not in {"HOLD", "REVIEW"}:
        raise ValueError("Qwen advisory response cannot contain a trade instruction")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("Qwen advisory confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("Qwen advisory confidence must be between 0 and 1")
    if not isinstance(payload.get("reason"), str):
        raise ValueError("Qwen advisory reason must be a string")
    risk_flags = payload.get("risk_flags")
    if not isinstance(risk_flags, list) or not all(
        isinstance(flag, str) for flag in risk_flags
    ):
        raise ValueError("Qwen advisory risk_flags must be a list of strings")
    return payload


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls",
        type=int,
        default=3,
        help="Number of ambiguous candidates that may make a real Qwen call; an equal "
        "number of clear candidates is evaluated and skipped.",
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/qwen_local_paper_soak.json"))
    return parser.parse_args()


def _candidate(index: int, *, conflicted: bool) -> dict[str, object]:
    return {
        "signal_id": f"SOAK-{index:03d}",
        "symbol": f"SOAK{index:03d}",
        "timeframe": "15m",
        "trade_horizon": "SCALP",
        "signal_type": "BUY",
        "strategy": "momentum",
        "strategy_version": "soak-validated-v1",
        "feature_snapshot_id": f"feature-{index}",
        "feature_version": "features_v3",
        "signal_candle_timestamp": f"2026-08-11T10:{15 + index:02d}:00+05:30",
        "supporting_strategies": ["momentum", "breakout"],
        "opposing_strategies": ["mean_reversion"] if conflicted else [],
        "agreement_level": "MEDIUM",
        "conflict_level": "HIGH" if conflicted else "NONE",
        "technical_quality": 0.79,
        "confidence": 0.79,
        "risk_reward_ratio": 2.0,
        "ml_status": "VALIDATED",
        "ml_probability": 0.64,
        "ml_model_version": "soak-model-v1",
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "target_price": 104.0,
    }


def _state() -> dict[str, object]:
    return {
        "trade_horizon": "SCALP",
        "regime": "trending_up",
        "regime_confidence": 0.88,
        "shared_market_context": {
            "context_version": "soak-market-v1",
            "major_market_event_risk": "NONE",
            "market_sentiment": 0.05,
            "major_news_summary": [],
        },
    }


def _risk_must_reject(index: int) -> list[str]:
    # Deliberately invalid long stop. Even SUPPORTIVE Qwen context cannot bypass
    # deterministic risk. ExecutionService is never constructed or called.
    decision = PaperRiskReservations().evaluate(
        FinalPaperOrder(
            symbol=f"SOAK{index:03d}",
            side="BUY",
            quantity=1,
            entry_price=100.0,
            stop_loss=101.0,
            target_price=102.0,
            strategy="momentum",
        ),
        positions=[],
        equity=1_000_000.0,
        entries_today=0,
        daily_entry_cap=100,
        max_positions=100,
        max_position_pct=1.0,
        max_total_exposure_pct=100.0,
        risk_per_trade=1.0,
        max_total_risk=1.0,
        realized_daily_pnl=0.0,
        daily_loss_limit=100_000.0,
    )
    if decision.approved:
        raise RuntimeError("deterministic risk unexpectedly approved invalid stop geometry")
    return decision.reasons


def main() -> int:
    args = _arguments()
    settings = get_settings()
    if current_provider() != "qwen":
        raise RuntimeError(f"Real-Qwen soak requires LLM_PROVIDER=qwen, got {current_provider()}")
    key = settings.dashscope_api_key.get_secret_value()
    if not key:
        raise RuntimeError("Real-Qwen soak requires DASHSCOPE_API_KEY")
    if args.calls < 1:
        raise ValueError("--calls must be positive")

    model = model_for_tier("REVIEW")
    breaker = get_llm_circuit_breaker()
    limiter = get_llm_limiter()
    policy = QwenReviewPolicy(
        regime_uncertain_below=settings.qwen_regime_uncertain_below,
        ml_conflict_below=settings.qwen_ml_conflict_below,
    )
    cache = CandidateVerdictCache()
    state = _state()
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    failures = 0
    schema_failures = 0
    risk_bypasses = 0
    qwen_calls = 0
    qwen_skipped_clear = 0
    qwen_cache_hits = 0
    total_input_tokens = 0
    total_output_tokens = 0

    candidates = [
        _candidate(index, conflicted=bool(index % 2)) for index in range(args.calls * 2)
    ]
    for index, candidate in enumerate(candidates):
        review_decision = policy.assess(
            candidate,
            regime_confidence=float(state["regime_confidence"]),
            shared_context=state["shared_market_context"],
        )
        if not review_decision.required:
            qwen_skipped_clear += 1
            rows.append(
                {
                    "candidate": candidate["signal_id"],
                    "qwen_required": False,
                    "handling": "CONTINUE",
                    "reason": "CLEAR_DETERMINISTIC_CONTEXT",
                }
            )
            continue

        fingerprint = build_candidate_fingerprint(
            candidate,
            regime=str(state["regime"]),
            market_context_version=str(state["shared_market_context"]["context_version"]),
        )
        cached = cache.get(fingerprint)
        if cached is not None:
            qwen_cache_hits += 1
            rows.append(
                {
                    "candidate": candidate["signal_id"],
                    "qwen_required": True,
                    "cache_hit": True,
                    "handling": cached.handling,
                }
            )
            continue

        started = perf_counter()
        try:
            if settings.enable_rate_limiting:
                limiter.acquire_sync()
            prompt = serialize_context_packets(
                [candidate],
                state,
                max_input_tokens=settings.qwen_max_input_tokens_per_review,
            )
            qwen_calls += 1
            response = invoke_with_fallback(
                [
                    SystemMessage(content=COMPACT_VALIDATION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                circuit_breaker=breaker,
                max_tokens=settings.qwen_max_output_tokens_review,
                temperature=0.0,
                models_to_try=(model,),
                trade_horizon="SCALP",
            )
            elapsed = perf_counter() - started
            latencies.append(elapsed)
            usage_record = record_llm_response(
                "qwen_local_paper_context_soak",
                response,
                model=model,
                trade_horizon="SCALP",
            )
            reviews = parse_context_reviews(response.content, [candidate])
            review = reviews[str(candidate["signal_id"])]
            input_tokens = int(usage_record.input_tokens) if usage_record is not None else 0
            output_tokens = int(usage_record.output_tokens) if usage_record is not None else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            cache.set(
                CandidateVerdict.create(
                    fingerprint=fingerprint,
                    decision=review.downstream_decision,
                    confidence=review.confidence,
                    reason_codes=list(review.supporting_factors),
                    risk_flags=list(review.risk_factors),
                    ttl_seconds=3600,
                    model_used=model,
                    market_context_version=str(
                        state["shared_market_context"]["context_version"]
                    ),
                    symbol_context_version="",
                    context_state=review.state.value,
                    handling=review.handling.value,
                    unresolved_conflicts=list(review.unresolved_conflicts),
                )
            )
            # Prove unchanged replay is a cache hit without issuing another request.
            if cache.get(fingerprint) is not None:
                qwen_cache_hits += 1
            try:
                risk_reasons = _risk_must_reject(index)
                risk_rejected = True
            except Exception as exc:
                risk_bypasses += 1
                risk_rejected = False
                risk_reasons = [str(exc)]
            rows.append(
                {
                    "candidate": candidate["signal_id"],
                    "qwen_required": True,
                    "cache_hit": False,
                    "latency_seconds": elapsed,
                    "schema_ok": True,
                    "context_state": review.state.value,
                    "handling": review.handling.value,
                    "response_sha256": hashlib.sha256(
                        str(response.content).encode()
                    ).hexdigest(),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "deterministic_risk_rejected": risk_rejected,
                    "risk_reasons": risk_reasons,
                }
            )
        except (ValueError, json.JSONDecodeError) as exc:
            schema_failures += 1
            rows.append({"candidate": candidate["signal_id"], "schema_error": str(exc)})
        except Exception as exc:
            failures += 1
            rows.append({"candidate": candidate["signal_id"], "error": str(exc)})

    report = {
        "provider": current_provider(),
        "model": model,
        "execution_mode": "local_paper_no_execution_service",
        "candidates_considered": len(candidates),
        "qwen_reviews_required": args.calls,
        "qwen_calls": qwen_calls,
        "qwen_skipped_clear": qwen_skipped_clear,
        "qwen_cache_hits": qwen_cache_hits,
        "failures": failures,
        "schema_failures": schema_failures,
        "deterministic_risk_bypasses": risk_bypasses,
        "latency_p50_seconds": median(latencies) if latencies else None,
        "latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "tokens_per_review": (
            (total_input_tokens + total_output_tokens) / qwen_calls if qwen_calls else 0.0
        ),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if failures == schema_failures == risk_bypasses == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
