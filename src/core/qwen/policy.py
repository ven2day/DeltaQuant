"""Deterministic policy and schema for selective Qwen candidate review.

This module does not make trading decisions.  It decides whether compact contextual
reasoning is useful and validates the shape of Qwen's advisory response.  Deterministic
signal validation and risk compliance remain downstream authorities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContextReviewState(str, Enum):
    SUPPORTIVE = "SUPPORTIVE"
    NEUTRAL = "NEUTRAL"
    CAUTION = "CAUTION"
    MATERIAL_CONFLICT = "MATERIAL_CONFLICT"


class ContextHandling(str, Enum):
    CONTINUE = "CONTINUE"
    CONTINUE_WITH_CAUTION = "CONTINUE_WITH_CAUTION"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    CONTEXT_REJECT = "CONTEXT_REJECT"


@dataclass(frozen=True)
class QwenReviewDecision:
    required: bool
    reason_codes: tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class ContextReview:
    signal_id: str
    state: ContextReviewState
    confidence: float
    supporting_factors: tuple[str, ...] = ()
    risk_factors: tuple[str, ...] = ()
    unresolved_conflicts: tuple[str, ...] = ()
    handling: ContextHandling = ContextHandling.REQUIRE_REVIEW

    @property
    def downstream_decision(self) -> str:
        return (
            "approve"
            if self.handling
            in {ContextHandling.CONTINUE, ContextHandling.CONTINUE_WITH_CAUTION}
            else "reject"
        )

    def validation_dict(self, *, cache_hit: bool, model_used: str) -> dict[str, Any]:
        return {
            "decision": self.downstream_decision,
            "context_state": self.state.value,
            "handling": self.handling.value,
            "confidence": self.confidence,
            "reason_codes": list(self.supporting_factors),
            "risk_flags": list(self.risk_factors),
            "unresolved_conflicts": list(self.unresolved_conflicts),
            "reasoning": ", ".join((*self.risk_factors, *self.unresolved_conflicts))
            or self.state.value,
            "cache_hit": cache_hit,
            "model_used": model_used,
        }


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _timeframe_conflict(signal: dict[str, Any]) -> bool:
    states = signal.get("timeframe_states") or signal.get("multi_timeframe_context") or {}
    if not isinstance(states, dict):
        return False
    direction = str(signal.get("signal_type", signal.get("direction", ""))).upper()
    opposed = {"BEARISH", "SELL", "DOWN", "TRENDING_DOWN"}
    if direction == "SELL":
        opposed = {"BULLISH", "BUY", "UP", "TRENDING_UP"}
    return any(str(value).upper() in opposed for value in states.values())


class QwenReviewPolicy:
    """Request context only for material ambiguity, never by rank or Top-K."""

    def __init__(
        self,
        *,
        regime_uncertain_below: float = 0.60,
        ml_conflict_below: float = 0.45,
        borderline_margin: float = 0.05,
    ) -> None:
        self.regime_uncertain_below = regime_uncertain_below
        self.ml_conflict_below = ml_conflict_below
        self.borderline_margin = borderline_margin

    def assess(
        self,
        signal: dict[str, Any],
        *,
        regime_confidence: float,
        shared_context: dict[str, Any] | None = None,
    ) -> QwenReviewDecision:
        reasons: list[str] = []
        priority = 0
        if "conflict_level" not in signal or "ml_status" not in signal:
            # Legacy callers did not persist enough material state to prove that a
            # candidate is clear.  Review rather than silently blessing incomplete
            # context; the event runtime always supplies both fields.
            reasons.append("CONTEXT_STATE_INCOMPLETE")
            priority += 15
        conflict = str(signal.get("conflict_level", "NONE")).upper()
        if conflict in {"MEDIUM", "HIGH"} or signal.get("opposing_strategies"):
            reasons.append("STRATEGY_CONFLICT")
            priority += 40 if conflict == "HIGH" else 30
        if _timeframe_conflict(signal):
            reasons.append("MULTI_TIMEFRAME_CONFLICT")
            priority += 35

        context = shared_context or {}
        event_risk = str(
            signal.get("news_event_risk", context.get("major_market_event_risk", "NONE"))
        ).upper()
        if event_risk not in {"", "NONE", "LOW", "NEUTRAL"}:
            reasons.append("MATERIAL_NEWS_EVENT")
            priority += 50
        if bool(signal.get("material_corporate_event", False)):
            reasons.append("MATERIAL_CORPORATE_EVENT")
            priority += 60

        if regime_confidence < self.regime_uncertain_below or str(
            signal.get("regime", "")
        ).upper() in {"UNKNOWN", "UNCERTAIN", "TRANSITION"}:
            reasons.append("REGIME_UNCERTAINTY")
            priority += 20

        ranking = signal.get("ranking") or {}
        if not isinstance(ranking, dict):
            ranking = {}
        probability = _as_number(
            signal.get("ml_probability", ranking.get("ml_direction_probability"))
        )
        ml_status = str(signal.get("ml_status", signal.get("artifact_validation_status", "")))
        if probability is not None and ml_status.upper() in {"VALIDATED", "AVAILABLE"}:
            direction = str(signal.get("signal_type", "BUY")).upper()
            conflict_probability = probability if direction == "BUY" else 1.0 - probability
            if conflict_probability < self.ml_conflict_below:
                reasons.append("TECHNICAL_ML_CONFLICT")
                priority += 35

        confidence = _as_number(
            signal.get("technical_quality", signal.get("confidence"))
        )
        threshold = _as_number(signal.get("qualification_threshold"))
        if confidence is not None and threshold is not None and abs(confidence - threshold) <= self.borderline_margin:
            reasons.append("BORDERLINE_SETUP")
            priority += 10

        return QwenReviewDecision(bool(reasons), tuple(dict.fromkeys(reasons)), priority)


def build_context_packet(signal: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Whitelist compact evidence.  Raw OHLC/history/dataframes cannot enter."""

    ranking = signal.get("ranking") or {}
    if not isinstance(ranking, dict):
        ranking = {}
    indicators = signal.get("indicators") or {}
    if not isinstance(indicators, dict):
        indicators = {}
    shared = state.get("shared_market_context") or {}
    headlines = shared.get("major_news_summary") or []
    return {
        "signal_id": signal.get("signal_id"),
        "symbol": signal.get("symbol"),
        "timeframe": signal.get("timeframe"),
        "trade_horizon": signal.get("trade_horizon", state.get("trade_horizon", "SWING")),
        "direction": signal.get("signal_type"),
        "strategies": {
            "supporting": list(signal.get("supporting_strategies") or [signal.get("strategy")]),
            "opposing": list(signal.get("opposing_strategies") or []),
            "agreement": signal.get("agreement_level", "SINGLE"),
            "conflict": signal.get("conflict_level", "NONE"),
        },
        "regime": {
            "state": state.get("regime", signal.get("regime", "unknown")),
            "confidence": state.get("regime_confidence", 0.0),
        },
        "multi_timeframe": signal.get("timeframe_states", {}),
        "features": {
            "adx": signal.get("adx", indicators.get("adx")),
            "rsi": signal.get("rsi", indicators.get("rsi")),
            "relative_volume": signal.get("volume_ratio"),
            "vwap_distance_pct": signal.get("vwap_distance_pct"),
            "technical_quality": signal.get("technical_quality", signal.get("confidence")),
            "expected_r": signal.get(
                "expected_r", ranking.get("expected_r_multiple", signal.get("risk_reward_ratio"))
            ),
        },
        "ml": {
            "status": signal.get("ml_status", "ABSTAIN"),
            "probability": signal.get(
                "ml_probability", ranking.get("ml_direction_probability")
            ),
            "model_version": signal.get("ml_model_version", ""),
        },
        "news": {
            "event_risk": shared.get("major_market_event_risk", "NONE"),
            "sentiment": shared.get("market_sentiment"),
            "summaries": [str(headline)[:180] for headline in list(headlines)[:3]],
            "context_version": shared.get("context_version", ""),
        },
    }


def serialize_context_packets(
    signals: list[dict[str, Any]], state: dict[str, Any], *, max_input_tokens: int
) -> str:
    payload = {
        "reviews": [build_context_packet(signal, state) for signal in signals],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    # Conservative four characters/token guard.  Packets are already whitelisted;
    # refusing an oversized batch is safer than silently dropping fields/candidates.
    if len(text) > max(1, max_input_tokens) * 4:
        raise ValueError("compact Qwen context exceeds configured input-token ceiling")
    return text


def parse_context_reviews(
    content: str, expected_signals: list[dict[str, Any]]
) -> dict[str, ContextReview]:
    """Strict schema parser.  Missing/malformed reviews are absent and fail closed."""

    text = content.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        text = text[start : text.find("```", start)].strip()
    elif "```" in text:
        start = text.find("```") + 3
        text = text[start : text.find("```", start)].strip()
    elif "{" in text and not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        text = text[start : end + 1]
    parsed = json.loads(text)
    items = parsed.get("context_reviews")
    legacy = False
    if items is None and isinstance(parsed.get("validations"), list):
        items = parsed["validations"]
        legacy = True
    if not isinstance(items, list):
        raise ValueError("Qwen response missing context_reviews array")
    expected_ids = {str(signal.get("signal_id")) for signal in expected_signals}
    result: dict[str, ContextReview] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Qwen context review item is not an object")
        signal_id = str(item.get("signal_id", ""))
        if signal_id not in expected_ids or signal_id in result:
            raise ValueError("Qwen context review has unknown or duplicate signal_id")
        if legacy:
            approved = str(item.get("decision", "reject")).lower() == "approve"
            item = {
                **item,
                "state": "SUPPORTIVE" if approved else "MATERIAL_CONFLICT",
                "handling": "CONTINUE" if approved else "CONTEXT_REJECT",
                "supporting_factors": item.get("reason_codes", []),
                "risk_factors": item.get("risk_flags", []),
                "unresolved_conflicts": [],
            }
        state = ContextReviewState(str(item.get("state", "")).upper())
        handling = ContextHandling(str(item.get("handling", "")).upper())
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        result[signal_id] = ContextReview(
            signal_id=signal_id,
            state=state,
            confidence=confidence,
            supporting_factors=tuple(str(v)[:80] for v in item.get("supporting_factors", [])[:6]),
            risk_factors=tuple(str(v)[:80] for v in item.get("risk_factors", [])[:6]),
            unresolved_conflicts=tuple(
                str(v)[:80] for v in item.get("unresolved_conflicts", [])[:6]
            ),
            handling=handling,
        )
    if set(result) != expected_ids:
        raise ValueError("Qwen response omitted one or more requested candidates")
    return result
