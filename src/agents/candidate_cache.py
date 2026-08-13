"""Material-state fingerprinting and TTL cache for Qwen candidate verdicts.

The cache stores only the contextual validation verdict. It never stores or reuses
position sizing, deterministic risk decisions, reservations, stale-price results, or
execution outcomes. Those controls must run again on every cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


def _quantize(value: Any, step: float) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number / step) * step


def _price_bucket(value: Any, material_move_bps: float) -> int | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    ratio = math.log1p(material_move_bps / 10_000.0)
    return int(round(math.log(price) / ratio)) if ratio > 0 else None


def candidate_material_state(
    signal: dict[str, Any],
    *,
    regime: str,
    market_context_version: str,
    symbol_context_version: str = "",
    material_price_move_bps: float = 20.0,
    ml_probability_step: float = 0.05,
    expected_r_step: float = 0.10,
    entry_score_step: float = 5.0,
) -> dict[str, Any]:
    """Return the canonical material state used to fingerprint one candidate."""
    ranking = signal.get("ranking") or signal.get("indicators", {}).get("ranking", {})
    if not isinstance(ranking, dict):
        ranking = {}
    ml_probability = signal.get("ml_probability")
    if ml_probability is None:
        ml_probability = ranking.get("ml_direction_probability")
    expected_r = signal.get("expected_r")
    if expected_r is None:
        expected_r = ranking.get("expected_r_multiple", signal.get("risk_reward_ratio"))
    entry_score = signal.get("entry_quality_score", signal.get("confidence"))
    try:
        normalized_entry_score = float(entry_score) if entry_score is not None else None
    except (TypeError, ValueError):
        normalized_entry_score = None
    if normalized_entry_score is not None and normalized_entry_score <= 1.0:
        normalized_entry_score *= 100.0

    horizon = str(signal.get("trade_horizon", "SWING")).upper()
    return {
        "market": str(signal.get("market", "NSE")).upper(),
        "provider": str(signal.get("provider", signal.get("data_provider", ""))).upper(),
        "symbol": str(signal.get("symbol", "")).upper(),
        "trade_horizon": horizon,
        "primary_timeframe": str(signal.get("primary_timeframe", signal.get("timeframe", ""))),
        "direction": str(signal.get("signal_type", signal.get("direction", ""))).upper(),
        "signal_candle_timestamp": str(
            signal.get(
                "signal_candle_timestamp",
                signal.get("candle_timestamp", signal.get("timestamp", "")),
            )
        ),
        "strategy_identity": str(
            signal.get("strategy_consensus_identity", signal.get("strategy", ""))
        ),
        "strategy_version": str(signal.get("strategy_version", "")),
        "feature_snapshot_id": str(signal.get("feature_snapshot_id", "")),
        "feature_version": str(signal.get("feature_version", "")),
        "agreement_level": str(signal.get("agreement_level", "")),
        "conflict_level": str(signal.get("conflict_level", "")),
        "supporting_strategies": sorted(
            str(item) for item in signal.get("supporting_strategies", [])
        ),
        "opposing_strategies": sorted(
            str(item) for item in signal.get("opposing_strategies", [])
        ),
        "regime": regime,
        "entry_state": str(
            signal.get(
                "entry_state",
                signal.get("final_decision", signal.get("candidate_action", "")),
            )
        ),
        "entry_score_bucket": _quantize(normalized_entry_score, entry_score_step),
        "ml_probability_bucket": _quantize(ml_probability, ml_probability_step),
        "ml_status": str(signal.get("ml_status", "")),
        "ml_model_version": str(signal.get("ml_model_version", "")),
        "expected_r_bucket": _quantize(expected_r, expected_r_step),
        "entry_price_bucket": _price_bucket(signal.get("entry_price"), material_price_move_bps),
        "eligibility_model_version": str(
            signal.get("model_version", signal.get("strategy_version", ""))
        ),
        "eligibility_status": str(
            signal.get("eligibility_status", signal.get("validation_status", ""))
        ),
        "regime_policy": str(signal.get("regime_policy", "")),
        "market_context_version": market_context_version,
        "news_context_version": str(signal.get("news_context_version", "")),
        "sentiment_version": str(signal.get("sentiment_version", "")),
        "symbol_context_version": symbol_context_version,
    }


def build_candidate_fingerprint(
    signal: dict[str, Any],
    *,
    regime: str,
    market_context_version: str,
    symbol_context_version: str = "",
    material_price_move_bps: float = 20.0,
    ml_probability_step: float = 0.05,
    expected_r_step: float = 0.10,
    entry_score_step: float = 5.0,
) -> str:
    state = candidate_material_state(
        signal,
        regime=regime,
        market_context_version=market_context_version,
        symbol_context_version=symbol_context_version,
        material_price_move_bps=material_price_move_bps,
        ml_probability_step=ml_probability_step,
        expected_r_step=expected_r_step,
        entry_score_step=entry_score_step,
    )
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateVerdict:
    fingerprint: str
    decision: str
    confidence: float
    reason_codes: tuple[str, ...]
    risk_flags: tuple[str, ...]
    reviewed_at: str
    expires_at: str
    model_used: str
    market_context_version: str
    symbol_context_version: str
    context_state: str = "NEUTRAL"
    handling: str = "REQUIRE_REVIEW"
    unresolved_conflicts: tuple[str, ...] = ()

    @property
    def expired(self) -> bool:
        return datetime.now(UTC) >= datetime.fromisoformat(self.expires_at)

    def validation_dict(self, *, cache_hit: bool) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "risk_flags": list(self.risk_flags),
            "context_state": self.context_state,
            "handling": self.handling,
            "unresolved_conflicts": list(self.unresolved_conflicts),
            "reasoning": ", ".join(self.reason_codes) or "NO_CONTEXTUAL_CONFLICT",
            "cache_hit": cache_hit,
            "reviewed_at": self.reviewed_at,
            "model_used": self.model_used,
        }

    @classmethod
    def create(
        cls,
        *,
        fingerprint: str,
        decision: str,
        confidence: float,
        reason_codes: list[str],
        risk_flags: list[str],
        ttl_seconds: int,
        model_used: str,
        market_context_version: str,
        symbol_context_version: str,
        context_state: str = "NEUTRAL",
        handling: str = "REQUIRE_REVIEW",
        unresolved_conflicts: list[str] | None = None,
    ) -> CandidateVerdict:
        now = datetime.now(UTC)
        return cls(
            fingerprint=fingerprint,
            decision=decision,
            confidence=max(0.0, min(1.0, float(confidence))),
            reason_codes=tuple(str(code) for code in reason_codes[:8]),
            risk_flags=tuple(str(flag) for flag in risk_flags[:8]),
            reviewed_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=max(0, ttl_seconds))).isoformat(),
            model_used=model_used,
            market_context_version=market_context_version,
            symbol_context_version=symbol_context_version,
            context_state=str(context_state),
            handling=str(handling),
            unresolved_conflicts=tuple(str(item) for item in (unresolved_conflicts or [])[:8]),
        )


class CandidateVerdictCache:
    """Thread-safe in-memory verdict cache keyed by a material fingerprint."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, CandidateVerdict] = {}
        self.hits = 0
        self.misses = 0

    def get(self, fingerprint: str) -> CandidateVerdict | None:
        with self._lock:
            verdict = self._entries.get(fingerprint)
            if verdict is None:
                self.misses += 1
                return None
            if verdict.expired:
                self._entries.pop(fingerprint, None)
                self.misses += 1
                return None
            self.hits += 1
            return verdict

    def set(self, verdict: CandidateVerdict) -> None:
        with self._lock:
            self._entries[verdict.fingerprint] = verdict

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
            }


_candidate_cache = CandidateVerdictCache()


def get_candidate_verdict_cache() -> CandidateVerdictCache:
    return _candidate_cache


def reset_candidate_verdict_cache() -> None:
    _candidate_cache.clear()
