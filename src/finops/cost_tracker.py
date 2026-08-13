"""
FinOps — LLM cost & token accounting.

Tracks provider token usage and paid-tier-equivalent cost by market, reason,
component, model, and IST day. The default
free tier bills $0, but we still account tokens so the same code works on the paid
tier and so the operator can see "this run would cost $X on the paid tier".

This module is pure accounting — no network I/O — so it is safe to call from inside
sync agent nodes. Durable writes are bounded and failure-isolated. Alerts and
notifications live in ``finops.alerts``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.finops.attribution import (
    CONTEXT_SUPPORT_REASONS,
    LLMCallReason,
    LLMMarket,
    normalize_call_reason,
    normalize_llm_market,
)
from src.finops.storage import FinOpsEventStore
from src.markets.nse.sessions.market_time import now_ist

logger = logging.getLogger(__name__)


# List prices in USD per 1M tokens (input, output), across every supported provider
# (see src/agents/llm_factory.py) -- model names are unique across providers, so one
# flat dict keyed by model name (rather than namespacing per-provider) is sufficient;
# CostTracker.price_for() just looks up whatever model string record_llm_response() was
# given. These are estimates for paid-tier-equivalent cost and are easily overridden via
# register_pricing(); the free tier bills $0 regardless. Unknown models use the most
# expensive listed text-model rates so a model rename cannot silently bypass the spend
# gate. Gemini/DeepSeek figures are approximate list prices at time of writing --
# verify against the provider's current pricing page before relying on them for a real
# budget decision.
DEFAULT_GROQ_PRICING: dict[str, tuple[float, float]] = {
    # Groq
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.6-27b": (0.60, 3.00),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama3-70b-8192": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
    # Gemini (Google AI Studio list pricing, approximate)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # DeepSeek (approximate; DeepSeek also has a separate, cheaper cache-hit input
    # rate this table does not model -- it uses the standard cache-miss rate)
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Qwen (Alibaba Cloud DashScope Model Marketplace, confirmed from the console:
    # $0.32-0.96/M input, $1.28-3.84/M output, tiered by context length -- the
    # higher/long-context tier is used here since underestimating a per-call cost is
    # the wrong direction to be wrong in for a spend gate)
    "qwen3.7-plus": (0.96, 3.84),
}
_FALLBACK_PRICING: tuple[float, float] = (0.60, 3.00)


@dataclass
class UsageRecord:
    """A single LLM call's usage."""

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str  # IST isoformat
    trade_horizon: str = "SHARED"
    purpose: str = ""
    cycle_id: str = ""
    candidate_id: str = ""
    symbol: str = ""
    cache_hit: bool = False
    market: str = LLMMarket.UNKNOWN_LEGACY.value
    call_reason: str = LLMCallReason.UNKNOWN_LEGACY.value
    component: str = "unknown"
    runtime_id: str = ""
    session_id: str = ""
    timeframe: str = ""
    trace_id: str = ""
    call_id: str = ""
    call_count: int = 1
    metadata: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
            "trade_horizon": self.trade_horizon,
            "purpose": self.purpose,
            "cycle_id": self.cycle_id,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "cache_hit": self.cache_hit,
            "market": self.market,
            "call_reason": self.call_reason,
            "component": self.component,
            "runtime_id": self.runtime_id,
            "session_id": self.session_id,
            "timeframe": self.timeframe,
            "trace_id": self.trace_id,
            "call_id": self.call_id,
            "call_count": self.call_count,
            "metadata": dict(self.metadata or {}),
        }


class CostTracker:
    """Thread-safe, durable and market-attributed LLM usage accounting."""

    def __init__(
        self,
        pricing: dict[str, tuple[float, float]] | None = None,
        *,
        store: FinOpsEventStore | None = None,
        runtime_id: str = "",
        session_id: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._pricing: dict[str, tuple[float, float]] = dict(pricing or DEFAULT_GROQ_PRICING)
        self._recent_records: list[UsageRecord] = []
        self._unpersisted_records: list[UsageRecord] = []
        self._store = store
        self.runtime_id = str(runtime_id or "deltaquant")
        self.session_id = str(session_id or uuid.uuid4())

    @staticmethod
    def _today() -> str:
        return str(now_ist().date().isoformat())

    def register_pricing(self, model: str, input_per_1m: float, output_per_1m: float) -> None:
        """Override pricing for a model (USD per 1M tokens)."""
        with self._lock:
            self._pricing[model] = (input_per_1m, output_per_1m)

    def price_for(self, model: str) -> tuple[float, float]:
        return self._pricing.get(model, _FALLBACK_PRICING)

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self.price_for(model)
        return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

    def record_usage(
        self,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        trade_horizon: str = "SHARED",
        *,
        purpose: str = "",
        cycle_id: str = "",
        candidate_id: str = "",
        symbol: str = "",
        cache_hit: bool = False,
        market: str | LLMMarket | None = None,
        call_reason: str | LLMCallReason | None = None,
        component: str = "",
        runtime_id: str = "",
        session_id: str = "",
        timeframe: str = "",
        trace_id: str = "",
        call_id: str = "",
        call_count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record one LLM call's token usage and accrue cost."""
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cost = self.estimate_cost(model, input_tokens, output_tokens)

        normalized_market = normalize_llm_market(market).value
        normalized_reason = normalize_call_reason(call_reason).value
        record = UsageRecord(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=now_ist().isoformat(),
            trade_horizon=str(trade_horizon or "SHARED").upper(),
            purpose=purpose or agent,
            cycle_id=str(cycle_id),
            candidate_id=str(candidate_id),
            symbol=str(symbol),
            cache_hit=bool(cache_hit),
            market=normalized_market,
            call_reason=normalized_reason,
            component=str(component or agent),
            runtime_id=str(runtime_id or self.runtime_id),
            session_id=str(session_id or self.session_id),
            timeframe=str(timeframe),
            trace_id=str(trace_id),
            call_id=str(call_id or uuid.uuid4()),
            call_count=max(1, int(call_count)),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._recent_records.append(record)
            if len(self._recent_records) > 500:
                self._recent_records = self._recent_records[-500:]
        if self._store is not None:
            try:
                payload = record.to_dict()
                payload["date_ist"] = self._today()
                self._store.upsert(payload)
            except Exception as exc:  # pragma: no cover - defensive persistence boundary
                with self._lock:
                    self._unpersisted_records.append(record)
                logger.warning("FinOps persistence degraded: %s", type(exc).__name__)
        logger.info(
            "LLM_CALL market=%s call_reason=%s component=%s model=%s "
            "input_tokens=%d output_tokens=%d total_tokens=%d cost_usd=%.8f "
            "runtime_id=%s session_id=%s cycle_id=%s candidate_id=%s symbol=%s "
            "timeframe=%s trace_id=%s",
            record.market,
            record.call_reason,
            record.component,
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.total_tokens,
            record.cost_usd,
            record.runtime_id,
            record.session_id,
            record.cycle_id,
            record.candidate_id,
            record.symbol,
            record.timeframe,
            record.trace_id,
        )
        return record

    def recent_records(
        self, limit: int = 100, *, market: str | None = None
    ) -> list[dict[str, Any]]:
        """Return bounded call attribution; no prompt or response content is retained."""

        safe_limit = max(0, min(int(limit), 500))
        if safe_limit == 0:
            return []
        if self._store is not None:
            try:
                normalized_market = normalize_llm_market(market).value if market else None
                records = self._store.records(
                    date_ist=self._today(), market=normalized_market, limit=10_000
                )
                return records[-safe_limit:]
            except Exception as exc:  # pragma: no cover - defensive read boundary
                logger.warning("FinOps history read degraded: %s", type(exc).__name__)
        with self._lock:
            records = [record.to_dict() for record in self._recent_records]
        if market:
            normalized_market = normalize_llm_market(market).value
            records = [item for item in records if item.get("market") == normalized_market]
        return records[-safe_limit:]

    @staticmethod
    def _empty_summary(date: str) -> dict[str, Any]:
        return {
            "date": date,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "candidate_review_calls": 0,
            "context_support_calls": 0,
            "by_agent": {},
            "by_horizon": {},
            "by_market": {},
            "by_reason": {},
            "by_component": {},
            "by_model": {},
        }

    @staticmethod
    def _add_bucket(container: dict[str, Any], key: str, record: dict[str, Any]) -> None:
        bucket = container.setdefault(
            key or "UNKNOWN_LEGACY",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0},
        )
        count = max(1, int(record.get("call_count", 1) or 1))
        input_tokens = int(record.get("input_tokens", 0) or 0)
        output_tokens = int(record.get("output_tokens", 0) or 0)
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += input_tokens + output_tokens
        bucket["cost_usd"] += float(record.get("cost_usd", 0.0) or 0.0)
        bucket["calls"] += count

    def _summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        summary = self._empty_summary(self._today())
        for record in records:
            count = max(1, int(record.get("call_count", 1) or 1))
            input_tokens = int(record.get("input_tokens", 0) or 0)
            output_tokens = int(record.get("output_tokens", 0) or 0)
            reason = normalize_call_reason(record.get("call_reason")).value
            summary["calls"] += count
            summary["input_tokens"] += input_tokens
            summary["output_tokens"] += output_tokens
            summary["total_tokens"] += input_tokens + output_tokens
            summary["total_cost_usd"] += float(record.get("cost_usd", 0.0) or 0.0)
            if reason == LLMCallReason.CANDIDATE_REVIEW.value:
                summary["candidate_review_calls"] += count
            if reason in CONTEXT_SUPPORT_REASONS:
                summary["context_support_calls"] += count
            for output_key, record_key in (
                ("by_agent", "agent"),
                ("by_horizon", "trade_horizon"),
                ("by_market", "market"),
                ("by_reason", "call_reason"),
                ("by_component", "component"),
                ("by_model", "model"),
            ):
                self._add_bucket(summary[output_key], str(record.get(record_key, "")), record)
        return summary

    def _records(
        self,
        *,
        market: str | None = None,
        session_id: str | None = None,
        cycle_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_market = normalize_llm_market(market).value if market else None
        if self._store is not None:
            try:
                records = self._store.records(
                    date_ist=self._today(),
                    market=normalized_market,
                    session_id=session_id,
                    cycle_id=cycle_id,
                    since=since,
                )
                with self._lock:
                    unpersisted = [record.to_dict() for record in self._unpersisted_records]
                if normalized_market:
                    unpersisted = [
                        item for item in unpersisted if item.get("market") == normalized_market
                    ]
                if session_id:
                    unpersisted = [
                        item for item in unpersisted if item.get("session_id") == session_id
                    ]
                if cycle_id:
                    unpersisted = [
                        item for item in unpersisted if item.get("cycle_id") == cycle_id
                    ]
                if since:
                    unpersisted = [
                        item
                        for item in unpersisted
                        if str(item.get("timestamp", "")) >= since
                    ]
                return [*records, *unpersisted]
            except Exception as exc:  # pragma: no cover - defensive read boundary
                logger.warning("FinOps aggregate read degraded: %s", type(exc).__name__)
        with self._lock:
            records = [record.to_dict() for record in self._recent_records]
        if normalized_market:
            records = [item for item in records if item.get("market") == normalized_market]
        if session_id:
            records = [item for item in records if item.get("session_id") == session_id]
        if cycle_id:
            records = [item for item in records if item.get("cycle_id") == cycle_id]
        if since:
            records = [item for item in records if str(item.get("timestamp", "")) >= since]
        return records

    def daily_summary(self, market: str | None = None) -> dict[str, Any]:
        """Snapshot of today's IST usage, optionally scoped to one market."""
        return self._summarize(self._records(market=market))

    def session_summary(self, market: str | None = None) -> dict[str, Any]:
        return self._summarize(self._records(market=market, session_id=self.session_id))

    def cycle_summary(self, cycle_id: str, market: str | None = None) -> dict[str, Any]:
        return self._summarize(self._records(market=market, cycle_id=str(cycle_id)))

    def summary_since(self, timestamp: str, market: str | None = None) -> dict[str, Any]:
        return self._summarize(self._records(market=market, since=timestamp))

    @property
    def today_tokens(self) -> int:
        s = self.daily_summary()
        return int(s["total_tokens"])

    @property
    def today_cost_usd(self) -> float:
        return float(self.daily_summary()["total_cost_usd"])

    @property
    def today_calls(self) -> int:
        return int(self.daily_summary()["calls"])

    def budget_status(self, trade_horizon: str | None = None) -> dict[str, Any]:
        """
        Compare today's usage against the configured daily budgets.

        A budget of 0 (the default) means "unlimited" and never breaches. Soft breach
        fires at ``finops_budget_soft_pct`` of either budget; hard breach at 100%.
        """
        settings = get_settings()

        def _int_setting(name: str) -> int:
            value = getattr(settings, name, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        legacy_token_budget = _int_setting("daily_token_budget")
        configured_total = _int_setting("llm_daily_budget_total")
        positive_totals = [value for value in (legacy_token_budget, configured_total) if value > 0]
        token_budget = min(positive_totals) if positive_totals else 0
        cost_budget = float(getattr(settings, "daily_cost_budget_usd", 0.0) or 0.0)
        soft_pct = float(getattr(settings, "finops_budget_soft_pct", 0.8) or 0.8)

        summary = self.daily_summary()
        tokens_used = int(summary["total_tokens"])
        cost_used = float(summary["total_cost_usd"])
        horizon = str(trade_horizon or "").upper()
        horizon_budget = 0
        if horizon == "SCALP":
            horizon_budget = _int_setting("llm_daily_budget_scalp")
        elif horizon == "SWING":
            horizon_budget = _int_setting("llm_daily_budget_swing")
        horizon_usage = summary["by_horizon"].get(horizon, {}) if horizon else {}
        horizon_tokens_used = int(
            float(horizon_usage.get("input_tokens", 0.0))
            + float(horizon_usage.get("output_tokens", 0.0))
        )

        token_soft = token_budget > 0 and tokens_used >= token_budget * soft_pct
        token_hard = token_budget > 0 and tokens_used >= token_budget
        cost_soft = cost_budget > 0 and cost_used >= cost_budget * soft_pct
        cost_hard = cost_budget > 0 and cost_used >= cost_budget
        horizon_soft = horizon_budget > 0 and horizon_tokens_used >= horizon_budget * soft_pct
        horizon_hard = horizon_budget > 0 and horizon_tokens_used >= horizon_budget

        return {
            "date": summary["date"],
            "tokens_used": tokens_used,
            "token_budget": token_budget,
            "cost_used_usd": cost_used,
            "cost_budget_usd": cost_budget,
            "trade_horizon": horizon or None,
            "horizon_tokens_used": horizon_tokens_used,
            "horizon_token_budget": horizon_budget,
            "soft_pct": soft_pct,
            "soft_breached": bool(token_soft or cost_soft or horizon_soft),
            "hard_breached": bool(token_hard or cost_hard or horizon_hard),
            "global_hard_breached": bool(token_hard or cost_hard),
            "horizon_hard_breached": bool(horizon_hard),
        }

    def is_over_hard_budget(self, trade_horizon: str | None = None) -> bool:
        return bool(self.budget_status(trade_horizon)["hard_breached"])


# --- Module-level singleton + LLM-response helper -------------------------------

_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def _import_legacy_snapshot_totals(store: FinOpsEventStore, repository_root: Path) -> None:
    """Preserve old global counters without inventing missing attribution.

    Pre-attribution snapshots contain aggregate calls/tokens/cost only. They cannot
    be decomposed into reliable per-call events, so one explicitly legacy aggregate
    is imported per market. The deterministic id makes concurrent worker startup
    idempotent.
    """
    today = CostTracker._today()
    for market in (LLMMarket.NSE.value, LLMMarket.FOREX.value):
        legacy_call_id = f"legacy-snapshot-{today}-{market}"
        existing = store.records(date_ist=today, market=market, limit=10_000)
        # Once a newly attributed call exists, the legacy baseline is frozen. This
        # prevents an old snapshot from overwriting or double-counting live events.
        if any(str(item.get("call_id")) != legacy_call_id for item in existing):
            continue
        path = repository_root / "run" / market.lower() / "market_snapshot.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        state = payload.get("dashboard_state", {}) if isinstance(payload, dict) else {}
        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        source = state if isinstance(state, dict) and state else status
        if not isinstance(source, dict):
            continue
        calls = int(source.get("llm_calls", 0) or 0)
        tokens = int(source.get("llm_tokens", 0) or 0)
        if calls <= 0 and tokens <= 0:
            continue
        if existing and max(int(item.get("call_count", 0) or 0) for item in existing) >= calls:
            continue
        input_tokens = int(source.get("llm_input_tokens", 0) or 0)
        output_tokens = int(source.get("llm_output_tokens", 0) or 0)
        if input_tokens + output_tokens == 0:
            input_tokens = tokens
        generated_at = str(payload.get("generated_at", now_ist().isoformat()))
        store.upsert(
            {
                "call_id": legacy_call_id,
                "timestamp": generated_at,
                "date_ist": today,
                "market": market,
                "call_reason": LLMCallReason.UNKNOWN_LEGACY.value,
                "component": "UNKNOWN_LEGACY",
                "agent": "UNKNOWN_LEGACY",
                "model": "UNKNOWN_LEGACY",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": float(source.get("llm_cost_usd", 0.0) or 0.0),
                "call_count": max(1, calls),
                "trade_horizon": "SHARED",
                "runtime_id": str(source.get("runtime_id", "legacy")),
                "session_id": "UNKNOWN_LEGACY",
                "cycle_id": "",
                "candidate_id": "",
                "symbol": "",
                "timeframe": "",
                "trace_id": "",
                "cache_hit": False,
                "metadata": {"source": "pre_attribution_market_snapshot"},
            }
        )


def get_cost_tracker() -> CostTracker:
    """Get or create the shared CostTracker."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                settings = get_settings()
                runtime_id = str(getattr(settings, "runtime_id", "deltaquant") or "deltaquant")
                # Tests construct explicit stores when persistence is under test. Keeping
                # the implicit singleton in-memory under pytest prevents a developer's
                # real daily ledger from contaminating unit-test assertions.
                store: FinOpsEventStore | None = None
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    configured = os.environ.get("FINOPS_DB_PATH", "").strip()
                    path = (
                        Path(configured)
                        if configured
                        else Path(__file__).resolve().parents[2]
                        / "run"
                        / "common"
                        / "finops"
                        / "llm_usage.sqlite3"
                    )
                    try:
                        store = FinOpsEventStore(path)
                        _import_legacy_snapshot_totals(
                            store, Path(__file__).resolve().parents[2]
                        )
                    except Exception as exc:  # pragma: no cover - runtime boundary
                        logger.warning("FinOps durable store unavailable: %s", type(exc).__name__)
                _tracker = CostTracker(store=store, runtime_id=runtime_id)
    return _tracker


def reset_cost_tracker() -> None:
    """Reset the shared CostTracker (test isolation)."""
    global _tracker
    with _tracker_lock:
        _tracker = None


def _extract_usage(response: Any) -> tuple[int, int] | None:
    """Pull (input_tokens, output_tokens) out of a LangChain chat response."""
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict) and usage_metadata:
        return (
            int(usage_metadata.get("input_tokens", 0) or 0),
            int(usage_metadata.get("output_tokens", 0) or 0),
        )
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        if isinstance(token_usage, dict) and token_usage:
            return (
                int(token_usage.get("prompt_tokens", 0) or 0),
                int(token_usage.get("completion_tokens", 0) or 0),
            )
    return None


def _extract_model(response: Any) -> str | None:
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        model = response_metadata.get("model_name") or response_metadata.get("model")
        if model:
            return str(model)
    return None


def record_llm_response(
    agent: str,
    response: Any,
    model: str | None = None,
    trade_horizon: str = "SHARED",
    *,
    purpose: str = "",
    cycle_id: str = "",
    candidate_id: str = "",
    symbol: str = "",
    cache_hit: bool = False,
    market: str | LLMMarket | None = None,
    call_reason: str | LLMCallReason | None = None,
    component: str = "",
    runtime_id: str = "",
    session_id: str = "",
    timeframe: str = "",
    candidate_metadata: dict[str, Any] | None = None,
) -> UsageRecord | None:
    """
    Record token usage from a LangChain chat response. Never raises — FinOps
    accounting must never crash an agent node.
    """
    try:
        settings = get_settings()
        if not getattr(settings, "finops_enabled", True):
            return None
        usage = _extract_usage(response)
        if usage is None:
            logger.debug("FinOps: no usage metadata on response from %s", agent)
            return None
        input_tokens, output_tokens = usage
        resolved_model = model or _extract_model(response) or settings.groq_model_primary
        tracker = get_cost_tracker()
        resolved_market = normalize_llm_market(market)
        resolved_reason = normalize_call_reason(call_reason)
        resolved_component = component or agent
        trace_metadata = {
            "market": resolved_market.value,
            "call_reason": resolved_reason.value,
            "component": resolved_component,
            "agent": agent,
            "runtime_id": runtime_id or str(getattr(settings, "runtime_id", "")),
            "session_id": session_id or tracker.session_id,
            "cycle_id": cycle_id,
            "candidate_id": candidate_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "model": resolved_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": tracker.estimate_cost(resolved_model, input_tokens, output_tokens),
        }
        trace_id = ""
        try:
            from src.observability.tracing import record_llm_call_trace

            trace_id = record_llm_call_trace(trace_metadata)
        except Exception as exc:  # pragma: no cover - observability boundary
            logger.debug("LLM Langfuse attribution failed safely: %s", type(exc).__name__)
        return tracker.record_usage(
            agent,
            resolved_model,
            input_tokens,
            output_tokens,
            trade_horizon=trade_horizon,
            purpose=purpose,
            cycle_id=cycle_id,
            candidate_id=candidate_id,
            symbol=symbol,
            cache_hit=cache_hit,
            market=resolved_market,
            call_reason=resolved_reason,
            component=resolved_component,
            runtime_id=runtime_id or str(getattr(settings, "runtime_id", "")),
            session_id=session_id,
            timeframe=timeframe,
            trace_id=trace_id,
            metadata=candidate_metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("FinOps record_llm_response failed (non-fatal): %s", exc)
        return None
