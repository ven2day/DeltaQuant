"""Optional Langfuse observability for agent and trading-cycle traces."""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from src.config import get_settings

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None
_OBSERVATION_TYPES = {
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
}


def setup_tracing() -> bool:
    """Configure the optional Langfuse client and verify its credentials."""
    global _langfuse_client

    settings = get_settings()
    if not settings.langfuse_tracing_enabled:
        logger.info("Langfuse tracing disabled by configuration")
        _langfuse_client = None
        return False

    public_key = settings.langfuse_public_key.get_secret_value()
    secret_key = settings.langfuse_secret_key.get_secret_value()
    if not public_key or not secret_key:
        logger.warning("Langfuse tracing enabled without both API keys; tracing disabled")
        _langfuse_client = None
        return False

    try:
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host,
            environment=settings.langfuse_environment,
        )
        client = get_client()
        if not client.auth_check():
            logger.warning("Langfuse authentication failed; tracing disabled")
            _langfuse_client = None
            return False

        _langfuse_client = client
        logger.info(
            "Langfuse tracing enabled for environment %s",
            settings.langfuse_environment,
        )
        return True
    except Exception as exc:
        _langfuse_client = None
        logger.warning("Langfuse tracing not available: %s", exc)
        return False


def get_langfuse_callback() -> CallbackHandler | None:
    """Return a callback for one LangGraph invocation when tracing is enabled."""
    if _langfuse_client is None:
        return None
    try:
        return CallbackHandler()
    except Exception as exc:
        logger.warning("Could not create Langfuse callback: %s", exc)
        return None


def trace_agent(
    agent_name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Decorate an agent function with a Langfuse observation."""
    observation_type = run_type if run_type in _OBSERVATION_TYPES else "chain"

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    "Agent %s completed in %dms",
                    agent_name,
                    int((time.time() - start_time) * 1000),
                )
                return result
            except Exception as exc:
                logger.error("Agent %s failed: %s", agent_name, exc)
                raise

        return observe(name=agent_name, as_type=observation_type)(wrapper)

    return decorator


@contextmanager
def trading_trace(
    workflow_id: str,
    regime: str | None = None,
    strategies: list[str] | None = None,
    signals_count: int = 0,
):
    """Trace one trading cycle without allowing observability to change its outcome."""
    metadata = {
        "workflow_id": workflow_id,
        "regime": regime,
        "strategies": strategies or [],
        "signals_count": signals_count,
        "started_at": datetime.now().isoformat(),
    }
    start_time = time.time()
    span_context = None
    span = None

    if _langfuse_client is not None:
        try:
            span_context = _langfuse_client.start_as_current_observation(
                as_type="span",
                name="trading_cycle",
                metadata=metadata,
            )
            span = span_context.__enter__()
        except Exception as trace_exc:
            logger.warning("Could not start Langfuse trading-cycle trace: %s", trace_exc)
            span_context = None

    try:
        yield metadata

        metadata["status"] = "success"
        metadata["duration_ms"] = int((time.time() - start_time) * 1000)
        if span is not None:
            try:
                span.update(metadata=metadata, output={"status": "success"})
            except Exception as trace_exc:
                logger.debug("Could not complete Langfuse trace: %s", trace_exc)
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = str(exc)
        metadata["duration_ms"] = int((time.time() - start_time) * 1000)
        if _langfuse_client is not None:
            try:
                _langfuse_client.update_current_span(
                    metadata=metadata,
                    status_message=str(exc),
                )
            except Exception as trace_exc:
                logger.debug("Could not record Langfuse trace error: %s", trace_exc)

        if span_context is not None:
            try:
                span_context.__exit__(type(exc), exc, exc.__traceback__)
                span_context = None
            except Exception as trace_exc:
                logger.debug("Could not close Langfuse trace: %s", trace_exc)
        raise
    finally:
        if span_context is not None:
            try:
                span_context.__exit__(None, None, None)
            except Exception as trace_exc:
                logger.debug("Could not close Langfuse trace: %s", trace_exc)
        logger.info(
            "Trading cycle %s completed: status=%s, duration=%sms",
            workflow_id,
            metadata.get("status"),
            metadata.get("duration_ms"),
        )


def add_trace_metadata(key: str, value: Any) -> None:
    """Attach one metadata field to the active Langfuse span when available."""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.update_current_span(metadata={key: value})
    except Exception as exc:
        logger.debug("Could not add Langfuse trace metadata: %s", exc)


def record_llm_call_trace(metadata: dict[str, Any]) -> str:
    """Attach one structured LLM-call attribution span to the active trace.

    Prompt and response bodies are deliberately excluded. The returned trace id is
    stored beside the FinOps event when the SDK exposes it; observability failure is
    always non-fatal.
    """
    if _langfuse_client is None:
        return ""
    safe_keys = {
        "market",
        "call_reason",
        "component",
        "agent",
        "runtime_id",
        "session_id",
        "cycle_id",
        "candidate_id",
        "symbol",
        "timeframe",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
    }
    safe = {key: value for key, value in metadata.items() if key in safe_keys}
    trace_id = ""
    try:
        current_trace_id = getattr(_langfuse_client, "get_current_trace_id", None)
        if callable(current_trace_id):
            trace_id = str(current_trace_id() or "")
        with _langfuse_client.start_as_current_observation(
            as_type="span",
            name="llm_call_attribution",
            metadata=safe,
        ) as span:
            if not trace_id:
                trace_id = str(getattr(span, "trace_id", "") or "")
            span.update(
                output={
                    "market": safe.get("market", "UNKNOWN_LEGACY"),
                    "call_reason": safe.get("call_reason", "UNKNOWN_LEGACY"),
                    "total_tokens": safe.get("total_tokens", 0),
                }
            )
    except Exception as exc:
        logger.debug("Could not record LLM attribution span: %s", exc)
    return trace_id


def record_candidate_review(metadata: dict[str, Any]) -> None:
    """Emit a bounded per-candidate span for fresh and cached Qwen verdicts.

    Only structured identifiers and decisions are accepted by callers; prompts,
    credentials, and model prose are deliberately excluded.
    """
    if _langfuse_client is None:
        return
    safe_keys = {
        "market",
        "provider",
        "provider_environment",
        "instrument",
        "symbol",
        "strategy",
        "trade_horizon",
        "timeframe",
        "model_version",
        "validation_status",
        "current_regime",
        "regime_policy",
        "original_confidence",
        "regime_adjusted_confidence",
        "ml_confidence",
        "registry_decision",
        "risk_decision",
        "final_outcome",
        "rejection_reason",
        "candidate_fingerprint",
        "cache_hit",
        "model",
        "context_version",
        "decision",
        "execution_result",
    }
    safe = {key: value for key, value in metadata.items() if key in safe_keys}
    try:
        with _langfuse_client.start_as_current_observation(
            as_type="span", name="qwen_candidate_review", metadata=safe
        ) as span:
            span.update(output={"decision": safe.get("decision", "unknown")})
    except Exception as exc:
        logger.debug("Could not record candidate-review span: %s", exc)


def record_candidate_execution(metadata: dict[str, Any]) -> None:
    """Emit the post-review execution result keyed by the same fingerprint."""
    if _langfuse_client is None:
        return
    safe_keys = {
        "market",
        "provider",
        "provider_environment",
        "instrument",
        "symbol",
        "strategy",
        "trade_horizon",
        "timeframe",
        "model_version",
        "validation_status",
        "current_regime",
        "regime_policy",
        "original_confidence",
        "regime_adjusted_confidence",
        "ml_confidence",
        "registry_decision",
        "risk_decision",
        "final_outcome",
        "rejection_reason",
        "candidate_fingerprint",
        "cache_hit",
        "model",
        "context_version",
        "decision",
        "execution_result",
    }
    safe = {key: value for key, value in metadata.items() if key in safe_keys}
    try:
        with _langfuse_client.start_as_current_observation(
            as_type="span", name="candidate_execution", metadata=safe
        ) as span:
            span.update(output={"execution_result": safe.get("execution_result", "unknown")})
    except Exception as exc:
        logger.debug("Could not record candidate-execution span: %s", exc)


def record_runtime_stage(
    stage: str,
    duration_seconds: float,
    metadata: dict[str, int | float | str | bool] | None = None,
) -> None:
    """Emit one small completed-stage span without attaching frames or model payloads."""
    safe = dict(metadata or {})
    safe["duration_ms"] = round(max(0.0, duration_seconds) * 1000, 2)
    if _langfuse_client is None:
        return
    try:
        with _langfuse_client.start_as_current_observation(
            as_type="span",
            name=f"runtime_{stage}",
            metadata=safe,
        ) as span:
            span.update(output={"status": "complete"})
    except Exception as exc:
        logger.debug("Could not record runtime-stage span: %s", exc)


def record_trading_cycle(metadata: dict[str, Any]) -> None:
    """Emit one bounded parent trace for a material settled-candle cycle.

    Raw candles, prompts, and model payloads are intentionally excluded. LangFuse is
    observability-only: every failure is swallowed after a debug log.
    """
    allowed = {
        "market",
        "provider",
        "provider_environment",
        "cycle_id",
        "execution_mode",
        "trade_horizon",
        "changed_timeframes",
        "symbols_considered",
        "symbols_scanned",
        "raw_strategy_signals",
        "technical_setups",
        "technical_candidates",
        "ml_candidates",
        "ml_evaluated",
        "ml_cache_hits",
        "ml_cache_misses",
        "all_llm_calls",
        "candidate_qwen_calls",
        "market_context_calls",
        "support_agent_calls",
        "qwen_reviewed",
        "qwen_cache_hits",
        "signals_validated",
        "risk_approved",
        "orders_created",
        "cycle_duration_ms",
    }
    safe = {key: value for key, value in metadata.items() if key in allowed}
    if _langfuse_client is None:
        return
    try:
        with _langfuse_client.start_as_current_observation(
            as_type="span", name="trading_cycle", metadata=safe
        ) as trace:
            trace.update(output={"status": "complete"})
    except Exception as exc:
        logger.debug("Could not record trading-cycle trace: %s", exc)


def tag_trace(
    trade_id: str | None = None,
    decision: str | None = None,
    signal_id: str | None = None,
) -> None:
    """Tag the current trace with trading-specific identifiers."""
    if trade_id:
        add_trace_metadata("trade_id", trade_id)
    if decision:
        add_trace_metadata("decision", decision)
    if signal_id:
        add_trace_metadata("signal_id", signal_id)


class TracingCallback:
    """In-memory callback used by the local dashboard and tests."""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.events = []

    def on_agent_start(self, agent_name: str, input_data: dict[str, Any]) -> None:
        """Record an agent start event."""
        self.events.append(
            {
                "type": "agent_start",
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "workflow_id": self.workflow_id,
            }
        )

    def on_agent_end(
        self,
        agent_name: str,
        output_data: dict[str, Any],
        latency_ms: int,
    ) -> None:
        """Record an agent completion event."""
        self.events.append(
            {
                "type": "agent_end",
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "workflow_id": self.workflow_id,
            }
        )

    def on_decision(
        self,
        agent_name: str,
        decision: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Record a decision and attach its key fields to the active trace."""
        self.events.append(
            {
                "type": "decision",
                "agent": agent_name,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning[:200],
                "timestamp": datetime.now().isoformat(),
                "workflow_id": self.workflow_id,
            }
        )
        add_trace_metadata(f"{agent_name}_decision", decision)
        add_trace_metadata(f"{agent_name}_confidence", confidence)

    def on_error(self, agent_name: str, error: str) -> None:
        """Record an agent error event."""
        self.events.append(
            {
                "type": "error",
                "agent": agent_name,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "workflow_id": self.workflow_id,
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of recorded local events."""
        return {
            "workflow_id": self.workflow_id,
            "event_count": len(self.events),
            "events": self.events,
            "agents_run": list(set(e["agent"] for e in self.events if "agent" in e)),
        }


def create_tracing_config(
    workflow_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create LangGraph invocation configuration with an optional Langfuse callback."""
    config: dict[str, Any] = {
        "configurable": {"thread_id": workflow_id},
        "metadata": {
            "workflow_id": workflow_id,
            "workflow_type": "trading_cycle",
            "started_at": datetime.now().isoformat(),
        },
        "callbacks": [],
    }
    if metadata:
        config["metadata"].update(metadata)

    callback = get_langfuse_callback()
    if callback is not None:
        config["callbacks"].append(callback)
    return config
