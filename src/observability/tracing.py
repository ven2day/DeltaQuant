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
