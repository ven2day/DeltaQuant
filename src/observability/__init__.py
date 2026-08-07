"""Observability module for optional Langfuse tracing and monitoring."""

from .tracing import setup_tracing, trace_agent

__all__ = ["setup_tracing", "trace_agent"]
