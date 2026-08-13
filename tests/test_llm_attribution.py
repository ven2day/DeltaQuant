"""Architectural guarantees for market-aware LLM accounting."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from src.observability import tracing

ROOT = Path(__file__).resolve().parents[1]


def test_every_production_record_hook_has_market_and_reason() -> None:
    missing: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.as_posix().endswith("src/finops/cost_tracker.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.id if isinstance(node.func, ast.Name) else ""
            if called != "record_llm_response":
                continue
            keywords = {item.arg for item in node.keywords}
            absent = {"market", "call_reason", "component"} - keywords
            if absent:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{sorted(absent)}")
    assert missing == []


class _Span:
    trace_id = "trace-123"

    def update(self, **kwargs: Any) -> None:
        self.updated = kwargs


class _Observation:
    def __init__(self, span: _Span) -> None:
        self.span = span

    def __enter__(self) -> _Span:
        return self.span

    def __exit__(self, *_args: Any) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def get_current_trace_id(self) -> str:
        return "trace-123"

    def start_as_current_observation(self, **kwargs: Any) -> _Observation:
        self.metadata = dict(kwargs["metadata"])
        return _Observation(_Span())


def test_langfuse_llm_span_contains_market_and_call_reason(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(tracing, "_langfuse_client", client)
    trace_id = tracing.record_llm_call_trace(
        {
            "market": "NSE",
            "call_reason": "REGIME_CONTEXT",
            "component": "market_regime",
            "runtime_id": "nse-runtime",
            "session_id": "session-1",
            "candidate_id": "",
            "prompt": "must not be retained",
        }
    )
    assert trace_id == "trace-123"
    assert client.metadata["market"] == "NSE"
    assert client.metadata["call_reason"] == "REGIME_CONTEXT"
    assert "prompt" not in client.metadata


def test_market_workspaces_render_only_their_scoped_finops() -> None:
    nse = (ROOT / "web/components/SessionCostPanel.tsx").read_text(encoding="utf-8")
    forex = (ROOT / "web/components/workspaces/ForexWorkspace.tsx").read_text(
        encoding="utf-8"
    )
    all_markets = (ROOT / "web/components/workspaces/AllMarketsWorkspace.tsx").read_text(
        encoding="utf-8"
    )
    assert "NSE LLM today" in nse
    assert "Forex LLM Today" in forex
    assert "LLM Today — All Markets" in all_markets

