from __future__ import annotations

import pytest

from scripts.benchmark_qwen_local_paper import _json_payload, _risk_must_reject


def test_qwen_soak_schema_accepts_strict_advisory_json() -> None:
    payload = _json_payload(
        '{"decision":"HOLD","confidence":0.61,"reason":"conflict","risk_flags":[]}'
    )
    assert payload["decision"] == "HOLD"


def test_qwen_soak_schema_fails_closed() -> None:
    with pytest.raises(ValueError):
        _json_payload('{"decision":"BUY","confidence":4,"reason":"bad","risk_flags":[]}')


def test_qwen_soak_cannot_bypass_deterministic_risk() -> None:
    reasons = _risk_must_reject(1)
    assert reasons
