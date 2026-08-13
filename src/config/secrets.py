"""Central secret redaction for diagnostics, exceptions, metrics, and APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

_SENSITIVE_KEY = re.compile(
    r"(?:access[_-]?token|api[_-]?key|api[_-]?secret|password|secret|authorization|account[_-]?id|(?<![a-z0-9])pin(?![a-z0-9])|totp)",
    re.IGNORECASE,
)


def redact_secrets(value: Any, *, key: str = "") -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "********" if value not in (None, "") else ""
    if isinstance(value, SecretStr):
        return "********" if value.get_secret_value() else ""
    if isinstance(value, Mapping):
        return {str(item_key): redact_secrets(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def safe_settings_dict(settings: Any) -> dict[str, Any]:
    raw = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings)
    redacted = redact_secrets(raw)
    return dict(redacted) if isinstance(redacted, Mapping) else {}
