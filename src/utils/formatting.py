"""
Safe display-formatting helpers.

Many numeric fields in the pipeline are *optional* — indicator values are ``None`` during
the warm-up window (``_safe_float`` maps NaN/inf to ``None``), so a symbol can produce a
signal while, say, ADX is still ``None``. Formatting such a value with a numeric spec
(``f"{value:.1f}"``) raises ``TypeError: unsupported format string passed to
NoneType.__format__`` and crashes the trading cycle. ``fmt_optional`` renders a placeholder
instead of raising.
"""

from __future__ import annotations

from typing import Any


def fmt_optional(value: Any, spec: str = "", default: str = "N/A") -> str:
    """
    Format ``value`` with ``spec``, returning ``default`` when it is ``None`` (or not
    formattable with a numeric spec).

    >>> fmt_optional(42.1234, ".1f")
    '42.1'
    >>> fmt_optional(None, ".1f")
    'N/A'
    """
    if value is None:
        return default
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return default


def plain_english_fallback_cause(error_msg: str) -> str:
    """Translate a raw agent-fallback exception string into a one-line, human-readable
    reason an LLM node was skipped this cycle.

    Shared by every place that surfaces *why* a node degraded to its rule-based/
    heuristic fallback (signal_validation.py's rejection reasons, the dashboard's
    agent_fallback_notice) -- previously each showed the raw exception (or nothing at
    all), which read as an unexplained error even though the cause is fully knowable
    from the message text (a Groq daily quota, an open circuit breaker, etc.).
    """
    lower = error_msg.lower()
    if "circuit breaker" in lower or "unavailable" in lower:
        return "the AI reviewer is temporarily paused after repeated failures"
    if "rate_limit" in lower or "429" in lower or "quota" in lower:
        return "the AI reviewer hit its daily usage limit"
    if "disabled via settings" in lower:
        return "AI review is turned off in settings"
    return "the AI reviewer failed unexpectedly"
