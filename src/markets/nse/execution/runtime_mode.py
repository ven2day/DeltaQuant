"""Explicit paper-runtime environment and persistence identity.

This is deliberately separate from :class:`src.markets.nse.execution.service.ExecutionMode`,
which describes the broker routing mechanism (local paper, shadow, or live).  The
runtime mode below describes which market-data lineage may drive decisions and which
persistence namespace owns the resulting paper state.
"""

from __future__ import annotations

from enum import Enum


class RuntimeExecutionMode(str, Enum):
    """Closed set of environments supported by the permanent paper workflow."""

    MARKET_PAPER = "market_paper"
    MOCK = "mock"

    @property
    def namespace(self) -> str:
        """Canonical namespace used by every execution-specific repository."""
        if self is RuntimeExecutionMode.MOCK:
            return "mock_simulated"
        return "paper_market_data"

    @property
    def expected_quote_source(self) -> str:
        """Price lineage that may create an entry in this environment."""
        if self is RuntimeExecutionMode.MOCK:
            return "simulated"
        return "real"

    @property
    def may_bypass_market_hours(self) -> bool:
        return self is RuntimeExecutionMode.MOCK

    @classmethod
    def parse(cls, value: RuntimeExecutionMode | str) -> RuntimeExecutionMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"Unknown runtime execution mode {value!r}; expected {choices}"
            ) from exc


def assert_namespace_matches_mode(mode: RuntimeExecutionMode, namespace: str) -> None:
    """Fail immediately when a repository is wired to the wrong environment."""
    if namespace != mode.namespace:
        raise ValueError(
            f"Execution-mode namespace mismatch: {mode.value} requires "
            f"{mode.namespace!r}, got {namespace!r}"
        )
