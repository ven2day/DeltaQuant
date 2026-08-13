"""Future crypto precision/volatility risk boundary."""

from typing import Any, Protocol


class CryptoRiskAdapter(Protocol):
    def evaluate(self, candidate: dict[str, Any], portfolio: dict[str, Any]) -> Any: ...

