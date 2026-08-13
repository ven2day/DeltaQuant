"""Market-scoped read models for the aggregate dashboard API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from src.config.secrets import redact_secrets
from src.core.models import Market
from src.core.namespaces import normalize_market


@dataclass(frozen=True)
class MarketApiView:
    market: Market | str
    status: Callable[[], dict[str, Any]]
    signals: Callable[[int], list[dict[str, Any]]]
    positions: Callable[[], list[dict[str, Any]]]
    candidates: Callable[[], list[dict[str, Any]]] = lambda: []
    strategies: Callable[[], list[dict[str, Any]]] = lambda: []
    models: Callable[[], list[dict[str, Any]]] = lambda: []

    @property
    def market_name(self) -> str:
        return normalize_market(self.market).value

    def scoped_status(self) -> dict[str, Any]:
        raw = dict(self.status())
        raw.pop("markets", None)
        raw["market"] = self.market_name
        result = redact_secrets(raw)
        return dict(result)

    def _scoped_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            row_market = str(row.get("market", self.market_name)).upper()
            if row_market != self.market_name:
                continue
            row["market"] = self.market_name
            redacted = redact_secrets(row)
            output.append(dict(redacted))
        return output

    def scoped_signals(self, days: int) -> list[dict[str, Any]]:
        return self._scoped_rows(self.signals(days))

    def scoped_positions(self) -> list[dict[str, Any]]:
        return self._scoped_rows(self.positions())

    def scoped_candidates(self) -> list[dict[str, Any]]:
        return self._scoped_rows(self.candidates())

    def scoped_strategies(self) -> list[dict[str, Any]]:
        return self._scoped_rows(self.strategies())

    def scoped_models(self) -> list[dict[str, Any]]:
        return self._scoped_rows(self.models())


class MarketApiRegistry:
    """Thread-safe registry; API aggregation never owns market processing."""

    def __init__(self) -> None:
        self._views: dict[Market, MarketApiView] = {}
        self._lock = RLock()

    def register(self, view: MarketApiView) -> None:
        market = normalize_market(view.market)
        with self._lock:
            self._views[market] = view

    def get(self, market: Market | str) -> MarketApiView | None:
        try:
            normalized = normalize_market(market)
        except ValueError:
            return None
        with self._lock:
            return self._views.get(normalized)

    def summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            views = dict(self._views)
        # Scope the aggregate workspace to the implemented peer domains.  Do
        # not make a dormant/future domain part of the public API implicitly.
        for market in (Market.NSE, Market.FOREX):
            view = views.get(market)
            if view is None:
                result[market.value] = {"market": market.value, "status": "DISABLED"}
                continue
            try:
                status = view.scoped_status()
                signals = view.scoped_signals(1)
                status["buy_count"] = sum(
                    1 for item in signals if str(item.get("side", item.get("signal_type", ""))).upper() == "BUY"
                )
                status["sell_count"] = sum(
                    1 for item in signals if str(item.get("side", item.get("signal_type", ""))).upper() == "SELL"
                )
                result[market.value] = status
            except Exception as exc:
                result[market.value] = {
                    "market": market.value,
                    "status": "UNHEALTHY",
                    "reason": type(exc).__name__,
                }
        return result

    def models_status(self) -> list[dict[str, Any]]:
        with self._lock:
            views = dict(self._views)
        rows: list[dict[str, Any]] = []
        for view in views.values():
            rows.extend(view.scoped_models())
        return rows

    def validation_status(self) -> list[dict[str, Any]]:
        with self._lock:
            views = dict(self._views)
        rows: list[dict[str, Any]] = []
        for view in views.values():
            rows.extend(view.scoped_strategies())
        return rows
