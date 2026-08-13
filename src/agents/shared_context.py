"""Shared market-wide context reused by SWING and SCALP graph invocations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from src.agents.market_regime import market_regime_node
from src.agents.news_analyst import NewsAnalyst
from src.agents.sentiment import MarketSentimentAgent
from src.agents.state import create_initial_state
from src.config import get_settings
from src.markets.nse.universe.sectors import get_stock_sector


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _quote_change(data: Any) -> float:
    if not isinstance(data, dict):
        return 0.0
    return _finite_float(data.get("change_percent", 0.0))


def _quote_price(data: Any) -> float:
    if not isinstance(data, dict):
        return 0.0
    return _finite_float(data.get("last_price", data.get("close", 0.0)))


@dataclass(frozen=True)
class SharedMarketContext:
    regime: str
    regime_confidence: float
    regime_reasoning: str
    market_sentiment: float
    market_breadth: float
    volatility_state: str
    volatility_score: float
    index_state: str
    sector_states: dict[str, str]
    major_market_event_risk: str
    major_news_summary: tuple[str, ...]
    news_sentiment: dict[str, Any]
    news_headlines: tuple[dict[str, Any], ...]
    market_mood: dict[str, Any]
    generated_at: str
    expires_at: str
    context_version: str
    errors: tuple[str, ...] = ()
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["major_news_summary"] = list(self.major_news_summary)
        data["news_headlines"] = [dict(item) for item in self.news_headlines]
        data["errors"] = list(self.errors)
        return data


class SharedMarketContextService:
    """Content-versioned, TTL-cached market context."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._context: SharedMarketContext | None = None
        self._expires_monotonic = 0.0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _market_summary(market_data: dict[str, Any]) -> dict[str, Any]:
        changes = [_quote_change(data) for data in market_data.values()]
        advances = sum(change > 0 for change in changes)
        declines = sum(change < 0 for change in changes)
        breadth = advances / max(1, advances + declines)
        avg_change = sum(changes) / max(1, len(changes))
        avg_abs_change = sum(abs(change) for change in changes) / max(1, len(changes))

        sector_changes: dict[str, list[float]] = {}
        for symbol, data in market_data.items():
            sector_changes.setdefault(get_stock_sector(symbol), []).append(_quote_change(data))
        sector_average = {
            sector: sum(values) / len(values) for sector, values in sector_changes.items() if values
        }
        sector_states = {
            sector: "bullish" if value > 0.25 else "bearish" if value < -0.25 else "neutral"
            for sector, value in sector_average.items()
        }
        volatility_state = (
            "high" if avg_abs_change >= 1.5 else "low" if avg_abs_change <= 0.5 else "normal"
        )
        index_state = (
            "bullish" if avg_change > 0.25 else "bearish" if avg_change < -0.25 else "neutral"
        )
        return {
            "breadth": breadth,
            "avg_change": avg_change,
            "avg_abs_change": avg_abs_change,
            "volatility_state": volatility_state,
            "index_state": index_state,
            "sector_states": sector_states,
        }

    @staticmethod
    def _representative_inputs(
        market_data: dict[str, Any], indicators: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        symbols = sorted(
            market_data,
            key=lambda symbol: abs(_quote_change(market_data[symbol])),
            reverse=True,
        )[:5]
        compact_market = {
            symbol: {
                "close": _quote_price(market_data[symbol]),
                "change_percent": _quote_change(market_data[symbol]),
            }
            for symbol in symbols
        }
        compact_indicators: dict[str, Any] = {}
        for symbol in symbols:
            value = indicators.get(symbol)
            if isinstance(value, dict) and value:
                first = next(iter(value.values()))
                value = first.to_dict() if hasattr(first, "to_dict") else first
            elif value is not None and hasattr(value, "to_dict"):
                value = value.to_dict()
            if isinstance(value, dict):
                compact_indicators[symbol] = value
        return compact_market, compact_indicators

    @staticmethod
    def _version(
        summary: dict[str, Any], headline_count: int, sentiment: float, settings: Any
    ) -> str:
        breadth_step = settings.market_context_breadth_invalidation_step
        volatility_step = settings.market_context_volatility_invalidation_step
        sentiment_step = settings.market_context_sentiment_invalidation_step
        # sector_states is a per-sector bullish/bearish/neutral label (~20 sectors,
        # each flipping on a fixed +-0.25% average-change threshold in
        # _market_summary). Fingerprinting it raw meant ANY single sector
        # crossing that line invalidated the whole cache -- with ~20 independent
        # coin-flips per cycle, at least one flips almost every cycle regardless
        # of how coarse breadth_step/volatility_step are, which is why widening
        # those alone didn't reduce call frequency. Collapse to a single bucketed
        # net-breadth-across-sectors number instead (same breadth_step, same
        # "material move" semantics as the symbol-level breadth_bucket below) so
        # one sector flipping doesn't matter, only a broad shift across many.
        sector_states = summary["sector_states"]
        sector_total = max(1, len(sector_states))
        sector_bullish = sum(1 for state in sector_states.values() if state == "bullish")
        sector_bearish = sum(1 for state in sector_states.values() if state == "bearish")
        sector_breadth_bucket = round((sector_bullish - sector_bearish) / sector_total / breadth_step)
        payload = {
            "breadth_bucket": round(summary["breadth"] / breadth_step),
            "avg_change_bucket": round(summary["avg_change"] / volatility_step),
            "volatility_bucket": round(summary["avg_abs_change"] / volatility_step),
            "sector_breadth_bucket": sector_breadth_bucket,
            # index_state is redundant with avg_change_bucket above (both derive
            # from the same avg_change at a similar threshold) -- dropped rather
            # than kept as a second raw, unbucketed trigger on the same signal.
            # Bucketed sentiment + headline count, not raw titles: headline text
            # reorders/refreshes almost every cycle even when nothing material
            # changed, which was invalidating this cache (and forcing a fresh
            # market_regime Qwen call) on nearly every cycle regardless of TTL.
            "sentiment_bucket": round(sentiment / sentiment_step),
            "headline_count_bucket": min(headline_count, 10) // 3,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    async def get_or_create(
        self,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
        memory_lessons: list[dict[str, Any]] | None = None,
        *,
        market: str = "NSE",
        cycle_id: str = "",
    ) -> SharedMarketContext:
        settings = get_settings()
        analyst = NewsAnalyst()
        news = await asyncio.to_thread(
            lambda: asyncio.run(
                analyst.get_market_sentiment(market=market, cycle_id=cycle_id)
            )
        )
        headlines = [
            {
                "title": item.title,
                "sentiment": "positive" if item.sentiment_score > 0 else "negative",
            }
            for item in news.items[:10]
        ]
        summary = self._market_summary(market_data)
        version = self._version(summary, len(headlines), news.avg_sentiment, settings)

        with self._lock:
            cached = self._context
            # TTL-only: a single cached slot can't distinguish "the market
            # moved to a new state" from "the market moved back to a state it
            # was in two cycles ago" -- oscillating between a small set of
            # states (observed in practice, especially under simulated data)
            # made the version-match requirement invalidate almost every
            # cycle even though the state space itself was small and stable.
            # Trust the TTL alone; version is still recorded on the context
            # (context_version) for observability, just not part of the
            # reuse decision anymore.
            if cached is not None and time.monotonic() < self._expires_monotonic:
                self.hits += 1
                return SharedMarketContext(**{**asdict(cached), "cache_hit": True})
            self.misses += 1

        mood = MarketSentimentAgent().analyze(
            news_sentiment=news.avg_sentiment,
            market_data=market_data,
            volatility=summary["avg_abs_change"],
        )
        compact_market, compact_indicators = self._representative_inputs(market_data, indicators)
        regime_state = create_initial_state(
            workflow_id=cycle_id or None,
            trade_horizon="SHARED",
            market=market,
            market_data_provider="DHAN" if market.upper() == "NSE" else "OANDA",
        )
        regime_state["market_data"] = compact_market
        regime_state["indicators"] = compact_indicators
        regime_state["memory_lessons"] = list(memory_lessons or [])
        regime_state["news_sentiment"] = {"avg_sentiment": news.avg_sentiment}
        regime_state["news_headlines"] = headlines
        regime_state["market_mood"] = mood.to_dict()
        regime_result = await asyncio.to_thread(market_regime_node, regime_state)

        generated = datetime.now(UTC)
        ttl = settings.market_context_ttl_seconds
        event_risk = "HIGH" if abs(news.avg_sentiment) >= 0.75 else "NONE"
        context = SharedMarketContext(
            regime=str(regime_result.get("regime", "unknown")),
            regime_confidence=float(regime_result.get("regime_confidence", 0.0) or 0.0),
            regime_reasoning=str(regime_result.get("regime_reasoning", "")),
            market_sentiment=float(news.avg_sentiment),
            market_breadth=float(summary["breadth"]),
            volatility_state=str(summary["volatility_state"]),
            volatility_score=float(summary["avg_abs_change"]),
            index_state=str(summary["index_state"]),
            sector_states=dict(summary["sector_states"]),
            major_market_event_risk=event_risk,
            major_news_summary=tuple(item["title"] for item in headlines[:5]),
            news_sentiment={"avg_sentiment": float(news.avg_sentiment)},
            news_headlines=tuple(headlines[:5]),
            market_mood=mood.to_dict(),
            generated_at=generated.isoformat(),
            expires_at=(generated + timedelta(seconds=ttl)).isoformat(),
            context_version=version,
            errors=tuple(str(error) for error in regime_result.get("errors", [])),
        )
        with self._lock:
            self._context = context
            self._expires_monotonic = time.monotonic() + ttl
        return context

    def clear(self) -> None:
        with self._lock:
            self._context = None
            self._expires_monotonic = 0.0
            self.hits = 0
            self.misses = 0


_shared_market_context_service = SharedMarketContextService()


def get_shared_market_context_service() -> SharedMarketContextService:
    return _shared_market_context_service


def reset_shared_market_context_service() -> None:
    _shared_market_context_service.clear()
