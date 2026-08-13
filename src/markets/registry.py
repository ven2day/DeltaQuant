"""Authoritative ownership map for independent market workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.core.models import Market
from src.core.namespaces import MarketNamespace, normalize_market


@dataclass(frozen=True)
class MarketRuntimeSpec:
    market: Market
    provider: str
    module: str
    env_profile: str
    schema: str
    config_root: Path
    artifact_root: Path
    log_root: Path

    def namespace(self, runtime_id: str) -> MarketNamespace:
        return MarketNamespace(self.market, self.provider, runtime_id)


def _spec(market: Market, provider: str, profile: str) -> MarketRuntimeSpec:
    slug = market.value.lower()
    return MarketRuntimeSpec(
        market=market,
        provider=provider,
        module=f"src.markets.{slug}.runtime",
        env_profile=profile,
        schema=slug,
        config_root=Path("config") / slug,
        artifact_root=Path("artifacts") / slug,
        log_root=Path("logs") / slug,
    )


MARKET_SPECS: dict[Market, MarketRuntimeSpec] = {
    Market.NSE: _spec(Market.NSE, "DHAN", "env/.env.nse"),
    Market.FOREX: _spec(Market.FOREX, "OANDA", "env/.env.forex.practice"),
    Market.CRYPTO: _spec(Market.CRYPTO, "UNCONFIGURED", "env/.env.crypto"),
}


def market_runtime_spec(
    market: Market | str, *, forex_environment: str = "practice"
) -> MarketRuntimeSpec:
    selected = normalize_market(market)
    spec = MARKET_SPECS[selected]
    if selected is Market.FOREX and forex_environment.strip().lower() == "live":
        return replace(spec, env_profile="env/.env.forex.live")
    return spec
