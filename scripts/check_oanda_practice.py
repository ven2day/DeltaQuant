"""OANDA Practice integration check: market data through shared core, never orders."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.prediction import PredictionAgent
from src.config import get_settings
from src.core.candidates import SignalEngine
from src.core.candles import CandleStore
from src.markets.forex.eligibility import create_registry
from src.markets.forex.market_data import create_forex_market_data_provider
from src.markets.forex.ml import create_artifact_registry
from src.markets.forex.persistence import bind_candle_repository, bind_trading_repository
from src.markets.forex.risk import ForexRiskLimits
from src.markets.forex.runtime.cycle import ForexSettledCandleCycle
from src.markets.forex.strategies import build_strategy_config


async def check() -> dict[str, object]:
    settings = get_settings()
    if settings.market != "FOREX" or settings.oanda_environment != "practice":
        raise RuntimeError(
            "Run with MARKET=FOREX and FOREX_ENVIRONMENT=practice; live is not accepted"
        )
    if settings.forex_execution_enabled or settings.allow_live_orders:
        raise RuntimeError("Practice integration check requires every execution gate disabled")
    provider = create_forex_market_data_provider(settings)
    try:
        health = await provider.validate_connectivity()
        if not health.healthy:
            raise RuntimeError(health.message)
        instruments = await provider.list_instruments()
        symbols = [item.symbol for item in instruments]
        if not symbols:
            raise RuntimeError("No configured Forex instruments available")
        await provider.start_price_stream(symbols)
        stream_received = await provider.wait_for_stream_message(
            float(settings.oanda_stream_startup_timeout_seconds)
        )
        stream_state = provider.stream_health_state().value
        stream_error = "" if stream_received else (provider.client.stream_error or "STREAM_TIMEOUT")
        health = await provider.validate_connectivity()

        candle_store = None
        trading_repository = None
        if settings.market_history_store_enabled:
            raw_candle_store = CandleStore(
                settings.market_history_database_url or settings.database_url,
                enable_timescale=settings.market_history_enable_timescale,
                require_timescale=settings.market_history_require_timescale,
                schema=str(settings.forex_db_schema),
            )
            candle_store = bind_candle_repository(raw_candle_store)
            trading_repository = bind_trading_repository(raw_candle_store.engine)
        config = build_strategy_config(settings)
        cycle = ForexSettledCandleCycle(
            provider=provider,
            signal_engine=SignalEngine(production_config=config),
            strategy_config=config,
            eligibility_registry=create_registry(settings),
            risk_limits=ForexRiskLimits.from_settings(settings),
            candle_store=candle_store,
            trading_repository=trading_repository,
            prediction_agent=PredictionAgent(artifact_registry=create_artifact_registry(settings)),
            settings=settings,
        )
        result = await cycle.run()
        return {
            "market": "FOREX",
            "provider": "OANDA",
            "environment": "PRACTICE",
            "orders_sent": 0,
            "health": health.to_dict(),
            "instrument_count": len(instruments),
            "stream_healthy": stream_state == "HEALTHY",
            "stream_state": stream_state,
            "stream_error": stream_error,
            "cycle": result.metrics.to_dict(),
            "conflicts": list(result.conflicts),
        }
    finally:
        await provider.close()


if __name__ == "__main__":
    report = asyncio.run(check())
    output = ROOT / "run" / "forex" / "oanda_practice_cycle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
