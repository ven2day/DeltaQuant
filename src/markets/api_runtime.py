"""Standalone aggregate read-only dashboard API with no broker credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv

from src.markets.api_registry import MarketApiRegistry
from src.markets.snapshots import MarketSnapshotStore
from src.webui.server import ConnectionHub, create_app


def _standalone_snapshot(
    store: MarketSnapshotStore, registry: MarketApiRegistry
) -> dict[str, Any]:
    """Return the legacy NSE dashboard state when its worker has published it.

    The separated API still serves the existing NSE workspace, whose WebSocket
    contract is ``type=state``.  Market summaries remain the safe fallback while
    the NSE worker is starting and continue to power the market-scoped REST APIs.
    """
    dashboard_state = store.read("NSE").get("dashboard_state")
    if isinstance(dashboard_state, dict) and dashboard_state:
        return {"type": "state", "data": dashboard_state}
    return {"type": "markets", "data": registry.summary()}


def create_standalone_app() -> Any:
    load_dotenv(Path("env/.env.common"), override=False)
    username = os.environ.get("WEB_UI_USERNAME", "admin")
    password_hash = os.environ.get("WEB_UI_PASSWORD_HASH", "")
    session_secret = os.environ.get("WEB_UI_SESSION_SECRET", "")
    if not password_hash or not session_secret:
        raise RuntimeError(
            "Standalone API requires WEB_UI_PASSWORD_HASH and WEB_UI_SESSION_SECRET in common env"
        )
    store = MarketSnapshotStore(
        os.environ.get("MARKET_SNAPSHOT_ROOT", "run"),
        stale_after_seconds=float(os.environ.get("MARKET_SNAPSHOT_STALE_SECONDS", "180")),
        writable=False,
    )
    registry = MarketApiRegistry()
    # The standalone API for this deployment intentionally exposes the two
    # implemented peer domains only.  A future market must opt in explicitly
    # instead of appearing merely because an enum value exists.
    for market in ("NSE", "FOREX"):
        registry.register(store.api_view(market))

    async def health(full: bool) -> dict[str, Any]:
        return {"status": "healthy", "markets": registry.summary(), "full": full}

    return create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: _standalone_snapshot(store, registry),
        get_signals=lambda days: [],
        get_health=health,
        get_candles=lambda symbol, timeframe, limit, preview: [],
        get_universe=lambda: [],
        cors_origins=[
            value.strip()
            for value in os.environ.get("WEB_UI_CORS_ORIGINS", "http://127.0.0.1:3000").split(",")
            if value.strip()
        ],
        username=username,
        password_hash=password_hash,
        session_secret=session_secret,
        market_api_registry=registry,
        websocket_poll_seconds=1.0,
    )


def main() -> None:
    uvicorn.run(
        create_standalone_app(),
        host=os.environ.get("WEB_UI_HOST", "127.0.0.1"),
        port=int(os.environ.get("WEB_UI_PORT", "8010")),
        log_level="warning",
    )
