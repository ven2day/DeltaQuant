"""FastAPI app + WebSocket broadcast hub for the live web dashboard.

Runs inside the same asyncio event loop as ``run_live_trading.py`` (as a
background task, mirroring the existing Dhan WebSocket ``listen_task`` pattern) —
not a separate process, no IPC.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class ConnectionHub:
    """Tracks connected WebSocket clients and broadcasts state to all of them."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    def connect(self, ws: WebSocket) -> None:
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send payload to every connected client; drop any that error out.

        Returns immediately with zero connections, so a session with the web UI
        enabled but no browser open pays no serialization/network cost per tick.
        """
        if not self._connections:
            return
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            logger.debug("Dropping unresponsive web UI client")
            self.disconnect(ws)


def create_app(
    hub: ConnectionHub,
    get_snapshot: Callable[[], dict[str, Any]],
    get_signals: Callable[[int], list[dict[str, Any]]],
    get_health: Callable[[bool], Awaitable[dict[str, Any]]],
    cors_origins: list[str],
) -> FastAPI:
    """Build the FastAPI app: snapshot, signal-history and health endpoints plus the WebSocket."""
    app = FastAPI(title="₹DeltaQuant Web UI")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return get_snapshot()

    @app.get("/api/signals")
    def get_signals_route(days: int = 7) -> list[dict[str, Any]]:
        return get_signals(days)

    @app.get("/api/trades")
    def get_trades_route(limit: int = 250) -> list[dict[str, Any]]:
        """Return the durable, net-of-charges paper trade ledger."""
        from src.execution.paper_engine import LocalPaperEngine

        return LocalPaperEngine().get_closed_trade_history(limit=limit)

    @app.get("/api/health")
    async def get_health_route(full: bool = False) -> dict[str, Any]:
        return await get_health(full)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        hub.connect(websocket)
        try:
            # Paint immediately so a client connecting mid-session isn't blank
            # until the next broadcast tick.
            await websocket.send_json(get_snapshot())
            while True:
                # Nothing is expected from the client — this just blocks until
                # it disconnects (any inbound text is ignored).
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(websocket)

    return app


class WebUIServer:
    """Runs a FastAPI app via uvicorn as a background asyncio task.

    Uses ``uvicorn.Server(...).serve()`` directly rather than ``uvicorn.run()``
    (which spins up its own event loop) so it can run concurrently with the
    existing trading loop in the same process/event loop.
    """

    def __init__(
        self,
        hub: ConnectionHub,
        get_snapshot: Callable[[], dict[str, Any]],
        get_signals: Callable[[int], list[dict[str, Any]]],
        get_health: Callable[[bool], Awaitable[dict[str, Any]]],
        host: str,
        port: int,
        cors_origins: list[str],
    ) -> None:
        app = create_app(hub, get_snapshot, get_signals, get_health, cors_origins)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._host = host
        self._port = port

    async def serve(self) -> None:
        """Run the server until stopped.

        Never lets a startup failure (e.g. the port already being in use)
        escape as an exception — uvicorn signals that with a raw
        ``sys.exit()``, which would otherwise surface as an unhandled
        ``SystemExit`` the next time this task is awaited (including during
        the trading loop's shutdown), and could take the whole process down
        with it. Swallow it here, loudly, and let the trading loop carry on
        without the web UI instead.
        """
        try:
            await self._server.serve()
        except SystemExit as e:
            logger.error(
                "Web UI failed to start on %s:%s (exit code %s) — is something else "
                "already listening on that port? The trading loop will continue without "
                "the web UI. Set WEB_UI_PORT in .env to use a different port.",
                self._host,
                self._port,
                e.code,
            )
        except Exception:
            logger.exception("Web UI server crashed unexpectedly")

    async def shutdown(self) -> None:
        self._server.should_exit = True
