"""FastAPI app + WebSocket broadcast hub for the live web dashboard.

Runs inside the same asyncio event loop as ``run_live_trading.py`` (as a
background task, mirroring the existing Dhan WebSocket ``listen_task`` pattern) —
not a separate process, no IPC.

Every route (REST and the ``/ws`` WebSocket) requires a valid session — see
``src/webui/auth.py``. There is no unauthenticated read path: the dashboard exposes
wallet balance, positions, and trading activity, and this app is designed to be
reachable over the network, not just loopback (M-8,
DeltaQuant-Quant-Risk-Review.md).
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.market.stock_discovery import MIDCAP_STOCKS, NIFTY50_STOCKS
from src.webui.auth import (
    LoginAttemptTracker,
    create_session_token,
    verify_password,
    verify_session_token,
)

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "dq_session"

# Symbols chartable from the Charts tab's symbol picker -- the same NIFTY50+midcap
# universe the discovery/signal loop scans (see stock_discovery.py), not limited to
# currently open positions. Static and small, so served straight from this list
# rather than threaded through as a callback like get_candles/get_snapshot.
TRADEABLE_UNIVERSE = sorted(set(NIFTY50_STOCKS) | set(MIDCAP_STOCKS))


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


class LoginRequest(BaseModel):
    username: str
    password: str


def create_app(
    hub: ConnectionHub,
    get_snapshot: Callable[[], dict[str, Any]],
    get_signals: Callable[[int], list[dict[str, Any]]],
    get_health: Callable[[bool], Awaitable[dict[str, Any]]],
    cors_origins: list[str],
    *,
    get_candles: Callable[[str, str, int, bool], list[dict[str, Any]]],
    username: str,
    password_hash: str,
    session_secret: str,
    session_ttl_minutes: int = 720,
    cookie_secure: bool = False,
    login_max_attempts: int = 5,
    login_lockout_minutes: int = 15,
) -> FastAPI:
    """Build the FastAPI app: snapshot, signal-history, health, and auth routes plus the WebSocket.

    ``password_hash`` is a ``hash_password()`` string (never a plaintext password —
    see ``scripts/set_dashboard_password.py``). ``session_secret`` signs session
    tokens; it must be long, random, and stable across restarts (a rotated secret
    invalidates every existing session, logging everyone out). ``cookie_secure``
    should be True whenever the app is served over HTTPS (it prevents the browser
    from ever sending the session cookie over plain HTTP) — see the caller in
    ``scripts/run_live_trading.py`` for how it's derived from settings.
    """
    app = FastAPI(title="₹DeltaQuant Web UI")
    attempts = LoginAttemptTracker(
        max_attempts=login_max_attempts, lockout_minutes=login_lockout_minutes
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _require_session(dq_session: str | None = Cookie(default=None)) -> str:
        """FastAPI dependency: 401s any request without a valid session cookie."""
        if dq_session is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        subject = verify_session_token(dq_session, secret=session_secret)
        if subject is None:
            raise HTTPException(status_code=401, detail="Session invalid or expired")
        return subject

    @app.post("/api/login")
    def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
        # Keyed on the direct TCP peer, not X-Forwarded-For: this port may be
        # reached directly (not only through a reverse proxy), and a client-
        # supplied header must never be trusted to key a security control — it
        # would let an attacker spoof a fresh IP on every request and bypass the
        # lockout entirely.
        client_ip = request.client.host if request.client else "unknown"
        if attempts.is_locked(client_ip):
            logger.warning("Login blocked (locked out): client_ip=%s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many failed login attempts. Try again in "
                    f"{login_lockout_minutes} minutes."
                ),
            )
        # Always run verify_password even when the username is already known to
        # be wrong, so a mismatched username doesn't short-circuit into a faster
        # response than a mismatched password — that timing gap is itself an
        # oracle for valid usernames.
        username_ok = payload.username == username
        password_ok = verify_password(payload.password, password_hash)
        if not (username_ok and password_ok):
            attempts.record_failure(client_ip)
            logger.warning(
                "Login failed: client_ip=%s username_ok=%s password_ok=%s",
                client_ip,
                username_ok,
                password_ok,
            )
            raise HTTPException(status_code=401, detail="Invalid username or password")

        logger.info("Login succeeded: client_ip=%s", client_ip)
        attempts.reset(client_ip)
        token = create_session_token(username, secret=session_secret, ttl_minutes=session_ttl_minutes)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
            max_age=session_ttl_minutes * 60,
            path="/",
        )
        return {"ok": True, "username": username}

    @app.post("/api/logout")
    def logout(response: Response) -> dict[str, Any]:
        response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/api/session")
    def session_status(dq_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        if dq_session is None:
            return {"authenticated": False}
        subject = verify_session_token(dq_session, secret=session_secret)
        return {"authenticated": subject is not None, "username": subject}

    @app.get("/api/state")
    def get_state(_subject: str = Depends(_require_session)) -> dict[str, Any]:
        return get_snapshot()

    @app.get("/api/signals")
    def get_signals_route(
        days: int = 7, _subject: str = Depends(_require_session)
    ) -> list[dict[str, Any]]:
        return get_signals(days)

    @app.get("/api/trades")
    def get_trades_route(
        limit: int = 250, _subject: str = Depends(_require_session)
    ) -> list[dict[str, Any]]:
        """Return the durable, net-of-charges paper trade ledger."""
        from src.execution.paper_engine import LocalPaperEngine

        return LocalPaperEngine().get_closed_trade_history(limit=limit)

    @app.get("/api/candles")
    def get_candles_route(
        symbol: str,
        timeframe: str = "15m",
        limit: int = 500,
        preview_simulated: bool = False,
        _subject: str = Depends(_require_session),
    ) -> list[dict[str, Any]]:
        """Historical OHLCV for the chart tab. See get_candles's implementation in
        run_live_trading.py (backed by HistoryManager) for the candle-shape contract.
        preview_simulated forces the simulated pipeline for this request regardless
        of open positions -- an explicit, opt-in preview, never the default."""
        return get_candles(symbol, timeframe, limit, preview_simulated)

    @app.get("/api/universe")
    def get_universe_route(_subject: str = Depends(_require_session)) -> list[str]:
        """NSE symbols the Charts tab's symbol picker can chart -- lets a user pull up
        any NIFTY50/midcap stock for analysis, not just symbols with an open position."""
        return TRADEABLE_UNIVERSE

    @app.get("/api/health")
    async def get_health_route(
        full: bool = False, _subject: str = Depends(_require_session)
    ) -> dict[str, Any]:
        return await get_health(full)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        # Auth happens BEFORE accept(): a browser's WebSocket handshake carries
        # cookies automatically (it's a normal HTTP request with Upgrade headers),
        # so the same session cookie set by /api/login is available here without
        # needing a token in the URL (which would otherwise leak into proxy/access
        # logs). Rejecting pre-accept sends a proper handshake failure instead of
        # accepting the socket and immediately closing it.
        token = websocket.cookies.get(SESSION_COOKIE_NAME)
        subject = verify_session_token(token, secret=session_secret) if token else None
        if subject is None:
            await websocket.close(code=1008)  # 1008 = policy violation
            return

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
        *,
        get_candles: Callable[[str, str, int, bool], list[dict[str, Any]]],
        username: str,
        password_hash: str,
        session_secret: str,
        session_ttl_minutes: int = 720,
        cookie_secure: bool = False,
        login_max_attempts: int = 5,
        login_lockout_minutes: int = 15,
    ) -> None:
        app = create_app(
            hub,
            get_snapshot,
            get_signals,
            get_health,
            cors_origins,
            get_candles=get_candles,
            username=username,
            password_hash=password_hash,
            session_secret=session_secret,
            session_ttl_minutes=session_ttl_minutes,
            cookie_secure=cookie_secure,
            login_max_attempts=login_max_attempts,
            login_lockout_minutes=login_lockout_minutes,
        )
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
