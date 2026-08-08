import json
from unittest.mock import patch

import pytest

from src.dashboard.stats import TradingStats
from src.webui.auth import hash_password
from src.webui.schema import stats_to_dict
from src.webui.server import ConnectionHub, WebUIServer, create_app

TEST_USERNAME = "admin"
TEST_PASSWORD = "correct-horse-battery-staple"
TEST_PASSWORD_HASH = hash_password(TEST_PASSWORD)
TEST_SESSION_SECRET = "test-session-secret"


async def _noop_health(full: bool) -> dict:
    """Stand-in get_health for tests that don't exercise the health route itself."""
    return {"status": "healthy", "services": [], "full": full}


def _make_app(**overrides):
    from fastapi.testclient import TestClient

    defaults = dict(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=lambda full: _noop_health(full),
        get_candles=lambda symbol, timeframe, limit, preview_simulated: [],
        cors_origins=["http://localhost:3000"],
        username=TEST_USERNAME,
        password_hash=TEST_PASSWORD_HASH,
        session_secret=TEST_SESSION_SECRET,
    )
    defaults.update(overrides)
    app = create_app(**defaults)
    return TestClient(app)


def _logged_in_client(**overrides):
    """A TestClient that has already completed a successful login (cookie carries
    across requests automatically via TestClient's underlying session)."""
    client = _make_app(**overrides)
    response = client.post(
        "/api/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200, response.text
    return client


# --- schema.stats_to_dict tests ---


@pytest.fixture
def stats():
    s = TradingStats()
    s.log_activity("hello", "INFO")
    s.current_signal = {"signal_type": "BUY", "symbol": "RELIANCE", "timeframe": "15m"}
    s.market_quotes = {"RELIANCE": {"last_price": 1275.0, "change_percent": -1.2}}
    return s


def test_stats_to_dict_is_json_serializable(stats):
    data = stats_to_dict(stats)
    # Must not raise — this is the whole point of the helper.
    dumped = json.dumps(data)
    assert isinstance(dumped, str)


def test_stats_to_dict_converts_session_start_to_isoformat(stats):
    data = stats_to_dict(stats)
    assert isinstance(data["session_start"], str)
    assert "T" in data["session_start"]  # isoformat marker


def test_stats_to_dict_includes_computed_properties(stats):
    stats.realized_pnl = 100.0
    stats.unrealized_pnl = 50.0
    stats.total_trades = 4
    stats.winning_trades = 3
    data = stats_to_dict(stats)
    assert data["total_pnl"] == 150.0
    assert data["win_rate"] == 75.0
    assert data["pnl_percent"] == pytest.approx(stats.pnl_percent)


def test_stats_to_dict_preserves_nested_fields(stats):
    data = stats_to_dict(stats)
    assert data["current_signal"]["symbol"] == "RELIANCE"
    assert data["market_quotes"]["RELIANCE"]["last_price"] == 1275.0
    assert data["activity_log"][-1]["message"] == "hello"


# --- ConnectionHub tests ---


class _FakeWebSocket:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("connection closed")
        self.sent.append(payload)


async def test_broadcast_with_no_connections_is_noop():
    hub = ConnectionHub()
    # Must not raise, and there's nothing to assert on except that it returns cleanly.
    await hub.broadcast({"type": "state", "data": {}})


async def test_broadcast_sends_to_all_connected_clients():
    hub = ConnectionHub()
    ws1, ws2 = _FakeWebSocket(), _FakeWebSocket()
    hub.connect(ws1)
    hub.connect(ws2)

    payload = {"type": "state", "data": {"x": 1}}
    await hub.broadcast(payload)

    assert ws1.sent == [payload]
    assert ws2.sent == [payload]


async def test_broadcast_disconnects_clients_that_raise():
    hub = ConnectionHub()
    good, bad = _FakeWebSocket(), _FakeWebSocket(fail=True)
    hub.connect(good)
    hub.connect(bad)

    await hub.broadcast({"type": "state", "data": {}})

    # The failing client should have been dropped; the good one stays and got the message.
    assert good.sent
    await hub.broadcast({"type": "state", "data": {}})
    assert len(good.sent) == 2  # still connected, received both broadcasts


# --- Auth: /api/login, /api/logout, /api/session ---


def test_login_with_correct_credentials_sets_cookie_and_succeeds():
    client = _make_app()
    response = client.post(
        "/api/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "dq_session" in response.cookies


def test_login_with_wrong_password_rejected():
    client = _make_app()
    response = client.post(
        "/api/login", json={"username": TEST_USERNAME, "password": "wrong"}
    )
    assert response.status_code == 401
    assert "dq_session" not in response.cookies


def test_login_with_wrong_username_rejected():
    client = _make_app()
    response = client.post(
        "/api/login", json={"username": "nobody", "password": TEST_PASSWORD}
    )
    assert response.status_code == 401


def test_login_lockout_after_max_attempts():
    client = _make_app(login_max_attempts=3, login_lockout_minutes=15)
    for _ in range(3):
        response = client.post(
            "/api/login", json={"username": TEST_USERNAME, "password": "wrong"}
        )
        assert response.status_code == 401

    # Correct credentials are now also rejected — the lockout is by IP, not by
    # correctness, exactly what stops online guessing regardless of luck.
    response = client.post(
        "/api/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == 429


def test_logout_clears_session():
    client = _logged_in_client()
    assert client.get("/api/session").json()["authenticated"] is True

    response = client.post("/api/logout")
    assert response.status_code == 200

    assert client.get("/api/session").json()["authenticated"] is False


def test_session_status_reflects_auth_state():
    client = _make_app()
    assert client.get("/api/session").json() == {"authenticated": False}

    client.post("/api/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD})
    status = client.get("/api/session").json()
    assert status["authenticated"] is True
    assert status["username"] == TEST_USERNAME


# --- Auth: protected REST routes ---


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/state"),
        ("get", "/api/signals"),
        ("get", "/api/trades"),
        ("get", "/api/health"),
    ],
)
def test_protected_routes_reject_unauthenticated_requests(method, path):
    client = _make_app()
    response = getattr(client, method)(path)
    assert response.status_code == 401


def test_protected_route_rejects_garbage_cookie():
    client = _make_app()
    client.cookies.set("dq_session", "not-a-valid-token")
    response = client.get("/api/state")
    assert response.status_code == 401


def test_protected_route_rejects_token_signed_with_wrong_secret():
    from src.webui.auth import create_session_token

    client = _make_app()
    forged = create_session_token(TEST_USERNAME, secret="wrong-secret", ttl_minutes=30)
    client.cookies.set("dq_session", forged)
    response = client.get("/api/state")
    assert response.status_code == 401


def test_state_route_accessible_after_login():
    client = _logged_in_client(get_snapshot=lambda: {"type": "state", "data": {"x": 1}})
    response = client.get("/api/state")
    assert response.status_code == 200
    assert response.json() == {"type": "state", "data": {"x": 1}}


# --- /api/signals route ---


def test_get_signals_route_passes_days_query_param_through():
    captured: dict[str, int] = {}

    def get_signals(days: int) -> list[dict]:
        captured["days"] = days
        return [{"symbol": "TCS", "side": "BUY", "status": "approved"}]

    client = _logged_in_client(get_signals=get_signals)

    response = client.get("/api/signals?days=3")

    assert response.status_code == 200
    assert response.json() == [{"symbol": "TCS", "side": "BUY", "status": "approved"}]
    assert captured["days"] == 3


def test_get_signals_route_defaults_to_seven_days():
    captured: dict[str, int] = {}

    def get_signals(days: int) -> list[dict]:
        captured["days"] = days
        return []

    client = _logged_in_client(get_signals=get_signals)

    client.get("/api/signals")

    assert captured["days"] == 7


# --- /api/health route ---


def test_get_health_route_defaults_to_fast_checks_only():
    captured: dict[str, bool] = {}

    async def get_health(full: bool) -> dict:
        captured["full"] = full
        return {"status": "healthy", "services": []}

    client = _logged_in_client(get_health=get_health)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "services": []}
    assert captured["full"] is False


def test_get_health_route_passes_full_query_param_through():
    captured: dict[str, bool] = {}

    async def get_health(full: bool) -> dict:
        captured["full"] = full
        return {"status": "healthy", "services": []}

    client = _logged_in_client(get_health=get_health)

    client.get("/api/health?full=true")

    assert captured["full"] is True


# --- /api/candles route ---


def test_candles_route_requires_authentication():
    client = _make_app()
    response = client.get("/api/candles?symbol=RELIANCE")
    assert response.status_code == 401


def test_candles_route_passes_symbol_timeframe_limit_through():
    captured: dict[str, object] = {}

    def get_candles(symbol: str, timeframe: str, limit: int, preview_simulated: bool) -> list[dict]:
        captured["symbol"] = symbol
        captured["timeframe"] = timeframe
        captured["limit"] = limit
        captured["preview_simulated"] = preview_simulated
        return [{"time": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}]

    client = _logged_in_client(get_candles=get_candles)

    response = client.get("/api/candles?symbol=RELIANCE&timeframe=1h&limit=100")

    assert response.status_code == 200
    assert captured == {
        "symbol": "RELIANCE",
        "timeframe": "1h",
        "limit": 100,
        "preview_simulated": False,
    }
    assert response.json()[0]["close"] == 1.0


def test_candles_route_defaults_timeframe_and_limit():
    captured: dict[str, object] = {}

    def get_candles(symbol: str, timeframe: str, limit: int, preview_simulated: bool) -> list[dict]:
        captured["timeframe"] = timeframe
        captured["limit"] = limit
        return []

    client = _logged_in_client(get_candles=get_candles)

    client.get("/api/candles?symbol=TCS")

    assert captured == {"timeframe": "15m", "limit": 500}


def test_candles_route_empty_result_is_valid_json_array():
    client = _logged_in_client(get_candles=lambda symbol, timeframe, limit, preview_simulated: [])
    response = client.get("/api/candles?symbol=UNKNOWN")
    assert response.status_code == 200
    assert response.json() == []


def test_candles_route_preview_simulated_query_param_passed_through():
    captured: dict[str, object] = {}

    def get_candles(symbol: str, timeframe: str, limit: int, preview_simulated: bool) -> list[dict]:
        captured["preview_simulated"] = preview_simulated
        return []

    client = _logged_in_client(get_candles=get_candles)

    client.get("/api/candles?symbol=RELIANCE&preview_simulated=true")

    assert captured["preview_simulated"] is True


# --- /api/universe route ---


def test_universe_route_requires_authentication():
    client = _make_app()
    response = client.get("/api/universe")
    assert response.status_code == 401


def test_universe_route_returns_nse_symbols_sorted_and_deduped():
    client = _logged_in_client()

    response = client.get("/api/universe")

    assert response.status_code == 200
    symbols = response.json()
    assert symbols == sorted(set(symbols))
    assert "RELIANCE" in symbols
    assert "TCS" in symbols


# --- WebSocket auth ---


def test_websocket_rejected_without_session_cookie():
    from starlette.websockets import WebSocketDisconnect

    client = _make_app()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass


def test_websocket_accepted_with_valid_session_cookie():
    client = _logged_in_client(get_snapshot=lambda: {"type": "state", "data": {"ok": True}})
    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
        assert message == {"type": "state", "data": {"ok": True}}


# --- WebUIServer startup-failure isolation ---


async def test_serve_swallows_systemexit_from_port_conflict():
    """uvicorn calls sys.exit() when it can't bind its port (e.g. already in use).

    That must never escape as an unhandled SystemExit from a background task —
    it previously could resurface later when the task was awaited during
    trading-loop shutdown, killing the whole process.
    """
    server = WebUIServer(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=lambda full: _noop_health(full),
        get_candles=lambda symbol, timeframe, limit, preview_simulated: [],
        host="127.0.0.1",
        port=0,
        cors_origins=["http://localhost:3000"],
        username=TEST_USERNAME,
        password_hash=TEST_PASSWORD_HASH,
        session_secret=TEST_SESSION_SECRET,
    )
    with patch.object(server._server, "serve", side_effect=SystemExit(3)):
        await server.serve()  # must not raise


async def test_serve_swallows_unexpected_exception():
    server = WebUIServer(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=lambda full: _noop_health(full),
        get_candles=lambda symbol, timeframe, limit, preview_simulated: [],
        host="127.0.0.1",
        port=0,
        cors_origins=["http://localhost:3000"],
        username=TEST_USERNAME,
        password_hash=TEST_PASSWORD_HASH,
        session_secret=TEST_SESSION_SECRET,
    )
    with patch.object(server._server, "serve", side_effect=RuntimeError("boom")):
        await server.serve()  # must not raise
