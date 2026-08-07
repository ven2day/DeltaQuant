import json
from unittest.mock import patch

import pytest

from src.dashboard.stats import TradingStats
from src.webui.schema import stats_to_dict
from src.webui.server import ConnectionHub, WebUIServer, create_app


async def _noop_health(full: bool) -> dict:
    """Stand-in get_health for tests that don't exercise the health route itself."""
    return {"status": "healthy", "services": [], "full": full}


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


# --- /api/signals route ---


def test_get_signals_route_passes_days_query_param_through():
    from fastapi.testclient import TestClient

    captured: dict[str, int] = {}

    def get_signals(days: int) -> list[dict]:
        captured["days"] = days
        return [{"symbol": "TCS", "side": "BUY", "status": "approved"}]

    app = create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=get_signals,
        get_health=lambda full: _noop_health(full),
        cors_origins=["http://localhost:3000"],
    )
    client = TestClient(app)

    response = client.get("/api/signals?days=3")

    assert response.status_code == 200
    assert response.json() == [{"symbol": "TCS", "side": "BUY", "status": "approved"}]
    assert captured["days"] == 3


def test_get_signals_route_defaults_to_seven_days():
    from fastapi.testclient import TestClient

    captured: dict[str, int] = {}

    def get_signals(days: int) -> list[dict]:
        captured["days"] = days
        return []

    app = create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=get_signals,
        get_health=lambda full: _noop_health(full),
        cors_origins=["http://localhost:3000"],
    )
    client = TestClient(app)

    client.get("/api/signals")

    assert captured["days"] == 7


# --- /api/health route ---


def test_get_health_route_defaults_to_fast_checks_only():
    from fastapi.testclient import TestClient

    captured: dict[str, bool] = {}

    async def get_health(full: bool) -> dict:
        captured["full"] = full
        return {"status": "healthy", "services": []}

    app = create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=get_health,
        cors_origins=["http://localhost:3000"],
    )
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "services": []}
    assert captured["full"] is False


def test_get_health_route_passes_full_query_param_through():
    from fastapi.testclient import TestClient

    captured: dict[str, bool] = {}

    async def get_health(full: bool) -> dict:
        captured["full"] = full
        return {"status": "healthy", "services": []}

    app = create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=get_health,
        cors_origins=["http://localhost:3000"],
    )
    client = TestClient(app)

    client.get("/api/health?full=true")

    assert captured["full"] is True


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
        host="127.0.0.1",
        port=0,
        cors_origins=["http://localhost:3000"],
    )
    with patch.object(server._server, "serve", side_effect=SystemExit(3)):
        await server.serve()  # must not raise


async def test_serve_swallows_unexpected_exception():
    server = WebUIServer(
        hub=ConnectionHub(),
        get_snapshot=lambda: {"type": "state", "data": {}},
        get_signals=lambda days: [],
        get_health=lambda full: _noop_health(full),
        host="127.0.0.1",
        port=0,
        cors_origins=["http://localhost:3000"],
    )
    with patch.object(server._server, "serve", side_effect=RuntimeError("boom")):
        await server.serve()  # must not raise
