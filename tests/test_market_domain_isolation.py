from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents.candidate_cache import build_candidate_fingerprint
from src.agents.model_artifacts import (
    ArtifactRejectedError,
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactRegistry,
)
from src.config.env_loader import resolve_env_files
from src.config.settings import Settings
from src.core.configuration import load_market_config
from src.core.indicators import IndicatorCache, Timeframe
from src.core.namespaces import MarketNamespace
from src.core.persistence import (
    MarketRepositoryBoundary,
    MarketRepositoryBoundaryError,
    SchemaBoundCandleRepository,
)
from src.core.risk import market_kill_switch_active
from src.core.utils.events import EventType, TradingEvent
from src.markets.api_registry import MarketApiRegistry, MarketApiView
from src.markets.api_runtime import _standalone_snapshot
from src.markets.forex.broker.oanda import OandaV20Client
from src.markets.nse.broker.dhan.historical import DhanHistoricalFeed
from src.markets.registry import market_runtime_spec
from src.markets.snapshots import MarketSnapshotStore
from src.webui.auth import hash_password
from src.webui.server import ConnectionHub, create_app


class _Model:
    pass


class _Store:
    timescale_enabled = False

    def upsert_frame(self, *args, **kwargs):
        self.last = (args, kwargs)
        return len(args[2])

    def load_frame(self, *args, **kwargs):
        self.last = (args, kwargs)
        return pd.DataFrame()


def _settings(market: str, **overrides) -> Settings:
    values = {
        "market": market,
        "groq_api_key": "test-common-key",
        "enable_web_ui": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100, 160),
            "high": range(101, 161),
            "low": range(99, 159),
            "close": range(100, 160),
            "volume": range(1000, 1060),
        },
        index=index,
    )


def _artifact_key(market: str) -> PredictionArtifactKey:
    return PredictionArtifactKey(
        strategy_version="ema_adx_trend:v1",
        timeframe="15m",
        trade_horizon="SCALP",
        regime="trending",
        model_version="model_v1",
        feature_version="features_v1",
        market=market,
    )


def _admit(registry: PredictionArtifactRegistry, key: PredictionArtifactKey) -> None:
    metadata = registry.metadata(
        key,
        oos_samples=100,
        folds_used=4,
        calibration_by_regime={"trending": 1.0},
    )
    registry.save(
        PredictionArtifact(
            metadata=metadata,
            scaler=_Model(),
            models={"model": _Model()},
            weights={"model": 1.0},
            calibrator=None,
        )
    )


def test_env_profiles_are_mutually_exclusive(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.mkdir()
    for name in (".env.common", ".env.nse", ".env.forex.practice", ".env.crypto"):
        (env / name).write_text("SAFE=true\n", encoding="utf-8")

    forex = resolve_env_files(tmp_path, {"MARKET": "FOREX", "FOREX_ENVIRONMENT": "practice"})
    nse = resolve_env_files(tmp_path, {"MARKET": "NSE"})
    crypto = resolve_env_files(tmp_path, {"MARKET": "CRYPTO"})

    assert forex == (str(env / ".env.common"), str(env / ".env.forex.practice"))
    assert nse == (str(env / ".env.common"), str(env / ".env.nse"))
    assert crypto == (str(env / ".env.common"), str(env / ".env.crypto"))


def test_legacy_root_env_is_never_loaded_by_market_workers(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DHAN_ACCESS_TOKEN=must-not-load\n", encoding="utf-8")
    env = tmp_path / "env"
    env.mkdir()
    (env / ".env.common").write_text("APP_ENV=test\n", encoding="utf-8")
    (env / ".env.nse").write_text("NSE_ENABLED=true\n", encoding="utf-8")
    (env / ".env.forex.practice").write_text("FOREX_ENABLED=true\n", encoding="utf-8")

    forex = resolve_env_files(tmp_path, {"MARKET": "FOREX"})
    nse = resolve_env_files(tmp_path, {"MARKET": "NSE"})

    assert str(tmp_path / ".env") not in forex
    assert str(tmp_path / ".env") not in nse
    assert forex == (str(env / ".env.common"), str(env / ".env.forex.practice"))
    assert nse == (str(env / ".env.common"), str(env / ".env.nse"))


def test_worker_settings_retain_only_their_broker_credentials() -> None:
    forex = _settings(
        "FOREX",
        forex_enabled=True,
        oanda_account_id="practice-account",
        oanda_access_token="oanda-secret",
        dhan_client_id="dhan-id",
        dhan_access_token="dhan-secret",
    )
    assert forex.oanda_access_token.get_secret_value() == "oanda-secret"
    assert forex.dhan_client_id is None
    assert forex.dhan_access_token is None

    nse = _settings(
        "NSE",
        dhan_client_id="dhan-id",
        dhan_access_token="dhan-secret",
        oanda_account_id="practice-account",
        oanda_access_token="oanda-secret",
    )
    assert nse.dhan_client_id == "dhan-id"
    assert nse.oanda_access_token.get_secret_value() == ""

    crypto = _settings("CRYPTO")
    assert crypto.dhan_client_id is None
    assert crypto.oanda_access_token.get_secret_value() == ""


def test_runtime_specs_are_peers_with_independent_modules() -> None:
    modules = {market: market_runtime_spec(market).module for market in ("NSE", "FOREX", "CRYPTO")}
    assert modules == {
        "NSE": "src.markets.nse.runtime",
        "FOREX": "src.markets.forex.runtime",
        "CRYPTO": "src.markets.crypto.runtime",
    }
    assert len(set(modules.values())) == 3


def test_market_configuration_bundles_have_independent_roots() -> None:
    bundles = {market: load_market_config(market) for market in ("NSE", "FOREX", "CRYPTO")}
    assert {bundle.root.name for bundle in bundles.values()} == {"nse", "forex", "crypto"}
    assert bundles["NSE"].strategies is not bundles["FOREX"].strategies
    assert bundles["FOREX"].sessions


def test_broker_implementations_are_owned_by_their_market_domains() -> None:
    assert DhanHistoricalFeed.__module__.startswith("src.markets.nse.")
    assert OandaV20Client.__module__.startswith("src.markets.forex.")


def test_market_repository_rejects_cross_market_access() -> None:
    nse = MarketRepositoryBoundary("NSE")
    assert nse.qualify("market_candles") == '"nse"."market_candles"'
    with pytest.raises(MarketRepositoryBoundaryError):
        nse.require_market("FOREX")

    delegate = _Store()
    repository = SchemaBoundCandleRepository(delegate, market="FOREX", provider="OANDA")
    with pytest.raises(MarketRepositoryBoundaryError):
        repository.upsert_frame(
            "EUR_USD", "15m", _frame(), source="oanda", market="NSE"
        )


def test_model_artifacts_are_physically_and_semantically_market_isolated(tmp_path: Path) -> None:
    registry = PredictionArtifactRegistry(tmp_path)
    _admit(registry, _artifact_key("NSE"))

    files = list((tmp_path / "nse").rglob("*.pkl"))
    assert len(files) == 1
    assert not (tmp_path / "forex").exists()
    with pytest.raises(ArtifactRejectedError):
        registry.load(_artifact_key("FOREX"))


def test_cache_and_event_namespaces_include_market() -> None:
    namespace = MarketNamespace("FOREX", "OANDA", "worker-1")
    assert namespace.cache_key("feature", "EUR_USD", "15m").startswith("forex:feature:")
    event = TradingEvent(EventType.CANDLE_SETTLED, {}, market="FOREX", provider="OANDA")
    assert event.topic == "forex.candle_settled"
    assert event.to_dict()["market"] == "FOREX"

    signal = {"symbol": "EUR_USD", "timeframe": "15m", "signal_type": "BUY"}
    nse_key = build_candidate_fingerprint(
        {**signal, "market": "NSE"}, regime="trending", market_context_version="v1"
    )
    forex_key = build_candidate_fingerprint(
        {**signal, "market": "FOREX"}, regime="trending", market_context_version="v1"
    )
    assert nse_key != forex_key


def test_indicator_cache_does_not_collide_across_markets() -> None:
    cache = IndicatorCache()
    frame = _frame()
    cache.get_or_compute(frame, "ABC", Timeframe.M15, market="NSE")
    cache.get_or_compute(frame, "ABC", Timeframe.M15, market="FOREX")
    assert cache.misses == 2


def test_market_and_global_kill_switches_are_independent() -> None:
    settings = SimpleNamespace(
        global_kill_switch=False,
        nse_kill_switch=True,
        forex_kill_switch=False,
        crypto_kill_switch=False,
    )
    assert market_kill_switch_active(settings, "NSE")
    assert not market_kill_switch_active(settings, "FOREX")
    settings.global_kill_switch = True
    assert market_kill_switch_active(settings, "FOREX")
    assert market_kill_switch_active(settings, "CRYPTO")


def test_market_scoped_api_filters_cross_market_rows_and_aggregates() -> None:
    registry = MarketApiRegistry()
    registry.register(
        MarketApiView(
            market="FOREX",
            status=lambda: {"status": "HEALTHY", "provider": "OANDA"},
            signals=lambda days: [
                {"market": "FOREX", "symbol": "EUR_USD", "side": "BUY"},
                {"market": "NSE", "symbol": "RELIANCE", "side": "SELL"},
            ],
            positions=lambda: [{"symbol": "GBP_USD", "side": "SELL"}],
            candidates=lambda: [
                {
                    "market": "FOREX",
                    "candidate_id": "fx-candidate",
                    "symbol": "USD_CAD",
                    "side": "BUY",
                }
            ],
        )
    )
    registry.register(
        MarketApiView(
            market="NSE",
            status=lambda: {"status": "HEALTHY", "provider": "DHAN"},
            signals=lambda days: [
                {"market": "NSE", "symbol": "RELIANCE", "side": "SELL"},
                {"market": "FOREX", "symbol": "EUR_USD", "side": "BUY"},
            ],
            positions=lambda: [],
        )
    )
    app = create_app(
        hub=ConnectionHub(),
        get_snapshot=lambda: {},
        get_signals=lambda days: [],
        get_health=lambda full: _health(),
        get_candles=lambda symbol, timeframe, limit, preview: [],
        cors_origins=["http://localhost:3000"],
        username="admin",
        password_hash=hash_password("password"),
        session_secret="test-session-secret",
        market_api_registry=registry,
    )
    client = TestClient(app)
    assert (
        client.post(
            "/api/login", json={"username": "admin", "password": "password"}
        ).status_code
        == 200
    )
    signals = client.get("/api/forex/signals").json()
    assert signals == [{"market": "FOREX", "symbol": "EUR_USD", "side": "BUY"}]
    assert client.get("/api/forex/candidates").json() == [
        {
            "market": "FOREX",
            "candidate_id": "fx-candidate",
            "symbol": "USD_CAD",
            "side": "BUY",
        }
    ]
    assert client.get("/api/nse/status").json()["provider"] == "DHAN"
    assert client.get("/api/nse/signals").json() == [
        {"market": "NSE", "symbol": "RELIANCE", "side": "SELL"}
    ]
    summary = client.get("/api/markets/summary").json()
    assert summary["FOREX"]["buy_count"] == 1
    assert summary["NSE"]["sell_count"] == 1


def test_market_snapshots_are_separate_and_secret_safe(tmp_path: Path) -> None:
    store = MarketSnapshotStore(tmp_path, min_write_seconds=0)
    assert store.publish(
        "FOREX",
        status={"status": "HEALTHY", "oanda_access_token": "never-persist"},
        signals=[{"symbol": "EUR_USD", "side": "BUY"}],
        positions=[],
        force=True,
    )
    assert store.publish(
        "NSE",
        status={"status": "HEALTHY", "dhan_access_token": "never-persist"},
        signals=[],
        positions=[],
        dashboard_state={
            "current_balance": 1_000_000.0,
            "dhan_access_token": "never-persist",
        },
        force=True,
    )

    forex = store.read("FOREX")
    nse = store.read("NSE")
    assert forex["market"] == "FOREX" and nse["market"] == "NSE"
    assert forex["status"]["oanda_access_token"] == "********"
    assert nse["status"]["dhan_access_token"] == "********"
    assert nse["dashboard_state"]["current_balance"] == 1_000_000.0
    assert nse["dashboard_state"]["dhan_access_token"] == "********"
    assert not (tmp_path / "crypto.json").exists()


def test_snapshot_publication_retries_transient_windows_permission_error(tmp_path: Path) -> None:
    store = MarketSnapshotStore(
        tmp_path,
        min_write_seconds=0,
        replace_attempts=3,
        retry_base_seconds=0,
        writer_runtime_id="forex-test",
    )
    real_replace = __import__("os").replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient lock")
        real_replace(source, target)

    with patch("src.markets.snapshots.os.replace", side_effect=flaky_replace):
        assert store.publish(
            "FOREX",
            status={"status": "HEALTHY"},
            signals=[],
            positions=[],
            force=True,
        )

    assert attempts == 3
    snapshot = store.read("FOREX")
    assert snapshot["writer_runtime_id"] == "forex-test"
    assert snapshot["health_state"] == "HEALTHY"


def test_permanent_snapshot_failure_is_degraded_and_non_fatal(tmp_path: Path) -> None:
    store = MarketSnapshotStore(
        tmp_path,
        min_write_seconds=0,
        replace_attempts=2,
        retry_base_seconds=0,
    )

    with patch("src.markets.snapshots.os.replace", side_effect=PermissionError("locked")):
        published = store.publish(
            "FOREX",
            status={"status": "HEALTHY"},
            signals=[],
            positions=[],
            force=True,
        )

    assert published is False
    assert store.metrics("FOREX") == {"write_failures": 1, "consecutive_failures": 1}
    assert store.read("FOREX")["status"]["status"] == "UNAVAILABLE"


def test_stale_snapshot_cannot_masquerade_as_current_green_health(tmp_path: Path) -> None:
    store = MarketSnapshotStore(tmp_path, min_write_seconds=0, stale_after_seconds=30)
    assert store.publish(
        "FOREX",
        status={"status": "HEALTHY", "pricing_stream": "HEALTHY"},
        signals=[],
        positions=[],
        force=True,
    )
    path = tmp_path / "forex" / "market_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generated_at"] = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = store.read("FOREX")

    assert snapshot["health_state"] == "STALE"
    assert snapshot["status"]["status"] == "STALE"
    assert snapshot["status"]["runtime_last_known_status"] == "HEALTHY"
    assert snapshot["status"]["dashboard_connectivity"] == "DISCONNECTED"
    assert snapshot["age_seconds"] >= 300


def test_api_snapshot_reader_is_read_only_and_does_not_own_publication(tmp_path: Path) -> None:
    reader = MarketSnapshotStore(tmp_path, writable=False)

    with pytest.raises(RuntimeError, match="Read-only"):
        reader.publish(
            "FOREX",
            status={"status": "HEALTHY"},
            signals=[],
            positions=[],
            force=True,
        )


def test_standalone_snapshot_serves_published_nse_dashboard_state(tmp_path: Path) -> None:
    store = MarketSnapshotStore(tmp_path, min_write_seconds=0)
    registry = MarketApiRegistry()
    registry.register(store.api_view("NSE"))
    assert _standalone_snapshot(store, registry)["type"] == "markets"

    store.publish(
        "NSE",
        status={"status": "HEALTHY"},
        signals=[],
        positions=[],
        dashboard_state={"current_balance": 1_000_000.0},
        force=True,
    )

    assert _standalone_snapshot(store, registry) == {
        "type": "state",
        "data": {"current_balance": 1_000_000.0},
    }


async def _health() -> dict[str, str]:
    return {"status": "healthy"}


def test_shared_strategy_module_has_no_broker_dependencies() -> None:
    source = Path("src/core/strategies/production.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any("dhan" in name.lower() or "oanda" in name.lower() for name in imports)


def test_actual_secret_env_files_are_gitignored() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    for name in (
        "env/.env.common",
        "env/.env.nse",
        "env/.env.forex.practice",
        "env/.env.forex.live",
        "env/.env.crypto",
    ):
        assert name in ignore
