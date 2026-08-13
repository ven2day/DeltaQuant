from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from src.core.ml import (
    MLPolicy,
    ModelGrain,
    ModelHotReloader,
    ModelRecord,
    ModelRegistry,
    ModelStatus,
    evaluate_ml_policy,
)
from src.markets.api_registry import MarketApiRegistry, MarketApiView
from src.markets.forex.config import load_forex_settings
from src.markets.forex.ml import ForexModelRegistry
from src.markets.nse.config import load_nse_settings
from src.markets.nse.ml import NSEModelRegistry


def _env_tree(root: Path) -> None:
    env = root / "env"
    env.mkdir()
    (env / ".env.common").write_text(
        "DATABASE_URL=postgresql://common\nTRADING_MODE=paper\n", encoding="utf-8"
    )
    (env / ".env.nse").write_text(
        "MARKET=NSE\nDHAN_CLIENT_ID=nse-client\nDHAN_ACCESS_TOKEN=nse-secret\n",
        encoding="utf-8",
    )
    (env / ".env.forex.practice").write_text(
        "MARKET=FOREX\nFOREX_ENVIRONMENT=practice\n"
        "OANDA_ACCOUNT_ID=practice-account\nOANDA_ACCESS_TOKEN=forex-secret\n",
        encoding="utf-8",
    )


def test_scoped_nse_settings_never_expose_oanda_credentials(tmp_path: Path) -> None:
    _env_tree(tmp_path)
    settings = load_nse_settings(base_dir=tmp_path, environ={"MARKET": "NSE"})
    assert settings.dhan_client_id.get_secret_value() == "nse-client"
    assert not hasattr(settings, "oanda_access_token")


def test_scoped_forex_settings_never_expose_dhan_credentials(tmp_path: Path) -> None:
    _env_tree(tmp_path)
    settings = load_forex_settings(
        base_dir=tmp_path,
        environ={"MARKET": "FOREX", "FOREX_ENVIRONMENT": "practice"},
    )
    assert settings.oanda_account_id.get_secret_value() == "practice-account"
    assert not hasattr(settings, "dhan_access_token")


def test_process_environment_overrides_only_active_profile(tmp_path: Path) -> None:
    _env_tree(tmp_path)
    settings = load_nse_settings(
        base_dir=tmp_path,
        environ={"MARKET": "NSE", "DHAN_CLIENT_ID": "process-client"},
    )
    assert settings.dhan_client_id.get_secret_value() == "process-client"


def test_core_has_no_broker_or_market_provider_imports() -> None:
    forbidden = ("dhan", "oanda", "src.markets.nse", "src.markets.forex")
    for path in Path("src/core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(item.name.lower() for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.lower())
        assert not any(token in module for token in forbidden for module in imported), path


def _record(grain: ModelGrain, version: str, status: ModelStatus) -> ModelRecord:
    return ModelRecord(
        grain=grain,
        model_version=version,
        status=status,
        artifact_location=f"artifacts/{grain.market.lower()}/{grain.strategy_name}/15m/{version}",
        artifact_checksum=f"checksum-{version}",
        trained_at=datetime.now(UTC),
        training_data_until=datetime(2026, 7, 31, tzinfo=UTC),
    )


def test_registry_uses_active_approved_not_latest_trained() -> None:
    registry = ModelRegistry(create_engine("sqlite:///:memory:"))
    grain = ModelGrain("NSE", "ema_adx_trend", "15m", "rules_v1")
    registry.register(_record(grain, "v18", ModelStatus.APPROVED))
    registry.register(_record(grain, "v19", ModelStatus.TRAINING))
    registry.promote(grain, "v18")
    assert registry.latest_active(grain).model_version == "v18"  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="APPROVED"):
        registry.promote(grain, "v19")
    assert registry.latest_active(grain).model_version == "v18"  # type: ignore[union-attr]


def test_atomic_promotion_retires_champion() -> None:
    registry = ModelRegistry(create_engine("sqlite:///:memory:"))
    grain = ModelGrain("FOREX", "trend_pullback", "30m", "rules_v1")
    registry.register(_record(grain, "v1", ModelStatus.APPROVED))
    registry.promote(grain, "v1")
    registry.register(_record(grain, "v2", ModelStatus.APPROVED))
    registry.promote(grain, "v2")
    assert registry.latest_active(grain).model_version == "v2"  # type: ignore[union-attr]
    statuses = {item.model_version: item.status for item in registry.list_records(market="FOREX")}
    assert statuses == {"v1": ModelStatus.RETIRED, "v2": ModelStatus.ACTIVE}


def test_failed_hot_reload_keeps_working_champion_in_memory() -> None:
    registry = ModelRegistry(create_engine("sqlite:///:memory:"))
    grain = ModelGrain("NSE", "ema_adx_trend", "15m", "rules_v1")
    registry.register(_record(grain, "v18", ModelStatus.APPROVED))
    registry.promote(grain, "v18")
    reloader = ModelHotReloader(registry, grain, lambda record: {"version": record.model_version})
    assert reloader.refresh() is True
    registry.register(_record(grain, "v20", ModelStatus.APPROVED))
    registry.promote(grain, "v20")
    reloader.loader = lambda record: (_ for _ in ()).throw(ValueError("bad artifact"))
    assert reloader.refresh() is False
    assert reloader.model == {"version": "v18"}
    assert reloader.active_record.model_version == "v18"  # type: ignore[union-attr]


def test_model_registry_fails_closed_on_cross_market_artifact_path() -> None:
    registry = ModelRegistry(create_engine("sqlite:///:memory:"))
    grain = ModelGrain("FOREX", "ema_adx_trend", "15m", "rules_v1")
    record = ModelRecord(
        grain=grain,
        model_version="v1",
        status=ModelStatus.APPROVED,
        artifact_location="artifacts/nse/ema_adx_trend/15m/v1",
        artifact_checksum="checksum",
    )
    with pytest.raises(ValueError, match="market namespace"):
        registry.register(record)


def test_market_bound_model_registries_reject_cross_market_access() -> None:
    engine = create_engine("sqlite:///:memory:")
    nse = NSEModelRegistry(engine)
    forex = ForexModelRegistry(engine)
    nse_grain = ModelGrain("NSE", "ema_adx_trend", "15m", "rules_v1")
    forex_grain = ModelGrain("FOREX", "ema_adx_trend", "15m", "rules_v1")
    nse.register(_record(nse_grain, "nse-v1", ModelStatus.APPROVED))
    forex.register(_record(forex_grain, "forex-v1", ModelStatus.APPROVED))
    with pytest.raises(ValueError, match="another market"):
        nse.latest_active(forex_grain)
    with pytest.raises(ValueError, match="another market"):
        forex.latest_active(nse_grain)


@pytest.mark.parametrize(
    ("policy", "available", "run", "reject"),
    [
        (MLPolicy.DISABLED, True, False, False),
        (MLPolicy.OPTIONAL, False, False, False),
        (MLPolicy.OPTIONAL, True, True, False),
        (MLPolicy.REQUIRED, False, False, True),
        (MLPolicy.REQUIRED, True, True, False),
    ],
)
def test_ml_policy_is_candidate_local(
    policy: MLPolicy, available: bool, run: bool, reject: bool
) -> None:
    decision = evaluate_ml_policy(policy, active_model_available=available)
    assert decision.run_inference is run
    assert decision.reject_candidate is reject


def test_market_api_strategies_and_models_are_strictly_scoped() -> None:
    registry = MarketApiRegistry()
    registry.register(
        MarketApiView(
            market="NSE",
            status=lambda: {"status": "HEALTHY"},
            signals=lambda days: [],
            positions=lambda: [],
            strategies=lambda: [
                {"market": "NSE", "strategy_name": "ema_adx_trend"},
                {"market": "FOREX", "strategy_name": "leak"},
            ],
            models=lambda: [{"market": "NSE", "model_version": "v18"}],
        )
    )
    view = registry.get("NSE")
    assert view is not None
    assert [item["strategy_name"] for item in view.scoped_strategies()] == ["ema_adx_trend"]
    assert registry.models_status() == [{"market": "NSE", "model_version": "v18"}]


def test_market_runtime_does_not_import_trainer_or_walk_forward() -> None:
    for path in (
        Path("src/markets/nse/runtime/worker.py"),
        Path("src/markets/forex/runtime/worker.py"),
        Path("src/markets/forex/runtime/engine.py"),
    ):
        source = path.read_text(encoding="utf-8").lower()
        assert "src.background" not in source
        assert "walk_forward" not in source
        assert "offlinepredictiontrainer" not in source
