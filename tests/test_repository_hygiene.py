"""Physical-boundary guards for the NSE/Forex peer-domain repository layout."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.backtesting.strategy_eligibility import eligibility_registry_from_settings
from src.config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_configuration_files_do_not_live_at_repository_root() -> None:
    forbidden = {
        ".dhan_previous_close_cache.json",
        ".dhan_token_cache.json",
        ".env",
        ".env.common.example",
        ".env.example",
        ".env.forex.live.example",
        ".env.forex.practice.example",
        ".env.nse.example",
        "DeltaQuant.zip",
        "dummy_journal.db",
        "exit_manager_state.paper_market_data.json",
        "live_trading_restart.err.log",
        "live_trading_restart.log",
        "run_live_trading.lock",
        "symbols.csv",
    }
    assert not {path.name for path in ROOT.iterdir()} & forbidden
    assert not (ROOT / "run" / "markets").exists()
    assert not list((ROOT / "run").rglob("*.log"))


def test_legacy_market_source_tree_is_removed() -> None:
    assert not (ROOT / "src" / "market").exists()


def test_compatibility_launcher_is_thin_and_targets_canonical_nse_runtime() -> None:
    launcher = (ROOT / "scripts" / "run_live_trading.py").read_text(encoding="utf-8")
    assert "from src.markets.nse.runtime.live import main" in launcher
    assert "def _run_cycle" not in launcher
    assert len(launcher.splitlines()) <= 25


def test_source_does_not_import_removed_market_aliases() -> None:
    offenders: list[str] = []
    removed_aliases = ("src.market.", "src.execution.", "src.risk.", "src.utils.")
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(alias in source for alias in removed_aliases):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_core_has_no_market_or_broker_imports() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src" / "core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        if any(
            module.startswith("src.markets.")
            or "dhan" in module.lower()
            or "oanda" in module.lower()
            for module in modules
        ):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_nse_runtime_path_defaults_are_market_owned() -> None:
    fields = Settings.model_fields
    assert fields["dhan_token_cache_file"].default == "run/nse/dhan/token_cache.json"
    assert (
        fields["dhan_previous_close_cache_file"].default
        == "run/nse/dhan/previous_close_cache.json"
    )
    assert fields["stock_universe_csv_path"].default == "config/nse/symbols.csv"
    assert fields["strategy_eligibility_registry_dir"].default.startswith("data/nse/")


def test_strategy_eligibility_paths_are_owned_by_the_active_market(tmp_path: Path) -> None:
    configured = tmp_path / "data" / "nse" / "strategy_eligibility"
    legacy = tmp_path / "data" / "nse" / "strategy_registry"

    with patch(
        "src.backtesting.strategy_eligibility.migrate_legacy_strategy_registry"
    ) as migrate:
        nse = eligibility_registry_from_settings(
            SimpleNamespace(
                market="NSE",
                strategy_registry_dir=str(legacy),
                strategy_eligibility_registry_dir=str(configured),
            )
        )
        forex = eligibility_registry_from_settings(
            SimpleNamespace(
                market="FOREX",
                strategy_registry_dir=str(legacy),
                strategy_eligibility_registry_dir=str(configured),
            )
        )

    assert nse.directory == configured
    assert forex.directory == tmp_path / "data" / "forex" / "strategy_eligibility"
    assert migrate.call_args_list[0].args[0].directory == legacy
    assert (
        migrate.call_args_list[1].args[0].directory
        == tmp_path / "data" / "forex" / "strategy_registry"
    )
