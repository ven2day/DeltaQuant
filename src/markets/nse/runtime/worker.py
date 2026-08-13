"""Launch the existing event-driven NSE runtime in its own process domain."""

from __future__ import annotations

from src.config import get_settings
from src.markets.nse.config import load_nse_settings
from src.markets.nse.runtime.live import main as run_nse_runtime


def main() -> None:
    scoped = load_nse_settings()
    if scoped.nse_execution_enabled and scoped.trading_mode != "paper":
        raise RuntimeError("NSE broker execution remains disabled by this deployment")
    settings = get_settings()
    if str(settings.market).upper() != "NSE":
        raise RuntimeError("NSE worker requires MARKET=NSE and loads only env/.env.nse")
    run_nse_runtime()
