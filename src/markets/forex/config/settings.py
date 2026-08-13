"""Forex-only settings loader; it never reads a Dhan profile."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr

from src.config.env_loader import resolve_env_files
from src.core.settings import CommonSettings


class ForexSettings(CommonSettings):
    """Configuration owned exclusively by the Forex worker."""

    market: Literal["FOREX"] = "FOREX"
    forex_enabled: bool = True
    forex_provider: Literal["oanda"] = "oanda"
    forex_environment: Literal["practice", "live"] = "practice"
    oanda_environment: Literal["practice", "live"] = "practice"
    oanda_account_id: SecretStr = SecretStr("")
    oanda_access_token: SecretStr = SecretStr("")
    forex_execution_enabled: bool = False
    forex_live_trading_ack: str = ""
    forex_kill_switch: bool = False
    forex_db_schema: Literal["forex"] = "forex"
    forex_config_root: str = "config/forex"
    forex_artifact_root: str = "artifacts/forex"
    forex_log_root: str = "logs/forex"
    forex_timezone: str = "UTC"
    forex_symbols: str = Field(default="EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD")


def load_forex_settings(
    *, base_dir: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> ForexSettings:
    process = dict(os.environ if environ is None else environ)
    process["MARKET"] = "FOREX"
    raw: dict[str, str] = {}
    for path in resolve_env_files(base_dir, process):
        raw.update({key: value for key, value in dotenv_values(path).items() if value is not None})
    raw.update(process)
    fields = ForexSettings.model_fields
    values = {name: raw[name.upper()] for name in fields if name.upper() in raw}
    values["market"] = "FOREX"
    return ForexSettings.model_validate(values)


__all__ = ["ForexSettings", "load_forex_settings"]
