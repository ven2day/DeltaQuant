"""NSE-only settings loader; it never reads an OANDA profile."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import SecretStr

from src.config.env_loader import resolve_env_files
from src.core.settings import CommonSettings


class NSESettings(CommonSettings):
    """Configuration owned exclusively by the NSE worker."""

    market: Literal["NSE"] = "NSE"
    nse_enabled: bool = True
    nse_broker: Literal["dhan"] = "dhan"
    dhan_client_id: SecretStr = SecretStr("")
    dhan_access_token: SecretStr = SecretStr("")
    nse_execution_enabled: bool = False
    nse_kill_switch: bool = False
    nse_db_schema: Literal["nse"] = "nse"
    nse_config_root: str = "config/nse"
    nse_artifact_root: str = "artifacts/nse"
    nse_log_root: str = "logs/nse"
    nse_timezone: str = "Asia/Kolkata"


def load_nse_settings(
    *, base_dir: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> NSESettings:
    process = dict(os.environ if environ is None else environ)
    process["MARKET"] = "NSE"
    raw: dict[str, str] = {}
    for path in resolve_env_files(base_dir, process):
        raw.update({key: value for key, value in dotenv_values(path).items() if value is not None})
    raw.update(process)
    fields = NSESettings.model_fields
    values = {name: raw[name.upper()] for name in fields if name.upper() in raw}
    values["market"] = "NSE"
    return NSESettings.model_validate(values)


__all__ = ["NSESettings", "load_nse_settings"]
