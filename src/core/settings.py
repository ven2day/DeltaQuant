"""Strict, broker-neutral settings used by independently bootstrapped workers.

``src.config.Settings`` remains a compatibility adapter for the legacy NSE loop.  New
market entry points validate one of the scoped settings models here before importing
that adapter, so a worker's configuration surface cannot expose another broker's
credentials.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr


class CommonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_env: str = "local"
    log_level: str = "INFO"
    database_url: SecretStr = SecretStr("")
    timescale_url: SecretStr = SecretStr("")
    redis_url: SecretStr = SecretStr("")
    langfuse_host: str = ""
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    qwen_model: str = ""
    qwen_api_base: str = ""
    qwen_api_key: SecretStr = SecretStr("")
    ml_model_root: str = "artifacts"
    strategy_config_root: str = "config"
    trading_mode: Literal["paper", "live"] = "paper"
    global_kill_switch: bool = False
    runtime_id: str = "deltaquant"


__all__ = ["CommonSettings"]
