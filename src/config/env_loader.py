"""Select exactly one market dotenv profile with deterministic precedence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def selected_market(environ: Mapping[str, str] | None = None) -> str:
    values = environ or os.environ
    market = values.get("MARKET", "NSE").strip().upper()
    if market not in {"NSE", "FOREX", "CRYPTO"}:
        raise ValueError("MARKET must be NSE, FOREX, or CRYPTO")
    return market


def selected_forex_environment(environ: Mapping[str, str] | None = None) -> str:
    values = environ or os.environ
    selected = values.get("FOREX_ENVIRONMENT", values.get("OANDA_ENVIRONMENT", "practice"))
    normalized = selected.strip().lower()
    if normalized not in {"practice", "live"}:
        raise ValueError("FOREX_ENVIRONMENT/OANDA_ENVIRONMENT must be practice or live")
    return normalized


def resolve_env_files(
    base_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return existing dotenv files in low-to-high precedence order.

    Broker profiles are mutually exclusive by construction. Repository-root dotenv
    fallbacks are intentionally unsupported: all profiles live below ``env/``.
    Pydantic Settings reads process environment variables after dotenv values, so
    process variables retain the highest precedence.
    """

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    values = environ or os.environ
    market = selected_market(values)
    env_root = root / "env"
    common = env_root / ".env.common"
    files: list[Path] = [common] if common.exists() else []
    profile: Path | None
    if market == "NSE":
        profile = env_root / ".env.nse"
    elif market == "FOREX":
        name = f".env.forex.{selected_forex_environment(values)}"
        profile = env_root / name
    elif market == "CRYPTO":
        profile = env_root / ".env.crypto"
    else:  # pragma: no cover - selected_market normalizes to a supported market
        profile = None
    if profile is not None and profile.exists():
        files.append(profile)
    return tuple(str(path) for path in files if path.exists())
