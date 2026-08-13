"""Crypto service scaffold with no implicit provider or execution fallback."""

from __future__ import annotations

import logging

from src.config import get_settings
from src.core.utils.logging_config import configure_runtime_logging
from src.markets.snapshots import MarketSnapshotStore

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    if str(settings.market).upper() != "CRYPTO":
        raise RuntimeError("Crypto worker requires MARKET=CRYPTO and loads only env/.env.crypto")
    configure_runtime_logging(
        None,
        market="CRYPTO",
        provider=str(getattr(settings, "crypto_provider", "UNCONFIGURED")),
        runtime_id=str(getattr(settings, "runtime_id", "crypto")),
    )
    if not bool(getattr(settings, "crypto_enabled", False)):
        MarketSnapshotStore(str(settings.market_snapshot_root)).publish(
            "CRYPTO",
            status={
                "status": "DISABLED",
                "provider": "UNCONFIGURED",
                "execution": "OFF",
                "runtime_id": str(getattr(settings, "runtime_id", "crypto")),
            },
            signals=[],
            positions=[],
            force=True,
        )
        logger.info("CRYPTO runtime disabled; no provider credentials loaded")
        return
    raise RuntimeError("CRYPTO_ENABLED=true but no crypto provider has been configured; fail closed")
