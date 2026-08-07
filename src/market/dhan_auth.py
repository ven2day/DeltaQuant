"""DhanHQ programmatic authentication: PIN + TOTP -> a fresh access token.

DhanHQ's manually-generated web-portal tokens expire in ~24 hours, a poor fit
for a bot meant to run unattended (see the runtime failures this caused
before this module existed: every historical/live Dhan call started failing
once the pasted token went stale). This uses DhanHQ's documented direct login
endpoint to mint a fresh token on demand and caches it to disk so a restart
within the token's validity window skips the login round-trip.

Verified against DhanHQ's docs — no guessing:
    POST https://auth.dhan.co/app/generateAccessToken
        ?dhanClientId=...&pin=......&totp=123456
    -> {"dhanClientId": ..., "accessToken": "...", "expiryTime": "...", ...}
No API key/secret header is required for this specific endpoint (Dhan's docs
are explicit about that — only the query parameters above).

Expiry comparison uses IST throughout (never bare ``datetime.now()``, which
reads the host's local clock — this dev environment's host clock has been
observed off-zone in the past, and a wrong "is my token still valid" judgement
here would either force needless re-logins or, worse, keep using a token past
its real expiry until DhanHQ itself rejects it).
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyotp
import requests

from src.config import get_settings
from src.utils.market_time import IST, now_ist

logger = logging.getLogger(__name__)

GENERATE_TOKEN_PATH = "/app/generateAccessToken"
REQUEST_TIMEOUT_SECONDS = 30
# Refresh a bit before the documented 24h expiry, not right at the edge — avoids a
# request failing mid-flight because the token expired seconds earlier.
REFRESH_MARGIN_MINUTES = 30


def _fetch_fresh_token(client_id: str, pin: str, totp_secret: str, auth_base_url: str) -> dict[str, Any]:
    """Call DhanHQ's generateAccessToken endpoint. Raises on any failure — the
    caller decides how to degrade."""
    totp_code = pyotp.TOTP(totp_secret).now()
    response = requests.post(
        f"{auth_base_url}{GENERATE_TOKEN_PATH}",
        params={"dhanClientId": client_id, "pin": pin, "totp": totp_code},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    if "accessToken" not in data or "expiryTime" not in data:
        raise ValueError(f"DhanHQ login response missing expected fields: {list(data.keys())}")
    return data


def _load_cache(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            cached: dict[str, Any] = json.load(f)
            return cached
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(path: Path, access_token: str, expiry_time: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"access_token": access_token, "expiry_time": expiry_time}, f)
    except OSError:
        logger.warning("Could not write DhanHQ token cache to %s", path, exc_info=True)


def _is_cache_valid(cache: dict[str, Any] | None) -> bool:
    if not cache or "access_token" not in cache or "expiry_time" not in cache:
        return False
    try:
        # DhanHQ's expiryTime has no timezone marker; treat it as IST (Dhan is an
        # Indian broker) rather than the host's local zone.
        expiry = datetime.fromisoformat(cache["expiry_time"]).replace(tzinfo=IST)
    except ValueError:
        return False
    return now_ist() < expiry - timedelta(minutes=REFRESH_MARGIN_MINUTES)


def get_valid_access_token() -> str | None:
    """Return a usable DhanHQ access token, preferring the PIN+TOTP auto-login flow.

    Order of preference:
    1. A cached auto-generated token that isn't close to expiring.
    2. A freshly-generated token via PIN+TOTP (cached for next time), if those
       are configured.
    3. The static ``dhan_access_token`` setting, if set (manual-token fallback
       for anyone not using PIN+TOTP).
    4. None — caller decides how to handle "no usable token" (e.g. degrade to
       simulated/local-paper mode, matching this codebase's fail-open
       convention for optional broker integrations).

    Never raises: any failure in the auto-login path logs a warning and falls
    through to the next option.
    """
    settings = get_settings()
    cache_path = Path(settings.dhan_token_cache_file)

    cache = _load_cache(cache_path)
    if _is_cache_valid(cache):
        assert cache is not None  # narrowed by _is_cache_valid
        return str(cache["access_token"])

    client_id = settings.dhan_client_id
    pin = settings.dhan_pin.get_secret_value() if settings.dhan_pin else None
    totp_secret = settings.dhan_totp_secret.get_secret_value() if settings.dhan_totp_secret else None

    if client_id and pin and totp_secret:
        try:
            data = _fetch_fresh_token(client_id, pin, totp_secret, settings.dhan_auth_base_url)
            _save_cache(cache_path, data["accessToken"], data["expiryTime"])
            logger.info("Generated a fresh DhanHQ access token (expires %s)", data["expiryTime"])
            return str(data["accessToken"])
        except Exception:
            logger.warning(
                "DhanHQ PIN+TOTP auto-login failed; falling back to static "
                "DHAN_ACCESS_TOKEN if configured",
                exc_info=True,
            )

    if settings.dhan_access_token:
        return settings.dhan_access_token.get_secret_value()

    return None


def get_dhan_client(client_id: str, access_token: str) -> Any:
    """Construct a ``dhanhq`` SDK client with a correct, complete header.

    Works around a real bug in the installed ``dhanhq`` package: its ``__init__``
    builds ``self.header`` with only ``access-token``/``Content-type``/``Accept`` —
    no ``client-id`` — and most data-API methods (``historical_daily_data``,
    ``intraday_minute_data``, etc.) send that incomplete header instead of building
    their own (unlike ``quote_data``, which happens to build a correct header
    inline). Confirmed live: every call through the broken header fails with
    ``DH-905 Input_Exception`` ("missing required fields"), and the exact same
    call succeeds once ``client-id`` is added. Patching the instance's header
    dict here, once, fixes every SDK method that reads it — far less fragile
    than bypassing the SDK with raw ``requests`` calls at each call site.
    """
    from dhanhq import dhanhq

    client = dhanhq(client_id, access_token)
    client.header["client-id"] = client_id
    return client
