"""Tests for src/market/dhan_auth.py — DhanHQ PIN+TOTP auto-login and token caching."""

import json
from datetime import timedelta
from unittest.mock import MagicMock, patch

from src.market.dhan_auth import _is_cache_valid, get_valid_access_token
from src.utils.market_time import now_ist


def _mock_settings(
    client_id="1111860304",
    pin="970097",
    totp_secret="JBSWY3DPEHPK3PXP",
    static_token=None,
    cache_file="dummy.json",
):
    settings = MagicMock()
    settings.dhan_client_id = client_id
    settings.dhan_pin = MagicMock() if pin else None
    if pin:
        settings.dhan_pin.get_secret_value.return_value = pin
    settings.dhan_totp_secret = MagicMock() if totp_secret else None
    if totp_secret:
        settings.dhan_totp_secret.get_secret_value.return_value = totp_secret
    settings.dhan_access_token = MagicMock() if static_token else None
    if static_token:
        settings.dhan_access_token.get_secret_value.return_value = static_token
    settings.dhan_auth_base_url = "https://auth.dhan.co"
    settings.dhan_token_cache_file = cache_file
    return settings


def _mock_login_response(access_token="fresh-token", expiry_time=None):
    if expiry_time is None:
        expiry_time = (now_ist() + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.000")
    resp = MagicMock()
    resp.json.return_value = {
        "dhanClientId": "1111860304",
        "accessToken": access_token,
        "expiryTime": expiry_time,
    }
    return resp


# --- _is_cache_valid ---


def test_cache_invalid_when_missing():
    assert _is_cache_valid(None) is False
    assert _is_cache_valid({}) is False


def test_cache_invalid_when_expiry_unparseable():
    assert _is_cache_valid({"access_token": "x", "expiry_time": "not-a-date"}) is False


def test_cache_valid_well_before_expiry():
    future = (now_ist() + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%S.000")
    assert _is_cache_valid({"access_token": "x", "expiry_time": future}) is True


def test_cache_invalid_within_refresh_margin_of_expiry():
    soon = (now_ist() + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.000")
    assert _is_cache_valid({"access_token": "x", "expiry_time": soon}) is False


def test_cache_invalid_when_already_expired():
    past = (now_ist() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000")
    assert _is_cache_valid({"access_token": "x", "expiry_time": past}) is False


# --- get_valid_access_token ---


def test_uses_valid_cached_token_without_hitting_network(tmp_path):
    cache_path = tmp_path / "cache.json"
    future = (now_ist() + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%S.000")
    cache_path.write_text(json.dumps({"access_token": "cached-token", "expiry_time": future}))

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(cache_file=str(cache_path)),
        ),
        patch("src.market.dhan_auth.requests.post") as mock_post,
    ):
        token = get_valid_access_token()

    assert token == "cached-token"
    mock_post.assert_not_called()


def test_generates_fresh_token_via_totp_when_cache_missing(tmp_path):
    cache_path = tmp_path / "cache.json"

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(cache_file=str(cache_path)),
        ),
        patch("src.market.dhan_auth.requests.post", return_value=_mock_login_response()),
    ):
        token = get_valid_access_token()

    assert token == "fresh-token"
    # Freshly generated token gets written back to the cache file.
    assert json.loads(cache_path.read_text())["access_token"] == "fresh-token"


def test_totp_code_is_generated_from_the_configured_secret(tmp_path):
    cache_path = tmp_path / "cache.json"

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(cache_file=str(cache_path)),
        ),
        patch(
            "src.market.dhan_auth.requests.post", return_value=_mock_login_response()
        ) as mock_post,
    ):
        get_valid_access_token()

    _, kwargs = mock_post.call_args
    assert kwargs["params"]["dhanClientId"] == "1111860304"
    assert kwargs["params"]["pin"] == "970097"
    assert len(kwargs["params"]["totp"]) == 6
    assert kwargs["params"]["totp"].isdigit()


def test_falls_back_to_static_token_when_totp_login_fails(tmp_path):
    cache_path = tmp_path / "cache.json"

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(
                cache_file=str(cache_path), static_token="manual-token"
            ),
        ),
        patch("src.market.dhan_auth.requests.post", side_effect=Exception("network down")),
    ):
        token = get_valid_access_token()

    assert token == "manual-token"


def test_returns_none_when_nothing_works(tmp_path):
    cache_path = tmp_path / "cache.json"

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(
                cache_file=str(cache_path), pin=None, totp_secret=None, static_token=None
            ),
        ),
    ):
        token = get_valid_access_token()

    assert token is None


def test_skips_totp_flow_when_pin_not_configured_uses_static_token(tmp_path):
    cache_path = tmp_path / "cache.json"

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(
                cache_file=str(cache_path), pin=None, static_token="manual-token"
            ),
        ),
        patch("src.market.dhan_auth.requests.post") as mock_post,
    ):
        token = get_valid_access_token()

    assert token == "manual-token"
    mock_post.assert_not_called()


def test_never_raises_on_malformed_login_response(tmp_path):
    cache_path = tmp_path / "cache.json"
    bad_response = MagicMock()
    bad_response.json.return_value = {"dhanClientId": "x"}  # missing accessToken/expiryTime

    with (
        patch(
            "src.market.dhan_auth.get_settings",
            return_value=_mock_settings(cache_file=str(cache_path), static_token="fallback"),
        ),
        patch("src.market.dhan_auth.requests.post", return_value=bad_response),
    ):
        token = get_valid_access_token()

    assert token == "fallback"
