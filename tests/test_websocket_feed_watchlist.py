"""Tests for websocket_feed.py's dynamic NSE_WATCHLIST refresh.

NSE_WATCHLIST/SECURITY_ID_TO_SYMBOL are module-level mutable state (deliberately,
so existing `from ... import NSE_WATCHLIST` call sites see updates without
re-importing) — every test restores the original contents afterward so this
file can't leak state into other tests that happen to run in the same process.
"""

from unittest.mock import patch

import pytest

import src.market.websocket_feed as websocket_feed


@pytest.fixture(autouse=True)
def _restore_watchlist():
    original_watchlist = dict(websocket_feed.NSE_WATCHLIST)
    original_reverse = dict(websocket_feed.SECURITY_ID_TO_SYMBOL)
    yield
    websocket_feed.NSE_WATCHLIST.clear()
    websocket_feed.NSE_WATCHLIST.update(original_watchlist)
    websocket_feed.SECURITY_ID_TO_SYMBOL.clear()
    websocket_feed.SECURITY_ID_TO_SYMBOL.update(original_reverse)


def test_refresh_watchlist_replaces_contents_on_success():
    with patch("src.market.websocket_feed.fetch_security_id_map", return_value={"FOO": "999"}):
        result = websocket_feed.refresh_watchlist(["FOO"])

    assert result == {"FOO": "999"}
    assert websocket_feed.NSE_WATCHLIST == {"FOO": "999"}
    assert websocket_feed.SECURITY_ID_TO_SYMBOL == {"999": "FOO"}


def test_refresh_watchlist_is_a_noop_when_resolution_returns_empty():
    original = dict(websocket_feed.NSE_WATCHLIST)
    with patch("src.market.websocket_feed.fetch_security_id_map", return_value={}):
        result = websocket_feed.refresh_watchlist(["NOTHING_RESOLVED"])

    # A failed/empty resolution must never wipe out an already-working watchlist.
    assert result == original
    assert websocket_feed.NSE_WATCHLIST == original


def test_refresh_watchlist_mutates_in_place_for_existing_imports():
    # Simulates `from src.market.websocket_feed import NSE_WATCHLIST` used elsewhere
    # (manager.py, run_live_trading.py): that binding is the SAME dict object, so it
    # must reflect the refresh without needing to be re-imported.
    aliased = websocket_feed.NSE_WATCHLIST
    with patch("src.market.websocket_feed.fetch_security_id_map", return_value={"BAR": "42"}):
        websocket_feed.refresh_watchlist(["BAR"])

    assert aliased is websocket_feed.NSE_WATCHLIST
    assert aliased == {"BAR": "42"}
