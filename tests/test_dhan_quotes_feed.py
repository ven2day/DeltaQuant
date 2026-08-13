"""Tests for src/markets/nse/broker/dhan/quotes.py — DhanHQ live quotes.

Response shapes mirror real live calls (quote_data double-nested under
"data"; historical_daily_data returning parallel arrays including "close") —
see the module docstring for why change_percent is computed from a separately
fetched previous close rather than the quote's own net_change field.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.markets.nse.broker.dhan.quotes import (
    DhanQuotesFeed,
    QuotesFeed,
    _fetch_previous_close,
    _parse_quote,
)


@pytest.fixture(autouse=True)
def _no_real_rate_limiting():
    # Tests must never touch the real global DhanHQ rate-limiter singleton — it's
    # shared process-wide state (see src/utils/rate_limiter.py); depleting its token
    # bucket across many tests causes genuine sleeps, not a mock-friendly no-op.
    with (
        patch("src.markets.nse.broker.dhan.quotes.get_dhan_data_api_limiter") as mock_data_limiter,
        patch("src.markets.nse.broker.dhan.quotes.get_dhan_quote_api_limiter") as mock_quote_limiter,
    ):
        yield mock_data_limiter, mock_quote_limiter


def _mock_settings(market_data_source="dhan", enable_dhan_quotes=True, cache_file="dummy.json"):
    settings = MagicMock()
    settings.dhan_client_id = "client-id"
    settings.dhan_exchange_segment = "NSE_EQ"
    settings.market_data_source = market_data_source
    settings.enable_dhan_quotes = enable_dhan_quotes
    settings.dhan_previous_close_cache_file = cache_file
    return settings


def _feed(security_ids=None, cache_file="dummy.json"):
    security_ids = security_ids or {"RELIANCE": "2885", "TCS": "11536"}
    with (
        patch(
            "src.markets.nse.broker.dhan.quotes.get_settings",
            return_value=_mock_settings(cache_file=cache_file),
        ),
        patch("src.markets.nse.broker.dhan.quotes.get_valid_access_token", return_value="token"),
        patch("src.markets.nse.broker.dhan.quotes.get_dhan_client"),
        patch("src.markets.nse.broker.dhan.quotes.fetch_security_id_map", return_value=security_ids),
    ):
        return DhanQuotesFeed(symbols=list(security_ids.keys()))


def _quote_data_response(entries: dict):
    """entries: {security_id_str: payload_dict}"""
    return {
        "status": "success",
        "data": {"data": {"NSE_EQ": entries}, "status": "success"},
    }


def _daily_response(closes: list):
    return {"status": "success", "data": {"close": closes}}


# --- _fetch_previous_close ---


def test_fetch_previous_close_returns_last_bar():
    client = MagicMock()
    client.historical_daily_data.return_value = _daily_response([100.0, 110.0, 1280.0])

    result = _fetch_previous_close(client, "NSE_EQ", "EQUITY", "2885")

    assert result == 1280.0


def test_fetch_previous_close_none_on_failure_status():
    client = MagicMock()
    client.historical_daily_data.return_value = {"status": "failure", "remarks": "x"}

    assert _fetch_previous_close(client, "NSE_EQ", "EQUITY", "2885") is None


def test_fetch_previous_close_none_on_empty_data():
    client = MagicMock()
    client.historical_daily_data.return_value = _daily_response([])

    assert _fetch_previous_close(client, "NSE_EQ", "EQUITY", "2885") is None


def test_fetch_previous_close_none_on_exception():
    client = MagicMock()
    client.historical_daily_data.side_effect = Exception("network error")

    assert _fetch_previous_close(client, "NSE_EQ", "EQUITY", "2885") is None


# --- _parse_quote ---


def test_parse_quote_computes_change_from_previous_close():
    payload = {
        "last_price": 1325.0,
        "ohlc": {"open": 1285.0, "high": 1325.2, "low": 1281.2, "close": 1325.0},
        "net_change": 0,  # confirmed live to be unreliable — must be ignored
        "volume": 20342297,
    }

    quote = _parse_quote("RELIANCE", payload, previous_close=1280.0)

    assert quote is not None
    assert quote.last_price == 1325.0
    assert quote.close == 1280.0
    assert quote.change == 45.0
    assert quote.change_percent == round(45.0 / 1280.0 * 100, 2)
    assert quote.open == 1285.0


def test_parse_quote_zero_change_when_previous_close_unavailable():
    payload = {"last_price": 100.0, "ohlc": {}, "volume": 0}

    quote = _parse_quote("X", payload, previous_close=None)

    assert quote is not None
    assert quote.change == 0.0
    assert quote.change_percent == 0.0
    assert quote.close == 0.0


def test_parse_quote_none_on_malformed_payload():
    assert _parse_quote("X", {"ohlc": {}}, previous_close=100.0) is None


# --- DhanQuotesFeed.fetch_quotes ---


def test_fetch_quotes_parses_batched_response_using_previous_close(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._client.historical_daily_data.side_effect = [
        _daily_response([1280.0]),  # RELIANCE
        _daily_response([2400.0]),  # TCS
    ]
    feed._client.quote_data.return_value = _quote_data_response(
        {
            "2885": {
                "last_price": 1325.0,
                "ohlc": {"open": 1285.0, "high": 1325.2, "low": 1281.2, "close": 1325.0},
                "net_change": 0,
                "volume": 100,
            },
            "11536": {
                "last_price": 2373.0,
                "ohlc": {"open": 2417.0, "high": 2434.5, "low": 2368.5, "close": 2373.0},
                "net_change": 0,
                "volume": 200,
            },
        }
    )

    quotes = feed.fetch_quotes()

    assert set(quotes.keys()) == {"RELIANCE", "TCS"}
    assert quotes["RELIANCE"].change == 45.0
    assert quotes["TCS"].change_percent == round(-27.0 / 2400.0 * 100, 2)


def test_fetch_quotes_skips_previous_close_requests_when_history_is_disabled(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._fetch_previous_closes = False
    feed._client.quote_data.return_value = _quote_data_response(
        {"2885": {"last_price": 1325.0, "ohlc": {}, "volume": 100}}
    )

    quotes = feed.fetch_quotes()

    feed._client.historical_daily_data.assert_not_called()
    assert quotes["RELIANCE"].last_price == 1325.0


def test_fetch_quotes_batches_at_1000_securities_per_request(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    security_ids = {f"SYM{i}": str(i) for i in range(1, 1002)}
    feed = _feed(security_ids=security_ids, cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([100.0])
    feed._client.quote_data.return_value = _quote_data_response({})

    feed.fetch_quotes()

    assert feed._client.quote_data.call_count == 2
    first_call_ids = feed._client.quote_data.call_args_list[0][0][0]["NSE_EQ"]
    assert len(first_call_ids) == 1000


def test_fetch_quotes_uses_the_correct_limiter_for_each_endpoint(tmp_path, _no_real_rate_limiting):
    # Regression guard: a live 276-symbol scan showed DhanHQ enforces one account-
    # wide 5 req/s cap across ALL its Data APIs combined (not per-endpoint) — a local
    # per-file delay isn't enough since other concurrent Dhan callers (e.g. the
    # scalping screener's historical fetches) draw from the same budget. Every
    # quote_data batch call must go through the shared limiter.
    cache_file = str(tmp_path / "cache.json")
    security_ids = {f"SYM{i}": str(i) for i in range(250)}
    feed = _feed(security_ids=security_ids, cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([100.0])
    feed._client.quote_data.return_value = _quote_data_response({})

    feed.fetch_quotes()

    # One acquire per previous-close fetch (250, one per symbol) plus one per
    # quote_data batch (one request for all 250) — every single DhanHQ call draws from
    # the shared limiter, not just quote_data.
    data_limiter, quote_limiter = _no_real_rate_limiting
    assert data_limiter.return_value.acquire_sync.call_count == 250
    assert quote_limiter.return_value.acquire_sync.call_count == 1


def test_fetch_quotes_skips_failed_batch_without_raising(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([100.0])
    feed._client.quote_data.return_value = {"status": "failure", "remarks": "DH-901"}

    assert feed.fetch_quotes() == {}


def test_fetch_quotes_retries_a_failed_batch(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([100.0])
    feed._client.quote_data.side_effect = [
        {"status": "failure", "remarks": "temporary error"},
        _quote_data_response({}),
    ]

    assert feed.fetch_quotes() == {}
    assert feed._client.quote_data.call_count == 2


def test_fetch_quotes_handles_client_exception(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([100.0])
    feed._client.quote_data.side_effect = Exception("network error")

    assert feed.fetch_quotes() == {}


# --- previous-close caching ---


def test_previous_closes_are_cached_to_disk_and_reused(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(cache_file=cache_file)
    feed._client.historical_daily_data.return_value = _daily_response([1280.0])
    feed._client.quote_data.return_value = _quote_data_response({})

    feed.fetch_quotes()

    assert feed._client.historical_daily_data.call_count == 2  # RELIANCE + TCS
    saved = json.loads((tmp_path / "cache.json").read_text())
    assert saved["closes"]["RELIANCE"] == 1280.0

    # A second feed instance sharing the same cache file should reuse it
    # instead of refetching.
    feed2 = _feed(cache_file=cache_file)
    feed2._client.quote_data.return_value = _quote_data_response({})
    feed2.fetch_quotes()

    feed2._client.historical_daily_data.assert_not_called()


def test_stale_cached_date_triggers_refetch(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"date": "2000-01-01", "closes": {"RELIANCE": 1.0}}))
    feed = _feed(cache_file=str(cache_file))
    feed._client.historical_daily_data.return_value = _daily_response([1280.0])
    feed._client.quote_data.return_value = _quote_data_response({})

    feed.fetch_quotes()

    assert feed._client.historical_daily_data.call_count == 2


def test_symbol_missing_from_previous_closes_still_returns_a_quote_with_zero_change(tmp_path):
    cache_file = str(tmp_path / "cache.json")
    feed = _feed(security_ids={"RELIANCE": "2885"}, cache_file=cache_file)
    feed._client.historical_daily_data.return_value = {"status": "failure", "remarks": "x"}
    feed._client.quote_data.return_value = _quote_data_response(
        {"2885": {"last_price": 1325.0, "ohlc": {}, "net_change": 0, "volume": 1}}
    )

    quotes = feed.fetch_quotes()

    assert quotes["RELIANCE"].change_percent == 0.0


# --- QuotesFeed facade ---


def test_quotes_facade_prefers_dhan_when_it_returns_data():
    with (
        patch(
            "src.markets.nse.broker.dhan.quotes.get_settings",
            return_value=_mock_settings("dhan", True),
        ),
        patch("src.markets.nse.broker.dhan.quotes.DhanQuotesFeed") as mock_dhan_cls,
    ):
        mock_dhan_cls.return_value.fetch_quotes.return_value = {"RELIANCE": object()}
        feed = QuotesFeed(symbols=["RELIANCE"])

        result = feed.fetch_quotes()

    assert "RELIANCE" in result


def test_quotes_facade_returns_empty_when_dhan_returns_empty():
    with (
        patch(
            "src.markets.nse.broker.dhan.quotes.get_settings",
            return_value=_mock_settings("dhan", True),
        ),
        patch("src.markets.nse.broker.dhan.quotes.DhanQuotesFeed") as mock_dhan_cls,
    ):
        mock_dhan_cls.return_value.fetch_quotes.return_value = {}
        feed = QuotesFeed(symbols=["RELIANCE"])

        result = feed.fetch_quotes()

    assert result == {}


def test_quotes_facade_skips_dhan_when_disabled():
    with (
        patch(
            "src.markets.nse.broker.dhan.quotes.get_settings",
            return_value=_mock_settings("dhan", False),
        ),
        patch("src.markets.nse.broker.dhan.quotes.DhanQuotesFeed") as mock_dhan_cls,
    ):
        feed = QuotesFeed(symbols=["RELIANCE"])
        result = feed.fetch_quotes()

    mock_dhan_cls.assert_not_called()
    assert result == {}
