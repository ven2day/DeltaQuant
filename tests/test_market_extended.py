from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.market.live_data import LiveMarketData, LiveQuote
from src.market.manager import MarketDataManager, MarketQuote, is_market_open
from src.market.simulated_data import SimulatedMarketData, SimulatedQuote
from src.market.stock_discovery import StockDiscovery
from src.market.websocket_feed import DhanWebSocketFeed, QuoteData

# --- MarketDataManager Tests ---


@pytest.fixture
def mock_settings():
    with patch("src.market.manager.get_settings") as mock:
        mock.return_value.market_data_source = "simulated"
        mock.return_value.dhan_client_id = "test_id"
        mock.return_value.dhan_access_token.get_secret_value.return_value = "test_token"
        yield mock


def test_is_market_open():
    # is_market_open() now evaluates in IST via utils.market_time.now_ist(),
    # so the clock seam to patch is now_ist (not manager.datetime).
    with patch("src.utils.market_time.now_ist") as mock_now:
        clock = MagicMock()
        mock_now.return_value = clock

        # Weekday 10:00 AM IST - Open
        clock.weekday.return_value = 0  # Monday
        clock.time.return_value = time(10, 0)
        assert is_market_open() is True

        # Weekend - Closed
        clock.weekday.return_value = 5  # Saturday
        assert is_market_open() is False

        # Weekday 8:00 AM IST - Closed (before open)
        clock.weekday.return_value = 0
        clock.time.return_value = time(8, 0)
        assert is_market_open() is False


@pytest.mark.asyncio
async def test_manager_start_simulated(mock_settings):
    mock_settings.return_value.market_data_source = "simulated"
    manager = MarketDataManager(symbols=["RELIANCE"])

    # Mock simulated data loading
    with patch.object(manager.simulated_data, "get_quotes") as mock_get_quotes:
        mock_get_quotes.return_value = {
            "RELIANCE": SimulatedQuote("RELIANCE", 1, 100, 100, 100, 100, 90, 10, 10, 1000)
        }

        is_live = await manager.start()

        assert is_live is False
        assert manager.data_source == "simulated"
        assert "RELIANCE" in manager.quotes
        assert manager.quotes["RELIANCE"].is_live is False


@pytest.mark.asyncio
async def test_manager_start_dhan(mock_settings):
    mock_settings.return_value.market_data_source = "dhan"

    with patch("src.market.manager.is_market_open", return_value=True):
        manager = MarketDataManager(symbols=["RELIANCE"])

        with patch("src.market.manager.DhanWebSocketFeed") as MockWS:
            mock_ws = MockWS.return_value
            mock_ws.connect = AsyncMock(return_value=True)
            mock_ws.subscribe_nse_stocks = AsyncMock()

            is_live = await manager.start()

            assert is_live is True
            assert manager.data_source == "dhan"
            assert manager.is_live is True


@pytest.mark.asyncio
async def test_manager_uses_rest_quotes_when_market_is_closed(mock_settings):
    mock_settings.return_value.market_data_source = "dhan"
    mock_settings.return_value.enable_dhan_quotes = True

    with (
        patch("src.market.manager.is_market_open", return_value=False),
        patch("src.market.manager.QuotesFeed") as mock_feed_cls,
    ):
        quote = MagicMock(
            symbol="RELIANCE",
            last_price=1325.0,
            open=1285.0,
            high=1325.2,
            low=1281.2,
            close=1280.0,
            change=45.0,
            change_percent=3.52,
            volume=100,
        )
        mock_feed_cls.return_value.fetch_quotes.return_value = {"RELIANCE": quote}
        manager = MarketDataManager(symbols=["RELIANCE"])

        assert await manager.start() is True

    assert manager.data_source == "dhan_rest"
    assert manager.quotes["RELIANCE"].is_live is True


def test_manager_on_websocket_quote(mock_settings):
    manager = MarketDataManager()
    quote = QuoteData(
        "RELIANCE", 1, "NSE_EQ", 100, 1, datetime.now(), 100, 1000, 0, 0, 90, 110, 110, 90
    )

    manager._on_websocket_quote(quote)
    assert "RELIANCE" in manager.quotes
    assert manager.quotes["RELIANCE"].last_price == 100


def test_manager_get_trading_candidates(mock_settings):
    manager = MarketDataManager()
    manager.quotes = {
        "A": MarketQuote("A", 100, 100, 100, 100, 100, 10, 10, 1000, False),  # 10%
        "B": MarketQuote("B", 100, 100, 100, 100, 100, 1, 1, 1000, False),  # 1%
    }

    candidates = manager.get_trading_candidates(min_change=5.0)
    assert len(candidates) == 1
    assert candidates[0].symbol == "A"


# --- LiveMarketData Tests ---


@pytest.fixture
def live_market():
    with (
        patch("src.market.live_data.get_settings") as mock_settings,
        patch("src.market.live_data.get_valid_access_token", return_value="token"),
    ):
        mock_settings.return_value.dhan_base_url = "http://test"
        mock_settings.return_value.dhan_client_id = "id"
        yield LiveMarketData()


def test_live_get_quotes(live_market):
    with patch("src.market.live_data.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "status": "success",
            "data": {
                "NSE_EQ": {
                    "2885": {
                        "last_price": 2500,
                        "ohlc": {"open": 2400, "high": 2550, "low": 2400, "close": 2400},
                    }
                }
            },
        }

        quotes = live_market.get_quotes(["RELIANCE"])
        assert "RELIANCE" in quotes
        assert quotes["RELIANCE"].last_price == 2500
        assert quotes["RELIANCE"].is_bullish is True


def test_live_get_trading_candidates(live_market):
    with patch.object(live_market, "get_quotes") as mock_get_quotes:
        mock_get_quotes.return_value = {
            "A": LiveQuote("A", 1, 110, 100, 110, 100, 100, 10, 10.0),
            "B": LiveQuote("B", 2, 100.1, 100, 100.1, 100, 100, 0.1, 0.1),
        }

        candidates = live_market.get_trading_candidates()
        assert len(candidates) == 1
        assert candidates[0].symbol == "A"


# --- SimulatedMarketData Tests ---


def test_simulated_get_quotes():
    sim = SimulatedMarketData()
    quotes = sim.get_quotes(["RELIANCE"])

    assert "RELIANCE" in quotes
    assert quotes["RELIANCE"].symbol == "RELIANCE"
    assert quotes["RELIANCE"].last_price > 0


def test_simulated_tick():
    sim = SimulatedMarketData()
    sim.get_quotes()  # Init

    initial_price = sim.current_prices["RELIANCE"]
    sim.tick()
    new_price = sim.current_prices["RELIANCE"]

    assert initial_price != new_price


def test_simulated_historical_feed_uses_canonical_stream():
    sim = SimulatedMarketData(symbols=["RELIANCE"], history_bars=500)

    feed_history = sim.get_historical("RELIANCE", period="10d", interval="15m")
    canonical_history = sim.get_history("RELIANCE", timeframe="15m")

    assert feed_history is not None
    assert canonical_history is not None
    assert feed_history.equals(canonical_history)


def test_simulated_get_trading_candidates():
    sim = SimulatedMarketData()
    # Force high volatility to ensure changes
    sim.volatility = 0.5
    candidates = sim.get_trading_candidates(min_change=0.0)
    assert len(candidates) > 0


# --- StockDiscovery Tests ---


@pytest.fixture
def discovery():
    with patch("src.market.stock_discovery.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = None
        return StockDiscovery(max_stocks=10)


def test_extract_stock_mentions(discovery):
    text = "Reliance and TCS are doing well today. Also Bajaj Auto."
    mentions = discovery._extract_stock_mentions(text)
    assert "RELIANCE" in mentions
    assert "TCS" in mentions
    assert "BAJAJ-AUTO" in mentions


def test_discover_from_news(discovery):
    with patch("src.market.stock_discovery.feedparser.parse") as mock_parse:
        mock_parse.return_value.entries = [
            {"title": "Reliance surges", "summary": "Reliance hits new high"},
            {"title": "Market down", "summary": "Nothing happening"},
        ]

        mentions = discovery.discover_from_news()
        # Since the code loops over 3 queries, and we mock the response for all of them,
        # "Reliance" will be found 3 times (once per query)
        assert mentions.get("RELIANCE") == 3


def test_discover_market_movers(discovery):
    from src.market.dhan_quotes_feed import Quote

    with patch("src.market.stock_discovery.QuotesFeed") as mock_quotes_feed_cls:
        mock_quotes_feed_cls.return_value.fetch_quotes.return_value = {
            s: Quote(
                symbol=s,
                last_price=110.0,
                open=100.0,
                high=112.0,
                low=99.0,
                close=100.0,
                change=10.0,
                change_percent=10.0,
                volume=1000,
            )
            for s in discovery.universe
        }

        movers = discovery.discover_market_movers(min_change=5.0)
        # Should find movers because we mocked a 10% gain for every symbol
        assert len(movers) > 0


@pytest.mark.asyncio
async def test_discover(discovery):
    with (
        patch.object(discovery, "discover_from_news", return_value={"RELIANCE": 5}),
        patch.object(discovery, "discover_market_movers", return_value=[]),
    ):
        stocks = await discovery.discover()
        assert "RELIANCE" in stocks
        # Should also have fallback stocks
        assert len(stocks) >= 10


# --- DhanWebSocketFeed Tests ---


@pytest.mark.asyncio
async def test_websocket_feed_connect():
    with (
        patch("src.market.websocket_feed.websockets.connect", new_callable=AsyncMock),
        patch("src.market.websocket_feed.get_settings") as mock_settings,
        patch("src.market.websocket_feed.get_valid_access_token", return_value="token"),
    ):
        mock_settings.return_value.dhan_client_id = "id"
        mock_settings.return_value.dhan_feed_url = "wss://api-feed.dhan.co"

        feed = DhanWebSocketFeed()
        success = await feed.connect()
        assert success is True
        assert feed.connected is True


@pytest.mark.asyncio
async def test_websocket_feed_subscribe():
    with (
        patch("src.market.websocket_feed.websockets.connect", new_callable=AsyncMock),
        patch("src.market.websocket_feed.get_settings") as mock_settings,
        patch("src.market.websocket_feed.get_valid_access_token", return_value="token"),
    ):
        mock_settings.return_value.dhan_client_id = "id"
        mock_settings.return_value.dhan_feed_url = "wss://api-feed.dhan.co"

        feed = DhanWebSocketFeed()
        # Fake connection
        feed.ws = AsyncMock()
        feed.connected = True

        success = await feed.subscribe_nse_stocks(["RELIANCE"])
        assert success is True
        feed.ws.send.assert_called()
