"""Tests for the CSV-overridable stock universe in src/market/stock_discovery.py."""

from unittest.mock import patch

from src.market.stock_discovery import (
    MIDCAP_STOCKS,
    NIFTY50_STOCKS,
    StockDiscovery,
    _load_universe_from_csv,
)


def test_load_universe_from_csv_reads_symbol_column(tmp_path):
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol\nRELIANCE\nTCS\nINFY\n", encoding="utf-8")

    symbols = _load_universe_from_csv(str(csv_path))

    assert symbols == ["RELIANCE", "TCS", "INFY"]


def test_load_universe_from_csv_dedupes_and_normalizes(tmp_path):
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol\n reliance \nTCS\ntcs\nRELIANCE\n\n", encoding="utf-8")

    symbols = _load_universe_from_csv(str(csv_path))

    assert symbols == ["RELIANCE", "TCS"]


def test_load_universe_from_csv_missing_file_returns_none():
    assert _load_universe_from_csv("Z:/does/not/exist.csv") is None


def test_load_universe_from_csv_missing_symbol_column_returns_none(tmp_path):
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("ticker\nRELIANCE\n", encoding="utf-8")

    assert _load_universe_from_csv(str(csv_path)) is None


def test_load_universe_from_csv_empty_file_returns_none(tmp_path):
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol\n\n\n", encoding="utf-8")

    assert _load_universe_from_csv(str(csv_path)) is None


def test_stock_discovery_uses_csv_universe_when_configured(tmp_path):
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol\nZOMATO\nPAYTM\n", encoding="utf-8")

    with patch("src.market.stock_discovery.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = str(csv_path)
        discovery = StockDiscovery(max_stocks=10)

    assert discovery.universe == ["ZOMATO", "PAYTM"]


def test_stock_discovery_falls_back_to_builtin_list_when_csv_unset():
    with patch("src.market.stock_discovery.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = None
        discovery = StockDiscovery(max_stocks=10)

    assert set(discovery.universe) == set(NIFTY50_STOCKS + MIDCAP_STOCKS)


def test_stock_discovery_falls_back_to_builtin_list_when_csv_missing():
    with patch("src.market.stock_discovery.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = "Z:/does/not/exist.csv"
        discovery = StockDiscovery(max_stocks=10)

    assert set(discovery.universe) == set(NIFTY50_STOCKS + MIDCAP_STOCKS)
