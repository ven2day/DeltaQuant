"""Tests for the CSV-overridable sector mapping in src/markets/nse/universe/sectors.py."""

from unittest.mock import patch

from src.markets.nse.universe import sectors
from src.markets.nse.universe.sectors import STOCK_SECTORS, get_stock_sector


def _clear_cache():
    sectors._sector_override_cache.clear()


def test_get_stock_sector_uses_builtin_map_when_no_csv_configured():
    _clear_cache()
    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = None
        assert get_stock_sector("TCS") == "IT"
        assert get_stock_sector("UNKNOWNSTOCK") == "Unknown"


def test_get_stock_sector_normalizes_ns_suffix():
    _clear_cache()
    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = None
        assert get_stock_sector("tcs.ns") == "IT"


def test_csv_sector_column_overrides_builtin_map(tmp_path):
    _clear_cache()
    csv_path = tmp_path / "symbols.csv"
    # TCS is "IT" in STOCK_SECTORS — the CSV column should win for it, and supply
    # a label for NEWSTOCK which isn't in STOCK_SECTORS at all.
    csv_path.write_text(
        "symbol,sector\nTCS,Custom IT\nNEWSTOCK,Emerging Tech\n", encoding="utf-8"
    )

    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = str(csv_path)
        assert get_stock_sector("TCS") == "Custom IT"
        assert get_stock_sector("NEWSTOCK") == "Emerging Tech"
        # A symbol absent from both the CSV override and STOCK_SECTORS still falls
        # back to "Unknown" rather than crashing.
        assert get_stock_sector("SOMETHINGELSE") == "Unknown"


def test_csv_without_sector_column_falls_back_to_builtin_map(tmp_path):
    _clear_cache()
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol\nTCS\nRELIANCE\n", encoding="utf-8")

    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = str(csv_path)
        assert get_stock_sector("TCS") == STOCK_SECTORS["TCS"]


def test_missing_csv_falls_back_to_builtin_map():
    _clear_cache()
    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = "Z:/does/not/exist.csv"
        assert get_stock_sector("TCS") == "IT"


def test_overrides_are_cached_per_path(tmp_path):
    _clear_cache()
    csv_path = tmp_path / "symbols.csv"
    csv_path.write_text("symbol,sector\nTCS,Custom IT\n", encoding="utf-8")

    with patch("src.markets.nse.universe.sectors.get_settings") as mock_get_settings:
        mock_get_settings.return_value.stock_universe_csv_path = str(csv_path)
        get_stock_sector("TCS")
        assert str(csv_path) in sectors._sector_override_cache

    _clear_cache()
