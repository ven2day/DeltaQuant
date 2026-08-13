import asyncio
import contextlib
from unittest.mock import MagicMock

from src.markets.nse.broker.dhan.quotes import Quote
from src.markets.nse.universe.sector_movers import SectorMoversTracker, compute_sector_movers


def _quote(symbol: str, change_percent: float) -> Quote:
    return Quote(
        symbol=symbol,
        last_price=100.0,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        change=change_percent,
        change_percent=change_percent,
        volume=1000,
    )


def test_groups_by_sector():
    quotes = {
        "TCS": _quote("TCS", 2.0),  # IT
        "INFY": _quote("INFY", 1.0),  # IT
        "HDFCBANK": _quote("HDFCBANK", -1.5),  # Banking
    }
    result = compute_sector_movers(quotes)
    assert "IT" in result
    assert "Banking" in result
    assert {q["symbol"] for q in result["IT"]["gainers"]} == {"TCS", "INFY"}
    assert {q["symbol"] for q in result["Banking"]["losers"]} == {"HDFCBANK"}


def test_gainers_sorted_descending_and_losers_ascending():
    quotes = {
        "TCS": _quote("TCS", 1.0),
        "INFY": _quote("INFY", 3.0),
        "WIPRO": _quote("WIPRO", 2.0),
        "HCLTECH": _quote("HCLTECH", -0.5),
        "TECHM": _quote("TECHM", -3.0),
    }
    result = compute_sector_movers(quotes)
    gainer_symbols = [q["symbol"] for q in result["IT"]["gainers"]]
    loser_symbols = [q["symbol"] for q in result["IT"]["losers"]]
    assert gainer_symbols == ["INFY", "WIPRO", "TCS"]
    assert loser_symbols == ["TECHM", "HCLTECH"]  # most negative first


def test_caps_at_top_n():
    # 7 IT gainers — only the top 5 should survive.
    quotes = {
        sym: _quote(sym, pct)
        for sym, pct in [
            ("TCS", 7.0),
            ("INFY", 6.0),
            ("WIPRO", 5.0),
            ("HCLTECH", 4.0),
            ("TECHM", 3.0),
            ("LTIM", 2.0),
            ("PERSISTENT", 1.0),
        ]
    }
    result = compute_sector_movers(quotes, top_n=5)
    assert len(result["IT"]["gainers"]) == 5
    assert [q["symbol"] for q in result["IT"]["gainers"]] == [
        "TCS",
        "INFY",
        "WIPRO",
        "HCLTECH",
        "TECHM",
    ]


def test_flat_sector_is_omitted():
    quotes = {
        "TCS": _quote("TCS", 0.0),
        "INFY": _quote("INFY", 0.0),
        "HDFCBANK": _quote("HDFCBANK", 1.0),
    }
    result = compute_sector_movers(quotes)
    assert "IT" not in result  # both flat -> no gainers or losers
    assert "Banking" in result


def test_unknown_symbol_falls_back_without_crashing():
    quotes = {"SOMENEWSTOCK": _quote("SOMENEWSTOCK", 4.2)}
    result = compute_sector_movers(quotes)
    assert "Unknown" in result
    assert result["Unknown"]["gainers"][0]["symbol"] == "SOMENEWSTOCK"


def test_empty_quotes_returns_empty_dict():
    assert compute_sector_movers({}) == {}


def test_serialized_fields():
    quotes = {"TCS": _quote("TCS", 2.5)}
    result = compute_sector_movers(quotes)
    entry = result["IT"]["gainers"][0]
    assert set(entry.keys()) == {"symbol", "last_price", "change_percent"}
    assert entry["change_percent"] == 2.5


async def _run_one_cycle(tracker: SectorMoversTracker) -> None:
    task = asyncio.create_task(tracker.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_tracker_reports_success_via_on_status():
    calls: list[tuple[str, str]] = []
    tracker = SectorMoversTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        on_status=lambda msg, level: calls.append((msg, level)),
    )
    tracker._feed.fetch_quotes = MagicMock(return_value={"TCS": _quote("TCS", 1.0)})

    await _run_one_cycle(tracker)

    assert calls
    assert calls[0][1] == "INFO"


async def test_tracker_reports_failure_via_on_status():
    calls: list[tuple[str, str]] = []
    tracker = SectorMoversTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        on_status=lambda msg, level: calls.append((msg, level)),
    )
    tracker._feed.fetch_quotes = MagicMock(side_effect=RuntimeError("boom"))

    await _run_one_cycle(tracker)

    assert calls
    assert calls[0][1] == "WARNING"
    assert "boom" in calls[0][0]


async def test_tracker_accepts_simulated_quote_provider():
    tracker = SectorMoversTracker(
        symbols=["TCS"],
        refresh_seconds=999,
        fetch_quotes=lambda: {"TCS": _quote("TCS", 1.0)},
        data_source="simulated",
    )

    await _run_one_cycle(tracker)

    assert tracker.scan_status == "ready"
    assert tracker.data_source == "simulated"
    assert tracker.sector_movers["IT"]["gainers"][0]["symbol"] == "TCS"
