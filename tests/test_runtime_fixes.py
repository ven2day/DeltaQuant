"""
Tests for the runtime/usability fixes:

- MarketDataManager.refresh() advances data each cycle (was: fetched once + frozen).
- HistoryManager never substitutes synthetic prices for failed DhanHQ data in live mode.
"""

from src.market.history_manager import HistoryManager
from src.market.indicators import Timeframe, calculate_indicators
from src.market.manager import MarketDataManager
from src.market.signals import SignalEngine


def test_manager_refresh_advances_simulated():
    manager = MarketDataManager(symbols=["RELIANCE", "TCS", "SBIN"])
    manager.is_live = False
    manager.data_source = "simulated"
    manager._load_simulated_quotes()

    before = {s: q.change_percent for s, q in manager.get_all_quotes().items()}
    manager.refresh()
    after = {s: q.change_percent for s, q in manager.get_all_quotes().items()}

    assert before  # there were quotes
    assert before != after  # refresh advanced the simulated movement


def test_history_manager_seeds_synthetic():
    hm = HistoryManager(symbols=[])
    hm._seed_synthetic("DEMO", bars=150)
    df = hm.get_history("DEMO")
    assert df is not None
    assert len(df) == 150
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_prefetch_leaves_symbol_unavailable_on_failure(monkeypatch):
    hm = HistoryManager(symbols=["AAA", "BBB"])
    # Simulate DhanHQ being unavailable.
    monkeypatch.setattr(hm, "fetch_history", lambda *a, **k: False)
    results = hm.prefetch_all()
    assert not any(results.values())
    assert not hm.has_sufficient_data("AAA")


def test_prefetch_leaves_symbol_unavailable_when_history_feed_is_disabled(monkeypatch):
    hm = HistoryManager(symbols=["AAA", "BBB"])
    monkeypatch.setattr(type(hm._feed), "is_available", property(lambda _: False))
    monkeypatch.setattr(hm, "fetch_history", lambda *args, **kwargs: (_ for _ in ()).throw())

    results = hm.prefetch_all()

    assert not any(results.values())
    assert not hm.has_sufficient_data("AAA")


def test_prefetch_seeds_synthetic_only_when_explicitly_allowed(monkeypatch):
    hm = HistoryManager(symbols=["AAA", "BBB"], allow_synthetic=True)
    monkeypatch.setattr(type(hm._feed), "is_available", property(lambda _: False))

    results = hm.prefetch_all()

    assert all(results.values())
    assert hm.has_sufficient_data("AAA")
    assert hm.has_sufficient_data("BBB")


def test_synthetic_history_drives_indicators_and_signals():
    # End-to-end: synthetic history -> real indicators -> SignalEngine produces output
    # without raising (proves the agent pipeline can run offline).
    hm = HistoryManager(symbols=[])
    hm._seed_synthetic("DEMO", bars=180)
    df = hm.get_history("DEMO", bars=200, include_forming=False)
    indicators = calculate_indicators(df, "DEMO", timeframe=Timeframe.D1)
    assert indicators.rsi is not None
    # Should not raise; may or may not produce signals depending on the random walk.
    signals = SignalEngine().generate_signals(indicators)
    assert isinstance(signals, list)
