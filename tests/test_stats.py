import pytest

from src.dashboard.stats import TradingDashboard, TradingStats

# --- TradingStats Tests ---


@pytest.fixture
def stats():
    return TradingStats()


def test_trading_stats_init(stats):
    assert stats.starting_balance == 1000000.0
    assert stats.total_trades == 0
    assert stats.win_rate == 0.0


def test_trading_stats_pnl(stats):
    stats.realized_pnl = 100
    stats.unrealized_pnl = 50
    assert stats.total_pnl == 150
    assert stats.pnl_percent == 0.015


def test_trading_stats_scalp_fields_default_empty(stats):
    # Mirrors candidate_decisions/scalping_candidates' existing default-empty
    # convention -- a fresh TradingStats (scalp_enabled=False) must show an empty
    # funnel/opportunity list, not a missing attribute.
    assert stats.scalp_opportunities == []
    assert stats.scalp_funnel == {}


def test_trading_stats_scalp_fields_are_independent_instances():
    # Dataclass mutable-default footgun regression: two TradingStats instances must
    # not share the same underlying list/dict object.
    a, b = TradingStats(), TradingStats()
    a.scalp_opportunities.append({"symbol": "RELIANCE"})
    a.scalp_funnel["raw_triggers"] = 5

    assert b.scalp_opportunities == []
    assert b.scalp_funnel == {}


def test_trading_stats_log_activity(stats):
    stats.log_activity("Test message", "INFO")
    assert len(stats.activity_log) == 1
    assert stats.activity_log[0]["message"] == "Test message"

    # Test capping
    for i in range(250):
        stats.log_activity(f"Msg {i}")
    assert len(stats.activity_log) == 200  # Cap size
    assert stats.activity_log[-1]["message"] == "Msg 249"


# --- TradingDashboard Tests ---


@pytest.fixture
def dashboard():
    return TradingDashboard()


def test_dashboard_start(dashboard):
    dashboard.start(balance=500000.0, mode="live", data_source="dhan")
    assert dashboard.stats.starting_balance == 500000.0
    assert dashboard.stats.trading_mode == "live"
    assert dashboard.running is True
    assert len(dashboard.stats.activity_log) == 3


def test_dashboard_update_regime(dashboard):
    dashboard.update_regime("bull", 0.9, ["strat1"])
    assert dashboard.stats.current_regime == "bull"
    assert dashboard.stats.regime_confidence == 0.9
    assert dashboard.stats.active_strategies == ["strat1"]


def test_dashboard_update_market_data(dashboard):
    quotes = {"A": 100}
    dashboard.update_market_data(quotes)
    assert dashboard.stats.market_quotes == quotes


def test_dashboard_set_current_signal(dashboard):
    dashboard.set_current_signal("BUY", "AAPL", "strat1", 0.8)
    assert dashboard.stats.current_signal["symbol"] == "AAPL"


def test_dashboard_set_decision_reason(dashboard):
    dashboard.set_decision_reason("Reason")
    assert dashboard.stats.last_decision_reason == "Reason"


def test_dashboard_log_signal(dashboard):
    dashboard.log_signal("AAPL", "BUY", "strat1", True)
    assert dashboard.stats.signals_generated == 1
    assert dashboard.stats.signals_validated == 1

    dashboard.log_signal("AAPL", "BUY", "strat1", False)
    assert dashboard.stats.signals_generated == 2
    assert dashboard.stats.signals_rejected == 1


def test_dashboard_log_trade(dashboard):
    dashboard.log_trade("AAPL", "BUY", 10, 100, True)
    assert dashboard.stats.trades_approved == 1

    dashboard.log_trade("AAPL", "BUY", 10, 100, False)
    assert dashboard.stats.trades_risk_rejected == 1


def test_dashboard_add_position(dashboard):
    dashboard.add_position("AAPL", "BUY", 10, 100)
    assert len(dashboard.stats.open_positions) == 1
    assert dashboard.stats.open_positions[0]["symbol"] == "AAPL"


def test_dashboard_remove_position(dashboard):
    dashboard.add_position("AAPL", "BUY", 10, 100)
    dashboard.add_position("MSFT", "BUY", 5, 200)
    dashboard.remove_position("AAPL")
    assert len(dashboard.stats.open_positions) == 1
    assert dashboard.stats.open_positions[0]["symbol"] == "MSFT"


def test_dashboard_remove_position_only_removes_one_match(dashboard):
    # Pyramided same-symbol positions: closing one shouldn't wipe both.
    dashboard.add_position("AAPL", "BUY", 10, 100)
    dashboard.add_position("AAPL", "BUY", 5, 110)
    dashboard.remove_position("AAPL")
    assert len(dashboard.stats.open_positions) == 1


def test_dashboard_remove_position_missing_symbol_is_noop(dashboard):
    dashboard.add_position("AAPL", "BUY", 10, 100)
    dashboard.remove_position("MSFT")
    assert len(dashboard.stats.open_positions) == 1


class _FakePosition:
    def __init__(self, symbol, side, quantity, entry_price, unrealized_pnl=0.0):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.unrealized_pnl = unrealized_pnl


def test_dashboard_sync_positions_mirrors_engine_state(dashboard):
    # Simulates positions restored from a persisted wallet on startup — never
    # seeded via add_position(), so only sync_positions() can surface them.
    dashboard.add_position("STALE", "BUY", 1, 1.0)
    positions = [
        _FakePosition("TCS", "BUY", 10, 4000.0, unrealized_pnl=500.0),
        _FakePosition("INFY", "SELL", 5, 1800.0, unrealized_pnl=-50.0),
    ]
    dashboard.sync_positions(positions)

    assert len(dashboard.stats.open_positions) == 2
    assert dashboard.stats.open_positions[0] == {
        "symbol": "TCS",
        "side": "BUY",
        "qty": 10,
        "entry": 4000.0,
        "pnl": 500.0,
    }
    # The stale entry from add_position() is gone — sync_positions replaces
    # the list wholesale rather than merging with it.
    assert all(p["symbol"] != "STALE" for p in dashboard.stats.open_positions)


def test_dashboard_sync_positions_empty_clears_display(dashboard):
    dashboard.add_position("AAPL", "BUY", 10, 100)
    dashboard.sync_positions([])
    assert dashboard.stats.open_positions == []


class _FakeManagedPosition:
    """Just enough shape of exit_manager.ManagedPosition for the sync_positions join."""

    def __init__(self, position_id, timeframe):
        self.position_id = position_id
        self.timeframe = timeframe


class _FakePaperPosition(_FakePosition):
    """Adds the fields only present on paper_engine.Position (gated by hasattr(p,
    "position_id") in sync_positions), matching the real object's shape."""

    def __init__(
        self, symbol, side, quantity, entry_price, position_id, entry_time, strategy, **kwargs
    ):
        super().__init__(symbol, side, quantity, entry_price, **kwargs)
        self.position_id = position_id
        self.entry_time = entry_time
        self.strategy = strategy
        self.current_price = entry_price
        self.target_price = entry_price * 1.02
        self.stop_loss = entry_price * 0.98


def test_dashboard_sync_positions_joins_timeframe_from_managed_positions(dashboard):
    # Position.entry_time is already an ISO string (see paper_engine.place_order());
    # timeframe lives only on the exit manager's ManagedPosition and must be joined
    # in by position_id rather than expected directly on the paper engine's Position.
    positions = [
        _FakePaperPosition(
            "TCS", "BUY", 10, 4000.0, position_id="POS-1", entry_time="2026-08-08T09:15:00", strategy="momentum"
        )
    ]
    managed = [_FakeManagedPosition(position_id="POS-1", timeframe="15m")]

    dashboard.sync_positions(positions, managed)

    assert dashboard.stats.open_positions[0]["entry_time"] == "2026-08-08T09:15:00"
    assert dashboard.stats.open_positions[0]["strategy"] == "momentum"
    assert dashboard.stats.open_positions[0]["timeframe"] == "15m"


def test_dashboard_sync_positions_timeframe_blank_without_managed_match(dashboard):
    positions = [
        _FakePaperPosition(
            "TCS", "BUY", 10, 4000.0, position_id="POS-1", entry_time="2026-08-08T09:15:00", strategy="momentum"
        )
    ]
    # No managed_positions passed at all -- must not raise, timeframe defaults blank.
    dashboard.sync_positions(positions)
    assert dashboard.stats.open_positions[0]["timeframe"] == ""


def test_dashboard_sync_paper_account_mirrors_durable_wallet(dashboard):
    dashboard.start(balance=1_000_000.0)
    dashboard.sync_paper_account(
        {
            "initial_balance": 1_000_000.0,
            "balance": 999_145.89,
            "realized_pnl": -854.11,
            "unrealized_pnl": 0.0,
            "total_trades": 6,
            "winning_trades": 0,
            "losing_trades": 6,
        }
    )

    assert dashboard.stats.current_balance == 999_145.89
    assert dashboard.stats.realized_pnl == -854.11
    assert dashboard.stats.total_trades == 6
    assert dashboard.stats.winning_trades == 0
    assert dashboard.stats.losing_trades == 6


def test_dashboard_close_trade(dashboard):
    dashboard.start()
    dashboard.close_trade(100.0)
    assert dashboard.stats.total_trades == 1
    assert dashboard.stats.winning_trades == 1
    assert dashboard.stats.realized_pnl == 100.0
    assert dashboard.stats.current_balance == 1000100.0

    dashboard.close_trade(-50.0)
    assert dashboard.stats.losing_trades == 1
    assert dashboard.stats.realized_pnl == 50.0


def test_dashboard_increment_cycle(dashboard):
    dashboard.increment_cycle()
    assert dashboard.stats.cycles_run == 1


def test_set_cycle_stage_updates_label_and_number(dashboard):
    dashboard.set_cycle_stage("scanning", "Scanning 50 symbols", cycle=7)

    assert dashboard.stats.cycle_stage == "scanning"
    assert dashboard.stats.cycle_stage_label == "Scanning 50 symbols"
    assert dashboard.stats.current_cycle_number == 7
    assert dashboard.stats.cycle_stage_started_at != ""


def test_set_cycle_stage_clears_next_cycle_countdown_when_not_waiting(dashboard):
    dashboard.set_next_cycle_countdown(120)
    assert dashboard.stats.next_cycle_at != ""

    dashboard.set_cycle_stage("scanning", "Scanning symbols")
    assert dashboard.stats.next_cycle_at == ""


def test_set_next_cycle_countdown_sets_waiting_stage(dashboard):
    dashboard.set_next_cycle_countdown(90)

    assert dashboard.stats.cycle_stage == "waiting"
    assert dashboard.stats.cycle_stage_label == "Next cycle in 90s"
    assert dashboard.stats.next_cycle_at != ""


def test_trading_stats_per_cycle_agent_detail_defaults(stats):
    """New fields (what each agent decided this cycle) must default to empty/falsy so
    a fresh session renders as 'nothing yet', not a crash or stale placeholder."""
    assert stats.regime_reasoning == ""
    assert stats.strategy_reasoning == ""
    assert stats.news_headlines == []
    assert stats.news_sentiment == 0.0
    assert stats.market_mood == {}
    assert stats.prediction_signals == []
    assert stats.agent_fallback_notice == ""


def test_stats_to_dict_serializes_per_cycle_agent_detail():
    from src.webui.schema import stats_to_dict

    stats = TradingStats()
    stats.regime_reasoning = "Trending up on strong breadth"
    stats.news_headlines = [{"title": "Sensex rallies", "sentiment": "positive"}]
    stats.market_mood = {"mood_index": 72, "mood_label": "greed"}
    stats.agent_fallback_notice = "Signal Validation: the AI reviewer hit its daily usage limit"

    data = stats_to_dict(stats)

    assert data["regime_reasoning"] == "Trending up on strong breadth"
    assert data["news_headlines"] == [{"title": "Sensex rallies", "sentiment": "positive"}]
    assert data["market_mood"]["mood_index"] == 72
    assert "daily usage limit" in data["agent_fallback_notice"]
