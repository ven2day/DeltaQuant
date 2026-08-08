"""
₹DeltaQuant Trading State

Shared trading-state schema and mutation methods. This is the single source
of truth for session stats — the web UI serializes it directly
(``src/webui/schema.py``); the live loop mutates it as events happen.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.market_time import now_ist

# In-memory activity log retention (this is the raw buffer TradingStats
# keeps; the web UI's Activity Log panel scrolls through all of it).
ACTIVITY_LOG_MAX = 200

# Real-time plain-text log of every log_activity() call — a dedicated,
# non-propagating logger (not the root logger via logging.basicConfig) so
# this surfaces on its own without also turning on every other module's
# existing logger.debug/info calls throughout the codebase.
_activity_logger = logging.getLogger("deltaquant.activity")
_activity_logger.setLevel(logging.INFO)
_activity_logger.propagate = False
if not _activity_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _activity_logger.addHandler(_handler)

_LEVEL_TO_LOGGING = {
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO,
    "TRADE": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


@dataclass
class TradingStats:
    """Real-time trading statistics."""

    # Session info
    session_start: datetime = field(default_factory=now_ist)
    trading_mode: str = "paper"
    data_source: str = "simulated"

    # True when the trading loop is currently allowed to open NEW positions — the same
    # is_trading_window()/force_trading_window check run_live_trading.py's cycle gate
    # itself uses, not an independently-guessed client-side clock. Exits still run
    # outside this window; this only reflects new-entry eligibility.
    market_open: bool = False
    # TESTING ONLY: mirrors settings.force_trading_window, so the UI can show that the
    # trading-hours gate is deliberately bypassed rather than implying a real market open.
    force_trading_window: bool = False

    # Account
    starting_balance: float = 1000000.0
    current_balance: float = 1000000.0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Open positions
    open_positions: list = field(default_factory=list)

    # Agent activity
    cycles_run: int = 0
    signals_generated: int = 0
    signals_validated: int = 0
    signals_rejected: int = 0
    trades_approved: int = 0
    trades_risk_rejected: int = 0

    # Current regime
    current_regime: str = "unknown"
    regime_confidence: float = 0.0
    active_strategies: list = field(default_factory=list)

    # FinOps (LLM cost tracking, today / IST)
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0

    # Profit-target goal (month-to-date pace)
    goal_enabled: bool = False
    goal_feasible: bool = True
    goal_target_amount: float = 0.0
    goal_mtd_pnl: float = 0.0
    goal_expected_to_date: float = 0.0
    goal_on_pace: bool = True
    goal_status: str = ""

    # Market data
    market_quotes: dict = field(default_factory=dict)  # symbol -> quote dict
    top_movers: list = field(default_factory=list)
    # sector -> {"gainers": [...], "losers": [...]}, refreshed every few
    # minutes across the full discovery universe (see SectorMoversTracker)
    sector_movers: dict = field(default_factory=dict)
    sector_movers_status: str = "disabled"
    sector_movers_data_source: str = ""
    # Ranked list of ScalpingCandidate dicts, refreshed on a long timer
    # (see ScalpingScreenerTracker)
    scalping_candidates: list = field(default_factory=list)
    scalping_screener_status: str = "disabled"
    scalping_screener_data_source: str = ""

    # Decision info
    last_decision_reason: str = ""
    current_signal: dict = field(default_factory=dict)
    candidate_decisions: list = field(default_factory=list)
    chart_symbol: str = ""
    chart_timeframes: dict = field(default_factory=dict)
    simulation_event_time: str = ""
    daily_entry_cap: int = 6
    daily_entries: int = 0

    # Recent activity log
    activity_log: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_percent(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.total_pnl / self.starting_balance) * 100

    def log_activity(self, message: str, level: str = "INFO"):
        timestamp = now_ist().strftime("%H:%M:%S")
        self.activity_log.append(
            {
                "time": timestamp,
                "level": level,
                "message": message,
            }
        )
        if len(self.activity_log) > ACTIVITY_LOG_MAX:
            self.activity_log = self.activity_log[-ACTIVITY_LOG_MAX:]
        _activity_logger.log(_LEVEL_TO_LOGGING.get(level, logging.INFO), f"{level:<8} {message}")


class TradingDashboard:
    """Trading session state manager."""

    def __init__(self):
        self.stats = TradingStats()
        self.running = False

    def start(
        self, balance: float = 1000000.0, mode: str = "paper", data_source: str = "simulated"
    ):
        """Start the session."""
        self.stats.starting_balance = balance
        self.stats.current_balance = balance
        self.stats.trading_mode = mode
        self.stats.data_source = data_source
        self.stats.session_start = now_ist()
        self.running = True

        self.stats.log_activity("Session started", "INFO")
        self.stats.log_activity(f"Mode: {mode.upper()}", "INFO")
        self.stats.log_activity(f"Data: {data_source.upper()}", "INFO")

    def update_regime(self, regime: str, confidence: float, strategies: list):
        """Update market regime info."""
        self.stats.current_regime = regime
        self.stats.regime_confidence = confidence
        self.stats.active_strategies = strategies
        self.stats.log_activity(f"Regime: {regime} ({confidence:.0%})", "INFO")

    def update_market_data(self, quotes: dict):
        """Update market quotes."""
        self.stats.market_quotes = quotes

    def set_current_signal(
        self,
        signal_type: str,
        symbol: str,
        strategy: str,
        confidence: float,
        timeframe: str = "",
        action: str = "BUY",
        rationale: list[str] | None = None,
        target_realistic: bool = True,
    ):
        """Set the current trading signal."""
        self.stats.current_signal = {
            "signal_type": signal_type,
            "symbol": symbol,
            "strategy": strategy,
            "confidence": confidence,
            "timeframe": timeframe,
            "action": action,
            "rationale": rationale or [],
            "target_realistic": target_realistic,
        }

    def set_decision_reason(self, reason: str):
        """Set the decision reasoning."""
        self.stats.last_decision_reason = reason

    def log_signal(
        self, symbol: str, signal_type: str, strategy: str, validated: bool, timeframe: str = ""
    ):
        """Log a signal."""
        tf_tag = f"/{timeframe}" if timeframe else ""
        self.stats.signals_generated += 1
        if validated:
            self.stats.signals_validated += 1
            self.stats.log_activity(f"✓ {signal_type} {symbol} [{strategy}{tf_tag}]", "SUCCESS")
        else:
            self.stats.signals_rejected += 1
            self.stats.log_activity(f"✗ {signal_type} {symbol} rejected", "WARNING")

    def log_trade(self, symbol: str, side: str, qty: int, price: float, approved: bool):
        """Log a trade decision."""
        if approved:
            self.stats.trades_approved += 1
            self.stats.log_activity(f"TRADE: {side} {qty}x {symbol} @ Rs.{price:,.2f}", "TRADE")
        else:
            self.stats.trades_risk_rejected += 1
            self.stats.log_activity(f"BLOCKED: {side} {symbol} by risk", "WARNING")

    def add_position(self, symbol: str, side: str, qty: int, entry_price: float):
        """Add an open position."""
        self.stats.open_positions.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry": entry_price,
                "pnl": 0.0,
            }
        )

    def remove_position(self, symbol: str):
        """Remove one closed position from the display list (first match by symbol).

        Only removes a single entry so pyramided same-symbol positions don't all
        vanish on one partial/full close of one of them.
        """
        for i, pos in enumerate(self.stats.open_positions):
            if pos["symbol"] == symbol:
                del self.stats.open_positions[i]
                break

    def sync_positions(self, positions, managed_positions=()) -> None:
        """Rebuild the display list from the execution engine's actual positions.

        This is the robust alternative to add_position()/remove_position(): those
        require a matching call at every single entry/exit site (easy to miss —
        e.g. positions restored from a persisted wallet on startup were never
        seeded this way, so they'd silently show as "no open positions" despite
        being real and holding real capital). Calling this every dashboard
        refresh instead makes the display a mirror of ground truth, not a
        separately-maintained shadow copy that can drift from it.

        ``positions`` are the paper engine's own ``Position`` objects (source of
        truth for qty/entry/current/pnl). ``managed_positions`` are the exit
        manager's ``ManagedPosition`` objects, joined in by position_id purely to
        surface ``timeframe`` (which the paper engine's Position doesn't carry —
        threading it through place_order()/ExecutionService as well wasn't
        justified just for a display field) alongside entry_time/strategy, which
        both objects already have.
        """
        timeframe_by_id = {
            str(mp.position_id): mp.timeframe for mp in managed_positions if mp.timeframe
        }
        synced = []
        for p in positions:
            item = {
                "symbol": p.symbol,
                "side": p.side,
                "qty": p.quantity,
                "entry": p.entry_price,
                "pnl": p.unrealized_pnl,
            }
            if hasattr(p, "position_id"):
                # Position.entry_time is already an ISO string (see place_order()).
                item.update(
                    {
                        "position_id": p.position_id,
                        "current": p.current_price or p.entry_price,
                        "target": p.target_price,
                        "stop": p.stop_loss,
                        "status": "OPEN",
                        "entry_time": p.entry_time,
                        "strategy": p.strategy,
                        "timeframe": timeframe_by_id.get(str(p.position_id), ""),
                        "entry_data_source": getattr(p, "entry_data_source", "real"),
                    }
                )
            synced.append(item)
        self.stats.open_positions = synced

    def sync_paper_account(self, account: dict) -> None:
        """Mirror the durable paper-wallet aggregates into the session dashboard."""
        self.stats.starting_balance = float(account["initial_balance"])
        self.stats.current_balance = float(account["balance"])
        self.stats.realized_pnl = float(account["realized_pnl"])
        self.stats.unrealized_pnl = float(account["unrealized_pnl"])
        self.stats.total_trades = int(account["total_trades"])
        self.stats.winning_trades = int(account["winning_trades"])
        self.stats.losing_trades = int(account["losing_trades"])

    def close_trade(self, pnl: float):
        """Record a closed trade."""
        self.stats.total_trades += 1
        self.stats.realized_pnl += pnl
        self.stats.current_balance += pnl

        # Track best/worst
        if pnl > self.stats.best_trade:
            self.stats.best_trade = pnl
        if pnl < self.stats.worst_trade:
            self.stats.worst_trade = pnl

        if pnl >= 0:
            self.stats.winning_trades += 1
            self.stats.log_activity(f"Trade closed: +Rs.{pnl:,.2f}", "SUCCESS")
        else:
            self.stats.losing_trades += 1
            self.stats.log_activity(f"Trade closed: Rs.{pnl:,.2f}", "ERROR")

    def increment_cycle(self):
        """Increment cycle counter."""
        self.stats.cycles_run += 1
        self.stats.log_activity(f"Cycle #{self.stats.cycles_run} complete", "INFO")
