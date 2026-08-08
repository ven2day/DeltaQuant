"""
₹DeltaQuant Live Trading Dashboard

Enhanced with:
- Real historical indicator calculation (replaces fabricated indicators)
- Real trade execution via paper engine (replaces random P&L)
- Dynamic exit management (trailing stops, time exits, partial profits)
- Performance tracking for strategy learning
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Force UTF-8 stdout/stderr: Windows defaults to the system codepage (cp1252) whenever
# stdout isn't a live console (redirected to a file/pipe, e.g. under nohup or `| Tee-Object`),
# which can't encode ₹ or other non-ASCII characters used in banners/log messages and
# crashes the whole process with an unhandled UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.agents.graph import create_trading_graph, run_trading_cycle
from src.agents.prediction import PredictionAgent
from src.agents.risk_compliance import check_kill_switch, compute_return_correlations
from src.config import get_settings
from src.dashboard.stats import TradingDashboard
from src.execution.averaging import CappedAveragingPolicy
from src.execution.exit_manager import ExitManager
from src.execution.journal import TradeJournal
from src.execution.lifecycle import (
    MOCK_NAMESPACE,
    PAPER_DATA_NAMESPACE,
    PaperTradeLifecycleStore,
)
from src.execution.paper_engine import LocalPaperEngine
from src.execution.service import ExecutionService, IdempotencyStore
from src.execution.signal_log import SignalLogger, SignalRecord
from src.finops import get_alert_manager, get_cost_tracker
from src.market.candidate_policy import CandidateAction, evaluate_long_candidate
from src.market.history_manager import HistoryManager
from src.market.indicators import Timeframe, get_indicator_cache
from src.market.manager import MarketDataManager, is_market_open
from src.market.signal_ranking import rank_signals, select_diversified_signals
from src.market.signals import SignalEngine
from src.market.sizing import calculate_position_size
from src.market.stock_discovery import StockDiscovery
from src.market.websocket_feed import NSE_WATCHLIST, refresh_watchlist
from src.memory import feedback
from src.memory.classifier import MistakeClassifier
from src.memory.database import AgentMemoryDB
from src.memory.injection import MemoryInjector
from src.memory.performance_tracker import get_performance_tracker
from src.notifications.telegram import get_notifier
from src.observability.tracing import setup_tracing
from src.profit import ProfitGoalEngine
from src.risk.daily_state import DailyRiskStore
from src.risk.guards import DrawdownTracker, is_circuit_locked
from src.risk.pretrade import FinalPaperOrder, PaperRiskReservations
from src.signal_discovery.live import DiscoveredSignalScorer
from src.signal_discovery.operators import build_ohlcv_panel
from src.signal_discovery.scheduler import AutomaticDiscoveryScheduler
from src.signal_discovery.store import DiscoveredSignalStore
from src.utils.formatting import fmt_optional
from src.utils.market_time import is_trading_window

console = Console()
logger = logging.getLogger(__name__)

# Safety cap on the kill-switch flatten loop: each pass closes at least the FIFO-first
# position of every open symbol, so this far exceeds the passes any realistic book needs.
MAX_FLATTEN_PASSES = 20


_TIMEFRAME_BY_VALUE = {tf.value: tf for tf in Timeframe}


def parse_signal_timeframes(raw: str) -> list[Timeframe]:
    """Parse a comma-separated ``signal_timeframes`` setting into Timeframe enums.

    Unknown tokens are ignored (logged); falls back to Daily if nothing parses so the
    pipeline always has at least one timeframe to run.
    """
    timeframes = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        tf = _TIMEFRAME_BY_VALUE.get(token)
        if tf is None:
            logger.warning(f"Unknown signal timeframe '{token}' in signal_timeframes setting")
            continue
        timeframes.append(tf)
    return timeframes or [Timeframe.D1]


def calculate_real_indicators(
    history_manager: HistoryManager, symbol: str, timeframe: Timeframe = Timeframe.D1
):
    """
    Calculate REAL indicators from historical data for a given candle timeframe.

    When ``signals_exclude_forming_bar`` is set, the still-forming bar is excluded on
    every timeframe to prevent repainting. Results are memoized via the shared
    IndicatorCache (keyed by symbol + timeframe) so unchanged data is not recomputed.
    """
    settings = get_settings()

    if timeframe == Timeframe.D1:
        include_forming = not settings.signals_exclude_forming_bar
        df = history_manager.get_history(symbol, bars=200, include_forming=include_forming)
    else:
        df = history_manager.get_multi_timeframe_history(symbol, timeframe, bars=200)
        if settings.signals_exclude_forming_bar and df is not None and len(df) > 1:
            df = df.iloc[:-1]

    if df is None or len(df) < 26:  # Need at least MACD slow period
        return None

    try:
        return get_indicator_cache().get_or_compute(df, symbol, timeframe=timeframe)
    except Exception as e:
        logger.warning(f"Indicator calc failed for {symbol} [{timeframe.value}]: {e}")
        return None


async def run_live_trading():
    """Run the permanent local-paper workflow with live or simulated market data."""

    settings = get_settings()
    if (
        settings.trading_mode != "paper"
        or settings.execution_mode != "local_paper"
        or settings.allow_live_orders
    ):
        raise RuntimeError(
            "DeltaQuant is permanently paper-only: require TRADING_MODE=paper, "
            "EXECUTION_MODE=local_paper, and ALLOW_LIVE_ORDERS=false."
        )
    simulated_session = (
        settings.enable_synthetic_history
        and not settings.enable_dhan_historical_data
        and not settings.enable_dhan_quotes
    )
    data_namespace = MOCK_NAMESPACE if simulated_session else PAPER_DATA_NAMESPACE
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Initialize dashboard
    data_source = (
        "live"
        if is_market_open()
        else "dhan_rest"
        if settings.market_data_source == "dhan" and settings.enable_dhan_quotes
        else "simulated"
    )
    dashboard = TradingDashboard()
    dashboard.start(
        balance=settings.paper_wallet_balance, mode=settings.trading_mode, data_source=data_source
    )

    # Setup tracing
    tracing_enabled = setup_tracing()
    dashboard.stats.log_activity(
        f"Langfuse: {'enabled' if tracing_enabled else 'disabled'}", "INFO"
    )

    # Create trading graph (now includes support agents)
    graph = create_trading_graph(include_support_agents=settings.enable_news_analysis)
    dashboard.stats.log_activity("Trading graph compiled (with support agents)", "SUCCESS")

    # Initialize memory & performance tracker
    memory_db = AgentMemoryDB(namespace=data_namespace)
    perf_tracker = get_performance_tracker(data_namespace)
    dashboard.stats.log_activity("Memory database + performance tracker ready", "INFO")

    # Learning feedback loop: classify closed trades into lessons and measure whether the
    # lessons that were injected actually helped. Optional (adds an LLM call per loss).
    mistake_classifier = MistakeClassifier() if settings.enable_learning else None
    memory_injector = MemoryInjector(memory_db=memory_db) if settings.enable_learning else None
    # Lesson IDs that were in the agents' context when each position was opened.
    dashboard.stats.log_activity(
        f"Learning loop: {'enabled' if settings.enable_learning else 'disabled'}", "INFO"
    )

    # Initialize paper trading engine
    paper_engine = LocalPaperEngine(initial_balance=settings.paper_wallet_balance)
    dashboard.sync_paper_account(paper_engine.get_stats())
    dashboard.stats.log_activity(
        f"Paper engine: Rs. {paper_engine.get_balance():,.0f} balance", "INFO"
    )

    # Persistent history of every signal considered (approved/rejected), independent of
    # actual fills (TradeJournal) — powers the web UI's 7-day signal history view. Live and
    # backfill records share one Postgres table (src.execution.signal_log), distinguished
    # by the `source` column, so read_recent() already returns both together.
    signal_logger = SignalLogger()

    # Durable trade journal (records every entry/exit). Persists to DATABASE_URL; falls back
    # to a non-persistent in-memory DB (logged loudly) if that is unreachable.
    journal = TradeJournal(namespace=data_namespace)
    lifecycle_store = PaperTradeLifecycleStore()
    averaging_policy = CappedAveragingPolicy(
        enabled=settings.paper_averaging_enabled,
        max_adds=settings.paper_averaging_max_adds,
        trigger_pct=settings.paper_averaging_trigger_pct,
        add_fraction=settings.paper_averaging_add_fraction,
    )
    dashboard.stats.log_activity(
        f"Trade journal: {'persistent' if journal.is_persistent else 'in-memory (NOT durable)'}",
        "INFO" if journal.is_persistent else "WARNING",
    )

    # Unified execution service: one mode-switched path for order submission with
    # idempotency (no double-submit on retry/restart) and shadow-mode safety.
    execution_service = ExecutionService.from_settings(
        engine=paper_engine,
        idempotency=IdempotencyStore(),
    )
    effective = execution_service.effective_mode.value
    real_orders_label = "YES" if execution_service.real_orders_active else "NO"
    dashboard.stats.log_activity(
        f"Execution mode: {effective} | REAL ORDERS: {real_orders_label}"
        + (" (SHADOW — mirrors live, sends no real orders)" if effective == "shadow" else ""),
        "WARNING" if execution_service.real_orders_active else "INFO",
    )

    # Live modes only: attach the broker executor and reconcile against the broker at
    # startup (broker = source of truth). Dormant by default — effective mode is shadow
    # unless allow_live_orders=True and Dhan creds are present.
    # Initialize exit manager
    exit_manager = ExitManager(
        trailing_atr_multiplier=1.5,
        breakeven_r_threshold=1.0,
        max_hold_minutes=240,
        partial_profit_r=1.0,
        partial_exit_pct=0.5,
        state_file=Path("exit_manager_state.json"),
    )
    reconciliation = lifecycle_store.reconcile(paper_engine.get_positions(), exit_manager)
    dashboard.stats.log_activity(
        f"Paper lifecycle reconciliation: {reconciliation.summary()}",
        "INFO" if reconciliation.in_sync else "WARNING",
    )

    # Profit-target goal engine: derive a risk-bounded plan from the configured target.
    # Advisory only — it never changes position sizing or relaxes risk limits.
    goal_engine = ProfitGoalEngine()
    goal_plan = goal_engine.build_plan(paper_engine.get_balance())
    dashboard.stats.goal_enabled = goal_plan.enabled
    dashboard.stats.goal_feasible = goal_plan.feasible
    dashboard.stats.goal_target_amount = goal_plan.monthly_target_amount
    if goal_plan.enabled:
        wr = goal_plan.required_win_rate
        wr_str = "n/a" if wr == float("inf") else f"{wr:.0%}"
        dashboard.stats.log_activity(
            f"Profit goal: Rs.{goal_plan.monthly_target_amount:,.0f}/mo "
            f"(needs ~{wr_str} win rate @ {goal_plan.expected_trades_per_day}/day; "
            f"{'feasible' if goal_plan.feasible else 'NOT feasible within risk'})",
            "INFO" if goal_plan.feasible else "WARNING",
        )
        for warning in goal_plan.warnings:
            dashboard.stats.log_activity(f"Goal: {warning}", "WARNING")
    else:
        dashboard.stats.log_activity("Profit goal: disabled (no target configured)", "INFO")

    # Intraday drawdown tracker (includes unrealized P&L) — feeds the kill switch's drawdown
    # limb, which was previously dead because the loop fed a hardcoded max_drawdown of 0.
    # Seed from the engine's actual equity (it may have RESTORED a persisted wallet/positions
    # on startup), not the static configured balance — otherwise peak/current are wrong on a
    # restart and the drawdown halt mis-fires (or never fires).
    starting_equity = paper_engine.get_total_value()
    drawdown_tracker = DrawdownTracker(
        peak_equity=starting_equity,
        current_equity=starting_equity,
    )

    # Daily trade count / P&L for the risk engine's daily limits — persisted to Postgres and
    # scoped to the IST calendar day (NOT reset by a restart, unlike the tracker above, which
    # is deliberately session-scoped). A plain in-memory counter would let the daily trade cap
    # and daily loss limit reset every time the process restarts, defeating the point of a
    # *daily* cap.
    daily_risk_store = DailyRiskStore()

    # Signal engine
    signal_engine = SignalEngine(
        mean_reversion_stop_loss_pct=settings.mean_reversion_stop_loss_pct,
        mean_reversion_target_pct=settings.mean_reversion_target_pct,
    )
    prediction_agent = PredictionAgent()
    discovery_timeframes = parse_signal_timeframes(settings.signal_discovery_timeframes)
    discovered_signal_scorers: dict[Timeframe, DiscoveredSignalScorer] = {}
    if settings.signal_discovery_enabled:
        for timeframe in discovery_timeframes:
            discovered_signal_scorers[timeframe] = DiscoveredSignalScorer(
                DiscoveredSignalStore(
                    Path(settings.signal_discovery_output_dir) / timeframe.value,
                    ic_threshold=settings.signal_discovery_ic_threshold,
                    p_value_threshold=settings.signal_discovery_p_value_threshold,
                ),
                max_probability_tilt=settings.signal_discovery_max_probability_tilt,
                min_cross_section=settings.signal_discovery_min_cross_section,
            )
    discovery_scheduler = (
        AutomaticDiscoveryScheduler(dashboard.stats.log_activity)
        if settings.signal_discovery_enabled and settings.signal_discovery_auto_run
        else None
    )
    signal_timeframes = parse_signal_timeframes(settings.signal_timeframes)
    if simulated_session:
        signal_timeframes = [
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
            Timeframe.H4,
        ]
    dashboard.stats.log_activity(
        f"Signal timeframes: {', '.join(tf.value for tf in signal_timeframes)}", "INFO"
    )
    if settings.signal_discovery_enabled:
        mode = (
            f"automatic every {settings.signal_discovery_refresh_hours:g}h"
            if discovery_scheduler is not None
            else "load-only"
        )
        dashboard.stats.log_activity(
            f"Signal discovery: {', '.join(tf.value for tf in discovery_timeframes)} ({mode})",
            "INFO",
        )

    # The configured CSV is the trading universe. Do not run a second full-universe
    # quote scan for ranking before the live Dhan feed starts: it consumes the same
    # account-wide quote budget and can make the initial live poll fail.
    dashboard.stats.log_activity("Loading configured stock universe...", "INFO")
    discovery = StockDiscovery(max_stocks=settings.max_active_stocks)
    trading_symbols = discovery.universe
    dashboard.stats.log_activity(f"Loaded {len(trading_symbols)} symbols from the universe", "INFO")

    # Resolve Dhan security IDs for the discovered universe before anything tries to
    # subscribe/quote against it — a stale/small hardcoded watchlist here is exactly
    # what silently drops non-hardcoded symbols from live Dhan data (see the unmapped-
    # symbols check below). Network fetch, so keep it off the event loop.
    if settings.market_data_source == "dhan" and settings.enable_dhan_instrument_lookup:
        loop = asyncio.get_running_loop()
        resolved = await loop.run_in_executor(None, refresh_watchlist, trading_symbols)
        dashboard.stats.log_activity(
            f"Resolved {len(resolved)}/{len(trading_symbols)} discovered symbols against "
            "DhanHQ's instrument master",
            "INFO",
        )

    # Initialize history manager and pre-fetch data
    dashboard.stats.log_activity("Pre-fetching historical data for real indicators...", "INFO")
    synthetic_history_allowed = (
        settings.enable_synthetic_history
        and settings.force_trading_window
        and settings.trading_mode == "paper"
        and effective == "local_paper"
        and not settings.allow_live_orders
        and not settings.enable_dhan_historical_data
        and not settings.enable_dhan_quotes
    )
    if settings.enable_synthetic_history and not synthetic_history_allowed:
        dashboard.stats.log_activity(
            "Synthetic history requested but blocked by the paper-only testing safety gate",
            "WARNING",
        )
    elif synthetic_history_allowed:
        dashboard.stats.log_activity(
            "TESTING ONLY: synthetic daily history enabled (Dhan quote/history calls disabled)",
            "WARNING",
        )
    market_manager = MarketDataManager(symbols=trading_symbols)
    history_manager = HistoryManager(
        symbols=trading_symbols,
        lookback_period="3mo",
        allow_synthetic=synthetic_history_allowed,
        simulated_stream=(market_manager.simulated_data if synthetic_history_allowed else None),
    )
    fetch_results = history_manager.prefetch_all()
    loaded = sum(1 for v in fetch_results.values() if v)
    history_kind = "synthetic demo" if synthetic_history_allowed else "real"
    dashboard.stats.log_activity(
        f"Historical data loaded: {loaded}/{len(trading_symbols)} symbols ({history_kind})",
        "SUCCESS",
    )

    console.print("\n[bold green]₹DeltaQuant Live Trading System Starting...[/]")
    # "LIVE"/"SIMULATED" here describes the PRICE FEED only (real-time NSE data vs a demo
    # feed when the market's closed) — it says nothing about money. `effective` (computed
    # above from execution_mode/allow_live_orders) is what actually gates whether any order
    # can touch a real account, so spell that out explicitly to avoid the two being confused.
    data_mode = (
        "LIVE"
        if is_market_open()
        else "DHAN REST (last available quote)"
        if settings.market_data_source == "dhan" and settings.enable_dhan_quotes
        else "SIMULATED"
    )
    money_mode = "REAL ORDERS" if effective == "live" else "PAPER (no real orders)"
    console.print(
        f"[dim]Data feed: {data_mode} | Execution: {money_mode} | "
        f"Stocks: {len(trading_symbols)} | Press Ctrl+C to stop[/]\n"
    )
    if settings.force_trading_window:
        console.print(
            "[bold red]TESTING MODE: FORCE_TRADING_WINDOW=true — trading cycles will run "
            "regardless of market hours or weekday.[/]\n"
        )
        dashboard.stats.log_activity(
            "TESTING MODE: FORCE_TRADING_WINDOW=true — market-hours/weekday check bypassed",
            "WARNING",
        )
    await asyncio.sleep(1)

    # Start market data
    is_live = await market_manager.start()
    dashboard.stats.data_source = market_manager.data_source
    dashboard.stats.log_activity(
        f"Data source: {market_manager.data_source}"
        + ("" if is_live else " (polled/simulated — refreshed each cycle)"),
        "SUCCESS",
    )

    # The Dhan live feed only knows how to subscribe to its own hardcoded symbol table
    # (`NSE_WATCHLIST`, ~20 large-caps) — completely decoupled from StockDiscovery's much
    # wider universe. Any discovered symbol outside that table is silently dropped from the
    # subscription (subscribe() still reports success even with zero instruments), so a
    # discovery run with poor overlap looks identical to a broken feed. Surface it loudly.
    if market_manager.data_source == "dhan":
        unmapped = [s for s in trading_symbols if s not in NSE_WATCHLIST]
        if unmapped:
            dashboard.stats.log_activity(
                f"No live Dhan quotes for {len(unmapped)}/{len(trading_symbols)} discovered "
                f"symbols (not in the live feed's watchlist): {', '.join(unmapped)}",
                "WARNING",
            )
        if len(unmapped) == len(trading_symbols):
            dashboard.stats.log_activity(
                "ZERO discovered symbols are covered by the live WebSocket feed — quotes "
                "for this cycle will come from REST-polled DhanHQ quotes instead "
                "(refresh()/QuotesFeed), or restart to re-roll discovery.",
                "ERROR",
            )

    # A live WebSocket connection only opens the socket and subscribes — quotes are
    # actually delivered by reading it in `listen()`. Without pumping that loop
    # concurrently with the trading cycle below, `market_manager.quotes` never
    # populates and every cycle sees "no candidates" forever. Run it as a background
    # task; it handles its own errors/reconnect-signalling internally and never raises.
    listen_task: asyncio.Task | None = None
    if is_live:
        listen_task = asyncio.create_task(market_manager.listen())
        dashboard.stats.log_activity("WebSocket listener started", "INFO")

    # Optional live web dashboard (FastAPI+WebSocket), mirroring the CLI panels in a
    # browser. Strictly additive: the terminal dashboard behaves identically whether
    # this is on or off. Imported lazily so a plain `uv sync` (no `web` extra) never
    # hits a missing-module error for users who haven't opted in.
    hub = None
    web_server = None
    web_task: asyncio.Task | None = None
    if settings.enable_web_ui:
        from src.api.health import health_check
        from src.webui.schema import stats_to_dict
        from src.webui.server import ConnectionHub, WebUIServer

        async def _get_health_dict(full: bool) -> dict[str, Any]:
            return (await health_check(include_slow_checks=full)).to_dict()

        hub = ConnectionHub()
        web_server = WebUIServer(
            hub=hub,
            get_snapshot=lambda: {"type": "state", "data": stats_to_dict(dashboard.stats)},
            get_signals=signal_logger.read_recent,
            get_health=lambda full: _get_health_dict(full),
            host=settings.web_ui_host,
            port=settings.web_ui_port,
            cors_origins=settings.web_ui_cors_origins.split(","),
        )
        web_task = asyncio.create_task(web_server.serve())
        dashboard.stats.log_activity(
            f"Web UI: http://{settings.web_ui_host}:{settings.web_ui_port}", "INFO"
        )

    # Sector-wide top gainers/losers: scans the FULL discovery universe (~70 NIFTY50+
    # midcap symbols), not just the ~15 symbols actively picked for trading. Runs on its
    # own multi-minute timer in a thread executor (the scan is blocking) so it never
    # competes with the fast trading-signal loop for the event loop.
    sector_tracker = None
    sector_task: asyncio.Task | None = None
    if settings.enable_sector_movers:
        from src.market.sector_movers import SectorMoversTracker

        sector_tracker = SectorMoversTracker(
            symbols=discovery.universe,
            refresh_seconds=settings.sector_movers_refresh_seconds,
            on_status=dashboard.stats.log_activity,
            fetch_quotes=(
                market_manager.get_all_quotes if market_manager.data_source == "simulated" else None
            ),
            data_source=market_manager.data_source,
        )
        sector_task = asyncio.create_task(sector_tracker.run())
        dashboard.stats.log_activity(
            f"Sector movers scan started ({len(discovery.universe)} symbols, "
            f"every {settings.sector_movers_refresh_seconds}s)",
            "INFO",
        )

    # Scalping candidates: ranks the discovery universe by how often each symbol swings
    # by settings.scalping_swing_threshold_pct (% of the symbol's own price) or more
    # within a day (zigzag swing count, averaged over the last several sessions) —
    # separate from the trading-signal loop, informational only. Refreshes on a much
    # longer timer than sector movers since each scan pulls ~10 days of 5-minute candles
    # per symbol.
    scalping_tracker = None
    scalping_task: asyncio.Task | None = None
    if settings.enable_scalping_screener:
        from src.market.scalping_screener import ScalpingScreenerTracker

        scalping_tracker = ScalpingScreenerTracker(
            symbols=discovery.universe,
            refresh_seconds=settings.scalping_screener_refresh_seconds,
            threshold_pct=settings.scalping_swing_threshold_pct,
            lookback_days=settings.scalping_screener_lookback_days,
            on_status=dashboard.stats.log_activity,
            feed=(
                market_manager.simulated_data if market_manager.data_source == "simulated" else None
            ),
            data_source=market_manager.data_source,
        )
        scalping_task = asyncio.create_task(scalping_tracker.run())
        dashboard.stats.log_activity(
            f"Scalping screener started ({len(discovery.universe)} symbols, "
            f"{settings.scalping_swing_threshold_pct:.2f}%+ swings, "
            f"every {settings.scalping_screener_refresh_seconds}s)",
            "INFO",
        )

    # Telegram startup notification (best-effort; no-op if unconfigured)
    notifier = get_notifier()
    if notifier.enabled:
        try:
            await notifier.send_startup_message()
            dashboard.stats.log_activity("Telegram notifications active", "INFO")
        except Exception as e:
            logger.warning("Telegram startup notification failed: %s", e)

    async def refresh_dashboard() -> None:
        """Single choke point for every dashboard state refresh.

        Updates sector movers from the background tracker and, when the web
        UI is enabled, broadcasts the current state to any connected
        WebSocket clients. Plain-text progress is printed separately, in
        real time, by TradingStats.log_activity() itself.
        """
        if sector_tracker is not None:
            dashboard.stats.sector_movers = sector_tracker.sector_movers
            dashboard.stats.sector_movers_status = sector_tracker.scan_status
            dashboard.stats.sector_movers_data_source = sector_tracker.data_source
        if scalping_tracker is not None:
            dashboard.stats.scalping_candidates = scalping_tracker.candidates
            dashboard.stats.scalping_screener_status = scalping_tracker.scan_status
            dashboard.stats.scalping_screener_data_source = scalping_tracker.data_source
        # Mirror the exact gate the cycle loop itself uses (below), not an
        # independently-computed guess — the Header badge and this must never disagree
        # about whether the system is actually willing to open new positions right now.
        dashboard.stats.market_open = settings.force_trading_window or is_trading_window(
            settings.no_trading_before, settings.no_trading_after
        )
        dashboard.stats.force_trading_window = settings.force_trading_window
        today = daily_risk_store.get_today()
        dashboard.stats.daily_entry_cap = settings.paper_daily_entry_cap
        dashboard.stats.daily_entries = int(today["trades_count"])
        if market_manager.data_source == "simulated":
            dashboard.stats.simulation_event_time = (
                market_manager.simulated_data.event_time.isoformat()
            )
        dashboard.sync_paper_account(paper_engine.get_stats())
        dashboard.sync_positions(paper_engine.get_positions())
        if hub is not None:
            await hub.broadcast({"type": "state", "data": stats_to_dict(dashboard.stats)})

    async def _log_signals_async(records: list[SignalRecord]) -> None:
        """Write a batch of signal records without blocking the event loop.

        signal_logger.log() is a synchronous Postgres round-trip; a cycle can produce
        dozens of these (one per signal per timeframe/strategy), so writing them one at a
        time directly on the loop would stall the WebSocket broadcast and market polling
        for the sum of all those round-trips. Running the whole batch in one executor call
        keeps the loop free while they write.
        """
        if not records:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: [signal_logger.log(r) for r in records])

    def _publish_chart(
        symbol: str,
        *,
        entry: float,
        current: float,
        target: float,
        stop: float,
        status: str,
    ) -> None:
        """Publish compact multi-timeframe chart data from the canonical feed."""
        chart: dict[str, Any] = {}
        for timeframe in (
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
            Timeframe.H4,
        ):
            frame = history_manager.get_multi_timeframe_history(symbol, timeframe, 90)
            if frame is None or frame.empty:
                continue
            points = []
            for timestamp, row in frame.tail(90).iterrows():
                points.append(
                    {
                        "time": timestamp.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )
            chart[timeframe.value] = {
                "points": points,
                "entry": entry,
                "current": current,
                "target": target,
                "stop": stop,
                "status": status,
            }
        dashboard.stats.chart_symbol = symbol
        dashboard.stats.chart_timeframes = chart

    async def _execute_managed_exit(
        pos: Any,
        *,
        quantity: int,
        price: float,
        reason: str,
        explanation: str,
    ) -> bool:
        """Execute and atomically fan out one canonical local-paper exit fill."""
        if quantity <= 0:
            return False
        event_key = (
            market_manager.simulated_data.sequence
            if market_manager.data_source == "simulated"
            else datetime.now().strftime("%Y%m%d%H%M%S")
        )
        result = await execution_service.submit_async(
            symbol=pos.symbol,
            side="SELL",
            quantity=quantity,
            price=price,
            idempotency_key=(
                f"{data_namespace}:{pos.position_id}:exit:{reason}:{event_key}:{quantity}"
            ),
            trade_id=pos.position_id,
            position_id=pos.position_id,
            reason=reason,
        )
        if not result.filled:
            dashboard.stats.log_activity(
                f"EXIT {result.status}: {pos.symbol} — {result.message}", "WARNING"
            )
            return False

        fully_closed = lifecycle_store.record_exit(
            pos.position_id,
            order_id=result.order_id,
            quantity=quantity,
            exit_price=result.fill_price,
            realized_pnl=result.realized_pnl,
            exit_charges=result.exit_charges,
            reason=reason,
        )
        daily_risk_store.record_exit_pnl(result.realized_pnl)
        exit_manager.apply_exit_fill(pos.position_id, quantity)
        dashboard.stats.log_activity(
            f"EXIT [{reason}]: {pos.symbol} {quantity} @ Rs.{result.fill_price:,.2f} "
            f"P&L: Rs.{result.realized_pnl:+,.2f} — {explanation}",
            "TRADE",
        )
        if not fully_closed:
            return True

        lifecycle = lifecycle_store.get(pos.position_id) or {}
        cumulative_pnl = float(lifecycle.get("cumulative_pnl", result.realized_pnl))
        original_quantity = int(lifecycle.get("original_quantity", quantity))
        weighted_entry = float(lifecycle.get("weighted_entry_price", pos.entry_price))
        notional = weighted_entry * original_quantity
        pnl_pct = cumulative_pnl / notional * 100 if notional else 0.0
        perf_tracker.record_trade(
            strategy=pos.strategy,
            regime=pos.regime_at_entry,
            pnl=cumulative_pnl,
            pnl_pct=pnl_pct,
            symbol=pos.symbol,
        )
        try:
            journal.close_trade(
                pos.position_id,
                result.fill_price,
                reason,
                mae=pos.mae,
                mfe=pos.mfe,
                pnl=cumulative_pnl,
                exit_quantity=original_quantity,
            )
        except Exception as exc:
            logger.warning("Journal close_trade failed for %s: %s", pos.position_id, exc)
        if settings.enable_learning and mistake_classifier and memory_injector:
            outcome = feedback.build_outcome(
                trade_id=pos.position_id,
                symbol=pos.symbol,
                strategy=pos.strategy,
                regime=pos.regime_at_entry,
                side=pos.side,
                entry_price=weighted_entry,
                exit_price=result.fill_price,
                stop_loss=pos.stop_loss,
                target_price=pos.target_price,
                pnl=cumulative_pnl,
                pnl_pct=pnl_pct,
                mae=pos.mae,
                mfe=pos.mfe,
                hold_minutes=max(0, int((datetime.now() - pos.entry_time).total_seconds() / 60)),
            )
            mistake = feedback.learn_from_outcome(memory_injector, mistake_classifier, outcome)
            if mistake:
                dashboard.stats.log_activity(
                    f"Lesson learned: [{mistake.severity}] {mistake.category}", "INFO"
                )
            feedback.mark_lessons_outcome(
                memory_db,
                lifecycle.get("active_lessons", []),
                was_successful=cumulative_pnl > 0,
            )
        return True

    last_review_fingerprint: tuple[str, ...] | None = None

    async def _run_cycle(cycle: int) -> None:
        """
        Run a single trading cycle.

        Raises on any unexpected failure; the caller catches it so one bad
        cycle never tears down the whole loop. Early ``return`` is used for
        benign "nothing to do" outcomes (no candidates / signals / history).
        """
        nonlocal last_review_fingerprint

        # Advance one authoritative 5-minute event before any exit, signal, or chart
        # reads simulated state. Quotes and all aggregate timeframes stay synchronized.
        if not is_live:
            market_manager.refresh()
            if market_manager.data_source == "simulated":
                history_manager.sync_simulated_stream()

        # ── Step 0: Check exits on existing positions ───────────
        quotes = market_manager.get_all_quotes()
        market_prices = {s: q.last_price for s, q in quotes.items()}

        # Calculate ATR for exit manager
        atr_values = {}
        for symbol in market_prices:
            ind = calculate_real_indicators(history_manager, symbol)
            if ind and ind.atr:
                atr_values[symbol] = ind.atr

        # Check all managed positions for exits
        current_regime = (
            dashboard.stats.current_regime if hasattr(dashboard.stats, "current_regime") else ""
        )
        exit_signals = exit_manager.check_exits(market_prices, current_regime, atr_values)

        for pos, exit_rule in exit_signals:
            exit_price = market_prices.get(pos.symbol, pos.entry_price)
            exit_qty = int(pos.quantity * exit_rule.partial_pct)
            await _execute_managed_exit(
                pos,
                quantity=exit_qty,
                price=exit_price,
                reason=exit_rule.exit_type,
                explanation=exit_rule.reason,
            )

        # Separate capped averaging experiment. It is disabled by default, can only add
        # to a still-valid long above its stop, consumes a daily entry slot, and is
        # bounded by both the position and aggregate exposure limits.
        if averaging_policy.enabled:
            equity = paper_engine.get_total_value()
            for pos in list(exit_manager.get_managed_positions()):
                lifecycle = lifecycle_store.get(pos.position_id)
                current = market_prices.get(pos.symbol)
                if lifecycle is None or current is None:
                    continue
                add = averaging_policy.evaluate(
                    original_quantity=int(lifecycle["original_quantity"]),
                    add_count=int(lifecycle["add_count"]),
                    weighted_entry_price=float(lifecycle["weighted_entry_price"]),
                    current_price=current,
                    stop_loss=float(lifecycle["stop_loss"]),
                    target_price=float(lifecycle["target_price"]),
                )
                if not add.should_add:
                    continue
                daily = daily_risk_store.get_today()
                expected_fill = paper_engine.cost_model.fill_price(current, "BUY")
                current_exposure = sum(
                    position.quantity * (position.current_price or position.entry_price)
                    for position in paper_engine.get_positions()
                )
                projected_total_pct = (
                    (current_exposure + add.quantity * expected_fill) / equity * 100
                    if equity > 0
                    else 100.0
                )
                projected_position_pct = (
                    (pos.quantity * pos.entry_price + add.quantity * expected_fill) / equity
                    if equity > 0
                    else 1.0
                )
                if (
                    int(daily["trades_count"]) >= settings.paper_daily_entry_cap
                    or projected_total_pct > settings.max_total_exposure_pct
                    or projected_position_pct > settings.max_position_pct
                ):
                    dashboard.stats.log_activity(
                        f"AVERAGING BLOCKED: {pos.symbol} failed cap/exposure controls",
                        "WARNING",
                    )
                    continue
                idempotency_key = (
                    f"{data_namespace}:{pos.position_id}:add:{int(lifecycle['add_count']) + 1}"
                )
                result = await execution_service.submit_async(
                    symbol=pos.symbol,
                    side="BUY",
                    quantity=add.quantity,
                    price=current,
                    idempotency_key=idempotency_key,
                    trade_id=pos.position_id,
                    position_id=pos.position_id,
                    stop_loss=float(lifecycle["stop_loss"]),
                    target_price=float(lifecycle["target_price"]),
                    strategy=pos.strategy,
                    reason="capped_average",
                )
                if result.filled:
                    lifecycle_store.record_add(
                        pos.position_id,
                        order_id=result.order_id,
                        quantity=add.quantity,
                        fill_price=result.fill_price,
                        entry_charges=result.entry_charges,
                    )
                    exit_manager.apply_add_fill(pos.position_id, add.quantity, result.fill_price)
                    daily_risk_store.record_entry()
                    dashboard.stats.log_activity(
                        f"CAPPED ADD: {pos.symbol} +{add.quantity} @ "
                        f"Rs.{result.fill_price:,.2f} — {add.reason}",
                        "TRADE",
                    )

        await refresh_dashboard()

        # ── Step 1: Refresh market data ────────────────────────
        # Pull fresh quotes for the active source every cycle. (Previously yfinance was
        # fetched once at startup and frozen; only the websocket push updated live.)
        quotes = market_manager.get_all_quotes()
        dashboard.update_market_data({s: q.to_dict() for s, q in quotes.items()})

        # Append new quotes to history for rolling indicator updates
        if market_manager.data_source != "simulated":
            for symbol, quote in quotes.items():
                history_manager.append_quote(
                    symbol=symbol,
                    open_price=quote.open,
                    high=quote.high,
                    low=quote.low,
                    close=quote.last_price,
                    volume=quote.volume,
                )

        # ── Data freshness gate ────────────────────────────────
        # If the feed has stalled (even the freshest quote is too old), skip NEW
        # entries this cycle — trading on stale prices is unsafe. Exits in Step 0
        # already ran on the last known price.
        max_stale = settings.max_quote_staleness_seconds
        if max_stale > 0 and quotes and market_manager.data_source != "simulated":
            freshest_age = min(q.age_seconds for q in quotes.values())
            if freshest_age > max_stale:
                await get_alert_manager().alert(
                    "data_staleness",
                    f"Market data stale: freshest quote is {freshest_age:.0f}s old "
                    f"(limit {max_stale}s). Skipping new entries this cycle.",
                    level="WARNING",
                )
                dashboard.stats.log_activity(
                    f"Stale data ({freshest_age:.0f}s old) — skipping trading this cycle",
                    "WARNING",
                )
                for _ in range(15):
                    await asyncio.sleep(1)
                    await refresh_dashboard()
                return

        # ── Step 2: Find trading candidates ────────────────────
        # Run every configured strategy/timeframe across every quoted universe symbol.
        # This is local computation plus broker history; no LLM tokens are spent here.
        scan_symbols = [symbol for symbol in trading_symbols if symbol in quotes]
        indicators_by_symbol: dict[str, dict[Timeframe, Any]] = {}
        raw_signals = []
        for index, symbol in enumerate(scan_symbols, start=1):
            indicators_by_tf: dict[Timeframe, Any] = {}
            for tf in signal_timeframes:
                # Broker history is synchronous. Run it off the event loop so quotes,
                # exits, and the web dashboard remain responsive during a bulk scan.
                ind = await asyncio.to_thread(
                    calculate_real_indicators, history_manager, symbol, tf
                )
                if ind is not None:
                    indicators_by_tf[tf] = ind

            if not indicators_by_tf:
                continue

            symbol_signals = []
            for ind in indicators_by_tf.values():
                symbol_signals.extend(signal_engine.generate_signals(ind))
            if settings.long_only:
                symbol_signals = [
                    signal for signal in symbol_signals if signal.signal_type.value == "BUY"
                ]
            if symbol_signals:
                indicators_by_symbol[symbol] = indicators_by_tf
                raw_signals.extend(symbol_signals)

            if index % 25 == 0:
                dashboard.stats.log_activity(
                    f"Strategy scan: {index}/{len(scan_symbols)} symbols, "
                    f"{len(raw_signals)} raw signals",
                    "INFO",
                )
                await refresh_dashboard()

        if discovery_scheduler is not None and discovery_scheduler.maybe_start(
            history_manager, trading_symbols
        ):
            dashboard.stats.log_activity(
                "Automatic signal-discovery research queued in the background",
                "INFO",
            )

        if not raw_signals:
            dashboard.stats.log_activity(
                f"Full-universe scan: {len(scan_symbols)} symbols, no strategy signals",
                "INFO",
            )
            for _ in range(15):
                await asyncio.sleep(1)
                await refresh_dashboard()
            return

        # The sklearn ensemble is CPU-only and runs only on symbol/timeframe pairs where
        # a deterministic strategy fired. Its confidence is directional agreement, not
        # historical accuracy; the ranker keeps those concepts separate.
        predictions = {}
        prediction_keys = {(signal.symbol, signal.timeframe) for signal in raw_signals}
        for symbol, timeframe in prediction_keys:
            frame = await asyncio.to_thread(
                history_manager.get_multi_timeframe_history, symbol, timeframe, 200
            )
            if frame is None or frame.empty:
                continue
            if settings.signals_exclude_forming_bar and len(frame) > 1:
                frame = frame.iloc[:-1]
            ml_frame = frame.rename(columns=lambda column: str(column).title())
            predictions[(symbol, timeframe.value)] = await asyncio.to_thread(
                prediction_agent.predict, ml_frame, symbol
            )

        discovered_tilts: dict[tuple[str, str], float] = {}
        for discovery_timeframe, scorer in discovered_signal_scorers.items():
            if not scorer.store.load_accepted():
                continue
            discovery_histories = {}
            for symbol in scan_symbols:
                frame = await asyncio.to_thread(
                    history_manager.get_multi_timeframe_history,
                    symbol,
                    discovery_timeframe,
                    500,
                )
                if frame is None or frame.empty:
                    continue
                if settings.signals_exclude_forming_bar and len(frame) > 1:
                    frame = frame.iloc[:-1]
                discovery_histories[symbol] = frame
            if not discovery_histories:
                continue
            timeframe_tilts = await asyncio.to_thread(
                scorer.score_panel,
                build_ohlcv_panel(discovery_histories),
            )
            discovered_tilts.update(
                {
                    (symbol, discovery_timeframe.value): tilt
                    for symbol, tilt in timeframe_tilts.items()
                }
            )
            if timeframe_tilts:
                dashboard.stats.log_activity(
                    f"Discovered {discovery_timeframe.value} alpha tilted "
                    f"{len(timeframe_tilts)} symbols",
                    "INFO",
                )

        # Shared with the graph's prediction_node (support_agents) so it doesn't
        # re-fetch history and re-run the ensemble for symbols already scored here.
        precomputed_predictions_map = {
            f"{symbol}|{timeframe}": pred.to_dict()
            for (symbol, timeframe), pred in predictions.items()
        }

        ranked = rank_signals(
            raw_signals,
            predictions,
            perf_tracker,
            discovered_signal_tilts=discovered_tilts,
        )
        candidate_decisions = {}
        qualifying_ranked = []
        for item in ranked:
            signal = item.signal
            histories = {
                timeframe: history_manager.get_multi_timeframe_history(
                    signal.symbol, timeframe, 240
                )
                for timeframe in (
                    Timeframe.M5,
                    Timeframe.M15,
                    Timeframe.M30,
                    Timeframe.H1,
                    Timeframe.H4,
                )
            }
            decision = evaluate_long_candidate(
                signal.to_dict(),
                histories,
                predictions.get((signal.symbol, signal.timeframe.value)),
                target_pct=settings.paper_target_pct,
                min_ml_confidence=settings.paper_min_ml_confidence,
                min_volume_ratio=settings.paper_min_volume_ratio,
                required_higher_timeframes=settings.paper_required_higher_timeframes,
            )
            candidate_decisions[signal.signal_id] = decision
            # The framework target is the exact target subsequently reviewed, risked,
            # submitted, managed, journaled, and displayed.
            signal.target_price = decision.target_price
            signal.risk_reward_ratio = (
                (decision.target_price - decision.entry_price)
                / (decision.entry_price - decision.stop_loss)
                if decision.entry_price > decision.stop_loss > 0
                else 0.0
            )
            if decision.action == CandidateAction.BUY:
                qualifying_ranked.append(item)

        dashboard.stats.candidate_decisions = [
            candidate_decisions[item.signal.signal_id].to_dict() for item in ranked[:25]
        ]
        if settings.llm_review_all_signals:
            survivors = qualifying_ranked
        else:
            survivors = select_diversified_signals(
                qualifying_ranked,
                max_symbols=settings.max_active_stocks,
                max_per_sector=settings.universe_max_per_sector,
                max_signals_per_symbol=settings.max_signals_per_symbol,
            )
        survivor_symbols = list(dict.fromkeys(item.signal.symbol for item in survivors))
        review_symbols = (
            survivor_symbols
            if settings.llm_review_all_signals
            else survivor_symbols[: settings.llm_review_max_symbols]
        )
        reviewed_symbol_set = set(review_symbols)
        reviewed_ranked = [item for item in survivors if item.signal.symbol in reviewed_symbol_set]
        signals = [item.signal for item in reviewed_ranked]

        display_item = reviewed_ranked[0] if reviewed_ranked else (ranked[0] if ranked else None)
        if display_item is not None:
            display_signal = display_item.signal
            display_decision = candidate_decisions[display_signal.signal_id]
            quote = quotes.get(display_signal.symbol)
            current = quote.last_price if quote is not None else display_decision.entry_price
            dashboard.set_current_signal(
                display_signal.signal_type.value,
                display_signal.symbol,
                display_signal.strategy.value,
                display_signal.confidence,
                timeframe=display_signal.timeframe.value,
                action=display_decision.action.value,
                rationale=display_decision.rationale,
                target_realistic=display_decision.target_realistic,
            )
            dashboard.set_decision_reason(" ".join(display_decision.rationale))
            _publish_chart(
                display_signal.symbol,
                entry=display_decision.entry_price,
                current=current,
                target=display_decision.target_price,
                stop=display_decision.stop_loss,
                status=display_decision.action.value,
            )

        dashboard.stats.log_activity(
            f"Ranked {len(raw_signals)} signals from {len(scan_symbols)} symbols; "
            f"{len(signals)} signals across {len(review_symbols)} symbols sent to AI",
            "SUCCESS",
        )
        best_by_symbol = {}
        for item in survivors:
            best_by_symbol.setdefault(item.signal.symbol, item)
        for position, symbol in enumerate(survivor_symbols, start=1):
            item = best_by_symbol[symbol]
            dashboard.stats.log_activity(
                f"Rank {position}: {symbol} "
                f"[{item.signal.strategy.value}/{item.signal.timeframe.value}] "
                f"p={item.estimated_win_probability:.1%} "
                f"ER={item.expected_r_multiple:+.2f} "
                f"history={item.historical_sample_size} ({item.accuracy_status})",
                "INFO",
            )

        indicators_dict: dict[str, dict[str, Any]] = {}
        for symbol in review_symbols:
            indicators_by_tf = indicators_by_symbol[symbol]
            primary_indicators = next(
                (indicators_by_tf[tf] for tf in signal_timeframes if tf in indicators_by_tf),
                next(iter(indicators_by_tf.values())),
            )
            indicators_dict[symbol] = primary_indicators.to_dict()

        if signals:
            sig = signals[0]
            decision = candidate_decisions[sig.signal_id]
            dashboard.set_current_signal(
                sig.signal_type.value,
                sig.symbol,
                sig.strategy.value,
                sig.confidence,
                timeframe=sig.timeframe.value,
                action=decision.action.value,
                rationale=decision.rationale,
                target_realistic=decision.target_realistic,
            )
            primary_quote = quotes[sig.symbol]
            primary_indicators = indicators_by_symbol[sig.symbol][sig.timeframe]
            direction = "bullish" if primary_quote.is_bullish else "bearish"
            # RSI/ADX can be None during the indicator warm-up window (a signal may come
            # from a strategy that doesn't need them), so format defensively — a bare
            # f"{None:.1f}" raises TypeError and kills the cycle (#17).
            reason = (
                f"{direction.title()} momentum ({primary_quote.change_percent:+.2f}%) "
                f"with {sig.strategy.value} strategy on {sig.timeframe.value} "
                f"(RSI: {fmt_optional(primary_indicators.rsi, '.1f')}, "
                f"ADX: {fmt_optional(primary_indicators.adx, '.1f')})"
            )
            dashboard.set_decision_reason(reason + " " + " ".join(decision.rationale))

        dashboard.stats.signals_generated += len(raw_signals)
        for signal in signals:
            dashboard.stats.log_activity(
                f"Signal: {signal.signal_type.value} {signal.symbol} "
                f"[{signal.strategy.value}/{signal.timeframe.value}] "
                f"conf={signal.confidence:.0%}",
                "INFO",
            )
        await refresh_dashboard()

        if not signals:
            waits = sum(
                decision.action == CandidateAction.WAIT for decision in candidate_decisions.values()
            )
            rejects = sum(
                decision.action == CandidateAction.REJECT
                for decision in candidate_decisions.values()
            )
            dashboard.stats.log_activity(
                f"No qualifying BUY candidates (WAIT {waits}, REJECT {rejects}); "
                "no trade was forced",
                "INFO",
            )
            for _ in range(15):
                await asyncio.sleep(1)
                await refresh_dashboard()
            return

        review_fingerprint = tuple(
            f"{item.signal.symbol}|{item.signal.strategy.value}|"
            f"{item.signal.timeframe.value}|{item.signal.entry_price:.4f}|"
            f"{item.estimated_win_probability:.4f}"
            for item in reviewed_ranked
        )
        if review_fingerprint == last_review_fingerprint:
            dashboard.stats.log_activity(
                "Ranked signal set unchanged; skipping duplicate Groq/news review", "INFO"
            )
            # Keep the normal no-op cycle cadence. Without this cooldown, the outer
            # loop immediately rescans the same unchanged set and busy-loops.
            for _ in range(15):
                await asyncio.sleep(1)
                await refresh_dashboard()
            return

        # ── FinOps spend gate ──────────────────────────────────
        # The agent pipeline below is the only LLM spend. If the daily token/cost
        # budget is exhausted, skip it (and any new entries) to conserve spend —
        # exits in Step 0 already ran, so risk is still managed.
        cost_tracker = get_cost_tracker()
        if cost_tracker.is_over_hard_budget():
            status = cost_tracker.budget_status()
            await get_alert_manager().alert(
                "finops_hard_budget",
                f"Daily LLM budget exhausted ({status['tokens_used']:,} tokens / "
                f"${status['cost_used_usd']:.4f} used) — pausing new agent cycles.",
                level="CRITICAL",
            )
            dashboard.stats.log_activity(
                "FinOps HARD budget reached — skipping agent pipeline this cycle", "WARNING"
            )
            for _ in range(15):
                await asyncio.sleep(1)
                await refresh_dashboard()
            return

        # ── Step 5: Run agent pipeline ─────────────────────────
        # The full universe is already covered by the deterministic screen. Keep
        # the LLM context to the reviewed names instead of dumping hundreds of
        # raw quotes into a prompt.
        reviewed_symbols = set(indicators_dict)
        market_data = {s: quotes[s].to_dict() for s in reviewed_symbols}

        market_breadth = sum(1 for quote in quotes.values() if quote.is_bullish) / max(
            len(quotes), 1
        )
        memory_lessons = memory_db.get_top_lessons_for_context(
            regime="trending_up" if market_breadth >= 0.5 else "trending_down",
            strategies=["momentum", "trend_following"],
            n=5,
        )

        workflow_id = f"LIVE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{cycle}"

        # Mark positions to market and update intraday drawdown (incl. unrealized) BEFORE
        # the kill-switch check, so a deep unrealized drawdown actually halts trading.
        paper_engine.update_positions_pnl(market_prices)
        drawdown_tracker.update(paper_engine.get_total_value())
        daily_risk_store.update_drawdown(drawdown_tracker.max_drawdown)

        # trades_count/profit_loss come from Postgres (today's IST-scoped row), not the
        # in-memory dashboard counters — those reset on every restart, which would silently
        # reset the daily trade limit and daily loss limit along with them.
        daily_stats = daily_risk_store.get_today()
        daily_stats["simulation_mode"] = simulated_session
        portfolio = {
            "capital": paper_engine.get_balance(),
            "positions": [p.to_dict() for p in paper_engine.get_positions()],
        }

        # Real return-correlation, not just a sector-membership proxy: two names in
        # different sectors can still move together. Only worth computing over the
        # symbols that could actually interact with an open-position risk check —
        # existing holdings plus this cycle's reviewed candidates.
        correlation_symbols = {p["symbol"] for p in portfolio["positions"]} | {
            s.symbol for s in signals
        }
        if len(correlation_symbols) > 1:
            price_histories = {
                symbol: frame["close"]
                for symbol in correlation_symbols
                if (
                    frame := history_manager.get_history(
                        symbol,
                        bars=settings.pairwise_correlation_lookback_days,
                        include_forming=False,
                    )
                )
                is not None
            }
            portfolio["return_correlations"] = compute_return_correlations(price_histories)

        signal_payloads = []
        for signal in signals:
            payload = signal.to_dict()
            decision = candidate_decisions[signal.signal_id]
            payload["candidate_action"] = decision.action.value
            payload["candidate_rationale"] = decision.rationale
            payload["target_realistic"] = decision.target_realistic
            payload["volume_ratio"] = decision.volume_ratio
            payload["vwap"] = decision.vwap
            payload["higher_timeframes_aligned"] = decision.higher_timeframes_aligned
            signal_payloads.append(payload)

        final_state = await run_trading_cycle(
            graph=graph,
            market_data=market_data,
            indicators=indicators_dict,
            signals=signal_payloads,
            memory_lessons=memory_lessons,
            portfolio=portfolio,
            daily_stats=daily_stats,
            thread_id=workflow_id,
            precomputed_predictions=precomputed_predictions_map,
        )
        last_review_fingerprint = review_fingerprint

        # ── Step 6: Process results ────────────────────────────
        regime = final_state.get("regime", "unknown")
        confidence = final_state.get("regime_confidence", 0)
        strategies = final_state.get("active_strategies", [])
        dashboard.update_regime(regime, confidence, strategies)
        if hasattr(dashboard.stats, "current_regime"):
            dashboard.stats.current_regime = regime
        await refresh_dashboard()

        validated = final_state.get("validated_signals", [])
        rejected = final_state.get("rejected_signals", [])
        dashboard.stats.signals_validated += len(validated)
        dashboard.stats.signals_rejected += len(rejected)

        for sig in validated:
            dashboard.stats.log_activity(
                f"VALIDATED: {sig.get('signal_type')} {sig.get('symbol')} "
                f"[{sig.get('timeframe', 'N/A')}]",
                "SUCCESS",
            )
        rejected_records = []
        for sig in rejected:
            dashboard.stats.log_activity(
                f"REJECTED: {sig.get('signal_type')} {sig.get('symbol')} "
                f"[{sig.get('timeframe', 'N/A')}]",
                "WARNING",
            )
            rejected_records.append(SignalRecord.from_signal(sig, "rejected_validation"))
        await _log_signals_async(rejected_records)
        await refresh_dashboard()

        # ── Step 7: Execute approved trades via paper engine ───
        approved = final_state.get("approved_trades", [])
        risk_rejected = final_state.get("risk_rejected", [])
        agent_fallbacks = [
            error for error in final_state.get("errors", []) if "fallback" in str(error).lower()
        ]
        degraded_blocked = []
        if agent_fallbacks:
            # The deterministic screen can shortlist candidates, but it must not
            # silently replace the requested LLM review. A degraded agent cycle
            # is analysis-only and can never create a new position.
            degraded_blocked = list(approved)
            approved = []
            if degraded_blocked:
                dashboard.stats.log_activity(
                    f"AGENT DEGRADED - blocking {len(degraded_blocked)} new entry "
                    f"{'signal' if len(degraded_blocked) == 1 else 'signals'}: "
                    f"{agent_fallbacks[0]}",
                    "WARNING",
                )
        long_only_blocked = []
        if settings.long_only:
            long_only_blocked = [
                sig for sig in approved if sig.get("signal_type", "").upper() == "SELL"
            ]
            approved = [sig for sig in approved if sig not in long_only_blocked]
        dashboard.stats.trades_approved += len(approved)
        dashboard.stats.trades_risk_rejected += len(risk_rejected)

        # Surface WHY entries were blocked so a run with signals-but-no-trades is not a
        # mystery (#19). The most common off-hours cause is the deterministic
        # trading-hours guard (09:15–15:15 IST) — every entry is blocked when the NSE is
        # closed, which is intentional, not a bug.
        outcome_records = []
        for sig in risk_rejected:
            failures = sig.get("risk_result", {}).get("failures", [])
            reason = failures[0].get("message") if failures else "risk rules"
            dashboard.stats.log_activity(
                f"RISK BLOCKED: {sig.get('signal_type')} {sig.get('symbol')} "
                f"[{sig.get('timeframe', 'N/A')}] — {reason}",
                "WARNING",
            )
            outcome_records.append(SignalRecord.from_signal(sig, "rejected_risk"))
        for sig in long_only_blocked:
            dashboard.stats.log_activity(
                f"LONG-ONLY BLOCKED: SELL {sig.get('symbol')} [{sig.get('timeframe', 'N/A')}]",
                "INFO",
            )
            outcome_records.append(
                SignalRecord.from_signal(
                    sig,
                    "rejected_risk",
                    reason="Long-only investing mode does not open short positions",
                )
            )
        for sig in degraded_blocked:
            outcome_records.append(
                SignalRecord.from_signal(
                    sig,
                    "rejected_risk",
                    reason="LLM agent review degraded; new entries fail closed",
                )
            )
        for sig in approved:
            outcome_records.append(SignalRecord.from_signal(sig, "approved"))
        await _log_signals_async(outcome_records)

        # Kill-switch gate: a daily-loss / drawdown breach must halt NEW entries
        # at the point of execution (exits in Step 0 still run, to flatten risk).
        # Previously the kill switch only ended the agent graph at the regime edge
        # and never re-checked here, so a mid-cycle breach still placed orders.
        # Use peak equity as the capital base so the drawdown % is true (the graph's
        # `portfolio.capital` is cash-only and would distort it).
        kill_state = {
            "daily_stats": daily_stats,
            "portfolio": {"capital": max(drawdown_tracker.peak_equity, 1.0)},
        }
        if check_kill_switch(kill_state):
            if approved:
                dashboard.stats.log_activity(
                    f"KILL SWITCH ACTIVE — blocking {len(approved)} approved "
                    f"{'entry' if len(approved) == 1 else 'entries'} "
                    f"(loss/drawdown {drawdown_tracker.drawdown_pct:.1f}%)",
                    "WARNING",
                )
                approved = []

            # Stop the bleed: flatten all open positions (not just block new entries).
            # place_order() closes by (symbol, side) FIFO, not by position id, so a single
            # snapshot pass may not fully flatten multiple/odd-sized same-symbol positions.
            # Loop until the engine reports flat (capped to avoid spinning), and clear the
            # exit-manager / open-trade tracking ONLY once it actually is — otherwise we'd
            # lose sight of still-open risk.
            if settings.kill_switch_flatten and paper_engine.get_positions():
                for pos in list(exit_manager.get_managed_positions()):
                    px = market_prices.get(pos.symbol, pos.entry_price)
                    await _execute_managed_exit(
                        pos,
                        quantity=pos.quantity,
                        price=px,
                        reason="kill_switch",
                        explanation="deterministic daily-loss/drawdown kill switch",
                    )
                dashboard.stats.current_balance = paper_engine.get_balance()
                if paper_engine.get_positions():
                    # Could not fully flatten — keep risk tracking intact and escalate
                    # rather than clearing (which would hide the residual exposure).
                    remaining = len(paper_engine.get_positions())
                    dashboard.stats.log_activity(
                        f"FLATTEN INCOMPLETE — {remaining} position(s) still open; "
                        "keeping risk tracking.",
                        "ERROR",
                    )
                    await get_alert_manager().alert(
                        "kill_switch_flatten_incomplete",
                        f"Kill switch fired but {remaining} position(s) could not be "
                        "flattened — risk tracking retained.",
                        level="CRITICAL",
                    )
                else:
                    await get_alert_manager().alert(
                        "kill_switch_flatten",
                        f"Kill switch fired (drawdown {drawdown_tracker.drawdown_pct:.1f}%) — "
                        "flattened all open positions.",
                        level="CRITICAL",
                    )

        reservations = PaperRiskReservations()
        for trade in approved:
            symbol = trade.get("symbol", "N/A")
            side = trade.get("signal_type", "BUY").upper()

            # Cash-investing mode does not treat a bearish signal as permission to
            # short. Existing long holdings are exited only by ExitManager rules.
            if settings.long_only and side == "SELL":
                dashboard.stats.log_activity(
                    f"LONG-ONLY: ignoring new SELL signal for {symbol}", "INFO"
                )
                continue

            # Circuit guard: don't open into a scrip at/through its NSE price band — near a
            # circuit you can't get a sane fill (limit-up: no sellers; limit-down: no buyers).
            if settings.circuit_guard_enabled:
                q = quotes.get(symbol)
                if q is not None and is_circuit_locked(
                    q.change_percent, settings.default_circuit_band_pct
                ):
                    dashboard.stats.log_activity(
                        f"CIRCUIT LOCKED: skipping {symbol} ({q.change_percent:+.1f}%)",
                        "WARNING",
                    )
                    continue

            quoted_price = float(market_prices.get(symbol, 0.0) or 0.0)
            reviewed_entry = float(trade.get("entry_price", 0.0) or 0.0)
            stop_loss = float(trade.get("stop_loss", 0.0) or 0.0)
            target_price = float(trade.get("target_price", 0.0) or 0.0)
            strategy = str(trade.get("strategy", "unknown"))
            if reviewed_entry <= 0 or quoted_price <= 0:
                dashboard.stats.log_activity(
                    f"WAIT: {symbol} has no exact reviewed/current price", "WARNING"
                )
                continue
            if abs(quoted_price - reviewed_entry) / reviewed_entry > 0.001:
                dashboard.stats.log_activity(
                    f"WAIT: {symbol} moved after review; exact approval is stale", "WARNING"
                )
                continue
            requested_price = reviewed_entry
            expected_fill = paper_engine.cost_model.fill_price(requested_price, side)

            # Risk-based position sizing: size from the risk budget (risk-per-trade and the
            # stop distance) using the real PositionSizer — and the strategy's actual
            # win-rate (Kelly) when enough history exists — instead of a flat 5% of cash.
            if expected_fill > 0 and 0 < stop_loss < expected_fill < target_price:
                win_rate = perf_tracker.get_strategy_performance(strategy, regime).win_rate
                sizing = calculate_position_size(
                    capital=paper_engine.get_total_value(),
                    entry_price=expected_fill,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    risk_per_trade=settings.risk_per_trade,
                    max_position_pct=settings.max_position_pct,
                    win_rate=win_rate,
                )
                quantity = sizing.shares
            else:
                quantity = 0

            final_order = FinalPaperOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=expected_fill,
                stop_loss=stop_loss,
                target_price=target_price,
                strategy=strategy,
            )
            current_daily = daily_risk_store.get_today()
            final_risk = reservations.evaluate(
                final_order,
                positions=paper_engine.get_positions(),
                equity=paper_engine.get_total_value(),
                entries_today=int(current_daily["trades_count"]),
                daily_entry_cap=min(settings.max_daily_trades, settings.paper_daily_entry_cap),
                max_positions=settings.max_concurrent_positions,
                max_position_pct=settings.max_position_pct,
                max_total_exposure_pct=settings.max_total_exposure_pct,
                risk_per_trade=settings.risk_per_trade,
                max_total_risk=settings.max_total_risk,
            )
            if not final_risk.approved:
                dashboard.stats.log_activity(
                    f"FINAL RISK BLOCKED: {symbol} — {final_risk.reasons[0]}", "WARNING"
                )
                continue
            reservations.reserve(final_order)

            signal_id = str(trade.get("signal_id", ""))
            decision = candidate_decisions.get(signal_id)
            rationale = (
                decision.rationale
                if decision is not None
                else list(trade.get("candidate_rationale", []))
            )
            idempotency_key = f"{data_namespace}:{workflow_id}:{signal_id}:{symbol}:entry"
            trade_id = lifecycle_store.new_trade_id()
            lifecycle_store.create_intent(
                namespace=data_namespace,
                run_id=run_id,
                workflow_id=workflow_id,
                idempotency_key=idempotency_key,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                strategy=strategy,
                timeframe=str(trade.get("timeframe", "")),
                quantity=quantity,
                entry_price=expected_fill,
                stop_loss=stop_loss,
                target_price=target_price,
                rationale=rationale,
                context={
                    "regime": regime,
                    "regime_confidence": confidence,
                    "validation": trade.get("validation", {}),
                    "risk_result": trade.get("risk_result", {}),
                    "news_sentiment": final_state.get("news_sentiment", {}),
                    "market_mood": final_state.get("market_mood", {}),
                    "prediction_signals": final_state.get("prediction_signals", []),
                    "target_realistic": (
                        decision.target_realistic if decision is not None else True
                    ),
                },
                active_lessons=(
                    feedback.lesson_ids(memory_lessons) if settings.enable_learning else []
                ),
                trade_id=trade_id,
            )

            # Execute via the unified execution service (idempotent; shadow-aware;
            # awaits the broker for live modes).
            result = await execution_service.submit_async(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=requested_price,
                idempotency_key=idempotency_key,
                trade_id=trade_id,
                position_id=trade_id,
                stop_loss=stop_loss,
                target_price=target_price,
                strategy=strategy,
                reason="entry",
            )
            reservations.release(final_order)

            if result.is_duplicate:
                dashboard.stats.log_activity(f"DUPLICATE suppressed: {side} {symbol}", "INFO")
                continue

            if result.filled:
                tf = trade.get("timeframe", "N/A")
                dashboard.stats.log_activity(
                    f"PAPER FILL: {side} {quantity} {symbol} @ Rs.{result.fill_price:,.2f} [{tf}]",
                    "TRADE",
                )
                dashboard.stats.current_balance = paper_engine.get_balance()
                lifecycle_store.mark_open(
                    trade_id,
                    order_id=result.order_id,
                    position_id=trade_id,
                    quantity=quantity,
                    fill_price=result.fill_price,
                    entry_charges=result.entry_charges,
                )
                daily_risk_store.record_entry()

                # Register with exit manager for tracking
                exit_manager.register_position(
                    position_id=trade_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=result.fill_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    strategy=strategy,
                    regime=regime,
                )
                try:
                    trade_record = {
                        **trade,
                        "quantity": quantity,
                        "entry_price": result.fill_price,
                        "stop_loss": stop_loss,
                        "target_price": target_price,
                        "candidate_decision": (decision.to_dict() if decision is not None else {}),
                    }
                    journal.record_trade(
                        trade_record,
                        workflow_id,
                        final_state,
                        trade_id=trade_id,
                        run_id=run_id,
                    )
                except Exception as e:
                    logger.warning("Journal record_trade failed: %s", e)
                _publish_chart(
                    symbol,
                    entry=result.fill_price,
                    current=result.fill_price,
                    target=target_price,
                    stop=stop_loss,
                    status="OPEN",
                )
            else:
                lifecycle_store.mark_rejected(trade_id, result.message or result.status)
                dashboard.stats.log_activity(
                    f"ORDER {result.status}: {symbol} — {result.message}", "WARNING"
                )

        await refresh_dashboard()

        # Update paper engine P&L with current prices
        paper_engine.update_positions_pnl(market_prices)

        # Update dashboard balance from paper engine
        dashboard.stats.current_balance = paper_engine.get_balance()

        # ── FinOps: surface today's LLM spend + soft-budget alert ──
        cost_summary = cost_tracker.daily_summary()
        dashboard.stats.llm_calls = cost_summary["calls"]
        dashboard.stats.llm_tokens = cost_summary["total_tokens"]
        dashboard.stats.llm_cost_usd = cost_summary["total_cost_usd"]
        budget = cost_tracker.budget_status()
        if budget["soft_breached"] and not budget["hard_breached"]:
            await get_alert_manager().alert(
                "finops_soft_budget",
                f"Approaching daily LLM budget: {budget['tokens_used']:,} tokens / "
                f"${budget['cost_used_usd']:.4f} used "
                f"(soft limit {budget['soft_pct']:.0%}).",
                level="WARNING",
            )

        # ── Profit goal: pace tracking (advisory only — never changes sizing) ──
        goal_report = goal_engine.evaluate(
            capital=paper_engine.get_balance(),
            realized_pnl=dashboard.stats.realized_pnl,
        )
        if goal_report.get("enabled"):
            dashboard.stats.goal_mtd_pnl = goal_report["month_to_date_pnl"]
            dashboard.stats.goal_expected_to_date = goal_report["expected_to_date"]
            dashboard.stats.goal_on_pace = goal_report["on_pace"]
            dashboard.stats.goal_status = goal_report["status"]
            if not goal_report["feasible"]:
                await get_alert_manager().alert(
                    "goal_infeasible",
                    "Profit target not feasible within risk budget. "
                    f"{goal_report['plan']['recommended_action']}",
                    level="WARNING",
                )
            elif not goal_report["on_pace"]:
                await get_alert_manager().alert(
                    "goal_off_pace",
                    f"Behind profit pace: month-to-date Rs.{goal_report['month_to_date_pnl']:,.0f} "
                    f"vs pace Rs.{goal_report['expected_to_date']:,.0f}. "
                    "Risk limits are fixed — do NOT increase position size to catch up.",
                    level="WARNING",
                )

        # Increment cycle
        dashboard.increment_cycle()
        await refresh_dashboard()

        # Wait before next cycle
        wait_time = settings.trading_cycle_seconds
        dashboard.stats.log_activity(f"Next cycle in {wait_time}s...", "INFO")
        await refresh_dashboard()

        for _ in range(wait_time):
            await asyncio.sleep(1)
            await refresh_dashboard()

    consecutive_errors = 0
    market_pause_announced = False

    try:
        cycle = 0

        while True:
            if not settings.force_trading_window and not is_trading_window(
                settings.no_trading_before, settings.no_trading_after
            ):
                if not market_pause_announced:
                    dashboard.stats.log_activity(
                        "Market closed: trading cycles paused until the next entry window.",
                        "INFO",
                    )
                    market_pause_announced = True
                await refresh_dashboard()
                await asyncio.sleep(60)
                continue

            market_pause_announced = False
            cycle += 1
            dashboard.stats.log_activity(f"=== Trading Cycle #{cycle} ===", "INFO")
            await refresh_dashboard()

            try:
                await _run_cycle(cycle)
                consecutive_errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Per-cycle isolation: log, back off, and keep the loop alive.
                consecutive_errors += 1
                backoff = min(60, 5 * consecutive_errors)
                logger.exception("Trading cycle #%d failed", cycle)
                dashboard.stats.log_activity(
                    f"Cycle #{cycle} ERROR (failure #{consecutive_errors}): {e} "
                    f"— recovering in {backoff}s",
                    "ERROR",
                )
                if consecutive_errors >= 5:
                    dashboard.stats.log_activity(
                        "5+ consecutive cycle failures — check data source and logs",
                        "WARNING",
                    )
                for _ in range(backoff):
                    await asyncio.sleep(1)
                    await refresh_dashboard()

    except KeyboardInterrupt:
        dashboard.stats.log_activity("Shutdown requested", "WARNING")
        await refresh_dashboard()

    finally:
        if discovery_scheduler is not None:
            await discovery_scheduler.stop()
        if listen_task is not None:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
        if web_task is not None:
            await web_server.shutdown()
            web_task.cancel()
            try:
                await web_task
            except asyncio.CancelledError:
                pass
        if sector_task is not None:
            sector_task.cancel()
            try:
                await sector_task
            except asyncio.CancelledError:
                pass
        if scalping_task is not None:
            scalping_task.cancel()
            try:
                await scalping_task
            except asyncio.CancelledError:
                pass
        await market_manager.stop()
        # Show final stats
        stats = paper_engine.get_stats()
        cost_summary = get_cost_tracker().daily_summary()
        console.print("\n[yellow]Trading stopped[/]")
        console.print(
            f"[dim]Final balance: Rs.{stats['balance']:,.2f} | "
            f"P&L: Rs.{stats['total_pnl']:+,.2f} | "
            f"Win rate: {stats['win_rate']:.1f}%[/]"
        )
        console.print(
            f"[dim]LLM spend today: {cost_summary['calls']} calls | "
            f"{cost_summary['total_tokens']:,} tokens | "
            f"${cost_summary['total_cost_usd']:.4f} (paid-tier equiv)[/]"
        )

        # Telegram shutdown + P&L summary (best-effort)
        if notifier.enabled:
            try:
                await notifier.send_pnl_summary(
                    balance=stats["balance"],
                    realized_pnl=stats["total_pnl"],
                    unrealized_pnl=0.0,
                    total_trades=stats.get("total_trades", 0),
                    win_rate=stats["win_rate"],
                )
                await notifier.send_shutdown_message(reason="Session ended")
            except Exception as e:
                logger.warning("Telegram shutdown notification failed: %s", e)


def _acquire_single_instance_lock() -> Any:
    """Refuse to start a second instance.

    A duplicate process doesn't just race for the web UI port (the failure mode this
    used to be caught by) — now that paper wallet/signal history/daily risk state all
    live in Postgres, two independent processes would each open their own connection
    pool against the same database and silently split writes between them. This has
    happened for real multiple times this session, so it's no longer just a cosmetic
    "port already in use" log line — it's a data-integrity risk.

    Uses an OS-level file lock (not a PID file): the OS releases it automatically even
    if the process is killed or crashes, so there's no stale-lock file to clean up by
    hand — unlike a PID file, which can wrongly claim a slot forever if the process that
    wrote it never got the chance to delete it on exit.
    """
    lock_path = Path("run_live_trading.lock")
    lock_file = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(
            "ERROR: another instance of run_live_trading.py is already running "
            f"(lock held on {lock_path}). Stop it first — a second instance would race "
            "for the web UI port and split Postgres writes between two processes.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return lock_file  # keep this referenced for the life of the process — closing releases the lock


def main():
    """Main entry point."""
    import atexit

    def suppress_threading_errors():
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

    atexit.register(suppress_threading_errors)

    lock_file = _acquire_single_instance_lock()  # noqa: F841 — held for process lifetime

    try:
        asyncio.run(run_live_trading())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
