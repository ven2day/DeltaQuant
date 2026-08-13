"""
Out-of-sample / walk-forward strategy validation — RUN THIS BEFORE RISKING CAPITAL.

Evaluates the live signal logic (RealSignalStrategy -> the same SignalEngine used live) on
rolling out-of-sample windows, **net of realistic NSE costs**, and prints a blunt
VALIDATED / NOT VALIDATED verdict. A green single in-sample backtest is meaningless; this is
the gate.

    uv run python scripts/validate_strategy.py

In addition to the mixed-strategy universe report, this walk-forward-validates every named
strategy and writes a StrategyEligibility record keyed by strategy, timeframe, and model
version. Passing validation grants paper approval only; live approval remains an explicit
operator-controlled status transition. The legacy artifact is also retained for migration
and historical reproducibility.

NOTE: uses a FIXED, explicit universe (not the live StockDiscovery output, which would inject
look-ahead/selection bias). See the survivorship caveat printed at the end.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.strategies import RealSignalStrategy
from src.backtesting.strategy_eligibility import (
    EligibilityStatus,
    StrategyEligibility,
    eligibility_registry_from_settings,
)
from src.backtesting.strategy_registry import StrategyRegistry, build_strategy_version
from src.backtesting.walk_forward import aggregate_reports, run_walk_forward
from src.config import get_settings
from src.core.candidates import StrategyType
from src.core.candles import CandleStore
from src.core.features import build_historical_market_relative_context
from src.core.indicators import Timeframe
from src.core.strategies import PRODUCTION_STRATEGY_NAMES
from src.markets.nse.market_data.historical_feed import HistoricalDataFeed
from src.markets.nse.risk.costs import CostModel
from src.markets.nse.strategies import build_strategy_config

# A fixed, sector-diversified universe (large + mid cap). Deliberately NOT StockDiscovery
# (which picks today's movers and would bias the evaluation). Replace with a point-in-time,
# survivorship-free list for production.
DEFAULT_UNIVERSE = [
    # Banking / financials
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    # IT
    "TCS",
    "INFY",
    "WIPRO",
    "TECHM",
    # Telecom
    "BHARTIARTL",
    # FMCG
    "ITC",
    "HINDUNILVR",
    # Auto
    "MARUTI",
    "TVSMOTOR",
    "M&M",
    # Pharma
    "SUNPHARMA",
    "DRREDDY",
    "CIPLA",
    # Metals
    "TATASTEEL",
    "HINDALCO",
    "JSWSTEEL",
    # Cement / infra
    "ULTRACEMCO",
    "ADANIPORTS",
    # Energy / PSU
    "RELIANCE",
    "NTPC",
    "POWERGRID",
    "ONGC",
    # Realty / consumer durables
    "DLF",
    "TITAN",
]
LINE = "=" * 84

# Daily NSE session is ~375 minutes (09:15-15:30), so an intraday interval needs its
# own period/warmup/test -- the "2y / 120 warmup / 40 test" defaults below are bar
# COUNTS tuned for 1d bars (~500 bars total for 2y). Applied unchanged to e.g. 5m bars
# (2y -> ~30,000+ bars/symbol), that blows up to ~750 folds/symbol instead of ~10 --
# across a couple hundred symbols and 4-7 strategies that is effectively unbounded
# compute, not a hang, but indistinguishable from one in practice. Keep period within
# DhanHQ's own MAX_REQUEST_DAYS (src/market/dhan_historical_feed.py) so a single fetch
# window covers it, and scale warmup/test off that interval's own bars/day so fold count
# stays the same order of magnitude as the validated 1d case.
_BARS_PER_DAY = {"5m": 75, "15m": 25, "30m": 12, "1h": 6}


def _interval_defaults(interval: str) -> tuple[str, int, int]:
    """Return (period, warmup_bars, test_bars) sized for this interval."""
    bars_per_day = _BARS_PER_DAY.get(interval)
    if bars_per_day is None:
        return "2y", 120, 40  # 1d and anything unrecognized: unchanged existing behavior
    return "90d", bars_per_day * 10, bars_per_day * 5  # ~10 trading days warmup, ~5 test


def _load_universe_from_csv(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    # First line is a header ("symbol"); skip it defensively either way.
    return [line for line in lines if line.lower() != "symbol"]


def _normalize_ohlcv(frame):
    """Adapt CandleStore's lowercase schema to the backtest engine contract."""
    if frame is None or frame.empty:
        return frame
    aliases = {str(column).lower(): column for column in frame.columns}
    required = ("open", "high", "low", "close", "volume")
    if not all(name in aliases for name in required):
        return frame
    return frame.rename(columns={aliases[name]: name.title() for name in required})


def _load_local_validation_frame(
    store: CandleStore,
    symbol: str,
    interval: str,
    bars: int | None,
):
    """Load native or shared NSE-aligned derived candles from TimescaleDB."""

    frame = store.load_analysis_frame(
        symbol,
        interval,
        bars=bars,
        complete_only=True,
    )
    return _normalize_ohlcv(frame)


def register_strategy_versions(
    symbols: list[str],
    period: str,
    warmup: int,
    test: int,
    cost_model: CostModel,
    interval: str = "1d",
    trade_horizon: str = "SWING",
    data_by_symbol: dict[str, object] | None = None,
    indicator_caches_by_symbol: dict[str, dict] | None = None,
    market_relative_by_symbol: dict[str, dict] | None = None,
) -> None:
    """Walk-forward-validate each named strategy and register its eligibility.

    The legacy artifact remains immutable and preserved. The new eligibility record
    uses the same offline verdict and never grants LIVE_APPROVED automatically.

    ``interval``/``trade_horizon`` default to ``"1d"``/``"SWING"`` -- today's exact
    behavior (interval was previously never passed at all, so ``get_historical``
    silently defaulted to daily bars regardless of what a caller might have assumed).
    Passing e.g. ``interval="5m", trade_horizon="SCALP"`` records the actual 5m
    validation timeframe. Horizon remains report metadata and does not create a second
    StrategyEligibility identity.

    ``data_by_symbol``, when supplied, is reused instead of refetching -- with
    ``StrategyType`` now at 7 members, refetching per strategy meant 7 (here) + 1 (the
    mixed-universe report in ``main``) = 8x more Dhan calls than symbols actually need.
    """
    settings = get_settings()
    legacy_registry = StrategyRegistry(settings.strategy_registry_dir)
    eligibility_registry = eligibility_registry_from_settings(settings)
    feed = None if data_by_symbol is not None else HistoricalDataFeed(symbols=symbols)
    dataset_id = f"historical_feed:{period}:{interval}:{','.join(sorted(symbols))}"
    production_config = build_strategy_config(settings)
    strategy_types = [
        strategy_type
        for strategy_type in StrategyType
        if strategy_type.value not in PRODUCTION_STRATEGY_NAMES
        or production_config.eligible(strategy_type.value, interval)
    ]

    print(LINE)
    print(
        " Per-strategy walk-forward validation for StrategyEligibilityRegistry "
        f"(interval={interval}; horizon metadata={trade_horizon})"
    )
    print(LINE)

    reports_by_strategy = {strategy_type: [] for strategy_type in strategy_types}
    for sym in symbols:
        df = (data_by_symbol or {}).get(sym) if data_by_symbol is not None else None
        if data_by_symbol is None:
            df = feed.get_historical(sym, period=period, interval=interval)  # type: ignore[union-attr]
        if df is None or df.empty or len(df) < warmup + test:
            continue
        indicator_cache = (
            indicator_caches_by_symbol.setdefault(sym, {})
            if indicator_caches_by_symbol is not None
            else {}
        )
        for strategy_type in strategy_types:
            report = run_walk_forward(
                df,
                lambda s=sym, st=strategy_type, cache=indicator_cache: RealSignalStrategy(
                    symbol=s,
                    strategy_types=[st],
                    timeframe=Timeframe(interval),
                    indicator_cache=cache,
                    production_config=production_config,
                    market_relative_by_timestamp=(market_relative_by_symbol or {}).get(sym, {}),
                ),
                symbol=sym,
                warmup_bars=warmup,
                test_bars=test,
                cost_model=cost_model,
                interval=interval,
            )
            reports_by_strategy[strategy_type].append(report)

    for strategy_type in strategy_types:
        reports = reports_by_strategy[strategy_type]
        # This loop tests len(strategy_types) strategies against the same universe in
        # pursuit of a handful of PAPER_APPROVED admissions -- the classic multiple-
        # comparisons setup. Bonferroni-correcting the significance bar (same pattern
        # already used in src/signal_discovery/workflow.py) means the bar tightens as
        # more strategies are tested in one run, instead of an uncorrected p<0.05
        # threshold letting through roughly one false "significant" edge per ~20
        # strategies tested purely by chance.
        agg = aggregate_reports(
            reports, num_tests=len(strategy_types), require_significance=True
        )
        is_production_strategy = strategy_type.value in PRODUCTION_STRATEGY_NAMES
        regime_validation_available = False
        version = build_strategy_version(
            strategy_type.value,
            owner="scripts/validate_strategy.py",
            parameters={
                "warmup_bars": warmup,
                "test_bars": test,
                "period": period,
                "interval": interval,
            },
            approved_universe=symbols,
            approved_regimes=[],  # no per-regime OOS split yet; admitted for any regime
            dataset_id=dataset_id,
            oos_trades=agg["total_oos_trades"],
            oos_expectancy=agg["weighted_oos_expectancy"],
            oos_return_pct=agg["summed_oos_return_pct"],
            fold_consistency=agg["avg_fold_consistency"],
            validity_days=settings.strategy_registry_validity_days,
            timeframe=interval,
            trade_horizon=trade_horizon,
            p_value=agg["p_value"],
            num_tests=agg["num_tests"],
            require_significance=True,
        )
        legacy_path = legacy_registry.register(version)
        eligibility = StrategyEligibility(
            strategy_name=strategy_type.value,
            timeframe=interval,
            model_version=version.version,
            validation_status=(
                EligibilityStatus.PAPER_APPROVED
                if version.verdict == "VALIDATED"
                and (not is_production_strategy or regime_validation_available)
                else EligibilityStatus.SHADOW
            ),
            validated_at=version.validated_at,
            validation_window={
                "dataset_id": dataset_id,
                "period": period,
                "warmup_bars": warmup,
                "test_bars": test,
                "interval": interval,
                "production_strategy_version": production_config.version,
                "strategy_parameters": dict(
                    production_config.parameters.get(strategy_type.value, {})
                ),
            },
            oos_trade_count=agg["total_oos_trades"],
            oos_profit_factor=agg["weighted_oos_profit_factor"],
            oos_max_drawdown=agg["max_oos_drawdown_pct"],
            oos_win_rate=agg["weighted_oos_win_rate"],
            minimum_model_confidence=0.0,
            allowed_regimes=(),
            disabled_regimes=(),
            status_reason=(
                "Offline walk-forward passed; approved for paper only"
                if version.verdict == "VALIDATED"
                and (not is_production_strategy or regime_validation_available)
                else "Aggregate OOS passed, but production strategy remains SHADOW until regime-specific validation is recorded"
                if version.verdict == "VALIDATED" and is_production_strategy
                else "Offline walk-forward did not pass; retained as shadow evidence"
            ),
            artifact_reference=None,
            requires_ml=False,
            regime_performance={},
            validation_metadata={
                "legacy_artifact": str(legacy_path),
                "trade_horizon_observed": trade_horizon,
                "oos_expectancy": agg["weighted_oos_expectancy"],
                "oos_return_pct": agg["summed_oos_return_pct"],
                "fold_consistency": agg["avg_fold_consistency"],
                "regime_validation_available": regime_validation_available,
            },
        )
        path = eligibility_registry.register(eligibility)
        p_display = f"{agg['p_value']:.4f}" if agg["p_value"] is not None else "n/a"
        print(
            f" {strategy_type.value:<16} verdict={version.verdict:<14} "
            f"trades={version.oos_trades:<5} p={p_display:<7} "
            f"(bonferroni-corrected bar={agg['bonferroni_corrected_p_threshold']:.4f}) "
            f"status={eligibility.validation_status.value:<15} -> {path.name}"
        )
    print(LINE)


def main(
    symbols: list[str] | None = None,
    period: str | None = None,
    warmup: int | None = None,
    test: int | None = None,
    interval: str = "1d",
    trade_horizon: str = "SWING",
    use_local_store: bool = False,
    local_bars: int | None = None,
) -> None:
    symbols = symbols or DEFAULT_UNIVERSE
    default_period, default_warmup, default_test = _interval_defaults(interval)
    period = period or default_period
    warmup = warmup if warmup is not None else default_warmup
    test = test if test is not None else default_test
    settings = get_settings()
    cost_model = CostModel.from_settings()
    feed = None if use_local_store else HistoricalDataFeed(symbols=symbols)
    store = None
    if use_local_store:
        database_url = settings.market_history_database_url or settings.database_url
        store = CandleStore(
            database_url,
            enable_timescale=settings.market_history_enable_timescale,
            require_timescale=settings.market_history_require_timescale,
        )
        if local_bars is None:
            # Keep approximately ten OOS windows, matching the interval-aware Dhan
            # default without loading two full intraday years into every backtest.
            local_bars = warmup + test * 10
    reports = []

    print(LINE)
    print(" Out-of-Sample / Walk-Forward Validation  (net of realistic costs)")
    print(LINE)
    print(
        f" Universe: {len(symbols)} symbols | period={period} | interval={interval} | "
        f"warmup={warmup} test={test} bars | trade_horizon={trade_horizon}"
    )
    print(
        f" Costs: slippage {cost_model.slippage_bps}bps | brokerage {cost_model.brokerage_bps}bps"
        f" (cap Rs.{cost_model.brokerage_max:.0f}) | statutory {cost_model.statutory_bps}bps"
        f" | GST {cost_model.gst_pct}%"
    )
    print(LINE)
    header = f"{'Symbol':<12}{'trades':>8}{'ret%':>9}{'exp':>9}{'win%':>7}{'PF':>7}{'maxDD%':>8}{'consist':>9}"
    print(header)
    print("-" * len(header))

    # Fetch every symbol's data exactly once and reuse it for both the mixed-universe
    # report below and the per-strategy registration pass -- StrategyType has 7 members,
    # so without this a 272-symbol run made 8x more Dhan calls than symbols require.
    data_by_symbol: dict[str, object] = {}
    indicator_caches_by_symbol: dict[str, dict] = {}
    for sym in symbols:
        df = (
            _load_local_validation_frame(store, sym, interval, local_bars)
            if store is not None
            else feed.get_historical(sym, period=period, interval=interval)  # type: ignore[union-attr]
        )
        data_by_symbol[sym] = df

    production_config = build_strategy_config(settings)
    market_relative_by_symbol = build_historical_market_relative_context(
        data_by_symbol,
        benchmark_symbol=production_config.benchmark_symbol,
        fallback=production_config.benchmark_fallback,
    )
    for i, sym in enumerate(symbols, 1):
        df = data_by_symbol[sym]
        if df is None or df.empty or len(df) < warmup + test:
            print(f"{sym:<12}{'(insufficient data)':>30}")
            continue
        report = run_walk_forward(
            df,
            lambda s=sym, cache=indicator_caches_by_symbol.setdefault(sym, {}): RealSignalStrategy(
                symbol=s,
                timeframe=Timeframe(interval),
                indicator_cache=cache,
                production_config=production_config,
                market_relative_by_timestamp=market_relative_by_symbol.get(sym, {}),
            ),
            symbol=sym,
            warmup_bars=warmup,
            test_bars=test,
            cost_model=cost_model,
            interval=interval,
        )
        reports.append(report)
        pf = (
            "inf" if report.oos_profit_factor == float("inf") else f"{report.oos_profit_factor:.2f}"
        )
        print(
            f"{sym:<12}{report.oos_trades:>8}{report.oos_return_pct:>9.2f}"
            f"{report.oos_expectancy:>9.3f}{report.oos_win_rate:>7.1f}{pf:>7}"
            f"{report.oos_max_drawdown_pct:>8.2f}{report.fold_consistency:>8.0%}"
        )
        if i % 25 == 0 or i == len(symbols):
            print(f" ... progress: {i}/{len(symbols)} symbols fetched+tested", flush=True)

    print(LINE)
    agg = aggregate_reports(reports)
    print(
        f" Universe OOS: {agg['total_oos_trades']} trades | summed return "
        f"{agg['summed_oos_return_pct']}% | weighted expectancy {agg['weighted_oos_expectancy']}"
        f" | avg consistency {agg['avg_fold_consistency']:.0%}"
    )
    print(f" Symbols net-positive OOS: {agg['symbols_positive']}/{agg['symbols_evaluated']}")
    print(LINE)
    print(f" VERDICT: {agg['verdict']}")
    for reason in agg["reasons"]:
        print(f"   - {reason}")
    print(LINE)
    print(" SURVIVORSHIP CAVEAT: this universe is current-listed names only. A production")
    print(" go/no-go REQUIRES a point-in-time, survivorship-free universe (incl. delisted/")
    print(" suspended names) — no free market-data API can provide this. VALIDATED here is")
    print(
        " necessary, NOT sufficient. Also: real fills face circuit limits, gaps, and liquidity that"
    )
    print(" a historical bar backtest cannot fully capture.")
    print(LINE)

    register_strategy_versions(
        symbols,
        period,
        warmup,
        test,
        cost_model,
        interval=interval,
        trade_horizon=trade_horizon,
        data_by_symbol=data_by_symbol,
        indicator_caches_by_symbol=indicator_caches_by_symbol,
        market_relative_by_symbol=market_relative_by_symbol,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval to validate against, e.g. 1d (default, current behavior), "
        "5m, 15m. Must be an interval the configured historical feed actually supports.",
    )
    parser.add_argument(
        "--trade-horizon",
        default="SWING",
        choices=["SWING", "SCALP"],
        help="Decision-horizon metadata retained in the legacy artifact and validation "
        "report. It is not part of StrategyEligibility identity.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Validate against every symbol in config/nse/symbols.csv (the NSE universe) "
        "instead of the smaller diversified DEFAULT_UNIVERSE. Substantially more Dhan "
        "API calls and compute -- expect a long run.",
    )
    parser.add_argument(
        "--symbols-file",
        default="config/nse/symbols.csv",
        help="NSE CSV to read when --all-symbols is set.",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Override the auto-selected lookback period (e.g. 2y, 90d). Default is "
        "interval-aware: 2y for 1d, 90d for intraday intervals.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override the auto-selected warmup bar count.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Override the auto-selected OOS test-window bar count.",
    )
    parser.add_argument(
        "--local-store",
        action="store_true",
        help="Read validation candles from MARKET_HISTORY_DATABASE_URL instead of Dhan",
    )
    parser.add_argument(
        "--local-bars",
        type=int,
        default=None,
        help="Newest local bars per symbol (default: warmup + ten test windows)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    _symbols = _load_universe_from_csv(_args.symbols_file) if _args.all_symbols else None
    main(
        symbols=_symbols,
        period=_args.period,
        warmup=_args.warmup,
        test=_args.test,
        interval=_args.interval,
        trade_horizon=_args.trade_horizon,
        use_local_store=_args.local_store,
        local_bars=_args.local_bars,
    )
