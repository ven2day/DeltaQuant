"""
Out-of-sample / walk-forward strategy validation — RUN THIS BEFORE RISKING CAPITAL.

Evaluates the live signal logic (RealSignalStrategy -> the same SignalEngine used live) on
rolling out-of-sample windows, **net of realistic NSE costs**, and prints a blunt
VALIDATED / NOT VALIDATED verdict. A green single in-sample backtest is meaningless; this is
the gate.

    uv run python scripts/validate_strategy.py

In addition to the mixed-strategy universe report, this ALSO walk-forward-validates each named
strategy (every src.market.signals.StrategyType member) individually and registers the
result in the H-8 strategy admission registry (``settings.strategy_registry_dir``,
DeltaQuant-Quant-Risk-Review.md) via ``StrategyRegistry``/``build_strategy_version``. Runtime
strategy selection (``src/agents/strategy_selection.py``) and the risk-compliance final gate
(``src/agents/risk_compliance.py``) both fail closed against that registry: a named strategy
with no current, non-expired VALIDATED entry there cannot trade live-paper. Re-run this
periodically -- entries expire after ``settings.strategy_registry_validity_days`` days.

NOTE: uses a FIXED, explicit universe (not the live StockDiscovery output, which would inject
look-ahead/selection bias). See the survivorship caveat printed at the end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.strategies import RealSignalStrategy
from src.backtesting.strategy_registry import StrategyRegistry, build_strategy_version
from src.backtesting.walk_forward import aggregate_reports, run_walk_forward
from src.config import get_settings
from src.execution.costs import CostModel
from src.market.historical_feed import HistoricalDataFeed
from src.market.signals import StrategyType

# A fixed large-cap universe. Deliberately NOT StockDiscovery (which picks today's movers and
# would bias the evaluation). Replace with a point-in-time, survivorship-free list for production.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "ITC", "LT", "AXISBANK", "BHARTIARTL",
]
LINE = "=" * 84


def register_strategy_versions(
    symbols: list[str],
    period: str,
    warmup: int,
    test: int,
    cost_model: CostModel,
) -> None:
    """Walk-forward-validate each named strategy individually and register the verdict
    (H-8). This is the ONLY place that writes to the admission registry; it always
    sources the verdict from ``edge_verdict()`` via ``build_strategy_version``, never a
    parallel pass/fail decision.
    """
    settings = get_settings()
    registry = StrategyRegistry(settings.strategy_registry_dir)
    feed = HistoricalDataFeed(symbols=symbols)
    dataset_id = f"historical_feed:{period}:{','.join(sorted(symbols))}"

    print(LINE)
    print(" Per-strategy walk-forward validation for the H-8 admission registry")
    print(LINE)

    for strategy_type in StrategyType:
        reports = []
        for sym in symbols:
            df = feed.get_historical(sym, period=period)
            if df is None or df.empty or len(df) < warmup + test:
                continue
            report = run_walk_forward(
                df,
                lambda s=sym, st=strategy_type: RealSignalStrategy(symbol=s, strategy_types=[st]),
                symbol=sym,
                warmup_bars=warmup,
                test_bars=test,
                cost_model=cost_model,
            )
            reports.append(report)

        agg = aggregate_reports(reports)
        version = build_strategy_version(
            strategy_type.value,
            owner="scripts/validate_strategy.py",
            parameters={"warmup_bars": warmup, "test_bars": test, "period": period},
            approved_universe=symbols,
            approved_regimes=[],  # no per-regime OOS split yet; admitted for any regime
            dataset_id=dataset_id,
            oos_trades=agg["total_oos_trades"],
            oos_expectancy=agg["weighted_oos_expectancy"],
            oos_return_pct=agg["summed_oos_return_pct"],
            fold_consistency=agg["avg_fold_consistency"],
            validity_days=settings.strategy_registry_validity_days,
        )
        path = registry.register(version)
        print(
            f" {strategy_type.value:<16} verdict={version.verdict:<14} "
            f"trades={version.oos_trades:<5} expiry={version.expires_at[:10]}  -> {path.name}"
        )
    print(LINE)


def main(symbols: list[str] | None = None, period: str = "2y", warmup: int = 120, test: int = 40) -> None:
    symbols = symbols or DEFAULT_UNIVERSE
    cost_model = CostModel.from_settings()
    feed = HistoricalDataFeed(symbols=symbols)
    reports = []

    print(LINE)
    print(" Out-of-Sample / Walk-Forward Validation  (net of realistic costs)")
    print(LINE)
    print(f" Universe: {len(symbols)} symbols | period={period} | warmup={warmup} test={test} bars")
    print(
        f" Costs: slippage {cost_model.slippage_bps}bps | brokerage {cost_model.brokerage_bps}bps"
        f" (cap Rs.{cost_model.brokerage_max:.0f}) | statutory {cost_model.statutory_bps}bps"
        f" | GST {cost_model.gst_pct}%"
    )
    print(LINE)
    header = f"{'Symbol':<12}{'trades':>8}{'ret%':>9}{'exp':>9}{'win%':>7}{'PF':>7}{'maxDD%':>8}{'consist':>9}"
    print(header)
    print("-" * len(header))

    for sym in symbols:
        df = feed.get_historical(sym, period=period)
        if df is None or df.empty or len(df) < warmup + test:
            print(f"{sym:<12}{'(insufficient data)':>30}")
            continue
        report = run_walk_forward(
            df,
            lambda s=sym: RealSignalStrategy(symbol=s),
            symbol=sym,
            warmup_bars=warmup,
            test_bars=test,
            cost_model=cost_model,
        )
        reports.append(report)
        pf = "inf" if report.oos_profit_factor == float("inf") else f"{report.oos_profit_factor:.2f}"
        print(
            f"{sym:<12}{report.oos_trades:>8}{report.oos_return_pct:>9.2f}"
            f"{report.oos_expectancy:>9.3f}{report.oos_win_rate:>7.1f}{pf:>7}"
            f"{report.oos_max_drawdown_pct:>8.2f}{report.fold_consistency:>8.0%}"
        )

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
    print(" necessary, NOT sufficient. Also: real fills face circuit limits, gaps, and liquidity that")
    print(" a historical bar backtest cannot fully capture.")
    print(LINE)

    register_strategy_versions(symbols, period, warmup, test, cost_model)


if __name__ == "__main__":
    main()
