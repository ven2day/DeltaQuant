"""
Tests for the walk-forward / OOS validation harness and the realistic-cost integration in
the backtest engine.
"""

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine, Strategy
from src.backtesting.walk_forward import (
    MIN_OOS_TRADES,
    _metrics_from_pnls,
    _pnl_significance_p_value,
    aggregate_reports,
    edge_verdict,
    generate_test_folds,
    run_walk_forward,
)
from src.markets.nse.risk.costs import CostModel


def _uptrend(n=240):
    closes = np.linspace(100, 200, n)
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
         "Close": closes, "Volume": 1000},
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


class _PingPong(Strategy):
    """Deterministic: alternate BUY/SELL so trades land predictably for attribution."""

    name = "PingPong"

    def on_bar(self, row, history):
        return "BUY" if len(history) % 2 == 0 else "SELL"


class _OneFoldStrategy(Strategy):
    """Trades only inside the first OOS fold (bars [120, 160)) so the edge lives in one window."""

    name = "OneFold"

    def on_bar(self, row, history):
        n = len(history)
        if 120 <= n < 160:
            return "BUY" if n % 2 == 0 else "SELL"
        return None


# ---------------------------------------------------------------------------
# Fold generation + metrics
# ---------------------------------------------------------------------------

def test_generate_test_folds():
    folds = generate_test_folds(n_bars=200, warmup_bars=120, test_bars=40)
    assert folds == [(120, 160), (160, 200)]


def test_generate_test_folds_none_when_too_short():
    assert generate_test_folds(n_bars=100, warmup_bars=120, test_bars=40) == []


def test_metrics_from_pnls():
    m = _metrics_from_pnls([10.0, -5.0, 20.0, -5.0], initial_capital=1000.0)
    assert m["trades"] == 4
    assert m["expectancy"] == 5.0  # (10-5+20-5)/4
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == 3.0  # 30 / 10
    assert m["return_pct"] == 2.0  # 20/1000


# ---------------------------------------------------------------------------
# run_walk_forward
# ---------------------------------------------------------------------------

def test_run_walk_forward_produces_oos_folds():
    data = _uptrend(240)
    report = run_walk_forward(
        data, _PingPong, symbol="X", warmup_bars=120, test_bars=40, cost_model=CostModel.zero()
    )
    assert report.symbol == "X"
    assert report.oos_trades > 0
    assert len(report.folds) == 3  # (120,160),(160,200),(200,240)
    # On a clean uptrend, ping-pong long trades are net positive.
    assert report.oos_return_pct > 0


def test_run_walk_forward_insufficient_data():
    report = run_walk_forward(_uptrend(80), _PingPong, warmup_bars=120, test_bars=40)
    assert report.oos_trades == 0
    assert report.folds == []
    # Regression: the no-folds branch once passed args positionally and put [] into the
    # fold_consistency (float) slot. It must be a number.
    assert report.fold_consistency == 0.0
    assert isinstance(report.fold_consistency, float)


def test_single_fold_edge_is_not_consistent():
    # An edge that only shows up in ONE fold must not read as consistent (it would otherwise
    # be 1/1 = 100% and sail through the verdict gate — the "one lucky regime" trap).
    data = _uptrend(240)
    report = run_walk_forward(
        data, _OneFoldStrategy, warmup_bars=120, test_bars=40, cost_model=CostModel.zero()
    )
    assert report.oos_trades > 0
    assert report.fold_consistency == 0.0
    v = edge_verdict(
        report.oos_trades, report.oos_expectancy, report.oos_return_pct, report.fold_consistency
    )
    assert v["validated"] is False


def test_trade_pnl_is_net_of_costs():
    # Trade.pnl must be net of brokerage/slippage so walk-forward expectancy/return are not
    # gross-inflated. The booked P&L across trades must equal the realized capital change.
    data = _uptrend(240)
    cm = CostModel(slippage_bps=50, brokerage_bps=10)
    result = BacktestEngine(initial_capital=100_000.0, cost_model=cm).run(_PingPong(), data, "X")
    assert result.trades
    booked = sum(t.pnl for t in result.trades)
    realized = result.final_capital - result.initial_capital
    assert abs(booked - realized) < 1e-6


# ---------------------------------------------------------------------------
# Cost integration in the engine
# ---------------------------------------------------------------------------

def test_costs_reduce_backtest_returns():
    data = _uptrend(240)
    free = BacktestEngine(cost_model=CostModel.zero()).run(_PingPong(), data, "X")
    costed = BacktestEngine(cost_model=CostModel(slippage_bps=50, brokerage_bps=10)).run(
        _PingPong(), data, "X"
    )
    assert costed.final_capital < free.final_capital  # realistic costs eat into P&L


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def test_edge_verdict_validated():
    v = edge_verdict(oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7)
    assert v["validated"] is True
    assert v["verdict"] == "VALIDATED"


def test_edge_verdict_rejects_negative_expectancy():
    v = edge_verdict(oos_trades=100, oos_expectancy=-0.2, oos_return_pct=-3.0, fold_consistency=0.7)
    assert v["validated"] is False
    assert any("expectancy" in r for r in v["reasons"])


def test_edge_verdict_rejects_too_few_trades():
    v = edge_verdict(oos_trades=5, oos_expectancy=2.0, oos_return_pct=5.0, fold_consistency=0.9)
    assert v["validated"] is False
    assert any(str(MIN_OOS_TRADES) in r for r in v["reasons"])


def test_edge_verdict_rejects_inconsistent_edge():
    v = edge_verdict(oos_trades=100, oos_expectancy=1.0, oos_return_pct=5.0, fold_consistency=0.3)
    assert v["validated"] is False
    assert any("folds" in r for r in v["reasons"])


def test_edge_verdict_rejects_exactly_half_consistency():
    # Docstring/message say ">50%" — exactly 50% (a coin-flip across folds) must NOT pass.
    v = edge_verdict(oos_trades=100, oos_expectancy=1.0, oos_return_pct=5.0, fold_consistency=0.5)
    assert v["validated"] is False
    assert any("folds" in r for r in v["reasons"])


def test_edge_verdict_ignores_p_value_when_significance_not_required():
    # require_significance defaults to False -- every pre-existing caller that never
    # passes a p_value must keep behaving exactly as before this feature was added.
    v = edge_verdict(
        oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7,
        p_value=0.99,  # would fail any significance bar, but the check isn't requested
    )
    assert v["validated"] is True
    assert "p_value" not in v


def test_edge_verdict_requires_significance_when_opted_in():
    v = edge_verdict(
        oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7,
        p_value=0.2, num_tests=1, require_significance=True,
    )
    assert v["validated"] is False
    assert any("not statistically significant" in r for r in v["reasons"])
    assert v["p_value"] == 0.2


def test_edge_verdict_missing_p_value_rejected_when_significance_required():
    # p_value=None (too few trades to test) must NOT be treated as a pass -- an
    # untested strategy is not the same thing as a proven one.
    v = edge_verdict(
        oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7,
        p_value=None, require_significance=True,
    )
    assert v["validated"] is False
    assert any("too few OOS trades to test" in r for r in v["reasons"])


def test_edge_verdict_bonferroni_threshold_tightens_with_more_tests():
    # p=0.03 clears the nominal 0.05 bar alone, but not once divided across 10 tests.
    solo = edge_verdict(
        oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7,
        p_value=0.03, num_tests=1, require_significance=True,
    )
    batched = edge_verdict(
        oos_trades=100, oos_expectancy=1.5, oos_return_pct=8.0, fold_consistency=0.7,
        p_value=0.03, num_tests=10, require_significance=True,
    )
    assert solo["validated"] is True
    assert batched["validated"] is False
    assert batched["bonferroni_corrected_p_threshold"] < solo["bonferroni_corrected_p_threshold"]


def test_pnl_significance_p_value_too_few_points_returns_none():
    assert _pnl_significance_p_value([]) is None
    assert _pnl_significance_p_value([5.0]) is None


def test_pnl_significance_p_value_zero_variance_returns_none():
    assert _pnl_significance_p_value([10.0, 10.0, 10.0]) is None


def test_pnl_significance_p_value_detects_clear_positive_edge():
    # A large, consistently positive sample should be very significant (small p).
    pnls = [100.0 + i % 5 for i in range(200)]
    p = _pnl_significance_p_value(pnls)
    assert p is not None
    assert p < 0.001


def test_pnl_significance_p_value_high_for_noise_around_zero():
    rng = np.random.default_rng(42)
    pnls = list(rng.normal(loc=0.0, scale=10.0, size=30))
    p = _pnl_significance_p_value(pnls)
    assert p is not None
    assert p > 0.05


def test_aggregate_reports_forwards_significance_settings():
    data = _uptrend(240)
    r1 = run_walk_forward(data, _PingPong, symbol="A", cost_model=CostModel.zero())
    r2 = run_walk_forward(data, _PingPong, symbol="B", cost_model=CostModel.zero())
    agg = aggregate_reports([r1, r2], num_tests=5, require_significance=True)
    assert agg["num_tests"] == 5
    assert agg["bonferroni_corrected_p_threshold"] == 0.05 / 5
    # Backward-compatible default: no significance fields leak in when not requested.
    agg_default = aggregate_reports([r1, r2])
    assert "p_value" not in agg_default


def test_aggregate_reports():
    data = _uptrend(240)
    r1 = run_walk_forward(data, _PingPong, symbol="A", cost_model=CostModel.zero())
    r2 = run_walk_forward(data, _PingPong, symbol="B", cost_model=CostModel.zero())
    agg = aggregate_reports([r1, r2])
    assert agg["symbols"] == 2
    assert agg["total_oos_trades"] == r1.oos_trades + r2.oos_trades
    assert "verdict" in agg
