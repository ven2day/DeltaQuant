import json

import numpy as np
import pandas as pd
import pytest

from src.signal_discovery.evaluator import RankICEvaluator
from src.signal_discovery.live import DiscoveredSignalScorer
from src.signal_discovery.models import DiscoveredSignal, DiscoveryResult, EvaluationMetrics
from src.signal_discovery.operators import FormulaCompiler, build_ohlcv_panel
from src.signal_discovery.store import DiscoveredSignalStore
from src.signal_discovery.workflow import SignalDiscoveryWorkflow


def _panel(periods: int = 100, symbols: int = 12):
    rng = np.random.default_rng(42)
    innovations = rng.normal(0, 0.008, size=(periods, symbols))
    returns = np.zeros_like(innovations)
    for index in range(1, periods):
        returns[index] = 0.65 * returns[index - 1] + innovations[index]
    close = 100 * pd.DataFrame(1 + returns).cumprod()
    close.index = pd.date_range("2025-01-01", periods=periods, freq="h")
    close.columns = [f"S{index}" for index in range(symbols)]
    volume = pd.DataFrame(
        rng.integers(10_000, 100_000, size=(periods, symbols)),
        index=close.index,
        columns=close.columns,
    )
    return {
        "Open": close.shift(1).fillna(close),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": volume,
    }


def _metrics(mean_ic: float = 0.03, p_value: float = 0.01):
    return EvaluationMetrics(mean_ic, 0.1, mean_ic / 0.1, 3.0, p_value, 80, 0.55, 12.0)


def _result(signal, metrics):
    return DiscoveryResult(
        status="accepted",
        request="momentum",
        iteration=1,
        total_iterations=1,
        selected_signal=signal,
        metrics=metrics,
        signals=[signal],
        thresholds={"ic_threshold": 0.015, "p_value_threshold": 0.05},
    )


def test_formula_compiler_evaluates_allowlisted_formula():
    panel = _panel()
    compiled = FormulaCompiler().compile("Rank(Div(TS_Return(Close, 5), TS_Std(Close, 10)))")

    values = compiled.evaluate(panel)

    assert values.shape == panel["Close"].shape
    assert compiled.fields_used == ("Close",)
    assert set(compiled.operators_used) == {"Div", "Rank", "TS_Return", "TS_Std"}


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os').system('whoami')",
        "Close.to_csv('stolen.csv')",
        "Close['S1']",
        "Unknown(Close, 5)",
    ],
)
def test_formula_compiler_rejects_arbitrary_python(formula):
    with pytest.raises(ValueError):
        FormulaCompiler().compile(formula)


def test_formula_rejects_future_delay_at_evaluation():
    compiled = FormulaCompiler().compile("Delay(Close, -5)")

    with pytest.raises(ValueError, match="rolling window"):
        compiled.evaluate(_panel())


def test_build_panel_converts_symbol_frames():
    panel = _panel(periods=20, symbols=3)
    histories = {}
    for symbol in panel["Close"].columns:
        histories[symbol] = pd.DataFrame(
            {field.lower(): panel[field][symbol] for field in panel}
        )

    rebuilt = build_ohlcv_panel(histories)

    assert list(rebuilt["Close"].columns) == ["S0", "S1", "S2"]
    pd.testing.assert_frame_equal(rebuilt["Close"], panel["Close"])


def test_rank_ic_evaluator_scores_predictive_formula():
    signal = DiscoveredSignal("Momentum", "TS_Return(Close, 2)", "Persistent returns")

    metrics = RankICEvaluator(forward_periods=1, min_cross_section=8, min_periods=30).evaluate(
        signal, _panel()
    )

    assert metrics.mean_ic > 0.2
    assert metrics.p_value < 0.05
    assert metrics.num_periods >= 30


def test_store_revalidates_formula_and_quality(tmp_path):
    signal = DiscoveredSignal("Momentum", "TS_Return(Close, 5)", "Momentum")
    store = DiscoveredSignalStore(tmp_path, ic_threshold=0.015, p_value_threshold=0.05)
    path = store.save_accepted(_result(signal, _metrics()))

    loaded = store.load_accepted()

    assert path.exists()
    assert [item.signal.name for item in loaded] == ["Momentum"]

    stricter = DiscoveredSignalStore(tmp_path, ic_threshold=0.05, p_value_threshold=0.05)
    assert stricter.load_accepted() == []


def test_live_scorer_inverts_negative_ic(tmp_path):
    signal = DiscoveredSignal("Reversal", "TS_Return(Close, 2)", "Inverted momentum")
    store = DiscoveredSignalStore(tmp_path, ic_threshold=0.015, p_value_threshold=0.05)
    store.save_accepted(_result(signal, _metrics(mean_ic=-0.03)))

    tilts = DiscoveredSignalScorer(store, max_probability_tilt=0.08).score_panel(_panel())
    latest_factor = FormulaCompiler().compile(signal.formula).evaluate(_panel()).iloc[-1]
    highest_symbol = str(latest_factor.idxmax())

    assert tilts[highest_symbol] < 0
    assert max(abs(value) for value in tilts.values()) <= 0.08


class _Response:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = None
        self.response_metadata = {}


class _FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def ainvoke(self, messages):
        return _Response(next(self.responses))


@pytest.mark.asyncio
async def test_workflow_accepts_and_persists_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.signal_discovery.workflow.record_llm_response", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.signal_discovery.workflow.get_cost_tracker",
        lambda: type("Tracker", (), {"is_over_hard_budget": lambda self: False})(),
    )
    response = json.dumps(
        [
            {
                "name": "Short Momentum",
                "formula": "TS_Return(Close, 2)",
                "meaning": "Return persistence",
                "category": "momentum",
                "data_fields_used": ["Close"],
                "operators_used": ["TS_Return"],
            }
        ]
    )
    workflow = SignalDiscoveryWorkflow(
        llm=_FakeLLM([response]),
        num_signals=1,
        max_iterations=1,
        forward_periods=1,
        ic_threshold=0.1,
        p_value_threshold=0.05,
        min_periods=30,
        output_directory=str(tmp_path),
    )

    result = await workflow.run("momentum", _panel())

    assert result.status == "accepted"
    assert result.saved_path is not None
    assert len(list(tmp_path.glob("*.json"))) == 1
    # A single formula was tested -> Bonferroni correction is a no-op (n=1).
    assert result.thresholds["bonferroni_corrected_p_value_threshold"] == pytest.approx(0.05)


def test_bonferroni_correction_tightens_as_more_formulas_are_tested(tmp_path):
    workflow = SignalDiscoveryWorkflow(
        llm=_FakeLLM([]),
        num_signals=1,
        max_iterations=1,
        p_value_threshold=0.05,
        output_directory=str(tmp_path),
    )

    assert workflow._corrected_p_threshold(1) == pytest.approx(0.05)
    assert workflow._corrected_p_threshold(5) == pytest.approx(0.01)

    # A p-value that clears the nominal 0.05 threshold on the first formula tested
    # must not clear it once four more formulas have also been tested this run.
    metrics = _metrics(mean_ic=0.03, p_value=0.02)
    assert workflow._accepted(metrics, workflow._corrected_p_threshold(1)) is True
    assert workflow._accepted(metrics, workflow._corrected_p_threshold(5)) is False
