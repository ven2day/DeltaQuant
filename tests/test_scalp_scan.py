"""
Orchestration tests for src/markets/nse/strategies/scalp_scan.py's run_scalp_scan() -- the
extracted, directly-testable core of run_live_trading.py's Stage-10 scalp scan
branch. Verifies funnel counters and final ranked output using plain async stub
fetch functions instead of a live HistoryManager/dashboard session.
"""

import contextlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.prediction import PredictionSignal
from src.core.candidates import SignalStrength, SignalType, StrategyType, TradingSignal
from src.core.indicators import Timeframe
from src.markets.nse.strategies.scalp_scan import (
    empty_funnel,
    run_scalp_scan,
    run_scalp_scan_detailed,
)


@dataclass
class _FakeIndicators:
    timeframe: Timeframe
    ema: dict | None = None
    atr: float | None = None


class _FakeSignalEngine:
    """generate_signals() returns whatever was pre-registered for that
    indicator's timeframe -- lets tests control exactly which timeframes fire."""

    def __init__(self, signals_by_timeframe: dict[Timeframe, list[TradingSignal]]):
        self.signals_by_timeframe = signals_by_timeframe

    def generate_signals(self, indicators: _FakeIndicators) -> list[TradingSignal]:
        return list(self.signals_by_timeframe.get(indicators.timeframe, []))


@dataclass
class _Performance:
    total_trades: int = 0
    winning_trades: int = 0


class _Tracker:
    def get_strategy_performance(self, strategy, regime, lookback_days=30):
        return _Performance()


def _signal(
    symbol="RELIANCE",
    *,
    timeframe,
    strategy=StrategyType.MOMENTUM,
    confidence=0.9,
    signal_type=SignalType.BUY,
    adx=30.0,
    plus_di=28.0,
    minus_di=12.0,
) -> TradingSignal:
    return TradingSignal(
        signal_id=f"SIG-{symbol}-{timeframe.value}",
        symbol=symbol,
        signal_type=signal_type,
        strength=SignalStrength.STRONG,
        strategy=strategy,
        timeframe=timeframe,
        entry_price=100.0,
        stop_loss=99.0,
        target_price=102.0,
        risk_reward_ratio=2.0,
        position_size_pct=5.0,
        confidence=confidence,
        indicators={"trend": {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}},
    )


def _quiet_frame(bars: int = 30) -> pd.DataFrame:
    """A calm, well-volumed consolidation -- entry_quality's ENTER_NOW fixture from
    test_entry_quality.py, reused here so the full pipeline reaches ENTER_NOW."""
    rows = [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(bars - 1)]
    rows.append([100.0, 100.4, 99.9, 100.2, 1500])
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _settings_mock(**overrides) -> MagicMock:
    settings = MagicMock()
    settings.long_only = True
    settings.scalp_max_active_symbols = 5
    settings.scalp_matrix_reject_score = 0.35
    settings.scalp_min_confidence = 0.6
    settings.scalp_required_mtf_alignment = 5
    settings.scalp_macro_filter_enabled = True
    settings.scalp_min_volume_ratio = 1.2
    settings.scalp_vwap_max_distance_pct = 0.5
    settings.scalp_ema9_max_distance_pct = 0.6
    settings.scalp_atr_extension_max_multiple = 1.5
    settings.scalp_wick_ratio_max = 0.6
    settings.scalp_swing_lookback_bars = 40
    settings.scalp_breakout_retest_lookback_bars = 12
    settings.scalp_resistance_min_distance_pct = 0.3
    settings.scalping_swing_threshold_pct = 0.5
    settings.scalp_ranking_weight_entry_quality = 1 / 6
    settings.scalp_ranking_weight_mtf_alignment = 1 / 6
    settings.scalp_ranking_weight_volume_liquidity = 1 / 6
    settings.scalp_ranking_weight_regime = 1 / 6
    settings.scalp_ranking_weight_historical_expectancy = 1 / 6
    settings.scalp_ranking_weight_ml_probability = 1 / 6
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


@contextlib.contextmanager
def _patched_settings(settings):
    """Every deterministic scalp module calls get_settings() independently
    (`from src.config import get_settings`), so each needs its own patch target."""
    with contextlib.ExitStack() as stack:
        for target in (
            "src.markets.nse.strategies.assessment_matrix.get_settings",
            "src.markets.nse.strategies.scalp_confirmation.get_settings",
            "src.markets.nse.strategies.entry_quality.get_settings",
            "src.markets.nse.strategies.scalp_ranking.get_settings",
        ):
            stack.enter_context(patch(target, return_value=settings))
        yield


ALL_TIMEFRAMES = [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4]


def _full_alignment_engine(symbol="RELIANCE") -> _FakeSignalEngine:
    return _FakeSignalEngine({tf: [_signal(symbol, timeframe=tf)] for tf in ALL_TIMEFRAMES})


async def _run(
    scan_symbols,
    engine,
    *,
    settings=None,
    tracker=None,
    frames: dict | None = None,
    predictions: dict | None = None,
    detailed: bool = False,
):
    settings = settings or _settings_mock()
    tracker = tracker or _Tracker()
    frames = frames or {}

    async def fetch_indicators(symbol, tf):
        return _FakeIndicators(timeframe=tf)

    async def fetch_5m_frame(symbol):
        return frames.get(symbol, _quiet_frame())

    async def fetch_prediction(symbol, timeframe, strategy_version):
        assert strategy_version in {strategy.value for strategy in StrategyType}
        return (predictions or {}).get((symbol, timeframe))

    with _patched_settings(settings):
        scan = run_scalp_scan_detailed if detailed else run_scalp_scan
        return await scan(
            scan_symbols,
            settings=settings,
            scalp_signal_engine=engine,
            scalp_confirmation_timeframes=ALL_TIMEFRAMES,
            scalp_origin_timeframes=[Timeframe.M5],
            scalp_cycle_regime="trending_up",
            performance_tracker=tracker,
            fetch_indicators=fetch_indicators,
            fetch_5m_frame=fetch_5m_frame,
            fetch_prediction=fetch_prediction if predictions is not None else None,
        )


@pytest.mark.asyncio
async def test_no_signals_produces_empty_result():
    engine = _FakeSignalEngine({})  # never fires

    opportunities, funnel = await _run(["RELIANCE"], engine)

    assert opportunities == []
    assert funnel == empty_funnel()


@pytest.mark.asyncio
async def test_full_alignment_produces_one_ranked_opportunity():
    engine = _full_alignment_engine()

    opportunities, funnel = await _run(["RELIANCE"], engine)

    assert funnel["raw_triggers"] == 5  # one signal per timeframe
    assert funnel["consolidated"] == 5  # one strategy each, no agreement-merging
    assert funnel["mtf_candidates"] == 1
    assert funnel["entry_quality_passed"] == 1
    assert funnel["regime_compatible"] == 1
    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp.symbol == "RELIANCE"
    assert opp.primary_strategy == "momentum"
    assert opp.primary_timeframe == "5m"
    assert opp.final_decision == "ENTER_NOW"
    assert opp.score > 0


@pytest.mark.asyncio
async def test_detailed_scan_blends_local_ml_and_returns_it_for_llm_reuse():
    engine = _full_alignment_engine()
    prediction = PredictionSignal(
        symbol="RELIANCE",
        direction="up",
        confidence=0.78,
        predicted_change_pct=0.25,
        reasoning="walk-forward local history",
        oos_samples=120,
    )
    result = await _run(
        ["RELIANCE"],
        engine,
        predictions={("RELIANCE", timeframe): prediction for timeframe in ALL_TIMEFRAMES},
        detailed=True,
    )

    assert result.funnel["ml_scored"] == len(ALL_TIMEFRAMES)
    assert result.predictions[("RELIANCE", "5m")] is prediction
    assert result.opportunities[0].ml_probability == pytest.approx(0.78)
    assert result.symbol_statuses[0].ml_probability == pytest.approx(0.78)


@pytest.mark.asyncio
async def test_insufficient_mtf_alignment_stops_before_entry_quality():
    # Only 5m fires -- 15m/30m/1h/4h have no data at all, required=5 can't be met.
    engine = _FakeSignalEngine({Timeframe.M5: [_signal("RELIANCE", timeframe=Timeframe.M5)]})

    opportunities, funnel = await _run(["RELIANCE"], engine)

    assert funnel["raw_triggers"] == 1
    assert funnel["mtf_candidates"] == 0
    assert funnel["entry_quality_passed"] == 0
    assert opportunities == []


@pytest.mark.asyncio
async def test_missing_higher_timeframe_data_is_not_reported_as_market_conflict():
    settings = _settings_mock()

    async def fetch_indicators(symbol, tf):
        return _FakeIndicators(timeframe=tf) if tf is Timeframe.M5 else None

    async def fetch_5m_frame(symbol):
        return _quiet_frame()

    with _patched_settings(settings):
        result = await run_scalp_scan_detailed(
            ["RELIANCE"],
            settings=settings,
            scalp_signal_engine=_FakeSignalEngine(
                {Timeframe.M5: [_signal("RELIANCE", timeframe=Timeframe.M5)]}
            ),
            scalp_confirmation_timeframes=ALL_TIMEFRAMES,
            scalp_origin_timeframes=[Timeframe.M5],
            scalp_cycle_regime="trending_up",
            performance_tracker=_Tracker(),
            fetch_indicators=fetch_indicators,
            fetch_5m_frame=fetch_5m_frame,
        )

    assert result.funnel["mtf_missing_data"] == 1
    assert result.funnel["mtf_conflict"] == 0
    assert result.symbol_statuses[0].stage == "MTF_MISSING_DATA"


@pytest.mark.asyncio
async def test_available_but_opposing_timeframes_are_reported_as_conflict():
    result = await _run(
        ["RELIANCE"],
        _FakeSignalEngine({Timeframe.M5: [_signal("RELIANCE", timeframe=Timeframe.M5)]}),
        detailed=True,
    )

    assert result.funnel["mtf_missing_data"] == 0
    assert result.funnel["mtf_conflict"] == 1
    assert result.symbol_statuses[0].stage == "MTF_CONFLICT"


@pytest.mark.asyncio
async def test_failed_deterministic_qualification_never_becomes_an_ml_candidate():
    engine = _FakeSignalEngine({Timeframe.M5: [_signal("RELIANCE", timeframe=Timeframe.M5)]})
    prediction = PredictionSignal(
        symbol="RELIANCE",
        direction="up",
        confidence=0.99,
        predicted_change_pct=1.0,
        reasoning="must never be fetched",
        oos_samples=500,
    )

    result = await _run(
        ["RELIANCE"],
        engine,
        predictions={("RELIANCE", timeframe): prediction for timeframe in ALL_TIMEFRAMES},
        detailed=True,
    )

    assert result.funnel["technical_setups"] == 1
    assert result.funnel["ml_candidates"] == 0
    assert result.funnel["ml_scored"] == 0
    assert result.predictions == {}


@pytest.mark.asyncio
async def test_entry_quality_reject_produces_no_opportunity():
    engine = _full_alignment_engine()
    # A wide-upper-wick last candle -- entry_quality.py's REJECT trigger.
    rejecting_frame = pd.DataFrame(
        [[100.0, 100.3, 99.8, 100.0, 1000] for _ in range(24)]
        + [[100.0, 105.0, 99.9, 100.2, 1500]],
        columns=["open", "high", "low", "close", "volume"],
    )

    opportunities, funnel = await _run(["RELIANCE"], engine, frames={"RELIANCE": rejecting_frame})

    assert funnel["mtf_candidates"] == 1  # got past confirmation
    assert funnel["entry_quality_passed"] == 0  # but rejected on entry timing
    assert opportunities == []


@pytest.mark.asyncio
async def test_regime_incompatible_signal_never_originates_an_opportunity():
    # mean_reversion is not compatible with a trending_up-inferred regime (see
    # regime_compatibility.py's table). assessment_matrix.py's BUY decision
    # requires score>=bar AND compatible (Stage 4) -- so an incompatible cell can
    # never read "BUY", which means it can never become an origin timeframe, which
    # means no opportunity is ever constructed. Regime incompatibility is therefore
    # blocked a layer earlier than filter_regime_compatible even runs; this test
    # locks in that stronger behavior instead of only relying on the later filter.
    engine = _FakeSignalEngine(
        {
            tf: [_signal("RELIANCE", timeframe=tf, strategy=StrategyType.MEAN_REVERSION)]
            for tf in ALL_TIMEFRAMES
        }
    )

    opportunities, funnel = await _run(["RELIANCE"], engine)

    assert funnel["raw_triggers"] == 5
    assert funnel["mtf_candidates"] == 0  # no cell ever reads BUY -> confirmation fails
    assert funnel["entry_quality_passed"] == 0
    assert funnel["regime_compatible"] == 0
    assert opportunities == []


@pytest.mark.asyncio
async def test_filter_regime_compatible_is_a_defensive_noop_given_matrix_gating():
    """Documents WHY filter_regime_compatible (Stage 8) rarely rejects anything in
    practice when wired through this pipeline: every opportunity that reaches it
    was only constructed because its origin cell already read "BUY", which itself
    required regime_compatible=True (Stage 4). Kept as defense-in-depth, not the
    primary mechanism -- this test protects that invariant from silently drifting."""
    engine = _full_alignment_engine()

    opportunities, funnel = await _run(["RELIANCE"], engine)

    assert len(opportunities) == 1
    assert opportunities[0].regime_compatible is True
    assert funnel["entry_quality_passed"] == funnel["regime_compatible"]


@pytest.mark.asyncio
async def test_long_only_drops_sell_signals_before_consolidation():
    engine = _FakeSignalEngine(
        {
            tf: [_signal("RELIANCE", timeframe=tf, signal_type=SignalType.SELL)]
            for tf in ALL_TIMEFRAMES
        }
    )

    opportunities, funnel = await _run(
        ["RELIANCE"], engine, settings=_settings_mock(long_only=True)
    )

    assert funnel["raw_triggers"] == 0
    assert opportunities == []


@pytest.mark.asyncio
async def test_valid_results_are_not_truncated_by_review_capacity():
    # run_scalp_scan calls generate_signals(indicators) once per (symbol, timeframe)
    # inside its own per-symbol loop, so a single fixed-per-timeframe engine (not
    # varying by symbol) still produces one independent opportunity per scan symbol
    # -- ScalpOpportunity.symbol is stamped from the outer loop variable, not from
    # whatever's on the fake signal itself.
    symbols = ["A", "B", "C"]
    engine = _full_alignment_engine()

    opportunities, funnel = await _run(
        symbols,
        engine,
        settings=_settings_mock(scalp_max_active_symbols=2),
    )

    assert funnel["regime_compatible"] == 3  # all 3 symbols qualified
    assert len(opportunities) == 3


@pytest.mark.asyncio
async def test_detailed_scan_keeps_one_status_for_every_symbol():
    result = await _run(
        ["RELIANCE", "TCS", "INFY"],
        _FakeSignalEngine({}),
        detailed=True,
    )

    assert result.opportunities == []
    assert [status.symbol for status in result.symbol_statuses] == ["INFY", "RELIANCE", "TCS"]
    assert {status.decision for status in result.symbol_statuses} == {"NO_BUY"}
    assert {status.stage for status in result.symbol_statuses} == {"NO_BUY_TRIGGER"}


@pytest.mark.asyncio
async def test_detailed_scan_keeps_all_buy_setups_even_when_review_capacity_is_lower():
    result = await _run(
        ["A", "B", "C"],
        _full_alignment_engine(),
        settings=_settings_mock(scalp_max_active_symbols=2),
        detailed=True,
    )

    assert len(result.opportunities) == 3
    assert len(result.symbol_statuses) == 3
    assert all(status.decision == "BUY" for status in result.symbol_statuses)
    assert sum(status.selected_for_review for status in result.symbol_statuses) == 3


@pytest.mark.asyncio
async def test_detailed_scan_uses_no_data_only_when_all_timeframes_are_missing():
    settings = _settings_mock()

    async def fetch_indicators(symbol, tf):
        if symbol == "MISSING":
            return None
        return _FakeIndicators(timeframe=tf)

    async def fetch_5m_frame(symbol):
        return _quiet_frame()

    with _patched_settings(settings):
        result = await run_scalp_scan_detailed(
            ["MISSING", "QUIET"],
            settings=settings,
            scalp_signal_engine=_FakeSignalEngine({}),
            scalp_confirmation_timeframes=ALL_TIMEFRAMES,
            scalp_origin_timeframes=[Timeframe.M5],
            scalp_cycle_regime="trending_up",
            performance_tracker=_Tracker(),
            fetch_indicators=fetch_indicators,
            fetch_5m_frame=fetch_5m_frame,
        )

    by_symbol = {status.symbol: status for status in result.symbol_statuses}
    assert by_symbol["MISSING"].decision == "NO_DATA"
    assert by_symbol["MISSING"].stage == "DATA_UNAVAILABLE"
    assert by_symbol["QUIET"].decision == "NO_BUY"
