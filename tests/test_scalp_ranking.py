from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.markets.nse.strategies.scalp_confirmation import ScalpConfirmationResult
from src.markets.nse.strategies.scalp_opportunity import ScalpOpportunity
from src.markets.nse.strategies.scalp_ranking import (
    ScalpRankingWeights,
    rank_scalp_opportunities,
    scalp_performance_key,
)


@dataclass
class _Performance:
    total_trades: int
    winning_trades: int


class _Tracker:
    """Records the exact (strategy, regime, lookback_days) key each lookup used, so
    tests can assert the scalp:: namespacing is actually applied."""

    def __init__(self, total: int = 0, winners: int = 0):
        self.performance = _Performance(total, winners)
        self.calls: list[tuple[str, str, int]] = []

    def get_strategy_performance(self, strategy, regime, lookback_days=30):
        self.calls.append((strategy, regime, lookback_days))
        return self.performance


def _confirmation(aligned=4, required=4) -> ScalpConfirmationResult:
    return ScalpConfirmationResult(
        execution_ok=True,
        primary_ok=True,
        directional_ok=True,
        context_ok=True,
        macro_ok=None,
        aligned_count=aligned,
        required=required,
        passed=aligned >= required,
    )


class _FakeEntryQuality:
    def __init__(self, status="ENTER_NOW", relative_volume=2.0):
        self.status = status
        self.relative_volume = relative_volume


def _opportunity(
    *,
    status="ENTER_NOW",
    relative_volume=2.0,
    aligned=4,
    required=4,
    regime_compatible=True,
    ml_probability=None,
    strategy="momentum",
) -> ScalpOpportunity:
    return ScalpOpportunity(
        symbol="RELIANCE",
        direction="BUY",
        entry_quality=_FakeEntryQuality(status, relative_volume),
        mtf_confirmation=_confirmation(aligned, required),
        regime_compatible=regime_compatible,
        ml_probability=ml_probability,
        primary_strategy=strategy,
    )


EQUAL_WEIGHTS = ScalpRankingWeights(
    entry_quality=1 / 6,
    mtf_alignment=1 / 6,
    volume_liquidity=1 / 6,
    regime=1 / 6,
    historical_expectancy=1 / 6,
    ml_probability=1 / 6,
)


def _settings_mock(min_volume_ratio=1.2):
    settings = MagicMock()
    settings.scalp_min_volume_ratio = min_volume_ratio
    return settings


def test_scalp_performance_key_is_namespaced():
    assert scalp_performance_key("momentum") == "scalp::momentum"
    assert scalp_performance_key("momentum") != "momentum"


def test_historical_win_probability_uses_namespaced_key_not_bare_strategy():
    tracker = _Tracker(total=0, winners=0)
    opp = _opportunity(strategy="momentum")

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert tracker.calls[0][0] == "scalp::momentum"


def test_cold_start_reads_as_neutral_fifty_percent_not_zero():
    tracker = _Tracker(total=0, winners=0)
    opp = _opportunity()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].historical_win_probability == 0.5
    assert ranked[0].historical_sample_size == 0


def test_missing_ml_probability_defaults_to_neutral_not_zero():
    tracker = _Tracker()
    opp = _opportunity(ml_probability=None)

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].ml_probability_score == 0.5


def test_entry_quality_score_maps_status_to_expected_values():
    tracker = _Tracker()
    cases = {"ENTER_NOW": 1.0, "WAIT_BREAKOUT": 0.5, "WAIT_PULLBACK": 0.5, "REJECT": 0.0}

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        for status, expected in cases.items():
            opp = _opportunity(status=status)
            ranked = rank_scalp_opportunities(
                [opp], tracker, "trending_up", weights=EQUAL_WEIGHTS
            )
            assert ranked[0].entry_quality_score == expected, status


def test_missing_entry_quality_scores_zero_not_a_crash():
    opp = ScalpOpportunity(symbol="RELIANCE", direction="BUY")  # entry_quality=None
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].entry_quality_score == 0.0
    assert ranked[0].volume_liquidity_score == 0.0


def test_mtf_alignment_score_capped_at_one():
    opp = _opportunity(aligned=10, required=4)  # more aligned than required
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].mtf_alignment_score == 1.0


def test_volume_liquidity_score_capped_at_one():
    opp = _opportunity(relative_volume=100.0)  # absurdly high
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].volume_liquidity_score == 1.0


def test_regime_incompatible_scores_zero_on_that_component():
    opp = _opportunity(regime_compatible=False)
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    assert ranked[0].regime_score == 0.0


def test_stronger_opportunity_ranks_first():
    strong = _opportunity(status="ENTER_NOW", relative_volume=3.0, regime_compatible=True)
    weak = _opportunity(status="REJECT", relative_volume=0.1, regime_compatible=False)
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities(
            [weak, strong], tracker, "trending_up", weights=EQUAL_WEIGHTS
        )

    assert ranked[0].opportunity is strong
    assert ranked[0].rank_score > ranked[1].rank_score


def test_rank_score_matches_manual_weighted_sum():
    opp = _opportunity(
        status="ENTER_NOW",  # entry_quality_score=1.0
        relative_volume=2.4,  # -> volume_liquidity_score=1.0 (ceiling=2*1.2=2.4)
        aligned=4,
        required=4,  # -> mtf_alignment_score=1.0
        regime_compatible=True,  # -> regime_score=1.0
        ml_probability=0.8,  # -> ml_probability_score=0.8
    )
    tracker = _Tracker(total=0, winners=0)  # -> historical_win_probability=0.5

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up", weights=EQUAL_WEIGHTS)

    expected = (1 / 6) * (1.0 + 1.0 + 1.0 + 1.0 + 0.5 + 0.8)
    assert ranked[0].rank_score == expected


def test_weights_default_to_settings_when_not_passed():
    settings = _settings_mock()
    settings.scalp_ranking_weight_entry_quality = 1.0
    settings.scalp_ranking_weight_mtf_alignment = 0.0
    settings.scalp_ranking_weight_volume_liquidity = 0.0
    settings.scalp_ranking_weight_regime = 0.0
    settings.scalp_ranking_weight_historical_expectancy = 0.0
    settings.scalp_ranking_weight_ml_probability = 0.0
    opp = _opportunity(status="ENTER_NOW")
    tracker = _Tracker()

    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=settings):
        ranked = rank_scalp_opportunities([opp], tracker, "trending_up")

    assert ranked[0].rank_score == 1.0  # only entry_quality (weight=1.0) contributes


def test_empty_input_returns_empty_list():
    with patch("src.markets.nse.strategies.scalp_ranking.get_settings", return_value=_settings_mock()):
        assert rank_scalp_opportunities([], _Tracker(), "trending_up") == []
