from unittest.mock import MagicMock, patch

from src.core.indicators import Timeframe
from src.markets.nse.strategies.assessment_matrix import TimeframeAssessment
from src.markets.nse.strategies.scalp_confirmation import confirm_multi_timeframe


def _cell(decision: str) -> TimeframeAssessment:
    return TimeframeAssessment(
        timeframe=Timeframe.M5,
        decision=decision,
        score=0.8 if decision == "BUY" else 0.2,
        strategy_consensus=1,
        ml_probability=None,
        regime_compatible=True,
        reasons=[],
    )


def _settings_mock(*, required=3, macro_enabled=True):
    settings = MagicMock()
    settings.scalp_required_mtf_alignment = required
    settings.scalp_macro_filter_enabled = macro_enabled
    return settings


def _full_matrix(overrides: dict | None = None) -> dict:
    matrix = {
        Timeframe.M5: _cell("BUY"),
        Timeframe.M15: _cell("BUY"),
        Timeframe.M30: _cell("BUY"),
        Timeframe.H1: _cell("BUY"),
        Timeframe.H4: _cell("BUY"),
    }
    matrix.update(overrides or {})
    return matrix


def test_full_alignment_passes():
    matrix = _full_matrix()
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=5),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.passed is True
    assert result.aligned_count == 5
    assert result.execution_ok is True
    assert result.primary_ok is True
    assert result.directional_ok is True
    assert result.context_ok is True
    assert result.macro_ok is True


def test_missing_30m_data_fails_closed_never_counted_as_aligned():
    """A timeframe with no data at all (never scanned) must not be silently treated
    as confirming -- this is the exact "fail closed on missing data" requirement."""
    matrix = _full_matrix()
    del matrix[Timeframe.M30]  # 30m simply has no entry -- not WAIT, not REJECT, absent

    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=5),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.directional_ok is False
    assert result.aligned_count == 4
    assert result.passed is False  # needed 5, only got 4


def test_explicit_wait_cell_also_fails_that_role():
    matrix = _full_matrix({Timeframe.M30: _cell("WAIT")})
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=4),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.directional_ok is False
    assert result.aligned_count == 4  # the other 4 still align
    assert result.passed is True  # meets the required=4 bar


def test_below_required_alignment_fails():
    matrix = _full_matrix(
        {Timeframe.M30: _cell("REJECT"), Timeframe.H1: _cell("REJECT")}
    )
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=4),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.aligned_count == 3
    assert result.passed is False


def test_macro_filter_disabled_excludes_4h_from_evaluation_and_denominator():
    matrix = _full_matrix({Timeframe.H4: _cell("REJECT")})  # 4h fails, but disabled
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=4, macro_enabled=False),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.macro_ok is None
    assert result.aligned_count == 4  # 4h's failure never counted against it
    assert result.passed is True
    assert any("disabled" in reason for reason in result.reasons)


def test_macro_filter_enabled_counts_4h_against_alignment():
    matrix = _full_matrix({Timeframe.H4: _cell("REJECT")})
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=5, macro_enabled=True),
    ):
        result = confirm_multi_timeframe(matrix)

    assert result.macro_ok is False
    assert result.aligned_count == 4
    assert result.passed is False


def test_empty_matrix_fails_every_role_closed():
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(required=1),
    ):
        result = confirm_multi_timeframe({})

    assert result.aligned_count == 0
    assert result.passed is False
    assert result.execution_ok is False


def test_result_is_json_serializable():
    matrix = _full_matrix()
    with patch(
        "src.markets.nse.strategies.scalp_confirmation.get_settings",
        return_value=_settings_mock(),
    ):
        result = confirm_multi_timeframe(matrix)

    import json

    json.dumps(result.to_dict())
