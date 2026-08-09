from src.market.regime_compatibility import (
    REGIME_STRATEGY_COMPATIBILITY,
    filter_regime_compatible,
    is_regime_compatible,
)
from src.market.scalp_opportunity import ScalpOpportunity


def test_momentum_compatible_with_trending_up():
    compatible, reason = is_regime_compatible("momentum", "trending_up")
    assert compatible is True
    assert "momentum" in reason


def test_breakout_incompatible_with_ranging():
    compatible, reason = is_regime_compatible("breakout", "ranging")
    assert compatible is False
    assert "breakout" in reason


def test_mean_reversion_compatible_with_ranging():
    compatible, _ = is_regime_compatible("mean_reversion", "ranging")
    assert compatible is True


def test_mean_reversion_incompatible_with_trending_up():
    compatible, _ = is_regime_compatible("mean_reversion", "trending_up")
    assert compatible is False


def test_unrecognized_regime_is_permissive_not_restrictive():
    """False negatives here just cost a little LLM spend; false positives would
    silently suppress real opportunities. Must default to permissive."""
    compatible, reason = is_regime_compatible("mean_reversion", "unknown")
    assert compatible is True
    assert "unknown" in reason


def test_table_covers_every_recognized_regime_with_a_nonempty_set():
    for regime, strategies in REGIME_STRATEGY_COMPATIBILITY.items():
        assert strategies, f"regime '{regime}' has an empty compatibility set"


# ---------------------------------------------------------------------------
# filter_regime_compatible: the ScalpOpportunity-level pre-LLM cost filter (req 7)
# ---------------------------------------------------------------------------


def _opportunity(symbol: str, *, regime_compatible: bool) -> ScalpOpportunity:
    return ScalpOpportunity(
        symbol=symbol, direction="BUY", regime_compatible=regime_compatible
    )


def test_filter_partitions_by_precomputed_regime_compatible_flag():
    compatible_opp = _opportunity("RELIANCE", regime_compatible=True)
    incompatible_opp = _opportunity("TCS", regime_compatible=False)

    compatible, rejected = filter_regime_compatible(
        [compatible_opp, incompatible_opp], "ranging"
    )

    assert compatible == [compatible_opp]
    assert len(rejected) == 1
    assert rejected[0].symbol == "TCS"


def test_filter_never_mutates_the_original_opportunity():
    incompatible_opp = _opportunity("TCS", regime_compatible=False)

    _, rejected = filter_regime_compatible([incompatible_opp], "ranging")

    # Original is frozen/untouched; the returned rejected copy carries the new reason.
    assert incompatible_opp.reason == []
    assert rejected[0].reason != []


def test_filter_appends_explanatory_reason_to_rejected_opportunities():
    incompatible_opp = _opportunity("TCS", regime_compatible=False)

    _, rejected = filter_regime_compatible([incompatible_opp], "ranging")

    assert any("ranging" in reason for reason in rejected[0].reason)


def test_filter_preserves_existing_reasons_on_rejection():
    incompatible_opp = ScalpOpportunity(
        symbol="TCS", direction="BUY", regime_compatible=False, reason=["prior note"]
    )

    _, rejected = filter_regime_compatible([incompatible_opp], "ranging")

    assert "prior note" in rejected[0].reason
    assert len(rejected[0].reason) == 2


def test_filter_handles_empty_input():
    compatible, rejected = filter_regime_compatible([], "trending_up")
    assert compatible == []
    assert rejected == []


def test_filter_never_touches_h8_registry_import():
    """Structural guardrail: this module must have no import relationship with the
    H-8 registry at all, so it cannot be silently 'extended' into an admission
    decision -- confirmed by inspecting the module's own imports."""
    import src.market.regime_compatibility as module

    assert "StrategyRegistry" not in dir(module)
