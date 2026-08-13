from src.core.indicators import Timeframe
from src.markets.nse.strategies.assessment_matrix import TimeframeAssessment
from src.markets.nse.strategies.scalp_opportunity import ScalpOpportunity


def test_default_opportunity_is_a_reject_with_no_evidence():
    opp = ScalpOpportunity(symbol="RELIANCE", direction="BUY")

    assert opp.final_decision == "REJECT"
    assert opp.entry_quality is None
    assert opp.mtf_confirmation is None
    assert opp.timeframe_states == {}
    assert opp.primary_strategy == ""
    assert opp.primary_timeframe == ""


def test_to_signal_dict_matches_the_shape_agent_nodes_expect():
    """strategy_selection_node/signal_validation_node/risk_compliance_node all read
    signal.get("strategy")/("timeframe")/("trade_horizon")/("confidence")/
    ("risk_reward_ratio") -- this is the one conversion point that must match."""
    opp = ScalpOpportunity(
        symbol="RELIANCE",
        direction="BUY",
        primary_strategy="momentum",
        primary_timeframe="5m",
        entry_price=100.0,
        stop_loss=99.0,
        target_price=102.0,
        expected_r=2.0,
        score=0.75,
        reason=["strong alignment"],
    )

    payload = opp.to_signal_dict()

    assert payload["symbol"] == "RELIANCE"
    assert payload["strategy"] == "momentum"
    assert payload["timeframe"] == "5m"
    assert payload["trade_horizon"] == "SCALP"
    assert payload["signal_type"] == "BUY"
    assert payload["entry_price"] == 100.0
    assert payload["stop_loss"] == 99.0
    assert payload["target_price"] == 102.0
    assert payload["risk_reward_ratio"] == 2.0
    assert payload["confidence"] == 0.75


def test_primary_strategy_and_timeframe_round_trip_through_to_dict():
    opp = ScalpOpportunity(
        symbol="RELIANCE",
        direction="BUY",
        primary_strategy="momentum",
        primary_timeframe="5m",
    )

    payload = opp.to_dict()

    assert payload["primary_strategy"] == "momentum"
    assert payload["primary_timeframe"] == "5m"


def test_to_dict_is_fully_json_shaped_with_no_evaluator_stages_yet():
    opp = ScalpOpportunity(
        symbol="RELIANCE",
        direction="BUY",
        timeframe_states={
            "15m": TimeframeAssessment(
                timeframe=Timeframe.M15,
                decision="BUY",
                score=0.75,
                strategy_consensus=2,
                ml_probability=0.6,
                regime_compatible=True,
                reasons=["2 strategies agree"],
            )
        },
        score=0.75,
        final_decision="ENTER_NOW",
        reason=["strong multi-timeframe alignment"],
        preferred_entry_low=99.5,
        preferred_entry_high=100.2,
        stop_loss=99.0,
        target_price=101.0,
        expected_r=1.8,
    )

    payload = opp.to_dict()

    assert payload["symbol"] == "RELIANCE"
    assert payload["direction"] == "BUY"
    assert payload["timeframe_states"]["15m"]["decision"] == "BUY"
    assert payload["entry_quality"] is None
    assert payload["mtf_confirmation"] is None
    assert payload["final_decision"] == "ENTER_NOW"
    assert payload["expected_r"] == 1.8

    # Must be JSON-serializable with zero extra conversion, matching the pattern
    # every other stats.* field on the dashboard follows.
    import json

    json.dumps(payload)
