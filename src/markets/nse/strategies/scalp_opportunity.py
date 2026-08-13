"""
``ScalpOpportunity`` (req 11): the canonical domain object for one scalp candidate,
carried unchanged from the scanner through ranking, the agent graph, the dashboard,
and execution -- so every stage of the scalp pipeline (and the UI) describes the same
opportunity the same way instead of each layer inventing its own shape.

``entry_quality``/``mtf_confirmation`` are populated by the EntryQualityEvaluator
(Stage 6) and multi-timeframe confirmation (Stage 7) respectively; both are optional
here (``None`` until those stages run) so this object can be constructed as soon as
the assessment matrix exists, without a forward dependency on modules built later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .assessment_matrix import TimeframeAssessment

if TYPE_CHECKING:
    from .entry_quality import EntryQualityResult
    from .scalp_confirmation import ScalpConfirmationResult

FinalDecision = Literal["ENTER_NOW", "WAIT_PULLBACK", "WAIT_BREAKOUT", "REJECT"]


@dataclass(frozen=True)
class ScalpOpportunity:
    """One candidate scalp trade, symbol + direction, with its full evidence trail."""

    symbol: str
    direction: Literal["BUY", "SELL"]
    # Keyed by timeframe value string ("5m", "15m", "30m", "1h", "4h") rather than
    # the Timeframe enum itself, matching how every other dict on this object
    # (to_dict()) is meant to serialize directly for the dashboard/webui.
    timeframe_states: dict[str, TimeframeAssessment] = field(default_factory=dict)

    # Which strategy/timeframe this opportunity is primarily evidenced by -- needed
    # downstream by the scalp ranker's historical-expectancy lookup (scalp_ranking.py)
    # and, once this opportunity is converted to a signal dict for the agent graph, by
    # strategy eligibility and risk_compliance's final backstop, both of
    # which key off `signal.get("strategy")`/`signal.get("timeframe")`. Empty string
    # defaults keep this object constructible before those consumers exist.
    primary_strategy: str = ""
    primary_timeframe: str = ""

    entry_quality: EntryQualityResult | None = None
    mtf_confirmation: ScalpConfirmationResult | None = None
    regime_compatible: bool = False
    ml_probability: float | None = None
    historical_scalp_expectancy: float | None = None

    score: float = 0.0
    final_decision: FinalDecision = "REJECT"
    reason: list[str] = field(default_factory=list)

    entry_price: float = 0.0
    preferred_entry_low: float = 0.0
    preferred_entry_high: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    expected_r: float = 0.0
    signal_candle_timestamp: str = ""

    def to_signal_dict(self) -> dict[str, Any]:
        """Convert to the same dict shape ``TradingSignal.to_dict()`` produces --
        the shape ``strategy_selection_node``/``signal_validation_node``/
        ``risk_compliance_node`` all read (``signal.get("strategy")``,
        ``signal.get("timeframe")``, ``signal.get("trade_horizon")``, etc.). This is
        the ONE place a ScalpOpportunity crosses into the agent graph, so every
        node downstream sees an ordinary signal dict, not a parallel schema.
        """
        return {
            "signal_id": f"SCALP-{self.symbol}-{self.primary_timeframe}",
            "symbol": self.symbol,
            "signal_type": self.direction,
            "strategy": self.primary_strategy,
            "timeframe": self.primary_timeframe,
            "trade_horizon": "SCALP",
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "risk_reward_ratio": self.expected_r,
            "confidence": self.score,
            "reasons": self.reason,
            "entry_state": self.final_decision,
            "entry_quality_score": self.score * 100.0,
            "mtf_confirmation_passed": bool(
                self.mtf_confirmation is not None and self.mtf_confirmation.passed
            ),
            "regime_compatible": self.regime_compatible,
            "ml_probability": self.ml_probability,
            "expected_r": self.expected_r,
            "signal_candle_timestamp": self.signal_candle_timestamp,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "timeframe_states": {
                tf: assessment.to_dict() for tf, assessment in self.timeframe_states.items()
            },
            "primary_strategy": self.primary_strategy,
            "primary_timeframe": self.primary_timeframe,
            "entry_quality": (
                self.entry_quality.to_dict() if self.entry_quality is not None else None
            ),
            "mtf_confirmation": (
                self.mtf_confirmation.to_dict() if self.mtf_confirmation is not None else None
            ),
            "regime_compatible": self.regime_compatible,
            "ml_probability": self.ml_probability,
            "historical_scalp_expectancy": self.historical_scalp_expectancy,
            "score": self.score,
            "final_decision": self.final_decision,
            "reason": self.reason,
            "entry_price": self.entry_price,
            "preferred_entry_low": self.preferred_entry_low,
            "preferred_entry_high": self.preferred_entry_high,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "expected_r": self.expected_r,
            "signal_candle_timestamp": self.signal_candle_timestamp,
        }
