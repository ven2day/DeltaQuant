"""Deterministic Buy/Wait/Reject policy for permanent paper trading.

Qwen and news enrich the later review, but only candidates with aligned local evidence
reach that stage. The policy never manufactures a trade when evidence is absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd

from src.core.indicators import Timeframe
from src.observability.signal_funnel import RejectionReason


class CandidateAction(str, Enum):
    BUY = "BUY"
    WAIT = "WAIT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CandidateDecision:
    symbol: str
    action: CandidateAction
    rationale: list[str]
    entry_price: float
    stop_loss: float
    target_price: float
    target_pct: float
    target_realistic: bool
    volume_ratio: float
    vwap: float
    higher_timeframes_aligned: int
    ml_direction: str
    ml_confidence: float
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["reason_codes"] = list(self.reason_codes)
        return data


def _ema(frame: pd.DataFrame, span: int) -> float:
    return float(frame["close"].ewm(span=span, adjust=False).mean().iloc[-1])


def _bullish_structure(frame: pd.DataFrame) -> bool:
    if len(frame) < 50:
        return False
    close = float(frame["close"].iloc[-1])
    ema20 = _ema(frame, 20)
    ema50 = _ema(frame, 50)
    return close > ema20 > ema50


def _session_vwap(frame: pd.DataFrame) -> float:
    recent = frame.iloc[-75:]
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    volume = recent["volume"].clip(lower=0)
    total = float(volume.sum())
    if total <= 0:
        return float(recent["close"].iloc[-1])
    return float((typical * volume).sum() / total)


def _target_realism(frame_h4: pd.DataFrame, entry: float, target_pct: float) -> bool:
    if entry <= 0 or len(frame_h4) < 20:
        return False
    previous_close = frame_h4["close"].shift(1)
    true_range = pd.concat(
        [
            frame_h4["high"] - frame_h4["low"],
            (frame_h4["high"] - previous_close).abs(),
            (frame_h4["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = float(true_range.iloc[-14:].mean() / entry * 100)
    # A target above roughly 2.5 recent 4h ATRs is displayed as aspirational rather than
    # silently treated as likely. It remains the framework target; the policy may WAIT.
    return target_pct <= max(atr_pct * 2.5, 0.5)


def evaluate_long_candidate(
    signal: dict[str, Any],
    histories: dict[Timeframe, pd.DataFrame | None],
    prediction: Any | None,
    *,
    target_pct: float = 3.5,
    min_ml_confidence: float = 0.55,
    min_volume_ratio: float = 1.0,
    required_higher_timeframes: int = 2,
    require_ml: bool = False,
) -> CandidateDecision:
    """Return an explainable local decision without invoking external services.

    ``require_ml`` is a versioned strategy-policy property, not a global runtime gate.
    A strategy validated without ML continues when the exact artifact is unavailable;
    a strategy version explicitly declaring ``requires_ml`` calls this with True and
    fails closed unless its exact validated artifact supports the setup.
    """
    symbol = str(signal.get("symbol", ""))
    side = str(signal.get("signal_type", "BUY")).upper()
    rationale: list[str] = []
    entry = float(signal.get("entry_price", 0.0) or 0.0)
    stop = float(signal.get("stop_loss", 0.0) or 0.0)
    target = entry * (1.0 + target_pct / 100.0) if entry > 0 else 0.0

    prediction_data = prediction.to_dict() if hasattr(prediction, "to_dict") else prediction or {}
    ml_direction = str(prediction_data.get("direction", "unknown"))
    ml_confidence = float(prediction_data.get("confidence", 0.0) or 0.0)

    five = histories.get(Timeframe.M5)
    h4 = histories.get(Timeframe.H4)
    if side != "BUY":
        return CandidateDecision(
            symbol,
            CandidateAction.REJECT,
            ["Permanent long-only policy rejects new SELL/short signals."],
            entry,
            stop,
            target,
            target_pct,
            False,
            0.0,
            0.0,
            0,
            ml_direction,
            ml_confidence,
            (RejectionReason.LONG_ONLY.value,),
        )
    if entry <= 0 or stop <= 0 or stop >= entry:
        return CandidateDecision(
            symbol,
            CandidateAction.REJECT,
            ["Entry and stop levels are invalid for a long paper trade."],
            entry,
            stop,
            target,
            target_pct,
            False,
            0.0,
            0.0,
            0,
            ml_direction,
            ml_confidence,
            (RejectionReason.INVALID_LEVELS.value,),
        )
    if five is None or len(five) < 50 or h4 is None or len(h4) < 20:
        return CandidateDecision(
            symbol,
            CandidateAction.WAIT,
            ["Waiting for sufficient coherent 5m and 4h history."],
            entry,
            stop,
            target,
            target_pct,
            False,
            0.0,
            0.0,
            0,
            ml_direction,
            ml_confidence,
            (RejectionReason.INSUFFICIENT_HISTORY.value,),
        )

    five = five.copy()
    last = five.iloc[-1]
    vwap = _session_vwap(five)
    previous_volume = float(five["volume"].iloc[-21:-1].mean())
    volume_ratio = float(last["volume"] / previous_volume) if previous_volume > 0 else 0.0
    five_confirmed = (
        float(last["close"]) > float(last["open"])
        and float(last["close"]) >= vwap
        and float(last["close"]) > _ema(five, 20)
    )
    rationale.append(
        "5m confirmation passed (bullish close above VWAP/EMA20)."
        if five_confirmed
        else "5m confirmation is not yet bullish above VWAP and EMA20."
    )

    aligned = sum(
        1
        for timeframe in (Timeframe.M30, Timeframe.H1, Timeframe.H4)
        if (frame := histories.get(timeframe)) is not None and _bullish_structure(frame)
    )
    rationale.append(
        f"Higher-timeframe trend aligned on {aligned}/3 frames; "
        f"requires {required_higher_timeframes}."
    )
    rationale.append(
        f"5m volume is {volume_ratio:.2f}x its 20-bar average "
        f"(minimum {min_volume_ratio:.2f}x)."
    )
    if require_ml:
        rationale.append(
            f"ML view is {ml_direction} at {ml_confidence:.0%} confidence "
            f"(minimum upward confidence {min_ml_confidence:.0%})."
        )
    else:
        rationale.append("ML is advisory/abstained for this strategy version.")

    realistic = _target_realism(h4, entry, target_pct)
    rationale.append(
        f"The {target_pct:.1f}% framework target is consistent with recent 4h range."
        if realistic
        else f"The {target_pct:.1f}% framework target is ambitious for recent 4h volatility."
    )

    deterministic_gates_pass = (
        five_confirmed
        and volume_ratio >= min_volume_ratio
        and aligned >= required_higher_timeframes
    )
    ml_gate_passed = ml_direction == "up" and ml_confidence >= min_ml_confidence
    gates_pass = deterministic_gates_pass and (ml_gate_passed if require_ml else True)
    action = CandidateAction.BUY if gates_pass else CandidateAction.WAIT
    reason_codes: list[str] = []
    if not five_confirmed:
        reason_codes.append(RejectionReason.WAIT_PULLBACK.value)
    if volume_ratio < min_volume_ratio:
        reason_codes.append(RejectionReason.LOW_RELATIVE_VOLUME.value)
    if aligned < required_higher_timeframes:
        reason_codes.append(RejectionReason.MTF_CONFLICT.value)
    if require_ml and not ml_gate_passed:
        reason_codes.append(
            RejectionReason.ML_ABSTAIN.value
            if ml_direction in {"unknown", "abstain", ""} or ml_confidence <= 0
            else RejectionReason.ML_BELOW_THRESHOLD.value
        )
    return CandidateDecision(
        symbol,
        action,
        rationale,
        entry,
        stop,
        target,
        target_pct,
        realistic,
        volume_ratio,
        vwap,
        aligned,
        ml_direction,
        ml_confidence,
        tuple(reason_codes),
    )
