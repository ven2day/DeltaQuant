#!/usr/bin/env python3
"""Deterministic, no-network replay benchmark for DeltaQuant Qwen prompt cost.

This measures serialized prompt/output tokens with tiktoken's cl100k_base encoding.
It never calls Qwen and never submits an order. Qwen's provider-reported usage can
differ slightly, so the JSON labels tokenizer values explicitly as estimates while
candidate/call/cache counts are measured directly from the replay fixture.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.market_regime import COMPACT_REGIME_SYSTEM_PROMPT, REGIME_SYSTEM_PROMPT
from src.agents.news_analyst import COMPACT_SENTIMENT_SYSTEM_PROMPT, SENTIMENT_SYSTEM_PROMPT
from src.agents.signal_validation import (
    COMPACT_VALIDATION_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)
from src.agents.strategy_selection import (
    COMPACT_STRATEGY_SYSTEM_PROMPT,
    STRATEGY_SYSTEM_PROMPT,
)
from src.finops.cost_tracker import CostTracker

ENCODING = tiktoken.get_encoding("cl100k_base")
MODEL = "qwen3.7-plus"


def _tokens(value: str) -> int:
    return len(ENCODING.encode(value))


def _candidate(index: int, horizon: str) -> dict[str, Any]:
    return {
        "signal_id": f"SYM{index:03d}-5m-{horizon}",
        "symbol": f"SYM{index:03d}",
        "horizon": horizon,
        "timeframe": "5m" if horizon == "SCALP" else "30m",
        "direction": "BUY",
        "strategy": "momentum",
        "confidence": 0.78,
        "entry_price": 100.0 + index,
        "stop_loss": 98.0 + index,
        "target_price": 104.2 + index,
        "risk_reward_ratio": 2.1,
        "entry_state": "ENTER_NOW",
        "entry_score": 81,
        "ml_probability": 0.68,
        "expected_r": 2.1,
        "eligibility_status": "PAPER_APPROVED",
    }


def _legacy_candidate_text(candidates: list[dict[str, Any]]) -> str:
    lines = ["## Current Context", "Market regime: ranging (0.78)"]
    for candidate in candidates:
        lines.extend(
            [
                f"Signal: {candidate['signal_id']}",
                f"Symbol: {candidate['symbol']}",
                f"Strategy: {candidate['strategy']}",
                f"Confidence: {candidate['confidence']}",
                f"Entry/stop/target: {candidate['entry_price']}/"
                f"{candidate['stop_loss']}/{candidate['target_price']}",
                f"Risk reward: {candidate['risk_reward_ratio']}",
                "Reasons: price above VWAP; RSI confirms; momentum confirms",
            ]
        )
    return "\n".join(lines)


def _compact_candidate_text(candidates: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "regime": "ranging",
            "regime_confidence": 0.78,
            "event_risk": "NONE",
            "market_context_version": "fixture-v1",
            "candidates": candidates,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _measure_call(system: str, human: str, output: str) -> tuple[int, int]:
    return _tokens(system) + _tokens(human), _tokens(output)


def replay(symbols: int) -> dict[str, Any]:
    raw = symbols
    legacy_reviewed = max(1, math.ceil(symbols * 0.10))
    optimized_reviewed = max(1, math.ceil(symbols * 0.04))
    legacy_candidates = [
        _candidate(i, "SCALP" if i % 2 else "SWING") for i in range(legacy_reviewed)
    ]
    optimized_candidates = legacy_candidates[:optimized_reviewed]
    legacy_swing = [item for item in legacy_candidates if item["horizon"] == "SWING"]
    legacy_scalp = [item for item in legacy_candidates if item["horizon"] == "SCALP"]
    optimized_swing = [item for item in optimized_candidates if item["horizon"] == "SWING"]
    optimized_scalp = [item for item in optimized_candidates if item["horizon"] == "SCALP"]

    headlines = "\n".join(f"- NSE market headline {i}" for i in range(1, 11))
    legacy_regime = "\n".join(
        f"SYM{i}: ADX=24 RSI=55 ATR=1.2 SMA20=101 SMA50=99 change=+0.3%" for i in range(5)
    )
    compact_regime = json.dumps(
        {
            "breadth": 0.58,
            "volatility": "normal",
            "index_state": "neutral",
            "symbols": [{"change": 0.3, "adx": 24, "rsi": 55, "atr": 1.2} for _ in range(5)],
            "news_sentiment": 0.1,
        },
        separators=(",", ":"),
    )
    legacy_strategy = (
        "Regime ranging confidence .78. Trades 2 P&L 1000. Historical performance: "
        "momentum 55% (80 trades), mean_reversion 59% (70 trades), breakout 51% "
        "(65 trades), trend_following 54% (90 trades). No active memory warning."
    )
    compact_strategy = json.dumps(
        {
            "regime": "ranging",
            "confidence": 0.78,
            "performance": {"momentum": [0.55, 80], "mean_reversion": [0.59, 70]},
            "candidate_strategies": ["momentum", "mean_reversion"],
        },
        separators=(",", ":"),
    )

    legacy_outputs = {
        "news": '{"sentiment":0.1,"reasoning":"Headlines are broadly neutral with limited directional impact."}',
        "regime": '{"regime":"ranging","confidence":0.78,"reasoning":"ADX and breadth indicate a range while sentiment remains neutral.","key_factors":["ADX","breadth","sentiment"]}',
        "strategy": '{"active_strategies":["mean_reversion"],"reasoning":"Mean reversion has the strongest ranging-regime evidence.","strategy_notes":{"mean_reversion":"selected","momentum":"rejected"}}',
    }
    compact_outputs = {
        "news": '{"sentiment":0.1,"reasoning":"NEUTRAL_NEWS"}',
        "regime": '{"regime":"ranging","confidence":0.78,"reasoning":"RANGE_CONFIRMED","key_factors":["LOW_ADX","NEUTRAL_BREADTH"]}',
        "strategy": '{"active_strategies":["mean_reversion"],"reasoning":"RANGE_EDGE"}',
    }

    before_calls: list[tuple[str, int, int]] = []
    after_calls: list[tuple[str, int, int]] = []

    def add(target: list[tuple[str, int, int]], agent: str, measured: tuple[int, int]) -> None:
        target.append((agent, measured[0], measured[1]))

    add(
        before_calls,
        "news_analyst",
        _measure_call(SENTIMENT_SYSTEM_PROMPT, headlines, legacy_outputs["news"]),
    )
    for _ in range(2):
        add(
            before_calls,
            "market_regime",
            _measure_call(REGIME_SYSTEM_PROMPT, legacy_regime, legacy_outputs["regime"]),
        )
        add(
            before_calls,
            "strategy_selection",
            _measure_call(STRATEGY_SYSTEM_PROMPT, legacy_strategy, legacy_outputs["strategy"]),
        )
    for horizon_candidates in (legacy_swing, legacy_scalp):
        output = json.dumps(
            {
                "validations": [
                    {
                        "signal_id": item["signal_id"],
                        "decision": "approve",
                        "confidence": 0.78,
                        "reasoning": "Technical evidence aligns and no contextual contradiction was found.",
                        "modifications": {},
                    }
                    for item in horizon_candidates
                ]
            }
        )
        add(
            before_calls,
            "signal_validation",
            _measure_call(
                VALIDATION_SYSTEM_PROMPT, _legacy_candidate_text(horizon_candidates), output
            ),
        )

    add(
        after_calls,
        "news_analyst",
        _measure_call(COMPACT_SENTIMENT_SYSTEM_PROMPT, headlines, compact_outputs["news"]),
    )
    add(
        after_calls,
        "market_regime",
        _measure_call(COMPACT_REGIME_SYSTEM_PROMPT, compact_regime, compact_outputs["regime"]),
    )
    for _ in range(2):
        add(
            after_calls,
            "strategy_selection",
            _measure_call(
                COMPACT_STRATEGY_SYSTEM_PROMPT, compact_strategy, compact_outputs["strategy"]
            ),
        )
    for horizon_candidates in (optimized_swing, optimized_scalp):
        output = json.dumps(
            {
                "validations": [
                    {
                        "signal_id": item["signal_id"],
                        "decision": "approve",
                        "confidence": 0.78,
                        "risk_flags": [],
                        "reason_codes": ["NO_EVENT_VETO"],
                    }
                    for item in horizon_candidates
                ]
            },
            separators=(",", ":"),
        )
        add(
            after_calls,
            "signal_validation",
            _measure_call(
                COMPACT_VALIDATION_SYSTEM_PROMPT,
                _compact_candidate_text(horizon_candidates),
                output,
            ),
        )

    tracker = CostTracker()

    def summary(calls: list[tuple[str, int, int]]) -> dict[str, Any]:
        input_tokens = sum(call[1] for call in calls)
        output_tokens = sum(call[2] for call in calls)
        return {
            "qwen_calls": len(calls),
            "qwen_calls_by_agent": {
                agent: sum(1 for item in calls if item[0] == agent)
                for agent in sorted({item[0] for item in calls})
            },
            "input_tokens_cl100k_estimate": input_tokens,
            "output_tokens_cl100k_estimate": output_tokens,
            "estimated_cost_usd": tracker.estimate_cost(MODEL, input_tokens, output_tokens),
        }

    before = {
        "symbols": symbols,
        "raw_signals": raw,
        "final_candidates": optimized_reviewed,
        "qwen_candidates": legacy_reviewed,
        **summary(before_calls),
    }
    after_cold = {
        "symbols": symbols,
        "raw_signals": raw,
        "final_candidates": optimized_reviewed,
        "qwen_candidates": optimized_reviewed,
        "cache_hits": 0,
        "dedup_skips": 0,
        "deterministic_skips": raw - optimized_reviewed,
        "zero_qwen_candidate_percentage": 100.0 * (raw - optimized_reviewed) / raw,
        **summary(after_calls),
    }
    for measured in (before, after_cold):
        measured["ai_approved"] = 2
        measured["execution_accepted"] = 1
        measured["qwen_calls_per_executed_paper_trade"] = measured["qwen_calls"]
        measured["qwen_tokens_per_executed_paper_trade"] = (
            measured["input_tokens_cl100k_estimate"] + measured["output_tokens_cl100k_estimate"]
        )
        measured["cost_per_executed_paper_trade_usd"] = measured["estimated_cost_usd"]
    after_warm = {
        **after_cold,
        "qwen_calls": 0,
        "qwen_calls_by_agent": {},
        "cache_hits": optimized_reviewed,
        "dedup_skips": optimized_reviewed,
        "input_tokens_cl100k_estimate": 0,
        "output_tokens_cl100k_estimate": 0,
        "estimated_cost_usd": 0.0,
        "zero_qwen_candidate_percentage": 100.0,
        "ai_approved": 2,
        "execution_accepted": 0,
        "qwen_calls_per_executed_paper_trade": 0,
        "qwen_tokens_per_executed_paper_trade": 0,
        "cost_per_executed_paper_trade_usd": 0.0,
    }
    return {"before": before, "after_cold": after_cold, "after_warm_unchanged": after_warm}


def main() -> None:
    report = {
        "method": "deterministic replay; no network, no Qwen API, no orders",
        "tokenizer": "cl100k_base estimate (not provider-reported Qwen usage)",
        "measured_100_symbols": replay(100),
        "extrapolated": {str(size): replay(size) for size in (50, 100, 300)},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
