from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep

import pandas as pd

from src.agents.prediction import PredictionAgent, PredictionSignal


def _frame(last: str, rows: int = 60) -> pd.DataFrame:
    index = pd.date_range(end=last, periods=rows, freq="15min")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100},
        index=index,
    )


def test_prediction_cache_reuses_horizon_and_regime_independent_artifact(monkeypatch):
    agent = PredictionAgent()
    calls = 0

    def fake_predict(data, symbol="UNKNOWN", *, artifact_key):
        nonlocal calls
        calls += 1
        return PredictionSignal(symbol, "up", 0.7, 0.1, "test", datetime.now()), True

    monkeypatch.setattr(agent, "predict_live", fake_predict)
    frame = _frame("2026-08-10 10:00")
    agent.predict_cached(frame, "RELIANCE", timeframe="15m", trade_horizon="SWING")
    agent.predict_cached(frame, "RELIANCE", timeframe="15m", trade_horizon="SWING")
    agent.predict_cached(frame, "RELIANCE", timeframe="15m", trade_horizon="SCALP")
    agent.predict_cached(frame, "RELIANCE", timeframe="5m", trade_horizon="SWING")
    agent.predict_cached(
        _frame("2026-08-10 10:15"),
        "RELIANCE",
        timeframe="15m",
        trade_horizon="SWING",
    )

    assert calls == 3
    metrics = agent.cache_metrics()
    assert metrics.hits == 2
    assert metrics.misses == 3
    assert metrics.training_runs == 0
    assert metrics.walk_forward_runs == 0
    assert metrics.inference_count == 3


def test_concurrent_duplicate_prediction_is_coalesced(monkeypatch):
    agent = PredictionAgent()
    calls = 0

    def fake_predict(data, symbol="UNKNOWN", *, artifact_key):
        nonlocal calls
        calls += 1
        sleep(0.05)
        return PredictionSignal(symbol, "up", 0.7, 0.1, "test"), True

    monkeypatch.setattr(agent, "predict_live", fake_predict)
    frame = _frame("2026-08-10 10:00")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: agent.predict_cached(
                    frame,
                    "TCS",
                    timeframe="15m",
                    trade_horizon="SWING",
                ),
                range(4),
            )
        )

    assert calls == 1
    assert len(results) == 4
    assert agent.cache_metrics().hits == 3
