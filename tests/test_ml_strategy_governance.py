"""
Regression tests for prediction train/inference leakage and preservation of the legacy
strict-grain validation records used as migration input.

H-7 (src/agents/prediction.py): the old ``predict()`` withheld only the final five rows of an
already fully-labeled feature matrix as a "test set", then reused the LAST row of that SAME
matrix (``X[-1]``, which is ``X_test[-1]`` by construction, i.e. a row with a KNOWN target) as
the "next candle" live-inference input.
``test_predict_end_to_end_uses_the_true_unseen_row_not_a_labeled_one`` below captures the exact
row fed into the final live-inference model call and checks it against an INDEPENDENTLY
re-derived reference computation of the true final bar's features (not prediction.py's own
helpers) -- manually verified via ``git stash`` to fail against the pre-fix source (the old
code's captured row does not match the true last bar; it's one step behind and has a known
target) and pass against the fix.

The legacy registry unit tests remain because those JSON records are migration input for
StrategyEligibilityRegistry; they no longer define runtime admission.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock settings for the duration of these imports only. Several modules under test
# call get_settings() at import time or in a plain (non-fixture) constructor
# (PredictionAgent.__init__), so importing this file standalone (rather than as part
# of the full `pytest` run) would otherwise hit a real, unconfigured Settings() build
# and fail on a missing GROQ_API_KEY -- same pattern used by test_agents_extended.py /
# test_agents_extra.py for the same reason.
with patch("src.config.get_settings") as _mock_get_settings:
    _mock_settings = MagicMock()
    _mock_settings.groq_api_key.get_secret_value.return_value = "token"
    _mock_settings.groq_model_primary = "llama"
    _mock_get_settings.return_value = _mock_settings

    from src.agents.prediction import FEATURE_COLS, MIN_LABELED_SAMPLES, PredictionAgent
    from src.backtesting.strategy_registry import StrategyRegistry, build_strategy_version

# ---------------------------------------------------------------------------
# Synthetic OHLCV data helper
# ---------------------------------------------------------------------------


def _noisy_ohlcv(n: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0005, 0.012)))
    return pd.DataFrame(
        {
            "Open": [p * 0.999 for p in prices],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": rng.integers(100_000, 1_000_000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


# ---------------------------------------------------------------------------
# H-7: prediction train/inference leakage
# ---------------------------------------------------------------------------


def test_compute_feature_frame_separates_live_row_from_labeled_matrix():
    """H-7 fix, white-box: the row available for live inference must have an
    unknown (NaN) target and must not appear in the labeled train/validation matrix."""
    df = _noisy_ohlcv(150)
    agent = PredictionAgent()

    frame = agent._compute_feature_frame(df)
    assert frame is not None

    labeled = frame.dropna(subset=["target"])
    live_rows = frame[frame["target"].isna()]

    # Exactly one genuinely unseen row: the true final bar of the input data.
    assert len(live_rows) == 1
    live_index = live_rows.index[0]
    assert live_index == df.index[-1]

    # It must not be part of the labeled matrix under any circumstance.
    assert live_index not in labeled.index
    assert bool(pd.isna(live_rows["target"].iloc[0]))

    # Cross-check against the OLD entry point (_create_features): the labeled matrix
    # it returns must be exactly `labeled` above, i.e. it must NOT include the live
    # row's features anywhere.
    X, _y = agent._create_features(df)
    assert X is not None
    live_feature_vector = live_rows[FEATURE_COLS].iloc[0].to_numpy()
    assert not any(np.allclose(row, live_feature_vector) for row in X)


def test_compute_feature_frame_sanitizes_infinite_volume_features():
    """A zero-volume bar (illiquid symbol, or a thin resampled candle) makes
    vol_change/vol_ratio divide by zero -> inf, not NaN. dropna() alone doesn't
    remove inf, so it used to reach StandardScaler().fit() and raise "Input X
    contains infinity" for the whole symbol -- caught upstream and silently
    degraded to a fixed 0.4-confidence fallback, which can never clear the
    >=0.55 ML-confidence entry gate (candidate_policy.py) regardless of how
    good the actual signal is. Only the contaminated row should be excluded,
    the same way an ordinary NaN row already is -- not the whole fit."""
    df = _noisy_ohlcv(150)
    df.loc[df.index[50], "Volume"] = 0.0

    agent = PredictionAgent()
    frame = agent._compute_feature_frame(df)

    assert frame is not None
    assert np.isfinite(frame[FEATURE_COLS].to_numpy()).all()

    X, y = agent._create_features(df)
    assert X is not None
    assert y is not None
    assert np.isfinite(X).all()


def _reference_last_row_features(df: pd.DataFrame) -> np.ndarray:
    """Independent re-derivation of the feature formulas, computed on the TRUE final
    row of ``df``. Deliberately duplicated here (not imported from prediction.py) so
    this check is a real external reference, not a tautology against the module's own
    internals -- run manually against the pre-fix source (git stash), this exact
    computation demonstrates the leak: the pre-fix live-inference row does NOT match
    the true last bar's features (it reuses an earlier, already-labeled row instead).
    Post-fix, they match exactly."""
    d = df.copy()
    d["returns"] = d["Close"].pct_change()
    d["returns_2"] = d["Close"].pct_change(2)
    d["returns_5"] = d["Close"].pct_change(5)
    d["sma_5"] = d["Close"].rolling(5).mean()
    d["sma_10"] = d["Close"].rolling(10).mean()
    d["sma_20"] = d["Close"].rolling(20).mean()
    d["sma_ratio_5_10"] = d["sma_5"] / d["sma_10"]
    d["sma_ratio_10_20"] = d["sma_10"] / d["sma_20"]
    d["vol_change"] = d["Volume"].pct_change()
    d["vol_sma_5"] = d["Volume"].rolling(5).mean()
    d["vol_ratio"] = d["Volume"] / d["vol_sma_5"]
    d["high_low_range"] = (d["High"] - d["Low"]) / d["Close"]
    d["close_position"] = (d["Close"] - d["Low"]) / (d["High"] - d["Low"] + 0.0001)
    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.0001)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["rsi_normalized"] = d["rsi"] / 100
    cols = [
        "returns",
        "returns_2",
        "returns_5",
        "sma_ratio_5_10",
        "sma_ratio_10_20",
        "vol_change",
        "vol_ratio",
        "high_low_range",
        "close_position",
        "rsi_normalized",
    ]
    return d[cols].iloc[-1].to_numpy()


def test_predict_end_to_end_uses_the_true_unseen_row_not_a_labeled_one():
    """Black-box, end-to-end, H-7 acceptance test.

    Capture the exact row fed into the final single-row live-inference call and prove
    it matches an INDEPENDENTLY re-derived computation of the true final bar's
    features -- not merely internally consistent with prediction.py's own helpers.

    Verified manually (git stash) that this specific check fails against the pre-fix
    ``predict()``: the old code withheld only the last 5 rows of an already-labeled
    matrix as a "test set" and then fed ``X[-1]`` (the same matrix's last row, which
    IS ``X_test[-1]`` by construction) back in as the "next candle" input -- a row one
    step behind the true final bar, and one with a known target. Post-fix, the single
    live-inference call's input matches this reference exactly.
    """
    df = _noisy_ohlcv(150)
    expected = _reference_last_row_features(df)

    from sklearn.preprocessing import StandardScaler

    captured_single_row_inputs: list[np.ndarray] = []
    original_transform = StandardScaler.transform

    def spy_transform(self, x_input, *args, **kwargs):  # type: ignore[no-untyped-def]
        if getattr(x_input, "shape", (0,))[0] == 1:
            captured_single_row_inputs.append(np.asarray(x_input).copy())
        return original_transform(self, x_input, *args, **kwargs)

    agent = PredictionAgent()
    with patch.object(StandardScaler, "transform", spy_transform):
        signal = agent.predict(df, "TESTSYM")

    assert not signal.abstained
    assert captured_single_row_inputs, "expected at least one single-row scaler.transform call"

    # The LAST single-row transform call is the live-inference one (walk-forward folds
    # only ever transform multi-row batches).
    final_call = captured_single_row_inputs[-1][0]
    assert np.allclose(final_call, expected, equal_nan=True)

    # And that row must not equal ANY row in the labeled training/validation matrix
    # _create_features() produces (the pre-fix code's only source of live features).
    X, _y = agent._create_features(df)
    assert X is not None
    assert not any(np.allclose(row, expected) for row in X)


def test_predict_refits_on_all_labeled_data_and_persists_versions():
    df = _noisy_ohlcv(150)
    agent = PredictionAgent()

    signal = agent.predict(df, "TESTSYM")

    assert not signal.abstained
    assert signal.oos_samples > 0
    assert signal.feature_version
    assert signal.model_version
    assert isinstance(signal.calibration_by_regime, dict)


def test_predict_abstains_when_labeled_sample_too_small():
    """H-7: insufficient walk-forward validation sample -> explicit no-signal abstain,
    never a confidence-floor direction."""
    df = _noisy_ohlcv(30)  # well under MIN_LABELED_SAMPLES after warm-up is dropped
    agent = PredictionAgent()

    signal = agent.predict(df, "TESTSYM")

    assert signal.abstained is True
    assert signal.direction == "flat"
    assert signal.confidence == 0.0


def test_min_labeled_samples_is_meaningfully_larger_than_old_five_row_holdout():
    # The old code validated on 5 rows. The fix must require a real sample.
    assert MIN_LABELED_SAMPLES >= 30


# ---------------------------------------------------------------------------
# Legacy edge_verdict registry behavior retained for migration compatibility
# ---------------------------------------------------------------------------


def _validated_version(tmp_path, strategy_name="momentum", expired=False, not_validated=False):
    now = datetime.now(UTC) - timedelta(days=40) if expired else datetime.now(UTC)
    version = build_strategy_version(
        strategy_name,
        owner="test",
        dataset_id="test-dataset",
        oos_trades=0 if not_validated else 100,
        oos_expectancy=-1.0 if not_validated else 2.0,
        oos_return_pct=-5.0 if not_validated else 10.0,
        fold_consistency=0.0 if not_validated else 0.8,
        validity_days=30,
        now=now,
    )
    registry = StrategyRegistry(tmp_path)
    registry.register(version)
    return registry, version


def test_strategy_registry_admits_current_validated_version(tmp_path):
    registry, version = _validated_version(tmp_path)
    assert version.status == "VALIDATED"
    assert registry.is_admitted("momentum") is True


def test_strategy_registry_rejects_not_validated_version(tmp_path):
    registry, version = _validated_version(tmp_path, not_validated=True)
    assert version.status == "NOT_VALIDATED"
    assert registry.is_admitted("momentum") is False


def test_strategy_registry_rejects_expired_version(tmp_path):
    registry, version = _validated_version(tmp_path, expired=True)
    assert version.status == "EXPIRED"
    assert registry.is_admitted("momentum") is False


def test_strategy_registry_rejects_unknown_strategy(tmp_path):
    registry = StrategyRegistry(tmp_path)
    assert registry.is_admitted("momentum") is False  # no entry at all -> fail closed


# ---------------------------------------------------------------------------
# Legacy-registry compatibility. These records remain readable for conservative
# migration into StrategyEligibilityRegistry; they are no longer the runtime gate.
# ---------------------------------------------------------------------------


def test_legacy_artifact_without_grain_keys_still_loads_and_admits_swing(tmp_path):
    """An on-disk JSON file written before timeframe/trade_horizon existed (i.e. no
    such keys at all) must still deserialize and keep admitting exactly what it always
    admitted -- SWING, unpinned to any timeframe. This is the backward-compatibility
    guarantee the whole registry-grain change depends on."""
    legacy_path = tmp_path / "momentum__legacy.json"
    legacy_payload = {
        "strategy_name": "momentum",
        "version": "legacy",
        "owner": "test",
        "parameters": {},
        "approved_universe": [],
        "approved_regimes": [],
        "dataset_id": "legacy-dataset",
        "validated_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
        "verdict": "VALIDATED",
        "reasons": [],
        "oos_trades": 100,
        "oos_expectancy": 2.0,
        "oos_return_pct": 10.0,
        "fold_consistency": 0.8,
        # deliberately NO "timeframe"/"trade_horizon" keys
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    registry = StrategyRegistry(tmp_path)
    version = registry.latest("momentum")
    assert version is not None
    assert version.timeframe == ""
    assert version.trade_horizon == "SWING"
    assert registry.is_admitted("momentum") is True  # default trade_horizon="SWING"
    assert registry.is_admitted("momentum", timeframe="15m") is True  # unpinned matches any


def test_legacy_swing_artifact_never_admits_a_scalp_request(tmp_path):
    """The central risk this stage exists to prevent: a strategy validated only on
    daily/swing bars must not silently cover a 5m scalp trade just because the
    strategy name matches."""
    _validated_version(tmp_path, strategy_name="momentum")  # trade_horizon defaults SWING

    registry = StrategyRegistry(tmp_path)
    assert registry.is_admitted("momentum", trade_horizon="SWING") is True
    assert registry.is_admitted("momentum", trade_horizon="SCALP") is False
    assert registry.is_admitted("momentum", timeframe="5m", trade_horizon="SCALP") is False


def test_fresh_scalp_artifact_admits_only_matching_horizon_and_timeframe(tmp_path):
    scalp_version = build_strategy_version(
        "momentum",
        owner="test",
        parameters={},
        approved_universe=[],
        approved_regimes=[],
        dataset_id="test-dataset-5m",
        oos_trades=100,
        oos_expectancy=2.0,
        oos_return_pct=10.0,
        fold_consistency=0.8,
        validity_days=30,
        timeframe="5m",
        trade_horizon="SCALP",
    )
    registry = StrategyRegistry(tmp_path)
    registry.register(scalp_version)

    # Matches: SCALP horizon, either unpinned-timeframe request or the exact 5m pin.
    assert registry.is_admitted("momentum", trade_horizon="SCALP") is True
    assert registry.is_admitted("momentum", timeframe="5m", trade_horizon="SCALP") is True
    # Does not match: wrong horizon, or a different pinned timeframe.
    assert registry.is_admitted("momentum", trade_horizon="SWING") is False
    assert registry.is_admitted("momentum", timeframe="15m", trade_horizon="SCALP") is False


def test_registry_keeps_distinct_scalp_timeframe_artifacts(tmp_path):
    registry = StrategyRegistry(tmp_path)
    versions = []
    for timeframe in ("5m", "15m"):
        version = build_strategy_version(
            "momentum",
            owner="test",
            parameters={},
            approved_universe=[],
            approved_regimes=[],
            dataset_id=f"test-dataset-{timeframe}",
            oos_trades=100,
            oos_expectancy=2.0,
            oos_return_pct=10.0,
            fold_consistency=0.8,
            validity_days=30,
            timeframe=timeframe,
            trade_horizon="SCALP",
        )
        registry.register(version)
        versions.append(version)

    assert len(list(tmp_path.glob("momentum__scalp__*__*.json"))) == 2
    assert registry.is_admitted("momentum", timeframe="5m", trade_horizon="SCALP")
    assert registry.is_admitted("momentum", timeframe="15m", trade_horizon="SCALP")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
