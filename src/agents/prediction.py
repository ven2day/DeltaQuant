"""
Price Prediction Agent

Uses ensemble machine learning to predict next candle direction.
Based on RandomForest and Linear Regression with technical features.

Features:
- Direction prediction (up/down), or an explicit "no usable signal" abstain state
- Walk-forward (rolling-origin) out-of-sample validation, not a single tiny holdout
- Probability calibration (Platt scaling) instead of a raw R^2-weighted vote
- Strict train/validation/live-inference time separation (see H-7,
  DeltaQuant-Quant-Risk-Review.md): the row used for live inference is never a row
  that also has a known target.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None
    LogisticRegression = None
    RandomForestRegressor = None
    GradientBoostingRegressor = None
    StandardScaler = None
    TimeSeriesSplit = None

from src.config import get_settings

logger = logging.getLogger(__name__)

# Bumped whenever the feature-construction logic changes materially (columns added/
# removed/redefined). Persisted on every PredictionSignal so a prediction is always
# traceable to the code that produced it (H-7 acceptance criterion).
FEATURE_VERSION = "features_v2_time_separated"
# Bumped whenever the model/ensemble/validation/calibration methodology changes.
MODEL_VERSION = "ensemble_lr_rf_gbr_walkforward_calibrated_v2"

FEATURE_COLS = [
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

# Minimum labeled (feature+known-target) rows required before attempting walk-forward
# validation. Below this, there isn't enough OOS data for the R^2/weight estimate or the
# calibrator to mean anything -- the agent abstains rather than reporting a confident
# direction off a tiny sample (H-7).
MIN_LABELED_SAMPLES = 40
# Rolling-origin (expanding window) folds for out-of-sample validation.
N_WALK_FORWARD_SPLITS = 5
MIN_FOLD_TEST_SIZE = 5
MIN_FOLD_TRAIN_SAMPLES = 10
# If every model's average out-of-sample R^2 clamps to ~0, the ensemble weights collapse
# and there is no real signal to combine -- abstain instead of emitting a confidence-floor
# direction.
WEIGHT_COLLAPSE_EPS = 0.01


@dataclass
class PredictionSignal:
    """Price prediction signal."""

    symbol: str
    direction: str  # "up", "down", or "flat" (abstain -- no usable signal)
    confidence: float  # 0-1
    predicted_change_pct: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    # True when the agent had no statistically usable signal (insufficient OOS sample or
    # collapsed ensemble weights) and deliberately reported no direction rather than a
    # confidence-floor guess. Consumers (signal_ranking._ml_probability) must treat this
    # the same as "no prediction available", not as a real up/down vote.
    abstained: bool = False
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    # Number of out-of-sample (walk-forward) observations behind this prediction's
    # calibration; 0 when abstained/fallback.
    oos_samples: int = 0
    # Per-local-regime-bucket calibration error (mean squared error between calibrated
    # probability and actual realized direction) computed on the pooled OOS folds. Local
    # regime buckets are a deterministic proxy (trend/volatility of the feature window),
    # independent of the LLM market-regime classification, since this runs before that
    # agent in the pipeline.
    calibration_by_regime: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # confidence / predicted_change_pct are derived from numpy/pandas ops (model.predict,
        # pct_change), so they are numpy.float64 unless coerced. Cast to native float here so
        # this dict is msgpack-serializable when it lands in TradingState (see #18).
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": float(self.confidence),
            "predicted_change_pct": float(self.predicted_change_pct),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "abstained": bool(self.abstained),
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "oos_samples": int(self.oos_samples),
            "calibration_by_regime": {k: float(v) for k, v in self.calibration_by_regime.items()},
        }


@dataclass
class _OOSRow:
    """One pooled out-of-sample observation collected during walk-forward validation."""

    preds: dict[str, float]  # per-model raw prediction on this held-out row
    actual: float  # actual next-bar return (known, since this is a validation fold)
    regime: str  # local regime bucket (diagnostic only)


@dataclass
class _ValidationResult:
    weights: dict[str, float]  # per-model avg OOS R^2 (clamped >= 0)
    oos_rows: list[_OOSRow]
    total_weight: float
    folds_used: int


class PredictionAgent:
    """
    Predicts price direction using ensemble ML.

    Uses multiple models (LinearRegression, RandomForest, GradientBoosting), validates
    them out-of-sample via rolling-origin walk-forward folds (not a single 5-row
    holdout), refits on all available labeled data before producing the live inference,
    and calibrates the ensemble's raw regression output into a probability via Platt
    scaling fit on the pooled out-of-sample predictions.
    """

    def __init__(self, lookback_periods: int = 20):
        """
        Initialize prediction agent.

        Args:
            lookback_periods: Number of candles to use for prediction
        """
        self.lookback = lookback_periods
        self.settings = get_settings()

    def _compute_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Build the full feature+target frame from OHLCV data.

        Rows are dropped only where the FEATURE columns are NaN (rolling-window
        warm-up). The target column (``returns.shift(-1)``) keeps its NaN on the most
        recent row -- there is no "next candle" for it yet. That NaN-target row is
        exactly the row that is safe to use for live inference; every other
        (feature-complete) row has a known target and may be used for
        training/validation, but the two sets are never allowed to overlap (H-7).
        """
        if len(df) < self.lookback + 5:
            return None

        df = df.copy()
        df["returns"] = df["Close"].pct_change()
        df["returns_2"] = df["Close"].pct_change(2)
        df["returns_5"] = df["Close"].pct_change(5)

        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_10"] = df["Close"].rolling(10).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["sma_ratio_5_10"] = df["sma_5"] / df["sma_10"]
        df["sma_ratio_10_20"] = df["sma_10"] / df["sma_20"]

        df["vol_change"] = df["Volume"].pct_change()
        df["vol_sma_5"] = df["Volume"].rolling(5).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_sma_5"]

        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 0.0001)

        # RSI-like momentum
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 0.0001)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_normalized"] = df["rsi"] / 100

        # Target: next candle's return. shift(-1) leaves the LAST row's target as NaN --
        # that unknown-target row is the live-inference candidate, never a training row.
        df["target"] = df["returns"].shift(-1)

        frame = df.dropna(subset=FEATURE_COLS)
        if frame.empty:
            return None
        return frame[[*FEATURE_COLS, "target"]]

    def _create_features(self, df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Labeled training/validation matrix only (rows with a KNOWN target).

        This intentionally excludes the trailing unlabeled row -- see
        ``_compute_feature_frame`` and ``predict()`` for how the live-inference row is
        derived separately so it never doubles as a training/validation observation.
        """
        frame = self._compute_feature_frame(df)
        if frame is None:
            return None, None
        labeled = frame.dropna(subset=["target"])
        if len(labeled) < 10:
            return None, None
        return labeled[FEATURE_COLS].to_numpy(), labeled["target"].to_numpy()

    def _model_factories(self) -> dict[str, Any]:
        return {
            "linear": lambda: LinearRegression(),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boost": lambda: GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=42,
            ),
        }

    def _local_regime_bucket(self, row: np.ndarray, feature_cols: list[str]) -> str:
        """Deterministic, self-contained regime proxy for calibration bucketing only.

        This is NOT the LLM market-regime classification (that agent runs later in the
        pipeline and isn't available here) -- it is a cheap local heuristic used solely
        to report calibration error by regime for traceability/diagnostics.
        """
        idx = {name: i for i, name in enumerate(feature_cols)}
        sma_ratio = float(row[idx["sma_ratio_10_20"]])
        hl_range = float(row[idx["high_low_range"]])
        if hl_range > 0.03:
            return "volatile"
        if sma_ratio > 1.01:
            return "trending_up"
        if sma_ratio < 0.99:
            return "trending_down"
        return "ranging"

    def _walk_forward_validate(
        self,
        x_matrix: np.ndarray,
        y: np.ndarray,
        feature_cols: list[str],
        model_factories: dict[str, Any],
    ) -> _ValidationResult:
        """Rolling-origin (expanding window) out-of-sample validation.

        Replaces the old "withhold the last 5 rows" holdout: each fold trains only on
        data strictly before the fold's test window (sklearn TimeSeriesSplit), so the
        per-model weight is an average over several genuinely out-of-sample windows
        instead of one small, arbitrary slice.
        """
        n = len(x_matrix)
        n_splits = min(N_WALK_FORWARD_SPLITS, max(2, n // MIN_FOLD_TEST_SIZE - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        per_model_r2: dict[str, list[float]] = {name: [] for name in model_factories}
        oos_rows: list[_OOSRow] = []
        folds_used = 0

        for train_idx, test_idx in tscv.split(x_matrix):
            if len(train_idx) < MIN_FOLD_TRAIN_SAMPLES or len(test_idx) == 0:
                continue
            X_train, y_train = x_matrix[train_idx], y[train_idx]
            X_test, y_test = x_matrix[test_idx], y[test_idx]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            fold_preds: dict[str, np.ndarray] = {}
            for name, factory in model_factories.items():
                try:
                    model = factory()
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    r2 = model.score(X_test_scaled, y_test)
                    per_model_r2[name].append(max(0.0, r2))
                    fold_preds[name] = preds
                except Exception as e:
                    logger.warning(f"Walk-forward fold failed for model {name}: {e}")

            if not fold_preds:
                continue
            folds_used += 1
            for i, idx in enumerate(test_idx):
                row_preds = {name: float(arr[i]) for name, arr in fold_preds.items()}
                oos_rows.append(
                    _OOSRow(
                        preds=row_preds,
                        actual=float(y_test[i]),
                        regime=self._local_regime_bucket(x_matrix[idx], feature_cols),
                    )
                )

        weights = {name: (sum(v) / len(v) if v else 0.0) for name, v in per_model_r2.items()}
        total_weight = sum(weights.values())
        return _ValidationResult(
            weights=weights, oos_rows=oos_rows, total_weight=total_weight, folds_used=folds_used
        )

    def _fit_calibrator(self, raw_preds: list[float], actual_dir: list[int]) -> Any:
        """Platt-scale the pooled OOS ensemble output into P(up).

        Falls back to a fixed monotonic squash (not a statistically fit calibrator) when
        there isn't enough class diversity in the OOS sample to fit logistic regression --
        this keeps ``predict()`` from crashing on a degenerate (e.g. all-up) validation
        window while still being explicit that it isn't a real calibration in that case.
        """
        if len(actual_dir) < 5 or len(set(actual_dir)) < 2:
            return None
        try:
            x = np.array(raw_preds).reshape(-1, 1)
            y_arr = np.array(actual_dir)
            clf = LogisticRegression()
            clf.fit(x, y_arr)
            return clf
        except Exception as e:
            logger.warning(f"Calibrator fit failed, using manual squash: {e}")
            return None

    def _calibrated_probability(self, calibrator: Any, raw_pred: float) -> float:
        if calibrator is not None:
            try:
                return float(calibrator.predict_proba(np.array([[raw_pred]]))[0][1])
            except Exception:
                pass
        # Bounded, monotonic fallback so probability stays in (0, 1) without pretending
        # to be a fitted calibration when we lacked the OOS class diversity for one.
        return float(1.0 / (1.0 + np.exp(-50.0 * raw_pred)))

    def _calibration_error_by_regime(
        self,
        raw_preds: list[float],
        actual_dir: list[int],
        regimes: list[str],
        calibrator: Any,
    ) -> dict[str, float]:
        """Per-regime-bucket Brier score (MSE of calibrated prob vs. actual outcome)."""
        buckets: dict[str, list[tuple[float, int]]] = {}
        for raw, actual, regime in zip(raw_preds, actual_dir, regimes, strict=True):
            prob = self._calibrated_probability(calibrator, raw)
            buckets.setdefault(regime, []).append((prob, actual))
        return {
            regime: round(sum((p - a) ** 2 for p, a in items) / len(items), 4)
            for regime, items in buckets.items()
        }

    def _abstain(self, symbol: str, reason: str) -> PredictionSignal:
        """Explicit "no usable signal" state (H-7): never emit a confident direction
        off an inadequate sample or a collapsed ensemble."""
        logger.info(f"Prediction abstains for {symbol}: {reason}")
        return PredictionSignal(
            symbol=symbol,
            direction="flat",
            confidence=0.0,
            predicted_change_pct=0.0,
            reasoning=f"No usable signal: {reason}",
            abstained=True,
            feature_version=FEATURE_VERSION,
            model_version=MODEL_VERSION,
        )

    def predict(
        self,
        historical_data: pd.DataFrame | dict[str, Any],
        symbol: str = "UNKNOWN",
    ) -> PredictionSignal:
        """
        Predict next candle direction using a walk-forward-validated, calibrated ensemble.

        Args:
            historical_data: DataFrame with OHLCV columns or dict
            symbol: Stock symbol

        Returns:
            PredictionSignal with direction and confidence, or an abstain signal
            (``abstained=True``, ``direction="flat"``) when there isn't a statistically
            usable signal.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using fallback prediction")
            return self._fallback_predict(historical_data, symbol)

        try:
            # Convert dict to DataFrame if needed
            if isinstance(historical_data, dict):
                df = pd.DataFrame(historical_data)
            else:
                df = historical_data.copy()

            # Ensure required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for {symbol}, using fallback")
                return self._fallback_predict(historical_data, symbol)

            frame = self._compute_feature_frame(df)
            if frame is None:
                logger.warning(f"Insufficient data for {symbol}, using fallback")
                return self._fallback_predict(historical_data, symbol)

            labeled = frame.dropna(subset=["target"])
            live_rows = frame[frame["target"].isna()]

            if live_rows.empty:
                # No row with an unknown target exists (e.g. caller pre-truncated the
                # frame). Reusing a labeled row here would be exactly the H-7 leak --
                # abstain instead.
                return self._abstain(symbol, "no unseen row available for live inference")

            live_features = live_rows[FEATURE_COLS].iloc[-1:].to_numpy()

            if len(labeled) < MIN_LABELED_SAMPLES:
                return self._abstain(
                    symbol,
                    f"only {len(labeled)} labeled samples available "
                    f"(need >= {MIN_LABELED_SAMPLES} for walk-forward validation)",
                )

            X = labeled[FEATURE_COLS].to_numpy()
            y = labeled["target"].to_numpy()

            model_factories = self._model_factories()
            validation = self._walk_forward_validate(X, y, FEATURE_COLS, model_factories)

            if not validation.oos_rows or validation.total_weight < WEIGHT_COLLAPSE_EPS:
                return self._abstain(
                    symbol,
                    "ensemble weights collapsed to ~0 across out-of-sample folds "
                    "(no model showed OOS skill)",
                )

            oos_raw = [
                sum(row.preds.get(name, 0.0) * w for name, w in validation.weights.items())
                / validation.total_weight
                for row in validation.oos_rows
            ]
            oos_actual_dir = [1 if row.actual > 0 else 0 for row in validation.oos_rows]
            oos_regimes = [row.regime for row in validation.oos_rows]

            calibrator = self._fit_calibrator(oos_raw, oos_actual_dir)
            calibration_by_regime = self._calibration_error_by_regime(
                oos_raw, oos_actual_dir, oos_regimes, calibrator
            )

            # Refit on ALL available labeled data before live inference (H-7): the
            # walk-forward folds above are for validation/weighting/calibration only.
            final_scaler = StandardScaler().fit(X)
            X_scaled = final_scaler.transform(X)
            live_scaled = final_scaler.transform(live_features)

            final_preds: dict[str, float] = {}
            for name, factory in model_factories.items():
                try:
                    model = factory()
                    model.fit(X_scaled, y)
                    final_preds[name] = float(model.predict(live_scaled)[0])
                except Exception as e:
                    logger.warning(f"Final refit failed for model {name} on {symbol}: {e}")

            if not final_preds:
                return self._abstain(symbol, "final refit failed for all models")

            ensemble_raw = (
                sum(final_preds.get(name, 0.0) * w for name, w in validation.weights.items())
                / validation.total_weight
            )
            probability_up = self._calibrated_probability(calibrator, ensemble_raw)
            confidence = max(probability_up, 1.0 - probability_up)
            direction = "up" if probability_up >= 0.5 else "down"

            recent_returns = df["Close"].pct_change().tail(5).mean() * 100
            trend = "upward" if recent_returns > 0 else "downward"
            reasoning = (
                f"Walk-forward-validated ensemble of {len(model_factories)} models "
                f"({validation.folds_used} OOS folds, {len(labeled)} labeled samples, "
                f"{len(validation.oos_rows)} pooled OOS observations). "
                f"Calibrated P(up)={probability_up:.2f}. Recent trend: {trend} "
                f"({recent_returns:+.2f}% avg). Predicted move: "
                f"{abs(ensemble_raw) * 100:.2f}% {direction}."
            )

            logger.info(f"Prediction for {symbol}: {direction} ({confidence:.1%} confidence)")

            return PredictionSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                predicted_change_pct=ensemble_raw * 100,
                reasoning=reasoning,
                abstained=False,
                feature_version=FEATURE_VERSION,
                model_version=MODEL_VERSION,
                oos_samples=len(validation.oos_rows),
                calibration_by_regime=calibration_by_regime,
            )

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return self._fallback_predict(historical_data, symbol)

    def _fallback_predict(
        self,
        data: Any,
        symbol: str,
    ) -> PredictionSignal:
        """Fallback prediction using simple momentum."""
        try:
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                recent_change = data["Close"].pct_change().tail(3).mean()
            elif isinstance(data, dict) and "close" in data:
                closes = data["close"]
                if len(closes) >= 3:
                    recent_change = (closes[-1] - closes[-3]) / closes[-3]
                else:
                    recent_change = 0
            else:
                recent_change = 0
        except Exception:
            recent_change = 0

        direction = "up" if recent_change > 0 else "down"

        return PredictionSignal(
            symbol=symbol,
            direction=direction,
            confidence=0.4,  # Low confidence for fallback
            predicted_change_pct=recent_change * 100,
            reasoning=f"Fallback momentum prediction based on recent trend ({recent_change * 100:+.2f}%)",
            abstained=False,
            feature_version=FEATURE_VERSION,
            model_version="fallback_momentum_v1",
        )


def prediction_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node for price prediction.

    Generates prediction signals for the symbols in the *raw* signals. This node
    runs as a support agent (before regime/validation), so ``validated_signals``
    is still empty at this point — sourcing from it produced zero predictions.
    Raw ``signals`` are already populated by the time the graph runs, so the
    regime and validation agents now receive real ML consensus.

    The live loop's own ranking pass (run_live_trading.py) already runs this same
    ensemble on every symbol/timeframe pair with a fired signal, to feed
    signal_ranking.rank_signals. Rather than fetch history and re-predict a second
    time, this node first looks up ``state["precomputed_predictions"]`` (keyed
    "SYMBOL|TIMEFRAME") and only falls back to its own fetch+predict for
    symbol/timeframe pairs the caller didn't already cover (e.g. direct graph
    invocations in tests or scripts/run_trading.py).
    """
    agent = PredictionAgent()
    precomputed = state.get("precomputed_predictions") or {}

    signals = state.get("signals", [])
    predictions = []
    seen_symbols: set[str] = set()

    for signal in signals:
        symbol = signal.get("symbol", "")
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        if len(seen_symbols) > 3:  # Limit to top 3 distinct symbols
            break

        cached = precomputed.get(f"{symbol}|{signal.get('timeframe', '')}")
        if cached is not None:
            predictions.append(cached)
            continue

        try:
            # Fetch historical data
            from src.market.historical_feed import HistoricalDataFeed

            feed = HistoricalDataFeed(symbols=[symbol])
            hist = feed.get_historical(symbol, period="1mo")

            if hist is not None and not hist.empty:
                pred = agent.predict(hist, symbol)
                predictions.append(pred.to_dict())
        except Exception as e:
            logger.warning(f"Could not generate prediction for {symbol}: {e}")

    return {
        "prediction_signals": predictions,
    }


def test_prediction_agent():
    """Test the prediction agent."""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("[PREDICTION] ₹DeltaQuant - Price Prediction Test")
    print("=" * 60)

    agent = PredictionAgent()

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=150, freq="D")

    # Simulate uptrend
    base_price = 100
    prices = [base_price]
    for i in range(149):
        change = np.random.normal(0.002, 0.02)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame(
        {
            "Open": [p * 0.999 for p in prices],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, 150),
        },
        index=dates,
    )

    print("\n[DATA] Generated sample data:")
    print(f"  Start price: ₹{prices[0]:.2f}")
    print(f"  End price: ₹{prices[-1]:.2f}")
    print(f"  Total return: {(prices[-1] / prices[0] - 1) * 100:.2f}%")

    print("\n[PREDICT] Running prediction...")
    prediction = agent.predict(df, "SAMPLE")

    print(f"\n  Direction: {prediction.direction.upper()}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Predicted change: {prediction.predicted_change_pct:+.2f}%")
    print(f"  Abstained: {prediction.abstained}")
    print(f"  Reasoning: {prediction.reasoning}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Prediction agent working!")
    print("=" * 60)


if __name__ == "__main__":
    test_prediction_agent()
