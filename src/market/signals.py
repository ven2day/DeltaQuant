"""
Signal Engine Module

Rule-based strategy signal generation for trading decisions.
Generates signals that are then validated by the agentic decision layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .indicators import IndicatorResult, Timeframe

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Signal strength classification."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class StrategyType(Enum):
    """Available trading strategies."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    # Adapted from a Traderversity.com strategy guide (EMA-Heiken-Ashi-RSI, EMA-Parabolic
    # SAR, EMA-CCI): standard, well-documented indicator combinations, not proprietary
    # black-box logic. Like every other strategy here, none of these three admit live
    # trades until they clear scripts/validate_strategy.py's walk-forward gate (H-8) --
    # the source guide ships no backtested evidence, just one narrated example chart.
    EMA_HEIKEN_ASHI_RSI = "ema_heiken_ashi_rsi"
    EMA_PSAR = "ema_psar"
    EMA_CCI = "ema_cci"


class TradeHorizon(Enum):
    """Which trading horizon a signal/strategy-version is governed under.

    Purely a classification tag at this layer -- it carries no risk/admission logic
    itself. SWING is the existing, long-established path (default for backward
    compatibility with every pre-existing signal/registry artifact); SCALP is the new
    5m/15m execution horizon. The H-8 registry (src/backtesting/strategy_registry.py)
    and the risk/validation agents are the actual enforcement points that key off this
    value -- see DeltaQuant-Quant-Risk-Review.md H-8.
    """

    SWING = "SWING"
    SCALP = "SCALP"


@dataclass
class TradingSignal:
    """Represents a trading signal from the signal engine."""

    signal_id: str
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    strategy: StrategyType
    timeframe: Timeframe

    # Entry/Exit levels
    entry_price: float
    stop_loss: float
    target_price: float

    # Risk metrics
    risk_reward_ratio: float
    position_size_pct: float  # Suggested position size as % of capital

    # Signal details
    confidence: float  # 0-1 confidence score
    reasons: list[str] = field(default_factory=list)
    indicators: dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    # Which trading horizon this signal belongs to. Defaults to SWING so every
    # existing call site (all of them, pre-dating this field) keeps producing
    # exactly the signals it always has.
    trade_horizon: TradeHorizon = TradeHorizon.SWING

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for agent consumption."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "strategy": self.strategy.value,
            "timeframe": self.timeframe.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size_pct": self.position_size_pct,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat(),
            "trade_horizon": self.trade_horizon.value,
        }


@dataclass
class SignalEngine:
    """
    Rule-based signal generation engine.

    Generates trading signals based on technical indicators.
    These signals are inputs to the agentic decision layer.
    """

    # Strategy parameters
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adx_trend_threshold: float = 25.0
    bb_squeeze_threshold: float = 0.1

    # Risk parameters
    default_stop_loss_pct: float = 2.0
    default_target_pct: float = 4.0
    max_position_size_pct: float = 10.0

    # Mean-reversion is a bounded-move strategy (bet on a snap back to the mid-band,
    # not a sustained trend), so it always uses a fixed % exit — never the ATR-based
    # sizing the other strategies use — sized to the kind of swing it's actually
    # catching, not a flat trend-following target.
    mean_reversion_stop_loss_pct: float = 1.2
    mean_reversion_target_pct: float = 2.5

    _signal_counter: int = field(default=0, repr=False)

    def generate_signals(
        self,
        indicators: IndicatorResult,
        active_strategies: list[StrategyType] | None = None,
    ) -> list[TradingSignal]:
        """
        Generate trading signals from indicator data.

        Args:
            indicators: Calculated indicators for a symbol
            active_strategies: List of strategies to run (all if None)

        Returns:
            List of generated trading signals
        """
        if active_strategies is None:
            active_strategies = list(StrategyType)

        signals = []

        for strategy in active_strategies:
            signal = self._run_strategy(indicators, strategy)
            if signal and signal.signal_type != SignalType.HOLD:
                signals.append(signal)

        return signals

    def _run_strategy(
        self,
        indicators: IndicatorResult,
        strategy: StrategyType,
    ) -> TradingSignal | None:
        """Run a specific strategy and return signal if generated."""

        if strategy == StrategyType.MOMENTUM:
            return self._momentum_strategy(indicators)
        elif strategy == StrategyType.MEAN_REVERSION:
            return self._mean_reversion_strategy(indicators)
        elif strategy == StrategyType.BREAKOUT:
            return self._breakout_strategy(indicators)
        elif strategy == StrategyType.TREND_FOLLOWING:
            return self._trend_following_strategy(indicators)
        elif strategy == StrategyType.EMA_HEIKEN_ASHI_RSI:
            return self._ema_heiken_ashi_rsi_strategy(indicators)
        elif strategy == StrategyType.EMA_PSAR:
            return self._ema_psar_strategy(indicators)
        elif strategy == StrategyType.EMA_CCI:
            return self._ema_cci_strategy(indicators)

        return None

    def _momentum_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        Momentum strategy based on RSI and MACD.

        BUY: RSI < 50 and rising, MACD histogram positive
        SELL: RSI > 50 and falling, MACD histogram negative
        """
        if ind.rsi is None or ind.macd_histogram is None:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        # BUY conditions
        if ind.rsi < 50 and ind.macd_histogram > 0:
            signal_type = SignalType.BUY
            reasons.append(f"RSI at {ind.rsi:.1f} with room to run")
            reasons.append("MACD histogram positive (bullish momentum)")

            # Strength based on RSI level
            if ind.rsi < self.rsi_oversold:
                strength = SignalStrength.STRONG
                confidence = 0.8
                reasons.append("RSI in oversold territory")
            elif ind.rsi < 40:
                strength = SignalStrength.MODERATE
                confidence = 0.6
            else:
                confidence = 0.4

        # SELL conditions
        elif ind.rsi > 50 and ind.macd_histogram < 0:
            signal_type = SignalType.SELL
            reasons.append(f"RSI at {ind.rsi:.1f} showing weakness")
            reasons.append("MACD histogram negative (bearish momentum)")

            if ind.rsi > self.rsi_overbought:
                strength = SignalStrength.STRONG
                confidence = 0.8
                reasons.append("RSI in overbought territory")
            elif ind.rsi > 60:
                strength = SignalStrength.MODERATE
                confidence = 0.6
            else:
                confidence = 0.4

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.MOMENTUM, confidence, reasons
        )

    def _mean_reversion_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        Mean reversion strategy based on Bollinger Bands and RSI.

        BUY: Price below lower BB and RSI oversold
        SELL: Price above upper BB and RSI overbought
        """
        if ind.bb_lower is None or ind.bb_upper is None or ind.rsi is None:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        # BUY conditions - price touched lower BB
        if ind.close <= ind.bb_lower and ind.rsi < self.rsi_oversold:
            signal_type = SignalType.BUY
            strength = SignalStrength.STRONG
            confidence = 0.75
            reasons.append("Price at lower Bollinger Band")
            reasons.append(f"RSI oversold at {ind.rsi:.1f}")
            reasons.append("Mean reversion opportunity")

        elif ind.close <= ind.bb_lower:
            signal_type = SignalType.BUY
            strength = SignalStrength.MODERATE
            confidence = 0.5
            reasons.append("Price at lower Bollinger Band")

        # SELL conditions - price touched upper BB
        elif ind.close >= ind.bb_upper and ind.rsi > self.rsi_overbought:
            signal_type = SignalType.SELL
            strength = SignalStrength.STRONG
            confidence = 0.75
            reasons.append("Price at upper Bollinger Band")
            reasons.append(f"RSI overbought at {ind.rsi:.1f}")
            reasons.append("Mean reversion expected")

        elif ind.close >= ind.bb_upper:
            signal_type = SignalType.SELL
            strength = SignalStrength.MODERATE
            confidence = 0.5
            reasons.append("Price at upper Bollinger Band")

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.MEAN_REVERSION, confidence, reasons
        )

    def _breakout_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        Breakout strategy based on Bollinger Band squeeze and price action.

        BUY: BB squeeze followed by upward breakout
        SELL: BB squeeze followed by downward breakout
        """
        if ind.bb_percent is None or ind.bb_upper is None or ind.bb_lower is None:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        # Check for squeeze (bands are tight)
        bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_middle if ind.bb_middle else 0
        is_squeeze = bb_width < self.bb_squeeze_threshold

        if is_squeeze:
            # Breakout conditions with volume confirmation
            if ind.close > ind.bb_upper:
                signal_type = SignalType.BUY
                strength = SignalStrength.STRONG
                confidence = 0.7
                reasons.append("Bollinger Band squeeze breakout (upward)")
                reasons.append(f"Price broke above upper band at {ind.bb_upper:.2f}")

            elif ind.close < ind.bb_lower:
                signal_type = SignalType.SELL
                strength = SignalStrength.STRONG
                confidence = 0.7
                reasons.append("Bollinger Band squeeze breakout (downward)")
                reasons.append(f"Price broke below lower band at {ind.bb_lower:.2f}")

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.BREAKOUT, confidence, reasons
        )

    def _trend_following_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        Trend following strategy based on ADX and moving average crossovers.

        BUY: Strong uptrend (ADX > threshold, +DI > -DI, price above EMAs)
        SELL: Strong downtrend (ADX > threshold, -DI > +DI, price below EMAs)
        """
        if ind.adx is None or ind.plus_di is None or ind.minus_di is None:
            return None

        if not ind.ema or 21 not in ind.ema:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        ema_21 = ind.ema.get(21, 0)

        # Check if in strong trend
        if ind.adx > self.adx_trend_threshold:
            # Uptrend
            if ind.plus_di > ind.minus_di and ind.close > ema_21:
                signal_type = SignalType.BUY
                reasons.append(f"Strong uptrend with ADX at {ind.adx:.1f}")
                reasons.append(f"+DI ({ind.plus_di:.1f}) > -DI ({ind.minus_di:.1f})")
                reasons.append(f"Price above EMA21 ({ema_21:.2f})")

                if ind.adx > 40:
                    strength = SignalStrength.STRONG
                    confidence = 0.8
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.6

            # Downtrend
            elif ind.minus_di > ind.plus_di and ind.close < ema_21:
                signal_type = SignalType.SELL
                reasons.append(f"Strong downtrend with ADX at {ind.adx:.1f}")
                reasons.append(f"-DI ({ind.minus_di:.1f}) > +DI ({ind.plus_di:.1f})")
                reasons.append(f"Price below EMA21 ({ema_21:.2f})")

                if ind.adx > 40:
                    strength = SignalStrength.STRONG
                    confidence = 0.8
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.6

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.TREND_FOLLOWING, confidence, reasons
        )

    def _ema_heiken_ashi_rsi_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        EMA + Heiken Ashi + RSI strategy (Traderversity.com guide, adapted).

        The source material's entry rule is: no trade in a sideways market; confirm trend
        direction; confirm momentum with RSI vs the 50 level; enter on a Heiken Ashi
        candle color flip in the confirmed direction. Two of those four steps reference
        things the guide only shows as hand-drawn chart annotations, not a defined
        formula ("flat support/resistance zones" for the sideways filter, "zones tilt" for
        trend direction) -- there's nothing precise to port for either, so this uses the
        indicators already computed here for the same job: ADX for trend-vs-sideways
        (matches the existing trend_following strategy's use of the same threshold), and
        price-vs-EMA21 for trend direction.

        BUY: trending (ADX above threshold) + price above EMA21 (uptrend) + RSI > 50
             (bullish momentum) + Heiken Ashi just flipped red -> green (entry trigger)
        SELL: trending + price below EMA21 (downtrend) + RSI < 50 (bearish momentum) +
              Heiken Ashi just flipped green -> red
        """
        if (
            ind.adx is None
            or ind.rsi is None
            or ind.ha_bullish is None
            or ind.ha_prev_bullish is None
        ):
            return None
        if not ind.ema or 21 not in ind.ema:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        ema_21 = ind.ema[21]
        is_trending = ind.adx > self.adx_trend_threshold
        ha_flipped_bullish = ind.ha_bullish and not ind.ha_prev_bullish
        ha_flipped_bearish = (not ind.ha_bullish) and ind.ha_prev_bullish

        if not is_trending:
            reasons.append(f"ADX at {ind.adx:.1f} -- sideways market, no trade")
        elif ind.close > ema_21 and ind.rsi > 50 and ha_flipped_bullish:
            signal_type = SignalType.BUY
            reasons.append(f"Uptrend: ADX {ind.adx:.1f}, price above EMA21 ({ema_21:.2f})")
            reasons.append(f"RSI at {ind.rsi:.1f} confirms bullish momentum")
            reasons.append("Heiken Ashi flipped red to green (entry trigger)")
            strong = ind.rsi > 60 and ind.adx > 40
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.75 if strong else 0.6

        elif ind.close < ema_21 and ind.rsi < 50 and ha_flipped_bearish:
            signal_type = SignalType.SELL
            reasons.append(f"Downtrend: ADX {ind.adx:.1f}, price below EMA21 ({ema_21:.2f})")
            reasons.append(f"RSI at {ind.rsi:.1f} confirms bearish momentum")
            reasons.append("Heiken Ashi flipped green to red (entry trigger)")
            strong = ind.rsi < 40 and ind.adx > 40
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.75 if strong else 0.6

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.EMA_HEIKEN_ASHI_RSI, confidence, reasons
        )

    def _ema_psar_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        EMA(20/40) + Parabolic SAR strategy (Traderversity.com guide, adapted).

        BUY: 20 EMA crosses above 40 EMA (uptrend) AND the Parabolic SAR dot is below
             price (bullish reversal/continuation confirmation)
        SELL: 20 EMA below 40 EMA (downtrend) AND the PSAR dot is above price

        The source guide enters on the *next* candle after both conditions align; this
        engine (like every other strategy here) evaluates and signals on the current bar
        instead of queuing a deferred entry -- consistent with how momentum/breakout/
        trend_following already work in this codebase.
        """
        if ind.psar_bullish is None:
            return None
        if not ind.ema or 20 not in ind.ema or 40 not in ind.ema:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        ema_20, ema_40 = ind.ema[20], ind.ema[40]
        ema_gap_pct = abs(ema_20 - ema_40) / ema_40 * 100 if ema_40 else 0.0

        if ema_20 > ema_40 and ind.psar_bullish:
            signal_type = SignalType.BUY
            reasons.append(f"EMA20 ({ema_20:.2f}) above EMA40 ({ema_40:.2f})")
            reasons.append(f"Parabolic SAR dot below price at {ind.psar:.2f}")
            strong = ema_gap_pct > 1.0
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.7 if strong else 0.55

        elif ema_20 < ema_40 and not ind.psar_bullish:
            signal_type = SignalType.SELL
            reasons.append(f"EMA20 ({ema_20:.2f}) below EMA40 ({ema_40:.2f})")
            reasons.append(f"Parabolic SAR dot above price at {ind.psar:.2f}")
            strong = ema_gap_pct > 1.0
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.7 if strong else 0.55

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.EMA_PSAR, confidence, reasons
        )

    def _ema_cci_strategy(self, ind: IndicatorResult) -> TradingSignal | None:
        """
        EMA(200) + CCI(14) strategy, from TraderVersity-EMACCI.tpl (a MetaTrader chart
        template naming exactly these two standard, well-documented indicators and their
        parameters -- 200-period EMA trend filter, 14-period CCI with the standard +-100
        levels).

        BUY: price above EMA200 (long-term uptrend) AND CCI above +100 (strong bullish
             momentum breakout)
        SELL: price below EMA200 (long-term downtrend) AND CCI below -100 (strong
              bearish momentum breakout)
        """
        if ind.cci is None or not ind.ema or 200 not in ind.ema:
            return None

        reasons = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0

        ema_200 = ind.ema[200]

        if ind.close > ema_200 and ind.cci > 100:
            signal_type = SignalType.BUY
            reasons.append(f"Price above EMA200 ({ema_200:.2f}) -- long-term uptrend")
            reasons.append(f"CCI at {ind.cci:.1f} confirms bullish momentum breakout")
            strong = ind.cci > 150
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.7 if strong else 0.55

        elif ind.close < ema_200 and ind.cci < -100:
            signal_type = SignalType.SELL
            reasons.append(f"Price below EMA200 ({ema_200:.2f}) -- long-term downtrend")
            reasons.append(f"CCI at {ind.cci:.1f} confirms bearish momentum breakout")
            strong = ind.cci < -150
            strength = SignalStrength.STRONG if strong else SignalStrength.MODERATE
            confidence = 0.7 if strong else 0.55

        if signal_type == SignalType.HOLD:
            return None

        return self._create_signal(
            ind, signal_type, strength, StrategyType.EMA_CCI, confidence, reasons
        )

    def _directional_confidence(self, ind: IndicatorResult, signal_type: SignalType) -> float:
        """
        Confidence in [0.35, 0.90] from how many independent indicators agree with the
        signal direction (RSI, MACD histogram, +DI/-DI, price vs moving average). More
        agreement → higher confidence; this replaces the old hardcoded confidence constants.
        """
        is_buy = signal_type == SignalType.BUY
        votes: list[bool] = []
        if ind.rsi is not None:
            votes.append(ind.rsi < 50 if is_buy else ind.rsi > 50)
        if ind.macd_histogram is not None:
            votes.append(ind.macd_histogram > 0 if is_buy else ind.macd_histogram < 0)
        if ind.plus_di is not None and ind.minus_di is not None:
            votes.append(ind.plus_di > ind.minus_di if is_buy else ind.minus_di > ind.plus_di)
        ref_ma = (ind.ema or {}).get(21) or (ind.sma or {}).get(20)
        if ref_ma is not None:
            votes.append(ind.close > ref_ma if is_buy else ind.close < ref_ma)
        if not votes:
            return 0.5
        agreement = sum(1 for v in votes if v) / len(votes)
        return round(0.35 + 0.55 * agreement, 2)

    def _create_signal(
        self,
        ind: IndicatorResult,
        signal_type: SignalType,
        strength: SignalStrength,
        strategy: StrategyType,
        confidence: float,
        reasons: list[str],
    ) -> TradingSignal:
        """Create a trading signal with proper risk management levels."""

        self._signal_counter += 1
        signal_id = f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._signal_counter:04d}"

        # Calculate entry, stop loss, and target
        entry_price = ind.close

        if strategy == StrategyType.MEAN_REVERSION:
            stop_pct = self.mean_reversion_stop_loss_pct
            target_pct = self.mean_reversion_target_pct
            use_atr = False
        else:
            stop_pct = self.default_stop_loss_pct
            target_pct = self.default_target_pct
            use_atr = bool(ind.atr)

        if signal_type == SignalType.BUY:
            # Use ATR-based stop if available (trend/momentum strategies), otherwise
            # a fixed percentage (always the case for mean-reversion, see above).
            if use_atr:
                stop_loss = entry_price - (2 * ind.atr)
                target_price = entry_price + (3 * ind.atr)
            else:
                stop_loss = entry_price * (1 - stop_pct / 100)
                target_price = entry_price * (1 + target_pct / 100)
        else:  # SELL
            if use_atr:
                stop_loss = entry_price + (2 * ind.atr)
                target_price = entry_price - (3 * ind.atr)
            else:
                stop_loss = entry_price * (1 + stop_pct / 100)
                target_price = entry_price * (1 - target_pct / 100)

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Position size based on strength
        position_pct = {
            SignalStrength.WEAK: 3.0,
            SignalStrength.MODERATE: 5.0,
            SignalStrength.STRONG: min(8.0, self.max_position_size_pct),
        }[strength]

        # Principled confidence: blend the strategy's base confidence with how strongly the
        # independent indicators actually agree with the signal direction — evidence-based,
        # not a fixed constant pulled from a hardcoded ladder.
        agreement = self._directional_confidence(ind, signal_type)
        final_confidence = round(min(0.95, 0.4 * confidence + 0.6 * agreement), 2)

        return TradingSignal(
            signal_id=signal_id,
            symbol=ind.symbol,
            signal_type=signal_type,
            strength=strength,
            strategy=strategy,
            timeframe=ind.timeframe,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward_ratio=risk_reward,
            position_size_pct=position_pct,
            confidence=final_confidence,
            reasons=reasons,
            indicators=ind.to_dict(),
        )
