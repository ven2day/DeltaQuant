"""
Risk & Compliance Agent Module

Hard gatekeeper that enforces risk rules and compliance constraints.
No LLM required - uses deterministic rules for safety.

Features:
- Position sizing limits
- Sector exposure limits
- Portfolio correlation checks
- Drawdown and daily loss limits
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.backtesting.strategy_eligibility import (
    eligibility_registry_from_settings,
    resolve_eligibility_environment,
)
from src.config import get_settings
from src.markets.nse.sessions.market_time import is_trading_window, now_ist
from src.markets.nse.universe.sectors import get_stock_sector

from .state import TradingState

logger = logging.getLogger(__name__)


def _decision_chain(signal: dict[str, Any], *, risk_result: dict[str, Any], final: str) -> dict[str, Any]:
    """Compact reproducible lineage; natural-language reasoning is never authoritative."""

    return {
        "symbol": signal.get("symbol"),
        "timeframe": signal.get("timeframe"),
        "trade_horizon": signal.get("trade_horizon"),
        "settled_candle_timestamp": signal.get("signal_candle_timestamp"),
        "feature_snapshot": {
            "id": signal.get("feature_snapshot_id"),
            "version": signal.get("feature_version"),
        },
        "strategies": {
            "strongest": signal.get("strategy"),
            "version": signal.get("strategy_version"),
            "supporting": signal.get("supporting_strategies", []),
            "opposing": signal.get("opposing_strategies", []),
            "agreement": signal.get("agreement_level"),
            "conflict": signal.get("conflict_level"),
        },
        "strategy_eligibility": {
            "model_version": signal.get("model_version", signal.get("strategy_version")),
            "validation_status": signal.get(
                "eligibility_status", signal.get("validation_status")
            ),
            "current_regime": signal.get("current_regime"),
            "regime_policy": signal.get("regime_policy"),
            "original_confidence": signal.get("original_confidence"),
            "regime_adjusted_confidence": signal.get("regime_adjusted_confidence"),
            "registry_decision": signal.get("registry_decision"),
            "registry_reason": signal.get("registry_reason"),
        },
        "ml": {
            "status": signal.get("ml_status", "ABSTAIN"),
            "probability": signal.get("ml_probability"),
            "model_version": signal.get("ml_model_version"),
            "feature_version": signal.get("ml_feature_version"),
            "required": bool(signal.get("ml_required", False)),
        },
        "qwen_context": signal.get("validation", {}),
        "risk": risk_result,
        "final": final,
    }

@dataclass
class RiskLimits:
    """Risk management limits."""

    # Position limits
    max_position_size_pct: float = 10.0  # Max % of capital per position
    max_total_exposure_pct: float = 50.0  # Max total exposure
    max_positions: int = 5  # Max concurrent positions

    # Daily limits
    max_daily_trades: int = 50
    max_daily_loss: float = 10000.0  # INR

    # Per-trade limits
    min_risk_reward: float = 1.5
    max_stop_loss_pct: float = 5.0

    # Drawdown limits
    max_drawdown_pct: float = 5.0

    # Time-based limits
    no_trading_before: str = "09:15"  # Market open
    no_trading_after: str = "15:15"  # Before close
    force_trading_window: bool = False  # TESTING ONLY: see settings.force_trading_window
    force_window_requires_simulation_marker: bool = False

    # Sector exposure limits
    max_sector_exposure_pct: float = 30.0  # Max % of portfolio per sector
    max_correlated_positions: int = 3  # Max positions in same sector

    # Pairwise return-correlation limit (catches cross-sector co-movement that
    # the sector-based checks above miss)
    max_pairwise_correlation: float = 0.80

    @classmethod
    def from_settings(cls, *, trade_horizon: str = "SWING") -> "RiskLimits":
        """Create risk limits from application settings.

        ``trade_horizon="SCALP"`` selects the tighter ``settings.scalp_max_position_pct``
        in place of swing's ``max_position_pct`` for check #3 (position_size) -- every
        other limit (exposure, daily caps, drawdown, trading hours, sector/correlation
        caps, and the strategy-eligibility check in ``_run_risk_checks``) is unchanged
        regardless of horizon. A single ``risk_compliance_node`` invocation processes
        one horizon's signals at a time (the scalp and swing batches go through
        separate graph invocations, see ``src/markets/nse/runtime/live.py``), so building
        one ``RiskLimits`` per call here matches that existing one-call-per-invocation
        shape rather than needing per-signal limits.
        """
        settings = get_settings()
        configured_daily = getattr(settings, "max_daily_trades", 50)
        configured_daily = configured_daily if isinstance(configured_daily, int) else 50
        paper_cap = getattr(settings, "paper_daily_entry_cap", configured_daily)
        paper_cap = paper_cap if isinstance(paper_cap, int) else configured_daily
        position_pct = (
            settings.scalp_max_position_pct
            if trade_horizon == "SCALP"
            else settings.max_position_pct
        )
        return cls(
            # Convert fractions to percentages (0.10 -> 10.0%)
            max_position_size_pct=position_pct * 100,
            max_total_exposure_pct=settings.max_total_exposure_pct,
            max_positions=settings.max_concurrent_positions,
            max_daily_trades=min(configured_daily, paper_cap),
            max_daily_loss=settings.daily_loss_limit,
            no_trading_before=settings.no_trading_before,
            no_trading_after=settings.no_trading_after,
            force_trading_window=settings.force_trading_window,
            force_window_requires_simulation_marker=True,
            max_sector_exposure_pct=settings.max_sector_exposure * 100,
            max_pairwise_correlation=settings.max_pairwise_correlation,
        )


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    rule: str
    message: str
    severity: str = "warning"  # warning, block

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity,
        }


def risk_compliance_node(state: TradingState) -> dict[str, Any]:
    """
    LangGraph node for risk and compliance checks.

    Performs deterministic risk checks on validated signals.
    This is the final gatekeeper before trade execution.

    Args:
        state: Current trading state with validated signals

    Returns:
        State updates with approved and rejected trades
    """
    logger.info("Running Risk & Compliance Agent...")

    validated_signals = state.get("validated_signals", [])

    if not validated_signals:
        logger.info("No signals to check")
        return {
            "approved_trades": [],
            "risk_rejected": [],
            "risk_warnings": [],
            "trades_to_execute": [],
        }

    # Get current state
    portfolio = state.get("portfolio", {})
    daily_stats = state.get("daily_stats", {})
    regime = state.get("regime", "unknown")
    execution_mode = str(state.get("execution_mode", "market_paper"))

    # Get risk limits (horizon-aware position-size cap; see from_settings docstring)
    limits = RiskLimits.from_settings(trade_horizon=state.get("trade_horizon", "SWING"))

    approved = []
    rejected = []
    warnings = []

    for signal in validated_signals:
        checks = _run_risk_checks(signal, portfolio, daily_stats, limits, regime=regime)
        signal_execution_mode = str(signal.get("execution_mode", execution_mode))
        checks.append(
            RiskCheckResult(
                passed=signal_execution_mode == execution_mode,
                rule="execution_mode_isolation",
                message=(
                    f"Signal execution mode {signal_execution_mode!r} does not match "
                    f"cycle mode {execution_mode!r}"
                ),
                severity="block",
            )
        )

        # Collect results
        blocking_failures = [c for c in checks if not c.passed and c.severity == "block"]
        warning_failures = [c for c in checks if not c.passed and c.severity == "warning"]

        if blocking_failures:
            # Reject the signal
            signal["risk_result"] = {
                "approved": False,
                "failures": [c.to_dict() for c in blocking_failures],
            }
            signal["decision_chain"] = _decision_chain(
                signal, risk_result=signal["risk_result"], final="REJECTED_BY_RISK"
            )
            rejected.append(signal)

            for failure in blocking_failures:
                logger.warning(f"Trade rejected: {failure.message}")
        else:
            # Approve (possibly with warnings)
            signal["risk_result"] = {
                "approved": True,
                "warnings": [c.to_dict() for c in warning_failures],
            }
            signal["decision_chain"] = _decision_chain(
                signal, risk_result=signal["risk_result"], final="PAPER_APPROVED"
            )
            approved.append(signal)

            for warning in warning_failures:
                warnings.append(warning.message)
                logger.info(f"Trade approved with warning: {warning.message}")

    logger.info(f"Risk check: {len(approved)} approved, {len(rejected)} rejected")

    return {
        "approved_trades": approved,
        "risk_rejected": rejected,
        "risk_warnings": warnings,
        "trades_to_execute": approved,  # Approved trades go to execution
    }


def _run_risk_checks(
    signal: dict[str, Any],
    portfolio: dict[str, Any],
    daily_stats: dict[str, Any],
    limits: RiskLimits,
    regime: str = "unknown",
    enforce_strategy_eligibility: bool = True,
) -> list[RiskCheckResult]:
    """Run all risk checks on a signal.

    Hard/soft severity matrix (H-3, docs/audits/DeltaQuant-Quant-Risk-Review.md): every check that
    represents a stated portfolio/risk *limit* is a hard "block" -- daily_trade_limit,
    daily_loss_limit, position_size, risk_reward, stop_loss_pct, max_positions,
    total_exposure, trading_hours, drawdown, duplicate_position, sector_exposure,
    correlated_positions, and pairwise_correlation. A failed block cannot be
    approved-with-a-warning; the system should refuse rather than silently pyramid the
    same symbol, sector, or correlated bet, or exceed the stop-distance/reward-risk
    economics that sizing assumed. Only "confidence" (rule=confidence) stays a soft
    "warning": it is the LLM's own self-reported combined confidence, a quality signal to
    surface for review rather than a portfolio limit, and is deliberately not part of this
    matrix.
    """

    checks = []
    capital = float(portfolio.get("capital", 100000) or 0.0)

    # 1. Daily trade limit
    trades_today = daily_stats.get("trades_count", 0)
    checks.append(
        RiskCheckResult(
            passed=trades_today < limits.max_daily_trades,
            rule="daily_trade_limit",
            message=f"Daily trade limit ({limits.max_daily_trades}) reached: {trades_today} trades today",
            severity="block",
        )
    )

    # 2. Daily loss limit
    daily_pnl = daily_stats.get("profit_loss", 0)
    checks.append(
        RiskCheckResult(
            passed=daily_pnl > -limits.max_daily_loss,
            rule="daily_loss_limit",
            message=f"Daily loss limit (₹{limits.max_daily_loss}) breached: ₹{abs(daily_pnl)} loss",
            severity="block",
        )
    )

    # 3. Position size limit
    position_pct = signal.get("position_size_pct", 0)
    checks.append(
        RiskCheckResult(
            passed=position_pct <= limits.max_position_size_pct,
            rule="position_size",
            message=f"Position size {position_pct}% exceeds limit of {limits.max_position_size_pct}%",
            severity="block",
        )
    )

    # 4. Risk-reward ratio (H-3: hardened from warning to block -- a sub-minimum R:R signal
    # is a real edge/economics problem, not a soft advisory; see docs/audits/DeltaQuant-Quant-Risk-Review.md)
    rr_ratio = signal.get("risk_reward_ratio", 0)
    checks.append(
        RiskCheckResult(
            passed=rr_ratio >= limits.min_risk_reward,
            rule="risk_reward",
            message=f"Risk-reward {rr_ratio:.2f} below minimum {limits.min_risk_reward}",
            severity="block",
        )
    )

    # 5. Stop loss percentage (H-3: hardened from warning to block -- an over-wide stop
    # directly inflates loss-at-stop beyond what sizing/risk assumed it was approving)
    entry = signal.get("entry_price", 0)
    stop = signal.get("stop_loss", 0)
    if entry > 0:
        stop_pct = abs(entry - stop) / entry * 100
        checks.append(
            RiskCheckResult(
                passed=stop_pct <= limits.max_stop_loss_pct,
                rule="stop_loss_pct",
                message=f"Stop loss {stop_pct:.1f}% exceeds limit of {limits.max_stop_loss_pct}%",
                severity="block",
            )
        )

    # 6. Max positions
    current_positions = len(portfolio.get("positions", []))
    checks.append(
        RiskCheckResult(
            passed=current_positions < limits.max_positions,
            rule="max_positions",
            message=f"Max positions ({limits.max_positions}) reached: {current_positions} open",
            severity="block",
        )
    )

    # 6b. Aggregate projected notional exposure (hard portfolio limit).
    current_exposure = sum(
        float(position.get("quantity", 0))
        * float(position.get("current_price") or position.get("entry_price", 0))
        for position in portfolio.get("positions", [])
    )
    candidate_quantity = int(signal.get("quantity", 0) or 0)
    candidate_notional = entry * candidate_quantity
    if candidate_quantity <= 0:
        candidate_notional = capital * max(0.0, float(position_pct)) / 100.0
    projected_exposure_pct = (
        (current_exposure + candidate_notional) / capital * 100 if capital > 0 else 100.0
    )
    checks.append(
        RiskCheckResult(
            passed=projected_exposure_pct <= limits.max_total_exposure_pct,
            rule="total_exposure",
            message=(
                f"Projected exposure {projected_exposure_pct:.1f}% exceeds "
                f"limit of {limits.max_total_exposure_pct:.1f}%"
            ),
            severity="block",
        )
    )

    # 7. Trading hours (evaluated in IST, not host-local time)
    current_time = now_ist().strftime("%H:%M")
    force_simulation_window = limits.force_trading_window and (
        not limits.force_window_requires_simulation_marker
        or bool(daily_stats.get("simulation_mode", False))
    )
    in_trading_hours = force_simulation_window or is_trading_window(
        limits.no_trading_before, limits.no_trading_after, now=current_time
    )
    checks.append(
        RiskCheckResult(
            passed=in_trading_hours,
            rule="trading_hours",
            message=(
                "Trading hours check bypassed (FORCE_TRADING_WINDOW testing mode)"
                if force_simulation_window
                else f"Outside trading hours ({limits.no_trading_before}-{limits.no_trading_after})"
            ),
            severity="block",
        )
    )

    # 8. Drawdown check
    max_dd = daily_stats.get("max_drawdown", 0)
    dd_pct = (max_dd / capital * 100) if capital > 0 else 0
    checks.append(
        RiskCheckResult(
            passed=dd_pct < limits.max_drawdown_pct,
            rule="drawdown",
            message=f"Drawdown {dd_pct:.1f}% exceeds limit of {limits.max_drawdown_pct}%",
            severity="block",
        )
    )

    # 9. Duplicate position check
    symbol = signal.get("symbol", "")
    existing_positions = [p.get("symbol") for p in portfolio.get("positions", [])]
    checks.append(
        RiskCheckResult(
            passed=symbol not in existing_positions,
            rule="duplicate_position",
            message=f"Already have position in {symbol}",
            severity="block",
        )
    )

    # 10. Confidence check
    confidence = signal.get("confidence", 0)
    validation_confidence = signal.get("validation", {}).get("confidence", 0)
    avg_confidence = (
        (confidence + validation_confidence) / 2 if validation_confidence else confidence
    )
    checks.append(
        RiskCheckResult(
            passed=avg_confidence >= 0.5,
            rule="confidence",
            message=f"Low confidence score: {avg_confidence:.2f}",
            severity="warning",
        )
    )

    # 11. Sector exposure check (H-3: hardened from warning to block -- this is a stated
    # portfolio diversification limit, not advisory; a warning let the system pyramid the
    # same macro bet across a sector while still reporting "risk approved")
    new_symbol = signal.get("symbol", "")
    new_sector = get_stock_sector(new_symbol)
    sector_exposure = _calculate_sector_exposure(portfolio, new_sector, capital)
    checks.append(
        RiskCheckResult(
            passed=sector_exposure <= limits.max_sector_exposure_pct,
            rule="sector_exposure",
            message=f"Sector exposure ({new_sector}: {sector_exposure:.1f}%) exceeds limit of {limits.max_sector_exposure_pct}%",
            severity="block",
        )
    )

    # 12. Correlated positions check (H-3: hardened from warning to block, same rationale
    # as sector_exposure above -- a correlated-cluster cap that can't actually block is not
    # a limit)
    sector_positions = _count_sector_positions(portfolio, new_sector)
    checks.append(
        RiskCheckResult(
            passed=sector_positions < limits.max_correlated_positions,
            rule="correlated_positions",
            message=f"Too many positions in {new_sector} sector: {sector_positions}/{limits.max_correlated_positions}",
            severity="block",
        )
    )

    # 13. Pairwise return-correlation check (H-3: hardened from warning to block). Sector
    # exposure/count above are a coarse proxy for diversification — two names in different
    # sectors can still move together (or two "same sector" names can be weakly
    # correlated). This uses actual daily-return correlation, computed upstream by the
    # caller (see compute_return_correlations) and threaded through
    # portfolio["return_correlations"] since this function stays pure/synchronous and
    # never fetches data itself. Insufficient history yields max_corr=0.0 (an unfilled
    # `default=`, not a fabricated correlation), which safely passes rather than blocking
    # on absence of evidence.
    return_correlations = portfolio.get("return_correlations", {})
    existing_symbols = [
        p.get("symbol") for p in portfolio.get("positions", []) if p.get("symbol") != symbol
    ]
    symbol_correlations = return_correlations.get(symbol, {})
    max_corr = max(
        (
            abs(symbol_correlations[other])
            for other in existing_symbols
            if other in symbol_correlations
        ),
        default=0.0,
    )
    checks.append(
        RiskCheckResult(
            passed=max_corr <= limits.max_pairwise_correlation,
            rule="pairwise_correlation",
            message=(
                f"Return correlation {max_corr:.2f} with an existing position exceeds "
                f"limit of {limits.max_pairwise_correlation:.2f}"
            ),
            severity="block",
        )
    )

    # 14. Final strategy-eligibility backstop. The exact registry decision is recomputed
    # after Qwen so contextual output can never promote a blocked strategy or regime.
    strategy_name = str(signal.get("strategy", ""))
    if enforce_strategy_eligibility and strategy_name:
        settings = get_settings()
        signal_timeframe = str(signal.get("timeframe", ""))
        registry = eligibility_registry_from_settings(settings)
        environment = resolve_eligibility_environment(
            str(signal.get("execution_mode", "market_paper")),
            requested_execution_mode=str(getattr(settings, "execution_mode", "local_paper")),
            trading_mode=str(getattr(settings, "trading_mode", "paper")),
        )
        eligibility = registry.evaluate(
            strategy_name=strategy_name,
            timeframe=signal_timeframe,
            model_version=str(
                signal.get("model_version", signal.get("strategy_version", ""))
            ),
            environment=environment,
            current_regime=regime,
            confidence=float(
                signal.get("original_confidence", signal.get("confidence", 0.0)) or 0.0
            ),
            action=str(signal.get("signal_type", "BUY")),
            market=str(signal.get("market", getattr(settings, "market", "NSE"))),
        )
        signal.update(eligibility.to_dict())
        signal["registry_reason"] = eligibility.reason_code
        signal["eligibility_status"] = eligibility.validation_status.value
        checks.append(
            RiskCheckResult(
                passed=eligibility.execution_allowed,
                rule="strategy_eligibility",
                message=eligibility.message,
                severity="block",
            )
        )

    return checks


def compute_return_correlations(
    price_histories: dict[str, Any],
    min_periods: int = 20,
) -> dict[str, dict[str, float]]:
    """
    Pairwise Pearson correlation of daily returns across the given close-price series.

    ``price_histories`` maps symbol -> a pandas Series of daily closes (chronological).
    Returns a symmetric nested dict ``{symbol_a: {symbol_b: corr, ...}, ...}``; a pair
    is omitted entirely (not zero) when there isn't enough overlapping history to trust
    the estimate, so missing data degrades to "no correlation signal" rather than a
    false "uncorrelated" reading.
    """
    import pandas as pd

    returns = {
        symbol: series.pct_change().dropna()
        for symbol, series in price_histories.items()
        if series is not None and len(series) > min_periods
    }
    symbols = list(returns)
    correlations: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}

    for i, symbol_a in enumerate(symbols):
        for symbol_b in symbols[i + 1 :]:
            aligned = pd.concat([returns[symbol_a], returns[symbol_b]], axis=1, join="inner")
            if len(aligned) < min_periods:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if corr is None or pd.isna(corr):
                continue
            correlations[symbol_a][symbol_b] = float(corr)
            correlations[symbol_b][symbol_a] = float(corr)

    return correlations


def _calculate_sector_exposure(
    portfolio: dict[str, Any],
    sector: str,
    total_capital: float,
) -> float:
    """
    Calculate current exposure to a sector as percentage of capital.

    Args:
        portfolio: Current portfolio state
        sector: Sector to check
        total_capital: Total capital

    Returns:
        Sector exposure as percentage
    """
    if total_capital <= 0:
        return 0.0

    positions = portfolio.get("positions", [])
    sector_value = 0.0

    for position in positions:
        pos_symbol = position.get("symbol", "")
        pos_sector = get_stock_sector(pos_symbol)
        if pos_sector == sector:
            # Use market value or cost basis
            qty = position.get("quantity", 0)
            price = position.get("current_price", position.get("entry_price", 0))
            sector_value += qty * price

    return (sector_value / total_capital) * 100


def _count_sector_positions(portfolio: dict[str, Any], sector: str) -> int:
    """
    Count number of positions in a sector.

    Args:
        portfolio: Current portfolio state
        sector: Sector to count

    Returns:
        Number of positions in the sector
    """
    positions = portfolio.get("positions", [])
    count = 0

    for position in positions:
        pos_symbol = position.get("symbol", "")
        if get_stock_sector(pos_symbol) == sector:
            count += 1

    return count


def check_kill_switch(state: TradingState, limits: RiskLimits | None = None) -> bool:
    """
    Check if kill switch should be triggered.

    Returns True if trading should be halted immediately.
    """
    if limits is None:
        limits = RiskLimits.from_settings()

    settings = get_settings()
    market = str(state.get("market", getattr(settings, "market", "NSE"))).upper()
    if bool(getattr(settings, "global_kill_switch", False)):
        logger.critical("KILL SWITCH: GLOBAL_KILL_SWITCH is active")
        return True
    market_switch = {
        "NSE": bool(getattr(settings, "nse_kill_switch", False)),
        "FOREX": bool(getattr(settings, "forex_kill_switch", False)),
        "CRYPTO": bool(getattr(settings, "crypto_kill_switch", False)),
    }.get(market, True)
    if market_switch:
        logger.critical("KILL SWITCH: %s_KILL_SWITCH is active", market)
        return True

    daily_stats = state.get("daily_stats", {})

    # Check daily loss
    daily_pnl = daily_stats.get("profit_loss", 0)
    if daily_pnl <= -limits.max_daily_loss:
        logger.critical(f"KILL SWITCH: Daily loss limit breached (₹{abs(daily_pnl)})")
        return True

    # Check drawdown
    portfolio = state.get("portfolio", {})
    max_dd = daily_stats.get("max_drawdown", 0)
    capital = portfolio.get("capital", 100000)
    dd_pct = (max_dd / capital * 100) if capital > 0 else 0

    if dd_pct >= limits.max_drawdown_pct:
        logger.critical(f"KILL SWITCH: Drawdown limit breached ({dd_pct:.1f}%)")
        return True

    return False
