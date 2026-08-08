"""
Fail-closed configuration safety tests (DeltaQuant-Quant-Risk-Review.md C-2 / M-5 / H-10).

Covers:
- resolve_effective_execution_mode(): the pure invariant-resolution function shared by
  Settings.validate_configuration, ExecutionService._resolve_mode, execute_trades(), and
  scripts/check_config.py.
- Settings.validate_configuration(): the hazardous trading_mode x execution_mode x
  allow_live_orders x Dhan-credential matrix now raises (fails startup) instead of only
  warning, and MAX_QUOTE_STALENESS_SECONDS=0 fails startup on a live-quote profile.
- The C-2 hazard is unreachable through the legacy execute_trades() adapter-selection path
  in src/execution/adapter.py too, not just through ExecutionService.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.config.settings import Settings, resolve_effective_execution_mode

# ---------------------------------------------------------------------------
# resolve_effective_execution_mode: pure invariant matrix
# ---------------------------------------------------------------------------


def test_dhan_paper_never_resolves_live_even_with_full_opt_in():
    # The exact C-2 combination: paper-labeled trading_mode, dhan_paper execution,
    # the master gate armed, and valid credentials -- must still resolve to shadow.
    resolved = resolve_effective_execution_mode(
        "dhan_paper",
        allow_live_orders=True,
        trading_mode="paper",
        has_dhan_credentials=True,
    )
    assert resolved == "shadow"


def test_dhan_paper_never_resolves_live_even_with_trading_mode_live():
    # Even if trading_mode genuinely says live, dhan_paper still can't reach a real
    # route -- it is redefined to mean "simulate against Dhan-shaped data", full stop.
    resolved = resolve_effective_execution_mode(
        "dhan_paper",
        allow_live_orders=True,
        trading_mode="live",
        has_dhan_credentials=True,
    )
    assert resolved == "shadow"


def test_live_requires_full_conjunction():
    base = {"trading_mode": "live", "has_dhan_credentials": True}
    assert (
        resolve_effective_execution_mode("live", allow_live_orders=True, **base) == "live"
    )
    assert (
        resolve_effective_execution_mode("live", allow_live_orders=False, **base) == "shadow"
    )
    assert (
        resolve_effective_execution_mode(
            "live", allow_live_orders=True, trading_mode="paper", has_dhan_credentials=True
        )
        == "shadow"
    )
    assert (
        resolve_effective_execution_mode(
            "live", allow_live_orders=True, trading_mode="live", has_dhan_credentials=False
        )
        == "shadow"
    )


def test_local_paper_and_shadow_pass_through_unchanged():
    for mode in ("local_paper", "shadow"):
        assert (
            resolve_effective_execution_mode(
                mode, allow_live_orders=True, trading_mode="live", has_dhan_credentials=True
            )
            == mode
        )


# ---------------------------------------------------------------------------
# Settings.validate_configuration(): fail-closed hazardous matrix
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    kwargs = dict(groq_api_key="x")
    kwargs.update(overrides)
    return kwargs


def test_clean_defaults_construct_without_raising():
    s = Settings(**_base_kwargs())
    assert s.config_warnings == []


def test_c2_hazardous_env_combination_fails_startup():
    """The exact audited C-2 combination refuses to even construct a Settings object:
    TRADING_MODE=paper, EXECUTION_MODE=dhan_paper, ALLOW_LIVE_ORDERS=true, valid creds."""
    with pytest.raises(ValidationError, match="ALLOW_LIVE_ORDERS=true requires TRADING_MODE=live"):
        Settings(
            **_base_kwargs(
                trading_mode="paper",
                execution_mode="dhan_paper",
                allow_live_orders=True,
                dhan_client_id="real-client-id",
                dhan_access_token="real-access-token",
            )
        )


def test_execution_mode_live_with_paper_trading_mode_fails_startup():
    with pytest.raises(ValidationError, match="EXECUTION_MODE=live requires TRADING_MODE=live"):
        Settings(**_base_kwargs(trading_mode="paper", execution_mode="live"))


def test_trading_mode_live_without_credentials_fails_startup():
    with pytest.raises(ValidationError, match="TRADING_MODE=live requires Dhan credentials"):
        Settings(**_base_kwargs(trading_mode="live"))


def test_trading_mode_live_with_full_valid_config_does_not_raise():
    s = Settings(
        **_base_kwargs(
            trading_mode="live",
            execution_mode="live",
            allow_live_orders=True,
            dhan_client_id="real-client-id",
            dhan_access_token="real-access-token",
        )
    )
    assert s.trading_mode == "live"


def test_zero_staleness_with_live_quotes_fails_startup():
    """H-10: MAX_QUOTE_STALENESS_SECONDS=0 on a live-quote profile fails startup."""
    with pytest.raises(ValidationError, match="MAX_QUOTE_STALENESS_SECONDS=0"):
        Settings(
            **_base_kwargs(
                max_quote_staleness_seconds=0,
                enable_dhan_quotes=True,
            )
        )


def test_zero_staleness_allowed_in_declared_simulation_profile():
    # No live quote feed at all -- staleness is meaningless, so 0 is fine.
    s = Settings(
        **_base_kwargs(
            max_quote_staleness_seconds=0,
            enable_dhan_quotes=False,
        )
    )
    assert s.max_quote_staleness_seconds == 0


def test_default_staleness_is_positive_and_safe():
    # The bare default must itself be safe for a live-quote profile (ENABLE_DHAN_QUOTES
    # defaults true) -- regression guard against reintroducing the H-10 default of 0.
    s = Settings(**_base_kwargs())
    assert s.enable_dhan_quotes is True
    assert s.max_quote_staleness_seconds > 0


def test_benign_warnings_still_only_warn_not_raise():
    # Scope check: risk-param sanity issues are NOT part of the hazardous matrix and
    # must keep degrading gracefully (existing behavior), not fail closed.
    s = Settings(**_base_kwargs(risk_per_trade=0.5, max_position_pct=0.1))
    assert any("risk_per_trade" in w for w in s.config_warnings)


# ---------------------------------------------------------------------------
# The legacy execute_trades() adapter-selection path (src/execution/adapter.py) must
# also be unable to construct a real broker-capable adapter under the C-2 combination --
# it previously checked only execution_mode + credential presence, ignoring
# allow_live_orders and trading_mode entirely.
# ---------------------------------------------------------------------------


def test_execute_trades_never_constructs_real_adapter_under_c2_combination(tmp_path):
    """The exact C-2 combination must not construct a real (DhanHQ-backed)
    ExecutionAdapter -- proven by intercepting the class constructor directly, not just
    by inference from the outcome."""
    import asyncio

    from src.execution.adapter import execute_trades

    fake_settings = MagicMock(
        execution_mode="dhan_paper",
        trading_mode="paper",
        allow_live_orders=True,
        dhan_client_id="real-client-id",
        max_position_size=10000,
        paper_wallet_balance=1_000_000,
    )
    fake_settings.dhan_access_token.get_secret_value.return_value = "real-access-token"

    trades = [{"symbol": "RELIANCE", "entry_price": 2500.0, "signal_type": "BUY"}]

    with (
        patch("src.execution.adapter.get_settings", return_value=fake_settings),
        patch("src.execution.adapter.ExecutionAdapter") as mock_broker_adapter_cls,
    ):
        # adapter=None forces the settings-driven auto-selection branch under test.
        results = asyncio.run(
            execute_trades(trades, adapter=None, market_prices={"RELIANCE": 2500.0})
        )

    mock_broker_adapter_cls.assert_not_called()
    assert len(results) == 1
    assert results[0].request.symbol == "RELIANCE"


def test_execute_trades_constructs_real_adapter_only_for_full_live_conjunction():
    """Positive control for the test above: when the full conjunction genuinely holds
    (trading_mode=live AND allow_live_orders AND execution_mode=live AND credentials),
    execute_trades is allowed to construct the real broker adapter."""
    import asyncio

    from src.execution.adapter import execute_trades

    fake_settings = MagicMock(
        execution_mode="live",
        trading_mode="live",
        allow_live_orders=True,
        dhan_client_id="real-client-id",
        max_position_size=10000,
        paper_wallet_balance=1_000_000,
    )
    fake_settings.dhan_access_token.get_secret_value.return_value = "real-access-token"

    trades = [{"symbol": "RELIANCE", "entry_price": 2500.0, "signal_type": "BUY"}]

    with (
        patch("src.execution.adapter.get_settings", return_value=fake_settings),
        patch("src.execution.adapter.ExecutionAdapter") as mock_broker_adapter_cls,
    ):
        broker_instance = mock_broker_adapter_cls.return_value
        # isinstance(adapter, LocalExecutionAdapter) is checked downstream on whatever
        # gets selected -- a MagicMock instance is fine, it just isn't a
        # LocalExecutionAdapter, so market_prices is skipped and entry_price is used as-is.
        broker_instance.place_order = AsyncMock(
            return_value=MagicMock(status="rejected", filled_quantity=0, average_price=0.0)
        )
        asyncio.run(execute_trades(trades, adapter=None, market_prices={"RELIANCE": 2500.0}))

    mock_broker_adapter_cls.assert_called_once()
