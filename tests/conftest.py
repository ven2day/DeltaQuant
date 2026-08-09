"""Session-wide safety net: no test may ever touch the real configured Postgres.

Several stores (LocalPaperEngine, SignalLogger, PerformanceTracker, IdempotencyStore,
DailyRiskStore) share one cached engine (src/db/base.py) that defaults to
settings.database_url when a test doesn't pass an explicit override. Most tests do pass
one, but it's easy to miss — this happened for real: test_strategy_selection_node
exercises strategy_selection_node's call to get_performance_tracker() (a module-level
singleton with no override) and wrote test data straight into the real production Neon
database. When a test misses its override, the fallback must be a harmless throwaway
database, never production.
"""

import pytest

import src.db.base as db_base
import src.execution.adapter as adapter_module
import src.market.data_feed as data_feed_module
import src.market.dhan_auth as dhan_auth_module
import src.market.dhan_historical_feed as dhan_historical_feed_module
import src.market.dhan_quotes_feed as dhan_quotes_feed_module
import src.market.live_data as live_data_module
import src.market.websocket_feed as websocket_feed_module
from src.utils.circuit_breaker import reset_all_circuit_breakers


@pytest.fixture(autouse=True, scope="session")
def _prevent_real_database_access(tmp_path_factory):
    fallback_path = tmp_path_factory.mktemp("db_safety_net") / "should_never_be_used.db"
    fallback_url = f"sqlite:///{fallback_path}"

    class _FallbackSettings:
        database_url = fallback_url

    # Reassigns the name src.db.base imported via `from src.config import get_settings`.
    # Deliberately not restored — this is process-lifetime for the whole test session,
    # and no test should ever want the real database_url back.
    db_base.get_settings = lambda: _FallbackSettings()


@pytest.fixture(autouse=True, scope="session")
def _prevent_real_dhan_auth():
    """No test may ever trigger a real DhanHQ login.

    get_valid_access_token() (src/market/dhan_auth.py) will hit DhanHQ's real
    generateAccessToken endpoint with real PIN+TOTP credentials from .env if
    called unmocked — this account now has live credentials configured, not
    just a static (often-expired) token, so an accidental real call here is a
    real login attempt, not a harmless no-op. Every module that imported the
    function via `from src.market.dhan_auth import get_valid_access_token`
    holds its own binding (patching the source module alone would miss them),
    so each is patched individually here — the same class of gap the database
    safety net above exists to close.
    """

    def _fake_token():
        return "test-fake-dhan-token"

    dhan_auth_module.get_valid_access_token = _fake_token
    websocket_feed_module.get_valid_access_token = _fake_token
    live_data_module.get_valid_access_token = _fake_token
    data_feed_module.get_valid_access_token = _fake_token
    adapter_module.get_valid_access_token = _fake_token
    dhan_historical_feed_module.get_valid_access_token = _fake_token
    dhan_quotes_feed_module.get_valid_access_token = _fake_token


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Every LLM-backed agent node shares one module-level circuit breaker per
    provider (get_llm_provider_circuit_breaker), keyed off the REAL configured
    LLM_PROVIDER from .env -- get_llm_circuit_breaker()'s own get_settings() call
    isn't touched by a test's `@patch("src.agents.X.get_settings")` (that only patches
    the name as imported into module X, not llm_factory's own reference). So a test
    that deliberately simulates a genuine (non-rate-limit) LLM failure trips the real
    shared breaker, and without a reset that failure count leaks into whichever
    unrelated test runs next in the same session -- confirmed to actually happen:
    genuine-failure tests here previously caused later, unrelated tests to fail with
    "circuit breaker open" for a breaker they never touched. Reset before AND after
    each test so failures from setup/teardown can't leak either direction.
    """
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()
