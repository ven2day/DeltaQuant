from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import SecretStr

from src.agents.model_artifacts import (
    ArtifactRejectedError,
    PredictionArtifact,
    PredictionArtifactKey,
    PredictionArtifactRegistry,
)
from src.backtesting.strategy_eligibility import (
    EligibilityEnvironment,
    EligibilityStatus,
    StrategyEligibility,
    StrategyEligibilityRegistry,
)
from src.config.env_loader import resolve_env_files, selected_forex_environment, selected_market
from src.config.secrets import redact_secrets, safe_settings_dict
from src.config.settings import Settings
from src.core import strategies as production_strategies
from src.core.candidates import SignalEngine
from src.core.candles import CandleStore
from src.core.models import (
    CanonicalCandle,
    CanonicalQuote,
    Instrument,
    Market,
    MarketProvider,
    ProviderHealth,
    VolumeType,
)
from src.markets.forex.broker.oanda.provider import (
    OANDA_TIMEFRAME_MAP,
    OandaAuthenticationError,
    OandaEnvironment,
    OandaExecutionDisabledError,
    OandaExecutionProvider,
    OandaMarketDataProvider,
    OandaOrderIntent,
    OandaProviderError,
    OandaStreamHealthState,
    OandaV20Client,
    normalize_oanda_candle,
    normalize_oanda_price,
    oanda_granularity,
)
from src.markets.forex.risk.costs import ForexCostModel
from src.markets.forex.risk.model import (
    ForexRiskLimits,
    evaluate_forex_pretrade,
    quote_to_home_rate,
    size_forex_position,
)
from src.markets.forex.runtime.cycle import ForexSettledCandleCycle
from src.markets.forex.sessions.calendar import ForexSession, ForexTradingCalendar
from src.markets.forex.strategies import FOREX_INITIAL_DISABLED, build_strategy_config
from src.markets.provider_factory import parse_forex_symbols


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "groq_api_key": "test",
        "market": "FOREX",
        "forex_enabled": True,
        "forex_environment": "practice",
        "oanda_environment": "practice",
        "oanda_account_id": "account",
        "oanda_access_token": "token",
        "trading_mode": "paper",
        "execution_mode": "local_paper",
        "allow_live_orders": False,
        "enable_web_ui": False,
    }
    values.update(overrides)
    return Settings(**values)


def _instrument(symbol: str = "EUR_USD") -> Instrument:
    base, quote = symbol.split("_")
    return Instrument(
        market=Market.FOREX,
        symbol=symbol,
        broker_symbol=symbol,
        provider=MarketProvider.OANDA,
        base_currency=base,
        quote_currency=quote,
        display_precision=5,
        pip_location=-4 if quote != "JPY" else -2,
        trade_units_precision=0,
        minimum_trade_size=1,
        maximum_order_units=1_000_000,
    )


def _quote(
    *,
    symbol: str = "EUR_USD",
    bid: float = 1.1000,
    ask: float = 1.1002,
    age_seconds: float = 0,
) -> CanonicalQuote:
    return CanonicalQuote(
        symbol=symbol,
        market=Market.FOREX,
        provider=MarketProvider.OANDA,
        timestamp=datetime.now(UTC) - timedelta(seconds=age_seconds),
        bid=bid,
        ask=ask,
        tradeable=True,
        source="oanda_pricing",
    )


def _price_payload() -> dict[str, Any]:
    return {
        "type": "PRICE",
        "instrument": "EUR_USD",
        "time": "2026-08-11T10:00:00.000000000Z",
        "tradeable": True,
        "status": "tradeable",
        "bids": [{"price": "1.1000", "liquidity": "1000000"}],
        "asks": [{"price": "1.1002", "liquidity": "1000000"}],
    }


def _candle_payload(*, complete: bool = True) -> dict[str, Any]:
    return {
        "time": "2026-08-11T09:45:00.000000000Z",
        "volume": 142,
        "complete": complete,
        "mid": {"o": "1.1000", "h": "1.1020", "l": "1.0990", "c": "1.1010"},
    }


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def text(self) -> str:
        import json

        return json.dumps(self.payload)


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


class FakeStreamContent:
    def __init__(self, messages: list[dict[str, Any]]):
        import json

        self.chunks = [(json.dumps(message) + "\n").encode() for message in messages]

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk


class FakeStreamResponse:
    status = 200

    def __init__(self, messages: list[dict[str, Any]]):
        self.content = FakeStreamContent(messages)

    async def __aenter__(self) -> FakeStreamResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


class FakeStreamSession:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = outcomes

    def get(self, *_args: Any, **_kwargs: Any) -> FakeStreamResponse:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        return None


def _client(
    session: Any, *, environment: str = "practice", sleep: Any = asyncio.sleep
) -> OandaV20Client:
    return OandaV20Client(
        environment=environment,
        account_id=SecretStr("account"),
        access_token=SecretStr("top-secret"),
        session=session,
        sleep=sleep,
    )


def test_market_enum_is_first_class() -> None:
    assert {item.value for item in Market} == {"NSE", "FOREX", "CRYPTO"}


def test_common_plus_forex_practice_env_selection(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / ".env.common").write_text("APP_ENV=test\n", encoding="utf-8")
    (env_dir / ".env.forex.practice").write_text(
        "FOREX_ENABLED=true\n", encoding="utf-8"
    )
    files = resolve_env_files(tmp_path, {"MARKET": "FOREX", "FOREX_ENVIRONMENT": "practice"})
    assert files == (str(env_dir / ".env.common"), str(env_dir / ".env.forex.practice"))


def test_common_plus_nse_env_selection(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / ".env.common").touch()
    (env_dir / ".env.nse").touch()
    assert resolve_env_files(tmp_path, {"MARKET": "NSE"})[-1].endswith(".env.nse")


def test_forex_process_discards_dhan_credentials() -> None:
    settings = _settings(dhan_client_id="dhan", dhan_access_token="dhan-token")
    assert settings.dhan_client_id is None
    assert settings.dhan_access_token is None


def test_nse_process_discards_oanda_credentials() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test",
        market="NSE",
        oanda_account_id="account",
        oanda_access_token="token",
    )
    assert settings.oanda_account_id.get_secret_value() == ""
    assert settings.oanda_access_token.get_secret_value() == ""


def test_process_market_selection_validation() -> None:
    assert selected_market({"MARKET": "forex"}) == "FOREX"
    with pytest.raises(ValueError):
        selected_market({"MARKET": "invalid"})


def test_forex_environment_selection_validation() -> None:
    assert selected_forex_environment({"FOREX_ENVIRONMENT": "LIVE"}) == "live"
    with pytest.raises(ValueError):
        selected_forex_environment({"FOREX_ENVIRONMENT": "sandbox"})


def test_process_environment_overrides_selected_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / ".env.common"
    forex = tmp_path / ".env.forex.practice"
    common.write_text("GROQ_API_KEY=test\nAPP_ENV=file\n", encoding="utf-8")
    forex.write_text(
        "MARKET=FOREX\nFOREX_ENABLED=true\nOANDA_ACCOUNT_ID=file-account\n"
        "OANDA_ACCESS_TOKEN=file-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_ENV", "process")
    settings = Settings(_env_file=(common, forex))
    assert settings.app_env == "process"
    assert settings.market == "FOREX"


def test_unselected_broker_profile_is_not_loaded(tmp_path: Path) -> None:
    common = tmp_path / ".env.common"
    forex = tmp_path / ".env.forex.practice"
    nse = tmp_path / ".env.nse"
    common.write_text("GROQ_API_KEY=test\n", encoding="utf-8")
    forex.write_text(
        "MARKET=FOREX\nFOREX_ENABLED=true\nOANDA_ACCOUNT_ID=account\nOANDA_ACCESS_TOKEN=token\n",
        encoding="utf-8",
    )
    nse.write_text(
        "DHAN_CLIENT_ID=must-not-load\nDHAN_ACCESS_TOKEN=must-not-load\n", encoding="utf-8"
    )
    settings = Settings(_env_file=(common, forex))
    assert settings.dhan_client_id is None
    assert settings.dhan_access_token is None


def test_conflicting_oanda_environment_fails_closed() -> None:
    with pytest.raises(ValueError, match="must match"):
        _settings(forex_environment="practice", oanda_environment="live")


def test_live_oanda_execution_is_refused() -> None:
    with pytest.raises(ValueError, match="not enabled"):
        _settings(
            forex_environment="live",
            oanda_environment="live",
            forex_execution_enabled=True,
        )


def test_oanda_secret_is_redacted() -> None:
    safe = safe_settings_dict(_settings())
    assert safe["oanda_access_token"] == "********"
    assert "'oanda_access_token': 'token'" not in repr(safe)


def test_recursive_secret_redaction() -> None:
    assert redact_secrets({"Authorization": "Bearer abc", "nested": {"password": "x"}}) == {
        "Authorization": "********",
        "nested": {"password": "********"},
    }


def test_practice_endpoints_are_exact() -> None:
    assert OandaEnvironment.PRACTICE.rest_url == "https://api-fxpractice.oanda.com"
    assert OandaEnvironment.PRACTICE.stream_url == "https://stream-fxpractice.oanda.com"


def test_live_endpoints_are_exact() -> None:
    assert OandaEnvironment.LIVE.rest_url == "https://api-fxtrade.oanda.com"
    assert OandaEnvironment.LIVE.stream_url == "https://stream-fxtrade.oanda.com"


def test_client_repr_never_exposes_secret() -> None:
    client = _client(FakeSession([]))
    assert "top-secret" not in repr(client)
    assert "********" in repr(client)


@pytest.mark.asyncio
async def test_invalid_account_auth_fails_closed() -> None:
    client = _client(FakeSession([FakeResponse(401, {"errorMessage": "denied"})]))
    with pytest.raises(OandaAuthenticationError):
        await client.list_accounts()


@pytest.mark.asyncio
async def test_account_id_must_match_authorized_account() -> None:
    client = _client(FakeSession([FakeResponse(200, {"accounts": [{"id": "other"}]})]))
    with pytest.raises(OandaAuthenticationError, match="not authorized"):
        await client.validate_account()


def test_timeframe_mapping_is_canonical() -> None:
    assert {key: OANDA_TIMEFRAME_MAP[key] for key in ("15m", "30m", "1h", "4h")} == {
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
    }
    with pytest.raises(ValueError):
        oanda_granularity("2h")


def test_candle_normalization_has_tick_volume_semantics() -> None:
    candle = normalize_oanda_candle("EUR_USD", "15m", _candle_payload())
    assert candle.market is Market.FOREX
    assert candle.volume == 142
    assert candle.volume_type is VolumeType.OANDA_TICK_COUNT
    assert candle.complete


@pytest.mark.asyncio
async def test_incomplete_candles_are_excluded() -> None:
    session = FakeSession(
        [FakeResponse(200, {"candles": [_candle_payload(), _candle_payload(complete=False)]})]
    )
    candles = await _client(session).candles("EUR_USD", "15m", count=2)
    assert len(candles) == 1
    assert candles[0].complete


def test_pricing_normalization_preserves_bid_ask() -> None:
    quote = normalize_oanda_price(_price_payload())
    assert quote.bid == pytest.approx(1.1000)
    assert quote.ask == pytest.approx(1.1002)
    assert quote.mid == pytest.approx(1.1001)


@pytest.mark.asyncio
async def test_instrument_discovery_uses_account_metadata_and_universe() -> None:
    class InstrumentClient:
        async def instruments(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "EUR_USD",
                    "type": "CURRENCY",
                    "displayPrecision": 5,
                    "pipLocation": -4,
                    "tradeUnitsPrecision": 0,
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "1000000",
                    "marginRate": "0.0333",
                },
                {"name": "XAU_USD", "type": "METAL"},
            ]

    provider = OandaMarketDataProvider(
        InstrumentClient(),  # type: ignore[arg-type]
        enabled_symbols=("EUR_USD",),
    )
    instruments = await provider.list_instruments()
    assert [item.symbol for item in instruments] == ["EUR_USD"]
    assert instruments[0].pip_size == pytest.approx(0.0001)
    assert instruments[0].base_currency == "EUR"


def test_buy_uses_ask_and_sell_uses_bid() -> None:
    quote = normalize_oanda_price(_price_payload())
    assert quote.executable_price("BUY") == quote.ask
    assert quote.executable_price("SELL") == quote.bid


def test_stale_price_detection() -> None:
    provider = OandaMarketDataProvider(_client(FakeSession([])), stale_seconds=5)
    provider._latest_quotes["EUR_USD"] = _quote(age_seconds=6)
    assert provider.quote_is_stale("EUR_USD")


@pytest.mark.asyncio
async def test_rate_limit_retries_safe_read() -> None:
    delays: list[float] = []

    async def sleep(value: float) -> None:
        delays.append(value)

    session = FakeSession(
        [
            FakeResponse(429, {}, {"Retry-After": "0.1"}),
            FakeResponse(200, {"accounts": [{"id": "account"}]}),
        ]
    )
    accounts = await _client(session, sleep=sleep).list_accounts()
    assert accounts[0]["id"] == "account"
    assert delays == [0.1]


@pytest.mark.asyncio
async def test_stream_heartbeat_updates_health_state() -> None:
    heartbeat = {"type": "HEARTBEAT", "time": "2026-08-11T10:00:00Z"}
    client = _client(FakeStreamSession([FakeStreamResponse([heartbeat])]))
    stream = client.stream_messages(["EUR_USD"])
    message = await anext(stream)
    await stream.aclose()
    assert message["type"] == "HEARTBEAT"
    assert client.last_stream_heartbeat_at is not None
    assert client.metrics.heartbeats == 1
    assert client.stream_state is OandaStreamHealthState.HEALTHY


@pytest.mark.asyncio
async def test_stream_reconnects_with_backoff() -> None:
    delays: list[float] = []

    async def sleep(value: float) -> None:
        delays.append(value)

    session = FakeStreamSession(
        [OandaProviderError("temporary"), FakeStreamResponse([_price_payload()])]
    )
    client = _client(session, sleep=sleep)
    stream = client.stream_messages(["EUR_USD"])
    message = await anext(stream)
    await stream.aclose()
    assert message["type"] == "PRICE"
    assert client.metrics.stream_reconnects == 1
    assert len(delays) == 1
    assert client.stream_state is OandaStreamHealthState.HEALTHY


def test_stream_health_state_starts_not_started_and_becomes_stale() -> None:
    client = _client(FakeStreamSession([]))
    provider = OandaMarketDataProvider(client, stale_seconds=5)
    assert provider.stream_health_state() is OandaStreamHealthState.NOT_STARTED
    client.stream_state = OandaStreamHealthState.HEALTHY
    client.last_stream_message_at = datetime.now(UTC) - timedelta(seconds=6)
    assert provider.stream_health_state() is OandaStreamHealthState.STALE


@pytest.mark.asyncio
async def test_stream_auth_failure_is_distinct() -> None:
    response = FakeStreamResponse([])
    response.status = 401
    client = _client(FakeStreamSession([response]))
    stream = client.stream_messages(["EUR_USD"])
    with pytest.raises(OandaAuthenticationError):
        await anext(stream)
    assert client.stream_state is OandaStreamHealthState.AUTH_FAILED


@pytest.mark.asyncio
async def test_provider_background_stream_accepts_heartbeat_as_healthy() -> None:
    heartbeat = {"type": "HEARTBEAT", "time": "2026-08-11T10:00:00Z"}

    class PersistentContent(FakeStreamContent):
        async def iter_any(self):
            for chunk in self.chunks:
                yield chunk
            await asyncio.Event().wait()

    response = FakeStreamResponse([heartbeat])
    response.content = PersistentContent([heartbeat])
    client = _client(FakeStreamSession([response]))
    provider = OandaMarketDataProvider(client)
    await provider.start_price_stream(["EUR_USD"])
    assert await provider.wait_for_stream_message(1.0)
    assert provider.stream_health_state() is OandaStreamHealthState.HEALTHY
    await provider.close()


@pytest.mark.asyncio
async def test_provider_stream_start_is_singleton_and_shutdown_is_graceful() -> None:
    class BlockingContent:
        async def iter_any(self):
            await asyncio.Event().wait()
            yield b""  # pragma: no cover

    response = FakeStreamResponse([])
    response.content = BlockingContent()
    provider = OandaMarketDataProvider(_client(FakeStreamSession([response])))
    await provider.start_price_stream(["EUR_USD"])
    first = provider._stream_task
    await provider.start_price_stream(["EUR_USD"])
    assert provider._stream_task is first
    await provider.close()
    assert first is not None and first.cancelled()


@pytest.mark.asyncio
async def test_stream_startup_timeout_is_unhealthy_until_a_real_message() -> None:
    class BlockingContent:
        async def iter_any(self):
            await asyncio.Event().wait()
            yield b""  # pragma: no cover

    response = FakeStreamResponse([])
    response.content = BlockingContent()
    provider = OandaMarketDataProvider(_client(FakeStreamSession([response])))
    await provider.start_price_stream(["EUR_USD"])
    assert not await provider.wait_for_stream_message(0.01)
    assert provider.stream_health_state() is OandaStreamHealthState.UNHEALTHY
    assert provider.client.stream_error == "STREAM_MESSAGE_TIMEOUT"
    await provider.close()


@pytest.mark.asyncio
async def test_stream_http_failure_reports_code_and_remains_reconnecting() -> None:
    sleeping = asyncio.Event()
    release = asyncio.Event()

    async def sleep(_value: float) -> None:
        sleeping.set()
        await release.wait()

    response = FakeStreamResponse([])
    response.status = 520
    provider = OandaMarketDataProvider(
        _client(FakeStreamSession([response]), sleep=sleep)
    )
    await provider.start_price_stream(["EUR_USD"])
    await asyncio.wait_for(sleeping.wait(), timeout=1.0)
    assert provider.stream_health_state() is OandaStreamHealthState.RECONNECTING
    assert provider.client.stream_error == "HTTP_520"
    await provider.close()


@pytest.mark.asyncio
async def test_order_creation_is_not_retried() -> None:
    session = FakeSession([FakeResponse(500, {})])
    with pytest.raises(Exception):
        await _client(session).create_order({"type": "MARKET"})
    assert len(session.requests) == 1


def test_forex_universe_is_configurable_and_deduplicated() -> None:
    assert parse_forex_symbols("EUR_USD, GBP_USD,EUR_USD") == ("EUR_USD", "GBP_USD")


def test_forex_weekend_is_closed() -> None:
    calendar = ForexTradingCalendar()
    assert calendar.classify(datetime(2026, 8, 8, 12, tzinfo=UTC)) is ForexSession.CLOSED


def test_forex_sunday_reopens_at_new_york_1700() -> None:
    calendar = ForexTradingCalendar()
    before = datetime(2026, 8, 9, 20, 30, tzinfo=UTC)  # 16:30 EDT
    after = datetime(2026, 8, 9, 21, 30, tzinfo=UTC)  # 17:30 EDT
    assert not calendar.is_open(before)
    assert calendar.is_open(after)


def test_london_new_york_overlap_is_dst_aware() -> None:
    calendar = ForexTradingCalendar()
    assert calendar.classify(datetime(2026, 7, 1, 14, tzinfo=UTC)) is ForexSession.OVERLAP
    assert calendar.classify(datetime(2026, 1, 5, 14, tzinfo=UTC)) is ForexSession.OVERLAP


def test_rollover_is_explicit() -> None:
    calendar = ForexTradingCalendar()
    assert calendar.classify(datetime(2026, 8, 11, 21, tzinfo=UTC)) is ForexSession.ROLLOVER


def _eligibility(market: str) -> StrategyEligibility:
    return StrategyEligibility(
        market=market,
        strategy_name="ema_adx_trend",
        timeframe="30m",
        model_version="v1",
        validation_status=EligibilityStatus.SHADOW,
        validated_at="2026-08-11T00:00:00+00:00",
        validation_window={},
        oos_trade_count=0,
        oos_profit_factor=None,
        oos_max_drawdown=None,
        oos_win_rate=None,
        minimum_model_confidence=0.0,
    )


def test_strategy_registry_identity_includes_market(tmp_path: Path) -> None:
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(_eligibility("NSE"))
    registry.register(_eligibility("FOREX"))
    assert len(registry.load_all()) == 2
    assert registry.latest("ema_adx_trend", "30m", market="FOREX").market == "FOREX"


def test_missing_forex_registry_record_does_not_use_nse(tmp_path: Path) -> None:
    registry = StrategyEligibilityRegistry(tmp_path)
    registry.register(_eligibility("NSE"))
    decision = registry.evaluate(
        market="FOREX",
        strategy_name="ema_adx_trend",
        timeframe="30m",
        model_version="v1",
        environment=EligibilityEnvironment.LIVE,
        current_regime="trending",
        confidence=0.8,
    )
    assert not decision.execution_allowed
    assert decision.reason_code == "ELIGIBILITY_REGISTRY_MISSING"


def test_prediction_artifact_identity_includes_market(tmp_path: Path) -> None:
    registry = PredictionArtifactRegistry(tmp_path)
    nse_key = PredictionArtifactKey("v1", "30m", "SWING", "all", "m1", "f1", market="NSE")
    metadata = registry.metadata(nse_key, oos_samples=50, folds_used=3, calibration_by_regime={})
    registry.save(
        PredictionArtifact(
            metadata=metadata,
            scaler={"ok": True},
            models={"model": {"ok": True}},
            weights={"model": 1.0},
            calibrator=None,
        )
    )
    forex_key = replace(nse_key, market="FOREX")
    with pytest.raises(ArtifactRejectedError):
        registry.load(forex_key)


def test_regime_change_does_not_change_artifact_identity() -> None:
    first = PredictionArtifactKey("v1", "30m", "SWING", "trending", "m1", "f1", market="FOREX")
    second = replace(first, regime="ranging")
    assert first.normalized().eligibility_identity == second.normalized().eligibility_identity


def test_quote_to_home_conversion_direct_and_inverse() -> None:
    instrument = _instrument()
    assert quote_to_home_rate(instrument, "USD", {}) == 1.0
    gbp_jpy = _instrument("GBP_JPY")
    assert quote_to_home_rate(gbp_jpy, "USD", {("USD", "JPY"): 150}) == pytest.approx(1 / 150)


def test_forex_position_sizing_uses_stop_and_pip_value() -> None:
    result = size_forex_position(
        instrument=_instrument(),
        quote=_quote(),
        side="BUY",
        stop_price=1.0952,
        account_equity=10_000,
        risk_fraction=0.01,
        account_home_currency="USD",
        conversion_rates={},
    )
    assert result.units == pytest.approx(20_000)
    assert result.risk_amount_home == 100


def test_forex_pretrade_incorporates_spread() -> None:
    decision = evaluate_forex_pretrade(
        instrument=_instrument(),
        quote=_quote(bid=1.1000, ask=1.1010),
        side="BUY",
        atr=0.005,
        open_positions=0,
        portfolio_exposure_ratio=0,
        currency_exposure_ratios={},
        correlated_currency_positions=0,
        limits=ForexRiskLimits(max_spread_pips=5),
        global_kill_switch=False,
        forex_kill_switch=False,
    )
    assert "SPREAD_PIPS_LIMIT" in decision.reason_codes
    assert not decision.approved


def test_forex_currency_exposure_is_blocked() -> None:
    decision = evaluate_forex_pretrade(
        instrument=_instrument(),
        quote=_quote(),
        side="BUY",
        atr=0.01,
        open_positions=0,
        portfolio_exposure_ratio=0,
        currency_exposure_ratios={"USD": 1.2},
        correlated_currency_positions=0,
        limits=ForexRiskLimits(max_currency_exposure=1.0),
        global_kill_switch=False,
        forex_kill_switch=False,
    )
    assert "CURRENCY_EXPOSURE_LIMIT" in decision.reason_codes


def test_forex_kill_switch_and_global_override() -> None:
    common = dict(
        instrument=_instrument(),
        quote=_quote(),
        side="BUY",
        atr=0.01,
        open_positions=0,
        portfolio_exposure_ratio=0,
        currency_exposure_ratios={},
        correlated_currency_positions=0,
        limits=ForexRiskLimits(),
    )
    assert (
        "FOREX_KILL_SWITCH"
        in evaluate_forex_pretrade(
            **common, global_kill_switch=False, forex_kill_switch=True
        ).reason_codes
    )
    assert (
        "GLOBAL_KILL_SWITCH"
        in evaluate_forex_pretrade(
            **common, global_kill_switch=True, forex_kill_switch=False
        ).reason_codes
    )


def test_forex_cost_model_does_not_use_nse_fees() -> None:
    cost = ForexCostModel(slippage_pips_per_side=0.2).round_trip_cost_quote(
        instrument=_instrument(), quote=_quote(), units=10_000
    )
    assert cost == pytest.approx(2.4)


def test_shared_strategy_module_has_no_broker_dependency() -> None:
    source = inspect.getsource(production_strategies).lower()
    assert "oanda" not in source
    assert "dhan" not in source


def test_forex_sensitive_strategies_are_disabled_initially() -> None:
    config = build_strategy_config(_settings())
    assert config.market == "FOREX"
    assert all(config.timeframe_map[name] == () for name in FOREX_INITIAL_DISABLED)


def test_candle_store_partitions_market_and_provider() -> None:
    store = CandleStore("sqlite:///:memory:", enable_timescale=False)
    index = pd.DatetimeIndex([datetime(2026, 8, 11, 10, tzinfo=UTC)])
    frame = pd.DataFrame(
        {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [10]},
        index=index,
    )
    store.upsert_frame("TEST", "15m", frame, market="NSE", provider="DHAN")
    store.upsert_frame(
        "TEST",
        "15m",
        frame,
        market="FOREX",
        provider="OANDA",
        volume_type="OANDA_TICK_COUNT",
    )
    assert store.coverage("TEST", "15m", market="NSE", provider="DHAN").rows == 1
    assert store.coverage("TEST", "15m", market="FOREX", provider="OANDA").rows == 1


@pytest.mark.asyncio
async def test_oanda_execution_disabled_by_default() -> None:
    provider = OandaExecutionProvider(
        _client(FakeSession([])),
        configured_enabled=False,
        trading_mode="paper",
        global_kill_switch=False,
        forex_kill_switch=False,
    )
    assert not provider.execution_enabled
    with pytest.raises(OandaExecutionDisabledError):
        await provider.submit_order(OandaOrderIntent("id", "EUR_USD", "BUY", 100, 1.09, 1.12))


@pytest.mark.asyncio
async def test_live_endpoint_alone_never_enables_execution() -> None:
    provider = OandaExecutionProvider(
        _client(FakeSession([]), environment="live"),
        configured_enabled=True,
        trading_mode="paper",
        global_kill_switch=False,
        forex_kill_switch=False,
    )
    assert not provider.execution_enabled


class CycleProvider:
    def __init__(self) -> None:
        self.instrument = _instrument()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        self.candles = [
            CanonicalCandle(
                symbol="EUR_USD",
                market=Market.FOREX,
                provider=MarketProvider.OANDA,
                timeframe="15m",
                timestamp=start + timedelta(minutes=15 * index),
                open=1.0 + index * 0.0001,
                high=1.001 + index * 0.0001,
                low=0.999 + index * 0.0001,
                close=1.0005 + index * 0.0001,
                volume=100 + index,
                complete=True,
                source="fixture",
                volume_type=VolumeType.OANDA_TICK_COUNT,
            )
            for index in range(260)
        ]

    async def list_instruments(self) -> list[Instrument]:
        return [self.instrument]

    async def get_candles(
        self, _symbol: str, _timeframe: str, **kwargs: Any
    ) -> list[CanonicalCandle]:
        return self.candles[-int(kwargs.get("count", 260)) :]

    async def get_prices(self, _symbols: list[str]) -> dict[str, CanonicalQuote]:
        return {"EUR_USD": _quote()}


@pytest.mark.asyncio
async def test_forex_cycle_reuses_shared_features_and_no_change_is_cheap(tmp_path: Path) -> None:
    settings = _settings()
    config = replace(
        build_strategy_config(settings),
        timeframe_map={"ema_adx_trend": ("15m",)},
    )
    cycle = ForexSettledCandleCycle(
        provider=CycleProvider(),  # type: ignore[arg-type]
        signal_engine=SignalEngine(production_config=config),
        strategy_config=config,
        eligibility_registry=StrategyEligibilityRegistry(tmp_path),
        risk_limits=ForexRiskLimits(),
        settings=settings,
    )
    first = await cycle.run(timeframes=("15m",))
    second = await cycle.run(timeframes=("15m",))
    assert first.metrics.feature_snapshots == 1
    assert first.metrics.strategy_evaluations > 0
    assert second.metrics.feature_snapshots == 0
    assert second.metrics.strategy_evaluations == 0
    assert second.metrics.ml_evaluated == second.metrics.qwen_reviewed == 0


@pytest.mark.asyncio
async def test_provider_failure_is_market_isolated() -> None:
    class FailedClient:
        environment = OandaEnvironment.PRACTICE

        async def validate_account(self) -> None:
            raise OandaAuthenticationError("failed")

    provider = OandaMarketDataProvider(FailedClient())  # type: ignore[arg-type]
    health = await provider.validate_connectivity()
    assert not health.healthy
    assert health.market is Market.FOREX


def test_health_payload_has_no_account_or_token() -> None:
    health = ProviderHealth(
        market=Market.FOREX,
        provider=MarketProvider.OANDA,
        healthy=True,
        market_data="healthy",
        execution="disabled",
        environment="PRACTICE",
    ).to_dict()
    assert "token" not in repr(health).lower()
    assert "account" not in repr(health).lower()


def test_forex_production_risk_requires_valid_stop_and_monetary_budget() -> None:
    common = dict(
        instrument=_instrument(),
        quote=_quote(),
        side="BUY",
        atr=0.01,
        open_positions=0,
        portfolio_exposure_ratio=0.0,
        currency_exposure_ratios={},
        correlated_currency_positions=0,
        limits=ForexRiskLimits(risk_per_trade=0.005),
        global_kill_switch=False,
        forex_kill_switch=False,
        account_equity_home=100_000.0,
        require_stop_loss=True,
    )
    missing_stop = evaluate_forex_pretrade(**common, risk_amount_home=500.0)
    assert "STOP_LOSS_REQUIRED" in missing_stop.reason_codes
    invalid_stop = evaluate_forex_pretrade(
        **common, stop_price=_quote().ask + 0.001, risk_amount_home=500.0
    )
    assert "INVALID_STOP_LOSS" in invalid_stop.reason_codes
    excessive_risk = evaluate_forex_pretrade(
        **common, stop_price=_quote().bid - 0.01, risk_amount_home=501.0
    )
    assert "MONETARY_RISK_LIMIT" in excessive_risk.reason_codes


def test_forex_production_risk_enforces_daily_loss_margin_and_duplicate() -> None:
    decision = evaluate_forex_pretrade(
        instrument=_instrument(),
        quote=_quote(),
        side="SELL",
        atr=0.01,
        open_positions=0,
        portfolio_exposure_ratio=0.0,
        currency_exposure_ratios={},
        correlated_currency_positions=0,
        limits=ForexRiskLimits(
            risk_per_trade=0.005,
            daily_loss_limit_fraction=0.02,
            max_margin_utilization=0.50,
        ),
        global_kill_switch=False,
        forex_kill_switch=False,
        stop_price=_quote().ask + 0.01,
        risk_amount_home=500.0,
        account_equity_home=100_000.0,
        daily_realized_pnl_home=-1_800.0,
        margin_used_home=49_000.0,
        required_margin_home=2_000.0,
        duplicate_position=True,
        require_stop_loss=True,
    )
    assert "DAILY_LOSS_LIMIT" in decision.reason_codes
    assert "MARGIN_UTILIZATION_LIMIT" in decision.reason_codes
    assert "DUPLICATE_POSITION" in decision.reason_codes
