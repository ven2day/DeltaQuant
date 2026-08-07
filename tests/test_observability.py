from unittest.mock import MagicMock, patch

import pytest

from src.observability.tracing import (
    TracingCallback,
    add_trace_metadata,
    create_tracing_config,
    setup_tracing,
    tag_trace,
    trace_agent,
    trading_trace,
)


@pytest.fixture
def mock_settings():
    with patch("src.observability.tracing.get_settings") as mock:
        mock.return_value.langfuse_public_key.get_secret_value.return_value = "pk-test"
        mock.return_value.langfuse_secret_key.get_secret_value.return_value = "sk-test"
        mock.return_value.langfuse_host = "https://langfuse.example"
        mock.return_value.langfuse_environment = "test"
        mock.return_value.langfuse_tracing_enabled = True
        yield mock


@pytest.fixture(autouse=True)
def reset_langfuse_client():
    with patch("src.observability.tracing._langfuse_client", None):
        yield


def test_setup_tracing(mock_settings):
    with (
        patch("src.observability.tracing.Langfuse") as mock_langfuse,
        patch("src.observability.tracing.get_client") as mock_get_client,
    ):
        mock_get_client.return_value.auth_check.return_value = True

        success = setup_tracing()

        assert success is True
        mock_langfuse.assert_called_once()


def test_setup_tracing_fail(mock_settings):
    with patch("src.observability.tracing.Langfuse", side_effect=Exception("Connect fail")):

        success = setup_tracing()
        assert success is False


def test_setup_tracing_disabled_does_not_create_client(mock_settings):
    mock_settings.return_value.langfuse_tracing_enabled = False
    with patch("src.observability.tracing.Langfuse") as mock_langfuse:
        success = setup_tracing()

    assert success is False
    mock_langfuse.assert_not_called()


def test_trace_agent_decorator():
    with patch("src.observability.tracing.observe") as mock_observe:
        mock_observe.return_value = lambda f: f

        @trace_agent("test_agent")
        def my_func(x):
            return x * 2

        result = my_func(2)
        assert result == 4


def test_trading_trace_context():
    with trading_trace("wf1", regime="bull") as meta:
        assert meta["workflow_id"] == "wf1"
        assert meta["regime"] == "bull"

    assert meta["status"] == "success"
    assert "duration_ms" in meta


def test_trading_trace_error():
    with pytest.raises(ValueError):
        with trading_trace("wf1") as meta:
            raise ValueError("Test error")

    assert meta["status"] == "error"
    assert "Test error" in meta["error"]


def test_add_trace_metadata():
    with patch("src.observability.tracing._langfuse_client", MagicMock()) as mock_client:

        add_trace_metadata("key", "value")

        mock_client.update_current_span.assert_called_once_with(metadata={"key": "value"})


def test_tag_trace():
    with patch("src.observability.tracing.add_trace_metadata") as mock_add:
        tag_trace(trade_id="t1", decision="buy", signal_id="s1")

        mock_add.assert_any_call("trade_id", "t1")
        mock_add.assert_any_call("decision", "buy")
        mock_add.assert_any_call("signal_id", "s1")


def test_tracing_callback():
    cb = TracingCallback("wf1")

    cb.on_agent_start("agent1", {})
    assert len(cb.events) == 1
    assert cb.events[0]["type"] == "agent_start"

    cb.on_agent_end("agent1", {}, 100)
    assert len(cb.events) == 2
    assert cb.events[1]["type"] == "agent_end"

    cb.on_decision("agent1", "buy", 0.9, "reason")
    assert len(cb.events) == 3
    assert cb.events[2]["type"] == "decision"

    cb.on_error("agent1", "err")
    assert len(cb.events) == 4
    assert cb.events[3]["type"] == "error"

    summary = cb.get_summary()
    assert summary["event_count"] == 4
    assert "agent1" in summary["agents_run"]


def test_create_tracing_config():
    config = create_tracing_config("wf1", metadata={"key": "val"})

    assert config["configurable"]["thread_id"] == "wf1"
    assert config["metadata"]["workflow_id"] == "wf1"
    assert config["metadata"]["key"] == "val"
    assert "started_at" in config["metadata"]
