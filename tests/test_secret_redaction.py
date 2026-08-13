"""Regression coverage for structure-preserving secret redaction."""

from src.config.secrets import redact_secrets
from src.markets.snapshots import MarketSnapshotStore


def test_pin_fields_are_redacted_without_masking_scalping_fields() -> None:
    candidates = [{"symbol": "RELIANCE", "last_price": 123.45}]
    payload = {
        "dhan_pin": "1234",
        "pin": "5678",
        "scalping_candidates": candidates,
        "scalping_screener_status": "ready",
        "scalping_screener_data_source": "dhan",
    }

    redacted = redact_secrets(payload)

    assert redacted["dhan_pin"] == "********"
    assert redacted["pin"] == "********"
    assert redacted["scalping_candidates"] == candidates
    assert redacted["scalping_screener_status"] == "ready"
    assert redacted["scalping_screener_data_source"] == "dhan"


def test_market_snapshot_preserves_scalping_dashboard_shapes(tmp_path) -> None:
    store = MarketSnapshotStore(root=tmp_path, min_write_seconds=0)
    candidates = [{"symbol": "RELIANCE", "last_price": 123.45}]

    assert store.publish(
        "NSE",
        status={"status": "HEALTHY"},
        signals=[],
        positions=[],
        dashboard_state={
            "scalping_candidates": candidates,
            "scalping_screener_status": "ready",
            "scalping_screener_data_source": "dhan",
        },
        force=True,
    )

    dashboard = store.read("NSE")["dashboard_state"]
    assert dashboard["scalping_candidates"] == candidates
    assert dashboard["scalping_screener_status"] == "ready"
    assert dashboard["scalping_screener_data_source"] == "dhan"
