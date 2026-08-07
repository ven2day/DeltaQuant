"""Serialization of TradingStats for the web UI.

Kept free of any FastAPI/uvicorn import so it stays importable (and unit-testable)
without the ``web`` extra installed.
"""

import dataclasses
from typing import Any

from src.dashboard.stats import TradingStats


def stats_to_dict(stats: TradingStats) -> dict[str, Any]:
    """Convert TradingStats to a JSON-serializable dict for the web UI.

    ``dataclasses.asdict`` covers every declared field as-is (none are nested
    dataclasses — open_positions/market_quotes/activity_log/current_signal are
    already plain dicts/lists), so only the datetime field and the computed
    ``@property`` values need to be added on top.
    """
    data = dataclasses.asdict(stats)
    data["session_start"] = stats.session_start.isoformat()
    data["win_rate"] = stats.win_rate
    data["total_pnl"] = stats.total_pnl
    data["pnl_percent"] = stats.pnl_percent
    return data
