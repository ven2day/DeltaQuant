"""Run the local Groq + Dhan quantitative signal-discovery workflow."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.core.indicators import Timeframe
from src.markets.nse.market_data.history_manager import HistoryManager
from src.markets.nse.universe.discovery import StockDiscovery
from src.signal_discovery.operators import build_ohlcv_panel
from src.signal_discovery.workflow import SignalDiscoveryWorkflow


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Signal family, e.g. volume-adjusted momentum")
    parser.add_argument("--timeframe", choices=[item.value for item in Timeframe], default=None)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 uses the full universe")
    parser.add_argument("--bars", type=int, default=500)
    return parser.parse_args()


async def _run() -> int:
    args = _arguments()
    settings = get_settings()
    default_timeframe = settings.signal_discovery_timeframes.split(",")[0].strip()
    timeframe = Timeframe(args.timeframe or default_timeframe)
    symbols = StockDiscovery(max_stocks=settings.max_active_stocks).universe
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    manager = HistoryManager(symbols=symbols)
    histories = {}
    for index, symbol in enumerate(symbols, start=1):
        if timeframe == Timeframe.D1:
            await asyncio.to_thread(manager.fetch_history, symbol)
            frame = manager.get_history(symbol, bars=args.bars)
        else:
            frame = await asyncio.to_thread(
                manager.get_multi_timeframe_history, symbol, timeframe, args.bars
            )
        if frame is not None and len(frame) >= settings.signal_discovery_min_periods:
            if settings.signals_exclude_forming_bar and len(frame) > 1:
                frame = frame.iloc[:-1]
            histories[symbol] = frame
        if index % 25 == 0:
            print(f"Loaded {index}/{len(symbols)} symbols ({len(histories)} usable)")

    if len(histories) < settings.signal_discovery_min_cross_section:
        raise RuntimeError(
            f"Only {len(histories)} usable symbols; need at least "
            f"{settings.signal_discovery_min_cross_section}"
        )

    panel = build_ohlcv_panel(histories)
    result = await SignalDiscoveryWorkflow.from_settings(timeframe=timeframe.value).run(
        args.request, panel
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
