"""Replay closed historical candles into an isolated, clearly tagged signal log.

This is intentionally not a backtest and never invokes the LangGraph validation or
risk nodes. Each emitted record is a raw SignalEngine result with status="approved"
only because the signal was generated; ``source="backfill"`` and its reason make
clear that it was not approved by the live decision pipeline.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.core.candidates import SignalEngine
from src.core.indicators import Timeframe, calculate_indicators
from src.markets.nse.execution.signal_log import SignalLogger, SignalRecord
from src.markets.nse.market_data.historical_feed import HistoricalDataFeed
from src.markets.nse.market_data.history_manager import fetch_timeframe_history
from src.markets.nse.sessions.market_time import now_ist
from src.markets.nse.universe.discovery import StockDiscovery

logger = logging.getLogger(__name__)

BACKFILL_REASON = "Historical raw signal; validation and risk checks were not replayed."
TIMEFRAME_BY_VALUE = {timeframe.value: timeframe for timeframe in Timeframe}
INTRADAY_DURATION = {
    Timeframe.M15: pd.Timedelta(minutes=15),
    Timeframe.M30: pd.Timedelta(minutes=30),
    Timeframe.H1: pd.Timedelta(hours=1),
    Timeframe.H4: pd.Timedelta(hours=4),
}


def parse_timeframes(raw: str) -> list[Timeframe]:
    """Parse the live setting's format without importing the live runner."""
    timeframes = []
    for token in raw.split(","):
        timeframe = TIMEFRAME_BY_VALUE.get(token.strip())
        if timeframe is None:
            logger.warning("Ignoring unsupported signal timeframe %r", token)
        elif timeframe not in timeframes:
            timeframes.append(timeframe)
    if not timeframes:
        raise ValueError("No supported timeframes were supplied")
    return timeframes


def candle_close_timestamp(timestamp: pd.Timestamp, timeframe: Timeframe) -> str:
    """Convert a left-labelled candle timestamp to its close time."""
    timestamp = pd.Timestamp(timestamp)
    if timeframe == Timeframe.D1:
        return (timestamp.normalize() + pd.Timedelta(hours=15, minutes=30)).isoformat()
    return (timestamp + INTRADAY_DURATION[timeframe]).isoformat()


def replay_signals(
    history: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    start_day: date,
    end_day: date,
    signal_engine: SignalEngine,
) -> list[SignalRecord]:
    """Generate raw signals using only data available at each completed candle."""
    records = []
    for position, candle_time in enumerate(history.index):
        candle_day = pd.Timestamp(candle_time).date()
        if candle_day < start_day or candle_day > end_day:
            continue

        # Indicators are recomputed from the expanding prefix. This deliberately avoids
        # using a later candle while evaluating the bar at ``position``.
        prefix = history.iloc[: position + 1]
        if len(prefix) < 26:
            continue
        indicators = calculate_indicators(prefix, symbol, timeframe=timeframe)
        for signal in signal_engine.generate_signals(indicators):
            signal_data = signal.to_dict()
            signal_data["timestamp"] = candle_close_timestamp(candle_time, timeframe)
            records.append(
                SignalRecord.from_signal(
                    signal_data,
                    "approved",
                    reason=BACKFILL_REASON,
                    source="backfill",
                )
            )
    return records


async def discover_symbols(max_stocks: int) -> list[str]:
    """Use the same current dynamic-discovery entry point as the live runner."""
    return await StockDiscovery(max_stocks=max_stocks).discover()


def parse_symbols(raw: str | None, max_stocks: int) -> list[str]:
    if raw:
        symbols = [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]
        if not symbols:
            raise ValueError("--symbols did not contain any symbols")
        return list(dict.fromkeys(symbols))
    return asyncio.run(discover_symbols(max_stocks))


def ensure_no_duplicate_backfill(
    signal_logger: SignalLogger, start_day: date, end_day: date, append: bool
) -> None:
    """Prevent accidentally duplicating a backfill run over the same date range.

    Both live and backfill records share one Postgres table now (distinguished by the
    ``source`` column), so the old "never write into the live directory" check no longer
    applies — this instead checks for existing ``source="backfill"`` rows in the
    requested range.
    """
    if append:
        return
    existing = signal_logger.read_recent(days=(now_ist().date() - start_day).days + 1)
    overlap = [
        r
        for r in existing
        if r.get("source") == "backfill" and start_day.isoformat() <= r["timestamp"][:10] <= end_day.isoformat()
    ]
    if overlap:
        raise ValueError(
            f"{len(overlap)} backfill record(s) already exist for {start_day}..{end_day}; "
            "use --append only when duplicates are intended"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Number of completed calendar days")
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Final completed day (YYYY-MM-DD); defaults to yesterday in IST",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols; skips dynamic discovery for a repeatable universe",
    )
    parser.add_argument("--max-stocks", type=int, default=15)
    parser.add_argument(
        "--timeframes",
        help="Comma-separated timeframes; defaults to the live signal_timeframes setting",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Allow duplicate backfill rows for a date range already replayed",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate and report records without writing"
    )
    return parser


def run(args: argparse.Namespace) -> tuple[int, list[str]]:
    if args.days < 1:
        raise ValueError("--days must be at least 1")
    if args.max_stocks < 1:
        raise ValueError("--max-stocks must be at least 1")

    end_day = args.end_date or (now_ist().date() - timedelta(days=1))
    start_day = end_day - timedelta(days=args.days - 1)
    settings = get_settings()
    timeframes = parse_timeframes(args.timeframes or settings.signal_timeframes)
    symbols = parse_symbols(args.symbols, args.max_stocks)

    signal_logger = SignalLogger()
    if not args.dry_run:
        ensure_no_duplicate_backfill(signal_logger, start_day, end_day, args.append)

    feed = HistoricalDataFeed(symbols=symbols)
    signal_engine = SignalEngine()
    records: list[SignalRecord] = []
    skipped: list[str] = []
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                history = fetch_timeframe_history(feed, symbol, timeframe)
            except Exception:
                logger.exception("Failed to fetch %s [%s]", symbol, timeframe.value)
                skipped.append(f"{symbol} [{timeframe.value}]")
                continue
            if history is None:
                logger.warning("No real historical data for %s [%s]", symbol, timeframe.value)
                skipped.append(f"{symbol} [{timeframe.value}]")
                continue
            records.extend(
                replay_signals(history, symbol, timeframe, start_day, end_day, signal_engine)
            )

    if not args.dry_run:
        for record in records:
            signal_logger.log(record)
    return len(records), skipped


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        record_count, skipped = run(args)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2

    mode = "Dry run generated" if args.dry_run else "Wrote"
    print(f"{mode} {record_count} backfill records to Postgres")
    if skipped:
        print(f"Skipped {len(skipped)} symbol/timeframe pairs with no usable data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
