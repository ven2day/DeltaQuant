"""Backfill or incrementally refresh the local candle database.

Examples:
    uv run python scripts/sync_market_history.py
    uv run python scripts/sync_market_history.py --years 2 --timeframes 5m,15m,1h,1d
    uv run python scripts/sync_market_history.py --symbols RELIANCE,TCS,HDFCBANK
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.core.candles import CandleStore  # noqa: E402
from src.markets.nse.market_data.historical_feed import HistoricalDataFeed  # noqa: E402
from src.markets.nse.market_data.history_ingestion import (  # noqa: E402
    HistoricalIngestionJob,
    parse_ingestion_timeframes,
)
from src.markets.nse.universe.discovery import StockDiscovery  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=None, help="Target history in years")
    parser.add_argument("--timeframes", default=None, help="Comma-separated native timeframes")
    parser.add_argument("--symbols", default=None, help="Comma-separated NSE symbols")
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for a smoke test")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Process only symbols with at least one requested timeframe below --minimum-rows",
    )
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=1,
        help="Coverage floor used by --missing-only (default: 1, meaning absent series)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _arguments()
    settings = get_settings()
    database_url = settings.market_history_database_url or settings.database_url
    store = CandleStore(
        database_url,
        enable_timescale=settings.market_history_enable_timescale,
        require_timescale=settings.market_history_require_timescale,
    )

    if args.symbols:
        symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()]
    else:
        symbols = StockDiscovery(max_stocks=settings.max_active_stocks).universe
    if args.limit is not None:
        symbols = symbols[: max(0, args.limit)]

    timeframes = parse_ingestion_timeframes(args.timeframes or settings.market_history_timeframes)
    if args.missing_only:
        minimum_rows = max(1, args.minimum_rows)
        requested = list(symbols)
        symbols = [
            symbol
            for symbol in requested
            if any(store.coverage(symbol, timeframe.value).rows < minimum_rows for timeframe in timeframes)
        ]
        print(
            f"Missing-only selection: {len(symbols)}/{len(requested)} symbols have a "
            f"requested series below {minimum_rows:,} rows"
        )
        if not symbols:
            print("All requested symbol/timeframe series already satisfy the coverage floor")
            return 0
    lookback_days = (
        max(30, int(args.years * 365.25))
        if args.years is not None
        else settings.market_history_lookback_days
    )
    job = HistoricalIngestionJob(
        store,
        HistoricalDataFeed(symbols=symbols),
        lookback_days=lookback_days,
        overlap_days=settings.market_history_overlap_days,
    )
    summary = job.run(symbols, timeframes)
    print(
        f"Done: {summary.series_completed} series, {summary.rows_upserted:,} rows, "
        f"{summary.series_failed} failed"
    )
    if summary.failures:
        for failure in summary.failures[:20]:
            print(f"  - {failure}")
    return 0 if summary.series_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
