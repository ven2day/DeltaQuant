"""Process-isolated strategy validation launcher.

This module deliberately has no import from either market runtime.  Validation may
run for hours, but a crash or resource spike cannot share a process with trading.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline DeltaQuant validator")
    parser.add_argument("--market", choices=("NSE", "FOREX"), required=True)
    parser.add_argument("validator_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    script = "validate_strategy.py" if args.market == "NSE" else "validate_forex_strategies.py"
    command = [sys.executable, f"scripts/{script}", *args.validator_args]
    completed = subprocess.run(command, check=False)  # noqa: S603 - fixed local script path
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
