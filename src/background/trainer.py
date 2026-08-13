"""Process-isolated model challenger training launcher."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train market-scoped model challengers")
    parser.add_argument("--market", choices=("NSE", "FOREX"), required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    script = "train_prediction_artifact.py" if args.market == "NSE" else "train_forex_ml.py"
    command = [sys.executable, f"scripts/{script}", *args.trainer_args]
    completed = subprocess.run(command, check=False)  # noqa: S603 - fixed local script path
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
