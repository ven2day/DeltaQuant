"""Independent NSE runtime package.

The worker import stays lazy so state/health helpers can be imported without
initializing the full trading graph.
"""

from typing import Any


def main(*args: Any, **kwargs: Any) -> Any:
    from src.markets.nse.runtime.worker import main as worker_main

    return worker_main(*args, **kwargs)


__all__ = ["main"]
