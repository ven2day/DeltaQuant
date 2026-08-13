"""Independent Forex worker entry point with lazy imports for compatibility."""


async def run() -> None:
    from src.markets.forex.runtime.worker import run as worker_run

    await worker_run()


def main() -> None:
    from src.markets.forex.runtime.worker import main as worker_main

    worker_main()


__all__ = ["main", "run"]
