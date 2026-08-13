"""NSE paper-execution and trade-journal services.

Broker adapters are imported from their owning market domains so importing the
paper engine cannot initialize Dhan recursively.
"""

from .journal import TradeJournal

__all__ = ["TradeJournal"]
