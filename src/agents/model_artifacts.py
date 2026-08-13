"""Compatibility alias for market-isolated artifacts in :mod:`src.core.ml`."""

import sys

from src.core.ml import artifacts as _implementation

sys.modules[__name__] = _implementation
