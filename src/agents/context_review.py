"""Compatibility alias for the shared selective-Qwen policy."""

import sys

from src.core.qwen import policy as _implementation

sys.modules[__name__] = _implementation
