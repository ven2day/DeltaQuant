"""Legacy launcher for the canonical NSE runtime.

New deployments should use ``uv run deltaquant-nse``.  This file intentionally
contains no trading implementation and remains only for operator compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markets.nse.runtime.live import main

if __name__ == "__main__":
    main()
