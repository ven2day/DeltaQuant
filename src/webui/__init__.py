"""Optional live web dashboard (FastAPI + WebSocket) mirroring the CLI dashboard.

Only imported when ``settings.enable_web_ui`` is true — see
``src/markets/nse/runtime/live.py``. Requires the ``web`` extra (``uv sync --extra web``).
"""
