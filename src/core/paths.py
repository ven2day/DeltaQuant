"""Repository-rooted filesystem locations shared by DeltaQuant services.

Runtime paths must not depend on the process working directory. Market-specific
components derive their mutable-state directories from these roots; no component
should write mutable state beside repository-level files.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CONFIG_ROOT = PROJECT_ROOT / "config"
DATA_ROOT = PROJECT_ROOT / "data"
ENV_ROOT = PROJECT_ROOT / "env"
LOG_ROOT = PROJECT_ROOT / "logs"
RUN_ROOT = PROJECT_ROOT / "run"


def project_path(value: str | Path) -> Path:
    """Return ``value`` unchanged if absolute, otherwise resolve below the repo root."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "ARTIFACT_ROOT",
    "CONFIG_ROOT",
    "DATA_ROOT",
    "ENV_ROOT",
    "LOG_ROOT",
    "PROJECT_ROOT",
    "RUN_ROOT",
    "project_path",
]
