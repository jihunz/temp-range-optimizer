from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


LATEST_RUN_FILENAME = "latest_run.txt"


def generate_run_id() -> str:
    """Return a timestamp-based run identifier (YYYYMMDD-HHMMSS)."""

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _latest_run_path(base_artifacts_root: Path) -> Path:
    return base_artifacts_root / LATEST_RUN_FILENAME


def write_latest_run_marker(base_artifacts_root: Path, run_id: str) -> None:
    """Persist the most recent run id under the base artifacts directory."""

    base_artifacts_root.mkdir(parents=True, exist_ok=True)
    _latest_run_path(base_artifacts_root).write_text(run_id)


def read_latest_run_marker(base_artifacts_root: Path) -> Optional[str]:
    """Return the latest recorded run id if it exists."""

    marker_path = _latest_run_path(base_artifacts_root)
    if not marker_path.exists():
        return None
    run_id = marker_path.read_text().strip()
    return run_id or None

