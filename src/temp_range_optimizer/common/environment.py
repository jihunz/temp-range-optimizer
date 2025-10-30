from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def ensure_matplotlib_config_dir(directory: Optional[Path] = None) -> Path:
    if directory is None:
        directory = Path.cwd() / ".matplotlib"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(directory)
    try:
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        try:
            matplotlib.rcParams["font.family"] = "AppleGothic"
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
    except Exception:
        pass
    return directory
