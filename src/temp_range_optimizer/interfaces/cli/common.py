from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ...common.config import ProjectConfig, load_project_config


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the configuration file (JSON or YAML).",
    )


def parse_project_config(args: argparse.Namespace) -> ProjectConfig:
    config_path: Optional[Path] = getattr(args, "config", None)
    return load_project_config(config_path)

