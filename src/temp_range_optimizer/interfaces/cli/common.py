from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ...common.config import ProjectConfig, load_project_config
from ...common.run import generate_run_id, read_latest_run_marker


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the configuration file (JSON or YAML).",
    )


def add_run_id_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--run-id",
        type=str,
        required=required,
        help="Identifier of the artifacts run directory (defaults to latest for read-only workflows).",
    )


def parse_project_config(args: argparse.Namespace) -> ProjectConfig:
    config_path: Optional[Path] = getattr(args, "config", None)
    return load_project_config(config_path)


def resolve_run_id(args: argparse.Namespace, base_artifacts_root: Path, *, create_new: bool) -> str:
    provided: Optional[str] = getattr(args, "run_id", None)
    if provided:
        return provided
    if create_new:
        return generate_run_id()
    latest = read_latest_run_marker(base_artifacts_root)
    if latest is None:
        raise ValueError(
            "No recorded run found. Provide --run-id explicitly or train a model first."
        )
    return latest

