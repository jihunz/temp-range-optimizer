from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("temp_range_optimizer")


DEFAULT_CONFIG: Dict[str, object] = {
    "paths": {
        "result_root": "result",
        "models_subdir": "models",
        "reports_subdir": "reports",
        "plots_subdir": "plots",
        "shap_subdir": "shap",
        "optimization_subdir": "optimization",
        "latest_run_filename": "latest_run.txt",
    },
    "data": {
        "train": "data/v4/lot_dataset_v4_train.csv",
        "validation": "data/v4/lot_dataset_v4_val.csv",
        # "test": "data/v4/lot_dataset_v4_test.csv",
        "test": "data/v5/lot_dataset_v4_all.csv",
        "target_column": "불량률",
        "lot_column": "LOT_NO",
        "exclude_columns": [
            "LOT_NO",
            "start_time",
            "end_time",
            "작업일",
            "작업일_최신",
            "양품수량",
            "불량수량",
            "총수량",
        ],
        "temperature_keywords": ["온도", "OP"],
    },
    "training": {
        "random_seed": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
        "model_params": {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.5,
            "min_child_weight": 5,
            "tree_method": "hist",
        },
        "target_scaler": {"enabled": True, "scale_factor": 100_000.0},
    },
    "shap": {
        "max_samples": 2000,
        "top_k_features": 15,
        "interaction_top_k": 10,
        "dependence_max_features": 5,
        "lot_top_features": 10,
        "lot_top_lots": 5,
    },
    "optimization": {
        "n_trials": 60,
        "timeout_seconds": None,
        "seed": 2024,
        "n_jobs": 1,
        "target_temperature_features": [
            "건조로_온도_1_Zone_mean",
            "건조로_온도_1_Zone_std",
            "건조로_온도_2_Zone_mean",
            "건조로_온도_2_Zone_std",
            "소입로_온도_1_Zone_mean",
            "소입로_온도_1_Zone_std",
            "소입로_온도_2_Zone_mean",
            "소입로_온도_2_Zone_std",
            "소입로_온도_3_Zone_mean",
            "소입로_온도_3_Zone_std",
            "소입로_온도_4_Zone_mean",
            "소입로_온도_4_Zone_std",
            "소입로_CP_값_mean",
            "소입로_CP_값_std",
            "솔트조_온도_1_Zone_mean",
            "솔트조_온도_1_Zone_std",
            "솔트조_온도_2_Zone_mean",
            "솔트조_온도_2_Zone_std",
            "솔트_컨베이어_온도_1_Zone_mean",
            "솔트_컨베이어_온도_1_Zone_std",
            "솔트_컨베이어_온도_2_Zone_mean",
            "솔트_컨베이어_온도_2_Zone_std",
            "세정기_mean",
            "세정기_std",
        ],
        "max_temperature_features": None,
        "max_lots": None,
    },
}


@dataclass(frozen=True)
class DatasetSplit:
    features: pd.DataFrame
    target: pd.Series
    lots: pd.Series


class TargetScaler:
    def __init__(self, enabled: bool = True, scale_factor: float = 100_000.0) -> None:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        self.enabled = enabled
        self.scale_factor = float(scale_factor)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "TargetScaler":
        enabled = bool(config.get("enabled", True))
        scale_factor = float(config.get("scale_factor", 100_000.0))
        return cls(enabled=enabled, scale_factor=scale_factor)

    def scale_series(self, series: pd.Series) -> pd.Series:
        if not self.enabled:
            return series
        return series.astype(float) * self.scale_factor

    def inverse_series(self, series: pd.Series) -> pd.Series:
        if not self.enabled:
            return series
        return series.astype(float) / self.scale_factor

    def scale_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if not self.enabled:
            return array
        return array * self.scale_factor

    def inverse_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if not self.enabled:
            return array
        return array / self.scale_factor

    def inverse_scalar(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        return float(value) / self.scale_factor


def _deep_update(base: Dict[str, object], overrides: Mapping[str, object]) -> Dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text())
    elif file_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pyyaml is required to load YAML configuration files.") from exc
        payload = yaml.safe_load(file_path.read_text()) or {}
    else:
        raise ValueError(f"Unsupported configuration format: {file_path.suffix}")

    if not isinstance(payload, Mapping):
        raise ValueError("Configuration payload must be a mapping")

    return _deep_update(DEFAULT_CONFIG, payload)


def configure_matplotlib(matplotlib_dir: Optional[Path] = None) -> None:
    directory = matplotlib_dir or Path.cwd() / ".matplotlib"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory))
    try:  # pragma: no cover - environment safeguard
        import matplotlib
        from matplotlib import font_manager

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")

        font_candidates = [
            "AppleGothic",
            "NanumGothic",
            "Malgun Gothic",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        chosen_font = None
        for name in font_candidates:
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                if path:
                    chosen_font = name
                    break
            except Exception:
                continue

        if chosen_font is None:
            chosen_font = font_candidates[-1]

        matplotlib.rcParams["font.family"] = chosen_font
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _latest_run_path(base_dir: Path, config: Mapping[str, object]) -> Path:
    filename = str(config["paths"]["latest_run_filename"])  # type: ignore[index]
    return base_dir / filename


def write_latest_run_marker(base_dir: Path, config: Mapping[str, object], run_id: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    _latest_run_path(base_dir, config).write_text(run_id)


def read_latest_run_marker(base_dir: Path, config: Mapping[str, object]) -> Optional[str]:
    marker_path = _latest_run_path(base_dir, config)
    if not marker_path.exists():
        return None
    value = marker_path.read_text().strip()
    return value or None


def ensure_result_directories(base_dir: Path, config: Mapping[str, object], run_id: str) -> Dict[str, Path]:
    paths_config: Mapping[str, object] = config["paths"]  # type: ignore[index]
    run_dir = base_dir / run_id
    models_dir = run_dir / str(paths_config["models_subdir"])
    reports_dir = run_dir / str(paths_config["reports_subdir"])
    plots_dir = run_dir / str(paths_config["plots_subdir"])
    shap_dir = plots_dir / str(paths_config["shap_subdir"])
    optimization_dir = run_dir / str(paths_config["optimization_subdir"])

    for directory in [run_dir, models_dir, reports_dir, plots_dir, shap_dir, optimization_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "models": models_dir,
        "reports": reports_dir,
        "plots": plots_dir,
        "shap": shap_dir,
        "optimization": optimization_dir,
    }


def load_dataset(split: str, config: Mapping[str, object]) -> DatasetSplit:
    data_cfg: Mapping[str, object] = config["data"]  # type: ignore[index]
    split_key = split.lower()
    if split_key in {"train", "training"}:
        path_key = "train"
    elif split_key in {"val", "validation"}:
        path_key = "validation"
    elif split_key == "test":
        path_key = "test"
    else:
        raise ValueError(f"Unsupported split: {split}")

    path_value = data_cfg.get(path_key)
    if path_value is None:
        raise ValueError(f"Path for split '{split}' is not configured")

    file_path = Path(str(path_value))
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    target_column = str(data_cfg.get("target_column", "불량률"))
    lot_column = str(data_cfg.get("lot_column", "LOT_NO"))

    if target_column not in dataframe.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset {file_path}")
    if lot_column not in dataframe.columns:
        raise KeyError(f"Lot column '{lot_column}' not found in dataset {file_path}")

    exclude_columns = set(data_cfg.get("exclude_columns", [])) | {target_column}
    columns_to_drop = [col for col in exclude_columns if col in dataframe.columns]
    features = dataframe.drop(columns=columns_to_drop)
    target = dataframe[target_column]
    lots = dataframe[lot_column]

    return DatasetSplit(features=features, target=target, lots=lots)


def list_temperature_features(columns: Iterable[str], config: Mapping[str, object]) -> Sequence[str]:
    keywords = tuple(str(keyword) for keyword in config["data"].get("temperature_keywords", []))  # type: ignore[index]
    selected = [column for column in columns if any(keyword in column for keyword in keywords)]
    return selected


def compute_feature_ranges(split: DatasetSplit, feature_names: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for feature in feature_names:
        if feature not in split.features.columns:
            continue
        column = split.features[feature]
        ranges[feature] = (float(column.min()), float(column.max()))
    return ranges


def ensure_logger(level: int = logging.INFO) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.setLevel(level)
    return LOGGER


def mean_absolute_shap(shap_values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap_values), axis=0)


