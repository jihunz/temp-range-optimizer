from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DataConfig:
    base_dir: Path = Path("data")
    train_file: str = "lot_dataset_train.csv"
    validation_file: str = "lot_dataset_val.csv"
    test_file: str = "lot_dataset_test.csv"
    target_column: str = "불량률"
    lot_column: str = "LOT_NO"
    datetime_columns: Sequence[str] = (
        "start_time",
        "end_time",
        "작업일",
        "작업일_최신",
    )
    exclude_columns: List[str] = field(
        default_factory=lambda: [
            "LOT_NO",
            "start_time",
            "end_time",
            "작업일",
            "작업일_최신",
            "양품수량",
            "불량수량",
            "총수량",
        ]
    )
    temperature_feature_keywords: Sequence[str] = ("온도", "OP")

    def __post_init__(self) -> None:
        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)

    @property
    def train_path(self) -> Path:
        return self.base_dir / self.train_file

    @property
    def validation_path(self) -> Path:
        return self.base_dir / self.validation_file

    @property
    def test_path(self) -> Path:
        return self.base_dir / self.test_file


@dataclass
class PathsConfig:
    artifacts_root: Path = Path("artifacts")
    models_subdir: str = "models"
    reports_subdir: str = "reports"
    plots_subdir: str = "plots"
    shap_subdir: str = "shap"
    optimization_subdir: str = "optimization"

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts_root, Path):
            self.artifacts_root = Path(self.artifacts_root)

    @property
    def models_dir(self) -> Path:
        return self.artifacts_root / self.models_subdir

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_root / self.reports_subdir

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_root / self.plots_subdir

    @property
    def shap_dir(self) -> Path:
        return self.plots_dir / self.shap_subdir

    @property
    def optimization_dir(self) -> Path:
        return self.artifacts_root / self.optimization_subdir


@dataclass
class TrainingConfig:
    random_seed: int = 42
    test_size: float = 0.2
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 0.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1,
            "tree_method": "hist",
        }
    )


@dataclass
class ShapConfig:
    max_samples: int = 2000
    top_k_features: int = 15
    interaction_top_k: int = 10
    dependence_target_features: Optional[Sequence[str]] = None
    lot_level_summary_top_k: int = 5


@dataclass
class OptimizationConfig:
    target_temperature_features: Optional[Sequence[str]] = None
    candidate_feature_patterns: Sequence[str] = (
        "건조",
        "소입로",
        "솔트",
    )
    n_trials: int = 60
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1
    seed: int = 2024
    lot_batch_size: int = 5


@dataclass
class EvaluationConfig:
    top_k_lots: int = 10
    output_prediction_details: bool = True


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        self.paths.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.plots_dir.mkdir(parents=True, exist_ok=True)
        self.paths.shap_dir.mkdir(parents=True, exist_ok=True)
        self.paths.optimization_dir.mkdir(parents=True, exist_ok=True)


def _load_from_mapping(mapping: Dict[str, Any]) -> ProjectConfig:
    def build(section_cls, key: str):
        section_data = mapping.get(key, {})
        return section_cls(**section_data)

    return ProjectConfig(
        data=build(DataConfig, "data"),
        paths=build(PathsConfig, "paths"),
        training=build(TrainingConfig, "training"),
        shap=build(ShapConfig, "shap"),
        optimization=build(OptimizationConfig, "optimization"),
        evaluation=build(EvaluationConfig, "evaluation"),
    )


def load_project_config(path: Optional[Path]) -> ProjectConfig:
    if path is None:
        return ProjectConfig()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if file_path.suffix.lower() in {".json"}:
        data = json.loads(file_path.read_text())
        return _load_from_mapping(data)

    if file_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded import
            raise ImportError(
                "pyyaml is required to load YAML configuration files."
            ) from exc
        data = yaml.safe_load(file_path.read_text())  # type: ignore
        return _load_from_mapping(data or {})

    raise ValueError(f"Unsupported configuration format: {file_path.suffix}")
