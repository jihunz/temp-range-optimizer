from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.train_model import TrainModelUseCase
from temp_range_optimizer.common.config import ProjectConfig
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import (
    FeatureImportanceCSVWriter,
    JoblibModelPersistence,
    MetricsJSONWriter,
    SklearnMetricEvaluator,
    XGBoostModelTrainer,
)
from temp_range_optimizer.interfaces.cli.common import add_config_argument, parse_project_config


def build_use_case(config: ProjectConfig) -> TrainModelUseCase:
    dataset_repo = PandasDatasetRepository(config.data)
    trainer = XGBoostModelTrainer(config.training)
    evaluator = SklearnMetricEvaluator()
    model_store = JoblibModelPersistence()
    metrics_writer = MetricsJSONWriter()
    feature_writer = FeatureImportanceCSVWriter()
    return TrainModelUseCase(
        dataset_repository=dataset_repo,
        trainer=trainer,
        evaluator=evaluator,
        model_store=model_store,
        metrics_writer=metrics_writer,
        feature_importance_writer=feature_writer,
        config=config,
        logger=logging.getLogger("TrainModelUseCase"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model for defect rate prediction.")
    add_config_argument(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_regressor",
        help="Name used when persisting the trained model and reports.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    args = parse_args()
    config = parse_project_config(args)
    use_case = build_use_case(config)
    result = use_case.execute(model_name=args.model_name)
    logging.info("Training completed. Model stored at %s", result.model_path)
    logging.info("Metrics summary: %s", result.metrics)
    if result.feature_importances_path:
        logging.info("Feature importances stored at %s", result.feature_importances_path)


if __name__ == "__main__":
    main()
