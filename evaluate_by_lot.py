from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.evaluate_by_lot import EvaluateByLotUseCase
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.domain.value_objects import DataSplit
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence
from temp_range_optimizer.interfaces.cli.common import add_config_argument, parse_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model performance at the LOT level and summarize SHAP contributions."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_regressor",
        help="Model artifact name to load from the artifacts directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=[split.value for split in DataSplit],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Disable SHAP computation for faster evaluation.",
    )
    return parser.parse_args()


def parse_split(value: str) -> DataSplit:
    for split in DataSplit:
        if split.value == value:
            return split
    raise ValueError(f"Unsupported split: {value}")


def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    args = parse_args()
    config = parse_project_config(args)
    split = parse_split(args.split)

    use_case = EvaluateByLotUseCase(
        dataset_repository=PandasDatasetRepository(config.data),
        model_store=JoblibModelPersistence(),
        config=config,
        logger=logging.getLogger("EvaluateByLotUseCase"),
    )

    result = use_case.execute(
        model_name=args.model_name,
        split=split,
        compute_shap=not args.no_shap,
    )

    logging.info("Per-LOT evaluation saved to %s", result.report_path)
    logging.info("Per-LOT comparison plot saved to %s", result.plot_path)
    logging.info("Split-level metrics saved to %s", result.metrics_path)
    if result.shap_summary_path:
        logging.info("LOT-level SHAP summary saved to %s", result.shap_summary_path)


if __name__ == "__main__":
    main()

