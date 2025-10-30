from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.optimize_temperature import (
    OptimizeTemperatureCombinationUseCase,
)
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.domain.value_objects import DataSplit
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence
from temp_range_optimizer.infrastructure.optimization.optuna_optimizer import (
    OptunaTemperatureOptimizer,
)
from temp_range_optimizer.infrastructure.optimization.writer import (
    OptimizationResultCSVWriter,
)
from temp_range_optimizer.interfaces.cli.common import add_config_argument, parse_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize temperature feature combinations to minimize defect rate predictions."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_regressor",
        help="Name of the trained model artifact to load.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=[split.value for split in DataSplit],
        help="Dataset split to use for optimization.",
    )
    parser.add_argument(
        "--lot-id",
        action="append",
        dest="lot_ids",
        help="Specific LOT IDs to optimize (can be repeated).",
    )
    parser.add_argument(
        "--max-lots",
        type=int,
        default=None,
        help="Maximum number of lots to optimize.",
    )
    return parser.parse_args()


def parse_split(value: str) -> DataSplit:
    for split in DataSplit:
        if split.value == value:
            return split
    raise ValueError(f"Unsupported split value: {value}")


def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    args = parse_args()
    config = parse_project_config(args)
    split = parse_split(args.split)

    optimizer = OptunaTemperatureOptimizer(
        n_trials=config.optimization.n_trials,
        timeout_seconds=config.optimization.timeout_seconds,
        seed=config.optimization.seed,
        n_jobs=config.optimization.n_jobs,
    )

    use_case = OptimizeTemperatureCombinationUseCase(
        dataset_repository=PandasDatasetRepository(config.data),
        model_store=JoblibModelPersistence(),
        optimizer=optimizer,
        result_writer=OptimizationResultCSVWriter(),
        config=config,
        logger=logging.getLogger("OptimizeTemperatureCombinationUseCase"),
    )

    output_path = use_case.execute(
        model_name=args.model_name,
        split=split,
        lot_ids=args.lot_ids,
        max_lots=args.max_lots,
    )
    logging.info("Optimization results saved to %s", output_path)


if __name__ == "__main__":
    main()

