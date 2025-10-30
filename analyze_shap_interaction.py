from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.analyze_shap import AnalyzeShapInteractionsUseCase
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence
from temp_range_optimizer.interfaces.cli.common import add_config_argument, parse_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP-based interaction analyses and visualizations."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_regressor",
        help="Name of the trained model to load from the artifacts directory.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    args = parse_args()
    config = parse_project_config(args)

    use_case = AnalyzeShapInteractionsUseCase(
        dataset_repository=PandasDatasetRepository(config.data),
        model_store=JoblibModelPersistence(),
        config=config,
        logger=logging.getLogger("AnalyzeShapInteractionsUseCase"),
    )

    artifacts = use_case.execute(model_name=args.model_name)
    logging.info("SHAP summary plot saved to %s", artifacts.summary_plot)
    logging.info("SHAP summary table saved to %s", artifacts.summary_table)
    for path in artifacts.dependence_plots:
        logging.info("Dependence plot saved to %s", path)
    if artifacts.interaction_heatmap:
        logging.info("Interaction heatmap saved to %s", artifacts.interaction_heatmap)
    if artifacts.lot_barplots:
        logging.info("Lot-level SHAP visualization saved to %s", artifacts.lot_barplots)
    if artifacts.surface_plot:
        logging.info("Surface plot saved to %s", artifacts.surface_plot)


if __name__ == "__main__":
    main()

