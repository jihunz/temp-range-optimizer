from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.analyze_shap import AnalyzeShapInteractionsUseCase
from temp_range_optimizer.common.config import load_project_config
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.common.run import read_latest_run_marker
from temp_range_optimizer.common.scaling import TargetScaler
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence


# ==== 사용자 설정 영역 ====
CONFIG_PATH: Optional[Path] = None
RUN_ID: Optional[str] = None  # None이면 latest_run 사용
MODEL_NAME: str = "xgb_baseline"
# ========================
def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    config = load_project_config(CONFIG_PATH)
    base_artifacts_root = config.paths.artifacts_root
    run_id = RUN_ID or read_latest_run_marker(base_artifacts_root)
    if run_id is None:
        raise ValueError("latest_run.txt 가 없으므로 RUN_ID를 직접 지정하세요.")
    config.paths.artifacts_root = base_artifacts_root / run_id
    target_scaler = TargetScaler.from_config(config.target_scaling)

    use_case = AnalyzeShapInteractionsUseCase(
        dataset_repository=PandasDatasetRepository(config.data),
        model_store=JoblibModelPersistence(),
        config=config,
        target_scaler=target_scaler,
        logger=logging.getLogger("AnalyzeShapInteractionsUseCase"),
    )

    logging.info("Using artifacts run %s", run_id)
    artifacts = use_case.execute(model_name=MODEL_NAME)
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

