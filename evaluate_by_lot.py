from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.evaluate_by_lot import EvaluateByLotUseCase
from temp_range_optimizer.common.config import load_project_config
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.common.run import read_latest_run_marker
from temp_range_optimizer.common.scaling import TargetScaler
from temp_range_optimizer.domain.value_objects import DataSplit
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence


# ==== 사용자 설정 영역 ====
CONFIG_PATH: Optional[Path] = None
RUN_ID: Optional[str] = None  # None이면 latest_run 사용
MODEL_NAME: str = "xgb_baseline"
SPLIT_NAME: str = "val"
COMPUTE_SHAP: bool = True
# ========================


def parse_split(value: str) -> DataSplit:
    for split in DataSplit:
        if split.value == value:
            return split
    raise ValueError(f"Unsupported split: {value}")


def main() -> None:
    configure_logging()
    ensure_matplotlib_config_dir()
    config = load_project_config(CONFIG_PATH)
    base_artifacts_root = config.paths.artifacts_root
    run_id = RUN_ID or read_latest_run_marker(base_artifacts_root)
    if run_id is None:
        raise ValueError("latest_run.txt 가 없으므로 RUN_ID를 직접 지정하세요.")
    config.paths.artifacts_root = base_artifacts_root / run_id
    split = parse_split(SPLIT_NAME)
    target_scaler = TargetScaler.from_config(config.target_scaling)

    use_case = EvaluateByLotUseCase(
        dataset_repository=PandasDatasetRepository(config.data),
        model_store=JoblibModelPersistence(),
        config=config,
        target_scaler=target_scaler,
        logger=logging.getLogger("EvaluateByLotUseCase"),
    )

    logging.info("Using artifacts run %s", run_id)

    result = use_case.execute(
        model_name=MODEL_NAME,
        split=split,
        compute_shap=COMPUTE_SHAP,
    )

    logging.info("Per-LOT evaluation saved to %s", result.report_path)
    logging.info("Per-LOT comparison plot saved to %s", result.plot_path)
    logging.info("Split-level metrics saved to %s", result.metrics_path)
    if result.shap_summary_path:
        logging.info("LOT-level SHAP summary saved to %s", result.shap_summary_path)


if __name__ == "__main__":
    main()

