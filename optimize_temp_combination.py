from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.optimize_temperature import (
    OptimizeTemperatureCombinationUseCase,
)
from temp_range_optimizer.common.config import load_project_config
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.common.run import read_latest_run_marker
from temp_range_optimizer.common.scaling import TargetScaler
from temp_range_optimizer.domain.value_objects import DataSplit
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import JoblibModelPersistence
from temp_range_optimizer.infrastructure.optimization.optuna_optimizer import (
    OptunaTemperatureOptimizer,
)
from temp_range_optimizer.infrastructure.optimization.writer import (
    OptimizationResultCSVWriter,
)


# ==== 사용자 설정 영역 ====
CONFIG_PATH: Optional[Path] = None
RUN_ID: Optional[str] = None  # None이면 latest_run 사용
MODEL_NAME: str = "xgb_baseline"
SPLIT_NAME: str = "val"  # "train", "val", "test"
LOT_IDS: Optional[Sequence[str]] = None  # 예) ["HZC01", "HZC02"]
MAX_LOTS: Optional[int] = 5
# ========================


def parse_split(value: str) -> DataSplit:
    for split in DataSplit:
        if split.value == value:
            return split
    raise ValueError(f"Unsupported split value: {value}")


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
        target_scaler=target_scaler,
        logger=logging.getLogger("OptimizeTemperatureCombinationUseCase"),
    )

    logging.info("Using artifacts run %s", run_id)
    output_path = use_case.execute(
        model_name=MODEL_NAME,
        split=split,
        lot_ids=list(LOT_IDS) if LOT_IDS is not None else None,
        max_lots=MAX_LOTS,
    )
    logging.info("Optimization results saved to %s", output_path)


if __name__ == "__main__":
    main()

