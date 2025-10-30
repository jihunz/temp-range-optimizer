from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temp_range_optimizer.application.use_cases.train_model import TrainModelUseCase
from temp_range_optimizer.common.config import ProjectConfig, load_project_config
from temp_range_optimizer.common.environment import ensure_matplotlib_config_dir
from temp_range_optimizer.common.logging import configure_logging
from temp_range_optimizer.common.run import generate_run_id, write_latest_run_marker
from temp_range_optimizer.common.scaling import TargetScaler
from temp_range_optimizer.infrastructure.data.pandas_repository import PandasDatasetRepository
from temp_range_optimizer.infrastructure.modeling.xgboost_trainer import (
    FeatureImportanceCSVWriter,
    JoblibModelPersistence,
    MetricsJSONWriter,
    SklearnMetricEvaluator,
    XGBoostModelTrainer,
)


# ==== 사용자 설정 영역 ====
CONFIG_PATH: Optional[Path] = None  # 예) Path("configs/experiment.yaml")
RUN_ID: Optional[str] = None  # None이면 자동 생성
MODEL_NAME: str = "xgb_baseline"
# ========================


def build_use_case(config: ProjectConfig) -> TrainModelUseCase:
    """학습에 필요한 인프라 의존성을 한 번에 조립한다.

    핵심 흐름
      1. 데이터 리포지토리 → 학습/평가/SHAP에서 동일한 데이터 분할 재사용
      2. TargetScaler → 작은 불량률 값을 스케일링하여 학습 안정화
      3. Trainer/Evaluator/Writer → XGBoost 모델 학습, 메트릭 계산, 산출물 저장
    """
    dataset_repo = PandasDatasetRepository(config.data)
    trainer = XGBoostModelTrainer(config.training)
    target_scaler = TargetScaler.from_config(config.target_scaling)
    evaluator = SklearnMetricEvaluator(target_scaler=target_scaler)
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
        target_scaler=target_scaler,
        logger=logging.getLogger("TrainModelUseCase"),
    )


def main() -> None:
    """Run ID를 생성하고 학습→평가→산출물 저장 전체 파이프라인을 실행한다."""
    configure_logging()
    ensure_matplotlib_config_dir()
    config = load_project_config(CONFIG_PATH)
    base_artifacts_root = config.paths.artifacts_root
    run_id = RUN_ID or generate_run_id()
    config.paths.artifacts_root = base_artifacts_root / run_id
    use_case = build_use_case(config)
    result = use_case.execute(model_name=MODEL_NAME)
    write_latest_run_marker(base_artifacts_root, run_id)
    logging.info("Artifacts stored under %s", config.paths.artifacts_root)
    logging.info("Training completed. Model stored at %s", result.model_path)
    logging.info("Metrics summary: %s", result.metrics)
    if result.feature_importances_path:
        logging.info("Feature importances stored at %s", result.feature_importances_path)


if __name__ == "__main__":
    main()
