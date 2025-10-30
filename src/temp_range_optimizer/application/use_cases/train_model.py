from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from ...common.config import ProjectConfig
from ...common.scaling import TargetScaler
from ...common.logging import get_logger
from ...domain.entities import DatasetSplit, FeatureImportance, ModelTrainingResult
from ...domain.repositories import (
    DatasetRepository,
    FeatureImportanceWriter,
    MetricEvaluator,
    MetricsWriter,
    ModelPersistence,
    ModelTrainer,
)
from ...domain.value_objects import DataSplit
from ...infrastructure.modeling.xgboost_trainer import compute_feature_importances


@dataclass
class TrainModelUseCase:
    dataset_repository: DatasetRepository
    trainer: ModelTrainer
    evaluator: MetricEvaluator
    model_store: ModelPersistence
    metrics_writer: MetricsWriter
    feature_importance_writer: FeatureImportanceWriter | None
    config: ProjectConfig
    target_scaler: TargetScaler
    logger: logging.Logger = get_logger("TrainModelUseCase")

    def execute(self, model_name: str = "xgboost_regressor") -> ModelTrainingResult:
        """Train XGBoost 모델을 학습하고 산출물을 저장한다.

        절차
          1. 데이터 분할 로드 (train/val/test)
          2. TargetScaler로 타깃을 스케일링한 뒤 모델 학습
          3. 각 분할에 대한 예측 성능 평가
          4. 모델, 메트릭, 피처 중요도를 Run 디렉터리에 저장
        """
        self.logger.info("Starting model training for %s", model_name)
        self.config.ensure_directories()

        train_split = self.dataset_repository.load_split(DataSplit.TRAIN)
        val_split = self.dataset_repository.load_split(DataSplit.VALIDATION)
        try:
            test_split = self.dataset_repository.load_split(DataSplit.TEST)
        except FileNotFoundError:
            self.logger.warning("Test split not found. Proceeding without test evaluation.")
            test_split = None

        scaled_train_split = self._scale_split(train_split)
        scaled_val_split = self._scale_split(val_split)

        model = self.trainer.train(scaled_train_split, scaled_val_split)
        self.logger.info("Model training completed.")

        metrics: Dict[str, Dict[str, float]] = {
            "train": self.evaluator.evaluate(model, train_split),
            "validation": self.evaluator.evaluate(model, val_split),
        }
        if test_split is not None:
            metrics["test"] = self.evaluator.evaluate(model, test_split)

        metrics_path = self.config.paths.reports_dir / f"{model_name}_metrics.json"
        self.metrics_writer.write(metrics, str(metrics_path))
        self.logger.info("Metrics written to %s", metrics_path)

        model_path = self.config.paths.models_dir / f"{model_name}.joblib"
        self.model_store.save(model, str(model_path))
        self.logger.info("Model saved to %s", model_path)

        importance_path: Optional[str] = None
        if self.feature_importance_writer is not None:
            importances = compute_feature_importances(model)
            importance_destination = self.config.paths.reports_dir / f"{model_name}_feature_importances.csv"
            self.feature_importance_writer.write(importances, str(importance_destination))
            importance_path = str(importance_destination)
            self.logger.info("Feature importances written to %s", importance_destination)

        return ModelTrainingResult(
            metrics=metrics,
            model_path=str(model_path),
            feature_importances_path=importance_path,
        )

    def _scale_split(self, split: DatasetSplit) -> DatasetSplit:
        if not self.target_scaler.enabled:
            return split
        return DatasetSplit(
            features=split.features,
            target=self.target_scaler.scale_series(split.target),
            lots=split.lots,
        )

