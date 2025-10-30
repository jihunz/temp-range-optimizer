from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from ...common.config import TrainingConfig
from ...common.scaling import TargetScaler
from ...domain.entities import DatasetSplit, FeatureImportance
from ...domain.repositories import (
    MetricEvaluator,
    ModelPersistence,
    ModelTrainer,
    RegressionModel,
)


@dataclass
class XGBoostModelTrainer(ModelTrainer):
    config: TrainingConfig

    def train(
        self,
        training_data: DatasetSplit,
        validation_data: DatasetSplit | None = None,
    ) -> RegressionModel:
        params = dict(self.config.model_params)
        params.setdefault("random_state", self.config.random_seed)
        params.setdefault("n_jobs", self.config.n_jobs)
        if self.config.early_stopping_rounds:
            params.setdefault("early_stopping_rounds", self.config.early_stopping_rounds)

        model = XGBRegressor(**params)
        eval_set = None
        if validation_data is not None:
            eval_set = [
                (training_data.features, training_data.target),
                (validation_data.features, validation_data.target),
            ]

        model.fit(
            training_data.features,
            training_data.target,
            eval_set=eval_set,
            verbose=True,
        )
        return model


@dataclass
class SklearnMetricEvaluator(MetricEvaluator):
    target_scaler: TargetScaler | None = None

    def evaluate(self, model: RegressionModel, dataset: DatasetSplit) -> Dict[str, float]:
        predictions = np.asarray(model.predict(dataset.features))
        target = dataset.target.to_numpy()
        if self.target_scaler is not None and self.target_scaler.enabled:
            predictions = self.target_scaler.inverse_values(predictions)
        mae = mean_absolute_error(target, predictions)
        mse = mean_squared_error(target, predictions)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(target, predictions)
        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


@dataclass
class JoblibModelPersistence(ModelPersistence):
    def save(self, model: RegressionModel, destination: str) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return str(path)

    def load(self, source: str) -> RegressionModel:
        return joblib.load(source)


@dataclass
class FeatureImportanceCSVWriter:
    def write(self, importances: Sequence[FeatureImportance], destination: str) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {"feature": item.feature, "importance": item.importance} for item in importances
        ]
        df = pd.DataFrame(records, columns=["feature", "importance"])
        if not df.empty:
            df = df.sort_values("importance", ascending=False)
        df.to_csv(path, index=False)
        return str(path)


@dataclass
class MetricsJSONWriter:
    def write(self, metrics: Dict[str, object], destination: str) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))
        return str(path)


def compute_feature_importances(model: RegressionModel) -> Sequence[FeatureImportance]:
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        return [
            FeatureImportance(feature=feature, importance=importance)
            for feature, importance in scores.items()
        ]

    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_")
        if isinstance(importances, np.ndarray):
            importances = importances.tolist()
        return [
            FeatureImportance(feature=str(index), importance=float(value))
            for index, value in enumerate(importances)
        ]

    raise AttributeError("Model does not provide feature importances.")
