from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...common.config import ProjectConfig
from ...common.logging import get_logger
from ...domain.entities import OptimizationRecommendation
from ...domain.repositories import DatasetRepository, ModelPersistence
from ...domain.value_objects import DataSplit, FeatureRange
from ...infrastructure.optimization.optuna_optimizer import OptunaTemperatureOptimizer
from ...infrastructure.optimization.writer import OptimizationResultCSVWriter


@dataclass
class OptimizeTemperatureCombinationUseCase:
    dataset_repository: DatasetRepository
    model_store: ModelPersistence
    optimizer: OptunaTemperatureOptimizer
    result_writer: OptimizationResultCSVWriter
    config: ProjectConfig
    logger: logging.Logger = get_logger("OptimizeTemperatureCombinationUseCase")

    def execute(
        self,
        model_name: str = "xgboost_regressor",
        split: DataSplit = DataSplit.VALIDATION,
        lot_ids: Optional[Sequence[str]] = None,
        max_lots: Optional[int] = None,
    ) -> Path:
        self.config.ensure_directories()
        dataset = self.dataset_repository.load_split(split)
        model_path = self.config.paths.models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train the model before optimization."
            )
        model = self.model_store.load(str(model_path))

        candidate_features = self._select_target_features(dataset)
        if not candidate_features:
            raise ValueError("No temperature features available for optimization.")

        feature_ranges = self.dataset_repository.feature_ranges(candidate_features)
        missing_ranges = [
            feature for feature in candidate_features if feature not in feature_ranges
        ]
        if missing_ranges:
            raise ValueError(f"Missing range information for features: {missing_ranges}")

        selected_lots = self._select_lots(dataset.lots, lot_ids, max_lots)

        results: List[OptimizationRecommendation] = []
        for lot_id in selected_lots:
            base_row = self._extract_lot_row(dataset, lot_id)
            baseline_prediction = float(model.predict(pd.DataFrame([base_row]))[0])
            best_params, best_value = self.optimizer.optimize(
                model=model,
                base_features=base_row,
                feature_ranges={name: feature_ranges[name] for name in candidate_features},
            )

            recommendation = OptimizationRecommendation(
                lot_id=lot_id,
                feature_values=best_params,
                predicted_defect_rate=best_value,
                baseline_prediction=baseline_prediction,
                improvement=baseline_prediction - best_value,
            )
            results.append(recommendation)
            self.logger.info(
                "Optimized lot %s: baseline %.6f -> %.6f",
                lot_id,
                baseline_prediction,
                best_value,
            )

        output_path = self.config.paths.optimization_dir / f"{model_name}_{split.value}_optimization.csv"
        self.result_writer.write(results, str(output_path))
        return output_path

    def _select_target_features(self, dataset) -> List[str]:
        if self.config.optimization.target_temperature_features:
            return [
                feature
                for feature in self.config.optimization.target_temperature_features
                if feature in dataset.features.columns
            ]

        temperature_features = [
            feature
            for feature in self.dataset_repository.list_temperature_features()
            if feature in dataset.features.columns
        ]
        return temperature_features[:5]

    def _select_lots(
        self,
        lots: pd.Series,
        lot_ids: Optional[Sequence[str]],
        max_lots: Optional[int],
    ) -> List[str]:
        if lot_ids:
            selected = [lot_id for lot_id in lot_ids if lot_id in lots.values]
        else:
            selected = list(dict.fromkeys(lots.values))
        if max_lots is not None:
            selected = selected[:max_lots]
        return selected

    def _extract_lot_row(self, dataset, lot_id: str) -> pd.Series:
        mask = dataset.lots == lot_id
        if not mask.any():
            raise ValueError(f"Lot {lot_id} not found in dataset.")
        row = dataset.features.loc[mask].iloc[0].copy()
        return row

