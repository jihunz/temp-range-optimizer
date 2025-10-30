from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    features: pd.DataFrame
    target: pd.Series
    lots: pd.Series

    def __post_init__(self) -> None:
        if len(self.features) != len(self.target):
            raise ValueError("Features and target must have matching rows.")
        if len(self.lots) != len(self.target):
            raise ValueError("Lots and target must have matching rows.")

    def select_columns(self, columns: Sequence[str]) -> "DatasetSplit":
        return DatasetSplit(
            features=self.features.loc[:, columns],
            target=self.target,
            lots=self.lots,
        )

    def subset_by_lots(self, lot_ids: Iterable[str]) -> "DatasetSplit":
        mask = self.lots.isin(list(lot_ids))
        return DatasetSplit(
            features=self.features.loc[mask],
            target=self.target.loc[mask],
            lots=self.lots.loc[mask],
        )


@dataclass(frozen=True)
class MetricSet:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


@dataclass(frozen=True)
class ModelTrainingResult:
    metrics: Dict[str, Dict[str, float]]
    model_path: str
    feature_importances_path: Optional[str] = None


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    importance: float


@dataclass(frozen=True)
class OptimizationRecommendation:
    lot_id: str
    feature_values: Dict[str, float]
    predicted_defect_rate: float
    baseline_prediction: float
    improvement: float


@dataclass(frozen=True)
class ShapExplanation:
    feature_frame: pd.DataFrame
    shap_values: np.ndarray
    expected_value: float
    lot_series: pd.Series
    interaction_values: Optional[np.ndarray] = None
