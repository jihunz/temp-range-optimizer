from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .entities import DatasetSplit, FeatureImportance, OptimizationRecommendation
from .value_objects import DataSplit, FeatureRange


class DatasetRepository(Protocol):
    @abstractmethod
    def load_split(self, split: DataSplit) -> DatasetSplit:
        raise NotImplementedError

    @abstractmethod
    def list_feature_names(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def list_temperature_features(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def target_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def feature_ranges(self, feature_names: Iterable[str]) -> Mapping[str, FeatureRange]:
        raise NotImplementedError


class RegressionModel(Protocol):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Sequence[float]:
        raise NotImplementedError


class ModelTrainer(Protocol):
    @abstractmethod
    def train(
        self,
        training_data: DatasetSplit,
        validation_data: DatasetSplit | None = None,
    ) -> RegressionModel:
        raise NotImplementedError


class MetricEvaluator(Protocol):
    @abstractmethod
    def evaluate(self, model: RegressionModel, dataset: DatasetSplit) -> Dict[str, float]:
        raise NotImplementedError


class ModelPersistence(Protocol):
    @abstractmethod
    def save(self, model: Any, destination: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self, source: str) -> RegressionModel:
        raise NotImplementedError


class FeatureImportanceWriter(Protocol):
    @abstractmethod
    def write(self, importances: Sequence[FeatureImportance], destination: str) -> str:
        raise NotImplementedError


class MetricsWriter(Protocol):
    @abstractmethod
    def write(self, metrics: Dict[str, float], destination: str) -> str:
        raise NotImplementedError


class VisualizationWriter(Protocol):
    @abstractmethod
    def save_plot(self, figure: Any, destination: str) -> str:
        raise NotImplementedError


class OptimizationResultWriter(Protocol):
    @abstractmethod
    def write(self, results: Sequence[OptimizationRecommendation], destination: str) -> str:
        raise NotImplementedError
