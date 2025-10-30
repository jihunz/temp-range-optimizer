from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from ...common.config import DataConfig
from ...domain.entities import DatasetSplit
from ...domain.repositories import DatasetRepository
from ...domain.value_objects import DataSplit, FeatureRange


@dataclass
class PandasDatasetRepository(DatasetRepository):
    config: DataConfig
    _cache: Dict[DataSplit, DatasetSplit] = field(default_factory=dict, init=False)

    def load_split(self, split: DataSplit) -> DatasetSplit:
        if split in self._cache:
            return self._cache[split]

        path = self._resolve_path(split)
        dataframe = pd.read_csv(path)
        lot_series = dataframe[self.config.lot_column]
        target_series = dataframe[self.config.target_column]

        drop_columns = set(self.config.exclude_columns) | {self.config.target_column}
        feature_frame = dataframe.drop(columns=[col for col in drop_columns if col in dataframe.columns])

        dataset_split = DatasetSplit(
            features=feature_frame,
            target=target_series,
            lots=lot_series,
        )
        self._cache[split] = dataset_split
        return dataset_split

    def list_feature_names(self) -> Sequence[str]:
        return list(self._training_split.features.columns)

    def list_temperature_features(self) -> Sequence[str]:
        keywords = tuple(self.config.temperature_feature_keywords)
        return [
            feature
            for feature in self.list_feature_names()
            if any(keyword in feature for keyword in keywords)
        ]

    def target_name(self) -> str:
        return self.config.target_column

    def feature_ranges(self, feature_names: Iterable[str]) -> Mapping[str, FeatureRange]:
        feature_frame = self._training_split.features
        ranges: Dict[str, FeatureRange] = {}
        for feature in feature_names:
            if feature not in feature_frame.columns:
                continue
            column = feature_frame[feature]
            lower = float(column.min())
            upper = float(column.max())
            ranges[feature] = FeatureRange(name=feature, bounds=(lower, upper))
        return ranges

    def _resolve_path(self, split: DataSplit) -> str:
        if split is DataSplit.TRAIN:
            return str(self.config.train_path)
        if split is DataSplit.VALIDATION:
            return str(self.config.validation_path)
        if split is DataSplit.TEST:
            return str(self.config.test_path)
        raise ValueError(f"Unsupported data split: {split}")

    @cached_property
    def _training_split(self) -> DatasetSplit:
        return self.load_split(DataSplit.TRAIN)

