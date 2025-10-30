from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import TargetScalingConfig


@dataclass
class TargetScaler:
    """Utility to scale small-magnitude regression targets for numerical stability."""

    enabled: bool = True
    scale_factor: float = 100_000.0

    @classmethod
    def from_config(cls, config: TargetScalingConfig) -> "TargetScaler":
        return cls(enabled=config.enabled, scale_factor=config.scale_factor)

    def scale_series(self, series: pd.Series) -> pd.Series:
        if not self.enabled:
            return series
        return series * self.scale_factor

    def inverse_series(self, series: pd.Series) -> pd.Series:
        if not self.enabled:
            return series
        return series / self.scale_factor

    def scale_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if not self.enabled:
            return array
        return array * self.scale_factor

    def inverse_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if not self.enabled:
            return array
        return array / self.scale_factor

    def inverse_scalar(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        return float(value) / self.scale_factor

    def scale_split(self, split: "DatasetSplit") -> "DatasetSplit":
        """Return a new dataset split with scaled target values."""

        if not self.enabled:
            return split

        from ..domain.entities import DatasetSplit  # Local import to avoid cycles

        return DatasetSplit(
            features=split.features,
            target=self.scale_series(split.target),
            lots=split.lots,
        )


