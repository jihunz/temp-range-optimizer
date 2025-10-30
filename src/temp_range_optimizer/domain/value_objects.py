from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass(frozen=True)
class FeatureRange:
    name: str
    bounds: Tuple[float, float]

    def clamp(self, value: float) -> float:
        lower, upper = self.bounds
        return max(lower, min(upper, value))

