from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ...domain.entities import OptimizationRecommendation


@dataclass
class OptimizationResultCSVWriter:
    def write(self, results: Sequence[OptimizationRecommendation], destination: str) -> str:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for item in results:
            record = {
                "LOT_NO": item.lot_id,
                "predicted_defect_rate": item.predicted_defect_rate,
                "baseline_prediction": item.baseline_prediction,
                "improvement": item.improvement,
            }
            record.update(item.feature_values)
            records.append(record)

        df = pd.DataFrame(records)
        df.sort_values("predicted_defect_rate", inplace=True)
        df.to_csv(path, index=False)
        return str(path)
