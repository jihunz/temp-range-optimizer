from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import optuna
import pandas as pd

from ...common.scaling import TargetScaler
from ...domain.value_objects import FeatureRange


@dataclass
class OptunaTemperatureOptimizer:
    n_trials: int
    timeout_seconds: int | None
    seed: int
    n_jobs: int = 1

    def optimize(
        self,
        model,
        base_features: pd.Series,
        feature_ranges: Mapping[str, FeatureRange],
        target_scaler: TargetScaler | None = None,
    ) -> Dict[str, float]:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            candidate = base_features.to_frame().T.copy()
            for feature_name, range_info in feature_ranges.items():
                low, high = range_info.bounds
                value = trial.suggest_float(feature_name, low, high)
                candidate.at[0, feature_name] = value
            prediction = float(model.predict(candidate)[0])
            if target_scaler is not None and target_scaler.enabled:
                prediction = target_scaler.inverse_scalar(prediction)
            return prediction

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            n_jobs=self.n_jobs,
        )
        best_value = float(study.best_value)
        return study.best_params, best_value
