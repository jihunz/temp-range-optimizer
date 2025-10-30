from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd

from common import (
    TargetScaler,
    compute_feature_ranges,
    ensure_logger,
    ensure_result_directories,
    list_temperature_features,
    load_config,
    load_dataset,
    read_latest_run_marker,
)


def _select_lots(series: pd.Series, lot_ids: Optional[Sequence[str]], max_lots: Optional[int]) -> List[str]:
    if lot_ids:
        selected = [lot for lot in lot_ids if lot in set(series.values)]
    else:
        # Preserve order while removing duplicates
        seen = set()
        selected = []
        for value in series.values:
            if value in seen:
                continue
            seen.add(value)
            selected.append(value)
    if max_lots is not None:
        selected = selected[: int(max_lots)]
    return selected


def _optimize_single_lot(
    model,
    scaler: TargetScaler,
    base_row: pd.Series,
    feature_ranges: Dict[str, Tuple[float, float]],
    opt_config: Dict[str, object],
) -> Tuple[Dict[str, float], float]:
    sampler = optuna.samplers.TPESampler(seed=opt_config.get("seed", 2024))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        candidate = base_row.to_frame().T.copy()
        for feature_name, bounds in feature_ranges.items():
            low, high = bounds
            value = trial.suggest_float(feature_name, low, high)
            candidate.at[0, feature_name] = value
        prediction = float(model.predict(candidate)[0])
        return scaler.inverse_scalar(prediction)

    study.optimize(
        objective,
        n_trials=int(opt_config.get("n_trials", 60)),
        timeout=opt_config.get("timeout_seconds"),
        n_jobs=int(opt_config.get("n_jobs", 1)),
        show_progress_bar=False,
    )
    best_params = {key: float(value) for key, value in study.best_params.items()}
    best_value = float(study.best_value)
    return best_params, best_value


def optimize(
    *,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name: str = "xgb_baseline",
    split: str = "validation",
    lot_ids: Optional[Sequence[str]] = None,
    max_lots: Optional[int] = None,
) -> Path:
    """Run Optuna-based temperature optimization for selected lots."""

    ensure_logger()
    config = load_config(config_path)
    base_result_dir = Path(str(config["paths"]["result_root"]))

    if run_id is None:
        run_id = read_latest_run_marker(base_result_dir, config)
        if run_id is None:
            raise ValueError("No recorded run. Train the model before running optimization.")

    run_paths = ensure_result_directories(base_result_dir, config, run_id)
    model_path = run_paths["models"] / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    scaler = TargetScaler.from_config(config["training"].get("target_scaler", {}))

    dataset = load_dataset(split, config)
    train_split = load_dataset("train", config)

    temperature_features = list_temperature_features(dataset.features.columns, config)

    opt_config = config["optimization"]
    target_feature_list = opt_config.get("target_temperature_features")
    candidate_features: Sequence[str]
    if target_feature_list:
        candidate_features = [
            feature for feature in target_feature_list if feature in dataset.features.columns
        ]
    else:
        max_feature_setting = opt_config.get("max_temperature_features")
        if max_feature_setting is None:
            candidate_features = temperature_features
        else:
            candidate_features = temperature_features[: int(max_feature_setting)]
    if not candidate_features:
        raise ValueError("No temperature features available for optimization.")

    feature_ranges = compute_feature_ranges(train_split, candidate_features)
    missing = [feature for feature in candidate_features if feature not in feature_ranges]
    if missing:
        raise ValueError(f"Missing feature ranges for: {missing}")

    if max_lots is None:
        max_lots_setting = opt_config.get("max_lots")
        if max_lots_setting is not None:
            max_lots = int(max_lots_setting)

    selected_lots = _select_lots(dataset.lots, lot_ids, max_lots)
    if not selected_lots:
        raise ValueError("No lots selected for optimization.")

    records: List[Dict[str, float]] = []
    for lot in selected_lots:
        mask = dataset.lots == lot
        if not mask.any():
            continue
        base_row = dataset.features.loc[mask].iloc[0].copy()
        baseline_prediction = float(model.predict(base_row.to_frame().T)[0])
        baseline_prediction = scaler.inverse_scalar(baseline_prediction)

        best_params, best_value = _optimize_single_lot(
            model,
            scaler,
            base_row,
            feature_ranges,
            config["optimization"],
        )

        record: Dict[str, float] = {
            "LOT_NO": lot,
            "predicted_defect_rate": float(best_value),
            "baseline_prediction": float(baseline_prediction),
            "improvement": float(baseline_prediction - best_value),
        }
        record.update(best_params)
        records.append(record)
        logging.info(
            "Optimized lot %s: baseline=%.6f, optimized=%.6f",
            lot,
            baseline_prediction,
            best_value,
        )

    output_path = run_paths["optimization"] / f"{model_name}_{split}_optimization.csv"
    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df.sort_values("predicted_defect_rate", inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    optimize()

