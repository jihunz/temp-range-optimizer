from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from common import (
    DatasetSplit,
    TargetScaler,
    ensure_logger,
    ensure_result_directories,
    generate_run_id,
    load_config,
    load_dataset,
    write_latest_run_marker,
)


def _scale_split(split: DatasetSplit, scaler: TargetScaler) -> DatasetSplit:
    if not scaler.enabled:
        return split
    return DatasetSplit(
        features=split.features,
        target=scaler.scale_series(split.target),
        lots=split.lots,
    )


def _evaluate(model: XGBRegressor, split: DatasetSplit, scaler: TargetScaler) -> Dict[str, float]:
    predictions = model.predict(split.features)
    predictions = scaler.inverse_values(predictions)
    actual = split.target.to_numpy(dtype=float)
    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    r2 = float(r2_score(actual, predictions))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train(
    *,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name: str = "xgb_baseline",
    save_feature_importances: bool = True,
) -> Dict[str, str]:
    """Train an XGBoost regression model and persist training artefacts."""

    logger = ensure_logger()
    config = load_config(config_path)

    base_result_dir = Path(str(config["paths"]["result_root"]))
    run_id = run_id or generate_run_id()
    run_paths = ensure_result_directories(base_result_dir, config, run_id)

    train_split = load_dataset("train", config)
    val_split = load_dataset("validation", config)
    try:
        test_split = load_dataset("test", config)
    except FileNotFoundError:
        test_split = None

    training_cfg = config["training"]
    scaler = TargetScaler.from_config(training_cfg.get("target_scaler", {}))

    scaled_train = _scale_split(train_split, scaler)
    scaled_val = _scale_split(val_split, scaler)

    params = dict(training_cfg.get("model_params", {}))
    params.setdefault("random_state", training_cfg.get("random_seed", 42))
    params.setdefault("n_jobs", training_cfg.get("n_jobs", -1))
    eval_metric = training_cfg.get("eval_metric")
    if eval_metric and "eval_metric" not in params:
        params["eval_metric"] = eval_metric
    
    # Early stopping을 모델 파라미터로 설정
    early_stop = training_cfg.get("early_stopping_rounds")
    if early_stop and val_split.features.shape[0] > 0:
        params["early_stopping_rounds"] = early_stop
        params["callbacks"] = None  # 기본 콜백 사용

    model = XGBRegressor(**params)
    eval_set = []
    fit_kwargs: Dict[str, object] = {}
    if val_split.features.shape[0] > 0:
        eval_set = [
            (scaled_train.features, scaled_train.target.to_numpy(dtype=float)),
            (scaled_val.features, scaled_val.target.to_numpy(dtype=float)),
        ]
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["verbose"] = False

    model.fit(
        scaled_train.features,
        scaled_train.target.to_numpy(dtype=float),
        **fit_kwargs,
    )

    metrics: Dict[str, Dict[str, float]] = {
        "train": _evaluate(model, train_split, scaler),
        "validation": _evaluate(model, val_split, scaler),
    }
    if test_split is not None:
        metrics["test"] = _evaluate(model, test_split, scaler)

    metrics_path = run_paths["reports"] / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    model_path = run_paths["models"] / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    feature_importance_path: Optional[Path] = None
    if save_feature_importances:
        try:
            booster = model.get_booster()
            scores = booster.get_score(importance_type="gain")
            records = (
                pd.DataFrame(
                    sorted(scores.items(), key=lambda item: item[1], reverse=True),
                    columns=["feature", "importance"],
                )
                if scores
                else pd.DataFrame(columns=["feature", "importance"])
            )
        except Exception:
            try:
                importances = getattr(model, "feature_importances_", None)
                if importances is None:
                    raise AttributeError
                records = pd.DataFrame(
                    {
                        "feature": train_split.features.columns,
                        "importance": np.asarray(importances, dtype=float),
                    }
                )
            except Exception:
                records = pd.DataFrame(columns=["feature", "importance"])

        feature_importance_path = run_paths["reports"] / f"{model_name}_feature_importances.csv"
        records.to_csv(feature_importance_path, index=False)

    write_latest_run_marker(base_result_dir, config, run_id)
    logger.info("Training completed. Run %s stored at %s", run_id, run_paths["run_dir"])

    result: Dict[str, str] = {
        "run_id": run_id,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }
    if feature_importance_path is not None:
        result["feature_importances_path"] = str(feature_importance_path)
    return result


if __name__ == "__main__":
    train()

