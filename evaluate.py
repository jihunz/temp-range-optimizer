from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import (
    TargetScaler,
    configure_matplotlib,
    ensure_logger,
    ensure_result_directories,
    load_config,
    load_dataset,
    read_latest_run_marker,
)


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mae = float(df["absolute_error"].mean())
    rmse = float(np.sqrt(df["squared_error"].mean()))
    actual = df["actual_defect_rate"].values
    predicted = df["predicted_defect_rate"].values
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _save_comparison_plot(df: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sorted_df = df.sort_values("actual_defect_rate")
    indices = np.arange(len(sorted_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices - width / 2, sorted_df["actual_defect_rate"], width, label="Actual")
    plt.bar(indices + width / 2, sorted_df["predicted_defect_rate"], width, label="Predicted")
    plt.xticks(indices, sorted_df["LOT_NO"], rotation=60, ha="right")
    plt.ylabel("Defect rate")
    plt.title("LOT-level actual vs predicted defect rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=300)
    plt.close()
    return destination


def _save_lot_shap_summary(
    model,
    scaler: TargetScaler,
    dataset,
    destination: Path,
    top_k: int = 5,
) -> Optional[Path]:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(dataset.features).values
    shap_values = scaler.inverse_values(shap_values)
    shap_abs = np.abs(shap_values)
    shap_df = pd.DataFrame(shap_abs, columns=dataset.features.columns)
    shap_df["LOT_NO"] = dataset.lots.values

    grouped = shap_df.groupby("LOT_NO").mean()
    records = []
    for lot_id, row in grouped.iterrows():
        top_features = row.sort_values(ascending=False).head(top_k)
        for rank, (feature, value) in enumerate(top_features.items(), start=1):
            records.append(
                {
                    "LOT_NO": lot_id,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                }
            )

    if not records:
        return None

    summary_df = pd.DataFrame(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(destination, index=False)
    return destination


def evaluate(
    *,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name: str = "xgb_baseline",
    split: str = "validation",
    compute_shap: bool = True,
) -> Dict[str, Optional[str]]:
    """Evaluate the trained model on a LOT-based split and persist reports."""

    ensure_logger()
    configure_matplotlib()
    config = load_config(config_path)
    base_result_dir = Path(str(config["paths"]["result_root"]))

    if run_id is None:
        run_id = read_latest_run_marker(base_result_dir, config)
        if run_id is None:
            raise ValueError("No recorded run. Train the model before running evaluation.")

    run_paths = ensure_result_directories(base_result_dir, config, run_id)
    model_path = run_paths["models"] / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    dataset = load_dataset(split, config)
    scaler = TargetScaler.from_config(config["training"].get("target_scaler", {}))

    predictions = scaler.inverse_values(model.predict(dataset.features))
    report_df = pd.DataFrame(
        {
            "LOT_NO": dataset.lots.values,
            "actual_defect_rate": dataset.target.values,
            "predicted_defect_rate": predictions,
        }
    )
    report_df["absolute_error"] = np.abs(
        report_df["actual_defect_rate"] - report_df["predicted_defect_rate"]
    )
    report_df["squared_error"] = (
        report_df["actual_defect_rate"] - report_df["predicted_defect_rate"]
    ) ** 2

    metrics = _compute_metrics(report_df)

    reports_dir = run_paths["reports"]
    plots_dir = run_paths["plots"]

    report_path = reports_dir / f"{model_name}_{split}_lot_evaluation.csv"
    metrics_path = reports_dir / f"{model_name}_{split}_lot_metrics.json"
    plot_path = plots_dir / f"{model_name}_{split}_lot_comparison.png"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    _save_comparison_plot(report_df, plot_path)

    shap_summary_path: Optional[Path] = None
    if compute_shap:
        shap_summary_path = _save_lot_shap_summary(
            model,
            scaler,
            dataset,
            reports_dir / f"{model_name}_{split}_lot_shap_summary.csv",
        )

    logging.info("Evaluation completed for run %s", run_id)

    result: Dict[str, Optional[str]] = {
        "run_id": run_id,
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path),
    }
    if shap_summary_path:
        result["shap_summary_path"] = str(shap_summary_path)
    return result


if __name__ == "__main__":
    evaluate()

