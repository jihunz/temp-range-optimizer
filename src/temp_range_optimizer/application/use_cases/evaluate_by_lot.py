from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ...common.config import ProjectConfig
from ...common.logging import get_logger
from ...common.scaling import TargetScaler
from ...domain.repositories import DatasetRepository, ModelPersistence
from ...domain.value_objects import DataSplit


@dataclass
class LotEvaluationResult:
    report_path: Path
    plot_path: Path
    metrics_path: Path
    shap_summary_path: Optional[Path]


@dataclass
class EvaluateByLotUseCase:
    dataset_repository: DatasetRepository
    model_store: ModelPersistence
    config: ProjectConfig
    target_scaler: TargetScaler
    logger: logging.Logger = get_logger("EvaluateByLotUseCase")

    def execute(
        self,
        model_name: str = "xgboost_regressor",
        split: DataSplit = DataSplit.VALIDATION,
        compute_shap: bool = True,
    ) -> LotEvaluationResult:
        self.config.ensure_directories()
        dataset = self.dataset_repository.load_split(split)
        model_path = self.config.paths.models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = self.model_store.load(str(model_path))

        predictions = np.asarray(model.predict(dataset.features))
        predictions = self.target_scaler.inverse_values(predictions)
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

        metrics = self._compute_metrics(report_df)

        report_path = (
            self.config.paths.reports_dir / f"{model_name}_{split.value}_lot_evaluation.csv"
        )
        metrics_path = (
            self.config.paths.reports_dir / f"{model_name}_{split.value}_lot_metrics.json"
        )
        plot_path = (
            self.config.paths.plots_dir / f"{model_name}_{split.value}_lot_comparison.png"
        )
        report_df.to_csv(report_path, index=False)
        metrics_path.write_text(pd.Series(metrics).to_json(indent=2))
        self._create_lot_comparison_plot(report_df, plot_path)

        shap_summary_path: Optional[Path] = None
        if compute_shap:
            shap_summary_path = self._compute_lot_shap_summary(
                model_name=model_name,
                split=split,
                model=model,
                dataset=dataset,
            )

        return LotEvaluationResult(
            report_path=report_path,
            plot_path=plot_path,
            metrics_path=metrics_path,
            shap_summary_path=shap_summary_path,
        )

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        mae = float(df["absolute_error"].mean())
        rmse = float(np.sqrt(df["squared_error"].mean()))
        actual = df["actual_defect_rate"].values
        predicted = df["predicted_defect_rate"].values
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def _create_lot_comparison_plot(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_df = df.sort_values("actual_defect_rate")
        indices = np.arange(len(sorted_df))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(indices - width / 2, sorted_df["actual_defect_rate"], width, label="Actual")
        plt.bar(indices + width / 2, sorted_df["predicted_defect_rate"], width, label="Predicted")
        plt.xticks(indices, sorted_df["LOT_NO"], rotation=60, ha="right")
        plt.ylabel("Defect Rate")
        plt.title("LOT-level Actual vs Predicted Defect Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

    def _compute_lot_shap_summary(
        self,
        model_name: str,
        split: DataSplit,
        model,
        dataset,
    ) -> Path:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(dataset.features).values
        shap_values = self.target_scaler.inverse_values(shap_values)
        shap_abs = np.abs(shap_values)
        shap_df = pd.DataFrame(shap_abs, columns=dataset.features.columns)
        shap_df["LOT_NO"] = dataset.lots.values

        lot_feature_importance = shap_df.groupby("LOT_NO").mean()
        records = []
        top_k = min(5, shap_df.shape[1] - 1)
        for lot_id, row in lot_feature_importance.iterrows():
            top_features = row.sort_values(ascending=False).head(top_k)
            for rank, (feature, value) in enumerate(top_features.items(), start=1):
                records.append(
                    {
                        "LOT_NO": lot_id,
                        "rank": rank,
                        "feature": feature,
                        "mean_abs_shap": value,
                    }
                )

        summary_df = pd.DataFrame(records)
        shap_summary_path = (
            self.config.paths.reports_dir / f"{model_name}_{split.value}_lot_shap_summary.csv"
        )
        summary_df.to_csv(shap_summary_path, index=False)
        return shap_summary_path
