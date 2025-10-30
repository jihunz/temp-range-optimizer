from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import shap

from ...common.config import PathsConfig, ShapConfig
from ...common.scaling import TargetScaler
from ...domain.entities import DatasetSplit, ShapExplanation
from ...domain.repositories import DatasetRepository, ModelPersistence, RegressionModel
from ...domain.value_objects import DataSplit


def _mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap_values), axis=0)


@dataclass
class ShapAnalysisArtifacts:
    summary_plot: Path
    summary_table: Path
    dependence_plots: List[Path]
    interaction_heatmap: Optional[Path]
    lot_barplots: Optional[Path]
    surface_plot: Optional[Path]


@dataclass
class ShapAnalysisService:
    dataset_repository: DatasetRepository
    model_store: ModelPersistence
    shap_config: ShapConfig
    paths_config: PathsConfig
    target_scaler: TargetScaler
    logger: logging.Logger
    random_state: int = 42

    def load_dataset_sample(self, split: DataSplit) -> DatasetSplit:
        dataset = self.dataset_repository.load_split(split)
        max_samples = self.shap_config.max_samples
        if max_samples is None or len(dataset.features) <= max_samples:
            return dataset

        sampled_features = dataset.features.sample(
            n=max_samples, random_state=self.random_state
        )
        sampled_target = dataset.target.loc[sampled_features.index]
        sampled_lots = dataset.lots.loc[sampled_features.index]
        return DatasetSplit(
            features=sampled_features.reset_index(drop=True),
            target=sampled_target.reset_index(drop=True),
            lots=sampled_lots.reset_index(drop=True),
        )

    def compute_explanation(
        self,
        model_path: Path,
        dataset: DatasetSplit,
    ) -> Tuple[RegressionModel, ShapExplanation]:
        model = self.model_store.load(str(model_path))
        self.logger.info("Loaded model from %s", model_path)

        feature_frame = dataset.features
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_result = explainer(feature_frame)
        shap_values = self.target_scaler.inverse_values(shap_result.values)
        interaction_values = None
        if self.shap_config.interaction_top_k > 0:
            raw_interactions = explainer.shap_interaction_values(feature_frame)
            interaction_values = self.target_scaler.inverse_values(raw_interactions)
        expected_value = self.target_scaler.inverse_scalar(float(np.mean(shap_result.base_values)))
        explanation = ShapExplanation(
            feature_frame=feature_frame,
            shap_values=shap_values,
            expected_value=expected_value,
            lot_series=dataset.lots,
            interaction_values=interaction_values,
        )
        return model, explanation

    def generate_summary_plot(
        self, explanation: ShapExplanation, destination: Path
    ) -> Path:
        self.logger.info("Creating SHAP summary plot at %s", destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            explanation.shap_values,
            explanation.feature_frame,
            show=False,
            max_display=self.shap_config.top_k_features,
        )
        plt.tight_layout()
        plt.savefig(destination, bbox_inches="tight", dpi=300)
        plt.close()
        return destination

    def export_summary_table(
        self, explanation: ShapExplanation, destination: Path
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        mean_abs = _mean_abs_shap(explanation.shap_values)
        df = pd.DataFrame(
            {
                "feature": explanation.feature_frame.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(destination, index=False)
        return destination

    def generate_dependence_plots(
        self,
        explanation: ShapExplanation,
        features: Sequence[str],
        destination_dir: Path,
    ) -> List[Path]:
        destination_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for feature in features:
            self.logger.info("Creating dependence plot for %s", feature)
            shap.dependence_plot(
                feature,
                explanation.shap_values,
                explanation.feature_frame,
                show=False,
            )
            plt.tight_layout()
            path = destination_dir / f"dependence_{feature}.png"
            plt.savefig(path, bbox_inches="tight", dpi=300)
            plt.close()
            paths.append(path)
        return paths

    def generate_interaction_heatmap(
        self,
        explanation: ShapExplanation,
        destination: Path,
        top_k: Optional[int] = None,
    ) -> Optional[Path]:
        if explanation.interaction_values is None:
            self.logger.warning("Interaction values were not computed; skipping heatmap.")
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)

        interaction_strength = np.mean(
            np.abs(explanation.interaction_values), axis=0
        )
        np.fill_diagonal(interaction_strength, 0.0)

        features = list(explanation.feature_frame.columns)
        interaction_df = pd.DataFrame(
            interaction_strength,
            index=features,
            columns=features,
        )

        if top_k is not None and top_k < len(features):
            top_features = (
                interaction_df.abs().sum(axis=0).sort_values(ascending=False).head(top_k).index
            )
            interaction_df = interaction_df.loc[top_features, top_features]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            interaction_df,
            cmap="coolwarm",
            center=0.0,
            annot=False,
            fmt=".3f",
        )
        plt.title("SHAP Interaction Values (Mean Absolute)")
        plt.tight_layout()
        plt.savefig(destination, bbox_inches="tight", dpi=300)
        plt.close()
        return destination

    def generate_lot_barplots(
        self,
        explanation: ShapExplanation,
        destination: Path,
        top_k_features: int,
        top_k_lots: int,
    ) -> Optional[Path]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shap_abs = np.abs(explanation.shap_values)
        shap_df = pd.DataFrame(shap_abs, columns=explanation.feature_frame.columns)
        shap_df["LOT_NO"] = explanation.lot_series.values

        lot_importance = shap_df.groupby("LOT_NO").mean()
        if lot_importance.empty:
            self.logger.warning("No SHAP data available for lot-level aggregation.")
            return None

        lot_scores = lot_importance.sum(axis=1).sort_values(ascending=False)
        selected_lots = lot_scores.head(top_k_lots).index

        num_lots = len(selected_lots)
        if num_lots == 0:
            self.logger.warning("No lots selected for visualization.")
            return None

        fig, axes = plt.subplots(num_lots, 1, figsize=(12, 4 * num_lots))
        if num_lots == 1:
            axes = [axes]

        for ax, lot in zip(axes, selected_lots):
            lot_series = lot_importance.loc[lot].sort_values(ascending=False).head(top_k_features)
            ax.barh(lot_series.index[::-1], lot_series.values[::-1])
            ax.set_title(f"LOT {lot} | Top SHAP Contributions")
            ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(destination, bbox_inches="tight", dpi=300)
        plt.close()
        return destination

    def generate_surface_plot(
        self,
        model: RegressionModel,
        explanation: ShapExplanation,
        feature_pair: Tuple[str, str],
        feature_ranges: Dict[str, Tuple[float, float]],
        destination: Path,
    ) -> Optional[Path]:
        feature_a, feature_b = feature_pair
        if feature_a not in feature_ranges or feature_b not in feature_ranges:
            self.logger.warning(
                "Feature ranges missing for %s or %s; skipping surface plot.",
                feature_a,
                feature_b,
            )
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)
        range_a = feature_ranges[feature_a]
        range_b = feature_ranges[feature_b]
        grid_a = np.linspace(range_a[0], range_a[1], 40)
        grid_b = np.linspace(range_b[0], range_b[1], 40)
        mesh_a, mesh_b = np.meshgrid(grid_a, grid_b)

        baseline = explanation.feature_frame.median()
        repeated_baseline = pd.DataFrame(
            np.repeat(baseline.to_frame().T.values, mesh_a.size, axis=0),
            columns=explanation.feature_frame.columns,
        )
        repeated_baseline[feature_a] = mesh_a.ravel()
        repeated_baseline[feature_b] = mesh_b.ravel()

        predictions = np.asarray(
            model.predict(repeated_baseline[explanation.feature_frame.columns])
        )
        surface = self.target_scaler.inverse_values(predictions).reshape(mesh_a.shape)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=grid_a,
                    y=grid_b,
                    z=surface,
                    colorscale="Viridis",
                    showscale=True,
                )
            ]
        )
        fig.update_layout(
            title=f"Predicted Defect Rate Surface ({feature_a} vs {feature_b})",
            scene=dict(
                xaxis_title=feature_a,
                yaxis_title=feature_b,
                zaxis_title="Predicted Defect Rate",
            ),
        )
        fig.write_html(destination)
        return destination


def select_top_features_by_shap(
    explanation: ShapExplanation, top_k: int
) -> List[str]:
    mean_abs = _mean_abs_shap(explanation.shap_values)
    feature_order = np.argsort(mean_abs)[::-1]
    top_indices = feature_order[:top_k]
    return [explanation.feature_frame.columns[i] for i in top_indices]
