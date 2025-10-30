from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ...common.config import ProjectConfig
from ...common.logging import get_logger
from ...domain.repositories import DatasetRepository, ModelPersistence
from ...domain.value_objects import DataSplit
from ...infrastructure.analysis.shap_service import (
    ShapAnalysisArtifacts,
    ShapAnalysisService,
    select_top_features_by_shap,
)


@dataclass
class AnalyzeShapInteractionsUseCase:
    dataset_repository: DatasetRepository
    model_store: ModelPersistence
    config: ProjectConfig
    logger: logging.Logger = get_logger("AnalyzeShapInteractionsUseCase")

    def __post_init__(self) -> None:
        self.service = ShapAnalysisService(
            dataset_repository=self.dataset_repository,
            model_store=self.model_store,
            shap_config=self.config.shap,
            paths_config=self.config.paths,
            logger=self.logger,
            random_state=self.config.training.random_seed,
        )

    def execute(self, model_name: str = "xgboost_regressor") -> ShapAnalysisArtifacts:
        self.config.ensure_directories()
        dataset = self.service.load_dataset_sample(DataSplit.TRAIN)
        model_path = self.config.paths.models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

        model, explanation = self.service.compute_explanation(model_path, dataset)

        summary_plot_path = self.service.generate_summary_plot(
            explanation, self.config.paths.shap_dir / f"{model_name}_summary.png"
        )
        summary_table_path = self.service.export_summary_table(
            explanation, self.config.paths.reports_dir / f"{model_name}_shap_summary.csv"
        )

        top_features = select_top_features_by_shap(
            explanation, min(self.config.shap.top_k_features, explanation.shap_values.shape[1])
        )
        dependence_targets = self._select_dependence_targets(top_features)
        dependence_dir = self.config.paths.shap_dir / "dependence"
        dependence_paths = self.service.generate_dependence_plots(
            explanation, dependence_targets, dependence_dir
        )

        heatmap_path = self.service.generate_interaction_heatmap(
            explanation,
            self.config.paths.shap_dir / f"{model_name}_interaction_heatmap.png",
            top_k=self.config.shap.interaction_top_k,
        )

        lot_barplot_path = self.service.generate_lot_barplots(
            explanation,
            self.config.paths.shap_dir / f"{model_name}_lot_shap.png",
            top_k_features=min(self.config.shap.top_k_features, 10),
            top_k_lots=self.config.shap.lot_level_summary_top_k,
        )

        surface_path = None
        feature_pair = self._select_surface_features(top_features)
        if feature_pair is not None:
            feature_ranges = self.dataset_repository.feature_ranges(feature_pair)
            surface_path = self.service.generate_surface_plot(
                model,
                explanation,
                feature_pair,
                {name: value.bounds for name, value in feature_ranges.items()},
                self.config.paths.shap_dir / f"{model_name}_surface.html",
            )

        return ShapAnalysisArtifacts(
            summary_plot=summary_plot_path,
            summary_table=summary_table_path,
            dependence_plots=dependence_paths,
            interaction_heatmap=heatmap_path,
            lot_barplots=lot_barplot_path,
            surface_plot=surface_path,
        )

    def _select_dependence_targets(self, top_features: List[str]) -> List[str]:
        if self.config.shap.dependence_target_features:
            return [
                feature
                for feature in self.config.shap.dependence_target_features
                if feature in top_features
            ][: self.config.shap.top_k_features]
        return top_features[: min(5, len(top_features))]

    def _select_surface_features(self, top_features: List[str]) -> Optional[Tuple[str, str]]:
        candidate_features = list(top_features)
        temperature_features = [
            feature
            for feature in self.dataset_repository.list_temperature_features()
            if feature in candidate_features
        ]
        if self.config.shap.dependence_target_features and len(
            self.config.shap.dependence_target_features
        ) >= 2:
            candidates = [
                feature
                for feature in self.config.shap.dependence_target_features
                if feature in temperature_features
            ]
            if len(candidates) >= 2:
                return candidates[0], candidates[1]

        if len(temperature_features) >= 2:
            return temperature_features[0], temperature_features[1]
        if len(candidate_features) >= 2:
            return candidate_features[0], candidate_features[1]
        return None

