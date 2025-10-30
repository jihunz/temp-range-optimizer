from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap

from common import (
    DatasetSplit,
    TargetScaler,
    compute_feature_ranges,
    configure_matplotlib,
    ensure_logger,
    ensure_result_directories,
    load_config,
    load_dataset,
    mean_absolute_shap,
    read_latest_run_marker,
)


def _sample_dataset(split: DatasetSplit, max_samples: Optional[int], random_state: int) -> DatasetSplit:
    if max_samples is None or len(split.features) <= max_samples:
        return split
    sampled_features = split.features.sample(n=max_samples, random_state=random_state)
    sampled_target = split.target.loc[sampled_features.index]
    sampled_lots = split.lots.loc[sampled_features.index]
    return DatasetSplit(
        features=sampled_features.reset_index(drop=True),
        target=sampled_target.reset_index(drop=True),
        lots=sampled_lots.reset_index(drop=True),
    )


def _select_top_features(columns: Sequence[str], shap_values: np.ndarray, top_k: int) -> List[str]:
    scores = mean_absolute_shap(shap_values)
    order = np.argsort(scores)[::-1]
    limit = min(top_k, len(columns))
    return [columns[idx] for idx in order[:limit]]


def _save_summary_plot(
    explanation: shap.Explanation,
    shap_values: np.ndarray,
    destination: Path,
    top_k: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, explanation.data, show=False, max_display=top_k)
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=300)
    plt.close()
    return destination


def _save_summary_table(columns: Sequence[str], shap_values: np.ndarray, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mean_abs = mean_absolute_shap(shap_values)
    df = pd.DataFrame({"feature": columns, "mean_abs_shap": mean_abs})
    df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    df.to_csv(destination, index=False)
    return destination


def _save_dependence_plots(
    shap_values: np.ndarray,
    feature_matrix: pd.DataFrame,
    features: Sequence[str],
    destination_dir: Path,
) -> List[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for feature in features:
        shap.dependence_plot(
            feature,
            shap_values,
            feature_matrix,
            show=False,
            feature_names=list(feature_matrix.columns),
        )
        plt.tight_layout()
        path = destination_dir / f"dependence_{feature}.png"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        paths.append(path)
    return paths


def _save_interaction_heatmap(
    feature_names: Sequence[str],
    interaction_values: np.ndarray,
    destination: Path,
    top_k: Optional[int],
) -> Optional[Path]:
    if interaction_values is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)

    strength = np.mean(np.abs(interaction_values), axis=0)
    np.fill_diagonal(strength, 0.0)
    interaction_df = pd.DataFrame(strength, index=feature_names, columns=feature_names)

    if top_k is not None and top_k < len(feature_names):
        totals = interaction_df.abs().sum(axis=0).sort_values(ascending=False)
        selected = totals.head(top_k).index
        interaction_df = interaction_df.loc[selected, selected]

    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_df, cmap="coolwarm", center=0.0)
    plt.title("Mean |SHAP interaction|")
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=300)
    plt.close()
    return destination


def _save_lot_barplots(
    shap_values: np.ndarray,
    columns: Sequence[str],
    lots: Sequence[str],
    destination: Path,
    top_k_features: int,
    top_k_lots: int,
) -> Optional[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shap_df = pd.DataFrame(np.abs(shap_values), columns=columns)
    shap_df["LOT_NO"] = list(lots)
    lot_mean = shap_df.groupby("LOT_NO").mean()
    if lot_mean.empty:
        return None

    lot_scores = lot_mean.sum(axis=1).sort_values(ascending=False)
    selected_lots = lot_scores.head(top_k_lots).index
    if len(selected_lots) == 0:
        return None

    rows = len(selected_lots)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes]
    for ax, lot in zip(axes, selected_lots):
        values = lot_mean.loc[lot].sort_values(ascending=False).head(top_k_features)
        ax.barh(list(values.index[::-1]), list(values.values[::-1]))
        ax.set_title(f"LOT {lot} | Top SHAP contributions")
        ax.set_xlabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=300)
    plt.close()
    return destination


def _save_surface_plot(
    model,
    scaler: TargetScaler,
    columns: Sequence[str],
    feature_pair: Tuple[str, str],
    feature_ranges: Dict[str, Tuple[float, float]],
    baseline_row: pd.Series,
    destination: Path,
) -> Optional[Path]:
    a, b = feature_pair
    if a not in feature_ranges or b not in feature_ranges:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    range_a = feature_ranges[a]
    range_b = feature_ranges[b]
    grid_a = np.linspace(range_a[0], range_a[1], 40)
    grid_b = np.linspace(range_b[0], range_b[1], 40)
    mesh_a, mesh_b = np.meshgrid(grid_a, grid_b)

    repeated = pd.DataFrame(
        np.repeat(baseline_row.to_frame().T.values, mesh_a.size, axis=0),
        columns=columns,
    )
    repeated[a] = mesh_a.ravel()
    repeated[b] = mesh_b.ravel()

    predictions = model.predict(repeated)
    surface = scaler.inverse_values(predictions).reshape(mesh_a.shape)

    figure = go.Figure(
        data=[
            go.Surface(x=grid_a, y=grid_b, z=surface, colorscale="Viridis", showscale=True)
        ]
    )
    figure.update_layout(
        title=f"Predicted defect rate surface ({a} vs {b})",
        scene=dict(xaxis_title=a, yaxis_title=b, zaxis_title="Predicted defect rate"),
    )
    figure.write_html(destination)
    return destination


def analyze(
    *,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name: str = "xgb_baseline",
    split: str = "train",
) -> Dict[str, Optional[str]]:
    """Generate SHAP-based analysis artefacts for a trained model."""

    ensure_logger()
    configure_matplotlib()
    config = load_config(config_path)

    base_result_dir = Path(str(config["paths"]["result_root"]))
    if run_id is None:
        run_id = read_latest_run_marker(base_result_dir, config)
        if run_id is None:
            raise ValueError("No recorded run. Execute training first or provide run_id explicitly.")

    run_paths = ensure_result_directories(base_result_dir, config, run_id)
    model_path = run_paths["models"] / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    dataset = load_dataset(split, config)
    shap_cfg = config["shap"]
    sampled_dataset = _sample_dataset(
        dataset,
        shap_cfg.get("max_samples"),
        random_state=config["training"].get("random_seed", 42),
    )

    scaler = TargetScaler.from_config(config["training"].get("target_scaler", {}))
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    explanation = explainer(sampled_dataset.features)
    shap_values = scaler.inverse_values(explanation.values)
    interaction_values = None
    if shap_cfg.get("interaction_top_k", 0) > 0:
        raw_interactions = explainer.shap_interaction_values(sampled_dataset.features)
        interaction_values = scaler.inverse_values(raw_interactions)

    columns = list(sampled_dataset.features.columns)
    top_features = _select_top_features(
        columns,
        shap_values,
        top_k=shap_cfg.get("top_k_features", 15),
    )

    summary_plot = _save_summary_plot(
        explanation,
        shap_values,
        run_paths["shap"] / f"{model_name}_summary.png",
        top_k=shap_cfg.get("top_k_features", 15),
    )

    summary_table = _save_summary_table(
        columns,
        shap_values,
        run_paths["reports"] / f"{model_name}_shap_summary.csv",
    )

    dependence_features = top_features[: shap_cfg.get("dependence_max_features", 5)]
    dependence_paths = _save_dependence_plots(
        shap_values,
        sampled_dataset.features,
        dependence_features,
        run_paths["shap"] / "dependence",
    )

    heatmap_path = _save_interaction_heatmap(
        columns,
        interaction_values,
        run_paths["shap"] / f"{model_name}_interaction_heatmap.png",
        top_k=shap_cfg.get("interaction_top_k"),
    )

    lot_barplot_path = _save_lot_barplots(
        shap_values,
        columns,
        sampled_dataset.lots,
        run_paths["shap"] / f"{model_name}_lot_shap.png",
        top_k_features=shap_cfg.get("lot_top_features", 10),
        top_k_lots=shap_cfg.get("lot_top_lots", 5),
    )

    surface_path: Optional[Path] = None
    if len(top_features) >= 2:
        feature_ranges = compute_feature_ranges(dataset, top_features[:2])
        baseline_row = dataset.features.median()
        surface_path = _save_surface_plot(
            model,
            scaler,
            columns,
            (top_features[0], top_features[1]),
            feature_ranges,
            baseline_row,
            run_paths["shap"] / f"{model_name}_surface.html",
        )

    logging.info("SHAP analysis completed for run %s", run_id)

    artefacts: Dict[str, Optional[str]] = {
        "run_id": run_id,
        "summary_plot": str(summary_plot),
        "summary_table": str(summary_table),
    }
    if dependence_paths:
        artefacts["dependence_plots"] = json.dumps([str(path) for path in dependence_paths], ensure_ascii=False)
    if heatmap_path:
        artefacts["interaction_heatmap"] = str(heatmap_path)
    if lot_barplot_path:
        artefacts["lot_barplot"] = str(lot_barplot_path)
    if surface_path:
        artefacts["surface_plot"] = str(surface_path)
    return artefacts


if __name__ == "__main__":
    analyze()

