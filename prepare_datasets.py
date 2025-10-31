from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sanitize_column(name: str) -> str:
    sanitized = (
        name.strip()
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return sanitized


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test datasets from raw furnace data."
    )
    default_features = Path(
        "/Users/jihunjang/Downloads/경진대회용 제조AI데이터셋/1. 열처리 공정최적화 AI 데이터셋/공식/품질전처리후데이터.csv"
    )
    default_label = Path(
        "/Users/jihunjang/Downloads/경진대회용 제조AI데이터셋/1. 열처리 공정최적화 AI 데이터셋/공식/label.xlsx"
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=default_features,
        help="Path to raw temperature CSV (e.g., 품질전처리후데이터.csv).",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=default_label,
        help="Path to label Excel file (e.g., label.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where train/val/test CSVs will be saved.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )
    return parser.parse_args()


def load_feature_dataframe(path: Path, temperature_columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {col: sanitize_column(col) for col in df.columns}
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=[sanitize_column(col) for col in temperature_columns])
    df["timestamp"] = pd.to_datetime(
        df["TAG_MIN"].astype(str).str.replace(".", "-", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["배정번호", "timestamp"])
    return df


def build_assignment_features(
    df: pd.DataFrame,
    temperature_columns: List[str],
) -> pd.DataFrame:
    group = df.groupby("배정번호", observed=True)

    agg_df = group[temperature_columns].agg(["mean", "std", "min", "max"])
    agg_df.columns = [
        f"{column}_{statistic}"
        for column, statistic in agg_df.columns.to_flat_index()
    ]

    for column in temperature_columns:
        agg_df[f"{column}_range"] = agg_df[f"{column}_max"] - agg_df[f"{column}_min"]
        cv = agg_df[f"{column}_std"] / agg_df[f"{column}_mean"]
        agg_df[f"{column}_cv"] = (
            cv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        agg_df[f"{column}_std"] = agg_df[f"{column}_std"].fillna(0.0)

    duration_minutes = group["timestamp"].agg(
        lambda s: (s.max() - s.min()).total_seconds() / 60 if len(s) > 1 else 0.0
    )
    record_count = group.size()
    avg_interval_seconds = group["timestamp"].agg(
        lambda s: s.diff().dt.total_seconds().dropna().mean() if len(s) > 1 else 0.0
    )

    agg_df["duration_minutes"] = duration_minutes
    agg_df["record_count"] = record_count
    agg_df["avg_interval_seconds"] = avg_interval_seconds.fillna(0.0)

    # Threshold features: proportion of readings above 850°C.
    high_temp_columns = [
        col
        for col in temperature_columns
        if "소입로_온도" in col or "소입" in col
    ]
    for column in high_temp_columns:
        ratio = group[column].apply(lambda s: float((s > 850.0).mean()))
        agg_df[f"{column}_gt850_ratio"] = ratio

    # Uniformity features.
    if {"건조_1존_OP", "건조_2존_OP"}.issubset(temperature_columns):
        agg_df["건조_OP_mean_diff"] = (
            agg_df["건조_1존_OP_mean"] - agg_df["건조_2존_OP_mean"]
        )
    if {"소입1존_OP", "소입4존_OP"}.issubset(temperature_columns):
        agg_df["소입_OP_front_back_diff"] = (
            agg_df["소입1존_OP_mean"] - agg_df["소입4존_OP_mean"]
        )
    if {"소입2존_OP", "소입3존_OP"}.issubset(temperature_columns):
        agg_df["소입_OP_inner_diff"] = (
            agg_df["소입2존_OP_mean"] - agg_df["소입3존_OP_mean"]
        )
    if {"소입로_온도_1_Zone", "소입로_온도_4_Zone"}.issubset(temperature_columns):
        agg_df["소입로_온도_1_4_mean_diff"] = (
            agg_df["소입로_온도_1_Zone_mean"]
            - agg_df["소입로_온도_4_Zone_mean"]
        )
        gradient = (
            agg_df["소입로_온도_4_Zone_mean"]
            - agg_df["소입로_온도_1_Zone_mean"]
        ) / 3.0
        agg_df["소입로_온도_gradient_mean"] = gradient
    if {"솔트조_온도_1_Zone", "솔트조_온도_2_Zone"}.issubset(temperature_columns):
        agg_df["솔트조_온도_mean_diff"] = (
            agg_df["솔트조_온도_1_Zone_mean"]
            - agg_df["솔트조_온도_2_Zone_mean"]
        )

    agg_df = agg_df.reset_index()
    return agg_df


def aggregate_to_lot_level(
    assignment_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    label_df = labels.copy()
    label_df = label_df.rename(columns=lambda c: sanitize_column(str(c)))
    label_df["불량률"] = label_df["불량수량"] / label_df["총수량"]

    merged = assignment_features.merge(
        label_df,
        left_on="배정번호",
        right_on="열처리_염욕_1",
        how="inner",
    )

    feature_columns = [
        column
        for column in assignment_features.columns
        if column not in {"배정번호"}
    ]

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        weights = group["총수량"].replace(0, np.nan)
        weights = weights.fillna(weights.mean() if not weights.empty else 1.0)
        weighted_features = (
            group[feature_columns]
            .multiply(weights, axis=0)
            .sum()
            / weights.sum()
        )

        sums = group[["양품수량", "불량수량", "총수량"]].sum()
        result = pd.concat([weighted_features, sums])
        result["불량률"] = sums["불량수량"] / sums["총수량"] if sums["총수량"] else 0.0
        return result

    lot_level = merged.groupby("LOT_NO").apply(_aggregate)
    lot_level = lot_level.reset_index()
    return lot_level


def create_splits(
    dataset: pd.DataFrame,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    indices = dataset.index.to_numpy()
    target = dataset["불량률"]

    stratify = None
    if target.nunique() >= 4:
        try:
            stratify = pd.qcut(target, q=4, duplicates="drop")
        except ValueError:
            stratify = None

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=stratify if stratify is not None else None,
    )

    stratify_train = (
        stratify.iloc[train_val_idx] if stratify is not None else None
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed,
        stratify=stratify_train if stratify_train is not None else None,
    )

    splits = {
        "train": dataset.loc[train_idx].reset_index(drop=True),
        "val": dataset.loc[val_idx].reset_index(drop=True),
        "test": dataset.loc[test_idx].reset_index(drop=True),
    }
    return splits


def main() -> None:
    args = parse_arguments()

    temperature_columns_original = [
        "건조 1존 OP",
        "건조 2존 OP",
        "건조로 온도 1 Zone",
        "건조로 온도 2 Zone",
        "세정기",
        "소입1존 OP",
        "소입2존 OP",
        "소입3존 OP",
        "소입4존 OP",
        "소입로 CP 값",
        "소입로 온도 1 Zone",
        "소입로 온도 2 Zone",
        "소입로 온도 3 Zone",
        "소입로 온도 4 Zone",
        "솔트 컨베이어 온도 1 Zone",
        "솔트 컨베이어 온도 2 Zone",
        "솔트조 온도 1 Zone",
        "솔트조 온도 2 Zone",
    ]
    temperature_columns = [sanitize_column(col) for col in temperature_columns_original]

    feature_df = load_feature_dataframe(args.features_path, temperature_columns_original)
    assignment_features = build_assignment_features(feature_df, temperature_columns)

    label_df = pd.read_excel(args.label_path)
    lot_dataset = aggregate_to_lot_level(assignment_features, label_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = create_splits(lot_dataset, args.random_seed)

    for split_name, frame in splits.items():
        output_path = args.output_dir / f"lot_dataset_{split_name}.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {split_name} split to {output_path} with shape {frame.shape}")


if __name__ == "__main__":
    main()
