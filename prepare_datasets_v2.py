from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ALL_REQUIRED_COLUMNS = [
    "건조로 온도 1 Zone",
    "건조로 온도 2 Zone",
    "소입로 온도 1 Zone",
    "소입로 온도 2 Zone",
    "소입로 온도 3 Zone",
    "소입로 온도 4 Zone",
    "소입로 CP 값",
    "솔트조 온도 1 Zone",
    "솔트조 온도 2 Zone",
]

OPTIONAL_COLUMNS = [
    "솔트 컨베이어 온도 1 Zone",
    "솔트 컨베이어 온도 2 Zone",
    "세정기",
]


def sanitize_column(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def parse_args() -> argparse.Namespace:
    default_features = Path(
        "/Users/jihunjang/Downloads/경진대회용 제조AI데이터셋/1. 열처리 공정최적화 AI 데이터셋/공식/품질전처리후데이터.csv"
    )
    default_labels = Path(
        "/Users/jihunjang/Downloads/경진대회용 제조AI데이터셋/1. 열처리 공정최적화 AI 데이터셋/공식/label.xlsx"
    )
    parser = argparse.ArgumentParser(
        description="Generate LOT-level dataset for XGBoost/SHAP/Optimization workflow (v2 schema)."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=default_features,
        help="Path to raw temperature CSV (실측 센서 데이터).",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=default_labels,
        help="Path to label Excel (lot 정보 및 수량 데이터).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for dataset splits.",
    )
    return parser.parse_args()


def load_feature_data(path: Path, required_cols: Iterable[str], optional_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {col: sanitize_column(col) for col in df.columns}
    df = df.rename(columns=rename_map)

    required_sanitized = [sanitize_column(col) for col in required_cols]
    optional_sanitized = [sanitize_column(col) for col in optional_cols if sanitize_column(col) in df.columns]

    cols_to_use = ["TAG_MIN", "배정번호"] + required_sanitized + optional_sanitized
    missing = [col for col in cols_to_use if col not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from features CSV: {missing}")

    df = df[cols_to_use]
    df = df.dropna(subset=required_sanitized + ["TAG_MIN", "배정번호"])
    df["timestamp"] = pd.to_datetime(
        df["TAG_MIN"].astype(str).str.replace(".", "-", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["배정번호", "timestamp"])
    return df, required_sanitized, optional_sanitized


def compute_group_statistics(
    df: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str],
) -> pd.DataFrame:
    group = df.groupby("배정번호", observed=True)

    stats_required = group[required_cols].agg(["mean", "std", "max", "min"])
    stats_required.columns = [
        f"{column}_{stat}" for column, stat in stats_required.columns.to_flat_index()
    ]

    for column in required_cols:
        stats_required[f"{column}_std"] = stats_required[f"{column}_std"].fillna(0.0)
        stats_required[f"{column}_range"] = (
            stats_required[f"{column}_max"] - stats_required[f"{column}_min"]
        )

    optional_frames: List[pd.DataFrame] = []
    if optional_cols:
        stats_optional = group[optional_cols].agg(["mean", "std"])
        stats_optional.columns = [
            f"{column}_{stat}" for column, stat in stats_optional.columns.to_flat_index()
        ]
        for column in optional_cols:
            stats_optional[f"{column}_std"] = stats_optional[f"{column}_std"].fillna(0.0)
        optional_frames.append(stats_optional)

    latest_timestamp = group["timestamp"].max().rename("작업일_최신")

    summary = pd.concat([stats_required] + optional_frames + [latest_timestamp], axis=1)
    summary = summary.reset_index()
    summary["작업일_최신"] = summary["작업일_최신"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return summary


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=lambda c: sanitize_column(str(c)))
    df["불량률"] = df["불량수량"] / df["총수량"]
    return df


def merge_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str],
) -> pd.DataFrame:
    if "열처리_염욕_1" in labels.columns:
        label_key = "열처리_염욕_1"
    elif "배정번호" in labels.columns:
        label_key = "배정번호"
    else:
        raise KeyError("Labels must contain either '열처리 염욕_1' or '배정번호' column.")

    merged = features.merge(
        labels,
        left_on="배정번호",
        right_on=label_key,
        how="inner",
    )

    merged["작업일"] = pd.to_datetime(merged["작업일"], errors="coerce")
    merged["작업일_최신"] = pd.to_datetime(merged["작업일_최신"], errors="coerce")

    feature_columns: List[str] = []
    for col in required_cols:
        feature_columns.extend(
            [
                f"{col}_mean",
                f"{col}_std",
                f"{col}_max",
                f"{col}_min",
                f"{col}_range",
            ]
        )
    for col in optional_cols:
        feature_columns.extend(
            [
                f"{col}_mean",
                f"{col}_std",
            ]
        )

    base_numeric = ["양품수량", "불량수량", "총수량"]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        weights = group["총수량"].replace(0, np.nan)
        if weights.isna().all():
            weights = pd.Series(np.ones(len(group)), index=group.index)
        else:
            weights = weights.fillna(weights.median())

        weighted_features = (
            group[feature_columns].multiply(weights, axis=0).sum() / weights.sum()
        )
        totals = group[base_numeric].sum()
        result = pd.concat([weighted_features, totals])
        result["불량률"] = (
            totals["불량수량"] / totals["총수량"] if totals["총수량"] else 0.0
        )
        result["작업일"] = group["작업일"].min()
        result["작업일_최신"] = group["작업일_최신"].max()
        return result

    aggregated = merged.groupby("LOT_NO").apply(aggregate)
    aggregated = aggregated.reset_index()
    aggregated["작업일"] = aggregated["작업일"].dt.strftime("%Y-%m-%d")
    aggregated["작업일_최신"] = aggregated["작업일_최신"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    column_order = (
        ["LOT_NO", "작업일", "작업일_최신", "양품수량", "불량수량", "총수량", "불량률"]
        + feature_columns
    )
    aggregated = aggregated[column_order]
    aggregated = aggregated.rename(
        columns=lambda c: c.replace(" ", "_").replace("Zone", "Zone")
    )
    return aggregated


def create_splits(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    indices = df.index.to_numpy()
    target = df["불량률"]
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
    stratify_train = stratify.iloc[train_val_idx] if stratify is not None else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed,
        stratify=stratify_train if stratify_train is not None else None,
    )
    return {
        "train": df.loc[train_idx].reset_index(drop=True),
        "val": df.loc[val_idx].reset_index(drop=True),
        "test": df.loc[test_idx].reset_index(drop=True),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_df, required_cols, optional_cols = load_feature_data(
        args.features_path,
        ALL_REQUIRED_COLUMNS,
        OPTIONAL_COLUMNS,
    )
    assignment_features = compute_group_statistics(feature_df, required_cols, optional_cols)
    label_df = load_labels(args.label_path)
    dataset = merge_features_labels(assignment_features, label_df, required_cols, optional_cols)

    splits = create_splits(dataset, args.random_seed)
    for split_name, frame in splits.items():
        output_path = args.output_dir / f"lot_dataset_v2_{split_name}.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {split_name} split to {output_path} shape={frame.shape}")


if __name__ == "__main__":
    main()
