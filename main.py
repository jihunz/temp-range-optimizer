from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from analyze import analyze
from evaluate import evaluate
from optimize import optimize
from train import train


def run_pipeline(
    *,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    model_name: str = "xgb_baseline",
    optimize_split: str = "validation",
    evaluate_split: str = "validation",
    lot_ids: Optional[Sequence[str]] = None,
    max_lots: Optional[int] = None,
) -> str:
    training_result = train(
        config_path=config_path,
        run_id=run_id,
        model_name=model_name,
    )
    current_run_id = training_result["run_id"]

    analyze(
        config_path=config_path,
        run_id=current_run_id,
        model_name=model_name,
    )

    optimize(
        config_path=config_path,
        run_id=current_run_id,
        model_name=model_name,
        split=optimize_split,
        lot_ids=lot_ids,
        max_lots=max_lots,
    )

    evaluate(
        config_path=config_path,
        run_id=current_run_id,
        model_name=model_name,
        split=evaluate_split,
    )

    return current_run_id


if __name__ == "__main__":
    run_pipeline()

