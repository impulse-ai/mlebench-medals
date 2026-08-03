"""Helpers for standardized EES prediction artifact contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_prediction_artifacts(
    output_dir: Path,
    *,
    test_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    validation_targets: pd.DataFrame,
    id_col: str,
    target_cols: list[str],
    validation_kind: str,
) -> dict[str, str | list[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test_predictions.csv"
    validation_prediction_path = output_dir / "validation_predictions.csv"
    validation_target_path = output_dir / "validation_targets.csv"
    ordered_cols = [id_col, *target_cols]
    test_predictions[ordered_cols].to_csv(test_path, index=False)
    validation_predictions[ordered_cols].to_csv(validation_prediction_path, index=False)
    validation_targets[ordered_cols].to_csv(validation_target_path, index=False)
    return {
        "test_prediction_path": str(test_path),
        "validation_prediction_path": str(validation_prediction_path),
        "validation_target_path": str(validation_target_path),
        "id_col": id_col,
        "target_cols": target_cols,
        "validation_kind": validation_kind,
    }


def one_hot_frame(
    ids,
    labels,
    *,
    id_col: str,
    target_cols: list[str],
    positive_value: float = 1.0,
    negative_value: float = 0.0,
) -> pd.DataFrame:
    label_values = [str(label) for label in labels]
    columns = {id_col: list(ids)}
    for target in target_cols:
        columns[target] = [
            positive_value if str(label) == str(target) else negative_value
            for label in label_values
        ]
    return pd.DataFrame(columns)
