"""Reshape Impulse's prediction CSV to match the per-task sample_submission schema.

MLE-bench's grader expects a submission CSV with task-specific column names
and the rows aligned to the test set (same ID set, same order in many cases).
Impulse's prediction output may use generic column names like 'prediction' or
'pred' or the user-specified target column — we coerce to the canonical schema
documented by `sample_submission.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .task_detection import TaskSchema


# Common column-name candidates Impulse may emit for a target prediction.
_PRED_NAME_CANDIDATES = [
    "{target}",
    "prediction",
    "predictions",
    "pred",
    "preds",
    "y_pred",
    "pred_{target}",
    "{target}_pred",
]


def _find_prediction_column(preds_df: pd.DataFrame, target: str) -> str:
    """Find the column in `preds_df` that maps to the expected `target`."""
    for tmpl in _PRED_NAME_CANDIDATES:
        candidate = tmpl.format(target=target)
        if candidate in preds_df.columns:
            return candidate
    # Last-resort: if there's a single non-id column, use that.
    non_id = [c for c in preds_df.columns if c.lower() not in {"id", "index"}]
    if len(non_id) == 1:
        return non_id[0]
    raise ValueError(
        f"could not find a column in predictions to map to target '{target}'. "
        f"available columns: {list(preds_df.columns)}"
    )


def format_submission(
    predictions_csv: Path,
    sample_submission_csv: Path,
    out_path: Path,
    schema: TaskSchema,
) -> Path:
    """Coerce `predictions_csv` into the schema expected by `sample_submission_csv`.

    Strategy:
      1. If predictions contain the ID column, merge on it (preserves sample order).
      2. Otherwise, fall back to row-order alignment (errors if lengths differ).
      3. Map each expected target column to a predictions column via name candidates.

    Returns the path of the written submission.
    """
    sample = pd.read_csv(sample_submission_csv)
    preds = pd.read_csv(predictions_csv)

    if schema.id_col in preds.columns:
        out = sample[[schema.id_col]].merge(preds, on=schema.id_col, how="left")
        if out[schema.id_col].isna().any():
            raise ValueError(
                f"some sample IDs were not present in predictions ("
                f"{int(out[schema.id_col].isna().sum())} missing)."
            )
    else:
        if len(preds) != len(sample):
            raise ValueError(
                f"cannot align predictions: {len(preds)} preds vs "
                f"{len(sample)} expected, and no '{schema.id_col}' column to merge on."
            )
        out = sample.copy()
        # Copy preds rows into out; rely on positional alignment.
        for col in preds.columns:
            if col not in out.columns:
                out[col] = preds[col].values

    # Map each expected target column to a predictions column.
    for target_col in schema.target_cols:
        if target_col in out.columns and not out[target_col].isna().all():
            continue  # already aligned by the merge above
        pred_col = _find_prediction_column(preds, target_col)
        if schema.id_col in preds.columns:
            # Reindex by ID
            src = preds.set_index(schema.id_col)[pred_col]
            out[target_col] = src.reindex(out[schema.id_col]).values
        else:
            out[target_col] = preds[pred_col].values

    out = out[schema.sample_submission_cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
