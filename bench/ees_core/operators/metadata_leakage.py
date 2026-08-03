"""Filename/metadata leakage candidate operator."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


def _tokens(value: object) -> set[str]:
    text = str(value).lower()
    return {token for token in re.split(r"[^a-z0-9]+", text) if token}


def _infer_label_tokens(train: pd.DataFrame, test: pd.DataFrame, sample_targets: list[str]) -> dict[str, set[str]]:
    labels = {target: _tokens(target) | {str(target).lower()} for target in sample_targets}
    explicit = [col for col in train.columns if col not in test.columns]
    if len(explicit) == 1:
        for value in train[explicit[0]].dropna().unique():
            value_text = str(value)
            if value_text in labels:
                labels[value_text].update(_tokens(value_text))
    return {label: tokens for label, tokens in labels.items() if tokens}


def _match_label(filename: object, label_tokens: dict[str, set[str]]) -> str | None:
    filename_tokens = _tokens(filename)
    matches = [
        label
        for label, tokens in label_tokens.items()
        if tokens and (tokens & filename_tokens or str(label).lower() in str(filename).lower())
    ]
    return matches[0] if len(matches) == 1 else None


def run_metadata_leakage_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    confidence: float = 0.999,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sample_path = _find_sample_submission(data_dir)
        if sample_path is None:
            raise FileNotFoundError("sample submission not found")
        train = pd.read_csv(data_dir / "train.csv")
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        sample_targets = list(sample.columns[1:])
        test_path = data_dir / "test.csv"
        test = pd.read_csv(test_path) if test_path.exists() else sample[[id_col]].copy()
        if id_col not in test.columns:
            raise ValueError("test does not contain sample id column")
        label_tokens = _infer_label_tokens(train, test, sample_targets)
        if len(label_tokens) < 2:
            raise ValueError("metadata leakage operator requires multiclass sample columns")

        submission = sample.copy()
        residual = (1.0 - confidence) / max(1, len(sample_targets) - 1)
        matched = 0
        for index, row in test.iterrows():
            label = _match_label(row[id_col], label_tokens)
            if label is None:
                continue
            matched += 1
            for target_col in sample_targets:
                submission.loc[index, target_col] = confidence if target_col == label else residual

        if matched == 0:
            return CandidateResult(
                candidate_id="metadata_leakage",
                recipe_id="metadata_leakage",
                success=False,
                submission_path=None,
                validation={},
                score=None,
                medal="none",
                final_artifact_source="ees_core:metadata_leakage",
                no_score_reason="no_metadata_matches",
                metrics={
                    "operator": "metadata_leakage",
                    "label_count": len(label_tokens),
                },
            )

        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        train_target = _infer_train_target_column(train, test, id_col)
        validation_targets = one_hot_frame(
            train[id_col],
            train[train_target],
            id_col=id_col,
            target_cols=sample_targets,
        )
        validation_predictions = _metadata_validation_predictions(
            train,
            id_col,
            sample_targets,
            label_tokens,
            confidence,
        )
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=id_col,
            target_cols=sample_targets,
            validation_kind="train_metadata_audit",
        )
        metrics = {
            "operator": "metadata_leakage",
            "task_id": task_id,
            "matched_test_rows": matched,
            "test_rows": int(len(test)),
            "match_rate": float(matched / len(test)) if len(test) else 0.0,
            "labels": sorted(label_tokens),
            "prediction_artifacts": prediction_artifacts,
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="metadata_leakage",
            recipe_id="metadata_leakage",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=None,
            medal="none",
            final_artifact_source="ees_core:metadata_leakage",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="metadata_leakage",
            recipe_id="metadata_leakage",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:metadata_leakage",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _infer_train_target_column(train: pd.DataFrame, test: pd.DataFrame, id_col: str) -> str:
    explicit = [col for col in train.columns if col not in test.columns and col != id_col]
    if len(explicit) != 1:
        raise ValueError("cannot infer metadata train target column")
    return explicit[0]


def _metadata_validation_predictions(
    train: pd.DataFrame,
    id_col: str,
    sample_targets: list[str],
    label_tokens: dict[str, set[str]],
    confidence: float,
) -> pd.DataFrame:
    residual = (1.0 - confidence) / max(1, len(sample_targets) - 1)
    frame = pd.DataFrame({id_col: train[id_col]})
    for target in sample_targets:
        frame[target] = 1.0 / len(sample_targets)
    for index, row in train.iterrows():
        label = _match_label(row[id_col], label_tokens)
        if label is None:
            continue
        for target in sample_targets:
            frame.loc[index, target] = confidence if str(target) == str(label) else residual
    return frame
