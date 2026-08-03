"""Exact train/test duplicate leakage candidate operator."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.operators.text_tfidf import _is_text_dtype
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


def _normalize_value(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _detect_match_column(train: pd.DataFrame, test: pd.DataFrame, id_col: str, target_cols: list[str]) -> str:
    excluded = {id_col, *target_cols}
    candidates = [
        col
        for col in train.columns
        if col in test.columns and col not in excluded and (_is_text_dtype(train[col]) or _is_text_dtype(test[col]))
    ]
    if not candidates:
        raise ValueError("no shared text/string column found for duplicate leakage")
    return max(candidates, key=lambda col: float(train[col].fillna("").astype(str).str.len().mean()))


def _detect_train_target(train: pd.DataFrame, test: pd.DataFrame, sample_targets: list[str]) -> tuple[str, str]:
    numeric_sample_targets = [
        col for col in sample_targets if col in train.columns and pd.api.types.is_numeric_dtype(train[col])
    ]
    if numeric_sample_targets:
        return "direct_numeric", numeric_sample_targets[0]
    explicit = [col for col in train.columns if col not in test.columns]
    if len(explicit) == 1:
        return "categorical", explicit[0]
    numeric_explicit = [col for col in explicit if pd.api.types.is_numeric_dtype(train[col])]
    if numeric_explicit:
        return "direct_numeric", numeric_explicit[0]
    raise ValueError("could not infer duplicate leakage target column")


def _fallback_submission(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.copy()


def run_duplicate_leakage_operator(
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
        test = pd.read_csv(data_dir / "test.csv")
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        sample_targets = list(sample.columns[1:])
        match_col = _detect_match_column(train, test, id_col, sample_targets)
        target_mode, train_target = _detect_train_target(train, test, sample_targets)

        normalized_to_targets: dict[str, list[object]] = {}
        for _, row in train.iterrows():
            key = _normalize_value(row[match_col])
            if not key:
                continue
            normalized_to_targets.setdefault(key, []).append(row[train_target])

        unique_targets: dict[str, object] = {}
        ambiguous = 0
        for key, values in normalized_to_targets.items():
            normalized_values = {str(value) for value in values}
            if len(normalized_values) == 1:
                unique_targets[key] = values[0]
            else:
                ambiguous += 1

        submission = _fallback_submission(sample)
        residual = (1.0 - confidence) / max(1, len(sample_targets) - 1)
        matched = 0
        for index, row in test.iterrows():
            key = _normalize_value(row[match_col])
            if key not in unique_targets:
                continue
            matched += 1
            target = unique_targets[key]
            if target_mode == "direct_numeric":
                for target_col in sample_targets:
                    if target_col == train_target:
                        submission.loc[index, target_col] = float(target)
                if train_target not in sample_targets and len(sample_targets) == 1:
                    submission.loc[index, sample_targets[0]] = float(target)
            else:
                for target_col in sample_targets:
                    submission.loc[index, target_col] = confidence if str(target_col) == str(target) else residual

        if matched == 0:
            return CandidateResult(
                candidate_id="duplicate_leakage",
                recipe_id="duplicate_leakage",
                success=False,
                submission_path=None,
                validation={},
                score=None,
                medal="none",
                final_artifact_source="ees_core:duplicate_leakage",
                no_score_reason="no_duplicate_matches",
                metrics={
                    "operator": "duplicate_leakage",
                    "match_col": match_col,
                    "ambiguous_duplicate_keys": ambiguous,
                },
            )

        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        validation_targets = _duplicate_validation_targets(train, id_col, sample_targets, target_mode, train_target)
        validation_predictions = validation_targets.copy()
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=id_col,
            target_cols=sample_targets,
            validation_kind="train_duplicate_audit",
        )
        metrics = {
            "operator": "duplicate_leakage",
            "task_id": task_id,
            "match_col": match_col,
            "target_mode": target_mode,
            "train_target": train_target,
            "matched_test_rows": matched,
            "test_rows": int(len(test)),
            "match_rate": float(matched / len(test)) if len(test) else 0.0,
            "ambiguous_duplicate_keys": ambiguous,
            "prediction_artifacts": prediction_artifacts,
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="duplicate_leakage",
            recipe_id="duplicate_leakage",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=None,
            medal="none",
            final_artifact_source="ees_core:duplicate_leakage",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="duplicate_leakage",
            recipe_id="duplicate_leakage",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:duplicate_leakage",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _duplicate_validation_targets(
    train: pd.DataFrame,
    id_col: str,
    sample_targets: list[str],
    target_mode: str,
    train_target: str,
) -> pd.DataFrame:
    if target_mode == "categorical":
        return one_hot_frame(
            train[id_col],
            train[train_target],
            id_col=id_col,
            target_cols=sample_targets,
        )
    frame = pd.DataFrame({id_col: train[id_col]})
    for target in sample_targets:
        if target == train_target:
            frame[target] = train[train_target].astype(float).to_numpy()
        elif len(sample_targets) == 1:
            frame[target] = train[train_target].astype(float).to_numpy()
        else:
            frame[target] = 0.0
    return frame
