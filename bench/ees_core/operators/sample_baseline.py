"""Sample-shaped fallback operator for unsupported MLE-bench modalities."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult


def run_sample_baseline_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    recipe_id: str,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sample_path = _find_sample_submission(data_dir)
        if sample_path is None:
            raise FileNotFoundError("sample submission not found")
        sample = pd.read_csv(sample_path)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        sample.to_csv(submission_path, index=False)
        metrics = {
            "operator": "sample_baseline",
            "task_id": task_id,
            "recipe_id": recipe_id,
            "sample_submission_path": str(sample_path),
            "rows": int(len(sample)),
            "columns": list(sample.columns),
            "runtime_seconds": round(time.time() - started, 3),
            "note": "No typed solver available; copied sample-shaped baseline to force grading/observability.",
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id="sample_baseline",
            recipe_id=recipe_id,
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(sample)), "columns": list(sample.columns)},
            score=None,
            medal="none",
            final_artifact_source="ees_core:sample_baseline",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            no_score_reason="sample_baseline_no_model",
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="sample_baseline",
            recipe_id=recipe_id,
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:sample_baseline",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
