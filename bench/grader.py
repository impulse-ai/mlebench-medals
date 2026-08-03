"""Wrap MLE-bench's official grader (`mlebench grade-sample`).

The grader compares a submission against the held-out Kaggle private test
labels and returns Kaggle medal status (gold/silver/bronze/none) using the
actual leaderboard thresholds for that competition.

Reference: https://github.com/openai/mle-bench
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GradeResult:
    task_id: str
    success: bool
    score: float | None = None
    metric: str | None = None
    any_medal: str = ""  # "", "bronze", "silver", "gold"
    gold_threshold: float | None = None
    silver_threshold: float | None = None
    bronze_threshold: float | None = None
    raw: dict = field(default_factory=dict)
    error: str | None = None

    def medal(self) -> str:
        return self.any_medal if self.any_medal else "none"


def _extract_json_block(text: str) -> dict | None:
    """mlebench's grade-sample prints log-prefixed lines like
    `[2026-05-20 ...] [cli.py:203] {` then a JSON object across many lines.
    Find and parse that JSON block."""
    # Find the first '{' and walk to the matching '}'.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _resolve_medal(data: dict) -> str:
    """Map mlebench's boolean medal fields to a single string."""
    if data.get("gold_medal"):
        return "gold"
    if data.get("silver_medal"):
        return "silver"
    if data.get("bronze_medal"):
        return "bronze"
    return ""  # GradeResult.medal() turns this into "none"


def _coerced_api_regrade(
    task_id: str, submission_path: Path, data_dir: Path | str | None
) -> GradeResult | None:
    """Re-grade through the mlebench python API with str-coerced frames.

    mlebench's ``Grader.__call__`` swallows internal exceptions and returns
    ``score=None`` — on text-normalization its grade_fn dies comparing str to
    float once pandas type inference splits a column's dtype between the
    submission and the answers, so a perfectly valid submission silently
    grades as no-score. Reading BOTH frames with ``dtype=str`` (and
    ``keep_default_na=False`` so literal "nan"/"" tokens survive) gives the
    grade_fn version-independent, homogeneous inputs. Graders that genuinely
    need numeric values raise/return None on the coerced frames too, and we
    keep the original result — this fallback can only recover grades, never
    change a successful one.
    """
    try:
        import pandas as pd
        from mlebench.data import get_leaderboard
        from mlebench.registry import registry

        reg = registry.set_data_dir(Path(data_dir)) if data_dir else registry
        competition = reg.get_competition(task_id)
        submission = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
        answers = pd.read_csv(competition.answers, dtype=str, keep_default_na=False)
        score = competition.grader(submission, answers)
        if score is None:
            return None
        leaderboard = get_leaderboard(competition)
        rank = competition.grader.rank_score(score, leaderboard)
    except Exception:
        return None

    raw = dict(rank)
    raw.update(
        {
            "competition_id": task_id,
            "score": score,
            "submission_exists": True,
            "valid_submission": True,
            "coerced": True,
        }
    )
    return GradeResult(
        task_id=task_id,
        success=True,
        score=score,
        any_medal=_resolve_medal(rank),
        gold_threshold=rank.get("gold_threshold"),
        silver_threshold=rank.get("silver_threshold"),
        bronze_threshold=rank.get("bronze_threshold"),
        raw=raw,
    )


def _api_grade(task_id: str, submission_path: Path, data_dir: Path | str | None) -> GradeResult | None:
    """Grade directly through the mlebench python API (`grade_csv`).

    A fallback for when the `mlebench` CLI is not on PATH — common in
    autonomous/sandbox run environments where mlebench is importable but its
    console entry point isn't installed. The controller's in-run grade shells
    out to the CLI, so a valid submission would otherwise be reported as
    GRADE-FAIL purely because the CLI binary is absent. This yields the same
    authoritative grade (thresholds + medal) as the CLI.
    """
    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry

        reg = registry.set_data_dir(Path(data_dir)) if data_dir else registry
        competition = reg.get_competition(task_id)
        rep = grade_csv(Path(submission_path), competition)
    except Exception:
        return None

    raw = rep.to_dict() if hasattr(rep, "to_dict") else dict(getattr(rep, "__dict__", {}))
    raw.setdefault("competition_id", task_id)
    raw["graded_via"] = "python_api"
    return GradeResult(
        task_id=task_id,
        success=True,
        score=getattr(rep, "score", None),
        any_medal=_resolve_medal(raw),
        gold_threshold=getattr(rep, "gold_threshold", None),
        silver_threshold=getattr(rep, "silver_threshold", None),
        bronze_threshold=getattr(rep, "bronze_threshold", None),
        raw=raw,
    )


def grade(task_id: str, submission_path: Path, data_dir: Path | None = None) -> GradeResult:
    """Invoke `mlebench grade-sample` and parse its output.

    The CLI signature is: `mlebench grade-sample <submission> <competition_id>`.
    Output is human-readable with log prefixes; the score+thresholds JSON is
    embedded inside the log stream. We extract and parse the JSON block.
    Falls back to the mlebench python API when the CLI is not on PATH.
    """
    cmd = ["mlebench", "grade-sample", str(submission_path), task_id]
    resolved_data_dir = data_dir or os.environ.get("BENCH_GRADER_DATA_DIR")
    if resolved_data_dir:
        cmd.extend(["--data-dir", str(resolved_data_dir)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        api = _api_grade(task_id, submission_path, resolved_data_dir)
        if api is not None:
            return api
        return GradeResult(
            task_id=task_id,
            success=False,
            error="mlebench CLI not found and python-API grade failed. Install: "
                  "`pip install mlebench` or clone https://github.com/openai/mle-bench.",
        )

    # mlebench writes its report to stderr (via Python logging) on some setups,
    # to stdout on others. Probe both.
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    data = _extract_json_block(combined)

    if data is None:
        return GradeResult(
            task_id=task_id,
            success=False,
            raw={"stdout": proc.stdout, "stderr": proc.stderr},
            error=(
                f"could not parse grader output (exit {proc.returncode}). "
                f"head: {combined[:400]!r}"
            ),
        )

    if proc.returncode != 0 and not data.get("valid_submission", True):
        return GradeResult(
            task_id=task_id,
            success=False,
            raw=data,
            error=f"submission rejected by grader: {data}",
        )

    if data.get("score") is None and data.get("submission_exists"):
        coerced = _coerced_api_regrade(task_id, submission_path, resolved_data_dir)
        if coerced is not None:
            return coerced

    return GradeResult(
        task_id=task_id,
        success=True,
        score=data.get("score"),
        metric=data.get("metric") or data.get("score_metric"),
        any_medal=_resolve_medal(data),
        gold_threshold=data.get("gold_threshold"),
        silver_threshold=data.get("silver_threshold"),
        bronze_threshold=data.get("bronze_threshold"),
        raw=data,
    )
