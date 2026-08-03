"""Reusable measurement logic for per-tag bench-run statistics + gates.

Both `bench/scripts/measure.py` (CLI) and `bench-service/app/routes.py`
(HTTP) import from here. Keep this module dependency-free (stdlib only) so
the CLI works without the bench-service venv activated.

The data flow:
  load_runs(runs_dir)          → list[RunRecord]
  stats_for_tag(runs, tag)     → dict[task_id, TaskStat]
  compute_diff(before, after)  → DiffResult           ← JSON-serializable

The DiffResult shape is intentionally JSON-friendly: the frontend
renders it as a table, the CLI renders it as a markdown table, the same
fields drive both.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---- metric direction ------------------------------------------------------
#
# Names below are lowercased + dash-separated as mlebench writes them.
# Unknown metrics default to higher-is-better (classification metrics
# dominate MLE-bench) but a warning is printed to stderr.

_HIGHER_IS_BETTER = {
    "accuracy", "balanced-accuracy", "f1", "f1-macro", "f1-micro",
    "f1-weighted", "f1-samples", "precision", "recall", "auc", "auc-roc",
    "roc-auc", "auc-pr", "average-precision", "ap", "map",
    "mean-average-precision", "iou", "miou", "score", "top1-accuracy",
    "top5-accuracy", "spearman", "spearman-correlation", "pearson",
    "kappa", "cohen-kappa", "quadratic-weighted-kappa", "qwk", "ndcg",
}

_LOWER_IS_BETTER_HINTS = (
    "rmse", "rmsle", "mae", "mape", "smape", "log-loss", "logloss", "mse",
    "huber", "wmae", "rmspe", "neg-log", "cross-entropy", "crossentropy",
    "deviance", "brier",
)

_seen_unknown_metrics: set[str] = set()


def higher_is_better(metric: str | None, *, warn: bool = True) -> bool:
    """Resolve metric direction. Defaults to higher=better on unknown names."""
    if not metric:
        return True
    m = metric.lower().strip()
    if m in _HIGHER_IS_BETTER:
        return True
    if any(h in m for h in _LOWER_IS_BETTER_HINTS):
        return False
    if warn and m not in _seen_unknown_metrics:
        _seen_unknown_metrics.add(m)
        print(
            f"warning: unknown metric direction for {metric!r}; assuming "
            f"higher=better. Add it to bench/stats.py if wrong.",
            file=sys.stderr,
        )
    return True


# ---- core types ------------------------------------------------------------


@dataclass
class RunRecord:
    """A single bench-service run, parsed from <runs-dir>/<id>/state.json."""
    run_id: str
    task_id: str
    tag: str | None
    status: str
    score: float | None
    metric: str | None


@dataclass
class TaskStat:
    """Aggregate of all runs for a (tag, task_id) pair."""
    task_id: str
    metric: str | None
    higher_better: bool
    n_total: int
    n_succeeded: int
    n_failed: int
    mean: float | None
    std: float | None
    scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class DiffRow:
    task_id: str
    metric: str | None
    higher_better: bool
    before: TaskStat | None
    after: TaskStat | None
    raw_delta: float | None
    improved: bool
    regression_amount: float
    per_task_gate: str  # "PASS" | "FAIL" | "MISSING"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "metric": self.metric,
            "higher_better": self.higher_better,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "raw_delta": self.raw_delta,
            "improved": self.improved,
            "regression_amount": self.regression_amount,
            "per_task_gate": self.per_task_gate,
        }


@dataclass
class DiffResult:
    before_tag: str
    after_tag: str
    rows: list[DiffRow]
    n_compared: int
    n_improving: int
    worst_regression: float
    min_improving_required: int
    max_regression_allowed: float
    aggregate_gate: str  # "PASS" | "FAIL"

    def to_dict(self) -> dict:
        return {
            "before_tag": self.before_tag,
            "after_tag": self.after_tag,
            "rows": [r.to_dict() for r in self.rows],
            "n_compared": self.n_compared,
            "n_improving": self.n_improving,
            "worst_regression": self.worst_regression,
            "min_improving_required": self.min_improving_required,
            "max_regression_allowed": self.max_regression_allowed,
            "aggregate_gate": self.aggregate_gate,
        }


# ---- I/O -----------------------------------------------------------------


def load_runs(runs_dir: Path) -> list[RunRecord]:
    """Read every <runs_dir>/*/state.json into a RunRecord. Missing or
    malformed files are silently skipped — the caller is the operator and
    will notice if records are unexpectedly absent."""
    if not runs_dir.exists():
        return []
    out: list[RunRecord] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "state.json"
        if not sf.exists():
            continue
        try:
            data = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        score = data.get("score")
        out.append(RunRecord(
            run_id=str(data.get("run_id") or d.name),
            task_id=str(data.get("task_id") or ""),
            tag=data.get("tag"),
            status=str(data.get("status") or ""),
            score=float(score) if isinstance(score, (int, float)) else None,
            metric=data.get("metric"),
        ))
    return out


# ---- aggregation + gates -------------------------------------------------


def stats_for_tag(
    runs: Iterable[RunRecord],
    tag: str,
    task_set: set[str] | None = None,
) -> dict[str, TaskStat]:
    """Per-task aggregation for runs carrying a specific `tag`."""
    by_task: dict[str, list[RunRecord]] = {}
    for r in runs:
        if r.tag != tag:
            continue
        if task_set is not None and r.task_id not in task_set:
            continue
        by_task.setdefault(r.task_id, []).append(r)

    out: dict[str, TaskStat] = {}
    for tid, rs in by_task.items():
        succeeded = [r for r in rs if r.status == "succeeded" and r.score is not None]
        scores = [r.score for r in succeeded if r.score is not None]
        # Prefer metric from a succeeded run; fall back to whatever's present.
        metric = next((r.metric for r in succeeded if r.metric), None)
        if not metric:
            metric = next((r.metric for r in rs if r.metric), None)
        out[tid] = TaskStat(
            task_id=tid,
            metric=metric,
            higher_better=higher_is_better(metric),
            n_total=len(rs),
            n_succeeded=len(succeeded),
            n_failed=len(rs) - len(succeeded),
            mean=statistics.mean(scores) if scores else None,
            std=(statistics.stdev(scores) if len(scores) >= 2
                 else (0.0 if scores else None)),
            scores=list(scores),
        )
    return out


def all_tags_summary(runs: Iterable[RunRecord]) -> list[dict[str, Any]]:
    """Per-tag overview for the UI: tag, total runs, succeeded, distinct tasks."""
    by_tag: dict[str | None, list[RunRecord]] = {}
    for r in runs:
        by_tag.setdefault(r.tag, []).append(r)
    out: list[dict[str, Any]] = []
    for tag, rs in by_tag.items():
        n_total = len(rs)
        n_succeeded = sum(1 for r in rs if r.status == "succeeded")
        n_failed = sum(1 for r in rs if r.status == "failed")
        task_ids = sorted({r.task_id for r in rs})
        out.append({
            "tag": tag,
            "n_total": n_total,
            "n_succeeded": n_succeeded,
            "n_failed": n_failed,
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
        })
    # Sort: tagged runs first (alpha), untagged last
    out.sort(key=lambda x: (x["tag"] is None, x["tag"] or ""))
    return out


def _gate_per_task_row(b: TaskStat | None, a: TaskStat | None,
                       max_regression: float) -> tuple[bool, float | None, float, str]:
    """Returns (improved, raw_delta_or_None, regression_amount, gate_label)."""
    if b is None or a is None:
        return (False, None, 0.0, "MISSING")
    if b.mean is None or a.mean is None:
        return (False, None, 0.0, "MISSING")
    raw_delta = a.mean - b.mean
    if a.higher_better:
        improved = raw_delta > 0
        regression = 0.0 if improved else -raw_delta
    else:
        improved = raw_delta < 0
        regression = 0.0 if improved else raw_delta
    gate = "PASS" if (improved or regression <= max_regression) else "FAIL"
    return (improved, raw_delta, regression, gate)


def compute_diff(
    before: dict[str, TaskStat],
    after: dict[str, TaskStat],
    task_set: set[str] | None,
    *,
    before_tag: str,
    after_tag: str,
    min_improving_required: int = 3,
    max_regression_allowed: float = 0.01,
) -> DiffResult:
    """Compute a per-task before/after diff + aggregate gate verdict."""
    effective_tasks = sorted(task_set) if task_set else sorted(set(before) | set(after))
    rows: list[DiffRow] = []
    n_improving = 0
    n_compared = 0
    worst_regression = 0.0
    for tid in effective_tasks:
        b = before.get(tid)
        a = after.get(tid)
        improved, raw_delta, regression, gate = _gate_per_task_row(
            b, a, max_regression_allowed,
        )
        metric = (a.metric if a else (b.metric if b else None))
        higher = (a.higher_better if a else (b.higher_better if b else True))
        if gate != "MISSING":
            n_compared += 1
            if improved:
                n_improving += 1
            worst_regression = max(worst_regression, regression)
        rows.append(DiffRow(
            task_id=tid,
            metric=metric,
            higher_better=higher,
            before=b,
            after=a,
            raw_delta=raw_delta,
            improved=improved,
            regression_amount=regression,
            per_task_gate=gate,
        ))

    aggregate_pass = (
        n_compared > 0
        and n_improving >= min_improving_required
        and worst_regression <= max_regression_allowed
    )
    return DiffResult(
        before_tag=before_tag,
        after_tag=after_tag,
        rows=rows,
        n_compared=n_compared,
        n_improving=n_improving,
        worst_regression=worst_regression,
        min_improving_required=min_improving_required,
        max_regression_allowed=max_regression_allowed,
        aggregate_gate="PASS" if aggregate_pass else "FAIL",
    )
