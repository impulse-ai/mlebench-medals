"""Task-metric-aware scoring for reliable, comparable candidate selection.

Selection must compare candidates on ONE metric (the task's) in the correct
direction. Operators score themselves in their own internal metrics, so a raw
cross-operator `score` comparison is meaningless; this module recomputes a
single canonical score per candidate from its held-out validation artifacts.
"""
from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd

from bench.stats import higher_is_better

if TYPE_CHECKING:
    from bench.ees_core.candidates import CandidateResult

Scorer = Callable[[np.ndarray, np.ndarray], float]

_AUC_WORD_RE = re.compile(r"\bauc\b")


def _canonicalize_metric(metric: str | None) -> str:
    """Map mlebench's DESCRIPTIVE grader names to a SHORT canonical token.

    `mlebench.registry.registry.get_competition(task_id).grader.name` returns
    human-readable strings like "root_mean_squared_error", "multi-class-log-loss",
    "mean-column-wise-roc-auc", or even "column-wise ROC AUC" (mixed case, spaces).
    Both `metric_direction` and `scorer_for_metric` need a single normalized
    vocabulary to key off, or they silently miss real names (that was the bug:
    `bench.stats.higher_is_better`'s hint list has "rmse"/"mse" but not the
    spelled-out "root_mean_squared_error", so RMSE tasks defaulted to maximize).

    This is pure string mapping — no task IDs, no per-competition branches.
    Unknown names pass through unchanged (normalized) so callers can still
    fall back sanely (default direction / None scorer).
    """
    if not metric:
        return ""
    normalized = metric.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    # Order matters: more specific variants must be checked before their
    # more generic substrings (rmsle before rmse before mse; log-loss
    # before generic).
    if "rmsle" in normalized or "root mean squared log error" in normalized:
        return "rmsle"
    if "rmse" in normalized or "root mean squared error" in normalized:
        return "rmse"
    if "mse" in normalized or "mean squared error" in normalized:
        return "mse"
    if "mae" in normalized or "mean absolute error" in normalized:
        return "mae"
    if "log loss" in normalized or "logloss" in normalized or "cross entropy" in normalized:
        return "log-loss"
    if "roc auc" in normalized or "auc roc" in normalized or _AUC_WORD_RE.search(normalized):
        return "auc"
    if "accuracy" in normalized:
        return "accuracy"
    if "quadratic weighted kappa" in normalized or "qwk" in normalized:
        return "qwk"
    if "kappa" in normalized:
        return "kappa"
    if "f1" in normalized:
        for average in ("macro", "micro", "weighted", "samples"):
            if average in normalized:
                return f"f1-{average}"
        return "f1"
    return normalized


def metric_direction(metric: str | None) -> Literal["minimize", "maximize"]:
    """Single source of truth for selection direction (unifies the two resolvers)."""
    canonical = _canonicalize_metric(metric)
    return "maximize" if higher_is_better(canonical, warn=False) else "minimize"


_MINIMIZE_CANONICAL_TOKENS = {"rmsle", "rmse", "mse", "mae", "log-loss"}
_MAXIMIZE_CANONICAL_TOKENS = {
    "auc", "accuracy", "qwk", "kappa", "f1", "f1-macro", "f1-micro", "f1-weighted", "f1-samples",
}


def internal_metric_direction(metric: str | None) -> Literal["minimize", "maximize"] | None:
    """Direction of an operator's SELF-REPORTED metric, or None when not certain.

    Unlike ``metric_direction`` this never defaults: it is used to populate
    ``CandidateResult.score_direction``, which drives selection when the task
    metric is unknown, so a guessed direction would be worse than no declaration.
    """
    canonical = _canonicalize_metric(metric)
    if canonical in _MINIMIZE_CANONICAL_TOKENS:
        return "minimize"
    if canonical in _MAXIMIZE_CANONICAL_TOKENS:
        return "maximize"
    return None


def _normalize(metric: str | None) -> str:
    return _canonicalize_metric(metric)


def _as_labels(y: np.ndarray) -> np.ndarray:
    """Collapse a one-hot / proba matrix to integer class labels; pass through 1-D."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1)
    return arr.reshape(-1)


def scorer_for_metric(metric: str | None) -> Scorer | None:
    """Return f(y_true, y_pred) -> float in the metric's native units, or None."""
    m = _normalize(metric)

    if m in {"rmse"}:
        from sklearn.metrics import mean_squared_error
        return lambda yt, yp: float(mean_squared_error(_as_labels(yt), _as_labels(yp)) ** 0.5)

    if m in {"mse"}:
        from sklearn.metrics import mean_squared_error
        return lambda yt, yp: float(mean_squared_error(_as_labels(yt), _as_labels(yp)))

    if m in {"rmsle"}:
        from sklearn.metrics import mean_squared_log_error
        return lambda yt, yp: float(
            mean_squared_log_error(
                np.clip(_as_labels(yt), 0, None), np.clip(_as_labels(yp), 0, None)
            ) ** 0.5
        )

    if m in {"mae"}:
        from sklearn.metrics import mean_absolute_error
        return lambda yt, yp: float(mean_absolute_error(_as_labels(yt), _as_labels(yp)))

    if m in {"auc", "roc-auc", "auc-roc"}:
        from sklearn.metrics import roc_auc_score

        def _auc(yt: np.ndarray, yp: np.ndarray) -> float:
            yt_a = np.asarray(yt, dtype=float)
            yp_a = np.asarray(yp, dtype=float)
            if yt_a.ndim == 2 and yt_a.shape[1] > 1:
                return float(roc_auc_score(yt_a, yp_a, average="macro"))
            return float(roc_auc_score(yt_a.reshape(-1), yp_a.reshape(-1)))

        return _auc

    if m in {"log-loss", "logloss", "cross-entropy", "crossentropy"}:
        from sklearn.metrics import log_loss

        def _ll(yt: np.ndarray, yp: np.ndarray) -> float:
            yt_a = np.asarray(yt, dtype=float)
            yp_a = np.asarray(yp, dtype=float)
            if yt_a.ndim == 2 and yt_a.shape[1] > 1:
                labels = list(range(yt_a.shape[1]))
                return float(log_loss(np.argmax(yt_a, axis=1), yp_a, labels=labels))
            return float(log_loss(yt_a.reshape(-1), yp_a.reshape(-1), labels=[0, 1]))

        return _ll

    if m in {"accuracy"}:
        from sklearn.metrics import accuracy_score

        def _acc(yt: np.ndarray, yp: np.ndarray) -> float:
            yt_l = _as_labels(yt)
            yp_a = np.asarray(yp, dtype=float)
            yp_l = np.argmax(yp_a, axis=1) if yp_a.ndim == 2 and yp_a.shape[1] > 1 else (yp_a.reshape(-1) >= 0.5).astype(int)
            return float(accuracy_score(yt_l.astype(int), yp_l))

        return _acc

    if m in {"f1", "f1-macro", "f1-micro", "f1-weighted", "f1-samples"}:
        from sklearn.metrics import f1_score
        average = m.split("-", 1)[1] if "-" in m else "binary"

        def _f1(yt: np.ndarray, yp: np.ndarray) -> float:
            yt_l = _as_labels(yt)
            yp_a = np.asarray(yp, dtype=float)
            yp_l = np.argmax(yp_a, axis=1) if yp_a.ndim == 2 and yp_a.shape[1] > 1 else (yp_a.reshape(-1) >= 0.5).astype(int)
            return float(f1_score(yt_l.astype(int), yp_l, average=average))

        return _f1

    if m in {"kappa", "cohen-kappa", "quadratic-weighted-kappa", "qwk"}:
        from sklearn.metrics import cohen_kappa_score
        weights = "quadratic" if m in {"quadratic-weighted-kappa", "qwk"} else None

        def _kappa(yt: np.ndarray, yp: np.ndarray) -> float:
            yt_l = _as_labels(yt)
            yp_a = np.asarray(yp, dtype=float)
            yp_l = np.argmax(yp_a, axis=1) if yp_a.ndim == 2 and yp_a.shape[1] > 1 else np.rint(yp_a.reshape(-1))
            return float(cohen_kappa_score(yt_l.astype(int), yp_l.astype(int), weights=weights))

        return _kappa

    return None


def _load_aligned(artifacts: dict) -> tuple[np.ndarray, np.ndarray] | None:
    pred_path = artifacts.get("validation_prediction_path")
    tgt_path = artifacts.get("validation_target_path")
    id_col = artifacts.get("id_col")
    target_cols = artifacts.get("target_cols")
    if not pred_path or not tgt_path or not id_col or not target_cols:
        return None
    try:
        preds = pd.read_csv(pred_path)
        targets = pd.read_csv(tgt_path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    cols = [id_col, *target_cols]
    if any(c not in preds.columns for c in cols) or any(c not in targets.columns for c in cols):
        return None
    if id_col in target_cols:
        # Some tasks (e.g. text-classification tasks with no genuine sample id)
        # have an operator-supplied id_col that IS a target column, because the
        # operator falls back to the first train column as a positional id. An
        # id-keyed merge is then ambiguous (duplicate column name) and would be
        # meaningless anyway (predictions vs. labels don't share values to key
        # on). Both frames are built from the same row order upstream, so align
        # positionally instead of by id.
        if len(preds) != len(targets) or len(targets) == 0:
            return None
        try:
            y_true = targets[target_cols].to_numpy(dtype=float)
            y_pred = preds[target_cols].to_numpy(dtype=float)
        except ValueError:
            return None
        return y_true, y_pred
    # align on id to be order-independent
    merged = targets[cols].merge(preds[cols], on=id_col, suffixes=("_true", "_pred"))
    if len(merged) == 0:
        return None
    try:
        y_true = merged[[f"{c}_true" for c in target_cols]].to_numpy(dtype=float)
        y_pred = merged[[f"{c}_pred" for c in target_cols]].to_numpy(dtype=float)
    except ValueError:
        return None
    return y_true, y_pred


def canonical_validation_score(candidate: "CandidateResult", metric: str | None) -> float | None:
    scorer = scorer_for_metric(metric)
    if scorer is None:
        return None
    artifacts = candidate.metrics.get("prediction_artifacts")
    if not isinstance(artifacts, dict):
        return None
    loaded = _load_aligned(artifacts)
    if loaded is None:
        return None
    y_true, y_pred = loaded
    try:
        return float(scorer(y_true, y_pred))
    except (ValueError, IndexError):
        return None


def attach_selection_scores(candidates: list["CandidateResult"], metric: str | None) -> list["CandidateResult"]:
    out = []
    for candidate in candidates:
        score = canonical_validation_score(candidate, metric)
        out.append(candidate if score is None else dataclasses.replace(candidate, selection_score=score))
    return out
