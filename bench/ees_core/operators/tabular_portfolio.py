"""Typed tabular portfolio operator for structured MLE-bench tasks."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.folds import assemble_oof, choose_n_splits, fold_safe_stratify_labels, make_folds
from bench.ees_core.metric_scoring import internal_metric_direction
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts

# --- row budget -------------------------------------------------------------
# EES_TABULAR_MAX_TRAIN_ROWS stays authoritative when set. When unset, the
# default cap escalates from _DEFAULT_MAX_TRAIN_ROWS to _ESCALATED_MAX_TRAIN_ROWS
# only for the narrow shape where rows were proven the biggest lever
# (tps-dec-2021: identical params scaled 0.9228 -> 0.9554 accuracy going
# 500k -> 3.2M rows): single-label multiclass classification (internal metric is
# plain accuracy) over a numeric-dominant feature frame whose float32 matrix at
# the escalated cap fits comfortably inside EES_TABULAR_MEMORY_BUDGET_BYTES.
_DEFAULT_MAX_TRAIN_ROWS = 120_000
_ESCALATED_MAX_TRAIN_ROWS = 2_000_000
_ROW_BUDGET_PROBE_ROWS = 4096
# Budget covers the float32 model-time feature matrix (rows x cols x 4 bytes);
# it is deliberately much smaller than typical RAM because CSV parsing, per-fold
# copies and per-class probability arrays all cost extra on top of it.
_DEFAULT_MEMORY_BUDGET_BYTES = 2_500_000_000
_BYTES_PER_FLOAT32 = 4
# Above this many train rows the classification path switches to "big data"
# behavior: 3 OOF folds instead of 5, float32 features, and LightGBM defaults
# with enough capacity to use the rows (see _make_classifier).
_BIG_DATA_ROW_THRESHOLD = 500_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _memory_budget_bytes() -> int:
    return _env_int("EES_TABULAR_MEMORY_BUDGET_BYTES", _DEFAULT_MEMORY_BUDGET_BYTES)


def _resolve_row_budget(train_path: Path, *, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[int, str]:
    """Return (effective train-row cap, source) — see the module-level note.

    Sources: "env" (EES_TABULAR_MAX_TRAIN_ROWS set — authoritative, <=0 means
    unlimited), "memory_aware_escalation", or "default". The escalation check is
    conservative on purpose: every condition below must hold, and any probe
    hiccup falls back to the default cap.
    """
    raw = os.environ.get("EES_TABULAR_MAX_TRAIN_ROWS")
    if raw is not None and str(raw).strip() != "":
        return int(raw), "env"
    default = _DEFAULT_MAX_TRAIN_ROWS
    try:
        head = pd.read_csv(train_path, nrows=_ROW_BUDGET_PROBE_ROWS)
    except Exception:
        return default, "default"
    if len(sample.columns) != 2:
        return default, "default"  # probability-per-class submissions are not the accuracy shape
    target_cols = [col for col in head.columns if col not in test.columns]
    if len(target_cols) != 1:
        return default, "default"
    target = head[target_cols[0]].dropna()
    class_count = int(target.nunique())
    # 3..20 classes: the multiclass label path whose internal metric is plain
    # accuracy. Binary (roc_auc) and regression shapes keep the historical cap.
    if not (3 <= class_count <= 20 and _looks_like_class_labels(target)):
        return default, "default"
    feature_cols = [col for col in head.columns if col != target_cols[0]]
    if not feature_cols:
        return default, "default"
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(head[col])]
    if len(numeric_cols) < 0.95 * len(feature_cols):
        return default, "default"  # text columns expand unpredictably (one-hot / hashing)
    estimated_footprint = _ESCALATED_MAX_TRAIN_ROWS * len(feature_cols) * _BYTES_PER_FLOAT32
    if estimated_footprint > _memory_budget_bytes():
        return default, "default"
    return _ESCALATED_MAX_TRAIN_ROWS, "memory_aware_escalation"


def _is_text_dtype(series: pd.Series) -> bool:
    # pandas 3 defaults text columns to `str` dtype rather than `object`; pandas 2
    # users may also carry StringDtype. `dtype != object` (or `== object`) silently
    # skips both under pandas 3 -- the June task_detection str-dtype bug family, and
    # the reason datetime/haversine engineering here stopped firing (and stopped
    # dropping leftover text columns) on real pandas-3 data. Accept any string-like
    # dtype instead of hardcoding `object`.
    from pandas.api.types import is_object_dtype, is_string_dtype

    return is_object_dtype(series) or is_string_dtype(series)


# --- general feature synthesis ---------------------------------------------
# All levers below are shape-gated (they fire only when the data has the right
# structure) and, where they add capacity, OOF-gated by the caller (kept only if
# out-of-fold score improves). None of them references a task id.

# Fixed-length string decomposition: an object/str column where (nearly) every
# value is the same length is almost always a positional code (e.g. a 10-char
# categorical string). Trees can't split "inside" such a token, so we expand it
# to per-position character codes + a distinct-character count + per-character
# occurrence counts. Bounds keep it from firing on free text (too long / too
# variable) or on constants.
_FIXED_STR_MIN_LEN = 2
_FIXED_STR_MAX_LEN = 40
_FIXED_STR_LENGTH_SHARE = 0.99  # (nearly) all values share one length
_FIXED_STR_MAX_ALPHABET = 128   # skip columns whose per-char count blows up


def _fixed_length_string_length(series: pd.Series) -> int | None:
    """Return the shared length if ``series`` is a fixed-length code column, else None.

    Gate (all must hold): text dtype; >= _FIXED_STR_LENGTH_SHARE of non-null
    values share one length; that length in [_FIXED_STR_MIN_LEN, _FIXED_STR_MAX_LEN];
    at least two distinct values; a small character alphabet. Datetime-looking
    columns are excluded (they have their own dedicated engineering)."""
    if not _is_text_dtype(series):
        return None
    values = series.dropna().astype(str)
    if len(values) == 0:
        return None
    lengths = values.str.len()
    top_len = int(lengths.mode().iloc[0])
    if not (_FIXED_STR_MIN_LEN <= top_len <= _FIXED_STR_MAX_LEN):
        return None
    if float((lengths == top_len).mean()) < _FIXED_STR_LENGTH_SHARE:
        return None
    if values.nunique() < 2:
        return None
    sample = values.iloc[: min(len(values), 2000)]
    # Datetime-looking columns have their own dedicated engineering; don't
    # decompose them character-by-character.
    if pd.to_datetime(sample, errors="coerce", utc=True).notna().mean() > 0.9:
        return None
    alphabet = set("".join(sample.tolist()))
    if len(alphabet) > _FIXED_STR_MAX_ALPHABET:
        return None
    return top_len


def _decompose_fixed_length_string(series: pd.Series, length: int, prefix: str) -> pd.DataFrame:
    """Expand a fixed-length code column into numeric features.

    Produces, for a value of the given ``length``:
      - ``{prefix}_pos{i}``: ordinal (byte) code of the char at position i
        (fit-free: a stable integer, so train/test agree without a vocabulary),
      - ``{prefix}_nunique``: number of distinct characters,
      - ``{prefix}_count_{code}``: occurrences of each character seen in this
        frame (train/test alignment is handled by the caller's column reindex).
    Vectorized via numpy fixed-width bytes with a pure-python fallback."""
    filled = series.fillna("").astype(str).str.slice(0, length)
    n = len(filled)
    try:
        raw = filled.to_numpy(dtype=f"S{length}")
        codes = raw.view(np.uint8).reshape(n, length)
    except Exception:
        codes = np.zeros((n, length), dtype=np.uint8)
        for row, val in enumerate(filled.to_numpy()):
            for pos, ch in enumerate(val[:length]):
                codes[row, pos] = ord(ch) % 256
    out: dict[str, np.ndarray] = {}
    for i in range(length):
        out[f"{prefix}_pos{i}"] = codes[:, i].astype(np.float64)
    present = [int(c) for c in np.unique(codes) if c != 0]
    nunique = np.zeros(n, dtype=np.float64)
    for code in present:
        col = (codes == code).sum(axis=1).astype(np.float64)
        out[f"{prefix}_count_{code}"] = col
        nunique += (col > 0)
    out[f"{prefix}_nunique"] = nunique
    return pd.DataFrame(out, index=series.index)


def _expand_fixed_length_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every fixed-length code column in ``df`` with its decomposition.

    Non-matching columns are untouched. General: no column names are hardcoded."""
    out = df
    replaced = False
    parts: list[pd.DataFrame] = []
    keep_cols: list[str] = []
    for col in df.columns:
        length = _fixed_length_string_length(df[col]) if _is_text_dtype(df[col]) else None
        if length is None:
            keep_cols.append(col)
            continue
        parts.append(_decompose_fixed_length_string(df[col], length, str(col)))
        replaced = True
    if not replaced:
        return df
    out = pd.concat([df[keep_cols]] + parts, axis=1)
    return out


def _engineer_tabular_features(
    df: pd.DataFrame, datetime_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer general numeric features (datetime parts + haversine).

    Returns the engineered frame and the list of columns treated as datetime.
    Pass that list back in as ``datetime_cols`` when engineering a second frame
    (e.g. test) so both frames get an identical datetime schema regardless of how
    cleanly the second frame happens to parse — otherwise a column that is genuine
    datetime in train but dips below the parse threshold in test would be silently
    dropped there and zero-filled on reindex.
    """
    out = df.copy()
    # Fixed-length code strings -> per-position / count features (before the
    # datetime pass and before the trailing non-numeric drop, so the signal is
    # captured instead of hashed away).
    out = _expand_fixed_length_strings(out)
    detected: list[str] = []
    # datetime columns -> numeric parts
    for col in list(out.columns):
        if not _is_text_dtype(out[col]):
            continue
        if datetime_cols is None:
            parsed = pd.to_datetime(out[col], errors="coerce", utc=True)
            is_datetime = parsed.notna().mean() > 0.9   # column is genuinely datetime
        else:
            is_datetime = col in datetime_cols
            parsed = (
                pd.to_datetime(out[col], errors="coerce", utc=True) if is_datetime else None
            )
        if is_datetime:
            out[f"{col}_year"] = parsed.dt.year
            out[f"{col}_month"] = parsed.dt.month
            out[f"{col}_day"] = parsed.dt.day
            out[f"{col}_hour"] = parsed.dt.hour
            out[f"{col}_weekday"] = parsed.dt.weekday
            out = out.drop(columns=[col])
            detected.append(col)
    # lat/lon pairs -> haversine between each (lat,lon) pair combination
    lats = [c for c in out.columns if "latitude" in c.lower()]
    lons = [c for c in out.columns if "longitude" in c.lower()]
    def _prefix(c):
        return c.lower().replace("latitude", "").replace("longitude", "").strip("_ ")
    pairs = {}
    for la in lats:
        pairs.setdefault(_prefix(la), {})["lat"] = la
    for lo in lons:
        pairs.setdefault(_prefix(lo), {})["lon"] = lo
    points = [(v["lat"], v["lon"]) for v in pairs.values() if "lat" in v and "lon" in v]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            (la1, lo1), (la2, lo2) = points[i], points[j]
            out[f"haversine_{i}_{j}"] = _haversine_km(out[la1], out[lo1], out[la2], out[lo2])
    # drop any remaining non-numeric columns so models get numeric input
    non_num = [c for c in out.columns if _is_text_dtype(out[c])]
    if non_num:
        out = out.drop(columns=non_num)
    return out, detected


def run_tabular_portfolio_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    random_state: int = 42,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sample_path = _find_sample_submission(data_dir)
        if sample_path is None:
            raise FileNotFoundError("sample submission not found")
        train_path = data_dir / "train.csv"
        if not train_path.exists() and (data_dir / "labels.csv").exists():
            train_path = data_dir / "labels.csv"
        test = pd.read_csv(data_dir / "test.csv")
        sample = pd.read_csv(sample_path)
        max_rows, row_budget_source = _resolve_row_budget(train_path, test=test, sample=sample)
        train = pd.read_csv(train_path, nrows=max_rows if max_rows > 0 else None)
        target_cols = [col for col in train.columns if col not in test.columns]
        if not target_cols:
            raise ValueError("no target column found")
        target_col = target_cols[0]
        train = _limit_training_rows(
            train, target_col=target_col, random_state=random_state, max_rows=max_rows
        )

        unique_targets = train[target_col].dropna().nunique()
        sample_targets = list(sample.columns[1:])
        numeric_sample_targets = [
            col
            for col in sample_targets
            if col in train.columns and pd.api.types.is_numeric_dtype(train[col])
        ]
        if len(numeric_sample_targets) > 1 and set(numeric_sample_targets).issubset(set(target_cols)):
            submission, validation, valid_pred_frame, valid_target_frame = _run_multioutput_regression(
                train,
                test,
                sample,
                numeric_sample_targets,
                random_state,
            )
            recipe_id = "tabular_multioutput_regression_portfolio"
            strategy = str(validation.get("strategy", "hist_gradient_boosting_multioutput_regression"))
        elif (
            pd.api.types.is_numeric_dtype(train[target_col])
            and len(sample.columns) == 2
            and not _looks_like_class_labels(train[target_col])
        ):
            submission, validation, valid_pred_frame, valid_target_frame = _run_regression(train, test, sample, target_col, random_state)
            recipe_id = "tabular_regression_portfolio"
            strategy = str(validation.get("strategy", "hist_gradient_boosting_regression"))
        elif len(sample.columns) == 2 and unique_targets <= 20:
            submission, validation, valid_pred_frame, valid_target_frame = _run_label_classification(train, test, sample, target_col, random_state)
            recipe_id = "tabular_label_classification_portfolio"
            strategy = str(validation.get("strategy", "extra_trees_label_classification"))
        elif pd.api.types.is_numeric_dtype(train[target_col]) and len(sample.columns) == 2:
            submission, validation, valid_pred_frame, valid_target_frame = _run_regression(train, test, sample, target_col, random_state)
            recipe_id = "tabular_regression_portfolio"
            strategy = str(validation.get("strategy", "hist_gradient_boosting_regression"))
        else:
            submission, validation, valid_pred_frame, valid_target_frame = _run_multiclass(train, test, sample, target_col, random_state)
            recipe_id = "tabular_multiclass_portfolio"
            strategy = str(validation.get("strategy", "extra_trees_multiclass"))

        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        valid_pred_frame = valid_pred_frame.sort_values(sample.columns[0], kind="stable").reset_index(drop=True)
        valid_target_frame = valid_target_frame.sort_values(sample.columns[0], kind="stable").reset_index(drop=True)
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=valid_pred_frame,
            validation_targets=valid_target_frame,
            id_col=sample.columns[0],
            target_cols=list(sample.columns[1:]),
            validation_kind="oof",
        )
        metrics = {
            "operator": "tabular_portfolio",
            "task_id": task_id,
            "target_col": target_col,
            "strategy": strategy,
            "validation": validation,
            "prediction_artifacts": prediction_artifacts,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "row_budget": int(max_rows),
            "row_budget_source": row_budget_source,
            "engineered_feature_count": int(validation.get("engineered_feature_count", 0)),
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id=strategy,
            recipe_id=recipe_id,
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation.get("score"),
            medal="none",
            final_artifact_source="ees_core:tabular_portfolio",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            score_direction=internal_metric_direction(validation.get("metric")),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="tabular_portfolio",
            recipe_id="tabular_portfolio",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:tabular_portfolio",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _oof_regression_pred(make_model, x: pd.DataFrame, y: np.ndarray, folds) -> np.ndarray:
    preds = []
    for tr, va in folds:
        model, _family = make_model()
        model.fit(x.iloc[tr], y[tr])
        preds.append((va, np.clip(model.predict(x.iloc[va]), 0, None).reshape(len(va), 1)))
    return assemble_oof(len(y), 1, preds)[:, 0]


def _oof_proba(make_model, x: pd.DataFrame, y: pd.Series, folds, labels: list) -> np.ndarray:
    preds = []
    for tr, va in folds:
        model, _family = make_model()
        model.fit(x.iloc[tr], y.iloc[tr])
        preds.append((va, _aligned_proba(model, x.iloc[va], labels)))
    return assemble_oof(len(y), len(labels), preds)


def _run_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    id_col = sample.columns[0]
    train_feature_frame = train.drop(columns=[id_col, target_col], errors="ignore")
    test_feature_frame = test.drop(columns=[id_col], errors="ignore")
    x, datetime_cols = _engineer_tabular_features(train_feature_frame)
    # Reuse train's datetime column set on test so both frames get an identical
    # datetime schema even if test parses less cleanly (avoids silent zero-fill).
    x_test, _ = _engineer_tabular_features(test_feature_frame, datetime_cols=datetime_cols)
    x_test = x_test.reindex(columns=x.columns, fill_value=0)
    # Geospatial hardening: bearings + data-derived anchor distances + cluster ids
    # (anchors fit on train, reused on test). Fires only with >=2 lat/lon pairs.
    x, geo_anchors = _augment_geospatial(x, anchors=None, random_state=random_state)
    x_test, _ = _augment_geospatial(x_test, anchors=geo_anchors, random_state=random_state)
    x_test = x_test.reindex(columns=x.columns, fill_value=0)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train[target_col].to_numpy(dtype=float)
    folds = make_folds(y, n_splits=choose_n_splits(y, stratified=False), stratified=False, random_state=random_state)

    def _reg_oof(frame: pd.DataFrame) -> np.ndarray:
        return _oof_regression_pred(lambda: _make_regressor(random_state=random_state), frame, y, folds)

    def _rmse(oof: np.ndarray) -> float:
        return float(mean_squared_error(y, oof) ** 0.5)

    base_oof = _reg_oof(x)
    base_rmse = _rmse(base_oof)
    # Interaction synthesis (OOF-gated): top-K important numeric features -> pairwise
    # products + safe ratios, kept only if OOF RMSE improves.
    top_k = max(2, _env_int("EES_TABULAR_INTERACTION_TOP_K", _INTERACTION_DEFAULT_TOP_K))
    importances = _quick_feature_importances(x, y, classifier=False, random_state=random_state)
    x, x_test, oof_pred, interaction_info = _select_interactions(
        x, x_test,
        importances=importances, base_oof=base_oof, base_score=base_rmse,
        oof_fn=_reg_oof, score_fn=_rmse, higher_better=False, top_k=top_k,
    )
    rmse = _rmse(oof_pred)
    # Skewed-target transform (OOF-gated): right-skewed non-negative target ->
    # fit on log1p(y), expm1 at predict; kept only if OOF RMSE improves.
    log_target = False
    target_info: dict = {"log1p_applied": False, "skewness": round(_skewness(y), 4)}
    if _should_log_transform_target(y):
        y_log = np.log1p(y)
        log_oof = np.expm1(
            _oof_regression_pred(lambda: _make_regressor(random_state=random_state), x, y_log, folds)
        )
        log_oof = np.clip(log_oof, 0, None)
        log_rmse = _rmse(log_oof)
        target_info["log1p_oof_rmse"] = round(log_rmse, 6)
        target_info["raw_oof_rmse"] = round(rmse, 6)
        if log_rmse < rmse:
            log_target = True
            oof_pred = log_oof
            rmse = log_rmse
            target_info["log1p_applied"] = True
    engineered_feature_count = int(x.shape[1])
    model, model_family = _make_regressor(random_state=random_state)
    if log_target:
        model.fit(x, np.log1p(y))
        pred = np.clip(np.expm1(model.predict(x_test)), 0, None)
    else:
        model.fit(x, y)
        pred = np.clip(model.predict(x_test), 0, None)
    submission = sample.copy()
    # Bracket assignment REPLACES the column (dtype inferred fresh from the new
    # values); an in-place `.iloc[:, 0] = ...` set instead forces the values into
    # `submission`'s existing column dtype. Under pandas 3's strict `str` dtype
    # that in-place set raises `TypeError: Invalid value '[...]' for dtype 'str'`
    # whenever the sample submission's id column is str-typed but test's id column
    # is int-typed (silently succeeded, dtype-upcasting, under pandas 2).
    submission[sample.columns[0]] = test.iloc[:, 0].to_numpy()
    submission[target_col] = pred
    valid_pred_frame = pd.DataFrame({id_col: train[id_col].to_numpy(), target_col: oof_pred})
    valid_target_frame = pd.DataFrame({id_col: train[id_col].to_numpy(), target_col: y})
    return (
        submission[list(sample.columns)],
        {
            "metric": "rmse",
            "score": rmse,
            "model_family": model_family,
            "strategy": f"{model_family}_regression",
            "engineered_feature_count": engineered_feature_count,
            "interactions": interaction_info,
            "target_transform": target_info,
        },
        valid_pred_frame,
        valid_target_frame,
    )


def _run_multioutput_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_cols: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    x = _tabular_features(train.drop(columns=target_cols), fit_columns=None)
    x_test = _tabular_features(test, fit_columns=list(x.columns))
    submission = sample.copy()
    # Bracket assignment, not .iloc[:, 0] in-place set: see _run_regression's note
    # (pandas 3 strict `str` dtype rejects int ids written into a str id column).
    submission[sample.columns[0]] = test.iloc[:, 0].to_numpy()
    id_col = sample.columns[0]
    valid_pred_frame = pd.DataFrame({id_col: train[id_col].to_numpy()})
    valid_target_frame = pd.DataFrame({id_col: train[id_col].to_numpy()})
    rmses = []
    for i, target_col in enumerate(target_cols):
        y = train[target_col].to_numpy(dtype=float)
        folds = make_folds(y, n_splits=choose_n_splits(y, stratified=False), stratified=False, random_state=random_state + i)
        oof_pred = _oof_regression_pred(lambda rs=random_state + i: _make_regressor(random_state=rs), x, y, folds)
        rmses.append(float(mean_squared_error(y, oof_pred) ** 0.5))
        model, model_family = _make_regressor(random_state=random_state + i)
        model.fit(x, y)
        submission[target_col] = np.clip(model.predict(x_test), 0, None)
        valid_pred_frame[target_col] = oof_pred
        valid_target_frame[target_col] = y
    return (
        submission[list(sample.columns)],
        {
            "metric": "mean_rmse",
            "score": float(np.mean(rmses)),
            "model_family": model_family if target_cols else "hist_gradient_boosting",
            "strategy": f"{model_family if target_cols else 'hist_gradient_boosting'}_multioutput_regression",
        },
        valid_pred_frame,
        valid_target_frame,
    )


def _limit_training_rows(
    train: pd.DataFrame,
    *,
    target_col: str,
    random_state: int,
    max_rows: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(train) <= max_rows:
        return train
    stratify = None
    if target_col in train.columns and train[target_col].nunique() <= 50:
        # Rare classes are merged into the majority label for subsampling
        # stratification only (train_test_split needs >= 2 members per class),
        # so one singleton row no longer disables stratified subsampling.
        stratify = fold_safe_stratify_labels(train[target_col].to_numpy(), n_splits=2)
    sampled, _ = train_test_split(
        train,
        train_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return sampled.reset_index(drop=True)


def _looks_like_class_labels(y: pd.Series) -> bool:
    unique_count = y.nunique(dropna=True)
    if unique_count > 20:
        return False
    if unique_count > 2 and unique_count / max(1, len(y.dropna())) > 0.8:
        return False
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty:
        return True
    return bool(np.allclose(numeric.to_numpy(dtype=float), np.round(numeric.to_numpy(dtype=float))))


def _run_multiclass(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    labels = list(sample.columns[1:])
    x = _tabular_features(train.drop(columns=[target_col]), fit_columns=None)
    x_test = _tabular_features(test, fit_columns=list(x.columns))
    y = train[target_col].astype(str)
    folds, _n_splits, _stratified = _label_fold_plan(
        y.to_numpy(), random_state=random_state, desired_splits=5
    )
    oof = _oof_proba(lambda: _make_classifier(random_state=random_state, class_count=len(labels)), x, y, folds, labels)
    model, model_family = _make_classifier(random_state=random_state, class_count=len(labels))
    validation = {
        "metric": "log_loss",
        "score": float(log_loss(y, oof, labels=labels)),
        "model_family": model_family,
        "strategy": f"{model_family}_multiclass",
    }
    model.fit(x, y)
    probs = _aligned_proba(model, x_test, labels)
    submission = sample.copy()
    # Bracket assignment, not .iloc[:, 0] in-place set: see _run_regression's note
    # (pandas 3 strict `str` dtype rejects int ids written into a str id column).
    submission[sample.columns[0]] = test.iloc[:, 0].to_numpy()
    for index, label in enumerate(labels):
        submission[label] = np.clip(probs[:, index], 1e-6, 1 - 1e-6)
    row_sums = submission[labels].sum(axis=1)
    submission[labels] = submission[labels].div(row_sums, axis=0)
    id_col = sample.columns[0]
    valid_pred_frame = pd.DataFrame({id_col: train[id_col].to_numpy()})
    for index, label in enumerate(labels):
        valid_pred_frame[label] = oof[:, index]
    valid_target_frame = one_hot_frame(
        train[id_col],
        y,
        id_col=id_col,
        target_cols=labels,
    )
    return submission[list(sample.columns)], validation, valid_pred_frame, valid_target_frame


def _label_fold_plan(
    y_arr: np.ndarray,
    *,
    random_state: int,
    desired_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, bool]:
    """Folds for label targets with the rare-class guard applied.

    Classes with fewer than ``desired_splits`` members are merged into the
    majority class for fold assignment only (see fold_safe_stratify_labels), so
    a singleton class neither crashes StratifiedKFold nor silently forces the
    whole task onto unstratified KFold. Returns (folds, n_splits, stratified).
    """
    stratify_labels = fold_safe_stratify_labels(y_arr, n_splits=desired_splits)
    if stratify_labels is not None:
        folds = make_folds(
            stratify_labels,
            n_splits=desired_splits,
            stratified=True,
            random_state=random_state,
        )
        return folds, desired_splits, True
    n_splits = choose_n_splits(y_arr, stratified=False, max_splits=desired_splits)
    folds = make_folds(y_arr, n_splits=n_splits, stratified=False, random_state=random_state)
    return folds, n_splits, False


def _run_label_classification(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    x = _tabular_features(train.drop(columns=[target_col]), fit_columns=None)
    x_test = _tabular_features(test, fit_columns=list(x.columns))
    y = train[target_col]
    id_col = sample.columns[0]
    submission = sample.copy()
    # Bracket assignment, not .iloc[:, 0] in-place set: see _run_regression's note
    # (pandas 3 strict `str` dtype rejects int ids written into a str id column).
    submission[sample.columns[0]] = test.iloc[:, 0].to_numpy()
    big_data = len(x) >= _BIG_DATA_ROW_THRESHOLD
    if big_data:
        # Halve the model-time matrix footprint; both LightGBM and sklearn trees
        # train natively on float32 (this is the footprint the row-budget
        # escalation heuristic estimated against).
        x = x.astype(np.float32)
        x_test = x_test.astype(np.float32)
    desired_splits = 3 if big_data else 5
    if y.nunique() == 2:
        # Native-dtype positive label (matches model.classes_, which preserves y's dtype) —
        # NOT a stringified label, since `_class_probability` indexes `model.classes_` by
        # the exact native value (int/bool/str labels would otherwise miss the dict lookup).
        positive_label = sorted(y.unique())[-1]
        y_pos_int = (y == positive_label).astype(int).to_numpy()
        folds, n_splits, stratified = _label_fold_plan(
            y.to_numpy(), random_state=random_state, desired_splits=desired_splits
        )

        def _binary_oof(frame: pd.DataFrame) -> np.ndarray:
            oof = np.zeros(len(y))
            for tr, va in folds:
                m, _fam = _make_classifier(random_state=random_state, class_count=2, big_data=big_data)
                m.fit(frame.iloc[tr], y.iloc[tr])
                # If positive_label is absent from this fold's training set, fall back to
                # base rate to degrade gracefully instead of crashing with KeyError.
                if positive_label in m.classes_:
                    oof[va] = _class_probability(m, frame.iloc[va], positive_label)
                else:
                    oof[va] = float((y.iloc[tr] == positive_label).mean())
            return oof

        def _auc(oof: np.ndarray) -> float:
            return float(roc_auc_score(y_pos_int, oof))

        base_oof = _binary_oof(x)
        can_score = y.nunique() > 1
        base_auc = _auc(base_oof) if can_score else 0.0
        interaction_info: dict = {"applied": False}
        if can_score:
            # Interaction synthesis (OOF-gated): top-K important numeric features ->
            # pairwise products + safe ratios, kept only if OOF AUC improves.
            top_k = max(2, _env_int("EES_TABULAR_INTERACTION_TOP_K", _INTERACTION_DEFAULT_TOP_K))
            importances = _quick_feature_importances(x, y_pos_int, classifier=True, random_state=random_state)
            x, x_test, oof_pos, interaction_info = _select_interactions(
                x, x_test,
                importances=importances, base_oof=base_oof, base_score=base_auc,
                oof_fn=_binary_oof, score_fn=_auc, higher_better=True, top_k=top_k,
            )
        else:
            oof_pos = base_oof
        score = _auc(oof_pos) if can_score else None
        model, family = _make_classifier(random_state=random_state, class_count=2, big_data=big_data)
        model.fit(x, y)
        submission[target_col] = np.clip(_class_probability(model, x_test, positive_label), 1e-6, 1 - 1e-6)
        validation: dict[str, float | str] = {
            "metric": "roc_auc",
            "model_family": family,
            "strategy": f"{family}_label_classification",
            "folds": {"n_splits": n_splits, "stratified": stratified},
            "interactions": interaction_info,
        }
        if score is not None:
            validation["score"] = score
        valid_pred_values = oof_pos
    else:
        y_arr = y.to_numpy()
        classes = np.unique(y_arr)
        folds, n_splits, stratified = _label_fold_plan(
            y_arr, random_state=random_state, desired_splits=desired_splits
        )
        families = _classification_families()
        oof_by_family: dict[str, np.ndarray] = {}
        test_by_family: dict[str, np.ndarray] = {}
        family_accuracy: dict[str, float] = {}
        for family in families:
            oof = np.zeros((len(y_arr), len(classes)), dtype=np.float32)
            for tr, va in folds:
                oof[va] = _family_predict_proba(
                    family,
                    x.iloc[tr],
                    y_arr[tr],
                    [x.iloc[va]],
                    classes=classes,
                    random_state=random_state,
                    big_data=big_data,
                )[0]
            test_by_family[family] = _family_predict_proba(
                family, x, y_arr, [x_test], classes=classes, random_state=random_state, big_data=big_data
            )[0]
            oof_by_family[family] = oof
            family_accuracy[family] = float(accuracy_score(y_arr, classes[oof.argmax(axis=1)]))

        # Selection is OOF-only: best single family vs. the OOF-swept soft blend.
        # The blend has to live here (not in the prediction registry) because
        # 2-column label submissions carry hard labels downstream — averaging
        # label values across candidates is meaningless, so probabilities must
        # be blended before the argmax collapses them.
        best_family = max(families, key=lambda name: family_accuracy[name])
        chosen_oof = oof_by_family[best_family]
        chosen_test = test_by_family[best_family]
        chosen_family = best_family
        chosen_score = family_accuracy[best_family]
        validation = {
            "metric": "accuracy",
            "family_oof_accuracy": family_accuracy,
            "folds": {"n_splits": n_splits, "stratified": stratified},
        }
        if len(families) == 2:
            first, second = families
            weight, blend_accuracy = _blend_weight_sweep(
                oof_by_family[first], oof_by_family[second], y_arr, classes
            )
            validation["blend"] = {
                f"weight_{first}": weight,
                "oof_accuracy": blend_accuracy,
            }
            if blend_accuracy > chosen_score:
                chosen_oof = weight * oof_by_family[first] + (1 - weight) * oof_by_family[second]
                chosen_test = weight * test_by_family[first] + (1 - weight) * test_by_family[second]
                chosen_family = f"blend_{first}_{second}"
                chosen_score = blend_accuracy
        validation["score"] = chosen_score
        validation["model_family"] = chosen_family
        validation["strategy"] = f"{chosen_family}_label_classification"
        # classes carries y's dtype, so argmax-indexed predictions store as native
        # labels, not boxed objects (accuracy_score chokes on object-dtype labels).
        oof_label = classes[chosen_oof.argmax(axis=1)]
        submission[target_col] = classes[chosen_test.argmax(axis=1)]
        valid_pred_values = oof_label
    valid_pred_frame = pd.DataFrame({id_col: train[id_col].to_numpy(), sample.columns[1]: valid_pred_values})
    valid_target_frame = pd.DataFrame({id_col: train[id_col].to_numpy(), sample.columns[1]: y.to_numpy()})
    return submission[list(sample.columns)], validation, valid_pred_frame, valid_target_frame


def _classification_families() -> list[str]:
    """Model families the label-classification path should field.

    "auto" fields both the LightGBM leg and the ExtraTrees strong leg (tps-dec
    proved the 0.3*LGBM + 0.7*ET soft blend at 0.96086 official, gold); an
    explicit EES_TABULAR_MODEL_FAMILY pin (e.g. an evolution variant) fields
    only that family.
    """
    family = _preferred_model_family()
    if family in {"lightgbm", "lgbm"}:
        return ["lightgbm"]
    if family in {"extra_trees", "et"}:
        return ["extra_trees"]
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return ["extra_trees"]
    return ["lightgbm", "extra_trees"]


def _family_predict_proba(
    family: str,
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    eval_frames: list[pd.DataFrame],
    *,
    classes: np.ndarray,
    random_state: int,
    big_data: bool,
) -> list[np.ndarray]:
    """Fit one family on (x_fit, y_fit) and return class-aligned probabilities
    for each frame in ``eval_frames``."""
    if family == "extra_trees":
        return _batched_extra_trees_probas(
            x_fit, y_fit, eval_frames, classes=classes, random_state=random_state
        )
    model, _resolved = _make_classifier(
        random_state=random_state, class_count=len(classes), big_data=big_data
    )
    model.fit(x_fit, y_fit)
    return [_class_aligned_proba(model, frame, classes) for frame in eval_frames]


def _batched_extra_trees_probas(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    eval_frames: list[pd.DataFrame],
    *,
    classes: np.ndarray,
    random_state: int,
) -> list[np.ndarray]:
    """ExtraTrees strong leg with a bounded memory profile.

    Proven params (tps-dec-2021, full rows: holdout 0.95896 / official 0.95954 —
    gold on its own): ~240 trees, min_samples_leaf=2, max_features=0.3. The
    forest is fit as independent seed-varied batches of <= EES_TABULAR_ET_BATCH_TREES
    trees whose probabilities are averaged — statistically equivalent to one
    large forest, but peak memory only ever holds one batch of trees (240 trees
    on millions of rows would otherwise hold several GB of tree nodes at once).
    """
    total_trees = max(1, _env_int("EES_TABULAR_ET_N_ESTIMATORS", _env_int("EES_TABULAR_N_ESTIMATORS", 240)))
    batch_trees = max(1, min(total_trees, _env_int("EES_TABULAR_ET_BATCH_TREES", 80)))
    min_samples_leaf = max(1, _env_int("EES_TABULAR_ET_MIN_SAMPLES_LEAF", 2))
    max_features = _extra_trees_max_features()
    sums = [np.zeros((len(frame), len(classes)), dtype=np.float64) for frame in eval_frames]
    built = 0
    batch_index = 0
    while built < total_trees:
        n_batch = min(batch_trees, total_trees - built)
        model = ExtraTreesClassifier(
            n_estimators=n_batch,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state + batch_index,
            n_jobs=-1,
        )
        model.fit(x_fit, y_fit)
        for index, frame in enumerate(eval_frames):
            sums[index] += _class_aligned_proba(model, frame, classes).astype(np.float64) * n_batch
        built += n_batch
        batch_index += 1
        del model
    return [(total / total_trees).astype(np.float32) for total in sums]


def _extra_trees_max_features() -> float | str:
    raw = os.environ.get("EES_TABULAR_ET_MAX_FEATURES", "").strip()
    if not raw:
        return 0.3
    try:
        return float(raw)
    except ValueError:
        return raw  # e.g. "sqrt"


def _class_aligned_proba(model, x: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """predict_proba aligned to the global (sorted, native-dtype) class order.

    A fold-trained model can miss globally rare classes; its columns are mapped
    into the global order and missing classes stay at probability 0.
    """
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)), dtype=np.float32)
    positions = np.searchsorted(classes, model.classes_)
    out[:, positions] = raw.astype(np.float32)
    return out


def _blend_weight_sweep(
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    y_arr: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float]:
    """Sweep w in [0, 1] maximizing OOF accuracy of argmax(w*a + (1-w)*b).

    OOF-only selection: the official test set is never touched. (tps-dec proved
    w=0.3 on LightGBM vs. ExtraTrees: 0.96035 holdout -> 0.96086 official.)
    """
    best_weight = 0.5
    best_accuracy = -1.0
    for weight in np.linspace(0.0, 1.0, 11):
        blended = weight * oof_a + (1.0 - weight) * oof_b
        accuracy = float(accuracy_score(y_arr, classes[blended.argmax(axis=1)]))
        if accuracy > best_accuracy:
            best_weight = float(weight)
            best_accuracy = accuracy
    return best_weight, best_accuracy


def _tabular_n_estimators() -> int:
    return max(10, int(os.environ.get("EES_TABULAR_N_ESTIMATORS", "160") or "160"))


def _preferred_model_family() -> str:
    return os.environ.get("EES_TABULAR_MODEL_FAMILY", "auto").strip().lower() or "auto"


# LightGBM defaults by data size. The small profile is the historical one; the
# big profile is what the tps-dec-2021 row ladder proved out (same params scaled
# 0.9228 -> 0.9554 accuracy from 500k to 3.2M rows): more rows only pay off with
# enough model capacity to use them. Env vars remain authoritative over both.
_LGBM_SMALL_DEFAULTS = {
    "n_estimators": 160,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "subsample_freq": 0,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
}
_LGBM_BIG_DEFAULTS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "num_leaves": 127,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
}


def _make_classifier(*, random_state: int, class_count: int, big_data: bool = False):
    family = _preferred_model_family()
    if family in {"auto", "lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier

            defaults = _LGBM_BIG_DEFAULTS if big_data else _LGBM_SMALL_DEFAULTS
            objective = "binary" if class_count <= 2 else "multiclass"
            extra_params = {"num_class": class_count} if class_count > 2 else {}
            return (
                LGBMClassifier(
                    objective=objective,
                    n_estimators=max(10, _env_int("EES_TABULAR_N_ESTIMATORS", defaults["n_estimators"])),
                    learning_rate=_env_float("EES_TABULAR_LGBM_LEARNING_RATE", defaults["learning_rate"]),
                    num_leaves=max(8, _env_int("EES_TABULAR_LGBM_NUM_LEAVES", defaults["num_leaves"])),
                    subsample=_env_float("EES_TABULAR_LGBM_SUBSAMPLE", defaults["subsample"]),
                    subsample_freq=_env_int("EES_TABULAR_LGBM_SUBSAMPLE_FREQ", defaults["subsample_freq"]),
                    colsample_bytree=_env_float("EES_TABULAR_LGBM_COLSAMPLE", defaults["colsample_bytree"]),
                    min_child_samples=max(1, _env_int("EES_TABULAR_LGBM_MIN_CHILD_SAMPLES", defaults["min_child_samples"])),
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    **extra_params,
                ),
                "lightgbm",
            )
        except Exception:
            if family in {"lightgbm", "lgbm"}:
                raise
    # Proven ET profile (tps-dec: 240 trees, min_samples_leaf=2, max_features=0.3
    # beat every LightGBM config); the multiclass label path fits this family in
    # memory-capped batches instead (see _batched_extra_trees_probas).
    return (
        ExtraTreesClassifier(
            n_estimators=max(1, _env_int("EES_TABULAR_ET_N_ESTIMATORS", _env_int("EES_TABULAR_N_ESTIMATORS", 240))),
            max_features=_extra_trees_max_features(),
            min_samples_leaf=max(1, _env_int("EES_TABULAR_ET_MIN_SAMPLES_LEAF", 2)),
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees",
    )


def _make_regressor(*, random_state: int):
    family = _preferred_model_family()
    if family in {"auto", "lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMRegressor

            return (
                LGBMRegressor(
                    objective="regression",
                    n_estimators=_tabular_n_estimators(),
                    learning_rate=float(os.environ.get("EES_TABULAR_LGBM_LEARNING_RATE", "0.05") or "0.05"),
                    num_leaves=max(8, int(os.environ.get("EES_TABULAR_LGBM_NUM_LEAVES", "31") or "31")),
                    subsample=float(os.environ.get("EES_TABULAR_LGBM_SUBSAMPLE", "0.9") or "0.9"),
                    colsample_bytree=float(os.environ.get("EES_TABULAR_LGBM_COLSAMPLE", "0.9") or "0.9"),
                    min_child_samples=max(1, int(os.environ.get("EES_TABULAR_LGBM_MIN_CHILD_SAMPLES", "20") or "20")),
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                "lightgbm",
            )
        except Exception:
            if family in {"lightgbm", "lgbm"}:
                raise
    return (
        HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=random_state,
        ),
        "hist_gradient_boosting",
    )


def _class_probability(model: ExtraTreesClassifier, x: pd.DataFrame, label) -> np.ndarray:
    probs = model.predict_proba(x)
    class_to_index = {value: index for index, value in enumerate(model.classes_)}
    return probs[:, class_to_index[label]]


def _aligned_proba(model: ExtraTreesClassifier, x: pd.DataFrame, labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.full((len(x), len(labels)), 1e-6, dtype=float)
    class_to_index = {str(label): idx for idx, label in enumerate(model.classes_)}
    for col_index, label in enumerate(labels):
        if label in class_to_index:
            out[:, col_index] = raw[:, class_to_index[label]]
    out /= out.sum(axis=1, keepdims=True)
    return out


def _tabular_features(df: pd.DataFrame, fit_columns: list[str] | None) -> pd.DataFrame:
    # Fixed-length code strings expand to per-position / count features rather
    # than being one-hot / hashed (which loses the positional signal). Train/test
    # per-character-count column alignment is handled by the fit_columns reindex.
    df = _expand_fixed_length_strings(df)
    feature_parts: list[pd.DataFrame | pd.Series] = []
    for col in df.columns:
        if col.lower() in {"key", "id"}:
            continue
        series = df[col]
        if "datetime" in col.lower() or "date" == col.lower():
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            feature_parts.append(pd.DataFrame({
                f"{col}_year": parsed.dt.year.fillna(0),
                f"{col}_month": parsed.dt.month.fillna(0),
                f"{col}_day": parsed.dt.day.fillna(0),
                f"{col}_hour": parsed.dt.hour.fillna(0),
                f"{col}_dow": parsed.dt.dayofweek.fillna(0),
            }, index=df.index))
        elif pd.api.types.is_numeric_dtype(series):
            feature_parts.append(pd.to_numeric(series, errors="coerce").rename(col))
        else:
            values = series.fillna("__missing__").astype(str)
            max_onehot = int(os.environ.get("EES_TABULAR_MAX_ONEHOT_CARDINALITY", "40") or "40")
            if values.nunique(dropna=False) <= max_onehot:
                encoded = pd.get_dummies(values, prefix=col)
                feature_parts.append(encoded)
            else:
                hashed = pd.util.hash_pandas_object(values, index=False).astype("uint64")
                feature_parts.append(
                    pd.DataFrame(
                        {
                            f"{col}_hash_mod": (hashed % 997).astype(float),
                            f"{col}_len": values.str.len().astype(float),
                        },
                        index=df.index,
                    )
                )

    if feature_parts:
        features = pd.concat(feature_parts, axis=1)
    else:
        features = pd.DataFrame(index=df.index)

    lon_lat = {
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    }
    if lon_lat.issubset(df.columns):
        lat1 = pd.to_numeric(df["pickup_latitude"], errors="coerce")
        lon1 = pd.to_numeric(df["pickup_longitude"], errors="coerce")
        lat2 = pd.to_numeric(df["dropoff_latitude"], errors="coerce")
        lon2 = pd.to_numeric(df["dropoff_longitude"], errors="coerce")
        features = pd.concat([
            features,
            pd.DataFrame({
                "abs_lat_diff": (lat1 - lat2).abs(),
                "abs_lon_diff": (lon1 - lon2).abs(),
                "haversine_km": _haversine_km(lat1, lon1, lat2, lon2),
            }, index=df.index),
        ], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    if fit_columns is not None:
        features = features.reindex(columns=fit_columns, fill_value=0)
    return features


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing (degrees) from point 1 to point 2 — a general
    geospatial signal (direction of travel), complementary to haversine distance."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# --- geospatial hardening ---------------------------------------------------
# When a frame carries >= 2 lat/lon-like column pairs we add, per pair, a
# bearing, and — using anchors *derived from the training coordinates* (KMeans
# cluster centroids, never hardcoded coordinates) — a nearest-cluster id plus the
# haversine distance from each point to every anchor. Anchors are fit on train
# and reused on test.
_GEO_DEFAULT_CLUSTERS = 10


def _detect_lat_lon_pairs(columns: list[str]) -> list[tuple[str, str, str]]:
    """Return [(lat_col, lon_col, prefix)] for columns whose names look like a
    latitude/longitude pair sharing a prefix. General name-based detection."""
    cols = list(columns)
    lats = [c for c in cols if "latitude" in c.lower()]
    lons = [c for c in cols if "longitude" in c.lower()]

    def _prefix(c: str) -> str:
        return c.lower().replace("latitude", "").replace("longitude", "").strip("_ ")

    pairs: dict[str, dict[str, str]] = {}
    for la in lats:
        pairs.setdefault(_prefix(la), {})["lat"] = la
    for lo in lons:
        pairs.setdefault(_prefix(lo), {})["lon"] = lo
    out: list[tuple[str, str, str]] = []
    for prefix, pair in pairs.items():
        if "lat" in pair and "lon" in pair:
            out.append((pair["lat"], pair["lon"], prefix or "geo"))
    return out


def _fit_geo_anchors(df: pd.DataFrame, pairs: list[tuple[str, str, str]], *, random_state: int) -> np.ndarray | None:
    """Fit cluster centroids over all (lat, lon) points stacked across pairs.

    Returns an array of shape (n_clusters, 2) or None if unavailable. Anchors are
    data-derived (KMeans centroids), NOT hardcoded landmark coordinates."""
    n_clusters = max(2, _env_int("EES_TABULAR_GEO_CLUSTERS", _GEO_DEFAULT_CLUSTERS))
    stacked = []
    for lat_c, lon_c, _ in pairs:
        block = np.column_stack([
            pd.to_numeric(df[lat_c], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[lon_c], errors="coerce").to_numpy(dtype=float),
        ])
        stacked.append(block)
    if not stacked:
        return None
    points = np.vstack(stacked)
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if len(points) < n_clusters:
        return None
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=4096)
        km.fit(points)
        return km.cluster_centers_.astype(float)
    except Exception:
        return None


def _augment_geospatial(
    df: pd.DataFrame,
    *,
    anchors: np.ndarray | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Add bearing + anchor-distance + cluster-id features when >=2 lat/lon pairs
    are present. Returns (augmented_frame, anchors). When ``anchors`` is None it
    fits new anchors on ``df`` (train); pass them back for the test frame."""
    pairs = _detect_lat_lon_pairs(list(df.columns))
    if len(pairs) < 2:
        return df, anchors
    if anchors is None:
        anchors = _fit_geo_anchors(df, pairs, random_state=random_state)
    new_cols: dict[str, np.ndarray] = {}
    for lat_c, lon_c, prefix in pairs:
        lat = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df[lon_c], errors="coerce").to_numpy(dtype=float)
        if anchors is not None:
            dists = np.column_stack([
                _haversine_km(pd.Series(lat), pd.Series(lon), pd.Series(np.full(len(df), c[0])), pd.Series(np.full(len(df), c[1])))
                for c in anchors
            ])
            for k in range(anchors.shape[0]):
                new_cols[f"geo_{prefix}_anchor{k}_km"] = dists[:, k]
            new_cols[f"geo_{prefix}_cluster"] = np.nan_to_num(dists, nan=np.inf).argmin(axis=1).astype(float)
    # pairwise bearings between the pairs' points (direction of travel)
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            la1 = pd.to_numeric(df[pairs[i][0]], errors="coerce").to_numpy(dtype=float)
            lo1 = pd.to_numeric(df[pairs[i][1]], errors="coerce").to_numpy(dtype=float)
            la2 = pd.to_numeric(df[pairs[j][0]], errors="coerce").to_numpy(dtype=float)
            lo2 = pd.to_numeric(df[pairs[j][1]], errors="coerce").to_numpy(dtype=float)
            new_cols[f"geo_bearing_{i}_{j}"] = _bearing_deg(la1, lo1, la2, lo2)
    if not new_cols:
        return df, anchors
    add = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, add], axis=1), anchors


# --- interaction synthesis --------------------------------------------------
# After a quick model fit we take the top-K most important numeric features and
# add pairwise products and safe ratios. K is bounded so the candidate set stays
# small. The caller keeps the augmented set only if OOF score improves.
_INTERACTION_DEFAULT_TOP_K = 6


def _interaction_pairs_from_importance(
    x: pd.DataFrame, importances: np.ndarray, *, top_k: int
) -> list[tuple[str, str]]:
    """Pick top-K numeric columns by importance and return their unordered pairs."""
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    if len(numeric_cols) < 2:
        return []
    col_index = {c: i for i, c in enumerate(x.columns)}
    scored = sorted(
        numeric_cols,
        key=lambda c: float(importances[col_index[c]]) if col_index[c] < len(importances) else 0.0,
        reverse=True,
    )
    chosen = scored[: max(2, top_k)]
    pairs: list[tuple[str, str]] = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            pairs.append((chosen[i], chosen[j]))
    return pairs


def _build_interactions(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Materialize product + safe-ratio features for the given column pairs.

    Ratios guard against divide-by-zero (denominator floored away from 0). The
    exact same ``pairs`` are applied to train and test so the schema matches."""
    out: dict[str, np.ndarray] = {}
    eps = 1e-6
    for a, b in pairs:
        va = pd.to_numeric(df[a], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(df[b], errors="coerce").to_numpy(dtype=float)
        out[f"mul__{a}__{b}"] = va * vb
        den_b = np.where(np.abs(vb) < eps, np.sign(vb) * eps + eps, vb)
        den_a = np.where(np.abs(va) < eps, np.sign(va) * eps + eps, va)
        out[f"div__{a}__{b}"] = va / den_b
        out[f"div__{b}__{a}"] = vb / den_a
    frame = pd.DataFrame(out, index=df.index)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def _quick_feature_importances(
    x: pd.DataFrame, y: np.ndarray, *, classifier: bool, random_state: int
) -> np.ndarray:
    """Fast feature-importance vector aligned to ``x.columns`` for pair selection.

    Uses a small LightGBM (falls back to ExtraTrees) purely to rank features; it
    is never used for prediction, so a cheap fit is fine."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if classifier else LGBMRegressor
        model = cls(n_estimators=120, random_state=random_state, n_jobs=-1, verbosity=-1)
        model.fit(x, y)
        return np.asarray(model.feature_importances_, dtype=float)
    except Exception:
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        cls = ExtraTreesClassifier if classifier else ExtraTreesRegressor
        model = cls(n_estimators=100, random_state=random_state, n_jobs=-1)
        model.fit(x, y)
        return np.asarray(model.feature_importances_, dtype=float)


def _select_interactions(
    x: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    importances: np.ndarray,
    base_oof: np.ndarray,
    base_score: float,
    oof_fn,
    score_fn,
    higher_better: bool,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """OOF-gated interaction augmentation.

    Builds product + safe-ratio features for the top-K important numeric columns,
    computes the augmented OOF, and keeps the augmented frames only if the OOF
    score beats ``base_score``. Returns (x, x_test, chosen_oof, info)."""
    pairs = _interaction_pairs_from_importance(x, importances, top_k=top_k)
    info: dict = {"applied": False, "n_pairs": len(pairs), "base_score": float(base_score)}
    if not pairs:
        info["reason"] = "no_numeric_pairs"
        return x, x_test, base_oof, info
    inter_train = _build_interactions(x, pairs)
    inter_test = _build_interactions(x_test, pairs)
    x_aug = pd.concat([x, inter_train], axis=1)
    x_test_aug = pd.concat([x_test, inter_test], axis=1)
    aug_oof = oof_fn(x_aug)
    aug_score = float(score_fn(aug_oof))
    info["aug_score"] = aug_score
    info["n_features_added"] = int(inter_train.shape[1])
    improved = aug_score > base_score if higher_better else aug_score < base_score
    if improved:
        info["applied"] = True
        return x_aug, x_test_aug, aug_oof, info
    return x, x_test, base_oof, info


# --- skewed-target transform ------------------------------------------------
def _skewness(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.0
    mu = v.mean()
    sigma = v.std()
    if sigma == 0:
        return 0.0
    return float(np.mean(((v - mu) / sigma) ** 3))


def _should_log_transform_target(y: np.ndarray, *, threshold: float | None = None) -> bool:
    """Right-skew detector for a non-negative regression target. General gate:
    fires only when all values are >= 0 and skewness exceeds the threshold."""
    thr = _env_float("EES_TABULAR_TARGET_SKEW_THRESHOLD", 0.75) if threshold is None else threshold
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or np.any(arr < 0):
        return False
    return _skewness(arr) > thr
