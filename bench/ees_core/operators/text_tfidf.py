"""Word/character TF-IDF operator for toxic-comment style NLP tasks."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidatePortfolio, CandidateResult
from bench.ees_core.folds import assemble_oof, choose_n_splits, make_folds


@dataclass(frozen=True)
class TargetSpec:
    mode: str
    target_cols: list[str]
    train_target_col: str | None = None
    class_labels: list[str] | None = None


@dataclass(frozen=True)
class ValidationScore:
    score: float | None
    primary_metric: str
    direction: str
    roc_auc: float | None = None
    log_loss: float | None = None


def _is_text_dtype(series: pd.Series) -> bool:
    # pandas 3 defaults text columns to `str` dtype rather than `object` (and
    # pandas 2 users may carry StringDtype); `dtype == object` misses both, which
    # under pandas 3 made every text task die with "no shared text column found"
    # (text_tfidf AND text_embedding share this detector). Same idiom as
    # seq2seq_lookup/tabular_portfolio's `_is_text_dtype`.
    from pandas.api.types import is_object_dtype, is_string_dtype

    return is_object_dtype(series) or is_string_dtype(series)


def _detect_text_column(train: pd.DataFrame, test: pd.DataFrame, target_cols: list[str]) -> str:
    return _detect_text_columns(train, test, target_cols)[0]


def _detect_text_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_cols: list[str],
    *,
    secondary_min_avg_len: float = 20.0,
) -> list[str]:
    """Return the shared text column(s) to vectorize, primary (longest) first.

    A mixed task can carry several free-text fields (random-acts-of-pizza has both
    ``request_title`` and ``request_text_edit_aware``); concatenating them lets the
    text leg exploit all of them instead of only the longest. Secondary columns are
    included only when genuinely textual (avg length >= ``secondary_min_avg_len``),
    so short/degenerate string columns (e.g. a timestamp string) don't add noise.
    Single-text tasks (jigsaw/spooky) are unaffected -- the list is just ``[primary]``.
    """
    excluded = set(target_cols)
    candidates = [col for col in train.columns if col in test.columns and col not in excluded]
    object_candidates = [col for col in candidates if _is_text_dtype(train[col]) or _is_text_dtype(test[col])]
    if not object_candidates:
        raise ValueError("no shared text column found")
    avg_len = {col: float(train[col].fillna("").astype(str).str.len().mean()) for col in object_candidates}
    ordered = sorted(object_candidates, key=lambda col: avg_len[col], reverse=True)
    primary = ordered[0]
    selected = [primary] + [col for col in ordered[1:] if avg_len[col] >= secondary_min_avg_len]
    return selected


def _join_text_columns(frame: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    joined = frame[text_cols[0]].fillna("").astype(str)
    for col in text_cols[1:]:
        joined = joined.str.cat(frame[col].fillna("").astype(str), sep=" ")
    return joined


def _target_spec(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> TargetSpec:
    sample_targets = [col for col in sample.columns[1:] if col in train.columns]
    sample_targets = [col for col in sample_targets if pd.api.types.is_numeric_dtype(train[col])]
    if sample_targets:
        return TargetSpec(mode="multilabel", target_cols=sample_targets)

    explicit = [col for col in train.columns if col not in test.columns]
    numeric_explicit = [col for col in explicit if pd.api.types.is_numeric_dtype(train[col])]
    if numeric_explicit:
        return TargetSpec(mode="multilabel", target_cols=numeric_explicit)

    if len(explicit) == 1:
        sample_targets = [col for col in sample.columns if col not in test.columns]
        if sample_targets and not pd.api.types.is_numeric_dtype(train[explicit[0]]):
            return TargetSpec(
                mode="multiclass",
                target_cols=sample_targets,
                train_target_col=explicit[0],
                class_labels=sample_targets,
            )

    raise ValueError("no numeric target columns found")


def _load_train_test_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if train_csv.exists() and test_csv.exists():
        return pd.read_csv(train_csv), pd.read_csv(test_csv)
    train_json = data_dir / "train.json"
    test_json = data_dir / "test.json"
    if train_json.exists() and test_json.exists():
        return pd.read_json(train_json), pd.read_json(test_json)
    raise FileNotFoundError("train/test CSV or JSON files not found")


_MISSING_STRING_TOKENS = {"", "n/a", "na", "none", "null", "nan"}


def _looks_like_list(series: pd.Series) -> bool:
    for value in series.dropna():
        return isinstance(value, (list, tuple, np.ndarray))
    return False


def build_structured_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str,
    exclude_cols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Engineer leak-safe numeric features from columns present in BOTH frames.

    Only columns shared by train and test are used, so any train-only column (the
    classic ``*_at_retrieval`` leak family in random-acts-of-pizza) is dropped by
    construction -- there is no hardcoded field list, only the intersection plus a
    small set of general, dtype-driven transforms:

    * numeric -> passthrough (plus hour/weekday when the name looks like a unix ts)
    * list-valued -> element count
    * string -> char length, word count, and a presence flag (catches the
      ``giver_username_if_known`` "N/A" signal generically)

    Returns ``(train_features, test_features, source_columns, dropped_leak_columns)``.
    """
    ignore = set(exclude_cols) | {id_col}
    shared = [c for c in train.columns if c in test.columns and c not in ignore]
    dropped_leaks = [c for c in train.columns if c not in test.columns and c not in ignore]

    train_parts: dict[str, np.ndarray] = {}
    test_parts: dict[str, np.ndarray] = {}
    used_sources: list[str] = []

    def add(name: str, tr_vals, te_vals) -> None:
        train_parts[name] = np.asarray(tr_vals, dtype=float)
        test_parts[name] = np.asarray(te_vals, dtype=float)

    def _len_or_zero(value) -> int:
        return len(value) if isinstance(value, (list, tuple, np.ndarray)) else 0

    for col in shared:
        tr, te = train[col], test[col]
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            add(col, tr.to_numpy(dtype=float), te.to_numpy(dtype=float))
            used_sources.append(col)
            lname = col.lower()
            if "timestamp" in lname or "unix" in lname or "_time" in lname:
                tr_dt = pd.to_datetime(tr, unit="s", errors="coerce", utc=True)
                te_dt = pd.to_datetime(te, unit="s", errors="coerce", utc=True)
                add(f"{col}__hour", tr_dt.dt.hour.fillna(-1), te_dt.dt.hour.fillna(-1))
                add(f"{col}__weekday", tr_dt.dt.weekday.fillna(-1), te_dt.dt.weekday.fillna(-1))
        elif _looks_like_list(tr) or _looks_like_list(te):
            add(f"{col}__count", tr.apply(_len_or_zero), te.apply(_len_or_zero))
            used_sources.append(col)
        elif _is_text_dtype(tr) or _is_text_dtype(te):
            tr_s = tr.fillna("").astype(str)
            te_s = te.fillna("").astype(str)
            add(f"{col}__len", tr_s.str.len(), te_s.str.len())
            add(f"{col}__wc", tr_s.str.split().apply(len), te_s.str.split().apply(len))
            add(
                f"{col}__present",
                (~tr_s.str.strip().str.lower().isin(_MISSING_STRING_TOKENS)).astype(float),
                (~te_s.str.strip().str.lower().isin(_MISSING_STRING_TOKENS)).astype(float),
            )
            used_sources.append(col)

    train_features = pd.DataFrame(train_parts, index=train.index).replace([np.inf, -np.inf], np.nan)
    test_features = pd.DataFrame(test_parts, index=test.index).replace([np.inf, -np.inf], np.nan)
    return train_features, test_features, used_sources, dropped_leaks


def _make_structured_head(random_state: int) -> HistGradientBoostingClassifier:
    # sklearn HGB (not LightGBM) on purpose: the text operator co-runs with the
    # torch-backed text_embedding operator in the same EES process, and importing
    # LightGBM alongside torch can segfault. HGB is competitive here and handles
    # NaNs natively.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=random_state,
    )


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    # sign-preserving log compression: tames the heavy right tails of count/karma
    # columns (and handles the negative upvotes-minus-downvotes case) so the linear
    # leg can exploit their monotonic signal. Trees don't need it; the linear head does.
    return np.sign(x) * np.log1p(np.abs(x))


def _make_linear_structured_head(random_state: int):
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        _LR(C=0.2, max_iter=2000, random_state=random_state),
    )


def _oof_single(make_model, x_train, x_test, y, folds) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    for fold_train_idx, fold_valid_idx in folds:
        if np.unique(y[fold_train_idx]).size < 2:
            oof[fold_valid_idx] = float(np.mean(y[fold_train_idx]))
            continue
        model = make_model()
        model.fit(x_train[fold_train_idx], y[fold_train_idx])
        oof[fold_valid_idx] = model.predict_proba(x_train[fold_valid_idx])[:, 1]
    model = make_model()
    model.fit(x_train, y)
    return oof, model.predict_proba(x_test)[:, 1]


def _fit_oof_structured(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
    objective_metric: str = "roc_auc",
) -> tuple[np.ndarray, np.ndarray]:
    """OOF + test probabilities for the structured leg, aligned to train row order.

    The leg is itself a small ensemble: a HistGradientBoosting tree head on the raw
    features plus a standardized logistic head on signed-log1p features, convex-blended
    at the OOF-optimal weight. Trees capture interactions/thresholds; the linear head
    captures monotonic karma/age/activity signals -- complementary on small tabular data.
    """
    hgb_oof, hgb_test = _oof_single(
        lambda: _make_structured_head(random_state), x_train, x_test, y, folds
    )
    x_train_l = _signed_log1p(x_train)
    x_test_l = _signed_log1p(x_test)
    lin_oof, lin_test = _oof_single(
        lambda: _make_linear_structured_head(random_state), x_train_l, x_test_l, y, folds
    )
    weight, _ = _choose_blend_weight(y, hgb_oof, lin_oof, objective_metric)
    oof = weight * hgb_oof + (1.0 - weight) * lin_oof
    test_pred = weight * hgb_test + (1.0 - weight) * lin_test
    return oof, test_pred


def _choose_blend_weight(
    y: np.ndarray,
    p_text: np.ndarray,
    p_struct: np.ndarray,
    objective_metric: str,
) -> tuple[float, float | None]:
    """Grid-search the convex blend weight on the text leg that optimizes the OOF metric."""
    minimize = objective_metric == "log_loss"
    if not minimize and np.unique(y).size < 2:
        return 1.0, None
    best_w, best_score = 1.0, None
    for w in np.linspace(0.0, 1.0, 21):
        p = w * p_text + (1.0 - w) * p_struct
        if minimize:
            score = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        else:
            score = float(roc_auc_score(y, p))
        if best_score is None:
            best_score, best_w = score, float(w)
            continue
        better = (score < best_score) if minimize else (score > best_score)
        if better:
            best_score, best_w = score, float(w)
    return best_w, best_score


def _fuse_structured_metadata(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_spec: TargetSpec,
    target_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    ensemble_validation_predictions: dict[str, np.ndarray],
    ensemble_test_predictions: dict[str, np.ndarray],
    validation_objective: tuple[str, str],
    random_state: int,
    extra_exclude: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Blend a leak-safe structured-feature leg into the text OOF/test predictions.

    Returns possibly-updated ``(validation_predictions, test_predictions, fusion_info)``.
    When no shared structured columns exist (pure text task) the predictions pass
    through unchanged and ``fusion_info["engaged"]`` is False.
    """
    if target_spec.mode == "multiclass":
        return ensemble_validation_predictions, ensemble_test_predictions, {
            "engaged": False,
            "reason": "multiclass_target",
        }

    exclude = set(target_cols)
    if target_spec.train_target_col:
        exclude.add(target_spec.train_target_col)
    if extra_exclude:
        exclude |= set(extra_exclude)
    id_col_name = str(train.columns[0])
    struct_train, struct_test, struct_sources, dropped_leaks = build_structured_features(
        train, test, id_col=id_col_name, exclude_cols=exclude
    )
    if not struct_sources:
        return ensemble_validation_predictions, ensemble_test_predictions, {
            "engaged": False,
            "reason": "no_shared_structured_columns",
            "dropped_leak_columns": dropped_leaks,
        }

    x_struct_train = np.nan_to_num(struct_train.to_numpy(dtype=float), nan=0.0)
    x_struct_test = np.nan_to_num(struct_test.to_numpy(dtype=float), nan=0.0)
    metric_name = validation_objective[0]
    fused_valid = dict(ensemble_validation_predictions)
    fused_test = dict(ensemble_test_predictions)
    blend_weights: dict[str, dict[str, float]] = {}
    struct_oof_scores: dict[str, float] = {}
    text_oof_scores: dict[str, float] = {}
    fused_oof_scores: dict[str, float] = {}
    any_fused = False
    for target in target_cols:
        y = train[target].to_numpy()
        if np.unique(y).size < 2:
            continue
        struct_oof, struct_test_pred = _fit_oof_structured(
            x_struct_train, x_struct_test, y, folds, random_state, objective_metric=metric_name
        )
        p_text_oof = np.asarray(ensemble_validation_predictions[target], dtype=float)
        weight, fused_score = _choose_blend_weight(y, p_text_oof, struct_oof, metric_name)
        fused_valid[target] = weight * p_text_oof + (1.0 - weight) * struct_oof
        fused_test[target] = (
            weight * np.asarray(ensemble_test_predictions[target], dtype=float)
            + (1.0 - weight) * struct_test_pred
        )
        blend_weights[target] = {"text": weight, "structured": round(1.0 - weight, 6)}
        if metric_name == "log_loss":
            struct_oof_scores[target] = float(log_loss(y, np.clip(struct_oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
            text_oof_scores[target] = float(log_loss(y, np.clip(p_text_oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
        else:
            struct_oof_scores[target] = float(roc_auc_score(y, struct_oof))
            text_oof_scores[target] = float(roc_auc_score(y, p_text_oof))
        if fused_score is not None:
            fused_oof_scores[target] = float(fused_score)
        any_fused = True

    if not any_fused:
        return ensemble_validation_predictions, ensemble_test_predictions, {
            "engaged": False,
            "reason": "no_binary_target",
            "dropped_leak_columns": dropped_leaks,
        }

    return fused_valid, fused_test, {
        "engaged": True,
        "structured_model": "hgb_plus_logistic",
        "structured_feature_count": int(x_struct_train.shape[1]),
        "structured_source_columns": struct_sources,
        "dropped_leak_columns": dropped_leaks,
        "blend_weights": blend_weights,
        "structured_oof_score": struct_oof_scores,
        "text_oof_score": text_oof_scores,
        "fused_oof_score": fused_oof_scores,
    }


def _auc_for_targets(y_true: pd.DataFrame, predictions: dict[str, np.ndarray]) -> float | None:
    scores = []
    for col, probs in predictions.items():
        labels = y_true[col].to_numpy()
        if np.unique(labels).size < 2:
            continue
        scores.append(float(roc_auc_score(labels, probs)))
    if not scores:
        return None
    return float(np.mean(scores))


def _score_validation_predictions(
    *,
    train: pd.DataFrame,
    valid_idx: np.ndarray,
    target_spec: TargetSpec,
    predictions: dict[str, np.ndarray],
) -> float | None:
    if target_spec.mode != "multiclass":
        return _auc_for_targets(train.iloc[valid_idx][target_spec.target_cols], predictions)
    assert target_spec.train_target_col is not None
    assert target_spec.class_labels is not None
    labels = train.iloc[valid_idx][target_spec.train_target_col].astype(str)
    if labels.nunique() < 2:
        return None
    probs = np.column_stack([predictions[label] for label in target_spec.class_labels])
    # Integer-encode y against the COLUMN order of `probs`. sklearn's log_loss
    # re-sorts string `labels` internally (np.unique), which silently swaps
    # probability columns whenever class_labels isn't lexicographically sorted.
    label_to_index = {label: index for index, label in enumerate(target_spec.class_labels)}
    y_index = np.asarray([label_to_index[value] for value in labels])
    return float(
        log_loss(y_index, np.clip(probs, 1e-6, 1 - 1e-6), labels=list(range(len(target_spec.class_labels))))
    )


def _infer_validation_objective(data_dir: Path, target_spec: TargetSpec) -> tuple[str, str]:
    description_path = data_dir / "description.md"
    description = description_path.read_text(encoding="utf-8", errors="ignore").lower() if description_path.exists() else ""
    if "roc auc" in description or "auc" in description or "area under" in description:
        return "roc_auc", "maximize"
    if "logarithmic loss" in description or "log loss" in description or "logloss" in description:
        return "log_loss", "minimize"
    if "accuracy" in description:
        return "accuracy", "maximize"
    if target_spec.mode == "multiclass":
        return "log_loss", "minimize"
    return "roc_auc", "maximize"


def _validation_score_bundle(
    *,
    train: pd.DataFrame,
    valid_idx: np.ndarray,
    target_spec: TargetSpec,
    predictions: dict[str, np.ndarray],
    objective: tuple[str, str],
) -> ValidationScore:
    objective_metric, direction = objective
    if target_spec.mode == "multiclass":
        score = _score_validation_predictions(
            train=train,
            valid_idx=valid_idx,
            target_spec=target_spec,
            predictions=predictions,
        )
        return ValidationScore(
            score=score,
            primary_metric="multiclass_log_loss",
            direction="minimize",
            log_loss=score,
        )

    auc = _auc_for_targets(train.iloc[valid_idx][target_spec.target_cols], predictions)
    if objective_metric == "roc_auc" or len(target_spec.target_cols) <= 1:
        return ValidationScore(
            score=auc,
            primary_metric="binary_roc_auc" if len(target_spec.target_cols) <= 1 else "multilabel_roc_auc",
            direction=direction if objective_metric == "roc_auc" else "maximize",
            roc_auc=auc,
        )

    losses = []
    for target in target_spec.target_cols:
        y_true = train.iloc[valid_idx][target].to_numpy()
        y_pred = np.clip(predictions[target], 1e-6, 1 - 1e-6)
        losses.append(float(log_loss(y_true, y_pred, labels=[0, 1])))
    mean_loss = float(np.mean(losses)) if losses else None
    return ValidationScore(
        score=mean_loss,
        primary_metric="multilabel_log_loss",
        direction=direction,
        roc_auc=auc,
        log_loss=mean_loss,
    )


def run_text_tfidf_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    random_state: int = 42,
    linear_solver: str = "auto",
    candidate_solvers: tuple[str, ...] | None = None,
    top_k: int = 3,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        timings: dict[str, float] = {}
        sample_path = _find_sample_submission(data_dir)
        if sample_path is None:
            raise FileNotFoundError("sample submission not found")
        checkpoint = time.time()
        train, test = _load_train_test_frames(data_dir)
        sample = pd.read_csv(sample_path)
        target_spec = _target_spec(train, test, sample)
        validation_objective = _infer_validation_objective(data_dir, target_spec)
        excluded_targets = list(target_spec.target_cols)
        if target_spec.train_target_col:
            excluded_targets.append(target_spec.train_target_col)
        target_cols = target_spec.target_cols
        text_cols = _detect_text_columns(train, test, excluded_targets)
        text_col = text_cols[0]
        timings["load_data_seconds"] = round(time.time() - checkpoint, 3)

        checkpoint = time.time()
        train_text = _join_text_columns(train, text_cols)
        test_text = _join_text_columns(test, text_cols)
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=220_000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=220_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        all_text = pd.concat([train_text, test_text], ignore_index=True)
        vectorizer.fit(all_text)
        timings["fit_vectorizer_seconds"] = round(time.time() - checkpoint, 3)
        checkpoint = time.time()
        x_all = vectorizer.transform(train_text)
        x_test = vectorizer.transform(test_text)
        timings["transform_text_seconds"] = round(time.time() - checkpoint, 3)

        checkpoint = time.time()
        # True full-train OOF (F.2 fold convention): every train row gets exactly
        # one out-of-fold prediction, so exported validation artifacts are
        # full-length, leak-free, and alignable across operators.
        if target_spec.mode == "multiclass":
            strat_source = train[target_spec.train_target_col].astype(str)
        elif len(target_cols) == 1:
            strat_source = train[target_cols[0]]
        else:
            strat_source = None
        stratified = (
            strat_source is not None
            and strat_source.nunique() > 1
            and int(strat_source.value_counts().min()) >= 2
        )
        fold_labels = strat_source.to_numpy() if stratified else np.zeros(len(train))
        n_splits = choose_n_splits(fold_labels, stratified=stratified)
        folds = make_folds(fold_labels, n_splits=n_splits, stratified=stratified, random_state=random_state)
        valid_idx = np.arange(len(train))  # OOF covers the whole train set
        timings["folds_seconds"] = round(time.time() - checkpoint, 3)

        solvers = _resolve_candidate_solvers(linear_solver, candidate_solvers, target_cols, target_spec.mode)
        candidate_results: list[CandidateResult] = []
        candidate_payloads: list[dict[str, Any]] = []
        validation_by_candidate: dict[str, dict[str, np.ndarray]] = {}
        test_by_candidate: dict[str, dict[str, np.ndarray]] = {}
        candidate_label_stats: dict[str, dict[str, dict[str, float | int | str]]] = {}
        candidate_label_timings: dict[str, dict[str, float]] = {}

        for solver_index, solver in enumerate(solvers):
            candidate_id = f"text_tfidf_{solver}"
            solver_validation_predictions: dict[str, np.ndarray] = {}
            solver_test_predictions: dict[str, np.ndarray] = {}
            label_stats: dict[str, dict[str, float | int | str]] = {}
            label_timings: dict[str, float] = {}
            if target_spec.mode == "multiclass":
                label_started = time.time()
                assert target_spec.train_target_col is not None
                assert target_spec.class_labels is not None
                y = train[target_spec.train_target_col].astype(str).to_numpy()
                if np.unique(y).size < 2:
                    counts = pd.Series(y).value_counts(normalize=True)
                    valid_probs = _constant_multiclass_probs(len(valid_idx), target_spec.class_labels, counts)
                    test_probs = _constant_multiclass_probs(len(test), target_spec.class_labels, counts)
                else:
                    fold_predictions = []
                    for fold_train_idx, fold_valid_idx in folds:
                        if np.unique(y[fold_train_idx]).size < 2:
                            fold_counts = pd.Series(y[fold_train_idx]).value_counts(normalize=True)
                            fold_probs = _constant_multiclass_probs(
                                len(fold_valid_idx), target_spec.class_labels, fold_counts
                            )
                        else:
                            fold_probs = _fit_predict_multiclass(
                                x_train=x_all[fold_train_idx],
                                y_train=y[fold_train_idx],
                                x_predict=x_all[fold_valid_idx],
                                class_labels=target_spec.class_labels,
                                solver=solver,
                                random_state=random_state + solver_index * 101,
                            )
                        fold_predictions.append((fold_valid_idx, fold_probs))
                    valid_probs = assemble_oof(len(train), len(target_spec.class_labels), fold_predictions)
                    test_probs = _fit_predict_multiclass(
                        x_train=x_all,
                        y_train=y,
                        x_predict=x_test,
                        class_labels=target_spec.class_labels,
                        solver=solver,
                        random_state=random_state + solver_index * 101,
                    )
                for index, label in enumerate(target_spec.class_labels):
                    solver_validation_predictions[label] = valid_probs[:, index]
                    solver_test_predictions[label] = test_probs[:, index]
                    label_stats[label] = {
                        "strategy": _strategy_name(solver),
                        "classes": int(np.unique(y).size),
                        "target_column": target_spec.train_target_col,
                    }
                label_timings[target_spec.train_target_col] = round(time.time() - label_started, 3)
            else:
                for target in target_cols:
                    label_started = time.time()
                    y = train[target].to_numpy()
                    if np.unique(y).size < 2:
                        rate = float(np.mean(y))
                        solver_validation_predictions[target] = np.full(len(valid_idx), rate)
                        solver_test_predictions[target] = np.full(len(test), rate)
                        label_stats[target] = {"strategy": "constant_rate", "positive_rate": rate}
                        label_timings[target] = round(time.time() - label_started, 3)
                        continue

                    fold_predictions = []
                    for fold_train_idx, fold_valid_idx in folds:
                        if np.unique(y[fold_train_idx]).size < 2:
                            fold_pred = np.full(len(fold_valid_idx), float(np.mean(y[fold_train_idx])))
                        else:
                            fold_pred = _fit_predict_label(
                                x_train=x_all[fold_train_idx],
                                y_train=y[fold_train_idx],
                                x_predict=x_all[fold_valid_idx],
                                solver=solver,
                                random_state=random_state + solver_index * 101,
                            )
                        fold_predictions.append((fold_valid_idx, fold_pred))
                    solver_validation_predictions[target] = assemble_oof(len(train), 1, fold_predictions).ravel()

                    solver_test_predictions[target] = _fit_predict_label(
                        x_train=x_all,
                        y_train=y,
                        x_predict=x_test,
                        solver=solver,
                        random_state=random_state + solver_index * 101,
                    )
                    label_stats[target] = {
                        "strategy": _strategy_name(solver),
                        "positive_rate": float(np.mean(y)),
                        "classes": int(np.unique(y).size),
                    }
                    label_timings[target] = round(time.time() - label_started, 3)

            validation_score = _validation_score_bundle(
                train=train,
                valid_idx=valid_idx,
                target_spec=target_spec,
                predictions=solver_validation_predictions,
                objective=validation_objective,
            )
            candidate_dir = output_dir / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_submission_path = candidate_dir / "submission.csv"
            candidate_metrics_path = candidate_dir / "metrics.json"
            _make_submission(sample, test, target_cols, solver_test_predictions).to_csv(
                candidate_submission_path,
                index=False,
            )
            candidate_metrics = {
                "candidate_id": candidate_id,
                "linear_solver": solver,
                "validation_score": validation_score.score,
                "validation_primary_metric": validation_score.primary_metric,
                "validation_direction": validation_score.direction,
                "validation_objective": {
                    "metric": validation_objective[0],
                    "direction": validation_objective[1],
                },
                "validation_roc_auc": validation_score.roc_auc,
                "validation_log_loss": validation_score.log_loss,
                "label_stats": label_stats,
                "fit_labels_seconds": label_timings,
                "prediction_artifacts": {
                    "validation_kind": "oof",
                },
            }
            candidate_metrics_path.write_text(json.dumps(candidate_metrics, indent=2, sort_keys=True) + "\n")
            candidate_result = CandidateResult(
                candidate_id=candidate_id,
                recipe_id="text_tfidf_portfolio",
                success=True,
                submission_path=candidate_submission_path,
                validation={"rows": int(len(test)), "columns": list(sample.columns)},
                score=validation_score.score,
                medal="none",
                final_artifact_source=f"ees_core:text_tfidf:{solver}",
                artifact_paths=[str(candidate_submission_path), str(candidate_metrics_path)],
                metrics=candidate_metrics,
                score_direction=validation_score.direction,
            )
            candidate_results.append(candidate_result)
            candidate_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "linear_solver": solver,
                    "validation_score": validation_score.score,
                    "validation_primary_metric": validation_score.primary_metric,
                    "validation_direction": validation_score.direction,
                    "validation_objective": {
                        "metric": validation_objective[0],
                        "direction": validation_objective[1],
                    },
                    "validation_roc_auc": validation_score.roc_auc,
                    "validation_log_loss": validation_score.log_loss,
                    "submission_path": str(candidate_submission_path),
                    "metrics_path": str(candidate_metrics_path),
                    "label_stats": label_stats,
                    "fit_labels_seconds": label_timings,
                }
            )
            validation_by_candidate[candidate_id] = solver_validation_predictions
            test_by_candidate[candidate_id] = solver_test_predictions
            candidate_label_stats[candidate_id] = label_stats
            candidate_label_timings[candidate_id] = label_timings

        portfolio = CandidatePortfolio(
            recipe_id="text_tfidf_portfolio",
            candidates=candidate_results,
            metric_direction=(
                "minimize"
                if candidate_results and candidate_results[0].metrics.get("validation_direction") == "minimize"
                else "maximize"
            ),
            selection_strategy="validation_top_k_average",
            top_k=top_k,
        )
        ranked_ids = [candidate.candidate_id for candidate in portfolio.ranked_valid_candidates]
        candidate_payloads.sort(key=lambda payload: ranked_ids.index(payload["candidate_id"]) if payload["candidate_id"] in ranked_ids else 10_000)
        top_candidates = portfolio.top_k_candidates
        if not top_candidates:
            successful_candidates = [
                candidate
                for candidate in candidate_results
                if candidate.success and candidate.submission_path is not None
            ]
            top_candidates = successful_candidates[: max(1, top_k)]
        top_ids = [candidate.candidate_id for candidate in top_candidates]
        ensemble_validation_predictions = _average_prediction_dicts(
            [validation_by_candidate[candidate_id] for candidate_id in top_ids],
            target_cols,
        )
        ensemble_test_predictions = _average_prediction_dicts(
            [test_by_candidate[candidate_id] for candidate_id in top_ids],
            target_cols,
        )

        # --- Text <-> structured metadata fusion (general to mixed text+tabular tasks) ---
        # The text legs above only see the request body. When the task also carries
        # structured fields shared by train AND test (e.g. random-acts-of-pizza's
        # account age / karma / timestamps / subreddit list / giver presence), fit a
        # tree leg on leak-safe engineered features and blend it with the text OOF at
        # the OOF-optimal weight. Train-only columns are dropped by construction.
        ensemble_validation_predictions, ensemble_test_predictions, fusion_info = _fuse_structured_metadata(
            train=train,
            test=test,
            target_spec=target_spec,
            target_cols=target_cols,
            folds=folds,
            ensemble_validation_predictions=ensemble_validation_predictions,
            ensemble_test_predictions=ensemble_test_predictions,
            validation_objective=validation_objective,
            random_state=random_state,
            extra_exclude={str(sample.columns[0])},
        )

        validation_score = _validation_score_bundle(
            train=train,
            valid_idx=valid_idx,
            target_spec=target_spec,
            predictions=ensemble_validation_predictions,
            objective=validation_objective,
        )
        submission = _make_submission(sample, test, target_cols, ensemble_test_predictions)

        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        validation_predictions_path = output_dir / "validation_predictions.csv"
        validation_targets_path = output_dir / "validation_targets.csv"
        submission.to_csv(submission_path, index=False)
        # Key OOF artifacts on the SAMPLE SUBMISSION's id column (a genuine unique
        # row id) rather than train.columns[0]. For JSON tasks like random-acts-of-pizza
        # the first train column is an arbitrary field (giver_username_if_known, mostly
        # "N/A" -> non-unique); an id-keyed merge during cross-operator/selection scoring
        # then cartesian-explodes and collapses the recomputed score. Fall back to the
        # positional first column when the sample id is absent from train.
        sample_id_col = str(sample.columns[0])
        oof_id_col = sample_id_col if sample_id_col in train.columns else str(train.columns[0])
        validation_prediction_frame = _make_validation_prediction_frame(
            train=train,
            valid_idx=valid_idx,
            target_cols=target_cols,
            predictions=ensemble_validation_predictions,
            id_col=oof_id_col,
        )
        validation_target_frame = _make_validation_target_frame(
            train=train,
            valid_idx=valid_idx,
            target_spec=target_spec,
            target_cols=target_cols,
            id_col=oof_id_col,
        )
        id_col_name = oof_id_col
        if id_col_name not in target_cols:
            # One frame convention: both frames carry the id in its NATIVE dtype and
            # are sorted by that same dtype, so their row orders are identical (the
            # registry aligns cross-operator OOF records by ordered id equality).
            # When the positional "id" col IS a target col (no genuine sample id),
            # sorting would scramble predictions vs targets independently — keep
            # construction order (metric_scoring aligns those positionally).
            validation_prediction_frame = validation_prediction_frame.sort_values(
                id_col_name, kind="stable"
            ).reset_index(drop=True)
            validation_target_frame = validation_target_frame.sort_values(
                id_col_name, kind="stable"
            ).reset_index(drop=True)
        validation_prediction_frame.to_csv(validation_predictions_path, index=False)
        validation_target_frame.to_csv(validation_targets_path, index=False)
        single_solver = len(solvers) == 1
        resolved_solver = solvers[0] if single_solver else "portfolio"
        result_candidate_id = (
            candidate_results[0].candidate_id
            if single_solver
            else f"text_tfidf_ensemble_top{len(top_candidates)}"
        )
        metrics = {
            "operator": "text_tfidf",
            "task_id": task_id,
            "text_col": text_col,
            "text_cols": text_cols,
            "target_cols": target_cols,
            "target_mode": target_spec.mode,
            "train_target_col": target_spec.train_target_col,
            "linear_solver": resolved_solver,
            "candidate_solvers": solvers,
            "validation_score": validation_score.score,
            "validation_primary_metric": validation_score.primary_metric,
            "validation_direction": validation_score.direction,
            "validation_objective": {
                "metric": validation_objective[0],
                "direction": validation_objective[1],
            },
            "validation_roc_auc": validation_score.roc_auc,
            "validation_log_loss": validation_score.log_loss,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "n_splits": int(n_splits),
            "label_stats": candidate_label_stats[top_ids[0]] if top_ids else {},
            "candidate_label_stats": candidate_label_stats,
            "fusion": fusion_info,
            "portfolio": portfolio.summary(),
            "portfolio_candidates": candidate_payloads,
            "selected_candidate_id": portfolio.selected.candidate_id,
            "prediction_strategy": "validation_top_k_average",
            "top_k": len(top_candidates),
            "top_k_candidate_ids": top_ids,
            "prediction_artifacts": {
                "test_prediction_path": str(submission_path),
                "validation_prediction_path": str(validation_predictions_path),
                "validation_target_path": str(validation_targets_path),
                "id_col": oof_id_col,
                "target_cols": target_cols,
                "validation_kind": "oof",
            },
            "timings": {
                **timings,
                "fit_labels_seconds": (
                    candidate_label_timings[top_ids[0]]
                    if single_solver and top_ids
                    else candidate_label_timings
                ),
            },
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id=result_candidate_id,
            recipe_id="text_tfidf_portfolio",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation_score.score,
            medal="none",
            final_artifact_source="ees_core:text_tfidf",
            artifact_paths=[
                str(submission_path),
                str(metrics_path),
                str(validation_predictions_path),
                str(validation_targets_path),
                *[
                    artifact
                    for candidate in candidate_results
                    for artifact in candidate.artifact_paths
                ],
            ],
            metrics=metrics,
            score_direction=validation_score.direction,
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="text_tfidf_logistic",
            recipe_id="text_tfidf_portfolio",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:text_tfidf",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _make_linear_head(solver: str, random_state: int):
    if solver in {"sgd", "nbsvm_blend"}:
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=35,
            tol=1e-3,
            n_jobs=-1,
            random_state=random_state,
        )
    if solver == "logistic":
        return LogisticRegression(
            C=4.0,
            solver="liblinear",
            max_iter=120,
            random_state=random_state,
        )
    raise ValueError(f"unknown linear_solver: {solver}")


def _resolve_candidate_solvers(
    linear_solver: str,
    candidate_solvers: tuple[str, ...] | None,
    target_cols: list[str],
    target_mode: str = "multilabel",
) -> list[str]:
    if candidate_solvers:
        solvers = list(candidate_solvers)
    elif linear_solver != "auto":
        solvers = [linear_solver]
    elif target_mode == "multiclass":
        solvers = ["logistic"]
    elif len(target_cols) > 3:
        solvers = ["sgd", "nbsvm_blend"]
    else:
        solvers = ["logistic"]

    valid = {"sgd", "nbsvm_blend", "logistic"}
    unknown = [solver for solver in solvers if solver not in valid]
    if unknown:
        raise ValueError(f"unknown linear_solver(s): {', '.join(unknown)}")
    deduped = []
    for solver in solvers:
        if solver not in deduped:
            deduped.append(solver)
    return deduped


def _average_prediction_dicts(
    predictions: list[dict[str, np.ndarray]],
    target_cols: list[str],
) -> dict[str, np.ndarray]:
    if not predictions:
        return {}
    return {
        target: np.mean([prediction[target] for prediction in predictions], axis=0)
        for target in target_cols
    }


def _make_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    target_cols: list[str],
    test_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    submission = sample.copy()
    id_col = str(sample.columns[0])
    predicted_rows = len(next(iter(test_predictions.values()))) if test_predictions else len(test)
    # Prefer id-keyed alignment to the sample submission over positional copying.
    # The JSON/frame load can hand the operator a test frame whose row count/order
    # does not match the sample submission -- a duplicate or extra id fans the frame
    # out to len(test) == len(sample) + 1 (observed as pandas' "Length of values
    # (1163) does not match length of index (1162)"). Positional assignment then
    # raises, erroring the whole operator so the controller falls back to the
    # sample baseline (0.5, no medal). Keying passthrough columns AND predictions on
    # the sample id makes the submission exactly one row per sample id regardless of
    # the test frame's shape/order; duplicate test ids collapse to their first
    # occurrence. Falls back to positional copy when no usable id column exists
    # (e.g. a headerless single-target sample) or the frames already line up.
    id_alignable = (
        id_col in test.columns
        and id_col not in target_cols
        and predicted_rows == len(test)
    )
    if id_alignable:
        lookup = pd.DataFrame({id_col: test[id_col].to_numpy()})
        for col in sample.columns:
            if col != id_col and col in test.columns and col not in target_cols:
                lookup[col] = test[col].to_numpy()
        for target in target_cols:
            if target in submission.columns:
                lookup[target] = np.clip(
                    np.asarray(test_predictions[target], dtype=float), 1e-6, 1 - 1e-6
                )
        lookup = lookup.drop_duplicates(subset=[id_col], keep="first").set_index(id_col)
        aligned = lookup.reindex(submission[id_col].to_numpy())
        for col in aligned.columns:
            submission[col] = aligned[col].to_numpy()
        return submission[list(sample.columns)]

    for col in sample.columns:
        if col in test.columns and col not in target_cols:
            submission[col] = test[col].values
    for target in target_cols:
        if target in submission.columns:
            submission[target] = np.clip(test_predictions[target], 1e-6, 1 - 1e-6)
    return submission[list(sample.columns)]


def _make_validation_prediction_frame(
    *,
    train: pd.DataFrame,
    valid_idx: np.ndarray,
    target_cols: list[str],
    predictions: dict[str, np.ndarray],
    id_col: str | None = None,
) -> pd.DataFrame:
    id_col = id_col or str(train.columns[0])
    frame = pd.DataFrame({id_col: train.iloc[valid_idx][id_col].to_numpy()})
    for target in target_cols:
        frame[target] = np.clip(predictions[target], 1e-6, 1 - 1e-6)
    return frame


def _make_validation_target_frame(
    *,
    train: pd.DataFrame,
    valid_idx: np.ndarray,
    target_spec: TargetSpec,
    target_cols: list[str],
    id_col: str | None = None,
) -> pd.DataFrame:
    id_col = id_col or str(train.columns[0])
    frame = pd.DataFrame({id_col: train.iloc[valid_idx][id_col].to_numpy()})
    if target_spec.mode == "multiclass":
        assert target_spec.train_target_col is not None
        labels = train.iloc[valid_idx][target_spec.train_target_col].astype(str)
        for target in target_cols:
            frame[target] = (labels == target).astype(int).to_numpy()
    else:
        for target in target_cols:
            frame[target] = train.iloc[valid_idx][target].to_numpy()
    return frame


def _strategy_name(solver: str) -> str:
    if solver == "sgd":
        return "tfidf_sgd_log_loss"
    if solver == "nbsvm_blend":
        return "tfidf_sgd_nbsvm_blend"
    return "tfidf_logistic"


def _fit_predict_label(
    *,
    x_train,
    y_train: np.ndarray,
    x_predict,
    solver: str,
    random_state: int,
) -> np.ndarray:
    if solver != "nbsvm_blend":
        model = _make_linear_head(solver, random_state)
        model.fit(x_train, y_train)
        return model.predict_proba(x_predict)[:, 1]

    raw_model = _make_linear_head("sgd", random_state)
    raw_model.fit(x_train, y_train)
    raw_probs = raw_model.predict_proba(x_predict)[:, 1]

    ratio = _nb_log_count_ratio(x_train, y_train)
    nb_train = x_train.multiply(ratio)
    nb_predict = x_predict.multiply(ratio)
    nb_model = _make_linear_head("sgd", random_state + 17)
    nb_model.fit(nb_train, y_train)
    nb_probs = nb_model.predict_proba(nb_predict)[:, 1]
    return np.clip(0.5 * raw_probs + 0.5 * nb_probs, 1e-6, 1 - 1e-6)


def _fit_predict_multiclass(
    *,
    x_train,
    y_train: np.ndarray,
    x_predict,
    class_labels: list[str],
    solver: str,
    random_state: int,
) -> np.ndarray:
    model = _make_linear_head("sgd" if solver == "nbsvm_blend" else solver, random_state)
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_predict)
    out = np.full((x_predict.shape[0], len(class_labels)), 1e-6, dtype=float)
    class_to_index = {str(label): index for index, label in enumerate(model.classes_)}
    for col_index, label in enumerate(class_labels):
        if label in class_to_index:
            out[:, col_index] = raw[:, class_to_index[label]]
    out /= out.sum(axis=1, keepdims=True)
    return np.clip(out, 1e-6, 1 - 1e-6)


def _constant_multiclass_probs(
    rows: int,
    class_labels: list[str],
    rates: pd.Series,
) -> np.ndarray:
    probs = np.array([float(rates.get(label, 1e-6)) for label in class_labels], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(len(class_labels), dtype=float)
    probs = probs / probs.sum()
    return np.tile(probs, (rows, 1))


def _nb_log_count_ratio(x_matrix, y: np.ndarray, alpha: float = 1.0):
    y = np.asarray(y)
    positive = np.asarray(x_matrix[y == 1].sum(axis=0)).ravel() + alpha
    negative = np.asarray(x_matrix[y == 0].sum(axis=0)).ravel() + alpha
    positive /= positive.sum()
    negative /= negative.sum()
    ratio = np.log(positive / negative)
    return sparse.csr_matrix(ratio)
