"""Generic local text embedding operator using TF-IDF SVD features."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.folds import assemble_oof, choose_n_splits, make_folds
from bench.ees_core.metric_scoring import internal_metric_direction
from bench.ees_core.operators.text_tfidf import _detect_text_column, _load_train_test_frames, _target_spec
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


def run_text_embedding_operator(
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
        train, test = _load_train_test_frames(data_dir)
        sample = pd.read_csv(sample_path)
        target_spec = _target_spec(train, test, sample)
        excluded = [*target_spec.target_cols]
        if target_spec.train_target_col:
            excluded.append(target_spec.train_target_col)
        text_col = _detect_text_column(train, test, excluded)
        train_text = train[text_col].fillna("").astype(str)
        test_text = test[text_col].fillna("").astype(str)
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        x_all_sparse = vectorizer.fit_transform(pd.concat([train_text, test_text], ignore_index=True))
        n_components = max(1, min(64, x_all_sparse.shape[0] - 1, x_all_sparse.shape[1] - 1))
        if n_components >= 2:
            embedder = make_pipeline(TruncatedSVD(n_components=n_components, random_state=random_state), Normalizer(copy=False))
            x_all = embedder.fit_transform(x_all_sparse)
        else:
            x_all = x_all_sparse.toarray()
        x_train = x_all[: len(train)]
        x_test = x_all[len(train) :]
        folds = _oof_folds(train, target_spec, random_state)
        submission, validation_predictions, validation_targets, validation = _fit_predict(
            train=train,
            sample=sample,
            target_spec=target_spec,
            x_train=x_train,
            x_test=x_test,
            folds=folds,
            random_state=random_state,
        )
        id_col = sample.columns[0]
        target_cols = list(sample.columns[1:])
        if id_col not in target_cols:
            # One frame convention: both frames carry the id in its NATIVE dtype and
            # sort by that same dtype, so their row orders stay identical (the
            # registry aligns cross-operator OOF records by ordered id equality).
            validation_predictions = validation_predictions.sort_values(id_col, kind="stable").reset_index(drop=True)
            validation_targets = validation_targets.sort_values(id_col, kind="stable").reset_index(drop=True)
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=id_col,
            target_cols=target_cols,
            validation_kind="oof",
        )
        metrics = {
            "operator": "text_embedding",
            "task_id": task_id,
            "text_col": text_col,
            "embedding_components": int(n_components),
            "validation": validation,
            "prediction_artifacts": prediction_artifacts,
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="text_embedding_svd_logistic",
            recipe_id="text_embedding",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation.get("score"),
            medal="none",
            final_artifact_source="ees_core:text_embedding",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            score_direction=internal_metric_direction(validation.get("metric")),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="text_embedding",
            recipe_id="text_embedding",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:text_embedding",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _oof_folds(train: pd.DataFrame, target_spec, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Full-train OOF folds (F.2 convention): stratify when every class supports it."""
    split_col = target_spec.train_target_col or target_spec.target_cols[0]
    y = train[split_col]
    stratified = bool(y.nunique() > 1 and int(y.value_counts().min()) >= 2)
    fold_labels = y.astype(str).to_numpy() if stratified else np.zeros(len(train))
    n_splits = choose_n_splits(fold_labels, stratified=stratified)
    return make_folds(fold_labels, n_splits=n_splits, stratified=stratified, random_state=random_state)


def _fit_predict(
    *,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    target_spec,
    x_train: np.ndarray,
    x_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    id_col = sample.columns[0]
    submission = sample.copy()
    ids = train[id_col].to_numpy() if id_col in train.columns else np.arange(len(train))
    if target_spec.mode == "multiclass":
        labels = list(target_spec.class_labels or sample.columns[1:])
        assert target_spec.train_target_col is not None
        y = train[target_spec.train_target_col].astype(str).to_numpy()
        oof_probs = assemble_oof(
            len(train),
            len(labels),
            [
                (fold_valid_idx, _classification_probs(x_train, fold_train_idx, fold_valid_idx, y, labels, random_state))
                for fold_train_idx, fold_valid_idx in folds
            ],
        )
        test_probs = _final_classification_probs(x_train, x_test, y, labels, random_state)
        for index, label in enumerate(labels):
            submission[label] = test_probs[:, index]
        validation_predictions = pd.DataFrame(oof_probs, columns=labels)
        validation_predictions.insert(0, id_col, ids)
        validation_targets = one_hot_frame(ids, y, id_col=id_col, target_cols=labels)
        clipped = np.clip(oof_probs, 1e-6, 1 - 1e-6)
        clipped = clipped / np.where(clipped.sum(axis=1, keepdims=True) == 0.0, 1.0, clipped.sum(axis=1, keepdims=True))
        # Integer-encode y against the COLUMN order of the probability matrix.
        # Passing string labels to sklearn's log_loss silently re-sorts them
        # (np.unique), swapping probability columns whenever `labels` isn't
        # lexicographically sorted — the score then disagrees with the canonical
        # score recomputed from the exported artifacts.
        label_to_index = {label: index for index, label in enumerate(labels)}
        y_index = np.asarray([label_to_index[value] for value in y])
        score = float(log_loss(y_index, clipped, labels=list(range(len(labels)))))
        return submission[list(sample.columns)], validation_predictions, validation_targets, {
            "metric": "log_loss",
            "score": score,
        }

    validation_predictions = pd.DataFrame({id_col: ids})
    validation_targets = pd.DataFrame({id_col: ids})
    scores = []
    for target in target_spec.target_cols:
        y = train[target].to_numpy()
        if pd.api.types.is_numeric_dtype(train[target]) and np.unique(y).size > 2:
            fold_predictions = []
            for fold_train_idx, fold_valid_idx in folds:
                model = Ridge(alpha=1.0, random_state=random_state)
                model.fit(x_train[fold_train_idx], y[fold_train_idx])
                fold_predictions.append((fold_valid_idx, model.predict(x_train[fold_valid_idx])))
            oof_pred = assemble_oof(len(train), 1, fold_predictions).ravel()
            final_model = Ridge(alpha=1.0, random_state=random_state)
            final_model.fit(x_train, y)
            test_pred = final_model.predict(x_test)
            score = float(mean_squared_error(y, oof_pred) ** 0.5)
        else:
            y_int = y.astype(int)
            fold_predictions = [
                (fold_valid_idx, _binary_probs(x_train, fold_train_idx, fold_valid_idx, y_int, random_state))
                for fold_train_idx, fold_valid_idx in folds
            ]
            oof_pred = assemble_oof(len(train), 1, fold_predictions).ravel()
            test_pred = _final_binary_probs(x_train, x_test, y_int, random_state)
            if np.unique(y_int).size > 1:
                score = float(roc_auc_score(y_int, oof_pred))
            else:
                score = float(accuracy_score(y_int, oof_pred >= 0.5))
        submission[target] = test_pred
        validation_predictions[target] = oof_pred
        validation_targets[target] = y
        scores.append(score)
    metric = "rmse" if len(target_spec.target_cols) == 1 and np.unique(train[target_spec.target_cols[0]]).size > 2 else "auc_or_accuracy"
    return submission[list(sample.columns)], validation_predictions, validation_targets, {
        "metric": metric,
        "score": float(np.mean(scores)) if scores else None,
    }


def _classification_probs(
    x_train: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    random_state: int,
) -> np.ndarray:
    if np.unique(y[train_idx]).size < 2:
        return _constant_probs(len(valid_idx), labels, pd.Series(y[train_idx]).value_counts(normalize=True))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train[train_idx], y[train_idx])
    return _aligned_probs(model, x_train[valid_idx], labels)


def _final_classification_probs(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    random_state: int,
) -> np.ndarray:
    if np.unique(y).size < 2:
        return _constant_probs(len(x_test), labels, pd.Series(y).value_counts(normalize=True))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y)
    return _aligned_probs(model, x_test, labels)


def _binary_probs(
    x_train: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> np.ndarray:
    if np.unique(y[train_idx]).size < 2:
        return np.full(len(valid_idx), float(np.mean(y[train_idx])))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train[train_idx], y[train_idx])
    if len(model.classes_) == 1:
        return np.full(len(valid_idx), float(model.classes_[0]))
    return model.predict_proba(x_train[valid_idx])[:, 1]


def _final_binary_probs(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> np.ndarray:
    if np.unique(y).size < 2:
        return np.full(len(x_test), float(np.mean(y)))
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y)
    if len(model.classes_) == 1:
        return np.full(len(x_test), float(model.classes_[0]))
    return model.predict_proba(x_test)[:, 1]


def _aligned_probs(model: LogisticRegression, features: np.ndarray, labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(features)
    probs = np.zeros((len(features), len(labels)), dtype=float)
    class_to_index = {str(label): index for index, label in enumerate(model.classes_)}
    for output_index, label in enumerate(labels):
        if str(label) in class_to_index:
            probs[:, output_index] = raw[:, class_to_index[str(label)]]
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row_sums == 0.0, 1.0, row_sums)


def _constant_probs(length: int, labels: list[str], frequencies: pd.Series) -> np.ndarray:
    row = np.array([float(frequencies.get(label, 0.0)) for label in labels], dtype=float)
    if row.sum() <= 0:
        row[:] = 1.0 / max(1, len(labels))
    else:
        row = row / row.sum()
    return np.tile(row, (length, 1))
