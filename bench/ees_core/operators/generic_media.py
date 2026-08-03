"""Generic media feature operators for image and audio tasks."""
from __future__ import annotations

import json
import math
import os
import time
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from bench.adapters.task_detection import AUDIO_EXTS, IMAGE_EXTS, _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.folds import assemble_oof, choose_n_splits, make_folds
from bench.ees_core.metric_scoring import internal_metric_direction
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


def _fit_balanced_classifier(model: ExtraTreesClassifier, features: pd.DataFrame, y) -> ExtraTreesClassifier:
    """Fit with balanced class weighting via precomputed sample weights.

    `class_weight="balanced"` inside the estimator builds a weight dict keyed by
    the class labels; sklearn >= 1.9's compute_class_weight int-coerces digit-string
    classes when resolving that dict, so stringified numeric labels ('0'/'1' --
    exactly what our `astype(str)` label canonicalization produces) miss the lookup
    and fitting dies with `ValueError: The classes, [0, 1], are not in class_weight`.
    `compute_sample_weight("balanced", y)` computes the identical per-sample weights
    (verified byte-identical predictions) without a user-facing dict, on every
    sklearn version.
    """
    return model.fit(features, y, sample_weight=compute_sample_weight("balanced", y))


@dataclass(frozen=True)
class MediaRef:
    path: Path
    member: str | None = None


def run_generic_image_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    random_state: int = 42,
) -> CandidateResult:
    return _run_generic_media_operator(
        data_dir,
        output_dir,
        task_id=task_id,
        operator="generic_image",
        final_artifact_source="ees_core:generic_image",
        suffixes=IMAGE_EXTS,
        extractor=_image_features,
        random_state=random_state,
    )


def run_generic_audio_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    random_state: int = 42,
) -> CandidateResult:
    return _run_generic_media_operator(
        data_dir,
        output_dir,
        task_id=task_id,
        operator="generic_audio",
        final_artifact_source="ees_core:generic_audio",
        suffixes=AUDIO_EXTS,
        extractor=_audio_features,
        random_state=random_state,
    )


def _run_generic_media_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    operator: str,
    final_artifact_source: str,
    suffixes: set[str],
    extractor: Callable[[Path], dict[str, float]],
    random_state: int,
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
        train, test = _load_media_metadata(data_dir, sample)
        id_col = sample.columns[0]
        target_cols = list(sample.columns[1:])
        train_target_cols = [col for col in train.columns if col not in test.columns and col != id_col]
        if not train_target_cols:
            raise ValueError("no target column found")
        target_col = train_target_cols[0]
        fit_train = _sample_fit_rows(train, target_col=target_col, operator=operator, random_state=random_state)
        media_paths = _media_path_index(data_dir, suffixes)
        train_features = _feature_frame(fit_train[id_col], media_paths, extractor)
        test_features = _feature_frame(test[id_col], media_paths, extractor)
        if train_features.empty or test_features.empty:
            raise ValueError("no usable media features extracted")
        submission, validation_predictions, validation_targets, validation = _fit_media_model(
            train=fit_train,
            test=test,
            sample=sample,
            target_col=target_col,
            train_features=train_features,
            test_features=test_features,
            n_estimators=_n_estimators(operator),
            random_state=random_state,
        )
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        if id_col not in target_cols:
            # One frame convention: both frames carry the id in its NATIVE dtype
            # and sort by that same dtype, so their row orders stay identical.
            validation_predictions = validation_predictions.sort_values(id_col, kind="stable").reset_index(drop=True)
            validation_targets = validation_targets.sort_values(id_col, kind="stable").reset_index(drop=True)
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=id_col,
            target_cols=target_cols,
            validation_kind=validation["validation_kind"],
        )
        metrics = {
            "operator": operator,
            "task_id": task_id,
            "target_col": target_col,
            "feature_count": int(len(train_features.columns)),
            "train_rows": int(len(train)),
            "fit_train_rows": int(len(fit_train)),
            "random_state": int(random_state),
            "test_rows": int(len(test)),
            "validation": validation,
            "prediction_artifacts": prediction_artifacts,
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id=f"{operator}_features",
            recipe_id=f"{operator}_features",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation.get("score"),
            medal="none",
            final_artifact_source=final_artifact_source,
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            score_direction=internal_metric_direction(validation.get("metric")),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=operator,
            recipe_id=f"{operator}_features",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source=final_artifact_source,
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _media_path_index(data_dir: Path, suffixes: set[str]) -> dict[str, MediaRef]:
    index: dict[str, MediaRef] = {}
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            ref = MediaRef(path=path)
            index[path.name] = ref
            index[path.stem] = ref
            index[str(path.relative_to(data_dir))] = ref
        elif path.is_file() and path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        if member.endswith("/") or Path(member).suffix.lower() not in suffixes:
                            continue
                        ref = MediaRef(path=path, member=member)
                        index[Path(member).name] = ref
                        index[Path(member).stem] = ref
                        index[member] = ref
            except zipfile.BadZipFile:
                continue
    return index


def _load_media_metadata(data_dir: Path, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (data_dir / "train.csv").exists():
        train = pd.read_csv(data_dir / "train.csv")
    elif (data_dir / "labels.csv").exists():
        train = pd.read_csv(data_dir / "labels.csv")
    else:
        raise FileNotFoundError(f"No train.csv or labels.csv found in {data_dir}")
    if (data_dir / "test.csv").exists():
        test = pd.read_csv(data_dir / "test.csv")
    else:
        test = pd.DataFrame({sample.columns[0]: sample.iloc[:, 0].astype(str)})
    return train, test


def _feature_frame(ids: pd.Series, media_paths: dict[str, MediaRef], extractor: Callable[[Path], dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    zip_cache: dict[Path, zipfile.ZipFile] = {}
    try:
        for value in ids.astype(str):
            ref = media_paths.get(value) or media_paths.get(Path(value).name) or media_paths.get(Path(value).stem)
            if ref is None:
                rows.append({"missing_media": 1.0})
                continue
            features = _extract_media_features(ref, extractor, zip_cache)
            features["missing_media"] = 0.0
            rows.append(features)
    finally:
        for handle in zip_cache.values():
            handle.close()
    return pd.DataFrame(rows).fillna(0.0)


def _sample_fit_rows(
    train: pd.DataFrame,
    *,
    target_col: str,
    operator: str,
    random_state: int,
) -> pd.DataFrame:
    limit = _max_train_rows(operator)
    if limit <= 0 or len(train) <= limit:
        return train.reset_index(drop=True)
    y = train[target_col]
    stratify = None
    value_counts = y.astype(str).value_counts(dropna=False)
    if 1 < len(value_counts) <= max(2, limit // 2) and int(value_counts.min()) >= 2:
        stratify = y.astype(str)
    try:
        sampled, _ = train_test_split(
            train,
            train_size=limit,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        sampled = train.sample(n=limit, random_state=random_state)
    return sampled.sort_index().reset_index(drop=True)


def _max_train_rows(operator: str) -> int:
    if operator == "generic_image":
        return _env_int("EES_IMAGE_MAX_TRAIN_ROWS", _env_int("EES_MEDIA_MAX_TRAIN_ROWS", 8000))
    if operator == "generic_audio":
        return _env_int("EES_AUDIO_MAX_TRAIN_ROWS", _env_int("EES_MEDIA_MAX_TRAIN_ROWS", 8000))
    return _env_int("EES_MEDIA_MAX_TRAIN_ROWS", 8000)


def _n_estimators(operator: str) -> int:
    if operator == "generic_image":
        return _env_int("EES_IMAGE_N_ESTIMATORS", _env_int("EES_MEDIA_N_ESTIMATORS", 192))
    if operator == "generic_audio":
        return _env_int("EES_AUDIO_N_ESTIMATORS", _env_int("EES_MEDIA_N_ESTIMATORS", 192))
    return _env_int("EES_MEDIA_N_ESTIMATORS", 192)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _extract_media_features(
    ref: MediaRef,
    extractor: Callable[[Path], dict[str, float]],
    zip_cache: dict[Path, zipfile.ZipFile],
) -> dict[str, float]:
    if ref.member is None:
        return extractor(ref.path)
    suffix = Path(ref.member).suffix.lower()
    zf = zip_cache.get(ref.path)
    if zf is None:
        zf = zipfile.ZipFile(ref.path)
        zip_cache[ref.path] = zf
    if suffix in IMAGE_EXTS:
        return _image_features_from_zip_handle(zf, ref.member)
    if suffix in AUDIO_EXTS:
        return _audio_features_from_zip_handle(zf, ref.member)
    return {}


def _fit_media_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_col: str,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    id_col = sample.columns[0]
    target_cols = list(sample.columns[1:])
    submission = sample.copy()
    submission[id_col] = test[id_col].to_numpy()
    if len(target_cols) > 1:
        labels = [str(col) for col in target_cols]
        y = train[target_col].astype(str)
        if y.nunique() > 1:
            model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
            _fit_balanced_classifier(model, train_features, y)
            probabilities = _aligned_probabilities(model, test_features, labels)
            # True full-train OOF validation (F.2): every train row scored by a
            # model that never saw it, so the artifacts are trusted evidence.
            validation_probabilities = _oof_classification_probabilities(
                train_features, y.to_numpy(), labels, n_estimators=n_estimators, random_state=random_state
            )
            metric_name = "oof_accuracy"
            validation_kind = "oof"
        else:
            # Degenerate single-class train: nothing is fitted/validated, the
            # constant prior is computed on train itself — keep the untrusted kind.
            probabilities = _constant_probs(len(test), labels, y.value_counts(normalize=True))
            validation_probabilities = _constant_probs(len(train), labels, y.value_counts(normalize=True))
            metric_name = "train_accuracy"
            validation_kind = "train_in_sample_audit"
        for index, label in enumerate(labels):
            submission[label] = probabilities[:, index]
        validation_predictions = pd.DataFrame(validation_probabilities, columns=labels)
        validation_predictions.insert(0, id_col, train[id_col].to_numpy())
        validation_targets = one_hot_frame(train[id_col], y, id_col=id_col, target_cols=labels)
        score = _classification_accuracy(y.to_numpy(), validation_probabilities, labels)
        return submission[list(sample.columns)], validation_predictions, validation_targets, {
            "metric": metric_name,
            "score": score,
            "validation_kind": validation_kind,
        }

    target = target_cols[0]
    y = train[target_col]
    if pd.api.types.is_numeric_dtype(y):
        if _is_binary_probability_target(y, sample[target]):
            positive_label = _positive_binary_label(y)
            if y.nunique(dropna=True) > 1:
                model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
                _fit_balanced_classifier(model, train_features, y)
                probabilities = _positive_class_probability(model, test_features, positive_label)
                validation_probabilities = _oof_positive_probabilities(
                    train_features,
                    y.to_numpy(),
                    positive_label,
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
                metric_name = "oof_log_loss"
                validation_kind = "oof"
            else:
                positive_rate = float((y == positive_label).mean())
                probabilities = np.full(len(test), positive_rate, dtype=float)
                validation_probabilities = np.full(len(train), positive_rate, dtype=float)
                metric_name = "train_log_loss"
                validation_kind = "train_in_sample_audit"
            submission[target] = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
            validation_predictions = pd.DataFrame({id_col: train[id_col].to_numpy(), target: np.clip(validation_probabilities, 1e-6, 1.0 - 1e-6)})
            binary_targets = (y == positive_label).astype(int).to_numpy()
            validation_targets = pd.DataFrame({id_col: train[id_col].to_numpy(), target: binary_targets})
            score = float(log_loss(binary_targets, np.clip(validation_probabilities, 1e-6, 1.0 - 1e-6), labels=[0, 1]))
            return submission[list(sample.columns)], validation_predictions, validation_targets, {
                "metric": metric_name,
                "score": score,
                "positive_label": str(positive_label),
                "validation_kind": validation_kind,
            }
        model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(train_features, y.astype(float))
        submission[target] = model.predict(test_features)
        oof_pred = _oof_regression_predictions(
            train_features,
            y.astype(float).to_numpy(),
            n_estimators=n_estimators,
            random_state=random_state,
        )
        validation_predictions = pd.DataFrame({id_col: train[id_col].to_numpy(), target: oof_pred})
        validation_targets = pd.DataFrame({id_col: train[id_col].to_numpy(), target: y.astype(float).to_numpy()})
        rmse = float(np.sqrt(np.mean((oof_pred - y.astype(float).to_numpy()) ** 2)))
        return submission[list(sample.columns)], validation_predictions, validation_targets, {
            "metric": "oof_rmse",
            "score": rmse,
            "validation_kind": "oof",
        }

    # String-label submission branch: predictions are class LABELS, not numeric
    # probabilities, so canonical metric scoring cannot recompute a score from
    # the artifacts. Marking these artifacts "oof" would admit an unscoreable
    # candidate into the trusted selection pool (where its raw self-score would
    # be compared against other operators' canonical scores). Intentionally kept
    # on the untrusted in-sample kind until label artifacts are canonically
    # scoreable.
    labels = sorted(str(value) for value in y.dropna().unique())
    if not labels:
        labels = [str(sample[target].iloc[0])]
    model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
    if len(labels) > 1:
        _fit_balanced_classifier(model, train_features, y.astype(str))
        probabilities = _aligned_probabilities(model, test_features, labels)
        train_probabilities = _aligned_probabilities(model, train_features, labels)
        predicted = [labels[int(np.argmax(row))] for row in probabilities]
    else:
        train_probabilities = _constant_probs(len(train), labels, y.value_counts(normalize=True))
        predicted = [labels[0] for _ in range(len(test))]
    submission[target] = predicted
    validation_predictions = pd.DataFrame({id_col: train[id_col].to_numpy(), target: [labels[int(np.argmax(row))] for row in train_probabilities]})
    validation_targets = pd.DataFrame({id_col: train[id_col].to_numpy(), target: y.astype(str).to_numpy()})
    return submission[list(sample.columns)], validation_predictions, validation_targets, {
        "metric": "train_accuracy",
        "score": float(np.mean(validation_predictions[target].astype(str).to_numpy() == y.astype(str).to_numpy())),
        "validation_kind": "train_in_sample_audit",
    }


def _classification_oof_folds(y: np.ndarray, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Full-train OOF folds: stratified when every class has >= 2 members."""
    string_labels = pd.Series(y).astype(str)
    counts = string_labels.value_counts(dropna=False)
    stratified = len(counts) > 1 and int(counts.min()) >= 2
    fold_labels = string_labels.to_numpy() if stratified else np.zeros(len(string_labels))
    n_splits = choose_n_splits(fold_labels, stratified=stratified)
    return make_folds(fold_labels, n_splits=n_splits, stratified=stratified, random_state=random_state)


def _oof_classification_probabilities(
    features: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    *,
    n_estimators: int,
    random_state: int,
) -> np.ndarray:
    fold_predictions = []
    for train_idx, valid_idx in _classification_oof_folds(y, random_state):
        model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
        _fit_balanced_classifier(model, features.iloc[train_idx], y[train_idx])
        fold_predictions.append((valid_idx, _aligned_probabilities(model, features.iloc[valid_idx], labels)))
    return assemble_oof(len(y), len(labels), fold_predictions)


def _oof_positive_probabilities(
    features: pd.DataFrame,
    y: np.ndarray,
    positive_label: object,
    *,
    n_estimators: int,
    random_state: int,
) -> np.ndarray:
    fold_predictions = []
    for train_idx, valid_idx in _classification_oof_folds(y, random_state):
        model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
        _fit_balanced_classifier(model, features.iloc[train_idx], y[train_idx])
        fold_predictions.append((valid_idx, _positive_class_probability(model, features.iloc[valid_idx], positive_label)))
    return assemble_oof(len(y), 1, fold_predictions).ravel()


def _oof_regression_predictions(
    features: pd.DataFrame,
    y: np.ndarray,
    *,
    n_estimators: int,
    random_state: int,
) -> np.ndarray:
    fold_labels = np.zeros(len(y))
    n_splits = choose_n_splits(fold_labels, stratified=False)
    fold_predictions = []
    for train_idx, valid_idx in make_folds(fold_labels, n_splits=n_splits, stratified=False, random_state=random_state):
        model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(features.iloc[train_idx], y[train_idx])
        fold_predictions.append((valid_idx, model.predict(features.iloc[valid_idx])))
    return assemble_oof(len(y), 1, fold_predictions).ravel()


def _aligned_probabilities(model: ExtraTreesClassifier, features: pd.DataFrame, labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(features)
    probabilities = np.zeros((len(features), len(labels)), dtype=float)
    class_to_index = {str(label): index for index, label in enumerate(model.classes_)}
    for output_index, label in enumerate(labels):
        if label in class_to_index:
            probabilities[:, output_index] = raw[:, class_to_index[label]]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.where(row_sums == 0.0, 1.0, row_sums)


def _constant_probs(length: int, labels: list[str], frequencies: pd.Series) -> np.ndarray:
    row = np.array([float(frequencies.get(label, 0.0)) for label in labels], dtype=float)
    if row.sum() <= 0:
        row[:] = 1.0 / max(1, len(labels))
    else:
        row = row / row.sum()
    return np.tile(row, (length, 1))


def _is_binary_probability_target(y: pd.Series, sample_target: pd.Series) -> bool:
    if y.nunique(dropna=True) != 2:
        return False
    if not pd.api.types.is_numeric_dtype(sample_target):
        return False
    finite_sample = pd.to_numeric(sample_target, errors="coerce").dropna()
    if finite_sample.empty:
        return True
    return bool(((finite_sample >= 0.0) & (finite_sample <= 1.0)).all())


def _positive_binary_label(y: pd.Series) -> object:
    values = sorted(pd.to_numeric(y.dropna(), errors="coerce").dropna().unique())
    if values:
        return values[-1]
    labels = sorted(str(value) for value in y.dropna().unique())
    return labels[-1] if labels else 1


def _positive_class_probability(model: ExtraTreesClassifier, features: pd.DataFrame, positive_label: object) -> np.ndarray:
    raw = model.predict_proba(features)
    class_to_index = {str(label): index for index, label in enumerate(model.classes_)}
    positive_index = class_to_index.get(str(positive_label))
    if positive_index is None:
        return np.zeros(len(features), dtype=float)
    return raw[:, positive_index]


def _classification_accuracy(labels: np.ndarray, probabilities: np.ndarray, class_labels: list[str]) -> float:
    predicted = np.array([class_labels[int(np.argmax(row))] for row in probabilities])
    return float(np.mean(predicted == labels.astype(str)))


def _image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        return _image_features_from_pil(image)


def _image_features_from_zip(zip_path: Path, member: str) -> dict[str, float]:
    with zipfile.ZipFile(zip_path) as zf:
        return _image_features_from_zip_handle(zf, member)


def _image_features_from_zip_handle(zf: zipfile.ZipFile, member: str) -> dict[str, float]:
    with zf.open(member) as handle:
        with Image.open(handle) as image:
            return _image_features_from_pil(image)


def _image_features_from_pil(image: Image.Image) -> dict[str, float]:
    rgb = image.convert("RGB").resize((32, 32))
    arr = np.asarray(rgb, dtype=float) / 255.0
    height, width = arr.shape[:2]
    channels = arr.reshape(-1, 3)
    features = {
        "width": float(width),
        "height": float(height),
        "aspect": float(width / max(1, height)),
    }
    for index, name in enumerate(["red", "green", "blue"]):
        values = channels[:, index]
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        features[f"{name}_p10"] = float(np.quantile(values, 0.10))
        features[f"{name}_p50"] = float(np.quantile(values, 0.50))
        features[f"{name}_p90"] = float(np.quantile(values, 0.90))
        hist, _ = np.histogram(values, bins=8, range=(0.0, 1.0), density=False)
        hist = hist.astype(float) / max(1, int(hist.sum()))
        for bin_index, value in enumerate(hist):
            features[f"{name}_hist_{bin_index}"] = float(value)
    grayscale = arr.mean(axis=2)
    green_excess = arr[:, :, 1] - 0.5 * (arr[:, :, 0] + arr[:, :, 2])
    features["gray_mean"] = float(grayscale.mean())
    features["gray_std"] = float(grayscale.std())
    features["green_excess_mean"] = float(green_excess.mean())
    features["green_excess_positive_fraction"] = float((green_excess > 0.05).mean())
    features["edge_energy"] = float(np.mean(np.abs(np.diff(grayscale, axis=0))) + np.mean(np.abs(np.diff(grayscale, axis=1))))
    for row in range(4):
        for col in range(4):
            patch = arr[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8, :]
            for channel, name in enumerate(["red", "green", "blue"]):
                features[f"grid_{row}_{col}_{name}"] = float(patch[:, :, channel].mean())
    return features


def _audio_features(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.getnframes()
            sample_width = handle.getsampwidth()
            raw = handle.readframes(frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        samples = np.frombuffer(raw, dtype=dtype).astype(float)
        if channels > 1 and len(samples):
            samples = samples.reshape(-1, channels).mean(axis=1)
        duration = frames / max(1, frame_rate)
        return {
            "duration": float(duration),
            "frame_rate": float(frame_rate),
            "channels": float(channels),
            "rms": float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0,
            "zero_crossing_rate": _zero_crossing_rate(samples),
            "log_size": math.log1p(path.stat().st_size),
        }
    data = path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8).astype(float)
    return {
        "duration": 0.0,
        "frame_rate": 0.0,
        "channels": 0.0,
        "byte_mean": float(arr.mean()) if len(arr) else 0.0,
        "byte_std": float(arr.std()) if len(arr) else 0.0,
        "log_size": math.log1p(len(data)),
    }


def _audio_features_from_zip(zip_path: Path, member: str) -> dict[str, float]:
    with zipfile.ZipFile(zip_path) as zf:
        return _audio_features_from_zip_handle(zf, member)


def _audio_features_from_zip_handle(zf: zipfile.ZipFile, member: str) -> dict[str, float]:
    data = zf.read(member)
    arr = np.frombuffer(data, dtype=np.uint8).astype(float)
    return {
        "duration": 0.0,
        "frame_rate": 0.0,
        "channels": 0.0,
        "byte_mean": float(arr.mean()) if len(arr) else 0.0,
        "byte_std": float(arr.std()) if len(arr) else 0.0,
        "log_size": math.log1p(len(data)),
    }


def _zero_crossing_rate(samples: np.ndarray) -> float:
    if len(samples) < 2:
        return 0.0
    signs = np.signbit(samples)
    return float(np.mean(signs[1:] != signs[:-1]))
