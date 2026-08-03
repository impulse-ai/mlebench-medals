"""Geometry-aware operator for NOMAD transparent conductor tasks."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
SPECIES = ["Al", "Ga", "In", "O"]
ATOMIC_Z = {"Al": 13, "Ga": 31, "In": 49, "O": 8}
ATOMIC_RADIUS = {"Al": 1.21, "Ga": 1.22, "In": 1.42, "O": 0.66}
PAIR_KEYS = [f"{left}_{right}" for i, left in enumerate(SPECIES) for right in SPECIES[i:]]


def extract_geometry_features(path: Path) -> dict[str, float]:
    """Extract lattice, atom-count, coordinate, and distance descriptors."""
    lattice: list[list[float]] = []
    atoms: list[str] = []
    coords: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "lattice_vector" and len(parts) >= 4:
            lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "atom" and len(parts) >= 5:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            atoms.append(parts[4])

    features: dict[str, float] = {}
    lattice_arr = np.asarray(lattice, dtype=float) if lattice else np.zeros((0, 3))
    coords_arr = np.asarray(coords, dtype=float) if coords else np.zeros((0, 3))
    counts = Counter(atoms)
    atom_count = len(atoms)

    if lattice_arr.shape == (3, 3):
        norms = np.linalg.norm(lattice_arr, axis=1)
        volume = float(abs(np.linalg.det(lattice_arr)))
        features.update(
            {
                "geom_volume": volume,
                "geom_lattice_norm_mean": float(norms.mean()),
                "geom_lattice_norm_std": float(norms.std()),
                "geom_lattice_norm_min": float(norms.min()),
                "geom_lattice_norm_max": float(norms.max()),
                "geom_density_atoms_per_vol": float(atom_count / volume) if volume else 0.0,
            }
        )
        for index, value in enumerate(lattice_arr.reshape(-1)):
            features[f"geom_lattice_vec_{index}"] = float(value)
    else:
        features.update(
            {
                "geom_volume": 0.0,
                "geom_lattice_norm_mean": 0.0,
                "geom_lattice_norm_std": 0.0,
                "geom_lattice_norm_min": 0.0,
                "geom_lattice_norm_max": 0.0,
                "geom_density_atoms_per_vol": 0.0,
            }
        )
        for index in range(9):
            features[f"geom_lattice_vec_{index}"] = 0.0

    for species in SPECIES:
        count = counts.get(species, 0)
        features[f"count_{species}"] = float(count)
        features[f"frac_{species}"] = float(count / atom_count) if atom_count else 0.0

    if atoms:
        z_values = np.asarray([ATOMIC_Z.get(atom, 0) for atom in atoms], dtype=float)
        radius_values = np.asarray([ATOMIC_RADIUS.get(atom, 0.0) for atom in atoms], dtype=float)
        features["atomic_z_mean"] = float(z_values.mean())
        features["atomic_z_std"] = float(z_values.std())
        features["atomic_radius_mean"] = float(radius_values.mean())
        features["atomic_radius_std"] = float(radius_values.std())
    else:
        features["atomic_z_mean"] = 0.0
        features["atomic_z_std"] = 0.0
        features["atomic_radius_mean"] = 0.0
        features["atomic_radius_std"] = 0.0

    if len(coords_arr):
        span = coords_arr.max(axis=0) - coords_arr.min(axis=0)
        mean = coords_arr.mean(axis=0)
        std = coords_arr.std(axis=0)
        for axis, name in enumerate(["x", "y", "z"]):
            features[f"coord_{name}_mean"] = float(mean[axis])
            features[f"coord_{name}_span"] = float(span[axis])
            features[f"coord_{name}_std"] = float(std[axis])
        features["coord_span_volume"] = float(np.prod(span))
    else:
        for name in ["x", "y", "z"]:
            features[f"coord_{name}_mean"] = 0.0
            features[f"coord_{name}_span"] = 0.0
            features[f"coord_{name}_std"] = 0.0
        features["coord_span_volume"] = 0.0

    distances: list[float] = []
    pair_distances: dict[str, list[float]] = {key: [] for key in PAIR_KEYS}
    if len(coords_arr) > 1:
        species_order = {species: index for index, species in enumerate(SPECIES)}
        for i in range(len(coords_arr) - 1):
            for j in range(i + 1, len(coords_arr)):
                distance = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
                distances.append(distance)
                left = atoms[i]
                right = atoms[j]
                if left in species_order and right in species_order:
                    key = "_".join(sorted([left, right], key=species_order.get))
                    pair_distances[key].append(distance)
    for key, values in pair_distances.items():
        if values:
            arr = np.asarray(values, dtype=float)
            features[f"dist_{key}_mean"] = float(arr.mean())
            features[f"dist_{key}_std"] = float(arr.std())
            features[f"dist_{key}_min"] = float(arr.min())
            features[f"dist_{key}_p10"] = float(np.quantile(arr, 0.10))
        else:
            features[f"dist_{key}_mean"] = 0.0
            features[f"dist_{key}_std"] = 0.0
            features[f"dist_{key}_min"] = 0.0
            features[f"dist_{key}_p10"] = 0.0
    if distances:
        dist = np.asarray(distances, dtype=float)
        features["dist_all_mean"] = float(dist.mean())
        features["dist_all_std"] = float(dist.std())
        features["dist_all_min"] = float(dist.min())
        features["dist_all_p10"] = float(np.quantile(dist, 0.10))
        features["dist_all_p90"] = float(np.quantile(dist, 0.90))
    else:
        for key in ["mean", "std", "min", "p10", "p90"]:
            features[f"dist_all_{key}"] = 0.0

    return features


def _build_features(data_dir: Path, split: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        extract_geometry_features(data_dir / split / str(int(row_id)) / "geometry.xyz")
        for row_id in frame["id"]
    ]
    geometry = pd.DataFrame(rows)
    base = frame.drop(columns=[col for col in TARGETS if col in frame.columns]).reset_index(drop=True)
    features = pd.concat([base, geometry], axis=1)
    if {"lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang"}.issubset(features.columns):
        features["cell_volume_csv"] = (
            features["lattice_vector_1_ang"]
            * features["lattice_vector_2_ang"]
            * features["lattice_vector_3_ang"]
        )
        features["atoms_per_csv_volume"] = features["number_of_total_atoms"] / features["cell_volume_csv"].replace(0, np.nan)
    if {"percent_atom_al", "percent_atom_ga", "percent_atom_in"}.issubset(features.columns):
        comp = features[["percent_atom_al", "percent_atom_ga", "percent_atom_in"]].clip(1e-9, 1)
        features["composition_entropy"] = -(comp * np.log(comp)).sum(axis=1)
        features["al_ga_ratio"] = features["percent_atom_al"] / (features["percent_atom_ga"] + 1e-6)
        features["al_in_ratio"] = features["percent_atom_al"] / (features["percent_atom_in"] + 1e-6)
        features["ga_in_ratio"] = features["percent_atom_ga"] / (features["percent_atom_in"] + 1e-6)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _mean_column_rmsle_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> tuple[float, list[float]]:
    per_column = np.sqrt(np.mean((y_pred_log - y_true_log) ** 2, axis=0))
    return float(per_column.mean()), [float(v) for v in per_column]


def _model_candidates(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "extra": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    }
    try:
        from lightgbm import LGBMRegressor

        models["lgbm"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=800,
                learning_rate=0.025,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=0.2,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        models["xgb"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=650,
                max_depth=3,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        models["cat"] = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=700,
                learning_rate=0.025,
                depth=5,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
        )
    except Exception:
        pass
    return models


def run_nomad_geometry_operator(
    data_dir: Path,
    output_dir: Path,
    random_state: int = 42,
) -> CandidateResult:
    """Train a geometry-aware tree portfolio and write a Nomad submission."""
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        x_train = _build_features(data_dir, "train", train)
        x_test = _build_features(data_dir, "test", test)
        feature_cols = [col for col in x_train.columns if col != "id"]
        y = np.log1p(train[TARGETS].values)

        if len(train) < 10:
            prediction = np.repeat(train[TARGETS].mean().values.reshape(1, -1), len(test), axis=0)
            validation_prediction = np.repeat(train[TARGETS].mean().values.reshape(1, -1), len(train), axis=0)
            score = None
            col_scores: list[float] = []
            members = ["mean_fallback"]
        else:
            models = _model_candidates(random_state)
            folds = min(5, max(2, len(train) // 20))
            kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
            oof: dict[str, np.ndarray] = {name: np.zeros_like(y) for name in models}
            test_log: dict[str, np.ndarray] = {name: np.zeros((len(test), len(TARGETS))) for name in models}
            scores: dict[str, dict[str, Any]] = {}
            for train_idx, valid_idx in kfold.split(x_train):
                for name, model in models.items():
                    import copy

                    fitted = copy.deepcopy(model)
                    fitted.fit(x_train.iloc[train_idx][feature_cols], y[train_idx])
                    oof[name][valid_idx] = np.asarray(fitted.predict(x_train.iloc[valid_idx][feature_cols]))
                    test_log[name] += np.asarray(fitted.predict(x_test[feature_cols])) / folds
            for name, pred in oof.items():
                model_score, model_cols = _mean_column_rmsle_log(y, pred)
                scores[name] = {"score": model_score, "col_scores": model_cols}
            ranked = sorted(scores, key=lambda key: scores[key]["score"])
            members = ranked[: min(3, len(ranked))]
            blended_oof = sum(oof[name] for name in members) / len(members)
            score, col_scores = _mean_column_rmsle_log(y, blended_oof)
            validation_prediction = np.clip(np.expm1(blended_oof), 0, None)
            prediction = np.clip(np.expm1(sum(test_log[name] for name in members) / len(members)), 0, None)

        pred_frame = pd.DataFrame(
            {
                "id": test["id"].astype(sample["id"].dtype, copy=False),
                TARGETS[0]: prediction[:, 0],
                TARGETS[1]: prediction[:, 1],
            }
        )
        submission = sample[["id"]].merge(pred_frame, on="id", how="left")
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        validation_predictions = pd.DataFrame(
            {
                "id": train["id"],
                TARGETS[0]: validation_prediction[:, 0],
                TARGETS[1]: validation_prediction[:, 1],
            }
        )
        validation_targets = train[["id", *TARGETS]].copy()
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col="id",
            target_cols=TARGETS,
            validation_kind="oof" if len(train) >= 10 else "train_mean_fallback_audit",
        )

        metrics = {
            "operator": "geometry_nomad",
            "metric": "mean-column-wise-rmsle",
            "oof_score": score,
            "oof_col_scores": col_scores,
            "members": members,
            "prediction_artifacts": prediction_artifacts,
            "feature_count": len(feature_cols),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id="nomad_geometry",
            recipe_id="geometry_files",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=score,
            medal="none",
            final_artifact_source="ees_core:geometry_nomad",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            score_direction="minimize",  # score is mean-column-wise RMSLE
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="nomad_geometry",
            recipe_id="geometry_files",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:geometry_nomad",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
