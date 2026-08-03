"""Prediction artifact registry for cross-operator EES selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from bench.ees_core.candidates import CandidateResult, MetricDirection


@dataclass(frozen=True)
class PredictionRecord:
    candidate_id: str
    recipe_id: str
    score: float | None
    submission_path: Path
    prediction_path: Path
    id_col: str
    target_cols: list[str]
    row_count: int
    validation_prediction_path: Path | None = None
    validation_target_path: Path | None = None
    validation_row_count: int | None = None
    validation_kind: str | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["submission_path"] = str(self.submission_path)
        payload["prediction_path"] = str(self.prediction_path)
        if self.validation_prediction_path is not None:
            payload["validation_prediction_path"] = str(self.validation_prediction_path)
        if self.validation_target_path is not None:
            payload["validation_target_path"] = str(self.validation_target_path)
        return payload


class PredictionRegistry:
    def __init__(self, root_dir: Path, sample_submission: Path):
        self.root_dir = Path(root_dir)
        self.sample_submission = Path(sample_submission)
        self.arrays_dir = self.root_dir / "prediction_arrays"
        self.manifest_path = self.root_dir / "prediction_registry.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PredictionRecord] = {}
        self._load_existing()

    @property
    def records(self) -> list[PredictionRecord]:
        return list(self._records.values())

    def _load_existing(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for item in payload.get("records", []):
                record = PredictionRecord(
                    candidate_id=item["candidate_id"],
                    recipe_id=item["recipe_id"],
                    score=item.get("score"),
                    submission_path=Path(item["submission_path"]),
                    prediction_path=Path(item["prediction_path"]),
                    id_col=item["id_col"],
                    target_cols=list(item["target_cols"]),
                    row_count=int(item["row_count"]),
                    validation_prediction_path=(
                        Path(item["validation_prediction_path"])
                        if item.get("validation_prediction_path")
                        else None
                    ),
                    validation_target_path=(
                        Path(item["validation_target_path"])
                        if item.get("validation_target_path")
                        else None
                    ),
                    validation_row_count=item.get("validation_row_count"),
                    validation_kind=item.get("validation_kind"),
                )
                self._records[record.candidate_id] = record
        except Exception:
            self._records = {}

    def _write_manifest(self) -> None:
        payload = {"records": [record.to_json() for record in self.records]}
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def register_candidate(self, candidate: CandidateResult) -> PredictionRecord | None:
        if not candidate.success or candidate.submission_path is None or not Path(candidate.submission_path).exists():
            return None
        sample = pd.read_csv(self.sample_submission)
        submission = pd.read_csv(candidate.submission_path)
        if list(submission.columns) != list(sample.columns) or len(submission) != len(sample):
            return None
        id_col = sample.columns[0]
        if submission[id_col].astype(str).tolist() != sample[id_col].astype(str).tolist():
            return None
        prediction_path = self.arrays_dir / f"{candidate.candidate_id}.csv"
        submission.to_csv(prediction_path, index=False)
        (
            validation_prediction_path,
            validation_target_path,
            validation_row_count,
            validation_kind,
        ) = self._copy_validation_artifacts(candidate)
        record = PredictionRecord(
            candidate_id=candidate.candidate_id,
            recipe_id=candidate.recipe_id,
            score=candidate.score,
            submission_path=Path(candidate.submission_path),
            prediction_path=prediction_path,
            id_col=id_col,
            target_cols=list(sample.columns[1:]),
            row_count=int(len(sample)),
            validation_prediction_path=validation_prediction_path,
            validation_target_path=validation_target_path,
            validation_row_count=validation_row_count,
            validation_kind=validation_kind,
        )
        self._records[record.candidate_id] = record
        self._write_manifest()
        return record

    def _copy_validation_artifacts(
        self,
        candidate: CandidateResult,
    ) -> tuple[Path | None, Path | None, int | None, str | None]:
        artifacts = candidate.metrics.get("prediction_artifacts", {}) if candidate.metrics else {}
        prediction_source = artifacts.get("validation_prediction_path")
        target_source = artifacts.get("validation_target_path")
        if not prediction_source or not target_source:
            return None, None, None, None
        prediction_source_path = Path(prediction_source)
        target_source_path = Path(target_source)
        if not prediction_source_path.exists() or not target_source_path.exists():
            return None, None, None, None
        validation_dir = self.root_dir / "validation_arrays"
        validation_dir.mkdir(parents=True, exist_ok=True)
        prediction_dest = validation_dir / f"{candidate.candidate_id}_validation_predictions.csv"
        target_dest = validation_dir / f"{candidate.candidate_id}_validation_targets.csv"
        predictions = pd.read_csv(prediction_source_path)
        targets = pd.read_csv(target_source_path)
        if list(predictions.columns) != list(targets.columns) or len(predictions) != len(targets):
            return None, None, None, None
        predictions.to_csv(prediction_dest, index=False)
        targets.to_csv(target_dest, index=False)
        return prediction_dest, target_dest, int(len(predictions)), artifacts.get("validation_kind")

    def compatible_records(self, records: list[PredictionRecord] | None = None) -> list[PredictionRecord]:
        records = records or self.records
        if not records:
            return []
        first = records[0]
        compatible = []
        first_ids = pd.read_csv(first.prediction_path)[first.id_col].astype(str).tolist()
        for record in records:
            if record.target_cols != first.target_cols or record.id_col != first.id_col:
                continue
            ids = pd.read_csv(record.prediction_path)[record.id_col].astype(str).tolist()
            if ids == first_ids:
                compatible.append(record)
        return compatible

    def blend_top_k(
        self,
        candidates: list[CandidateResult],
        *,
        metric_direction: MetricDirection,
        top_k: int = 3,
    ) -> CandidateResult | None:
        score_by_id = {candidate.candidate_id: candidate.score for candidate in candidates}
        scored = [
            record
            for record in self.compatible_records()
            if score_by_id.get(record.candidate_id) is not None
        ]
        if len(scored) < 2:
            return None
        reverse = metric_direction == "maximize"
        ranked = sorted(scored, key=lambda record: float(score_by_id[record.candidate_id]), reverse=reverse)
        selected = ranked[: max(1, top_k)]
        selected = self.compatible_records(selected)
        if len(selected) < 2:
            return None

        frames = [pd.read_csv(record.prediction_path) for record in selected]
        blended = frames[0].copy()
        for target in selected[0].target_cols:
            blended[target] = sum(frame[target].astype(float) for frame in frames) / len(frames)
        output_dir = self.root_dir / "global_ensembles"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"global_ensemble_top{len(selected)}.csv"
        metrics_path = output_dir / f"global_ensemble_top{len(selected)}.json"
        blended.to_csv(output_path, index=False)
        scores = [float(score_by_id[record.candidate_id]) for record in selected]
        score = sum(scores) / len(scores)
        portfolio = {
            "selection_strategy": "global_top_k_average",
            "top_k": len(selected),
            "top_k_candidate_ids": [record.candidate_id for record in selected],
            "candidate_count": len(candidates),
            "compatible_scored_candidate_count": len(scored),
            "metric_direction": metric_direction,
            "score_proxy": score,
        }
        metrics_path.write_text(json.dumps({"global_portfolio": portfolio}, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id=f"global_ensemble_top{len(selected)}",
            recipe_id="global_ensemble",
            success=True,
            submission_path=output_path,
            validation={"rows": int(len(blended)), "columns": list(blended.columns)},
            score=score,
            medal="none",
            final_artifact_source="ees_core:global_ensemble",
            artifact_paths=[str(output_path), str(metrics_path)],
            metrics={"global_portfolio": portfolio},
            # Deliberately NO score_direction: the direction fed into this blend is
            # the controller's resolved task direction, which is exactly the blind
            # default when the task metric is unknown. Declaring it would launder
            # that default into the declared-direction majority vote in selection.
        )

    def blend_learned_validation(
        self,
        candidates: list[CandidateResult],
        *,
        metric_direction: MetricDirection,
        top_k: int = 5,
        rounds: int = 25,
    ) -> CandidateResult | None:
        eligible, ignored_counts = self._validation_stack_records(candidates)
        if len(eligible) < 2:
            return None
        selected, weights, loss, objective = self._learn_stack_weights(
            eligible[: max(2, top_k)],
            rounds=max(1, rounds),
        )

        frames = [pd.read_csv(record.prediction_path) for record in selected]
        blended = frames[0][[selected[0].id_col]].copy()
        for target in selected[0].target_cols:
            blended[target] = sum(
                weights[record.candidate_id] * frame[target].astype(float)
                for record, frame in zip(selected, frames)
            )
        if objective == "multiclass_log_loss":
            blended = self._normalize_probability_frame(blended, selected[0].id_col, selected[0].target_cols)

        output_dir = self.root_dir / "global_ensembles"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"global_validation_stack_top{len(selected)}.csv"
        metrics_path = output_dir / f"global_validation_stack_top{len(selected)}.json"
        blended.to_csv(output_path, index=False)
        score_proxy = -loss if metric_direction == "maximize" else loss
        portfolio = {
            "selection_strategy": "validation_weighted_stack",
            "top_k": len(selected),
            "top_k_candidate_ids": [record.candidate_id for record in selected],
            "weights": weights,
            "candidate_count": len(candidates),
            "compatible_validation_candidate_count": len(eligible),
            "eligible_validation_kinds": ["holdout", "oof"],
            "ignored_validation_kind_counts": ignored_counts,
            "validation_objective": objective,
            "validation_loss": loss,
            "metric_direction": metric_direction,
            "score_proxy": score_proxy,
        }
        metrics_path.write_text(json.dumps({"global_portfolio": portfolio}, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id=f"global_validation_stack_top{len(selected)}",
            recipe_id="global_ensemble",
            success=True,
            submission_path=output_path,
            validation={"rows": int(len(blended)), "columns": list(blended.columns)},
            score=score_proxy,
            medal="none",
            final_artifact_source="ees_core:global_validation_stack",
            artifact_paths=[str(output_path), str(metrics_path)],
            metrics={"global_portfolio": portfolio},
            # Deliberately NO score_direction: score_proxy is oriented to the
            # caller-resolved task direction, which is the blind default when the
            # task metric is unknown (see blend_top_k for the full rationale).
        )

    def _validation_stack_records(
        self,
        candidates: list[CandidateResult],
    ) -> tuple[list[PredictionRecord], dict[str, int]]:
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        records = [record for record in self.compatible_records() if record.candidate_id in candidate_ids]
        ignored_counts: dict[str, int] = {}
        eligible = []
        reference_ids: list[str] | None = None
        for record in records:
            if not record.validation_prediction_path or not record.validation_target_path:
                continue
            if record.validation_kind not in {"holdout", "oof"}:
                kind = record.validation_kind or "missing"
                ignored_counts[kind] = ignored_counts.get(kind, 0) + 1
                continue
            predictions = pd.read_csv(record.validation_prediction_path)
            targets = pd.read_csv(record.validation_target_path)
            if list(predictions.columns) != [record.id_col, *record.target_cols]:
                continue
            if list(targets.columns) != [record.id_col, *record.target_cols]:
                continue
            prediction_ids = predictions[record.id_col].astype(str).tolist()
            target_ids = targets[record.id_col].astype(str).tolist()
            if prediction_ids != target_ids:
                continue
            if reference_ids is None:
                reference_ids = prediction_ids
            if prediction_ids == reference_ids:
                eligible.append(record)
        ranked = sorted(
            eligible,
            key=lambda record: self._validation_loss(record)[0],
        )
        return ranked, ignored_counts

    def _learn_stack_weights(
        self,
        records: list[PredictionRecord],
        *,
        rounds: int,
    ) -> tuple[list[PredictionRecord], dict[str, float], float, str]:
        target_frame = pd.read_csv(records[0].validation_target_path)
        y_true = target_frame[records[0].target_cols].astype(float).to_numpy()
        objective = self._objective_name(y_true)
        validation_arrays = [
            pd.read_csv(record.validation_prediction_path)[record.target_cols].astype(float).to_numpy()
            for record in records
        ]
        ensemble = None
        counts = [0 for _ in records]
        loss = math.inf
        for round_index in range(rounds):
            best_index = 0
            best_loss = math.inf
            best_prediction = None
            for index, predictions in enumerate(validation_arrays):
                candidate_prediction = predictions if ensemble is None else (ensemble * round_index + predictions) / (round_index + 1)
                candidate_loss = self._loss(y_true, candidate_prediction, objective)
                if candidate_loss < best_loss:
                    best_index = index
                    best_loss = candidate_loss
                    best_prediction = candidate_prediction
            counts[best_index] += 1
            ensemble = best_prediction
            loss = best_loss
        weights = {
            record.candidate_id: count / rounds
            for record, count in zip(records, counts)
        }
        return records, weights, float(loss), objective

    def _validation_loss(self, record: PredictionRecord) -> tuple[float, str]:
        predictions = pd.read_csv(record.validation_prediction_path)[record.target_cols].astype(float).to_numpy()
        targets = pd.read_csv(record.validation_target_path)[record.target_cols].astype(float).to_numpy()
        objective = self._objective_name(targets)
        return self._loss(targets, predictions, objective), objective

    def _objective_name(self, targets: np.ndarray) -> str:
        if targets.ndim == 2 and targets.shape[1] > 1:
            row_sums = targets.sum(axis=1)
            values_are_binary = np.all((targets == 0.0) | (targets == 1.0))
            if values_are_binary and np.allclose(row_sums, 1.0):
                return "multiclass_log_loss"
        return "mean_squared_error"

    def _loss(self, targets: np.ndarray, predictions: np.ndarray, objective: str) -> float:
        if objective == "multiclass_log_loss":
            clipped = np.clip(predictions.astype(float), 1e-15, 1.0)
            row_sums = clipped.sum(axis=1, keepdims=True)
            clipped = clipped / np.where(row_sums == 0.0, 1.0, row_sums)
            return float(-np.sum(targets * np.log(clipped)) / len(targets))
        return float(np.mean((predictions.astype(float) - targets.astype(float)) ** 2))

    def _normalize_probability_frame(
        self,
        frame: pd.DataFrame,
        id_col: str,
        target_cols: list[str],
    ) -> pd.DataFrame:
        normalized = frame.copy()
        values = normalized[target_cols].astype(float).clip(lower=1e-15)
        row_sums = values.sum(axis=1).replace(0.0, 1.0)
        normalized[target_cols] = values.div(row_sums, axis=0)
        return normalized
