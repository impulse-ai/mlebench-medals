"""General strategy discovery for benchmark-first EES."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import pandas as pd

from bench.adapters.task_detection import TaskSchema
from bench.ees_core.inventory import DatasetInventory


@dataclass(frozen=True)
class TaskEvidence:
    task_id: str
    modality: str
    metric: str | None
    description: str
    text_col: str | None
    id_col: str
    target_cols: list[str]
    sample_submission_cols: list[str]
    n_train: int | None
    n_test: int | None
    has_geometry_files: bool
    has_image_files: bool
    has_audio_files: bool
    has_archives: bool
    hints: list[str]

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StrategyHypothesis:
    strategy_id: str
    priority: int
    rationale: str
    signals: list[str]
    required_capabilities: list[str]
    probe_budget_seconds: int
    operator_budget_seconds: int
    requires_web: bool = False
    requires_external_data: bool = False
    expected_artifacts: list[str] | None = None
    promotion_thresholds: dict[str, float] | None = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StrategyProbeResult:
    strategy_id: str
    status: str
    promote: bool
    evidence_score: float
    coverage: float | None
    precision_estimate: float | None
    rationale: str
    risks: list[str]
    artifact_paths: list[str]

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StrategyPlan:
    evidence: TaskEvidence
    hypotheses: list[StrategyHypothesis]
    web_enabled: bool

    def to_json(self) -> dict:
        return {
            "evidence": self.evidence.to_json(),
            "hypotheses": [hypothesis.to_json() for hypothesis in self.hypotheses],
            "web_enabled": self.web_enabled,
        }


def _description_text(schema: TaskSchema, inventory: DatasetInventory) -> str:
    if schema.description:
        return schema.description
    if inventory.description_md and Path(inventory.description_md).exists():
        return Path(inventory.description_md).read_text(encoding="utf-8", errors="ignore")
    return ""


def _fallback_text_col(schema: TaskSchema, inventory: DatasetInventory) -> str | None:
    if schema.text_col:
        return schema.text_col
    if not inventory.train_csv or not Path(inventory.train_csv).exists():
        return None
    try:
        sample = pd.read_csv(inventory.train_csv, nrows=200)
    except Exception:
        return None
    preferred_names = {"text", "comment", "comment_text", "excerpt", "sentence", "content"}
    for column in sample.columns:
        if str(column).lower() in preferred_names:
            return str(column)
    best_col: str | None = None
    best_words = 0.0
    for column in sample.columns:
        series = sample[column]
        if series.dtype != object and str(series.dtype) != "str":
            continue
        values = series.dropna().astype(str)
        if values.empty:
            continue
        mean_words = float(values.str.split().str.len().mean())
        mean_chars = float(values.str.len().mean())
        if mean_words >= 8 and mean_chars >= 40 and mean_words > best_words:
            best_col = str(column)
            best_words = mean_words
    return best_col


def _hints(
    text: str,
    schema: TaskSchema,
    inventory: DatasetInventory,
    text_col: str | None,
) -> list[str]:
    lowered = text.lower()
    hints: set[str] = set()
    patterns = {
        "public_domain": r"public[- ]domain",
        "author_target": r"author|written by|excerpt|fiction|stories",
        "external_data_allowed_hint": r"original version|data can be found|citation|paper|repository|github",
        "synthetic_data": r"synthetic|generated",
        "filename_labels": r"file name|filename|named according|label as part of",
        "scientific_structure": r"geometry|crystal|molecule|dft|materials|lattice|atoms",
        "metric_log_loss": r"logarithmic loss|log loss",
        "metric_auc": r"auc|roc",
    }
    for hint, pattern in patterns.items():
        if re.search(pattern, lowered):
            hints.add(hint)
    if (
        "text normalization" in lowered
        or "spoken forms" in lowered
        or (("token" in lowered or "sentence_id" in lowered) and "before" in lowered and "after" in lowered)
    ):
        hints.add("sequence_normalization")
    if text_col:
        hints.add("has_text_column")
    if _has_author_like_train_target(inventory, schema):
        hints.add("author_target")
    if len(schema.target_cols) > 1:
        hints.add("probability_columns")
    if inventory.geometry_files:
        hints.add("geometry_files")
    if inventory.image_files or inventory.image_archive_files:
        hints.add("image_files")
    if inventory.audio_files or inventory.audio_archive_files:
        hints.add("audio_files")
    if inventory.sample_submission is not None:
        try:
            from bench.ees_core.operators.image_denoise import detect_denoise_task

            if detect_denoise_task(inventory.sample_submission.parent) is not None:
                hints.add("paired_image_denoise")
        except Exception:
            pass
    return sorted(hints)


def _has_author_like_train_target(inventory: DatasetInventory, schema: TaskSchema) -> bool:
    if not inventory.train_csv or not Path(inventory.train_csv).exists():
        return False
    try:
        train = pd.read_csv(inventory.train_csv, nrows=50)
    except Exception:
        return False
    test_cols: set[str] = set()
    if inventory.test_csv and Path(inventory.test_csv).exists():
        try:
            test_cols = set(pd.read_csv(inventory.test_csv, nrows=1).columns)
        except Exception:
            test_cols = set()
    explicit_targets = [
        str(column)
        for column in train.columns
        if column not in test_cols and column != schema.id_col
    ]
    author_names = {"author", "authors", "writer", "writers", "source", "provenance"}
    return any(column.lower() in author_names for column in explicit_targets)


def collect_task_evidence(
    task_id: str,
    schema: TaskSchema,
    inventory: DatasetInventory,
) -> TaskEvidence:
    description = _description_text(schema, inventory)
    text_col = _fallback_text_col(schema, inventory)
    return TaskEvidence(
        task_id=task_id,
        modality=schema.modality,
        metric=schema.metric,
        description=description,
        text_col=text_col,
        id_col=schema.id_col,
        target_cols=list(schema.target_cols),
        sample_submission_cols=list(schema.sample_submission_cols),
        n_train=schema.n_train,
        n_test=schema.n_test,
        has_geometry_files=bool(inventory.geometry_files),
        has_image_files=bool(inventory.image_files or inventory.image_archive_files),
        has_audio_files=bool(inventory.audio_files or inventory.audio_archive_files),
        has_archives=bool(inventory.archive_files),
        hints=_hints(description, schema, inventory, text_col),
    )


def discover_strategy_plan(evidence: TaskEvidence, *, web_enabled: bool = False) -> StrategyPlan:
    hypotheses: list[StrategyHypothesis] = []
    hints = set(evidence.hints)

    if evidence.modality == "nlp" or evidence.text_col:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="baseline_text",
                priority=50,
                rationale="free-text column supports word/char text baselines",
                signals=sorted(
                    hints
                    & {
                        "has_text_column",
                        "metric_log_loss",
                        "metric_auc",
                        "probability_columns",
                    }
                ),
                required_capabilities=["text_tfidf"],
                probe_budget_seconds=5,
                operator_budget_seconds=900,
                expected_artifacts=["submission.csv", "metrics.json"],
            )
        )

    if evidence.modality == "tabular" and not evidence.text_col:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="baseline_tabular",
                priority=40,
                rationale="structured table supports tree-model portfolio",
                signals=[],
                required_capabilities=["tabular_portfolio"],
                probe_budget_seconds=5,
                operator_budget_seconds=1200,
                expected_artifacts=["submission.csv", "metrics.json"],
            )
        )

    if evidence.has_geometry_files or "scientific_structure" in hints:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="auxiliary_structure",
                priority=90,
                rationale="task evidence indicates auxiliary scientific structure can add signal",
                signals=sorted(hints & {"geometry_files", "scientific_structure"}),
                required_capabilities=["geometry_features"],
                probe_budget_seconds=20,
                operator_budget_seconds=1800,
                expected_artifacts=["submission.csv", "metrics.json", "feature_manifest.json"],
                promotion_thresholds={"mapped_file_rate": 0.5},
            )
        )

    provenance_signals = hints & {"public_domain", "author_target", "has_text_column"}
    if {"author_target", "has_text_column"}.issubset(provenance_signals):
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="external_provenance",
                priority=95,
                rationale="description links text labels to named public-domain sources",
                signals=sorted(provenance_signals),
                required_capabilities=["web_research", "text_provenance"],
                probe_budget_seconds=120,
                operator_budget_seconds=1200,
                requires_web=True,
                requires_external_data=True,
                expected_artifacts=["submission.csv", "source_manifest.csv", "metrics.json"],
                promotion_thresholds={"train_precision": 0.995, "train_coverage": 0.2},
            )
        )

    if "paired_image_denoise" in hints:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="paired_image_denoise",
                priority=85,
                rationale="per-pixel submission (image_row_col) plus paired dirty/clean training image dirs indicate an image-to-image denoising task",
                signals=["paired_image_denoise"],
                required_capabilities=["image_denoise"],
                probe_budget_seconds=10,
                operator_budget_seconds=1800,
                expected_artifacts=["submission.csv", "metrics.json"],
            )
        )

    if evidence.has_image_files:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="image_provenance",
                priority=95,
                rationale=(
                    "task images may originate from a public source dataset; "
                    "hash-matching against a class-labelled corpus can recover labels"
                ),
                signals=sorted(hints & {"image_files", "external_data_allowed_hint", "public_domain"}),
                required_capabilities=["image_provenance"],
                probe_budget_seconds=300,
                operator_budget_seconds=1800,
                requires_external_data=True,
                expected_artifacts=["submission.csv", "matches.csv", "metrics.json"],
                promotion_thresholds={"train_precision": 0.995, "train_coverage": 0.2},
            )
        )
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="baseline_image",
                priority=60,
                rationale="image files or archives are present",
                signals=sorted(hints & {"image_files", "filename_labels", "metric_auc", "metric_log_loss"}),
                required_capabilities=["image_metadata", "image_embedding"],
                probe_budget_seconds=20,
                operator_budget_seconds=1800,
                expected_artifacts=["submission.csv", "metrics.json"],
            )
        )
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="metadata_leakage",
                priority=70,
                rationale="media filenames or metadata may carry label signal",
                signals=sorted(hints & {"image_files", "filename_labels"}),
                required_capabilities=["metadata_audit"],
                probe_budget_seconds=20,
                operator_budget_seconds=0,
                expected_artifacts=["metadata_probe.json"],
                promotion_thresholds={"filename_label_rate": 0.2},
            )
        )

    if evidence.has_audio_files:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="baseline_audio",
                priority=60,
                rationale="audio files or archives are present",
                signals=sorted(hints & {"audio_files", "filename_labels", "metric_auc"}),
                required_capabilities=["audio_features"],
                probe_budget_seconds=20,
                operator_budget_seconds=1800,
                expected_artifacts=["submission.csv", "metrics.json"],
            )
        )

    if "sequence_normalization" in hints:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="sequence_rule_induction",
                priority=80,
                rationale="token-level before/after task supports copy-rate and rule induction probes",
                signals=["sequence_normalization"],
                required_capabilities=["sequence_rules"],
                probe_budget_seconds=30,
                operator_budget_seconds=1200,
                expected_artifacts=["submission.csv", "rules.json", "metrics.json"],
            )
        )

    if web_enabled and ("external_data_allowed_hint" in hints or "public_domain" in hints):
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="web_research",
                priority=30,
                rationale="description cites external sources or public data that may inform strategy",
                signals=sorted(hints & {"external_data_allowed_hint", "public_domain"}),
                required_capabilities=["web_research"],
                probe_budget_seconds=60,
                operator_budget_seconds=0,
                requires_web=True,
                expected_artifacts=["web_research_manifest.json"],
            )
        )

    if evidence.n_train and evidence.n_test:
        hypotheses.append(
            StrategyHypothesis(
                strategy_id="duplicate_leakage",
                priority=20,
                rationale="train/test overlap can reveal direct or near-direct labels",
                signals=[],
                required_capabilities=["duplicate_probe"],
                probe_budget_seconds=30,
                operator_budget_seconds=0,
                expected_artifacts=["duplicate_probe.json"],
                promotion_thresholds={"overlap_rate": 0.01},
            )
        )

    hypotheses.append(
        StrategyHypothesis(
            strategy_id="calibration",
            priority=10,
            rationale="probability metrics benefit from calibration and clipping",
            signals=sorted(hints & {"metric_log_loss", "metric_auc", "probability_columns"}),
            required_capabilities=["calibration"],
            probe_budget_seconds=5,
            operator_budget_seconds=120,
            expected_artifacts=["calibration_metrics.json"],
        )
    )

    hypotheses = sorted(hypotheses, key=lambda hypothesis: hypothesis.priority, reverse=True)
    return StrategyPlan(evidence=evidence, hypotheses=hypotheses, web_enabled=web_enabled)
