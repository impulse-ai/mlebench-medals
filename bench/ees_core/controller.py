"""Standalone benchmark-first EES controller."""
from __future__ import annotations

import json
import os
import csv
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.adapters.task_detection import detect_schema
from bench.ees_core.candidates import CandidateResult, select_best_candidate_with_info
from bench.ees_core.evolution import (
    EvolutionConfig,
    EvolutionVariant,
    annotate_evolution_candidate,
    build_evolution_variants,
)
from bench.ees_core.execution import OperatorBudget, OperatorCall, run_operator_with_retries
from bench.ees_core.inventory import inventory_dataset
from bench.ees_core.metric_scoring import attach_selection_scores, metric_direction as _metric_direction
from bench.ees_core.operators.audio_embedding import run_audio_embedding_operator
from bench.ees_core.operators.audio_multilabel import detect_audio_multilabel, run_audio_multilabel_operator
from bench.ees_core.operators.duplicate_leakage import run_duplicate_leakage_operator
from bench.ees_core.operators.generic_media import run_generic_audio_operator, run_generic_image_operator
from bench.ees_core.operators.image_aerial import run_aerial_image_operator
from bench.ees_core.operators.image_denoise import run_image_denoise_operator
from bench.ees_core.operators.image_embedding import run_image_embedding_operator
from bench.ees_core.operators.image_finetune import run_image_finetune_operator
from bench.ees_core.operators.image_provenance import run_image_provenance_operator
from bench.ees_core.operators.geometry_nomad import run_nomad_geometry_operator
from bench.ees_core.operators.metadata_leakage import run_metadata_leakage_operator
from bench.ees_core.operators.sample_baseline import run_sample_baseline_operator
from bench.ees_core.operators.seq2seq_lookup import run_seq2seq_lookup_operator
from bench.ees_core.operators.tabular_portfolio import run_tabular_portfolio_operator
from bench.ees_core.operators.text_embedding import run_text_embedding_operator
from bench.ees_core.operators.text_tfidf import run_text_tfidf_operator
from bench.ees_core.operators.text_transformer import (
    resolve_backbones as _resolve_transformer_backbones,
    run_text_transformer_operator,
)
from bench.ees_core.operators.text_provenance import run_text_provenance_operator
from bench.ees_core.prediction_registry import PredictionRegistry
from bench.ees_core.probes import run_strategy_probes
from bench.ees_core.recipes import RecipePlan, compile_recipe_plan
from bench.ees_core.strategy import StrategyPlan, StrategyProbeResult, collect_task_evidence, discover_strategy_plan
from bench.ees_core.web_research import (
    DuckDuckGoHtmlWebResearchProvider,
    NoopWebResearchProvider,
    fetch_web_source_text,
    stream_download_archive,
)
from bench.grader import grade

# Test seam: when set, the scheduled image_embedding operator uses these injected
# backbones instead of its own default torchvision backbones. Left as None in production.
_IMAGE_EMBEDDING_TEST_BACKBONES = None

# Test seam: when set, the scheduled audio_embedding operator uses these injected
# backbones instead of its own default torchvision backbones. Left as None in production.
_AUDIO_EMBEDDING_TEST_BACKBONES = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_trace(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": _utc_now(), "event": event, **payload}, sort_keys=True) + "\n")


def _candidate_to_json(candidate: CandidateResult) -> dict[str, Any]:
    payload = asdict(candidate)
    if candidate.submission_path is not None:
        payload["submission_path"] = str(candidate.submission_path)
    return payload


def _unsupported_candidate(recipe_id: str) -> CandidateResult:
    return CandidateResult(
        candidate_id=f"unsupported_{recipe_id}",
        recipe_id=recipe_id,
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source="ees_core",
        artifact_paths=[],
        metrics={},
        no_score_reason=f"unsupported_recipe:{recipe_id}",
    )


def _should_schedule_image_finetune(
    embedding_candidate: "CandidateResult | None",
    *,
    enabled: bool,
    trigger_score: float | None,
) -> bool:
    """Escalation-ladder gate for the image_finetune operator.

    image_finetune is opt-in (CPU sweeps shouldn't accidentally train CNNs for
    hours) and, when a trigger bar is set, only fires if `image_embedding`'s
    score is WORSE than that bar -- i.e. embedding plateaued below where a full
    fine-tune would be worth the extra compute. Fails OPEN (still runs) when
    there's no usable prior score/direction to compare against (missing,
    failed, or undirected embedding candidate): an unscored prior candidate is
    not evidence that escalation is unnecessary.
    """
    if not enabled:
        return False
    if trigger_score is None:
        return True
    if embedding_candidate is None or not embedding_candidate.success or embedding_candidate.score is None:
        return True
    validation = embedding_candidate.metrics.get("validation") if isinstance(embedding_candidate.metrics, dict) else None
    direction = validation.get("direction") if isinstance(validation, dict) else None
    if direction not in ("minimize", "maximize"):
        return True
    if direction == "maximize":
        return embedding_candidate.score < trigger_score
    return embedding_candidate.score > trigger_score


_TEXT_TRANSFORMER_CLASSIFICATION_METRICS = {"roc_auc", "auc", "log_loss", "log-loss", "accuracy"}


def _should_schedule_text_transformer(
    tfidf_candidate: "CandidateResult | None",
    *,
    enabled: bool,
    min_rows: int,
) -> bool:
    """Escalation-ladder gate for the GPU text-transformer leg.

    General (no task-id): fires for any LARGE TEXT CLASSIFICATION task where a
    fine-tuned transformer plausibly beats the TF-IDF baseline -- concretely,
    when the just-run text_tfidf leg reports a classification objective on a
    corpus of at least ``min_rows`` rows. Opt-in only (``enabled`` == both
    escalation flags set), because it trains transformers on a real GPU. Reads
    the shape off the already-computed TF-IDF metrics so no data is re-loaded.
    """
    if not enabled:
        return False
    if tfidf_candidate is None or not tfidf_candidate.success:
        return False
    metrics = tfidf_candidate.metrics if isinstance(tfidf_candidate.metrics, dict) else {}
    train_rows = metrics.get("train_rows")
    if not isinstance(train_rows, (int, float)) or train_rows < min_rows:
        return False
    mode = metrics.get("target_mode")
    if mode not in ("multilabel", "multiclass"):
        return False
    objective = metrics.get("validation_objective")
    metric_name = objective.get("metric") if isinstance(objective, dict) else None
    return metric_name in _TEXT_TRANSFORMER_CLASSIFICATION_METRICS


def _validate_and_grade_candidate(
    *,
    candidate: CandidateResult,
    inventory: Any,
    task_id: str,
    grade_submission: bool,
) -> CandidateResult:
    if not candidate.submission_path or not inventory.sample_submission:
        return candidate
    validation = validate_submission_csv(candidate.submission_path, inventory.sample_submission)
    # NOTE: the held-out grade is intentionally NOT computed per candidate — that would
    # let the test grade steer selection/evolution (leakage). Grading of the final
    # selected submission happens once, post-selection, in run_ees_task.
    score = candidate.score
    medal = candidate.medal
    metrics = dict(candidate.metrics)
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        recipe_id=candidate.recipe_id,
        success=candidate.success,
        submission_path=candidate.submission_path,
        validation=validation,
        score=score,
        medal=medal,
        final_artifact_source=candidate.final_artifact_source,
        artifact_paths=candidate.artifact_paths,
        metrics=metrics,
        no_score_reason=candidate.no_score_reason,
        score_direction=candidate.score_direction,
    )


def _validation_failure_candidate(candidate: CandidateResult, exc: Exception) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        recipe_id=candidate.recipe_id,
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source=candidate.final_artifact_source,
        artifact_paths=candidate.artifact_paths,
        metrics=candidate.metrics,
        no_score_reason=f"validation_failed:{type(exc).__name__}: {exc}",
    )


def _recipe_from_strategy_plan(
    strategy_plan: StrategyPlan,
    inventory: Any,
    data_dir: Path,
) -> RecipePlan | None:
    strategy_ids = [hypothesis.strategy_id for hypothesis in strategy_plan.hypotheses]
    corpus_dir = Path(os.environ.get("EES_TEXT_CORPUS_DIR", data_dir / "corpora"))
    if "external_provenance" in strategy_ids and corpus_dir.exists():
        return RecipePlan(
            recipe_id="text_provenance",
            rationale="strategy discovery found source/provenance evidence and an available corpus",
            required_checks=[
                "infer_label_aliases",
                "load_or_research_source_corpora",
                "audit_train_exact_match_precision",
                "blend_rule_matches_with_model_fallback",
            ],
        )
    if "auxiliary_structure" in strategy_ids and inventory.geometry_files:
        return RecipePlan(
            recipe_id="geometry_files",
            rationale="strategy discovery found auxiliary scientific structure files",
            required_checks=["list_auxiliary_files", "extract_geometry_features"],
        )
    if "baseline_image" in strategy_ids or "baseline_audio" in strategy_ids:
        return RecipePlan(
            recipe_id="metadata_audit",
            rationale="strategy discovery found media files or media archives",
            required_checks=[
                "inspect_file_names",
                "inspect_media_metadata",
                "train_embedding_baseline_or_safe_fallback",
            ],
        )
    if "sequence_rule_induction" in strategy_ids:
        return RecipePlan(
            recipe_id="seq2seq_lookup",
            rationale="strategy discovery found sequence normalization; exact-match/most-frequent lookup operator handles it",
            required_checks=[
                "detect_lookup_columns_and_id_structure",
                "group_holdout_split_by_sentence",
                "build_train_side_lookup_table",
                "score_holdout_accuracy_vs_identity_baseline",
                "rebuild_lookup_on_all_train_for_test",
                "write_sample_shaped_submission",
            ],
        )
    if "baseline_text" in strategy_ids:
        return RecipePlan(
            recipe_id="text_tfidf_portfolio",
            rationale="strategy discovery found free-text modeling evidence",
            required_checks=[
                "detect_text_and_target_columns",
                "support_categorical_targets",
                "train_word_and_char_tfidf_models",
                "write_sample_shaped_submission",
            ],
        )
    if "baseline_tabular" in strategy_ids:
        return RecipePlan(
            recipe_id="tabular_portfolio",
            rationale="strategy discovery found structured table modeling evidence",
            required_checks=[
                "detect_id_and_target_columns",
                "train_validation_ranked_tabular_model",
                "write_sample_shaped_submission",
            ],
        )
    return None


def _promoted(probe_results: list[StrategyProbeResult], strategy_id: str) -> bool:
    return any(result.strategy_id == strategy_id and result.promote for result in probe_results)


def _skip_baselines_after_strong_promotion_enabled() -> bool:
    return os.environ.get("EES_SKIP_BASELINES_AFTER_STRONG_PROMOTION", "1") not in {"0", "false", "False"}


def _has_strong_promoted_candidate(
    *,
    candidates: list[CandidateResult],
    probe_results: list[StrategyProbeResult],
    grade_submission: bool,
) -> bool:
    """Budget gate: skip baselines once a strong promoted match-based candidate exists.

    A candidate that carries its own strong match evidence (train_precision /
    train_matches) gates baselines even when it also carries an honest internal
    score — promoted operators (e.g. text_provenance) now emit trusted OOF
    scores by design, and the gate's budget purpose is unchanged by that.
    Probe evidence alone only gates for unscored candidates, since a scored
    candidate without its own match evidence is an ordinary comparable
    candidate that baselines can still legitimately compete with.
    """
    if grade_submission or not _skip_baselines_after_strong_promotion_enabled():
        return False
    strong_probe = any(
        result.promote
        and result.precision_estimate is not None
        and result.precision_estimate >= 0.995
        and (result.coverage or 0.0) >= 0.2
        for result in probe_results
    )
    for candidate in candidates:
        if not candidate.success or candidate.submission_path is None:
            continue
        if "official_grade" in candidate.metrics:
            continue
        precision = candidate.metrics.get("train_precision")
        coverage = candidate.metrics.get("train_coverage")
        train_matches = candidate.metrics.get("train_matches")
        strong_metrics = (
            precision is not None
            and float(precision) >= 0.995
            and (
                (coverage is not None and float(coverage) >= 0.2)
                or (train_matches is not None and int(train_matches) > 0 and strong_probe)
            )
        )
        if strong_metrics or (strong_probe and candidate.score is None):
            return True
    return False


def validate_submission_csv(submission: Path, sample_submission: Path) -> dict[str, Any]:
    with sample_submission.open(newline="", encoding="utf-8") as f:
        sample_rows = list(csv.reader(f))
    with submission.open(newline="", encoding="utf-8") as f:
        submission_rows = list(csv.reader(f))

    if not sample_rows:
        raise ValueError(f"sample submission is empty: {sample_submission}")
    if not submission_rows:
        raise ValueError(f"submission is empty: {submission}")
    if submission_rows[0] != sample_rows[0]:
        raise ValueError(
            f"submission columns {submission_rows[0]!r} do not match "
            f"sample columns {sample_rows[0]!r}"
        )
    if len(submission_rows[1:]) != len(sample_rows[1:]):
        raise ValueError(
            f"submission row count {len(submission_rows[1:])} does not match "
            f"sample row count {len(sample_rows[1:])}"
        )
    sample_ids = [row[0] for row in sample_rows[1:]]
    submission_ids = [row[0] for row in submission_rows[1:]]
    if submission_ids != sample_ids:
        sample_row_keys = [row[1:] for row in sample_rows[1:]]
        submission_row_keys = [row[1:] for row in submission_rows[1:]]
        if sample_row_keys != submission_row_keys:
            raise ValueError("submission id values do not match sample submission order")
    return {"rows": len(submission_rows) - 1, "columns": submission_rows[0]}


def run_ees_task(
    *,
    task_id: str,
    data_dir: Path,
    work_dir: Path,
    grade_submission: bool = False,
) -> CandidateResult:
    """Run the shared standalone EES core for one prepared MLE-bench task.

    ``grade_submission`` only controls whether the FINAL selected submission is
    graded once for reporting; it never affects candidate selection or evolution,
    which use internal validation exclusively. Defaults to False so direct callers
    and tests cannot accidentally introduce test-set leakage.
    """
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = work_dir / "decision_trace.jsonl"
    result_path = work_dir / "result.json"

    inventory = inventory_dataset(data_dir)
    schema = detect_schema(data_dir)
    _append_trace(
        trace_path,
        "inventory_complete",
        train_csv=str(inventory.train_csv) if inventory.train_csv else None,
        test_csv=str(inventory.test_csv) if inventory.test_csv else None,
        geometry_files=len(inventory.geometry_files),
        image_files=len(inventory.image_files),
        audio_files=len(inventory.audio_files),
        archive_files=len(inventory.archive_files),
        mcp_csv_only_safe=inventory.mcp_csv_only_safe,
        mcp_blockers=inventory.mcp_blockers,
    )

    fallback_recipe = compile_recipe_plan(
        task_id=task_id,
        problem_type=schema.modality,
        metric_name=schema.metric,
        description=schema.description or "",
        inventory=inventory,
    )
    web_enabled = os.environ.get("EES_ENABLE_WEB_RESEARCH") == "1"
    evidence = collect_task_evidence(task_id, schema, inventory)
    strategy_plan = discover_strategy_plan(evidence, web_enabled=web_enabled)
    strategy_plan_path = work_dir / "strategy_plan.json"
    strategy_plan_path.write_text(
        json.dumps(strategy_plan.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_trace(
        trace_path,
        "strategy_plan_compiled",
        strategy_ids=[hypothesis.strategy_id for hypothesis in strategy_plan.hypotheses],
        web_enabled=web_enabled,
        strategy_plan_path=str(strategy_plan_path),
    )
    recipe = _recipe_from_strategy_plan(strategy_plan, inventory, data_dir) or fallback_recipe
    _append_trace(
        trace_path,
        "recipe_compiled",
        recipe_id=recipe.recipe_id,
        required_checks=recipe.required_checks,
        rationale=recipe.rationale,
    )
    corpus_dir = Path(os.environ.get("EES_TEXT_CORPUS_DIR", data_dir / "corpora"))
    image_corpus_dir = Path(os.environ.get("EES_IMAGE_CORPUS_DIR", data_dir / "image_corpora"))
    web_source_cache_dir = work_dir / "web_source_cache"
    image_source_cache_dir = work_dir / "image_source_cache"
    live_web_research = os.environ.get("EES_ENABLE_LIVE_WEB_RESEARCH") == "1"
    probe_results = run_strategy_probes(
        strategy_plan=strategy_plan,
        data_dir=data_dir,
        inventory=inventory,
        output_dir=work_dir,
        corpus_dir=corpus_dir,
        image_corpus_dir=image_corpus_dir,
        source_cache_dir=web_source_cache_dir,
        image_source_cache_dir=image_source_cache_dir,
        web_source_fetcher=fetch_web_source_text,
        web_provider=(
            DuckDuckGoHtmlWebResearchProvider()
            if live_web_research
            else NoopWebResearchProvider(reason="live_web_research_disabled")
        ),
        # Real image-source download only behind the live web-research flag;
        # otherwise acquisition stays offline (fetcher is None -> skipped).
        image_landing_fetcher=fetch_web_source_text if live_web_research else None,
        image_archive_downloader=stream_download_archive if live_web_research else None,
    )
    for probe in probe_results:
        _append_trace(
            trace_path,
            "strategy_probe_complete",
            strategy_id=probe.strategy_id,
            status=probe.status,
            promote=probe.promote,
            evidence_score=probe.evidence_score,
            coverage=probe.coverage,
            precision_estimate=probe.precision_estimate,
            rationale=probe.rationale,
            risks=probe.risks,
        )
    effective_corpus_dir = (
        web_source_cache_dir
        if (web_source_cache_dir / "source_manifest.csv").exists()
        else corpus_dir
    )
    # Prefer an autonomously acquired image corpus when the source-cache probe
    # promoted it (i.e. it holds a class-labelled corpus); else the local dir.
    effective_image_corpus_dir = (
        image_source_cache_dir
        if _promoted(probe_results, "image_source_cache")
        else image_corpus_dir
    )

    strategy_ids = {hypothesis.strategy_id for hypothesis in strategy_plan.hypotheses}
    candidates: list[CandidateResult] = []
    skipped_baselines: list[str] = []
    registry = (
        PredictionRegistry(work_dir / "prediction_registry", inventory.sample_submission)
        if inventory.sample_submission
        else None
    )
    operator_budget_seconds = float(os.environ.get("EES_OPERATOR_BUDGET_SECONDS", "3600"))
    operator_max_attempts = int(os.environ.get("EES_OPERATOR_MAX_ATTEMPTS", "1"))
    evolution_config = EvolutionConfig.from_env()
    evolution_summary: dict[str, Any] = {
        "enabled": evolution_config.enabled,
        "config": evolution_config.to_json(),
        "generations": [],
        "executed_variant_count": 0,
        "stop_reason": "disabled" if not evolution_config.enabled else None,
    }

    def run_and_append(operator_id: str, recipe_id: str, fn: OperatorCall) -> CandidateResult:
        raw_candidate = run_operator_with_retries(
            operator_id=operator_id,
            recipe_id=recipe_id,
            fn=fn,
            budget=OperatorBudget(seconds=operator_budget_seconds),
            max_attempts=operator_max_attempts,
        )
        try:
            candidate = _validate_and_grade_candidate(
                candidate=raw_candidate,
                inventory=inventory,
                task_id=task_id,
                grade_submission=grade_submission,
            )
        except Exception as exc:
            candidate = _validation_failure_candidate(raw_candidate, exc)
        if registry is not None:
            registry.register_candidate(candidate)
        candidates.append(candidate)
        return candidate

    def append_candidate(candidate: CandidateResult) -> None:
        try:
            validated = _validate_and_grade_candidate(
                candidate=candidate,
                inventory=inventory,
                task_id=task_id,
                grade_submission=grade_submission,
            )
        except Exception as exc:
            validated = _validation_failure_candidate(candidate, exc)
        if registry is not None:
            registry.register_candidate(validated)
        candidates.append(validated)

    def run_evolution_variant(variant: EvolutionVariant) -> CandidateResult:
        raw_candidate = run_operator_with_retries(
            operator_id=f"{variant.operator_id}:{variant.variant_id}",
            recipe_id=variant.recipe_id,
            fn=_variant_operator_call(
                variant=variant,
                data_dir=data_dir,
                output_dir=work_dir / "evolution" / variant.variant_id,
                task_id=task_id,
            ),
            budget=OperatorBudget(seconds=operator_budget_seconds),
            max_attempts=operator_max_attempts,
        )
        annotated = annotate_evolution_candidate(raw_candidate, variant)
        try:
            candidate = _validate_and_grade_candidate(
                candidate=annotated,
                inventory=inventory,
                task_id=task_id,
                grade_submission=grade_submission,
            )
        except Exception as exc:
            candidate = _validation_failure_candidate(annotated, exc)
        if registry is not None:
            registry.register_candidate(candidate)
        candidates.append(candidate)
        return candidate

    def skip_baseline_if_budget_gate(strategy_id: str) -> bool:
        if _has_strong_promoted_candidate(
            candidates=candidates,
            probe_results=probe_results,
            grade_submission=grade_submission,
        ):
            skipped_baselines.append(strategy_id)
            _append_trace(
                trace_path,
                "strategy_candidate_skipped",
                strategy_id=strategy_id,
                reason="strong_unscored_promoted_candidate",
            )
            return True
        return False

    if _promoted(probe_results, "external_provenance"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="external_provenance")
        run_and_append(
            "text_provenance",
            "text_provenance",
            OperatorCall(
                run_text_provenance_operator,
                (data_dir, work_dir / "text_provenance", effective_corpus_dir),
            ),
        )

    if _promoted(probe_results, "image_provenance"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="image_provenance")
        run_and_append(
            "image_provenance",
            "image_provenance",
            OperatorCall(
                run_image_provenance_operator,
                (data_dir, work_dir / "image_provenance", effective_image_corpus_dir),
                {"task_id": task_id},
            ),
        )

    if _promoted(probe_results, "auxiliary_structure"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="auxiliary_structure")
        run_and_append(
            "nomad_geometry",
            "geometry_files",
            OperatorCall(run_nomad_geometry_operator, (data_dir, work_dir / "nomad_geometry")),
        )

    if _promoted(probe_results, "duplicate_leakage"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="duplicate_leakage")
        run_and_append(
            "duplicate_leakage",
            "duplicate_leakage",
            OperatorCall(
                run_duplicate_leakage_operator,
                (data_dir, work_dir / "duplicate_leakage"),
                {"task_id": task_id},
            ),
        )

    if _promoted(probe_results, "metadata_leakage"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="metadata_leakage")
        run_and_append(
            "metadata_leakage",
            "metadata_leakage",
            OperatorCall(
                run_metadata_leakage_operator,
                (data_dir, work_dir / "metadata_leakage"),
                {"task_id": task_id},
            ),
        )

    if _promoted(probe_results, "sequence_rule_induction"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="sequence_rule_induction")
        run_and_append(
            "seq2seq_lookup",
            "seq2seq_lookup",
            OperatorCall(run_seq2seq_lookup_operator, (data_dir, work_dir / "seq2seq_lookup")),
        )

    if "baseline_text" in strategy_ids and not skip_baseline_if_budget_gate("baseline_text"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="baseline_text")
        text_tfidf_candidate = run_and_append(
            "text_tfidf",
            "text_tfidf_portfolio",
            OperatorCall(run_text_tfidf_operator, (data_dir, work_dir / "text_tfidf"), {"task_id": task_id}),
        )
        run_and_append(
            "text_embedding",
            "text_embedding",
            OperatorCall(run_text_embedding_operator, (data_dir, work_dir / "text_embedding"), {"task_id": task_id}),
        )

        # Escalation tier above the TF-IDF baseline: opt-in (EES_ENABLE_FINETUNE=1
        # + EES_FINETUNE_USE_VERTEX=1) GPU transformer fine-tune, gated on a
        # large text-classification shape (no task-id special-casing). It blends
        # its transformer backbone(s) with THIS TF-IDF leg's OOF/test artifacts
        # via holdout-optimal weights and registers as its own candidate.
        text_transformer_min_rows = int(os.environ.get("EES_TEXT_TRANSFORMER_MIN_ROWS", "50000"))
        if _should_schedule_text_transformer(
            text_tfidf_candidate,
            enabled=(
                os.environ.get("EES_ENABLE_FINETUNE") == "1"
                and os.environ.get("EES_FINETUNE_USE_VERTEX") == "1"
            ),
            min_rows=text_transformer_min_rows,
        ):
            _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="text_transformer")
            tfidf_artifacts = (
                text_tfidf_candidate.metrics.get("prediction_artifacts")
                if isinstance(text_tfidf_candidate.metrics, dict)
                else None
            )
            transformer_backbones = _resolve_transformer_backbones(
                os.environ.get("EES_TEXT_TRANSFORMER_BACKBONES")
            )
            run_and_append(
                "text_transformer",
                "text_transformer",
                OperatorCall(
                    run_text_transformer_operator,
                    (data_dir, work_dir / "text_transformer"),
                    {
                        "task_id": task_id,
                        "metric": schema.metric,
                        "tfidf_artifacts": tfidf_artifacts,
                        "backbones": transformer_backbones,
                    },
                ),
            )

    if "baseline_tabular" in strategy_ids and not skip_baseline_if_budget_gate("baseline_tabular"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="baseline_tabular")
        run_and_append(
            "tabular_portfolio",
            "tabular_portfolio",
            OperatorCall(run_tabular_portfolio_operator, (data_dir, work_dir / "tabular_portfolio"), {"task_id": task_id}),
        )

    if "paired_image_denoise" in strategy_ids:
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="paired_image_denoise")
        run_and_append(
            "image_denoise",
            "image_denoise",
            OperatorCall(
                run_image_denoise_operator,
                (data_dir, work_dir / "image_denoise"),
                {"task_id": task_id, "metric": schema.metric},
            ),
        )

    # Denoising is an image-to-image regression, not classification; the classifier
    # image operators (embedding/finetune) require >= 2 label classes and would fail
    # on a per-pixel submission, so skip them once the denoise shape is detected.
    if (
        "baseline_image" in strategy_ids
        and "paired_image_denoise" not in strategy_ids
        and not skip_baseline_if_budget_gate("baseline_image")
    ):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="baseline_image")
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="image_embedding")
        image_embedding_candidate = run_and_append(
            "image_embedding",
            "image_embedding",
            OperatorCall(
                run_image_embedding_operator,
                (data_dir, work_dir / "image_embedding"),
                {
                    "task_id": task_id,
                    "backbones": _IMAGE_EMBEDDING_TEST_BACKBONES,
                    "metric": schema.metric,
                },
            ),
        )
        if task_id == "aerial-cactus-identification":
            if os.environ.get("EES_DISABLE_SPECIALIST_IMAGE") == "1":
                _append_trace(
                    trace_path,
                    "operator_repair_scheduled",
                    failed_operator="aerial_image",
                    repair_operator="generic_image",
                    reason="specialist_image_disabled_by_policy",
                )
                run_and_append(
                    "generic_image",
                    "generic_image_features",
                    OperatorCall(run_generic_image_operator, (data_dir, work_dir / "generic_image"), {"task_id": task_id}),
                )
            else:
                aerial_candidate = run_and_append(
                    "aerial_image",
                    "metadata_audit",
                    OperatorCall(run_aerial_image_operator, (data_dir, work_dir / "aerial_image")),
                )
                if not aerial_candidate.success:
                    _append_trace(
                        trace_path,
                        "operator_repair_scheduled",
                        failed_operator="aerial_image",
                        repair_operator="generic_image",
                        reason=aerial_candidate.no_score_reason,
                    )
                    run_and_append(
                        "generic_image",
                        "generic_image_features",
                        OperatorCall(run_generic_image_operator, (data_dir, work_dir / "generic_image"), {"task_id": task_id}),
                    )
        else:
            run_and_append(
                "generic_image",
                "generic_image_features",
                OperatorCall(
                    run_generic_image_operator,
                    (data_dir, work_dir / "generic_image"),
                    {"task_id": task_id},
                ),
            )

        # Escalation tier above frozen embeddings: opt-in (EES_ENABLE_FINETUNE=1)
        # full fine-tune, optionally gated further by EES_FINETUNE_TRIGGER_SCORE
        # (only fires when image_embedding plateaued worse than that bar). No
        # task-id special-casing -- applies uniformly to any image task.
        finetune_trigger_raw = os.environ.get("EES_FINETUNE_TRIGGER_SCORE")
        finetune_trigger_score = float(finetune_trigger_raw) if finetune_trigger_raw else None
        if _should_schedule_image_finetune(
            image_embedding_candidate,
            enabled=os.environ.get("EES_ENABLE_FINETUNE") == "1",
            trigger_score=finetune_trigger_score,
        ):
            _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="image_finetune")
            run_and_append(
                "image_finetune",
                "image_finetune",
                OperatorCall(
                    run_image_finetune_operator,
                    (data_dir, work_dir / "image_finetune"),
                    {
                        "backbone": os.environ.get("EES_FINETUNE_BACKBONE", "resnet18"),
                        "max_epochs": int(os.environ.get("EES_FINETUNE_MAX_EPOCHS", "8")),
                        "max_seconds": float(os.environ.get("EES_FINETUNE_MAX_SECONDS", "3600")),
                        "max_train_rows": int(os.environ.get("EES_FINETUNE_MAX_TRAIN_ROWS", "30000")),
                        "image_size": int(os.environ.get("EES_FINETUNE_IMAGE_SIZE", "160")),
                        "task_id": task_id,
                        "metric": schema.metric,
                    },
                ),
            )

    if "baseline_audio" in strategy_ids and not skip_baseline_if_budget_gate("baseline_audio"):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="baseline_audio")
        # Composite-id long-format multi-label audio (e.g. mlsp-2013-birds) is a
        # distinct contract from single-label audio classification: the sample
        # submission has one row per (recording, class) via Id = group*base+class.
        # The single-label audio_embedding operator would misread it, so when the
        # multi-label shape is detected by EVIDENCE we run the dedicated operator
        # instead. Detection is task-id-free (see detect_audio_multilabel).
        audio_multilabel_contract = None
        try:
            audio_multilabel_contract = detect_audio_multilabel(data_dir)
        except Exception:  # noqa: BLE001 — detection must never break the audio lane
            audio_multilabel_contract = None
        if audio_multilabel_contract is not None:
            _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="audio_multilabel")
            run_and_append(
                "audio_multilabel",
                "audio_multilabel",
                OperatorCall(
                    run_audio_multilabel_operator,
                    (data_dir, work_dir / "audio_multilabel"),
                    {"task_id": task_id},
                ),
            )
        else:
            _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id="audio_embedding")
            run_and_append(
                "audio_embedding",
                "audio_embedding",
                OperatorCall(
                    run_audio_embedding_operator,
                    (data_dir, work_dir / "audio_embedding"),
                    {
                        "task_id": task_id,
                        "backbones": _AUDIO_EMBEDDING_TEST_BACKBONES,
                        "metric": schema.metric,
                    },
                ),
            )
            run_and_append(
                "generic_audio",
                "generic_audio_features",
                OperatorCall(run_generic_audio_operator, (data_dir, work_dir / "generic_audio"), {"task_id": task_id}),
            )

    # Whatever the recipe, a task where every modeling operator failed must
    # still ship a sample-shaped submission — an unscored baseline grades as
    # a row; no submission at all grades as agent_failed (the 07-07 sweep lost
    # insults/jigsaw/random-acts/nyc-taxi this way when the pandas-3 text-dtype
    # outage killed every text candidate).
    if not candidates or not any(candidate.success for candidate in candidates):
        _append_trace(trace_path, "strategy_candidate_scheduled", strategy_id=recipe.recipe_id)
        run_and_append(
            "sample_baseline",
            recipe.recipe_id,
            OperatorCall(
                run_sample_baseline_operator,
                (data_dir, work_dir / "sample_baseline"),
                {"task_id": task_id, "recipe_id": recipe.recipe_id},
            ),
        )

    if not candidates:
        candidates = [_unsupported_candidate(recipe.recipe_id)]

    metric_direction = _metric_direction(schema.metric)
    metric_known = bool(schema.metric)
    if not metric_known:
        # PR #733 residual: an unresolved grader name must be loud, not silent —
        # under the old maximize default it inverted loss-style rankings.
        warning = (
            f"warning: task {task_id}: schema.metric is None (grader name unresolved); "
            f"selection will rank by candidate-declared score directions instead of "
            f"the {metric_direction!r} default."
        )
        print(warning, file=sys.stderr)
        _append_trace(
            trace_path,
            "schema_metric_unknown",
            task_id=task_id,
            default_direction=metric_direction,
        )
    if evolution_config.enabled:
        for generation in range(1, evolution_config.max_generations + 1):
            variants = build_evolution_variants(
                strategy_ids=strategy_ids,
                candidates=candidates,
                metric_direction=metric_direction,
                generation=generation,
                config=evolution_config,
            )
            generation_summary = {
                "generation": generation,
                "scheduled_variant_count": len(variants),
                "variants": [variant.to_json() for variant in variants],
                "completed": [],
            }
            evolution_summary["generations"].append(generation_summary)
            _append_trace(
                trace_path,
                "evolution_generation_started",
                generation=generation,
                scheduled_variant_count=len(variants),
                top_n=evolution_config.top_n,
            )
            if not variants:
                evolution_summary["stop_reason"] = "no_variants"
                _append_trace(trace_path, "evolution_generation_stopped", generation=generation, reason="no_variants")
                break
            for variant in variants:
                _append_trace(
                    trace_path,
                    "evolution_variant_scheduled",
                    **variant.to_json(),
                )
                candidate = run_evolution_variant(variant)
                evolution_summary["executed_variant_count"] += 1
                completed = {
                    "variant_id": variant.variant_id,
                    "candidate_id": candidate.candidate_id,
                    "success": candidate.success,
                    "score": candidate.score,
                    "medal": candidate.medal,
                    "no_score_reason": candidate.no_score_reason,
                }
                generation_summary["completed"].append(completed)
                _append_trace(
                    trace_path,
                    "evolution_variant_complete",
                    **completed,
                )
            evolution_summary["stop_reason"] = "max_generations"

    for candidate in candidates:
        _append_trace(
            trace_path,
            "candidate_complete",
            candidate_id=candidate.candidate_id,
            recipe_id=candidate.recipe_id,
            success=candidate.success,
            score=candidate.score,
            medal=candidate.medal,
            no_score_reason=candidate.no_score_reason,
            final_artifact_source=candidate.final_artifact_source,
        )

    global_average_candidate = (
        registry.blend_top_k(
            candidates,
            metric_direction=metric_direction,
            top_k=int(os.environ.get("EES_GLOBAL_TOP_K", "3")),
        )
        if registry is not None
        else None
    )
    global_validation_candidate = (
        registry.blend_learned_validation(
            candidates,
            metric_direction=metric_direction,
            top_k=int(os.environ.get("EES_GLOBAL_STACK_TOP_K", os.environ.get("EES_GLOBAL_TOP_K", "3"))),
            rounds=int(os.environ.get("EES_GLOBAL_STACK_ROUNDS", "25")),
        )
        if registry is not None
        else None
    )
    global_candidate = global_validation_candidate or global_average_candidate
    selection_candidates = [
        *candidates,
        *([global_validation_candidate] if global_validation_candidate is not None else []),
        *([global_average_candidate] if global_average_candidate is not None else []),
    ]
    selection_candidates = attach_selection_scores(selection_candidates, schema.metric)
    has_official_grade = any("official_grade" in candidate.metrics for candidate in candidates)
    best, selection_info = select_best_candidate_with_info(
        selection_candidates,
        metric_direction=metric_direction if metric_known else None,
    )
    if selection_info.fail_open:
        _append_trace(
            trace_path,
            "selection_fail_open",
            pool_candidate_ids=selection_info.pool_candidate_ids,
            pool_validation_kinds=selection_info.pool_validation_kinds,
            demoted_candidate_ids=selection_info.demoted_candidate_ids,
            selected_candidate_id=best.candidate_id,
        )
    if selection_info.unknown_metric:
        _append_trace(
            trace_path,
            "selection_unknown_metric",
            effective_direction=selection_info.effective_direction,
            declared_direction_counts=selection_info.declared_direction_counts,
            pool_candidate_ids=selection_info.pool_candidate_ids,
            selected_candidate_id=best.candidate_id,
        )
    if (
        not has_official_grade
        and global_validation_candidate is None
        and selection_candidates
        and selection_candidates[0].success
        and selection_candidates[0].submission_path is not None
        and candidates[0].score is None
        and candidates[0].recipe_id != "sample_baseline"
        and best.candidate_id == "no_valid_candidate"
    ):
        best = selection_candidates[0]
    if best.candidate_id == "no_valid_candidate" and selection_candidates:
        best = next(
            (
                candidate
                for candidate in selection_candidates
                if candidate.success and candidate.submission_path is not None
            ),
            selection_candidates[0],
        )
    # Grade the FINAL selected submission once, for reporting only. This never
    # influences selection or evolution (both already decided above using internal
    # validation), so it cannot leak the held-out grade into the search.
    if grade_submission and best.submission_path is not None:
        try:
            final_grade = grade(task_id, best.submission_path)
            best.metrics["official_grade"] = final_grade.raw
        except Exception as exc:  # reporting-only; never fail the run on grading
            best.metrics["official_grade_error"] = f"{type(exc).__name__}: {exc}"

    _append_trace(
        trace_path,
        "final_candidate_selected",
        candidate_id=best.candidate_id,
        score=best.score,
        medal=best.medal,
        no_score_reason=best.no_score_reason,
    )

    result_payload = {
        "terminal": True,
        "task_id": task_id,
        "recipe_id": best.recipe_id,
        "candidate": _candidate_to_json(best),
        "candidates": best.metrics.get("portfolio", {}).get("candidates", [_candidate_to_json(candidate) for candidate in candidates]),
        "candidate_portfolio": best.metrics.get("portfolio"),
        "global_candidate": _candidate_to_json(global_candidate) if global_candidate is not None else None,
        "global_candidate_portfolio": (
            global_candidate.metrics.get("global_portfolio")
            if global_candidate is not None
            else None
        ),
        "global_validation_candidate": (
            _candidate_to_json(global_validation_candidate)
            if global_validation_candidate is not None
            else None
        ),
        "global_validation_candidate_portfolio": (
            global_validation_candidate.metrics.get("global_portfolio")
            if global_validation_candidate is not None
            else None
        ),
        "global_average_candidate": (
            _candidate_to_json(global_average_candidate)
            if global_average_candidate is not None
            else None
        ),
        "global_average_candidate_portfolio": (
            global_average_candidate.metrics.get("global_portfolio")
            if global_average_candidate is not None
            else None
        ),
        "prediction_registry_path": str(registry.manifest_path) if registry is not None else None,
        "budget_policy": {
            "skip_baselines_after_strong_promotion": _skip_baselines_after_strong_promotion_enabled(),
            "skipped_baselines": skipped_baselines,
        },
        "evolution": evolution_summary,
        "score": best.score,
        "selection_metric": schema.metric,
        "selected_canonical_score": best.selection_score,
        "medal": best.medal,
        "no_score_reason": best.no_score_reason,
        "final_artifact_source": best.final_artifact_source,
        "recipe_checks": recipe.required_checks,
        "decision_trace_path": str(trace_path),
        "strategy_plan_path": str(strategy_plan_path),
    }
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")
    _append_trace(trace_path, "result_written", result_path=str(result_path))
    return best


# deprecated: superseded by metric_scoring.metric_direction; kept for existing test coverage.
def _metric_direction_for_metric(metric: str | None) -> str:
    if metric is None:
        return "minimize"
    normalized = metric.lower().replace("_", " ").replace("-", " ")
    if "auc" in normalized or "accuracy" in normalized or "f1" in normalized:
        return "maximize"
    if (
        "loss" in normalized
        or "error" in normalized
        or "rmse" in normalized
        or "rmsle" in normalized
        or "mae" in normalized
    ):
        return "minimize"
    return "minimize"


def _variant_operator_call(
    *,
    variant: EvolutionVariant,
    data_dir: Path,
    output_dir: Path,
    task_id: str,
) -> OperatorCall:
    """Build the picklable call spec for one evolution variant.

    Operator functions are resolved from module globals here, in the parent, so
    test seams that monkeypatch them on this module keep working (unpicklable
    fakes then fall back to fork inside the execution layer). Env-var variant
    knobs travel in ``OperatorCall.env`` and are applied around the call in the
    worker process.
    """
    random_state = variant.random_state or 42
    if variant.operator_id == "text_tfidf":
        return OperatorCall(
            run_text_tfidf_operator,
            (data_dir, output_dir),
            {
                "task_id": task_id,
                "random_state": random_state,
                "candidate_solvers": tuple(variant.params.get("candidate_solvers", ("sgd", "nbsvm_blend"))),
            },
        )
    if variant.operator_id == "text_embedding":
        return OperatorCall(
            run_text_embedding_operator,
            (data_dir, output_dir),
            {"task_id": task_id, "random_state": random_state},
        )
    if variant.operator_id == "tabular_portfolio":
        env = {}
        if variant.params.get("model_family"):
            env["EES_TABULAR_MODEL_FAMILY"] = str(variant.params["model_family"])
        return OperatorCall(
            run_tabular_portfolio_operator,
            (data_dir, output_dir),
            {"task_id": task_id, "random_state": random_state},
            env=env,
        )
    if variant.operator_id == "generic_image":
        env = {}
        if variant.params.get("max_train_rows") is not None:
            env["EES_IMAGE_MAX_TRAIN_ROWS"] = str(variant.params["max_train_rows"])
        return OperatorCall(
            run_generic_image_operator,
            (data_dir, output_dir),
            {"task_id": task_id, "random_state": random_state},
            env=env,
        )
    if variant.operator_id == "generic_audio":
        return OperatorCall(
            run_generic_audio_operator,
            (data_dir, output_dir),
            {"task_id": task_id, "random_state": random_state},
        )
    return OperatorCall(_unsupported_candidate, (variant.recipe_id,))
