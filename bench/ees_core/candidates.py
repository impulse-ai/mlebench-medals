"""Candidate result contracts and final selection for benchmark-first EES."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


MetricDirection = Literal["minimize", "maximize"]

# Validation kinds whose scores were computed on data the model never trained on.
TRUSTED_VALIDATION_KINDS = {"holdout", "oof", "global_holdout", "global_oof"}

# Train-side audit/probe evidence: self-scores computed ON the training data
# (train_in_sample_audit, train_metadata_audit, train_duplicate_audit,
# train_mean_fallback_audit, ...). These are known-inflated and must never
# outrank other self-reported validation in a fail-open pool.
_TRAIN_SIDE_EVIDENCE_TOKENS = ("train", "audit", "probe")


def _rank_key(candidate: "CandidateResult") -> float:
    return candidate.selection_score if candidate.selection_score is not None else candidate.score


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    recipe_id: str
    success: bool
    submission_path: Path | None
    validation: dict[str, Any]
    score: float | None
    medal: str
    final_artifact_source: str
    artifact_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    selection_score: float | None = None
    no_score_reason: str | None = None
    # Direction of this candidate's OWN `score` as declared by the operator that
    # computed it ("minimize" for loss-style, "maximize" for auc/accuracy-style).
    # Used only when the task metric is unknown; None means "not declared".
    score_direction: MetricDirection | None = None

    @property
    def valid_for_selection(self) -> bool:
        return self.success and self.submission_path is not None and self.score is not None


@dataclass(frozen=True)
class CandidatePortfolio:
    recipe_id: str
    candidates: list[CandidateResult]
    metric_direction: MetricDirection
    selection_strategy: str
    top_k: int = 1

    @property
    def ranked_valid_candidates(self) -> list[CandidateResult]:
        reverse = self.metric_direction == "maximize"
        valid = [candidate for candidate in self.candidates if candidate.valid_for_selection]
        return sorted(valid, key=_rank_key, reverse=reverse)

    @property
    def selected(self) -> CandidateResult:
        ranked = self.ranked_valid_candidates
        if ranked:
            return ranked[0]
        return select_best_candidate(self.candidates, metric_direction=self.metric_direction)

    @property
    def top_k_candidates(self) -> list[CandidateResult]:
        return self.ranked_valid_candidates[: max(1, self.top_k)]

    def summary(self) -> dict[str, Any]:
        selected = self.selected
        return {
            "recipe_id": self.recipe_id,
            "selection_strategy": self.selection_strategy,
            "metric_direction": self.metric_direction,
            "candidate_count": len(self.candidates),
            "valid_candidate_count": len(self.ranked_valid_candidates),
            "selected_candidate_id": selected.candidate_id,
            "selected_score": selected.score,
            "selected_medal": selected.medal,
            "top_k": len(self.top_k_candidates),
            "top_k_candidate_ids": [candidate.candidate_id for candidate in self.top_k_candidates],
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "recipe_id": candidate.recipe_id,
                    "success": candidate.success,
                    "score": candidate.score,
                    "medal": candidate.medal,
                    "submission_path": str(candidate.submission_path) if candidate.submission_path else None,
                    "no_score_reason": candidate.no_score_reason,
                }
                for candidate in self.candidates
            ],
        }


@dataclass(frozen=True)
class SelectionInfo:
    """Structured evidence about how the final candidate was selected.

    ``fail_open`` — no candidate carried trusted (holdout/oof) validation, so the
    pool fell open to ALL valid candidates with train-side audit/probe evidence
    demoted below every other self-reported validation.
    ``unknown_metric`` — selection ran without a known task metric and ranked by
    the candidates' own declared ``score_direction``.
    """

    fail_open: bool = False
    unknown_metric: bool = False
    effective_direction: MetricDirection | None = None
    pool_candidate_ids: list[str] = field(default_factory=list)
    pool_validation_kinds: dict[str, int] = field(default_factory=dict)
    demoted_candidate_ids: list[str] = field(default_factory=list)
    declared_direction_counts: dict[str, int] = field(default_factory=dict)


def select_best_candidate(
    candidates: list[CandidateResult],
    metric_direction: MetricDirection | None,
) -> CandidateResult:
    best, _ = select_best_candidate_with_info(candidates, metric_direction=metric_direction)
    return best


def select_best_candidate_with_info(
    candidates: list[CandidateResult],
    metric_direction: MetricDirection | None,
) -> tuple[CandidateResult, SelectionInfo]:
    """Select the final candidate and report how the decision was made.

    Ordering rules:
    1. Only ``valid_for_selection`` candidates compete.
    2. Trusted-validation candidates (``TRUSTED_VALIDATION_KINDS``) form the pool
       when any exist. Otherwise selection FAILS OPEN to all valid candidates,
       partitioned by evidence quality: train-side audit/probe kinds rank below
       every other self-reported validation, regardless of raw score. Score
       ordering is preserved within each partition.
    3. With a known ``metric_direction``, candidates rank by score in that
       direction (declared ``score_direction`` is ignored — behavior unchanged).
    4. With ``metric_direction=None`` (unknown task metric), candidates rank by
       their own declared ``score_direction``: the majority declared direction
       wins (ties resolve to "minimize", because loss-style scores ranked under
       a maximize default are the known catastrophic failure mode); candidates
       declaring the majority direction rank first (by score in that direction),
       then undeclared candidates (in submission order — their scores are not
       orderable), then minority-direction candidates (by score in their own
       direction). If NO candidate declares a direction, the legacy maximize
       default applies to the whole pool.
    """
    unknown_metric = metric_direction is None
    valid = [candidate for candidate in candidates if candidate.valid_for_selection]
    if not valid:
        return (
            CandidateResult(
                candidate_id="no_valid_candidate",
                recipe_id="none",
                success=False,
                submission_path=None,
                validation={},
                score=None,
                medal="none",
                final_artifact_source="none",
                no_score_reason="no_valid_candidate",
            ),
            SelectionInfo(unknown_metric=unknown_metric),
        )

    pool, fail_open = _comparable_selection_pool(valid)
    demoted = [
        candidate.candidate_id
        for candidate in pool
        if fail_open and _is_train_side_evidence(_validation_kind(candidate))
    ]

    declared_counts: dict[str, int] = {}
    effective_direction: MetricDirection = metric_direction or "maximize"
    if unknown_metric:
        declared = [c.score_direction for c in pool if c.score_direction in ("minimize", "maximize")]
        declared_counts = {
            "minimize": declared.count("minimize"),
            "maximize": declared.count("maximize"),
            "undeclared": len(pool) - len(declared),
        }
        if declared:
            # Majority declared direction; tie-break to "minimize" (see docstring).
            effective_direction = (
                "minimize"
                if declared_counts["minimize"] >= declared_counts["maximize"]
                else "maximize"
            )

    use_declared_directions = unknown_metric and (
        declared_counts["minimize"] + declared_counts["maximize"] > 0
    )

    def _oriented(score: float, direction: MetricDirection) -> float:
        return score if direction == "minimize" else -score

    def _sort_key(candidate: CandidateResult) -> tuple[int, int, float]:
        evidence_rank = (
            1 if fail_open and _is_train_side_evidence(_validation_kind(candidate)) else 0
        )
        if not use_declared_directions:
            return (evidence_rank, 0, _oriented(_rank_key(candidate), effective_direction))
        if candidate.score_direction == effective_direction:
            return (evidence_rank, 0, _oriented(_rank_key(candidate), effective_direction))
        if candidate.score_direction is None:
            # Direction unknown: the score is not orderable against the pool.
            return (evidence_rank, 1, 0.0)
        return (evidence_rank, 2, _oriented(_rank_key(candidate), candidate.score_direction))

    best = sorted(pool, key=_sort_key)[0]
    pool_kinds: dict[str, int] = {}
    for candidate in pool:
        kind = _validation_kind(candidate) or "missing"
        pool_kinds[kind] = pool_kinds.get(kind, 0) + 1
    info = SelectionInfo(
        fail_open=fail_open,
        unknown_metric=unknown_metric,
        effective_direction=effective_direction,
        pool_candidate_ids=[candidate.candidate_id for candidate in pool],
        pool_validation_kinds=pool_kinds,
        demoted_candidate_ids=demoted,
        declared_direction_counts=declared_counts,
    )
    return best, info


def _is_train_side_evidence(kind: str | None) -> bool:
    if not kind:
        return False
    return any(token in kind for token in _TRAIN_SIDE_EVIDENCE_TOKENS)


def _comparable_selection_pool(
    candidates: list[CandidateResult],
) -> tuple[list[CandidateResult], bool]:
    # The held-out MLE-bench grade must never drive selection (test-set leakage).
    # Selection uses only internal validation evidence; the official grade is
    # computed once on the final selected submission for reporting (see run_ees_task).
    trusted_validation = [
        candidate
        for candidate in candidates
        if _validation_kind(candidate) in TRUSTED_VALIDATION_KINDS
    ]
    if trusted_validation:
        return trusted_validation, False
    # Fail-open: no trusted evidence anywhere. Return everything so the controller
    # can still submit, but flag it so callers can trace the degraded decision.
    return candidates, True


def _validation_kind(candidate: CandidateResult) -> str | None:
    artifacts = candidate.metrics.get("prediction_artifacts")
    if isinstance(artifacts, dict) and artifacts.get("validation_kind"):
        return str(artifacts["validation_kind"])
    validation = candidate.metrics.get("validation")
    if isinstance(validation, dict) and validation.get("validation_kind"):
        return str(validation["validation_kind"])
    global_portfolio = candidate.metrics.get("global_portfolio")
    if isinstance(global_portfolio, dict) and global_portfolio.get("validation_kind"):
        return str(global_portfolio["validation_kind"])
    return None
