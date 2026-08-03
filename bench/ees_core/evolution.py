"""Budgeted evolutionary scheduling for benchmark-first EES."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from bench.ees_core.candidates import CandidateResult, MetricDirection


@dataclass(frozen=True)
class EvolutionConfig:
    enabled: bool = False
    max_generations: int = 1
    top_n: int = 3
    seeds: tuple[int, ...] = (101, 202, 303)
    max_variants_per_generation: int = 6
    text_solvers: tuple[str, ...] = ("sgd", "nbsvm_blend")
    tabular_families: tuple[str, ...] = ("lightgbm", "extra_trees")
    media_train_rows: tuple[int, ...] = (8000, 16000)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "EvolutionConfig":
        env = environ or os.environ
        return cls(
            enabled=_env_bool(env, "EES_EVOLUTION_ENABLED", False),
            max_generations=max(0, _env_int(env, "EES_EVOLUTION_MAX_GENERATIONS", 1)),
            top_n=max(1, _env_int(env, "EES_EVOLUTION_TOP_N", 3)),
            seeds=_env_int_tuple(env, "EES_EVOLUTION_SEEDS", (101, 202, 303)),
            max_variants_per_generation=max(1, _env_int(env, "EES_EVOLUTION_MAX_VARIANTS", 6)),
            text_solvers=_env_str_tuple(env, "EES_EVOLUTION_TEXT_SOLVERS", ("sgd", "nbsvm_blend")),
            tabular_families=_env_str_tuple(env, "EES_EVOLUTION_TABULAR_FAMILIES", ("lightgbm", "extra_trees")),
            media_train_rows=_env_int_tuple(env, "EES_EVOLUTION_MEDIA_TRAIN_ROWS", (8000, 16000)),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionVariant:
    variant_id: str
    operator_id: str
    recipe_id: str
    generation: int
    random_state: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    parent_candidate_ids: tuple[str, ...] = ()
    rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ranked_frontier(
    candidates: list[CandidateResult],
    *,
    metric_direction: MetricDirection,
    top_n: int,
) -> list[CandidateResult]:
    valid = [
        candidate
        for candidate in candidates
        if candidate.success and candidate.submission_path is not None and candidate.score is not None
    ]
    reverse = metric_direction == "maximize"
    ranked = sorted(valid, key=lambda candidate: float(candidate.score), reverse=reverse)
    if ranked:
        return ranked[: max(1, top_n)]
    successful = [
        candidate
        for candidate in candidates
        if candidate.success and candidate.submission_path is not None
    ]
    return successful[: max(1, top_n)]


def build_evolution_variants(
    *,
    strategy_ids: set[str],
    candidates: list[CandidateResult],
    metric_direction: MetricDirection,
    generation: int,
    config: EvolutionConfig,
) -> list[EvolutionVariant]:
    if not config.enabled or generation > config.max_generations:
        return []
    parents = ranked_frontier(candidates, metric_direction=metric_direction, top_n=config.top_n)
    parent_ids = tuple(candidate.candidate_id for candidate in parents)
    variants: list[EvolutionVariant] = []

    if "baseline_text" in strategy_ids:
        for seed in config.seeds:
            variants.append(
                EvolutionVariant(
                    variant_id=f"g{generation:02d}_text_tfidf_seed{seed}",
                    operator_id="text_tfidf",
                    recipe_id="text_tfidf_portfolio",
                    generation=generation,
                    random_state=seed,
                    params={"candidate_solvers": config.text_solvers},
                    parent_candidate_ids=parent_ids,
                    rationale="evolve text portfolio with a different validation split and solver portfolio",
                )
            )
            variants.append(
                EvolutionVariant(
                    variant_id=f"g{generation:02d}_text_embedding_seed{seed}",
                    operator_id="text_embedding",
                    recipe_id="text_embedding",
                    generation=generation,
                    random_state=seed,
                    parent_candidate_ids=parent_ids,
                    rationale="evolve text embedding baseline with a different SVD/validation seed",
                )
            )

    if "baseline_tabular" in strategy_ids:
        for seed in config.seeds:
            for family in config.tabular_families:
                variants.append(
                    EvolutionVariant(
                        variant_id=f"g{generation:02d}_tabular_{family}_seed{seed}",
                        operator_id="tabular_portfolio",
                        recipe_id="tabular_portfolio",
                        generation=generation,
                        random_state=seed,
                        params={"model_family": family},
                        parent_candidate_ids=parent_ids,
                        rationale="evolve tabular model family and random seed",
                    )
                )

    if "baseline_image" in strategy_ids:
        for seed in config.seeds:
            for max_rows in config.media_train_rows:
                variants.append(
                    EvolutionVariant(
                        variant_id=f"g{generation:02d}_image_rows{max_rows}_seed{seed}",
                        operator_id="generic_image",
                        recipe_id="generic_image_features",
                        generation=generation,
                        random_state=seed,
                        params={"max_train_rows": max_rows},
                        parent_candidate_ids=parent_ids,
                        rationale="evolve image feature model with a different sample cap and seed",
                    )
                )

    if "baseline_audio" in strategy_ids:
        for seed in config.seeds:
            variants.append(
                EvolutionVariant(
                    variant_id=f"g{generation:02d}_audio_seed{seed}",
                    operator_id="generic_audio",
                    recipe_id="generic_audio_features",
                    generation=generation,
                    random_state=seed,
                    parent_candidate_ids=parent_ids,
                    rationale="evolve audio feature model with a different seed",
                )
            )

    return variants[: config.max_variants_per_generation]


def annotate_evolution_candidate(candidate: CandidateResult, variant: EvolutionVariant) -> CandidateResult:
    metrics = dict(candidate.metrics)
    metrics["evolution"] = {
        **variant.to_json(),
        "base_candidate_id": candidate.candidate_id,
    }
    return CandidateResult(
        candidate_id=f"{candidate.candidate_id}__{variant.variant_id}",
        recipe_id=candidate.recipe_id,
        success=candidate.success,
        submission_path=candidate.submission_path,
        validation=candidate.validation,
        score=candidate.score,
        medal=candidate.medal,
        final_artifact_source=f"{candidate.final_artifact_source}:evolution:{variant.variant_id}",
        artifact_paths=candidate.artifact_paths,
        metrics=metrics,
        no_score_reason=candidate.no_score_reason,
        score_direction=candidate.score_direction,
    )


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_tuple(env: dict[str, str], name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = env.get(name)
    if value is None:
        return default
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue
    return tuple(parsed) or default


def _env_str_tuple(env: dict[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = env.get(name)
    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default
