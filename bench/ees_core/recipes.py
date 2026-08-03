"""Recipe routing for benchmark-first EES."""
from __future__ import annotations

from dataclasses import dataclass

from .inventory import DatasetInventory


@dataclass(frozen=True)
class RecipePlan:
    recipe_id: str
    rationale: str
    required_checks: list[str]
    blocks_generic_evolution: bool = True


def compile_recipe_plan(
    *,
    task_id: str,
    problem_type: str,
    metric_name: str | None,
    description: str,
    inventory: DatasetInventory,
) -> RecipePlan:
    """Compile a hard recipe plan from task metadata and file inventory."""
    text = " ".join([problem_type, metric_name or "", description or ""]).lower()

    if any(token in text for token in ["text normalization", "spoken forms", "before column", "after column"]):
        return RecipePlan(
            recipe_id="sample_baseline",
            rationale="sequence normalization operator is not implemented yet; submit sample-shaped baseline",
            required_checks=[
                "locate_zipped_sample_submission",
                "write_sample_shaped_submission",
                "record_no_model_baseline",
            ],
            blocks_generic_evolution=True,
        )

    if inventory.geometry_files or any(
        token in text for token in ["geometry", "xyz", "molecule", "conformer", "protein"]
    ):
        return RecipePlan(
            recipe_id="geometry_files",
            rationale="scientific task exposes per-sample structure files",
            required_checks=["list_auxiliary_files", "extract_geometry_features"],
        )

    if (
        inventory.image_files
        or inventory.audio_files
        or inventory.image_archive_files
        or inventory.audio_archive_files
        or any(
        token in text for token in ["image", "jpg", "jpeg", "png", "exif", "metadata"]
        )
    ):
        return RecipePlan(
            recipe_id="metadata_audit",
            rationale="media task requires file/metadata-aware handling before generic modeling",
            required_checks=[
                "inspect_file_names",
                "inspect_image_metadata",
                "train_embedding_baseline",
            ],
        )

    if any(
        token in text
        for token in [
            "toxic",
            "toxicity",
            "insult",
            "comment",
            "social commentary",
            "nlp",
            "free-text",
        ]
    ):
        return RecipePlan(
            recipe_id="text_tfidf_portfolio",
            rationale="free-text classification task benefits from enforced word/char TF-IDF baselines",
            required_checks=[
                "detect_text_and_target_columns",
                "train_word_and_char_tfidf_models",
                "save_validation_scores",
                "write_sample_shaped_submission",
            ],
            blocks_generic_evolution=True,
        )

    if any(
        token in text
        for token in ["author", "excerpt", "quote", "source text", "text provenance", "public-domain"]
    ):
        return RecipePlan(
            recipe_id="text_provenance",
            rationale="text task may benefit from source/provenance matching",
            required_checks=[
                "build_author_or_label_corpora",
                "audit_train_exact_match_precision",
                "sweep_match_thresholds",
                "save_match_manifest",
                "blend_rule_matches_with_model_fallback",
            ],
        )

    return RecipePlan(
        recipe_id="tabular_portfolio",
        rationale="no mandatory auxiliary-file recipe found",
        required_checks=[
            "run_fast_baseline",
            "audit_leakage_and_duplicates",
            "run_diverse_model_portfolio",
            "save_oof_and_test_arrays",
        ],
        blocks_generic_evolution=False,
    )
