"""Cheap evidence probes for EES strategy promotion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from bench.ees_core.inventory import DatasetInventory
from bench.ees_core.operators.image_provenance import (
    build_corpus_index,
    match_ids,
    normalize_class_name,
)
from bench.ees_core.operators.seq2seq_lookup import detect_lookup_structure, resolve_lookup_files
from bench.ees_core.operators.text_provenance import _load_corpus_index, _unique_author
from bench.ees_core.operators.text_tfidf import _is_text_dtype
from bench.ees_core.strategy import StrategyPlan, StrategyProbeResult
from bench.ees_core.web_research import (
    NoopWebResearchProvider,
    SourceCacheBuildResult,
    WebResearchProvider,
    acquire_image_source_corpus,
    acquire_label_source_texts,
    build_source_cache_from_web_manifest,
    fetch_gutendex_catalog,
    fetch_web_source_text,
    run_web_research_probe,
    stream_download_archive,
)


def _result(
    *,
    strategy_id: str,
    status: str,
    promote: bool,
    evidence_score: float,
    rationale: str,
    coverage: float | None = None,
    precision_estimate: float | None = None,
    risks: list[str] | None = None,
    artifact_paths: list[str] | None = None,
) -> StrategyProbeResult:
    return StrategyProbeResult(
        strategy_id=strategy_id,
        status=status,
        promote=promote,
        evidence_score=evidence_score,
        coverage=coverage,
        precision_estimate=precision_estimate,
        rationale=rationale,
        risks=risks or [],
        artifact_paths=artifact_paths or [],
    )


def _read_csv(path: Path | None, *, nrows: int | None = None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


def _hash_values(values: Iterable[object]) -> set[str]:
    hashed: set[str] = set()
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        hashed.add(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest())
    return hashed


def _corpus_available(corpus_dir: Path | None) -> bool:
    if corpus_dir is None or not Path(corpus_dir).exists():
        return False
    corpus_dir = Path(corpus_dir)
    if (corpus_dir / "source_manifest.csv").exists():
        return True
    return any(
        path.is_file() and path.suffix.lower() in {".txt", ".text", ".utf-8"}
        for path in corpus_dir.rglob("*")
    )


def _image_corpus_available(corpus_dir: Path | None) -> bool:
    """True when a class-labelled image corpus already exists on disk (an image
    whose immediate parent is a class subdirectory, not the corpus root)."""
    if corpus_dir is None or not Path(corpus_dir).is_dir():
        return False
    corpus_dir = Path(corpus_dir)
    if (corpus_dir / "source_manifest.csv").exists():
        return True
    from bench.adapters.task_detection import IMAGE_EXTS

    for path in corpus_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parent = path.parent
        if parent == corpus_dir or parent.name.startswith("."):
            continue
        return True
    return False


def _probe_submission_contract(inventory: DatasetInventory) -> StrategyProbeResult:
    sample = _read_csv(inventory.sample_submission)
    if sample is None:
        return _result(
            strategy_id="submission_contract",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale="sample submission is missing or unreadable",
            risks=["missing_sample_submission"],
        )
    return _result(
        strategy_id="submission_contract",
        status="success",
        promote=True,
        evidence_score=1.0,
        coverage=1.0,
        rationale=f"sample submission has {len(sample)} rows and {len(sample.columns)} columns",
    )


def _probe_calibration(strategy_ids: set[str], metric: str | None) -> StrategyProbeResult | None:
    if "calibration" not in strategy_ids:
        return None
    metric_text = (metric or "").lower()
    promote = any(token in metric_text for token in ["log", "auc", "roc"])
    return _result(
        strategy_id="calibration",
        status="success",
        promote=promote,
        evidence_score=1.0 if promote else 0.2,
        rationale="metric rewards probability quality" if promote else "metric does not clearly require calibration",
    )


def _probe_duplicate_leakage(strategy_plan: StrategyPlan, inventory: DatasetInventory) -> StrategyProbeResult:
    train = _read_csv(inventory.train_csv)
    test = _read_csv(inventory.test_csv)
    if train is None or test is None:
        return _result(
            strategy_id="duplicate_leakage",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="train or test csv unavailable",
            risks=["missing_train_or_test_csv"],
        )

    text_col = strategy_plan.evidence.text_col
    candidate_cols = [text_col] if text_col and text_col in train.columns and text_col in test.columns else []
    if not candidate_cols:
        candidate_cols = [
            col
            for col in train.columns
            if col in test.columns and col != strategy_plan.evidence.id_col and _is_text_dtype(train[col])
        ][:3]
    if not candidate_cols:
        return _result(
            strategy_id="duplicate_leakage",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="no shared text/string columns for duplicate probe",
        )

    train_hashes: set[str] = set()
    test_hashes: set[str] = set()
    for col in candidate_cols:
        train_hashes.update(_hash_values(train[col].dropna()))
        test_hashes.update(_hash_values(test[col].dropna()))
    overlap = train_hashes & test_hashes
    coverage = len(overlap) / max(1, len(test_hashes))
    promote = coverage >= 0.01
    return _result(
        strategy_id="duplicate_leakage",
        status="success",
        promote=promote,
        evidence_score=coverage,
        coverage=coverage,
        rationale=f"found {len(overlap)} overlapping values across {candidate_cols}",
    )


def _probe_auxiliary_structure(strategy_ids: set[str], strategy_plan: StrategyPlan, inventory: DatasetInventory) -> StrategyProbeResult | None:
    if "auxiliary_structure" not in strategy_ids:
        return None
    train = _read_csv(inventory.train_csv)
    test = _read_csv(inventory.test_csv)
    id_col = strategy_plan.evidence.id_col
    if train is None or test is None or id_col not in train.columns or id_col not in test.columns:
        return _result(
            strategy_id="auxiliary_structure",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="cannot map auxiliary files without train/test ids",
            risks=["missing_ids"],
        )
    expected_ids = {str(value) for value in train[id_col].tolist() + test[id_col].tolist()}
    mapped_ids = {path.parent.name for path in inventory.geometry_files}
    coverage = len(expected_ids & mapped_ids) / max(1, len(expected_ids))
    promote = coverage >= 0.5
    return _result(
        strategy_id="auxiliary_structure",
        status="success",
        promote=promote,
        evidence_score=coverage,
        coverage=coverage,
        rationale=f"geometry files map to {len(expected_ids & mapped_ids)}/{len(expected_ids)} row ids",
        risks=[] if promote else ["low_auxiliary_mapping_rate"],
    )


def _probe_metadata_leakage(strategy_ids: set[str], strategy_plan: StrategyPlan, inventory: DatasetInventory) -> StrategyProbeResult | None:
    if "metadata_leakage" not in strategy_ids:
        return None
    train = _read_csv(inventory.train_csv)
    test = _read_csv(inventory.test_csv)
    if train is None:
        return _result(
            strategy_id="metadata_leakage",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="train csv unavailable for filename metadata probe",
            risks=["missing_train_csv"],
        )
    id_col = strategy_plan.evidence.id_col
    target_cols = [col for col in strategy_plan.evidence.target_cols if col in train.columns]
    if not target_cols:
        test_cols = set(test.columns) if test is not None else set()
        explicit = [col for col in train.columns if col != id_col and col not in test_cols]
        if len(explicit) == 1:
            target_cols = explicit
    if id_col not in train.columns or not target_cols:
        return _result(
            strategy_id="metadata_leakage",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="cannot compare filenames with labels without id and target columns",
            risks=["missing_id_or_target"],
        )
    target_col = target_cols[0]
    if pd.api.types.is_numeric_dtype(train[target_col]):
        return _result(
            strategy_id="metadata_leakage",
            status="success",
            promote=False,
            evidence_score=0.0,
            rationale="numeric target values are not semantic filename labels",
            risks=["numeric_target_labels"],
        )
    rows = train[[id_col, target_col]].dropna()
    if rows.empty:
        return _result(
            strategy_id="metadata_leakage",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="no non-null filename/label pairs",
        )
    hits = 0
    for _, row in rows.iterrows():
        filename = str(row[id_col]).lower()
        label = str(row[target_col]).lower()
        if label and label in filename:
            hits += 1
    coverage = hits / len(rows)
    promote = coverage >= 0.2
    return _result(
        strategy_id="metadata_leakage",
        status="success",
        promote=promote,
        evidence_score=coverage,
        coverage=coverage,
        rationale=f"filename labels matched {hits}/{len(rows)} training rows",
        risks=[] if promote else ["low_filename_label_signal"],
    )


def _probe_provenance(
    strategy_ids: set[str],
    strategy_plan: StrategyPlan,
    inventory: DatasetInventory,
    corpus_dir: Path | None,
) -> StrategyProbeResult | None:
    if "external_provenance" not in strategy_ids:
        return None
    if corpus_dir is None or not Path(corpus_dir).exists():
        return _result(
            strategy_id="external_provenance",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="no corpus directory available for provenance probe",
            risks=["missing_corpus_dir"],
        )
    train = _read_csv(inventory.train_csv)
    if train is None or "text" not in train.columns or "author" not in train.columns:
        return _result(
            strategy_id="external_provenance",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="train csv lacks text/author columns for provenance audit",
            risks=["unsupported_provenance_schema"],
        )
    shingle_words = 8
    corpus_index = _load_corpus_index(Path(corpus_dir), shingle_words=shingle_words)
    corpora = corpus_index.corpora
    if not any(corpora.values()):
        return _result(
            strategy_id="external_provenance",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale="corpus directory loaded no text",
            risks=["empty_corpora"],
        )
    matches = 0
    wrong = 0
    for row in train.itertuples(index=False):
        predicted = _unique_author(
            getattr(row, "text"),
            corpora,
            min_chars=20,
            shingle_sets=corpus_index.shingle_sets,
            shingle_words=shingle_words,
        )
        if predicted is None:
            continue
        matches += 1
        wrong += int(predicted != getattr(row, "author"))
    coverage = matches / max(1, len(train))
    precision = 1.0 - wrong / matches if matches else None
    promote = matches > 0 and precision is not None and precision >= 0.995 and coverage >= 0.2
    return _result(
        strategy_id="external_provenance",
        status="success",
        promote=promote,
        evidence_score=(precision or 0.0) * coverage,
        coverage=coverage,
        precision_estimate=precision,
        rationale=f"matched {matches}/{len(train)} train rows with {wrong} wrong matches",
        risks=[] if promote else ["low_provenance_coverage_or_precision"],
    )


# Precision needs enough matched samples to clear the 0.995 bar with confidence,
# but hashing every train image at probe time is wasteful for very large tasks:
# a deterministic subsample bounds probe cost while keeping the estimate honest.
IMAGE_PROVENANCE_PROBE_MAX_IMAGES = 2000


def _probe_image_provenance(
    strategy_ids: set[str],
    inventory: DatasetInventory,
    image_corpus_dir: Path | None,
) -> StrategyProbeResult | None:
    """Audit-only train-side probe for the image_provenance strategy.

    Matches TRAIN images against the corpus index and gates promotion on label
    precision >= 0.995 and coverage >= 0.2 — mirroring ``_probe_provenance``.
    Probe numbers are evidence for scheduling only; they produce no prediction
    artifacts and therefore can never feed learned stacking.
    """
    if "image_provenance" not in strategy_ids:
        return None
    if image_corpus_dir is None or not Path(image_corpus_dir).is_dir():
        return _result(
            strategy_id="image_provenance",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="no image corpus directory available for provenance probe",
            risks=["missing_image_corpus_dir"],
        )
    try:
        from bench.ees_core.operators.image_embedding import load_image_task

        task = load_image_task(inventory.data_dir)
    except Exception as exc:
        return _result(
            strategy_id="image_provenance",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale=f"could not load image task for provenance probe: {type(exc).__name__}: {exc}",
            risks=["image_task_load_failed"],
        )
    if len(task.target_cols) < 2:
        return _result(
            strategy_id="image_provenance",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="single-column submission has no class-name label space to map corpus classes into",
            risks=["unsupported_label_space"],
        )
    try:
        index = build_corpus_index(Path(image_corpus_dir), list(task.target_cols))
    except Exception as exc:
        return _result(
            strategy_id="image_provenance",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale=f"corpus indexing failed: {type(exc).__name__}: {exc}",
            risks=["corpus_index_failed"],
        )
    if not index.mapping.class_to_target and not index.mapping.unmapped_classes:
        return _result(
            strategy_id="image_provenance",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale="image corpus loaded no class-labelled images",
            risks=["empty_image_corpus"],
        )
    if index.mapping.mapped_fraction < 0.8:
        return _result(
            strategy_id="image_provenance",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale=(
                f"only {len(index.mapping.class_to_target)} of "
                f"{len(index.mapping.class_to_target) + len(index.mapping.unmapped_classes)} "
                "corpus classes map onto the submission label columns"
            ),
            risks=["low_corpus_class_mapping"],
        )
    if not index.exact and index.phash_matrix.shape[0] == 0:
        return _result(
            strategy_id="image_provenance",
            status="failed",
            promote=False,
            evidence_score=0.0,
            rationale="image corpus produced no usable hash keys",
            risks=["empty_image_corpus_index"],
        )
    train_ids = list(task.train_ids)
    if len(train_ids) > IMAGE_PROVENANCE_PROBE_MAX_IMAGES:
        step = len(train_ids) / IMAGE_PROVENANCE_PROBE_MAX_IMAGES
        keep = sorted({int(i * step) for i in range(IMAGE_PROVENANCE_PROBE_MAX_IMAGES)})
    else:
        keep = list(range(len(train_ids)))
    sampled_ids = [train_ids[i] for i in keep]
    sampled_labels = [normalize_class_name(task.classes[task.y[i]]) for i in keep]
    matches = 0
    wrong = 0
    for predicted, label in zip(match_ids(sampled_ids, task.train_paths, index), sampled_labels):
        if predicted is None:
            continue
        matches += 1
        wrong += int(normalize_class_name(predicted) != label)
    coverage = matches / max(1, len(sampled_ids))
    precision = 1.0 - wrong / matches if matches else None
    promote = matches > 0 and precision is not None and precision >= 0.995 and coverage >= 0.2
    return _result(
        strategy_id="image_provenance",
        status="success",
        promote=promote,
        evidence_score=(precision or 0.0) * coverage,
        coverage=coverage,
        precision_estimate=precision,
        rationale=(
            f"matched {matches}/{len(sampled_ids)} sampled train images "
            f"with {wrong} wrong matches"
        ),
        risks=[] if promote else ["low_image_provenance_coverage_or_precision"],
    )


# The probe only needs a data sample to establish the lookup structure, so its
# train/test reads are bounded -- the real text-normalization train.csv.zip is
# ~700MB unzipped and must not be fully read at probe time. The sample
# submission is read in full (it is the id universe the composite-id check
# verifies against, and is small relative to train).
SEQUENCE_PROBE_MAX_ROWS = 50_000


def _probe_sequence_rule_induction(
    strategy_ids: set[str],
    inventory: DatasetInventory,
) -> StrategyProbeResult | None:
    if "sequence_rule_induction" not in strategy_ids:
        return None
    # Always resolve through the SAME consistency-driven resolver the operator
    # uses, so probe and operator agree by construction. Inventory paths alone
    # are not enough: real dirs may ship prefixed/zipped variants
    # (en_train.csv.zip) that inventory misses, or -- worse -- stale plain
    # duplicates (test.csv from an older prepare) whose id space does not match
    # the sample submission (the live round-3 failure).
    resolved = resolve_lookup_files(inventory.data_dir)
    train = _read_csv(resolved.train if resolved else None, nrows=SEQUENCE_PROBE_MAX_ROWS)
    test = _read_csv(resolved.test if resolved else None, nrows=SEQUENCE_PROBE_MAX_ROWS)
    sample = _read_csv(resolved.sample_submission if resolved else None)
    if train is None or test is None or sample is None:
        return _result(
            strategy_id="sequence_rule_induction",
            status="skipped",
            promote=False,
            evidence_score=0.0,
            rationale="train/test/sample submission unavailable for lookup structure probe",
            risks=["missing_train_test_or_sample"],
        )
    structure = detect_lookup_structure(train, test, sample)
    if structure is None:
        return _result(
            strategy_id="sequence_rule_induction",
            status="success",
            promote=False,
            evidence_score=0.0,
            rationale=(
                "no exact-match lookup structure detected (no shared high-identity "
                "input/output column pair, or no recoverable id/group structure)"
            ),
            risks=["not_a_lookup_task"],
        )
    return _result(
        strategy_id="sequence_rule_induction",
        status="success",
        promote=True,
        evidence_score=structure.identity_rate,
        coverage=structure.identity_rate,
        rationale=(
            f"detected lookup structure: input={structure.input_col!r} "
            f"output={structure.output_col!r} group={structure.group_col!r} "
            f"identity_rate={structure.identity_rate:.3f}"
        ),
    )


def run_strategy_probes(
    *,
    strategy_plan: StrategyPlan,
    data_dir: Path,
    inventory: DatasetInventory,
    output_dir: Path,
    corpus_dir: Path | None = None,
    image_corpus_dir: Path | None = None,
    web_provider: WebResearchProvider | None = None,
    web_source_fetcher: Callable[[str], str] | None = None,
    web_catalog_fetcher: Callable[[str], dict] | None = None,
    source_cache_dir: Path | None = None,
    image_source_cache_dir: Path | None = None,
    image_landing_fetcher: Callable[[str], str] | None = None,
    image_archive_downloader: Callable[..., Path] | None = None,
) -> list[StrategyProbeResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_ids = {hypothesis.strategy_id for hypothesis in strategy_plan.hypotheses}
    results: list[StrategyProbeResult] = [_probe_submission_contract(inventory)]
    effective_corpus_dir = corpus_dir
    effective_image_corpus_dir = image_corpus_dir

    if "web_research" in strategy_ids:
        web_hypothesis = next(
            hypothesis for hypothesis in strategy_plan.hypotheses if hypothesis.strategy_id == "web_research"
        )
        web_result = run_web_research_probe(
            web_provider or NoopWebResearchProvider(),
            strategy_plan.evidence,
            web_hypothesis,
            output_dir,
        )
        results.append(web_result)
        if (
            web_result.promote
            and "external_provenance" in strategy_ids
            and not _corpus_available(corpus_dir)
        ):
            cache_dir = Path(source_cache_dir or output_dir / "web_source_cache")
            cache_result = build_source_cache_from_web_manifest(
                output_dir / "web_research_manifest.json",
                cache_dir,
                strategy_plan.evidence,
                fetcher=web_source_fetcher or fetch_web_source_text,
            )
            if cache_result.sources_written <= 0:
                # Writeup-oriented manifests rarely link labelled source texts;
                # fall back to searching for the labels' source texts directly.
                try:
                    acquisition = acquire_label_source_texts(
                        strategy_plan.evidence,
                        cache_dir,
                        provider=web_provider or NoopWebResearchProvider(),
                        fetcher=web_source_fetcher or fetch_web_source_text,
                        catalog_fetcher=web_catalog_fetcher or fetch_gutendex_catalog,
                    )
                except Exception as exc:
                    acquisition = SourceCacheBuildResult(
                        status="failed",
                        sources_written=0,
                        manifest_path=cache_dir / "source_manifest.csv",
                        notes=["label_source_acquisition_error"],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if acquisition.sources_written > 0:
                    cache_result = acquisition
                else:
                    cache_result = SourceCacheBuildResult(
                        status=cache_result.status,
                        sources_written=0,
                        manifest_path=cache_result.manifest_path,
                        notes=[*cache_result.notes, *acquisition.notes],
                        error="; ".join(
                            error for error in [cache_result.error, acquisition.error] if error
                        )
                        or None,
                    )
            promote_cache = cache_result.status == "success" and cache_result.sources_written > 0
            # A corpus that covers only one class cannot separate authors: when
            # several labels have inferable sources, require at least two covered.
            insufficient_coverage = (
                promote_cache
                and cache_result.labels_with_aliases >= 2
                and cache_result.labels_with_sources < 2
            )
            if insufficient_coverage:
                promote_cache = False
            cache_risks: list[str] = []
            if cache_result.status != "success":
                cache_risks.append("no_cached_web_sources")
            if insufficient_coverage:
                cache_risks.append("insufficient_label_coverage")
            cache_probe = _result(
                strategy_id="web_source_cache",
                status=cache_result.status,
                promote=promote_cache,
                evidence_score=float(cache_result.sources_written),
                rationale="; ".join(cache_result.notes) or cache_result.status,
                risks=cache_risks,
                artifact_paths=[str(cache_result.manifest_path)],
            )
            results.append(cache_probe)
            if cache_probe.promote:
                effective_corpus_dir = cache_dir

    # Autonomous image-corpus acquisition (mirror of the text label-source
    # acquisition above): when image_provenance is hypothesized but no local
    # image corpus exists, infer + download the source dataset named in the
    # description. Gated by ``image_landing_fetcher`` being supplied -- the
    # controller passes a real fetcher only behind the web-research enable
    # flags, so this stays offline unless live research is turned on. The
    # acquired corpus is only a hypothesis; the image_provenance probe's
    # train-precision gate still decides whether it may promote.
    if (
        "image_provenance" in strategy_ids
        and image_landing_fetcher is not None
        and not _image_corpus_available(image_corpus_dir)
    ):
        image_cache_dir = Path(image_source_cache_dir or output_dir / "image_source_cache")
        try:
            acquisition = acquire_image_source_corpus(
                strategy_plan.evidence,
                image_cache_dir,
                landing_fetcher=image_landing_fetcher,
                archive_downloader=image_archive_downloader or stream_download_archive,
            )
        except Exception as exc:
            acquisition = SourceCacheBuildResult(
                status="failed",
                sources_written=0,
                manifest_path=image_cache_dir,
                notes=["image_source_acquisition_error"],
                error=f"{type(exc).__name__}: {exc}",
            )
        promote_image_cache = (
            acquisition.status == "success"
            and acquisition.sources_written > 0
            and _image_corpus_available(image_cache_dir)
        )
        results.append(
            _result(
                strategy_id="image_source_cache",
                status=acquisition.status,
                promote=promote_image_cache,
                evidence_score=float(acquisition.sources_written),
                rationale="; ".join(acquisition.notes) or acquisition.status,
                risks=[] if promote_image_cache else ["no_acquired_image_corpus"],
                artifact_paths=[str(acquisition.manifest_path)],
            )
        )
        if promote_image_cache:
            effective_image_corpus_dir = image_cache_dir

    provenance = _probe_provenance(strategy_ids, strategy_plan, inventory, effective_corpus_dir)
    if provenance is not None:
        results.append(provenance)

    image_provenance = _probe_image_provenance(strategy_ids, inventory, effective_image_corpus_dir)
    if image_provenance is not None:
        results.append(image_provenance)

    aux = _probe_auxiliary_structure(strategy_ids, strategy_plan, inventory)
    if aux is not None:
        results.append(aux)

    metadata = _probe_metadata_leakage(strategy_ids, strategy_plan, inventory)
    if metadata is not None:
        results.append(metadata)

    results.append(_probe_duplicate_leakage(strategy_plan, inventory))

    sequence_rule_induction = _probe_sequence_rule_induction(strategy_ids, inventory)
    if sequence_rule_induction is not None:
        results.append(sequence_rule_induction)

    calibration = _probe_calibration(strategy_ids, strategy_plan.evidence.metric)
    if calibration is not None:
        results.append(calibration)

    output_path = output_dir / "probe_results.json"
    output_path.write_text(
        json.dumps([result.to_json() for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [
        StrategyProbeResult(
            strategy_id=result.strategy_id,
            status=result.status,
            promote=result.promote,
            evidence_score=result.evidence_score,
            coverage=result.coverage,
            precision_estimate=result.precision_estimate,
            rationale=result.rationale,
            risks=result.risks,
            artifact_paths=[*result.artifact_paths, str(output_path)],
        )
        for result in results
    ]
