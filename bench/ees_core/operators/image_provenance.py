"""Image provenance operator: match task images against an external source corpus.

Second-modality sibling of ``text_provenance``. Some image competitions draw
their images from a public source dataset (e.g. an ImageNet subset) whose
directory layout carries the class label. When such a corpus is available
(``EES_IMAGE_CORPUS_DIR`` or ``<data_dir>/image_corpora``), matching task
images against it by content hash recovers labels directly.

Evidence chain (NO task-id logic anywhere):
1. the corpus directory exists and contains class-labelled images,
2. corpus class names map onto the sample-submission label columns after
   generic normalization (taxonomy prefixes such as ``n02085620-`` stripped),
3. train-image matches agree with the train labels (precision audited by the
   ``image_provenance`` strategy probe and re-measured here).

Where the corpus directory comes from is the caller's job (Stage 2 will add
autonomous source inference/acquisition); this module only consumes it.
"""
from __future__ import annotations

import json
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import log_loss

from bench.adapters.task_detection import IMAGE_EXTS
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.operators.embedding_core import resolve_media_paths
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts

CORPUS_INDEX_VERSION = "image_provenance_index_v1"
# Taxonomy prefixes like ImageNet synset ids: "n02085620-Chihuahua" -> "Chihuahua".
_TAXONOMY_PREFIX = re.compile(r"^n?\d{4,}[-_.]+")
_HASH_SIZE = 8
# Sentinel for hash keys observed under more than one mapped label; such keys
# can never produce a unique match.
_AMBIGUOUS = "__ambiguous__"


def normalize_class_name(name: object) -> str:
    """Generic label normalization shared by corpus dir names and target columns.

    Lowercase, strip taxonomy prefixes (``n02085620-``), fold unicode to ascii,
    collapse punctuation/whitespace runs to a single underscore.
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = _TAXONOMY_PREFIX.sub("", text.strip())
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())
    return text.strip("_")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def exact_image_hash(image: Image.Image) -> str:
    """Crypto hash of the decoded RGB pixel bytes (byte-identical pixels only)."""
    digest = sha256()
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _resize_gray(image: Image.Image, width: int, height: int) -> np.ndarray:
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    resized = image.convert("L").resize((width, height), resample)
    return np.asarray(resized, dtype=np.float64)


def _bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8).ravel()).tobytes().hex()


def dhash(image: Image.Image, hash_size: int = _HASH_SIZE) -> str:
    """Difference hash: horizontal gradient signs on a (hash_size+1) x hash_size thumbnail."""
    pixels = _resize_gray(image, hash_size + 1, hash_size)
    return _bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def ahash(image: Image.Image, hash_size: int = _HASH_SIZE) -> str:
    """Average hash: above-mean mask on a hash_size x hash_size thumbnail."""
    pixels = _resize_gray(image, hash_size, hash_size)
    return _bits_to_hex(pixels > pixels.mean())


def _dhash_bits(image: Image.Image, hash_size: int = _HASH_SIZE) -> np.ndarray:
    pixels = _resize_gray(image, hash_size + 1, hash_size)
    return (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8).ravel()


def _ahash_bits(image: Image.Image, hash_size: int = _HASH_SIZE) -> np.ndarray:
    pixels = _resize_gray(image, hash_size, hash_size)
    return (pixels > pixels.mean()).astype(np.uint8).ravel()


def perceptual_image_hash(image: Image.Image) -> str:
    """Combined dHash+aHash key (128 bits) as hex.

    Kept for exact-key round-trips and readable diagnostics; the operator itself
    matches by Hamming distance over the packed bit vector (see below), so a few
    bits flipped by re-encoding/resizing no longer break the match.
    """
    return dhash(image) + ahash(image)


def perceptual_hash_bits(image: Image.Image, hash_size: int = _HASH_SIZE) -> np.ndarray:
    """Packed uint8 bit vector (dHash ++ aHash) for Hamming-distance matching."""
    bits = np.concatenate([_dhash_bits(image, hash_size), _ahash_bits(image, hash_size)])
    return np.packbits(bits)


# Popcount lookup: number of set bits in each byte value 0..255.
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint16)


def hamming_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between one packed hash and a packed hash matrix."""
    return _POPCOUNT[np.bitwise_xor(matrix, query)].sum(axis=1)


@dataclass(frozen=True)
class CorpusClassMapping:
    """How corpus class directories map onto the task's label columns."""

    class_to_target: dict[str, str]
    unmapped_classes: list[str]

    @property
    def mapped_fraction(self) -> float:
        total = len(self.class_to_target) + len(self.unmapped_classes)
        return len(self.class_to_target) / total if total else 0.0


# Default max combined (128-bit) Hamming distance for a perceptual match. Kaggle
# re-encodes/resizes source images, flipping a few hash bits; an exact 128-bit
# key match (radius 0) leaves ~20% of true matches on the table. A small radius
# recovers them while the precision gate keeps false matches out (measured on
# dog-breed: radius 0 -> 78% coverage, radius <=4 -> 100% at 0.998 precision).
DEFAULT_MAX_HAMMING = 6


@dataclass(frozen=True)
class ImageCorpusIndex:
    exact: dict[str, str]
    phash_matrix: np.ndarray  # (N, 16) packed uint8: dHash ++ aHash per corpus image
    phash_labels: np.ndarray  # (N,) task label per corpus row
    mapping: CorpusClassMapping
    corpus_image_count: int
    ambiguous_exact_keys: int
    unreadable_images: int
    cache_status: str
    cache_path: Path
    max_hamming: int = DEFAULT_MAX_HAMMING


# source_manifest.csv column aliases for the image path and its class label,
# so an acquired corpus that ships flat images + a mapping file (rather than a
# class-directory layout) is still consumable.
_MANIFEST_PATH_COLS = ("image", "path", "local_path", "filepath", "file")
_MANIFEST_CLASS_COLS = ("class", "label", "breed", "category", "target")


def _manifest_class_images(corpus_dir: Path) -> dict[str, list[Path]]:
    """Read an optional ``source_manifest.csv`` (image->class) into class buckets.

    Mirrors the manifest fallback ``text_provenance._iter_corpus_sources`` uses:
    a corpus whose images are not laid out one-directory-per-class can instead
    declare an image->class mapping. Paths may be absolute or relative to the
    corpus directory. Returns an empty mapping when no readable manifest exists.
    """
    manifest = corpus_dir / "source_manifest.csv"
    if not manifest.exists():
        return {}
    try:
        rows = pd.read_csv(manifest)
    except Exception:
        return {}
    columns = {str(col).lower(): col for col in rows.columns}
    path_col = next((columns[name] for name in _MANIFEST_PATH_COLS if name in columns), None)
    class_col = next((columns[name] for name in _MANIFEST_CLASS_COLS if name in columns), None)
    if path_col is None or class_col is None:
        return {}
    classes: dict[str, list[Path]] = {}
    for row in rows.itertuples(index=False):
        raw_path = str(getattr(row, path_col)).strip()
        class_name = str(getattr(row, class_col)).strip()
        if not raw_path or not class_name:
            continue
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = corpus_dir / image_path
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        classes.setdefault(class_name, []).append(image_path)
    return classes


def discover_corpus_class_images(corpus_dir: Path) -> dict[str, list[Path]]:
    """Map corpus class names to their image files.

    A corpus image's class is its immediate parent directory name (e.g.
    ``Images/n02085620-Chihuahua/img.jpg`` -> ``n02085620-Chihuahua``). Images
    sitting directly at the corpus root carry no class and are ignored. If a
    ``source_manifest.csv`` (image->class) is present its rows are merged in,
    so an acquired flat-layout corpus is consumable too.
    """
    corpus_dir = Path(corpus_dir)
    classes: dict[str, list[Path]] = {}
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parent = path.parent
        if parent == corpus_dir or parent.name.startswith("."):
            continue
        classes.setdefault(parent.name, []).append(path)
    for class_name, paths in _manifest_class_images(corpus_dir).items():
        bucket = classes.setdefault(class_name, [])
        known = set(bucket)
        bucket.extend(path for path in paths if path not in known)
    return classes


def map_corpus_classes_to_targets(
    class_names: list[str],
    target_cols: list[str],
) -> CorpusClassMapping:
    """Match normalized corpus class names against normalized label columns."""
    normalized_targets: dict[str, str] = {}
    for target in target_cols:
        normalized_targets.setdefault(normalize_class_name(target), str(target))
    class_to_target: dict[str, str] = {}
    unmapped: list[str] = []
    for class_name in class_names:
        target = normalized_targets.get(normalize_class_name(class_name))
        if target is None:
            unmapped.append(class_name)
        else:
            class_to_target[class_name] = target
    return CorpusClassMapping(class_to_target=class_to_target, unmapped_classes=sorted(unmapped))


def _corpus_fingerprint(class_images: dict[str, list[Path]], target_cols: list[str]) -> str:
    payload = {
        "version": CORPUS_INDEX_VERSION,
        "hash_size": _HASH_SIZE,
        "target_cols": [str(col) for col in target_cols],
        "files": [
            {
                "class": class_name,
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for class_name in sorted(class_images)
            for path in class_images[class_name]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()[:24]


def _insert_key(table: dict[str, str], key: str, label: str) -> None:
    existing = table.get(key)
    if existing is None:
        table[key] = label
    elif existing != label:
        table[key] = _AMBIGUOUS


def build_corpus_index(
    corpus_dir: Path,
    target_cols: list[str],
    *,
    max_hamming: int = DEFAULT_MAX_HAMMING,
) -> ImageCorpusIndex:
    """Index corpus images by exact crypto hash + a packed perceptual-hash matrix.

    The perceptual matrix (one packed dHash++aHash row per corpus image, with a
    parallel task-label array) supports Hamming-nearest-neighbor matching so a
    few bits flipped by re-encoding no longer break a match. Byte-identical
    exact-hash keys observed under >1 label are marked ambiguous. The index is
    cached next to the corpus (keyed by file stats + target columns) so probe
    and operator share one build.
    """
    corpus_dir = Path(corpus_dir)
    class_images = discover_corpus_class_images(corpus_dir)
    mapping = map_corpus_classes_to_targets(sorted(class_images), list(target_cols))
    fingerprint = _corpus_fingerprint(class_images, list(target_cols))
    cache_dir = corpus_dir / ".ees_cache"
    cache_path = cache_dir / f"{CORPUS_INDEX_VERSION}_{fingerprint}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return ImageCorpusIndex(
                exact=payload["exact"],
                phash_matrix=payload["phash_matrix"],
                phash_labels=payload["phash_labels"],
                mapping=mapping,
                corpus_image_count=payload["corpus_image_count"],
                ambiguous_exact_keys=payload["ambiguous_exact_keys"],
                unreadable_images=payload["unreadable_images"],
                cache_status="hit",
                cache_path=cache_path,
                max_hamming=max_hamming,
            )
        except Exception:
            pass

    exact: dict[str, str] = {}
    phash_rows: list[np.ndarray] = []
    phash_labels: list[str] = []
    image_count = 0
    unreadable = 0
    for class_name, label in mapping.class_to_target.items():
        for path in class_images[class_name]:
            try:
                image = _open_rgb(path)
            except Exception:
                unreadable += 1
                continue
            image_count += 1
            _insert_key(exact, exact_image_hash(image), label)
            phash_rows.append(perceptual_hash_bits(image))
            phash_labels.append(label)
    ambiguous_exact = sum(1 for label in exact.values() if label == _AMBIGUOUS)
    phash_matrix = (
        np.vstack(phash_rows) if phash_rows else np.empty((0, 2 * (_HASH_SIZE * _HASH_SIZE // 8)), dtype=np.uint8)
    )
    phash_labels_arr = np.asarray(phash_labels, dtype=object)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "exact": exact,
                    "phash_matrix": phash_matrix,
                    "phash_labels": phash_labels_arr,
                    "corpus_image_count": image_count,
                    "ambiguous_exact_keys": ambiguous_exact,
                    "unreadable_images": unreadable,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass
    return ImageCorpusIndex(
        exact=exact,
        phash_matrix=phash_matrix,
        phash_labels=phash_labels_arr,
        mapping=mapping,
        corpus_image_count=image_count,
        ambiguous_exact_keys=ambiguous_exact,
        unreadable_images=unreadable,
        cache_status="miss",
        cache_path=cache_path,
        max_hamming=max_hamming,
    )


@dataclass
class MatchStats:
    exact_matches: int = 0
    perceptual_matches: int = 0
    ambiguous_hits: int = 0
    unreadable_images: int = 0


def _nearest_perceptual_label(image: Image.Image, index: ImageCorpusIndex) -> tuple[str | None, bool]:
    """Nearest corpus label by combined Hamming distance.

    Returns ``(label, ambiguous)``. A match requires the minimum distance to be
    within ``index.max_hamming`` AND all corpus rows achieving that minimum to
    share one label; otherwise the nearest neighbourhood is ambiguous and no
    label is emitted (falls back to uniform downstream).
    """
    if index.phash_matrix.shape[0] == 0:
        return None, False
    query = perceptual_hash_bits(image)
    distances = hamming_distances(query, index.phash_matrix)
    best = int(distances.min())
    if best > index.max_hamming:
        return None, False
    tied_labels = set(index.phash_labels[distances == best].tolist())
    if len(tied_labels) != 1:
        return None, True
    return next(iter(tied_labels)), False


def match_image_path(path: Path, index: ImageCorpusIndex, stats: MatchStats | None = None) -> str | None:
    """Match one image against the corpus: exact crypto hash first, then Hamming NN.

    Returns the mapped task label, or None (no match / ambiguous / unreadable).
    """
    try:
        image = _open_rgb(Path(path))
    except Exception:
        if stats is not None:
            stats.unreadable_images += 1
        return None
    label = index.exact.get(exact_image_hash(image))
    if label is not None:
        if label == _AMBIGUOUS:
            if stats is not None:
                stats.ambiguous_hits += 1
            return None
        if stats is not None:
            stats.exact_matches += 1
        return label
    label, ambiguous = _nearest_perceptual_label(image, index)
    if label is None:
        if ambiguous and stats is not None:
            stats.ambiguous_hits += 1
        return None
    if stats is not None:
        stats.perceptual_matches += 1
    return label


def match_ids(
    ids: list,
    paths: dict,
    index: ImageCorpusIndex,
    stats: MatchStats | None = None,
) -> list[str | None]:
    """Match task image ids (zip-backed refs supported) against the corpus index."""
    resolved = resolve_media_paths(ids, paths)
    return [match_image_path(path, index, stats) for path in resolved]


def run_image_provenance_operator(
    data_dir: Path,
    output_dir: Path,
    corpus_dir: Path,
    *,
    task_id: str = "",
    confidence: float = 0.98,
    calibrate_confidence: bool = True,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    min_mapped_class_fraction: float = 0.8,
) -> CandidateResult:
    """Emit near-one-hot predictions for corpus-matched test images.

    Matched rows get the matched class ``confidence`` and the uniform remainder
    elsewhere (log-loss safe); unmatched rows stay uniform, leaving them to
    global ensembling with the embedding operator's predictions — exactly how
    ``text_provenance`` handles partial coverage.

    When ``calibrate_confidence`` is set (default), the confidence placed on a
    match is the AUDITED train-match precision (clipped to [0.5, 0.999]) rather
    than a fixed constant. That is the log-loss-optimal weight for a match rule
    whose empirical hit rate is ``train_precision``, and it is honest: the
    precision is measured on train labels only, never on the held-out test set.

    Fails honestly (success=False, reasoned) when the corpus is missing, empty,
    or its class names do not map onto the task label space.
    """
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    corpus_dir = Path(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _failure(reason: str) -> CandidateResult:
        return CandidateResult(
            candidate_id="image_provenance",
            recipe_id="image_provenance",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_provenance",
            no_score_reason=reason,
        )

    if not corpus_dir.is_dir():
        return _failure("missing_image_corpus_dir")

    try:
        # Late import: image_embedding pulls embedding_core's torch-optional
        # stack; reuse its task loader so id/label/path discovery cannot drift.
        from bench.ees_core.operators.image_embedding import load_image_task

        task = load_image_task(data_dir)
        target_cols = list(task.target_cols)
        if len(target_cols) < 2:
            # Corpus class names map onto label COLUMNS; a single-column
            # (binary) submission has no class-name label space to map into.
            return _failure("unsupported_label_space:single_target_column")
        index = build_corpus_index(corpus_dir, target_cols, max_hamming=max_hamming)
        if not index.mapping.class_to_target and not index.mapping.unmapped_classes:
            return _failure("no_corpus_classes_found")
        if index.mapping.mapped_fraction < min_mapped_class_fraction:
            return _failure(
                "insufficient_class_mapping:"
                f"{len(index.mapping.class_to_target)}/"
                f"{len(index.mapping.class_to_target) + len(index.mapping.unmapped_classes)}"
                f"_classes_mapped_below_{min_mapped_class_fraction}"
            )
        if not index.exact and index.phash_matrix.shape[0] == 0:
            return _failure("empty_corpus_index")

        # Train labels must live in the sample-submission label space for the
        # precision audit and one-hot validation targets to mean anything.
        # Map via the SAME normalization as corpus classes; fail honestly if
        # any train label has no target column.
        normalized_targets: dict[str, str] = {}
        for col in target_cols:
            normalized_targets.setdefault(normalize_class_name(col), str(col))
        raw_train_labels = [str(task.classes[c]) for c in task.y]
        train_labels = [normalized_targets.get(normalize_class_name(label)) for label in raw_train_labels]
        if any(label is None for label in train_labels):
            unmapped_count = sum(label is None for label in train_labels)
            return _failure(f"train_labels_not_in_target_space:{unmapped_count}_rows")

        stats = MatchStats()
        train_matches = match_ids(task.train_ids, task.train_paths, index, stats)
        test_matches = match_ids(task.test_ids, task.test_paths, index, stats)

        train_match_count = sum(match is not None for match in train_matches)
        wrong = sum(
            1
            for match, label in zip(train_matches, train_labels)
            if match is not None and match != label
        )
        train_coverage = train_match_count / max(1, len(task.train_ids))
        train_precision = 1.0 - wrong / train_match_count if train_match_count else None
        test_match_count = sum(match is not None for match in test_matches)
        test_coverage = test_match_count / max(1, len(task.test_ids))

        n_classes = len(target_cols)
        uniform = 1.0 / n_classes
        # Calibrate match confidence to the audited train precision (log-loss
        # optimal, leak-free); fall back to the fixed constant when calibration
        # is off or precision is unmeasurable (no train matches).
        if calibrate_confidence and train_precision is not None:
            effective_confidence = float(min(max(train_precision, 0.5), 0.999))
        else:
            effective_confidence = confidence
        residual = (1.0 - effective_confidence) / (n_classes - 1)

        def _row_probabilities(match: str | None) -> np.ndarray:
            if match is None:
                return np.full(n_classes, uniform, dtype=float)
            return np.asarray(
                [effective_confidence if col == match else residual for col in target_cols],
                dtype=float,
            )

        # Honest validation over ALL train rows: matched rows get the same
        # deterministic one-hot-with-confidence used for test (the hash-match
        # rule has no fitted parameters, so scoring it on train is leak-free),
        # unmatched rows stay uniform. Mirrors text_provenance's blend contract.
        blend = np.vstack([_row_probabilities(match) for match in train_matches])
        validation_blend_log_loss = _safe_log_loss(train_labels, blend, target_cols)
        uniform_log_loss = float(np.log(n_classes))

        test_rows = np.vstack([_row_probabilities(match) for match in test_matches])
        submission = pd.DataFrame(test_rows, columns=target_cols)
        submission.insert(0, task.id_col, task.test_ids)
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        matches_path = output_dir / "matches.csv"
        pd.DataFrame(
            {"id": task.test_ids, "matched_label": test_matches}
        ).to_csv(matches_path, index=False)

        validation_predictions = pd.DataFrame(blend, columns=target_cols)
        validation_predictions.insert(0, task.id_col, list(task.train_ids))
        validation_targets = one_hot_frame(
            list(task.train_ids),
            train_labels,
            id_col=task.id_col,
            target_cols=target_cols,
        )
        # Sort by id in the NATIVE id dtype so predictions and targets stay
        # row-aligned (same convention as text_provenance / image_embedding).
        validation_predictions = validation_predictions.sort_values(task.id_col, kind="stable").reset_index(drop=True)
        validation_targets = validation_targets.sort_values(task.id_col, kind="stable").reset_index(drop=True)
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=task.id_col,
            target_cols=target_cols,
            validation_kind="oof",
        )

        metrics = {
            "operator": "image_provenance",
            "task_id": task_id,
            "train_matches": train_match_count,
            "train_wrong_matches": wrong,
            "train_precision": train_precision,
            "train_coverage": train_coverage,
            "test_matches": test_match_count,
            "test_coverage": test_coverage,
            "exact_matches": stats.exact_matches,
            "perceptual_matches": stats.perceptual_matches,
            "ambiguous_hits": stats.ambiguous_hits,
            "unreadable_task_images": stats.unreadable_images,
            "confidence": effective_confidence,
            "confidence_calibrated": bool(calibrate_confidence and train_precision is not None),
            "max_hamming": max_hamming,
            "n_classes": n_classes,
            "corpus_image_count": index.corpus_image_count,
            "corpus_mapped_classes": len(index.mapping.class_to_target),
            "corpus_unmapped_classes": index.mapping.unmapped_classes,
            "corpus_mapped_class_fraction": index.mapping.mapped_fraction,
            "corpus_ambiguous_exact_keys": index.ambiguous_exact_keys,
            "corpus_unreadable_images": index.unreadable_images,
            "corpus_index_cache_status": index.cache_status,
            "corpus_index_cache_path": str(index.cache_path),
            "prediction_artifacts": prediction_artifacts,
            "validation_blend_log_loss": validation_blend_log_loss,
            "uniform_log_loss": uniform_log_loss,
            "train_rows": int(len(task.train_ids)),
            "test_rows": int(len(task.test_ids)),
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="image_provenance",
            recipe_id="image_provenance",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation_blend_log_loss,
            medal="none",
            final_artifact_source="ees_core:image_provenance",
            artifact_paths=[str(submission_path), str(matches_path), str(metrics_path)],
            metrics=metrics,
            score_direction="minimize",  # score is validation blend log-loss
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _safe_log_loss(labels: list[str], probabilities: np.ndarray, classes: list[str]) -> float | None:
    try:
        return float(log_loss(labels, probabilities, labels=[str(c) for c in classes]))
    except ValueError:
        return None
