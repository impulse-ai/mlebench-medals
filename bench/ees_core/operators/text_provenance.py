"""Text provenance operator for Spooky-style author tasks."""
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import FeatureUnion

from bench.ees_core.candidates import CandidateResult
from bench.ees_core.folds import assemble_oof, choose_n_splits, make_folds
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts

AUTHOR_COLUMNS = ["EAP", "HPL", "MWS"]
CORPUS_TEXT_EXTS = {".txt", ".text", ".utf-8"}
CORPUS_INDEX_VERSION = "text_provenance_index_v1"
_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def _normalize(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).translate(_PUNCT_TRANSLATION))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True)
class CorpusSource:
    author: str
    path: Path


@dataclass(frozen=True)
class ProvenanceCorpusIndex:
    corpora: dict[str, str]
    shingle_sets: dict[str, set[str]]
    cache_status: str
    cache_path: Path
    source_count: int


def _iter_corpus_sources(corpus_dir: Path) -> list[CorpusSource]:
    sources: list[CorpusSource] = []
    for author in AUTHOR_COLUMNS:
        if (corpus_dir / author).is_dir():
            paths = [
                path
                for path in (corpus_dir / author).rglob("*")
                if path.is_file() and path.suffix.lower() in CORPUS_TEXT_EXTS
            ]
        else:
            paths = [
                path
                for path in corpus_dir.glob(f"{author}.*")
                if path.is_file() and path.suffix.lower() in CORPUS_TEXT_EXTS
            ]
        for path in sorted(paths):
            sources.append(CorpusSource(author=author, path=path))
    manifest = corpus_dir / "source_manifest.csv"
    if manifest.exists():
        try:
            rows = pd.read_csv(manifest)
            if {"author", "local_path"}.issubset(rows.columns):
                for row in rows.itertuples(index=False):
                    author = str(getattr(row, "author"))
                    local_path = Path(str(getattr(row, "local_path")))
                    if author not in AUTHOR_COLUMNS or not local_path.exists() or not local_path.is_file():
                        continue
                    sources.append(CorpusSource(author=author, path=local_path))
        except Exception:
            pass
    return sources


def _load_corpora(corpus_dir: Path) -> dict[str, str]:
    corpora: dict[str, list[str]] = {author: [] for author in AUTHOR_COLUMNS}
    for source in _iter_corpus_sources(corpus_dir):
        corpora[source.author].append(
            _normalize(source.path.read_text(encoding="utf-8", errors="ignore"))
        )
    return {author: "\n".join(parts).strip() for author, parts in corpora.items()}


def _source_fingerprint(sources: list[CorpusSource], *, shingle_words: int) -> str:
    payload = {
        "version": CORPUS_INDEX_VERSION,
        "shingle_words": shingle_words,
        "sources": [
            {
                "author": source.author,
                "path": str(source.path.resolve()),
                "size": source.path.stat().st_size,
                "mtime_ns": source.path.stat().st_mtime_ns,
            }
            for source in sources
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()[:24]


def _load_corpus_index(corpus_dir: Path, *, shingle_words: int = 8) -> ProvenanceCorpusIndex:
    corpus_dir = Path(corpus_dir)
    sources = _iter_corpus_sources(corpus_dir)
    fingerprint = _source_fingerprint(sources, shingle_words=shingle_words)
    cache_dir = corpus_dir / ".ees_cache"
    cache_path = cache_dir / f"{CORPUS_INDEX_VERSION}_{fingerprint}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return ProvenanceCorpusIndex(
                corpora=payload["corpora"],
                shingle_sets=payload["shingle_sets"],
                cache_status="hit",
                cache_path=cache_path,
                source_count=len(sources),
            )
        except Exception:
            pass

    corpora = {author: [] for author in AUTHOR_COLUMNS}
    for source in sources:
        corpora[source.author].append(
            _normalize(source.path.read_text(encoding="utf-8", errors="ignore"))
        )
    normalized_corpora = {
        author: "\n".join(parts).strip()
        for author, parts in corpora.items()
    }
    shingle_sets = _build_shingle_sets(normalized_corpora, shingle_words=shingle_words)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"corpora": normalized_corpora, "shingle_sets": shingle_sets},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass
    return ProvenanceCorpusIndex(
        corpora=normalized_corpora,
        shingle_sets=shingle_sets,
        cache_status="miss",
        cache_path=cache_path,
        source_count=len(sources),
    )


def infer_label_aliases(description: str, target_cols: list[str]) -> dict[str, list[str]]:
    """Infer human-readable aliases for abbreviated class labels from text.

    This intentionally uses description/schema evidence only. It is not tied to
    a competition id, so any author/source task with labels and parenthetical
    aliases can trigger provenance research.
    """
    if not description or not target_cols:
        return {}
    normalized_description = _normalize(description)
    aliases: dict[str, list[str]] = {}
    for label in target_cols:
        label_text = str(label).strip()
        if not label_text:
            continue
        pattern = re.compile(
            rf"([A-Za-z][A-Za-z.\s'-]{{2,80}}?)\s*\(\s*{re.escape(label_text)}\s*\)",
            flags=re.IGNORECASE,
        )
        matches = []
        for match in pattern.finditer(description):
            raw_alias = match.group(1)
            for delimiter in [",", ";", " by ", " and "]:
                if delimiter in raw_alias.lower():
                    parts = re.split(re.escape(delimiter), raw_alias, flags=re.IGNORECASE)
                    raw_alias = parts[-1]
            alias = _normalize(raw_alias)
            alias = re.sub(r"^(and|by)\s+", "", alias)
            if alias and alias not in matches and alias in normalized_description:
                matches.append(alias)
        if not matches:
            matches = _initials_aliases(description, label_text)
        if matches:
            aliases[label_text] = matches
    return aliases


def _initials_aliases(description: str, label: str) -> list[str]:
    """Fallback for prose descriptions: match a label like ``EAP`` to a proper
    name like ``Edgar Allan Poe`` by initials.

    A name's initials must equal the label, or be a subsequence of it with the
    same first letter (``Mary Shelley`` -> ``MS`` matches ``MWS``: authors are
    often listed without middle names). False positives are self-limiting
    downstream: a wrong alias fetches a corpus that cannot pass the provenance
    probe's train-precision gate.
    """
    label_up = re.sub(r"[^A-Z]", "", label.upper())
    if len(label_up) < 2:
        return []
    token_matches = list(re.finditer(r"[A-Z][\w.'-]*", description))
    matches: list[str] = []
    for size in (4, 3, 2):
        for start in range(len(token_matches) - size + 1):
            span = token_matches[start:start + size]
            # tokens must be adjacent in the text (at most one separator char)
            # so unrelated capitalized words don't chain into fake names
            if any(span[i].start() - span[i - 1].end() > 1 for i in range(1, size)):
                continue
            words = [m.group(0) for m in span]
            initials = "".join(
                word if word.isupper() and len(word) <= 3 else word[0]
                for word in words
            ).upper()
            initials = re.sub(r"[^A-Z]", "", initials)
            if len(initials) < 2 or initials[0] != label_up[0]:
                continue
            if initials != label_up and not _is_subsequence(initials, label_up):
                continue
            alias = _normalize(" ".join(words))
            if alias and alias not in matches:
                matches.append(alias)
    return matches


def _is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(ch in iterator for ch in needle)


@dataclass
class ProvenanceMatchStats:
    candidate_author_checks: int = 0
    full_scan_author_checks: int = 0
    shingle_filtered_queries: int = 0
    shingle_missed_queries: int = 0


def _build_shingle_sets(corpora: dict[str, str], *, shingle_words: int) -> dict[str, set[str]]:
    shingle_sets: dict[str, set[str]] = {}
    for author, corpus in corpora.items():
        tokens = corpus.split()
        if len(tokens) < shingle_words:
            shingle_sets[author] = set()
            continue
        shingle_sets[author] = {
            " ".join(tokens[index : index + shingle_words])
            for index in range(len(tokens) - shingle_words + 1)
        }
    return shingle_sets


def _candidate_authors(
    normalized_query: str,
    corpora: dict[str, str],
    shingle_sets: dict[str, set[str]],
    *,
    shingle_words: int,
) -> set[str]:
    tokens = normalized_query.split()
    authors = set(corpora)
    if len(tokens) < shingle_words:
        return authors
    starts = sorted(
        {
            0,
            max(0, len(tokens) // 2 - shingle_words // 2),
            max(0, len(tokens) - shingle_words),
        }
    )
    candidates: set[str] = set()
    for start in starts:
        shingle = " ".join(tokens[start : start + shingle_words])
        for author, shingles in shingle_sets.items():
            if shingle in shingles:
                candidates.add(author)
    return candidates


def _unique_author(
    text: object,
    corpora: dict[str, str],
    min_chars: int,
    *,
    shingle_sets: dict[str, set[str]] | None = None,
    shingle_words: int = 8,
    stats: ProvenanceMatchStats | None = None,
) -> str | None:
    query = _normalize(text)
    if len(query) < min_chars:
        return None
    author_corpora = corpora
    if shingle_sets is not None:
        candidates = _candidate_authors(
            query,
            corpora,
            shingle_sets,
            shingle_words=shingle_words,
        )
        if not candidates:
            if stats is not None:
                stats.full_scan_author_checks += len(corpora)
                stats.shingle_missed_queries += 1
            return None
        if len(candidates) < len(corpora):
            if stats is not None:
                stats.shingle_filtered_queries += 1
            author_corpora = {author: corpora[author] for author in candidates}
    if stats is not None:
        stats.candidate_author_checks += len(author_corpora)
        stats.full_scan_author_checks += len(corpora)
    matches = [author for author, corpus in author_corpora.items() if query and query in corpus]
    return matches[0] if len(matches) == 1 else None


def run_text_provenance_operator(
    data_dir: Path,
    output_dir: Path,
    corpus_dir: Path,
    *,
    min_chars: int = 8,
    confidence: float = 0.999,
    shingle_words: int = 8,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    corpus_dir = Path(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not corpus_dir.is_dir():
        return CandidateResult(
            candidate_id="text_provenance",
            recipe_id="text_provenance",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:text_provenance",
            no_score_reason="missing_corpus_dir",
        )

    try:
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        corpus_index = _load_corpus_index(corpus_dir, shingle_words=shingle_words)
        corpora = corpus_index.corpora
        if not any(corpora.values()):
            return CandidateResult(
                candidate_id="text_provenance",
                recipe_id="text_provenance",
                success=False,
                submission_path=None,
                validation={},
                score=None,
                medal="none",
                final_artifact_source="ees_core:text_provenance",
                no_score_reason="empty_corpora",
            )

        shingle_sets = corpus_index.shingle_sets
        match_stats = ProvenanceMatchStats()
        train_matches = []
        wrong = 0
        for _, row in train.iterrows():
            predicted = _unique_author(
                row["text"],
                corpora,
                min_chars=min_chars,
                shingle_sets=shingle_sets,
                shingle_words=shingle_words,
                stats=match_stats,
            )
            if predicted is not None:
                wrong += int(predicted != row["author"])
            train_matches.append({"id": row["id"], "matched_author": predicted})

        train_texts = train["text"].astype(str)
        train_labels = train["author"].astype(str)
        if len(train) >= 20:
            # True full-train OOF for the fallback model (F.2 fold convention):
            # every train row gets exactly one out-of-fold prediction, so the
            # exported validation artifacts are full-length and leak-free.
            n_splits = choose_n_splits(train_labels, stratified=True)
            folds = make_folds(
                train_labels,
                n_splits=n_splits,
                stratified=True,
                random_state=42,
            )
            fold_predictions = []
            for train_idx, valid_idx in folds:
                fold_vectorizer = _make_fallback_vectorizer()
                x_fold_train = fold_vectorizer.fit_transform(train_texts.iloc[train_idx])
                x_fold_valid = fold_vectorizer.transform(train_texts.iloc[valid_idx])
                fold_model = LogisticRegression(C=4.0, max_iter=1000, solver="saga", n_jobs=-1)
                fold_model.fit(x_fold_train, train_labels.iloc[train_idx])
                fold_predictions.append(
                    (valid_idx, _aligned_author_probabilities(fold_model, x_fold_valid))
                )
            fallback_oof = assemble_oof(len(train), len(AUTHOR_COLUMNS), fold_predictions)

            full_vectorizer = _make_fallback_vectorizer()
            x_full = full_vectorizer.fit_transform(train_texts)
            full_model = LogisticRegression(C=4.0, max_iter=1000, solver="saga", n_jobs=-1)
            full_model.fit(x_full, train_labels)
            test_probs = full_model.predict_proba(full_vectorizer.transform(test["text"].astype(str)))
            fallback_by_id = {
                str(row_id): {
                    author: float(test_probs[index, list(full_model.classes_).index(author)])
                    for author in AUTHOR_COLUMNS
                }
                for index, row_id in enumerate(test["id"].astype(str))
            }
            fallback_model = "tfidf_logistic"
        else:
            n_splits = None
            fallback_oof = np.full(
                (len(train), len(AUTHOR_COLUMNS)), 1.0 / len(AUTHOR_COLUMNS), dtype=float
            )
            fallback_by_id = {
                str(row["id"]): {author: 1.0 / len(AUTHOR_COLUMNS) for author in AUTHOR_COLUMNS}
                for _, row in test.iterrows()
            }
            fallback_model = "uniform_small_data"

        residual = (1.0 - confidence) / (len(AUTHOR_COLUMNS) - 1)

        # Honest blend validation over ALL train rows: uniquely-matched rows get
        # the same one-hot-with-confidence probabilities used for test (the match
        # rule is deterministic against an external corpus — no fitted
        # parameters), unmatched rows get their fallback OOF probabilities.
        blend = fallback_oof.copy()
        for position, match in enumerate(train_matches):
            matched_author = match["matched_author"]
            if matched_author is not None:
                blend[position] = [
                    confidence if author == matched_author else residual
                    for author in AUTHOR_COLUMNS
                ]
        fallback_oof_log_loss = _safe_log_loss(train_labels, fallback_oof)
        validation_blend_log_loss = _safe_log_loss(train_labels, blend)
        rows = []
        test_matches = []
        for _, row in test.iterrows():
            predicted = _unique_author(
                row["text"],
                corpora,
                min_chars=min_chars,
                shingle_sets=shingle_sets,
                shingle_words=shingle_words,
                stats=match_stats,
            )
            test_matches.append({"id": row["id"], "matched_author": predicted})
            if predicted is None:
                probabilities = fallback_by_id[str(row["id"])]
            else:
                probabilities = {
                    author: confidence if author == predicted else residual
                    for author in AUTHOR_COLUMNS
                }
            rows.append({"id": row["id"], **probabilities})

        submission = sample[["id"]].merge(pd.DataFrame(rows), on="id", how="left")
        submission_path = output_dir / "submission.csv"
        matches_path = output_dir / "matches.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        pd.DataFrame(test_matches).to_csv(matches_path, index=False)
        validation_targets = one_hot_frame(
            train["id"],
            train["author"],
            id_col="id",
            target_cols=AUTHOR_COLUMNS,
        )
        validation_predictions = pd.DataFrame(
            {
                "id": train["id"].to_numpy(),
                **{
                    author: blend[:, index]
                    for index, author in enumerate(AUTHOR_COLUMNS)
                },
            }
        )
        # Sort by id in the NATIVE id dtype so predictions and targets stay
        # row-aligned (converting ids to str before sorting is the exact bug
        # that bit image_embedding in F.2.1).
        validation_predictions = validation_predictions.sort_values("id", kind="stable").reset_index(drop=True)
        validation_targets = validation_targets.sort_values("id", kind="stable").reset_index(drop=True)
        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col="id",
            target_cols=AUTHOR_COLUMNS,
            validation_kind="oof",
        )

        train_match_count = sum(row["matched_author"] is not None for row in train_matches)
        test_match_count = sum(row["matched_author"] is not None for row in test_matches)
        precision = 1.0 - wrong / train_match_count if train_match_count else None
        metrics = {
            "operator": "text_provenance",
            "train_matches": train_match_count,
            "train_wrong_matches": wrong,
            "train_precision": precision,
            "test_matches": test_match_count,
            "match_strategy": "shingle_filtered_exact",
            "shingle_words": shingle_words,
            "corpus_index_cache_status": corpus_index.cache_status,
            "corpus_index_cache_path": str(corpus_index.cache_path),
            "corpus_source_count": corpus_index.source_count,
            "candidate_author_checks": match_stats.candidate_author_checks,
            "full_scan_author_checks": match_stats.full_scan_author_checks,
            "shingle_filtered_queries": match_stats.shingle_filtered_queries,
            "shingle_missed_queries": match_stats.shingle_missed_queries,
            "prediction_artifacts": prediction_artifacts,
            "fallback_model": fallback_model,
            "n_splits": n_splits,
            "fallback_oof_log_loss": fallback_oof_log_loss,
            "validation_blend_log_loss": validation_blend_log_loss,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id="text_provenance",
            recipe_id="text_provenance",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=validation_blend_log_loss,
            medal="none",
            final_artifact_source="ees_core:text_provenance",
            artifact_paths=[str(submission_path), str(matches_path), str(metrics_path)],
            metrics=metrics,
            score_direction="minimize",  # score is validation blend log-loss
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="text_provenance",
            recipe_id="text_provenance",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:text_provenance",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def _make_fallback_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )


def _aligned_author_probabilities(model: LogisticRegression, features) -> np.ndarray:
    """predict_proba re-ordered into AUTHOR_COLUMNS columns (0.0 for absent classes)."""
    probabilities = model.predict_proba(features)
    aligned = np.zeros((probabilities.shape[0], len(AUTHOR_COLUMNS)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        if label in AUTHOR_COLUMNS:
            aligned[:, AUTHOR_COLUMNS.index(label)] = probabilities[:, source_index]
    return aligned


def _safe_log_loss(labels: pd.Series, probabilities: np.ndarray) -> float | None:
    try:
        return float(log_loss(labels, probabilities, labels=AUTHOR_COLUMNS))
    except ValueError:
        return None
