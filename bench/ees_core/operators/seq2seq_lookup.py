"""General seq2seq lookup operator for token-level normalization tasks.

Ports the exact-match / most-frequent-fallback lookup approach that already
earns a bronze medal on text-normalization-english (0.99059 vs. a
0.98784-class bronze threshold) into a reusable, non-task-specific EES
operator.

Detection is structural only -- no task-id and no hardcoded column-name
special-casing. It looks for:
  1. an (input_col, output_col) pair, learned from the data, where a large
     fraction of train rows are identity (``input == output``); and
  2. a "sentence + position" id structure that reconstructs the sample
     submission's id scheme (either the id column is present directly, or it
     is a composite of two integer-like columns, e.g. ``f"{a}_{b}"``).

Both signals are evidence-only: any two string columns / any two integer
columns can play these roles, so long as the statistics hold up. Datasets
that don't have this shape (e.g. plain numeric tabular data) fail detection
cheaply, without building anything or raising.

The model is: for each input token, look up the most-frequent observed
output for that exact token; optionally refine with a (previous_token,
token) context key when it beats the plain unigram lookup on held-out data;
fall back to identity for tokens never seen in training.

Validation is honest: a GroupShuffleSplit on the detected "sentence" group
column (so no sentence straddles the split) builds the lookup on the train
side only and scores holdout accuracy. The final lookup used for test
predictions is rebuilt on ALL train rows.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

MIN_IDENTITY_RATE = 0.3
HOLDOUT_FRACTION = 0.2
MATCH_TARGET_COL = "match"
BOS_TOKEN = "<BOS>"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ResolvedLookupFiles:
    train: Path
    test: Path
    sample_submission: Path


# Bounded head read used by the resolver's consistency check (finding the
# candidate id columns never needs the full multi-hundred-MB train/test file).
RESOLVER_PROBE_ROWS = 50_000
# Combinatorial safety valve: at most this many candidates per role are tried,
# in preference order (canonical name first). Real dirs have 1-2 per role.
MAX_ROLE_CANDIDATES = 3


def resolve_lookup_files(data_dir: Path) -> ResolvedLookupFiles | None:
    """Find a mutually CONSISTENT train/test/sample-submission triple.

    Real prepared task dirs are messy in two independent ways:
      1. naming variants -- text-normalization ships ``en_train.csv.zip`` /
         ``en_test_2.csv.zip`` / ``en_sample_submission_2.csv`` rather than
         plain ``train.csv``/``test.csv``/``sample_submission.csv``; and
      2. stale duplicates -- a dir can carry BOTH a plain ``test.csv`` (from
         an older prepare) and the original ``en_test_2.csv.zip``, where the
         plain file is a DIFFERENT test version whose id space does not match
         the sample submission. Picking by name alone then produces a
         submission that cannot cover every sample id (the live round-3
         ``submission_ids_unmatched`` failure).

    So resolution is consistency-driven, not name-driven: candidates per role
    are matched by role token in the normalized basename (structure only --
    no language or task-specific strings; ``.csv`` and ``.csv.zip`` accepted,
    non-zip preferred among siblings), then combinations are tried in
    preference order (canonical exact name first, as a tie-break among
    consistent combos) and the first combination whose test file can
    construct EVERY sample-submission id wins. Only if no combination is
    consistent do we give up.
    """
    data_dir = Path(data_dir)
    candidates: dict[str, Path] = {}
    # plain .csv first so it wins the dedup against its own .csv.zip sibling
    for path in [*sorted(data_dir.glob("*.csv")), *sorted(data_dir.glob("*.csv.zip"))]:
        base = path.name[:-4] if path.name.lower().endswith(".zip") else path.name
        if base not in candidates:
            candidates[base] = path
    roles: dict[str, list[Path]] = {"train": [], "test": [], "sample_submission": []}
    for base, path in sorted(candidates.items()):
        normalized = base.lower().replace("_", "").replace("-", "")
        if "samplesubmission" in normalized:
            roles["sample_submission"].append(path)
        elif "train" in normalized:
            roles["train"].append(path)
        elif "test" in normalized:
            roles["test"].append(path)
    if any(not paths for paths in roles.values()):
        return None

    canonical = {"train": "train.csv", "test": "test.csv",
                 "sample_submission": "sample_submission.csv"}
    ordered = {
        role: sorted(paths, key=lambda p: (p.name != canonical[role], p.name))[:MAX_ROLE_CANDIDATES]
        for role, paths in roles.items()
    }

    # Consistency depends only on the (test, sample) pair; cache per pair so
    # multiple train candidates don't repeat file reads.
    consistency: dict[tuple[Path, Path], bool] = {}
    for train_path in ordered["train"]:
        for test_path in ordered["test"]:
            for sample_path in ordered["sample_submission"]:
                key = (test_path, sample_path)
                if key not in consistency:
                    consistency[key] = _test_sample_ids_consistent(test_path, sample_path)
                if consistency[key]:
                    return ResolvedLookupFiles(
                        train=train_path,
                        test=test_path,
                        sample_submission=sample_path,
                    )
    return None


def _test_sample_ids_consistent(test_path: Path, sample_path: Path) -> bool:
    """True iff the test file can construct EVERY sample-submission id.

    This is the sample-side coverage guarantee the operator ultimately needs
    (an uncovered sample id means an unfillable submission row). Column
    discovery uses a bounded head read; the coverage check then reads only
    the identified id columns of the full test file (cheap even for
    million-row zipped tests).
    """
    try:
        sample = pd.read_csv(sample_path)
        test_head = pd.read_csv(test_path, nrows=RESOLVER_PROBE_ROWS)
    except Exception:
        return False
    if sample.empty or sample.shape[1] < 1 or test_head.empty:
        return False
    id_col = str(sample.columns[0])
    sample_ids = set(sample[id_col].astype(str))

    if id_col in test_head.columns:
        try:
            test_ids = set(pd.read_csv(test_path, usecols=[id_col])[id_col].astype(str))
        except Exception:
            return False
        if sample_ids.issubset(test_ids):
            return True

    pair = _find_composite_id_pair(test_head, sample_ids)
    if pair is None:
        return False
    a, b, sep = pair
    try:
        full = pd.read_csv(test_path, usecols=[a, b])
        constructed = set(_compose_ids(full[a], full[b], sep))
    except Exception:
        return False
    return sample_ids.issubset(constructed)


@dataclass(frozen=True)
class LookupStructure:
    input_col: str
    output_col: str
    id_col: str
    group_col: str | None
    id_cols: tuple[str, str] | None  # (group_col, position_col) if id is composite; None if id_col is direct
    id_separator: str
    identity_rate: float


def _string_values(series: pd.Series) -> pd.Series:
    """Canonicalize a value column to real strings with missing -> "".

    pandas 2's ``astype(str)`` stringifies NaN to the literal ``"nan"``;
    pandas 3's ``str`` dtype PRESERVES missing values through ``astype(str)``.
    The real en text-normalization test has rows with a missing ``before``
    token, so under pandas 3 NaN flowed through the lookup into the
    predictions and tripped the submission NaN guard (live-gate failure).
    Filling with "" first gives identical, version-independent semantics:
    a missing token is the empty string (which is also the data-faithful
    reading -- train's dominant mapping for missing ``before`` is missing
    ``after``).
    """
    return series.fillna("").astype(str)


def _identity_rate(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0:
        return 0.0
    return float((_string_values(a) == _string_values(b)).mean())


def _is_int_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return bool((numeric == numeric.round()).all())


def _int_like_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _is_int_like(df[col])]


def _is_text_dtype(series: pd.Series) -> bool:
    # pandas 3 defaults text columns to `str` dtype and pandas 2 users may
    # carry StringDtype — `dtype != object` silently rejects both (the June
    # task_detection str-dtype bug family). Accept any string-like dtype.
    from pandas.api.types import is_object_dtype, is_string_dtype

    return is_object_dtype(series) or is_string_dtype(series)


def _find_input_output_columns(
    train: pd.DataFrame, test: pd.DataFrame, id_col: str, target_col: str
) -> tuple[str, float] | None:
    if target_col not in train.columns:
        return None
    if not _is_text_dtype(train[target_col]):
        return None
    shared = [col for col in train.columns if col in test.columns and col not in {target_col, id_col}]
    best_col: str | None = None
    best_rate = 0.0
    for col in shared:
        if not _is_text_dtype(train[col]):
            continue
        values = train[col].dropna().astype(str)
        if values.nunique() < 2:
            continue
        rate = _identity_rate(train[col], train[target_col])
        if rate >= MIN_IDENTITY_RATE and rate > best_rate:
            best_rate = rate
            best_col = col
    if best_col is None:
        return None
    return best_col, best_rate


def _compose_ids(a: pd.Series, b: pd.Series, sep: str) -> pd.Series:
    return (
        pd.to_numeric(a).astype("Int64").astype(str)
        + sep
        + pd.to_numeric(b).astype("Int64").astype(str)
    )


def _find_composite_id_pair(
    test: pd.DataFrame,
    sample_ids: set[str],
    *,
    allowed_cols: set[str] | None = None,
) -> tuple[str, str, str] | None:
    """Find ``(col_a, col_b, sep)`` such that ``f"{a}{sep}{b}"`` over the given
    test rows lands entirely inside the sample-submission id set.

    Coverage is measured on the CONSTRUCTED side: every constructed id must
    exist in ``sample_ids``. Measured this way (and not sample-side) so
    callers may pass a bounded prefix of test (probe and resolver read
    nrows-limited frames to avoid fully reading multi-hundred-MB files) and
    the check still holds; the full sample-side guarantee is enforced
    separately by the resolver's consistency check and the operator's
    submission guard.
    """
    candidates = [
        col
        for col in _int_like_columns(test)
        if allowed_cols is None or col in allowed_cols
    ]
    best: tuple[str, str, str] | None = None
    best_overlap = 0.0
    for sep in ("_", "-"):
        for a in candidates:
            for b in candidates:
                if a == b:
                    continue
                try:
                    constructed = _compose_ids(test[a], test[b], sep)
                except (TypeError, ValueError):
                    continue
                constructed_ids = set(constructed)
                if not constructed_ids:
                    continue
                overlap = len(constructed_ids & sample_ids) / len(constructed_ids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (a, b, sep)
    if best is None or best_overlap < 1.0:
        return None
    return best


def _find_id_construction(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, id_col: str
) -> tuple[str | None, tuple[str, str] | None, str]:
    """Return ``(direct_id_col, (group_col, position_col), separator)``.

    Exactly one of ``direct_id_col`` / the composite pair is non-None (or
    both are None when no id structure can be recovered).
    """
    if id_col in test.columns and id_col in train.columns:
        return id_col, None, ""
    if sample.empty:
        return None, None, ""

    sample_ids = set(sample[id_col].astype(str))
    best = _find_composite_id_pair(test, sample_ids, allowed_cols=set(train.columns))
    if best is None:
        return None, None, ""
    a, b, sep = best
    # The group ("sentence") column has more distinct values than the
    # within-group position column, which is a small bounded range (e.g.
    # 0..max_sentence_length) reused across many groups.
    if train[a].nunique(dropna=True) >= train[b].nunique(dropna=True):
        group_col, position_col = a, b
    else:
        group_col, position_col = b, a
    return None, (group_col, position_col), sep


def detect_lookup_structure(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> LookupStructure | None:
    """Evidence-only detection of the exact-match lookup structure.

    Returns ``None`` (cheaply, no crash) when the structure isn't present --
    e.g. plain numeric tabular data with no shared high-identity string
    column, or no recoverable id/group structure.
    """
    if sample.shape[1] < 2:
        return None
    id_col = str(sample.columns[0])
    target_col = str(sample.columns[1])

    io = _find_input_output_columns(train, test, id_col, target_col)
    if io is None:
        return None
    input_col, identity_rate = io

    direct_id, composite, separator = _find_id_construction(train, test, sample, id_col)
    if direct_id is None and composite is None:
        return None

    if composite is not None:
        group_col = composite[0]
    else:
        exclude = {id_col, input_col, target_col}
        group_candidates = [
            col
            for col in train.columns
            if col in test.columns and col not in exclude and 0 < train[col].nunique(dropna=True) < len(train)
        ]
        group_col = group_candidates[0] if group_candidates else None

    return LookupStructure(
        input_col=input_col,
        output_col=target_col,
        id_col=id_col,
        group_col=group_col,
        id_cols=composite,
        id_separator=separator,
        identity_rate=identity_rate,
    )


def _build_ids(df: pd.DataFrame, structure: LookupStructure) -> pd.Series:
    if structure.id_cols is not None:
        a, b = structure.id_cols
        return _compose_ids(df[a], df[b], structure.id_separator)
    return df[structure.id_col].astype(str)


def _group_values(df: pd.DataFrame, structure: LookupStructure) -> np.ndarray:
    if structure.group_col is not None and structure.group_col in df.columns:
        return _string_values(df[structure.group_col]).to_numpy()
    return np.arange(len(df)).astype(str)


def _previous_token_series(df: pd.DataFrame, structure: LookupStructure) -> np.ndarray:
    """Previous input-token within the same group, aligned to ``df``'s row order.

    Falls back to a beginning-of-sequence sentinel when there is no group
    column, or for the first token of each group.
    """
    tokens = _string_values(df[structure.input_col]).to_numpy()
    n = len(df)
    if structure.group_col is None or structure.group_col not in df.columns:
        return np.full(n, BOS_TOKEN, dtype=object)

    sort_keys = [structure.group_col]
    if structure.id_cols is not None:
        sort_keys.append(structure.id_cols[1])
    order_frame = df[sort_keys].copy()
    order_frame["_orig"] = np.arange(n)
    order = order_frame.sort_values(sort_keys + ["_orig"], kind="stable")["_orig"].to_numpy()

    sorted_groups = _string_values(df[structure.group_col]).to_numpy()[order]
    sorted_tokens = tokens[order]
    prev_sorted = np.empty(n, dtype=object)
    prev_sorted[0] = BOS_TOKEN
    if n > 1:
        prev_sorted[1:] = sorted_tokens[:-1]
        same_group = sorted_groups[1:] == sorted_groups[:-1]
        prev_sorted[1:][~same_group] = BOS_TOKEN

    result = np.empty(n, dtype=object)
    result[order] = prev_sorted
    return result


def _build_lookup_tables(
    inputs: np.ndarray, prev_inputs: np.ndarray, outputs: np.ndarray
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    unigram_counter: dict[str, Counter] = {}
    context_counter: dict[tuple[str, str], Counter] = {}
    for prev, inp, out in zip(prev_inputs, inputs, outputs):
        unigram_counter.setdefault(inp, Counter())[out] += 1
        context_counter.setdefault((prev, inp), Counter())[out] += 1
    unigram_map = {key: counter.most_common(1)[0][0] for key, counter in unigram_counter.items()}
    context_map = {key: counter.most_common(1)[0][0] for key, counter in context_counter.items()}
    return unigram_map, context_map


def _predict(
    inputs: np.ndarray,
    prev_inputs: np.ndarray,
    unigram_map: dict[str, str],
    context_map: dict[tuple[str, str], str],
    *,
    use_context: bool,
) -> np.ndarray:
    if use_context:
        return np.array(
            [
                context_map.get((prev, inp), unigram_map.get(inp, inp))
                for prev, inp in zip(prev_inputs, inputs)
            ],
            dtype=object,
        )
    return np.array([unigram_map.get(inp, inp) for inp in inputs], dtype=object)


def _holdout_split(n: int, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
    if len(unique_groups) < 2 or n < 5:
        return next(splitter.split(np.zeros(n), groups=np.arange(n)))
    return next(splitter.split(np.zeros(n), groups=groups))


def _failure(reason: str) -> CandidateResult:
    return CandidateResult(
        candidate_id="seq2seq_lookup",
        recipe_id="seq2seq_lookup",
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source="ees_core:seq2seq_lookup",
        no_score_reason=reason,
    )


def run_seq2seq_lookup_operator(data_dir: Path, output_dir: Path) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = resolve_lookup_files(data_dir)
    if files is None:
        return _failure("missing_or_ambiguous_data_files")
    try:
        train = pd.read_csv(files.train)
        test = pd.read_csv(files.test)
        sample = pd.read_csv(files.sample_submission)
    except Exception as exc:
        return _failure(f"unreadable_data_files:{type(exc).__name__}: {exc}")

    structure = detect_lookup_structure(train, test, sample)
    if structure is None:
        return _failure("not_a_lookup_task")

    try:
        inputs = _string_values(train[structure.input_col]).to_numpy()
        outputs = _string_values(train[structure.output_col]).to_numpy()
        prev_inputs = _previous_token_series(train, structure)
        groups = _group_values(train, structure)
        n_train = len(train)

        train_idx, holdout_idx = _holdout_split(n_train, groups)

        # Train-side lookup only, for honest holdout scoring.
        unigram_map_ho, context_map_ho = _build_lookup_tables(
            inputs[train_idx], prev_inputs[train_idx], outputs[train_idx]
        )
        holdout_true = outputs[holdout_idx]
        holdout_inputs = inputs[holdout_idx]
        holdout_prev = prev_inputs[holdout_idx]

        identity_pred = holdout_inputs
        unigram_pred = _predict(holdout_inputs, holdout_prev, unigram_map_ho, context_map_ho, use_context=False)
        context_pred = _predict(holdout_inputs, holdout_prev, unigram_map_ho, context_map_ho, use_context=True)

        identity_accuracy = float((identity_pred == holdout_true).mean())
        unigram_accuracy = float((unigram_pred == holdout_true).mean())
        context_accuracy = float((context_pred == holdout_true).mean())

        use_context = context_accuracy > unigram_accuracy
        holdout_accuracy = context_accuracy if use_context else unigram_accuracy
        holdout_pred_final = context_pred if use_context else unigram_pred

        # Rebuild on ALL train rows for the final test predictions.
        unigram_map_full, context_map_full = _build_lookup_tables(inputs, prev_inputs, outputs)

        test_inputs = _string_values(test[structure.input_col]).to_numpy()
        test_prev = _previous_token_series(test, structure)
        test_pred = _predict(test_inputs, test_prev, unigram_map_full, context_map_full, use_context=use_context)
        test_lookup_hit = np.array(
            [
                (use_context and (prev, inp) in context_map_full) or (inp in unigram_map_full)
                for prev, inp in zip(test_prev, test_inputs)
            ],
            dtype=float,
        )

        test_ids = _build_ids(test, structure)
        # Fill sample-order predictions via an id->prediction map (string-keyed
        # on both sides, so an int-dtype sample id column cannot silently
        # mismatch string-built composite ids the way an id merge would).
        prediction_by_id = dict(zip(test_ids, test_pred))
        submission = sample[[structure.id_col]].copy()
        submission[structure.output_col] = submission[structure.id_col].astype(str).map(prediction_by_id)
        if submission[structure.output_col].isna().any():
            return _failure("submission_ids_unmatched")
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)

        train_ids = _build_ids(train, structure)
        validation_targets = pd.DataFrame(
            {structure.id_col: train_ids.to_numpy()[holdout_idx], MATCH_TARGET_COL: 1.0}
        )
        validation_predictions = pd.DataFrame(
            {
                structure.id_col: train_ids.to_numpy()[holdout_idx],
                MATCH_TARGET_COL: (holdout_pred_final == holdout_true).astype(float),
            }
        )
        test_predictions_artifact = pd.DataFrame(
            {structure.id_col: test_ids.to_numpy(), MATCH_TARGET_COL: test_lookup_hit}
        )

        prediction_artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=test_predictions_artifact,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=structure.id_col,
            target_cols=[MATCH_TARGET_COL],
            validation_kind="holdout",
        )

        metrics = {
            "operator": "seq2seq_lookup",
            "train_file": str(files.train),
            "test_file": str(files.test),
            "sample_submission_file": str(files.sample_submission),
            "input_col": structure.input_col,
            "output_col": structure.output_col,
            "id_col": structure.id_col,
            "group_col": structure.group_col,
            "id_cols": list(structure.id_cols) if structure.id_cols else None,
            "id_separator": structure.id_separator,
            "train_identity_rate": structure.identity_rate,
            "identity_baseline_accuracy": identity_accuracy,
            "unigram_holdout_accuracy": unigram_accuracy,
            "context_holdout_accuracy": context_accuracy,
            "used_context_refinement": use_context,
            "holdout_accuracy": holdout_accuracy,
            "holdout_rows": int(len(holdout_idx)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "prediction_artifacts": prediction_artifacts,
            "test_predictions_match_semantics": (
                "1.0 if the token was found in the (context or unigram) lookup, else 0.0 "
                "(identity fallback); this is a coverage proxy, NOT a ground-truth correctness "
                "signal -- test labels are unknown."
            ),
            "runtime_seconds": round(time.time() - started, 3),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

        return CandidateResult(
            candidate_id="seq2seq_lookup",
            recipe_id="seq2seq_lookup",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=holdout_accuracy,
            medal="none",
            final_artifact_source="ees_core:seq2seq_lookup",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")
