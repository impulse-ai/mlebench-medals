"""Detect task schema from MLE-bench task files (sample_submission.csv + description.md).

The bench harness needs to know per task:
  - which column is the ID
  - which column(s) are the target(s)
  - the expected submission shape (column order)
  - the task description, to construct a deterministic Impulse prompt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}

# A column is treated as "free text" (→ NLP) only when its sampled values are
# long enough on average to be sentences rather than short categoricals. Names,
# cabin codes, and country labels stay well below these thresholds, so tabular
# tasks with a stray string column are not misclassified.
_NLP_MIN_MEAN_WORDS = 8
_NLP_MIN_MEAN_CHARS = 40


def detect_modality(data_dir) -> str:
    """Cheap modality probe — see :func:`detect_modality_and_text`."""
    return detect_modality_and_text(data_dir)[0]


def detect_modality_and_text(data_dir) -> tuple[str, str | None]:
    """Probe the task directory and return ``(modality, text_col)``.

    Priority order (a task may contain several file types):
      1. ``vision`` — any image file is present (recursively).
      2. ``audio``  — any audio file is present (recursively).
      3. ``nlp``    — ``train.csv`` has a dominant free-text column (long
                      sentences), e.g. detecting-insults-in-social-commentary.
                      ``text_col`` names that column.
      4. ``tabular``— everything else.

    Image/audio win over text because vision/audio tasks routinely ship a CSV
    manifest of labels alongside the media files.
    """
    data_dir = Path(data_dir)
    has_image = False
    has_audio = False
    zips: list[Path] = []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            has_image = True
            break  # vision is top priority — stop early
        if ext in AUDIO_EXTS:
            has_audio = True
        elif ext == ".zip":
            zips.append(p)

    # Media is often delivered zipped rather than as loose files (e.g. the
    # whale audio task ships train2.zip/test2.zip of .aiff clips). When no loose
    # media is found, peek at archive member names — central directory only, no
    # extraction — so these tasks are not mislabelled tabular.
    if not has_image and not has_audio and zips:
        kind = _media_kind_in_zips(zips)
        if kind == "vision":
            has_image = True
        elif kind == "audio":
            has_audio = True

    if has_image:
        return "vision", None
    if has_audio:
        return "audio", None

    text_col = _detect_text_column(data_dir)
    if text_col is not None:
        return "nlp", text_col
    return "tabular", None


def _media_kind_in_zips(zips: list[Path]) -> str | None:
    """Peek at zip member names (no extraction) and report 'vision'/'audio'/None.

    Vision wins over audio if both extensions appear. Reads only the archive's
    central directory via ``namelist()``, so it stays cheap even for large zips.
    """
    import zipfile

    has_audio = False
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    ext = Path(name).suffix.lower()
                    if ext in IMAGE_EXTS:
                        return "vision"
                    if ext in AUDIO_EXTS:
                        has_audio = True
        except Exception:
            continue
    return "audio" if has_audio else None


def _detect_text_column(data_dir: Path) -> str | None:
    """Return the name of the dominant free-text column in train.csv, or None.

    Excludes the id column and target column(s) (inferred from
    sample_submission.csv when present) so the target never counts as text.
    """
    train_path = data_dir / "train.csv"
    train_json_path = data_dir / "train.json"
    if not train_path.exists():
        if not train_json_path.exists():
            return None
        try:
            sample = pd.read_json(train_json_path).head(200)
        except Exception:
            return None
    else:
        try:
            sample = pd.read_csv(train_path, nrows=200)
        except Exception:
            return None

    exclude: set[str] = set()
    sub_path = data_dir / "sample_submission.csv"
    if sub_path.exists():
        try:
            sub_cols = list(pd.read_csv(sub_path, nrows=1).columns)
            exclude.update(sub_cols)  # id + target columns
        except Exception:
            pass
    if sample.columns.size:
        exclude.add(sample.columns[0])  # leading id column, defensively

    from pandas.api.types import is_object_dtype, is_string_dtype

    best_col: str | None = None
    best_words = 0.0
    for col in sample.columns:
        if col in exclude:
            continue
        series = sample[col]
        # pandas 3 defaults text columns to `str` dtype (not `object`), so a
        # `dtype != object` gate silently rejects every text column and NLP tasks
        # misroute to the tabular path. Accept any string-like dtype.
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue
        values = series.dropna().astype(str)
        if values.empty:
            continue
        mean_chars = values.str.len().mean()
        mean_words = values.str.split().str.len().mean()
        if mean_words >= _NLP_MIN_MEAN_WORDS and mean_chars >= _NLP_MIN_MEAN_CHARS:
            if mean_words > best_words:
                best_words = mean_words
                best_col = col
    return best_col


@dataclass
class TaskSchema:
    id_col: str
    target_cols: list[str]
    sample_submission_cols: list[str]
    n_train: int | None
    n_test: int | None
    description: str | None
    # The metric mlebench actually uses to grade. Critical to surface in the
    # prompt — without it, the LLM frequently picks a default like AUC for
    # binary classification even when the competition grades on accuracy
    # (or vice versa), leading to misleadingly high CV scores that don't
    # translate to a medal. Names come from mlebench.registry verbatim
    # (e.g. "accuracy", "auc-roc", "mean-column-wise-rmsle").
    metric: str | None = None
    # Coarse data modality, from detect_modality_and_text():
    # "vision" | "audio" | "nlp" | "tabular".
    modality: str = "tabular"
    # For NLP tasks: the dominant free-text column in train.csv (so the agent
    # prompt/preamble can name it deterministically). None for other modalities.
    text_col: str | None = None

    @property
    def is_multi_target(self) -> bool:
        return len(self.target_cols) > 1


def _find_sample_submission(data_dir: Path) -> Path | None:
    """Locate the sample-submission CSV, tolerating naming variants.

    MLE-bench tasks are inconsistent: most ship ``sample_submission.csv`` but
    some (e.g. the-icml-2013-whale-challenge-right-whale-redux) ship
    ``sampleSubmission.csv`` and text-normalization ships zipped language-
    prefixed samples. Try the common names, then fall back to any case-
    insensitive ``*sample*submission*.csv`` in the directory, extracting a
    matching single CSV from ``*.zip`` when needed.
    """
    for name in ("sample_submission.csv", "sampleSubmission.csv", "SampleSubmission.csv"):
        p = data_dir / name
        if p.exists():
            return p
    for p in sorted(data_dir.glob("*.csv")):
        low = p.name.lower().replace("_", "").replace("-", "")
        if "samplesubmission" in low:
            return p
    for p in sorted(data_dir.glob("*.zip")):
        low = p.name.lower().replace("_", "").replace("-", "")
        if "samplesubmission" not in low:
            continue
        extracted = _extract_sample_submission_zip(p)
        if extracted is not None:
            return extracted
    return None


def _extract_sample_submission_zip(zip_path: Path) -> Path | None:
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [
                name
                for name in zf.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() == ".csv"
                and "samplesubmission" in Path(name).name.lower().replace("_", "").replace("-", "")
            ]
            if not csv_members:
                csv_members = [
                    name
                    for name in zf.namelist()
                    if not name.endswith("/") and Path(name).suffix.lower() == ".csv"
                ]
            if len(csv_members) != 1:
                return None
            member = csv_members[0]
            destination = zip_path.with_name(Path(member).name)
            if not destination.exists():
                destination.write_bytes(zf.read(member))
            return destination
    except zipfile.BadZipFile:
        return None


def detect_schema(data_dir: Path) -> TaskSchema:
    """Detect the task schema from an MLE-bench prepared-data dir.

    Convention:
      - `sample_submission.csv` exists; first column is ID; rest are targets.
      - `train.csv` and `test.csv` exist (most tasks).
      - `description.md` exists (most tasks; competition description).
    """
    sample_sub_path = _find_sample_submission(data_dir)
    if sample_sub_path is None:
        raise FileNotFoundError(
            f"No sample-submission CSV found in {data_dir}. "
            f"Most MLE-bench tasks ship one (sample_submission.csv / "
            f"sampleSubmission.csv); verify the task is prepared."
        )

    sample = pd.read_csv(sample_sub_path, nrows=5)
    cols = list(sample.columns)
    if len(cols) < 2:
        raise ValueError(
            f"sample_submission.csv has {len(cols)} columns; expected ≥2 (id + target)."
        )

    id_col = cols[0]
    target_cols = cols[1:]

    description: str | None = None
    desc_path = data_dir / "description.md"
    if desc_path.exists():
        description = desc_path.read_text(errors="replace")

    def _row_count(p: Path) -> int | None:
        if not p.exists():
            json_path = p.with_suffix(".json")
            if not json_path.exists():
                return None
            try:
                return int(pd.read_json(json_path).shape[0])
            except Exception:
                return None
        # Cheap row count: read one column only.
        try:
            return int(pd.read_csv(p, usecols=[0]).shape[0])
        except Exception:
            return None

    # Resolve the grading metric from mlebench.registry. data_dir is
    # `<cache>/<task_id>/prepared/public`; the task id is two levels up.
    metric: str | None = None
    try:
        task_id = data_dir.parts[-3]
        from mlebench.registry import registry  # type: ignore
        grader = registry.get_competition(task_id).grader
        metric = getattr(grader, "name", None)
    except Exception:
        # Unknown task or mlebench import failed; carry on without it.
        metric = None

    modality, text_col = detect_modality_and_text(data_dir)
    return TaskSchema(
        id_col=id_col,
        target_cols=target_cols,
        sample_submission_cols=cols,
        n_train=_row_count(data_dir / "train.csv"),
        n_test=_row_count(data_dir / "test.csv"),
        description=description,
        metric=metric,
        modality=modality,
        text_col=text_col,
    )


def construct_training_prompt(schema: TaskSchema, task_id: str) -> str:
    """Build a deterministic prompt that keyword-routes to dataset_training.

    The gateway's chat route forces `tool_choice = dataset_training` when the
    message contains words like 'train' / 'fit' / 'build model' AND a dataset
    is active (per `gateway/src/routes/chat.ts:140-249`). We always start the
    prompt with 'Train' to be safe.
    """
    if schema.is_multi_target:
        target_str = ", ".join(schema.target_cols)
        target_phrase = f"Predict multiple targets: {target_str}."
    else:
        target_phrase = f"Target column: {schema.target_cols[0]}."

    base = (
        f"Train the best model you can on this dataset for the "
        f"'{task_id}' competition. {target_phrase} "
        f"Use thorough mode — try multiple algorithms and ensemble. "
        f"Optimize for the competition metric."
    )
    if schema.description:
        snippet = schema.description[:800].strip()
        base += f"\n\nCompetition context:\n{snippet}"
    return base


def construct_prediction_prompt(schema: TaskSchema) -> str:
    cols_str = ", ".join(schema.sample_submission_cols)
    return (
        f"Generate predictions on this test dataset using the most recently "
        f"trained model. Return a CSV with columns: {cols_str}."
    )
