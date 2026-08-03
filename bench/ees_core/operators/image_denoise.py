"""Image-to-image denoising operator for per-pixel regression tasks.

Shape (detected GENERALLY, no task-id): a task whose sample submission melts each
test image into one row PER PIXEL -- ids parse as ``<imageId>_<row>_<col>`` with a
single numeric value column -- AND whose training data ships paired *dirty* input
images alongside a parallel *cleaned* / ground-truth image directory (same file
names). This is the classic "denoising dirty documents" shape (RMSE on pixel
intensities, lower is better).

Pipeline (CPU-cheap, no torch): estimate the page background with a large-kernel
box blur and build per-pixel neighborhood features (the pixel + local
mean/median/min/max/std in a small window + the large-kernel background estimate +
background-subtracted and background-normalized signals). A bounded gradient-boosted
regressor (sklearn ``HistGradientBoostingRegressor``, squared-error loss) is trained
on paired (dirty->clean) pixels and predicts cleaned test pixels, clipped to [0, 1].

Validation is HONEST: whole train images are held out (never a random split of
pixels from an image that also trains), the reported score is the holdout RMSE on
those held-out images, and the exported validation artifacts are exactly the pixels
that score was computed on. The final test submission is produced by a model refit
on ALL train images.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bench.adapters.task_detection import IMAGE_EXTS, _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

# id like "1_2_3" -> image "1", row 2, col 3. The image token may itself contain
# underscores; only the trailing two integer tokens are row/col, so split from the right.
_PIXEL_ID_RE = re.compile(r"^(?P<img>.+)_(?P<row>\d+)_(?P<col>\d+)$")

# Directory-name signal for the ground-truth ("cleaned") side of the training pairs.
_CLEAN_DIR_RE = re.compile(r"clean|denoise|target|ground.?truth|\bgt\b|label", re.IGNORECASE)

_BG_WINDOW = int(os.environ.get("EES_DENOISE_BG_WINDOW", "25"))
_SMALL_WINDOW = int(os.environ.get("EES_DENOISE_SMALL_WINDOW", "3"))
_NBR_WINDOW = int(os.environ.get("EES_DENOISE_NBR_WINDOW", "5"))
_MAX_TRAIN_PIXELS = int(os.environ.get("EES_DENOISE_MAX_TRAIN_PIXELS", "600000"))
_MAX_HOLDOUT_PIXELS = int(os.environ.get("EES_DENOISE_MAX_HOLDOUT_PIXELS", "200000"))
_HOLDOUT_FRACTION = float(os.environ.get("EES_DENOISE_HOLDOUT_FRACTION", "0.2"))


@dataclass(frozen=True)
class DenoiseTask:
    id_col: str
    value_col: str
    sample_submission: Path
    train_dir: Path  # dirty training inputs
    clean_dir: Path  # ground-truth cleaned images (paired with train_dir by filename)
    test_dir: Path  # dirty test inputs
    image_ids: list[str]  # unique test image ids, in first-seen submission order


def _pixel_id_signal(sample_submission: Path, *, probe_rows: int = 400) -> bool:
    """True iff the sample submission is a per-pixel (image_row_col, value) contract."""
    try:
        head = pd.read_csv(sample_submission, nrows=probe_rows)
    except Exception:
        return False
    if head.shape[1] != 2:
        return False
    ids = head.iloc[:, 0].astype(str)
    if ids.empty:
        return False
    return bool(ids.map(lambda s: bool(_PIXEL_ID_RE.match(s))).all())


def _image_dirs(data_dir: Path) -> dict[Path, set[str]]:
    """Map each immediate subdirectory containing images -> set of image stems."""
    dirs: dict[Path, set[str]] = {}
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        stems = {p.stem for p in child.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        if stems:
            dirs[child] = stems
    return dirs


def detect_denoise_task(data_dir: Path) -> DenoiseTask | None:
    """Detect the paired-image per-pixel denoising shape, generally (no task-id).

    Signals (ALL required):
      1. sample submission ids parse as ``<img>_<row>_<col>`` with one value column;
      2. an image directory whose *name* marks it as cleaned/ground-truth exists and
         shares (nearly) all its file names with another image directory (the paired
         dirty inputs);
      3. a third image directory (not the pair) whose image stems match the test
         image ids in the submission (the dirty test inputs).
    """
    data_dir = Path(data_dir)
    sample = _find_sample_submission(data_dir)
    if sample is None or not _pixel_id_signal(sample):
        return None
    head = pd.read_csv(sample, nrows=1)
    id_col, value_col = str(head.columns[0]), str(head.columns[1])

    dirs = _image_dirs(data_dir)
    if len(dirs) < 2:
        return None

    # (2) clean dir = a name-marked dir with a heavy filename overlap partner.
    best_pair: tuple[Path, Path, float] | None = None
    for clean_dir, clean_stems in dirs.items():
        if not _CLEAN_DIR_RE.search(clean_dir.name):
            continue
        for train_dir, train_stems in dirs.items():
            if train_dir == clean_dir:
                continue
            overlap = len(clean_stems & train_stems) / max(1, len(clean_stems))
            if overlap >= 0.8 and (best_pair is None or overlap > best_pair[2]):
                best_pair = (train_dir, clean_dir, overlap)
    if best_pair is None:
        return None
    train_dir, clean_dir, _ = best_pair

    # (3) test dir = the remaining image dir whose stems best match submission image ids.
    image_ids = _submission_image_ids(sample, id_col)
    id_set = set(image_ids)
    best_test: tuple[Path, float] | None = None
    for cand_dir, cand_stems in dirs.items():
        if cand_dir in (train_dir, clean_dir):
            continue
        match = len(id_set & cand_stems) / max(1, len(id_set))
        if match >= 0.5 and (best_test is None or match > best_test[1]):
            best_test = (cand_dir, match)
    if best_test is None:
        return None

    return DenoiseTask(
        id_col=id_col,
        value_col=value_col,
        sample_submission=sample,
        train_dir=train_dir,
        clean_dir=clean_dir,
        test_dir=best_test[0],
        image_ids=image_ids,
    )


def _submission_image_ids(sample_submission: Path, id_col: str) -> list[str]:
    """Unique image ids in first-seen submission order (the required output order)."""
    ids = pd.read_csv(sample_submission, usecols=[id_col])[id_col].astype(str)
    imgs = ids.str.rsplit("_", n=2, expand=True)[0]
    return list(dict.fromkeys(imgs.tolist()))


def _load_gray01(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return arr / 255.0


def _features(img01: np.ndarray) -> np.ndarray:
    """Per-pixel neighborhood features for a HxW grayscale image -> (H*W, n_feat)."""
    from scipy import ndimage as ndi

    x = img01
    mean_s = ndi.uniform_filter(x, size=_SMALL_WINDOW)
    sq_s = ndi.uniform_filter(x * x, size=_SMALL_WINDOW)
    std_s = np.sqrt(np.clip(sq_s - mean_s * mean_s, 0.0, None))
    med_s = ndi.median_filter(x, size=_SMALL_WINDOW)
    mn = ndi.minimum_filter(x, size=_NBR_WINDOW)
    mx = ndi.maximum_filter(x, size=_NBR_WINDOW)
    bg = ndi.uniform_filter(x, size=_BG_WINDOW)  # large-kernel page background
    sub = x - bg
    norm = x / (bg + 1e-3)
    feats = [x, mean_s, std_s, med_s, mn, mx, bg, sub, norm]
    return np.stack([f.ravel() for f in feats], axis=1).astype(np.float32)


def _stack_pairs(dirty_dir: Path, clean_dir: Path, stems: list[str]):
    """Build (features, targets) over all pixels of the given paired image stems."""
    feat_blocks, tgt_blocks = [], []
    dirty_by_stem = {p.stem: p for p in dirty_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    clean_by_stem = {p.stem: p for p in clean_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    for stem in stems:
        if stem not in dirty_by_stem or stem not in clean_by_stem:
            continue
        dirty = _load_gray01(dirty_by_stem[stem])
        clean = _load_gray01(clean_by_stem[stem])
        if dirty.shape != clean.shape:
            continue
        feat_blocks.append(_features(dirty))
        tgt_blocks.append(clean.ravel().astype(np.float32))
    if not feat_blocks:
        raise ValueError("no usable dirty/clean image pairs found")
    return np.vstack(feat_blocks), np.concatenate(tgt_blocks)


def _subsample(X: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator):
    if len(y) <= cap:
        return X, y, np.arange(len(y))
    keep = rng.choice(len(y), size=cap, replace=False)
    keep.sort()
    return X[keep], y[keep], keep


def _fit_regressor(X: np.ndarray, y: np.ndarray, random_state: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(os.environ.get("EES_DENOISE_MAX_ITER", "300")),
        learning_rate=float(os.environ.get("EES_DENOISE_LR", "0.1")),
        max_depth=None,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _predict01(model, img01: np.ndarray) -> np.ndarray:
    pred = model.predict(_features(img01))
    return np.clip(pred, 0.0, 1.0).reshape(img01.shape)


def run_image_denoise_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    metric: str | None = None,
    random_state: int = 101,
) -> CandidateResult:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        task = detect_denoise_task(Path(data_dir))
        if task is None:
            raise ValueError("image_denoise: paired per-pixel denoising shape not detected")
        rng = np.random.default_rng(random_state)

        # Whole-image holdout split (never split pixels within an image across fit/holdout).
        clean_stems = sorted(
            {p.stem for p in task.clean_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
            & {p.stem for p in task.train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        )
        if len(clean_stems) < 2:
            raise ValueError("image_denoise: need >= 2 paired train images")
        perm = rng.permutation(len(clean_stems))
        n_holdout = max(1, int(round(len(clean_stems) * _HOLDOUT_FRACTION)))
        holdout_stems = [clean_stems[i] for i in perm[:n_holdout]]
        fit_stems = [clean_stems[i] for i in perm[n_holdout:]] or holdout_stems

        # Holdout evaluation: fit on fit_stems, score on held-out whole images.
        Xf, yf = _stack_pairs(task.train_dir, task.clean_dir, fit_stems)
        Xf, yf, _ = _subsample(Xf, yf, _MAX_TRAIN_PIXELS, rng)
        holdout_model = _fit_regressor(Xf, yf, random_state)

        dirty_by_stem = {p.stem: p for p in task.train_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        clean_by_stem = {p.stem: p for p in task.clean_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        hold_ids, hold_pred, hold_true = [], [], []
        for stem in holdout_stems:
            dirty = _load_gray01(dirty_by_stem[stem])
            clean = _load_gray01(clean_by_stem[stem])
            pred = _predict01(holdout_model, dirty)
            H, W = dirty.shape
            rows, cols = np.meshgrid(np.arange(1, H + 1), np.arange(1, W + 1), indexing="ij")
            hold_ids.append(
                np.char.add(
                    np.char.add(np.char.add(f"{stem}_", rows.ravel().astype(str)), "_"),
                    cols.ravel().astype(str),
                )
            )
            hold_pred.append(pred.ravel())
            hold_true.append(clean.ravel())
        hold_ids = np.concatenate(hold_ids)
        hold_pred = np.concatenate(hold_pred).astype(np.float64)
        hold_true = np.concatenate(hold_true).astype(np.float64)

        # Cap holdout artifact rows; the reported score is computed on EXACTLY these rows.
        if len(hold_ids) > _MAX_HOLDOUT_PIXELS:
            sel = rng.choice(len(hold_ids), size=_MAX_HOLDOUT_PIXELS, replace=False)
            sel.sort()
            hold_ids, hold_pred, hold_true = hold_ids[sel], hold_pred[sel], hold_true[sel]
        holdout_rmse = float(np.sqrt(np.mean((hold_pred - hold_true) ** 2)))

        # Final model: refit on ALL paired train images, predict test.
        Xa, ya = _stack_pairs(task.train_dir, task.clean_dir, clean_stems)
        Xa, ya, _ = _subsample(Xa, ya, _MAX_TRAIN_PIXELS, rng)
        final_model = _fit_regressor(Xa, ya, random_state)

        test_by_stem = {p.stem: p for p in task.test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        pred_by_img: dict[str, np.ndarray] = {}
        for stem in task.image_ids:
            if stem in test_by_stem:
                pred_by_img[stem] = _predict01(final_model, _load_gray01(test_by_stem[stem]))

        # Emit submission in EXACT sample order, indexing per-image prediction arrays.
        sample = pd.read_csv(task.sample_submission)
        ids = sample[task.id_col].astype(str)
        parts = ids.str.rsplit("_", n=2, expand=True)
        img_tok = parts[0].to_numpy()
        row_idx = parts[1].astype(int).to_numpy() - 1
        col_idx = parts[2].astype(int).to_numpy() - 1
        values = np.full(len(sample), 1.0, dtype=np.float64)  # default white for any gap
        for img, positions in pd.Series(img_tok).groupby(img_tok, sort=False).indices.items():
            pred = pred_by_img.get(img)
            if pred is None:
                continue
            values[positions] = pred[row_idx[positions], col_idx[positions]]
        values = np.clip(values, 0.0, 1.0)
        sub = pd.DataFrame({task.id_col: sample[task.id_col], task.value_col: values})

        submission_path = output_dir / "submission.csv"
        sub.to_csv(submission_path, index=False)

        val_pred = pd.DataFrame({task.id_col: hold_ids, task.value_col: hold_pred})
        val_targets = pd.DataFrame({task.id_col: hold_ids, task.value_col: hold_true})
        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=sub,
            validation_predictions=val_pred,
            validation_targets=val_targets,
            id_col=task.id_col,
            target_cols=[task.value_col],
            validation_kind="holdout",
        )
        metrics = {
            "operator": "image_denoise",
            "task_id": task_id,
            "holdout_rmse": holdout_rmse,
            "n_train_images": len(clean_stems),
            "n_holdout_images": len(holdout_stems),
            "n_test_images": len(pred_by_img),
            "fit_pixels": int(len(ya)),
            "bg_window": _BG_WINDOW,
            "features": ["pixel", "mean", "std", "median", "min", "max", "bg", "bg_sub", "bg_norm"],
            "prediction_artifacts": artifacts,
            "validation": {
                "validation_kind": "holdout",
                "score": holdout_rmse,
                "objective": "rmse",
                "direction": "minimize",
            },
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="image_denoise",
            recipe_id="image_denoise",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(sub)), "columns": list(sub.columns)},
            score=holdout_rmse,
            medal="none",
            final_artifact_source="ees_core:image_denoise",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
            score_direction="minimize",
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="image_denoise",
            recipe_id="image_denoise",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_denoise",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
