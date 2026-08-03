"""Pretrained image-embedding operator with OOF-only backbone/calibration selection."""
from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bench.adapters.task_detection import IMAGE_EXTS, _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.operators.embedding_core import (
    EmbeddingBackbone,
    OofSelection,
    _embed_ids,
    _materialize_zip_member,
    blend_by_oof,
    fake_backbone,
    scorer_for_metric,
    select_by_oof,
    torchvision_backbones,
)
from bench.ees_core.operators.generic_media import MediaRef
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


@dataclass(frozen=True)
class ImageTask:
    id_col: str
    target_cols: list[str]
    train_ids: list
    y: np.ndarray
    classes: list[str]
    test_ids: list
    train_paths: dict
    test_paths: dict
    train_tab: "np.ndarray | None" = None
    test_tab: "np.ndarray | None" = None


def _classify_split(relpath: str) -> str:
    parts = relpath.replace("\\", "/").lower().split("/")
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return "both"


def _split_image_indices(data_dir: Path) -> tuple[dict, dict]:
    """Build ``(train_index, test_index)`` mapping image ids -> MediaRef, scoped by split.

    A whole-tree index keyed by bare filename/stem collapses cross-split collisions
    (e.g. ``train/1.jpg`` vs ``test/1.jpg``) with last-writer-wins, silently embedding the
    wrong split's image. Resolving train ids against the train index and test ids against
    the test index prevents that. Images not clearly under a train/ or test/ location
    (e.g. a shared ``images/`` directory) are added to both indices.
    """
    data_dir = Path(data_dir)
    train_index: dict = {}
    test_index: dict = {}

    def place(split: str, keys: list[str], ref: MediaRef) -> None:
        for key in keys:
            if split in ("train", "both"):
                train_index[key] = ref
            if split in ("test", "both"):
                test_index[key] = ref

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            rel = str(path.relative_to(data_dir))
            place(_classify_split(rel), [path.name, path.stem, rel], MediaRef(path=path))
        elif suffix == ".zip":
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                continue
            with archive:
                for member in archive.namelist():
                    if member.endswith("/") or Path(member).suffix.lower() not in IMAGE_EXTS:
                        continue
                    split = _classify_split(member)
                    if split == "both":
                        split = _classify_split(path.name)  # e.g. train.zip / test.zip
                    place(split, [Path(member).name, Path(member).stem, member], MediaRef(path=path, member=member))
    return train_index, test_index


def _train_labels(data_dir: Path, sample: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # Accept the common mle-bench / Kaggle name `train_labels.csv` (e.g.
    # histopathologic-cancer-detection) as an alias for `labels.csv` -- an id +
    # label table. Without this the loader falls through to the filename-token
    # fallback, which for hash-named images yields one bogus class per image.
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        alt = data_dir / "train_labels.csv"
        if alt.exists():
            labels_path = alt
    if labels_path.exists():
        frame = pd.read_csv(labels_path)
        first = frame.columns[0]
        label_col = next((c for c in frame.columns if c != first), None)
        return frame.rename(columns={first: id_col, label_col: "__label__"})[[id_col, "__label__"]]
    train_path = data_dir / "train.csv"
    if train_path.exists():
        frame = pd.read_csv(train_path)
        first = frame.columns[0]
        non_id = [c for c in frame.columns if c != first]
        target_cols = list(sample.columns[1:])
        # Binary tasks: the single target column IS the train label column (e.g. aerial's
        # `has_cactus`). Multiclass: the label column holds class names and is not a sample column.
        if len(target_cols) == 1 and target_cols[0] in non_id:
            label_col = target_cols[0]
        elif len(target_cols) > 1 and all(col in non_id for col in target_cols):
            # One-hot encoded multi-class train.csv (e.g. plant-pathology): the non-id
            # columns ARE the sample submission's target columns, holding 0/1 (or soft)
            # class membership per row, with no separate string label column. Picking
            # any single one-hot column as "the label" (the generic fallback below)
            # mistakes its 0/1 values for the whole label space -- decode the real
            # label per row via argmax across the one-hot block instead.
            renamed = frame.rename(columns={first: id_col})
            label_values = renamed[target_cols].astype(float).idxmax(axis=1)
            return renamed.assign(__label__=label_values)[[id_col, "__label__"]]
        else:
            label_col = next((c for c in non_id if c not in sample.columns), non_id[0] if non_id else None)
        if label_col is not None:
            renamed = frame.rename(columns={first: id_col})
            return renamed.assign(__label__=renamed[label_col])[[id_col, "__label__"]]
    # filename-token fallback: id encoded as "<label>.<n>"
    train_dir = data_dir / "train"
    rows = []
    for path in sorted(train_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        token = path.stem.split(".")[0]
        rows.append({id_col: path.stem, "__label__": token})
    if rows:
        return pd.DataFrame(rows)
    raise ValueError("could not resolve train labels (no labels.csv, train.csv label col, or filename tokens)")


def load_image_task(data_dir: Path) -> ImageTask:
    data_dir = Path(data_dir)
    sample_path = _find_sample_submission(data_dir)
    if sample_path is None:
        raise ValueError("sample submission not found")
    sample = pd.read_csv(sample_path)
    id_col = sample.columns[0]
    target_cols = list(sample.columns[1:])
    labels = _train_labels(data_dir, sample, id_col)
    label_values = [str(v) for v in labels["__label__"].tolist()]
    semantic = set(label_values) <= set(target_cols) and len(target_cols) > 1
    classes = target_cols if semantic else sorted(set(label_values))
    class_index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_index[v] for v in label_values])
    train_paths, test_paths = _split_image_indices(data_dir)
    test_ids = sample[id_col].tolist()
    train_ids = labels[id_col].tolist()
    train_tab, test_tab = _load_tabular_block(data_dir, train_ids, test_ids, id_col) or (None, None)
    return ImageTask(id_col, target_cols, train_ids, y, list(classes), test_ids, train_paths, test_paths, train_tab, test_tab)


def _load_tabular_block(data_dir: Path, train_ids: list, test_ids: list, id_col: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect and align a tabular feature block from train.csv/test.csv, if present.

    Feature columns are numeric columns present in BOTH files, excluding each file's
    own id column and the task ``id_col``. Rows are aligned to ``train_ids``/``test_ids``
    order by string id (csv ids may be int while task ids are str, or vice versa).
    Returns ``None`` if the csvs are missing, no shared numeric feature columns exist,
    or alignment produces NaNs (e.g. an id missing from the csv) so the caller can
    safely fall back to image-only embeddings.
    """
    tr_path = Path(data_dir) / "train.csv"
    te_path = Path(data_dir) / "test.csv"
    if not (tr_path.exists() and te_path.exists()):
        return None
    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    tr_id = tr.columns[0]
    te_id = te.columns[0]
    tr_num = set(tr.select_dtypes("number").columns)
    te_num = set(te.select_dtypes("number").columns)
    feats = [c for c in tr.columns if c in tr_num and c in te_num and c not in (tr_id, te_id, id_col)]
    if not feats:
        return None
    tr_map = tr.set_index(tr[tr_id].astype(str))[feats]
    te_map = te.set_index(te[te_id].astype(str))[feats]
    train_tab = tr_map.reindex([str(i) for i in train_ids]).to_numpy(dtype=float)
    test_tab = te_map.reindex([str(i) for i in test_ids]).to_numpy(dtype=float)
    if np.isnan(train_tab).any() or np.isnan(test_tab).any():
        return None
    return train_tab, test_tab


def _default_backbones() -> list[EmbeddingBackbone]:
    names = os.environ.get("EES_IMAGE_EMBED_BACKBONES", "resnet18,resnet50,convnext_small").split(",")
    return torchvision_backbones([n.strip() for n in names if n.strip()])


def run_image_embedding_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    backbones: list[EmbeddingBackbone] | None = None,
    metric: str | None = None,
    random_state: int = 101,
) -> CandidateResult:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        task = load_image_task(Path(data_dir))
        n_classes = len(task.classes)
        if n_classes < 2:
            raise ValueError("image_embedding requires >= 2 classes")
        used = backbones if backbones is not None else _default_backbones()
        max_rows = int(os.environ.get("EES_IMAGE_EMBED_MAX_TRAIN_ROWS", "8000"))
        train_ids, y, train_tab = task.train_ids, task.y, task.train_tab
        # Capping train rows here (or, upstream, filename-derived string ids) can make
        # image's OOF id set a strict subset of a full-OOF tabular operator's, so the
        # id sets won't be identical. That's safe by construction -- the registry just
        # drops the pair from the learned stack on id mismatch, it never mis-aligns --
        # but it does mean the F.2 cross-operator stack won't engage for that pair.
        # Tracked by the align-by-id follow-up.
        if len(train_ids) > max_rows:
            rng = np.random.default_rng(random_state)
            keep = rng.permutation(len(train_ids))[:max_rows]
            train_ids = [train_ids[i] for i in keep]
            y = y[keep]
            if train_tab is not None:
                train_tab = train_tab[keep]
        emb_by_backbone = {}
        for bb in used:
            emb, emb_test = _embed_ids(bb, train_ids, task.train_paths), _embed_ids(bb, task.test_ids, task.test_paths)
            train_feat = np.hstack([train_tab, emb]) if train_tab is not None else emb
            test_feat = np.hstack([task.test_tab, emb_test]) if task.test_tab is not None else emb_test
            emb_by_backbone[bb.name] = (train_feat, test_feat)
        # Cap CV folds by the smallest class count so StratifiedKFold is valid for
        # small or class-imbalanced tasks (min 2 folds, max 5).
        cv_splits = max(2, min(5, int(np.unique(y, return_counts=True)[1].min())))
        scorer = scorer_for_metric(metric)
        # Validation-reliable ENSEMBLE blend (not argmax-single-backbone): weights each
        # backbone by OOF quality but flattens toward uniform -- and disables temperature
        # sharpening -- when per-class test support is tiny, where OOF ranking is an
        # unreliable predictor of the test score. See embedding_core.blend_by_oof.
        sel = blend_by_oof(emb_by_backbone, y, n_classes, n_splits=cv_splits, random_state=random_state, scorer=scorer)

        # sample-shaped submission
        if len(task.target_cols) == 1:  # binary: single prob column = P(last class)
            sub = pd.DataFrame({task.id_col: task.test_ids, task.target_cols[0]: sel.test_proba[:, -1]})
        else:
            sub = pd.DataFrame(sel.test_proba, columns=task.classes)[task.target_cols]
            sub.insert(0, task.id_col, task.test_ids)

        # full-OOF validation artifacts for the selected config (consistent with sel.oof_score).
        # Note: `train_ids`/`y` here are the (possibly row-capped) ids actually fit/embedded
        # above, not the task's full un-capped `task.train_ids` -- they're what `sel.oof_proba`
        # is aligned to.
        oof = sel.oof_proba
        val_pred = pd.DataFrame(oof, columns=task.classes)
        val_pred.insert(0, task.id_col, list(train_ids))
        # Use the SAME native id values as val_pred (not str) so both frames sort
        # into an identical row order — otherwise a native-vs-str id dtype mismatch
        # sorts them differently (numeric vs lexicographic), leaving pred/target rows
        # misaligned and the registry silently dropping this candidate from the stack.
        val_targets = one_hot_frame(
            list(train_ids),
            [task.classes[c] for c in y],
            id_col=task.id_col,
            target_cols=task.classes,
        )
        if len(task.target_cols) == 1:
            val_pred = val_pred[[task.id_col, task.classes[-1]]].rename(columns={task.classes[-1]: task.target_cols[0]})
            val_targets = val_targets[[task.id_col, task.classes[-1]]].rename(columns={task.classes[-1]: task.target_cols[0]})
            report_targets = task.target_cols
        else:
            val_pred = val_pred[[task.id_col, *task.target_cols]]
            val_targets = val_targets[[task.id_col, *task.target_cols]]
            report_targets = task.target_cols
        # sort by id so cross-operator OOF records align (registry uses ordered id equality)
        val_pred = val_pred.sort_values(task.id_col, kind="stable").reset_index(drop=True)
        val_targets = val_targets.sort_values(task.id_col, kind="stable").reset_index(drop=True)

        submission_path = output_dir / "submission.csv"
        sub.to_csv(submission_path, index=False)
        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=sub,
            validation_predictions=val_pred,
            validation_targets=val_targets,
            id_col=task.id_col,
            target_cols=report_targets,
            validation_kind="oof",
        )
        metrics = {
            "operator": "image_embedding",
            "task_id": task_id,
            "selected_backbone": sel.backbone,
            "selected_c": sel.c,
            "selected_temperature": sel.temperature,
            "blend_weights": sel.weights,
            "blend_alpha": sel.alpha,
            "oof_log_loss": sel.oof_score,
            "scorer": scorer,
            "n_classes": n_classes,
            "fit_train_rows": int(len(train_ids)),
            "backbones": [bb.name for bb in used],
            "fused_tabular": task.train_tab is not None,
            "tabular_dims": int(task.train_tab.shape[1]) if task.train_tab is not None else 0,
            "prediction_artifacts": artifacts,
            "validation": {
                "validation_kind": "oof",
                "score": sel.oof_score,
                "objective": scorer,
                "direction": "maximize" if scorer in ("auc", "qwk") else "minimize",
            },
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="image_embedding",
            recipe_id="image_embedding",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(sub)), "columns": list(sub.columns)},
            score=sel.oof_score,
            medal="none",
            final_artifact_source=f"ees_core:image_embedding:{sel.backbone}",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
            score_direction="maximize" if scorer in ("auc", "qwk") else "minimize",
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="image_embedding",
            recipe_id="image_embedding",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_embedding",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
