"""Pretrained audio-embedding operator: log-spectrogram image -> reuse image embedding core."""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.operators.generic_media import MediaRef

AUDIO_EXTS = {".aif", ".aiff", ".wav"}


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    suf = path.suffix.lower()
    raw = path.read_bytes()
    if suf in (".aif", ".aiff"):
        import aifc  # lazy: aifc is removed in Python 3.13 — keep it out of module import so the controller imports cleanly there
        a = aifc.open(io.BytesIO(raw)); n = a.getnframes(); fr = a.getframerate(); ch = a.getnchannels()
        data = np.frombuffer(a.readframes(n), dtype=">i2").astype(np.float32)
    else:
        w = wave.open(io.BytesIO(raw)); n = w.getnframes(); fr = w.getframerate(); ch = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / 32768.0, fr


def audio_to_image(samples: np.ndarray, framerate: int) -> Image.Image:
    f, t, Sxx = scipy.signal.spectrogram(samples, fs=framerate, nperseg=256, noverlap=224)
    S = np.log(Sxx + 1e-10)
    S = S - np.median(S, axis=1, keepdims=True)          # general denoising: remove stationary background per freq bin
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    return Image.fromarray((S * 255).astype(np.uint8)).convert("RGB").resize((224, 224))


@dataclass(frozen=True)
class AudioTask:
    id_col: str
    target_cols: list[str]
    train_ids: list
    y: np.ndarray
    classes: list[str]
    test_ids: list
    train_paths: dict
    test_paths: dict


def _classify_split(relpath: str) -> str:
    # Match a path segment that starts with "train"/"test" so trainN/testN layouts
    # (e.g. whale's train2.zip -> "train2/...") classify correctly, not just exact "train".
    parts = relpath.replace("\\", "/").lower().split("/")
    if any(p.startswith("train") for p in parts):
        return "train"
    if any(p.startswith("test") for p in parts):
        return "test"
    return "both"


def _split_audio_indices(data_dir: Path):
    import zipfile
    data_dir = Path(data_dir); train_index: dict = {}; test_index: dict = {}
    def place(split, keys, ref):
        for k in keys:
            if split in ("train", "both"): train_index[k] = ref
            if split in ("test", "both"): test_index[k] = ref
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTS:
            rel = str(path.relative_to(data_dir))
            place(_classify_split(rel), [path.name, path.stem, rel], MediaRef(path=path))
        elif suffix == ".zip":
            try:
                arc = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                continue
            with arc:
                for member in arc.namelist():
                    if member.endswith("/") or Path(member).suffix.lower() not in AUDIO_EXTS:
                        continue
                    split = _classify_split(member)
                    if split == "both":
                        split = _classify_split(path.name)
                    place(split, [Path(member).name, Path(member).stem, member], MediaRef(path=path, member=member))
    return train_index, test_index


def _audio_train_labels(data_dir: Path, sample: pd.DataFrame, id_col: str, train_paths: dict) -> pd.DataFrame:
    for fname in ("labels.csv", "train.csv"):
        fpath = data_dir / fname
        if fpath.exists():
            frame = pd.read_csv(fpath); first = frame.columns[0]
            target_cols = list(sample.columns[1:]); non_id = [c for c in frame.columns if c != first]
            if len(target_cols) == 1 and target_cols[0] in non_id:
                label_col = target_cols[0]
            else:
                label_col = next((c for c in non_id if c not in sample.columns), non_id[0] if non_id else None)
            if label_col is not None:
                r = frame.rename(columns={first: id_col})
                return r.assign(__label__=r[label_col])[[id_col, "__label__"]]
    # filename-token fallback: enumerate unique train clips (incl. zip members) from the
    # train index; label is the last underscore token, else the first dot token.
    seen = set(); stems = []
    for ref in train_paths.values():
        member = getattr(ref, "member", None); path = getattr(ref, "path", ref)
        ident = (str(path), member)
        if ident in seen:
            continue
        seen.add(ident)
        stems.append(Path(member).stem if member else Path(path).stem)
    rows = []
    for stem in sorted(stems):
        token = stem.split("_")[-1] if "_" in stem else stem.split(".")[0]
        rows.append({id_col: stem, "__label__": token})
    if rows:
        return pd.DataFrame(rows)
    raise ValueError("could not resolve audio train labels")


def load_audio_task(data_dir: Path) -> AudioTask:
    data_dir = Path(data_dir)
    sample_path = _find_sample_submission(data_dir)
    if sample_path is None:
        raise ValueError("sample submission not found")
    sample = pd.read_csv(sample_path)
    id_col = sample.columns[0]; target_cols = list(sample.columns[1:])
    train_paths, test_paths = _split_audio_indices(data_dir)
    labels = _audio_train_labels(data_dir, sample, id_col, train_paths)
    label_values = [str(v) for v in labels["__label__"].tolist()]
    semantic = set(label_values) <= set(target_cols) and len(target_cols) > 1
    classes = target_cols if semantic else sorted(set(label_values))
    class_index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_index[v] for v in label_values])
    return AudioTask(id_col, target_cols, labels[id_col].tolist(), y, list(classes), sample[id_col].tolist(), train_paths, test_paths)


import json, os, time
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.operators.embedding_core import (
    _materialize_zip_member, scorer_for_metric, select_by_oof, torchvision_backbones,
    stratified_subsample as _stratified_subsample,
)
from bench.ees_core.prediction_artifacts import one_hot_frame, write_prediction_artifacts


def _clip_image(ref, zip_handles):
    member = getattr(ref, "member", None)
    path = getattr(ref, "path", ref)
    if member is not None:
        real = _materialize_zip_member(path, member, zip_handles)
        samples, fr = decode_audio(real)
    else:
        samples, fr = decode_audio(path)
    return audio_to_image(samples, fr)


def _embed_audio_ids(backbone, ids, paths):
    imgs = []
    zip_handles: dict = {}
    try:
        for identifier in ids:
            ref = paths.get(str(identifier)) or paths.get(identifier)
            if ref is None:
                raise ValueError(f"no audio for id {identifier!r}")
            imgs.append(_clip_image(ref, zip_handles))
    finally:
        for handle in zip_handles.values():
            handle.close()
    return backbone.embed(imgs)  # imgs are PIL spectrograms; torchvision_backbones now accepts them


def _default_audio_backbones():
    names = os.environ.get("EES_AUDIO_EMBED_BACKBONES", "resnet50").split(",")
    return torchvision_backbones([n.strip() for n in names if n.strip()])


def run_audio_embedding_operator(data_dir, output_dir, *, task_id, backbones=None, metric=None, random_state=101):
    started = time.time(); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    try:
        task = load_audio_task(Path(data_dir)); n_classes = len(task.classes)
        if n_classes < 2:
            raise ValueError("audio_embedding requires >= 2 classes")
        used = backbones if backbones is not None else _default_audio_backbones()
        max_rows = int(os.environ.get("EES_AUDIO_EMBED_MAX_TRAIN_ROWS", "10000"))
        train_ids, y = task.train_ids, task.y
        if len(train_ids) > max_rows:
            keep = _stratified_subsample(y, max_rows, np.random.default_rng(random_state))
            train_ids = [train_ids[i] for i in keep]; y = y[keep]
        emb = {bb.name: (_embed_audio_ids(bb, train_ids, task.train_paths), _embed_audio_ids(bb, task.test_ids, task.test_paths)) for bb in used}
        scorer = scorer_for_metric(metric)
        cv = max(2, min(5, int(np.unique(y, return_counts=True)[1].min())))
        sel = select_by_oof(emb, y, n_classes, n_splits=cv, random_state=random_state, scorer=scorer, class_weight="balanced")

        if len(task.target_cols) == 1:
            sub = pd.DataFrame({task.id_col: task.test_ids, task.target_cols[0]: sel.test_proba[:, -1]})
        else:
            sub = pd.DataFrame(sel.test_proba, columns=task.classes)[task.target_cols]; sub.insert(0, task.id_col, task.test_ids)

        # full-OOF validation artifacts for the selected config (consistent with
        # sel.oof_score) — mirrors image_embedding. One frame convention: both
        # frames carry the SAME native id values (never str-converted) so a
        # single native-dtype sort yields an identical row order in both. Sorting
        # str ids and native ids independently misaligns predictions vs targets
        # (numeric vs lexicographic order) — the exact bug found in image_embedding.
        oof = sel.oof_proba
        val_pred = pd.DataFrame(oof, columns=task.classes); val_pred.insert(0, task.id_col, list(train_ids))
        val_targets = one_hot_frame(list(train_ids), [task.classes[c] for c in y], id_col=task.id_col, target_cols=task.classes)
        if len(task.target_cols) == 1:
            val_pred = val_pred[[task.id_col, task.classes[-1]]].rename(columns={task.classes[-1]: task.target_cols[0]})
            val_targets = val_targets[[task.id_col, task.classes[-1]]].rename(columns={task.classes[-1]: task.target_cols[0]})
            report_targets = task.target_cols
        else:
            val_pred = val_pred[[task.id_col, *task.target_cols]]; val_targets = val_targets[[task.id_col, *task.target_cols]]; report_targets = task.target_cols
        # sort by id so cross-operator OOF records align (registry uses ordered id equality)
        val_pred = val_pred.sort_values(task.id_col, kind="stable").reset_index(drop=True)
        val_targets = val_targets.sort_values(task.id_col, kind="stable").reset_index(drop=True)

        submission_path = output_dir / "submission.csv"; sub.to_csv(submission_path, index=False)
        artifacts = write_prediction_artifacts(output_dir, test_predictions=sub, validation_predictions=val_pred,
                                                validation_targets=val_targets, id_col=task.id_col, target_cols=report_targets, validation_kind="oof")
        metrics = {"operator": "audio_embedding", "task_id": task_id, "selected_backbone": sel.backbone,
                   "selected_c": sel.c, "selected_temperature": sel.temperature, "oof_score": sel.oof_score,
                   "scorer": scorer, "n_classes": n_classes, "fit_train_rows": int(len(train_ids)),
                   "prediction_artifacts": artifacts,
                   "validation": {"validation_kind": "oof", "score": sel.oof_score, "objective": scorer,
                                  "direction": "maximize" if scorer in ("auc", "qwk") else "minimize"},
                   "runtime_seconds": round(time.time() - started, 3)}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(candidate_id="audio_embedding", recipe_id="audio_embedding", success=True,
                               submission_path=submission_path, validation={"rows": int(len(sub))}, score=sel.oof_score,
                               medal="none", final_artifact_source=f"ees_core:audio_embedding:{sel.backbone}",
                               artifact_paths=[str(submission_path), str(output_dir / "metrics.json")], metrics=metrics,
                               score_direction="maximize" if scorer in ("auc", "qwk") else "minimize")
    except Exception as exc:
        return CandidateResult(candidate_id="audio_embedding", recipe_id="audio_embedding", success=False,
                               submission_path=None, validation={}, score=None, medal="none",
                               final_artifact_source="ees_core:audio_embedding", no_score_reason=f"{type(exc).__name__}: {exc}")
