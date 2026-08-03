"""GPU transformer fine-tune leg for large text-classification tasks.

`text_tfidf.py` is a fast, CPU, in-process linear baseline. It plateaus below
medal bars on large toxic-comment / topic-classification corpora where a
fine-tuned transformer captures word order and long-range context TF-IDF cannot
(jigsaw-toxic-comment: text_tfidf ~0.9805 mean-column AUC, no medal). This
operator is the escalation tier above it: it fine-tunes one or more pretrained
transformer backbones (distilbert / roberta, sigmoid + BCE for multilabel) on a
real Vertex GPU through the Cap 0 seam, then blends every transformer with the
existing TF-IDF leg by holdout-optimal weights.

It is GENERAL, not task-specific: the controller schedules it for ANY text
classification task of a transformer-favouring shape (many rows + a
classification metric) when the escalation flags are set. Nothing here mentions
jigsaw; the winning jigsaw recipe is merely the default backbone hyper-parameter
table (validated 2026-07-09 on a T4).

Design (mirrors image_finetune's Vertex lane):

- The heavy training runs INSIDE the GPU container (self-contained script built
  by `_build_transformer_job_code`); the host never imports torch/transformers,
  so this operator co-runs safely with the sklearn/HGB text_tfidf leg (no
  torch+lightgbm co-load) and unit tests mock the seam entirely.
- Each backbone job trains on a FIXED stratified 5% holdout (seed 42), exports
  its holdout probabilities+labels and full test probabilities, and the launcher
  upload tail ships them back; the seam pulls them into the operator's output.
- Validation is HONEST holdout: the same 5% split is used to pick the best epoch,
  to select the blend weights, and as the exported validation artifact
  (`validation_kind="holdout"`), so the reported score is exactly the score of
  what's exported. The transformer holdout ids are a subset of train, so the
  TF-IDF full-train OOF (leak-free for every row) aligns to them by id.
- The blend is holdout-selected across {each transformer, TF-IDF}: a raw-space
  simplex grid for <= 3 sources (exactly the validated jigsaw recipe), greedy
  Caruana averaging above that. The blended candidate registers with its own
  OOF/holdout artifacts so the registry / final selection compares it against
  the pure TF-IDF candidate on trusted validation evidence alone.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bench.adapters.task_detection import _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.metric_scoring import internal_metric_direction
from bench.ees_core.operators.text_tfidf import (
    _detect_text_columns,
    _load_train_test_frames,
    _target_spec,
)
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

# Validated jigsaw T4 recipe (2026-07-09) as the default per-backbone table. A
# backbone not listed here uses `_GENERIC_RECIPE`. Wall/infer budgets are the
# in-container hard budgets; the seam watchdog is sized from them.
_DEFAULT_RECIPES: dict[str, dict[str, Any]] = {
    "distilbert-base-uncased": {
        "max_len": 220,
        "batch_size": 64,
        "lr": 2e-5,
        "epochs": 2,
        "wall_seconds": 4500,
        "infer_reserve_seconds": 1080,
    },
    "roberta-base": {
        "max_len": 256,
        "batch_size": 40,
        "lr": 1e-5,
        "epochs": 3,
        "wall_seconds": 7500,
        "infer_reserve_seconds": 1200,
    },
}
_GENERIC_RECIPE: dict[str, Any] = {
    "max_len": 256,
    "batch_size": 32,
    "lr": 2e-5,
    "epochs": 2,
    "wall_seconds": 5400,
    "infer_reserve_seconds": 1200,
}
_DEFAULT_BACKBONES = ("distilbert-base-uncased", "roberta-base")
_HOLDOUT_FRACTION = 0.05
_SEED = 42


@dataclass(frozen=True)
class _SourcePreds:
    """One blend source: holdout probs (aligned to Y) + test probs (by id)."""

    name: str
    holdout: np.ndarray  # (n_holdout, n_targets)
    test: pd.DataFrame  # id_col + target_cols


def _recipe_for(backbone: str) -> dict[str, Any]:
    return dict(_DEFAULT_RECIPES.get(backbone, _GENERIC_RECIPE))


def resolve_backbones(raw: str | None) -> list[str]:
    """Parse the EES_TEXT_TRANSFORMER_BACKBONES override, else the default pair."""
    if not raw:
        return list(_DEFAULT_BACKBONES)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or list(_DEFAULT_BACKBONES)


# ---------------------------------------------------------------------------
# In-container training script (self-contained; no bench import inside the box)
# ---------------------------------------------------------------------------

def _build_transformer_job_code(
    *,
    backbone: str,
    max_len: int,
    batch_size: int,
    lr: float,
    epochs: int,
    wall_seconds: float,
    infer_reserve_seconds: float,
    seed: int,
    holdout_fraction: float,
) -> str:
    """Return the GPU-container training script for one backbone.

    Runs entirely on the staged data in cwd, derives id/targets/text columns
    from the files (no hardcoded schema), fine-tunes a sigmoid+BCE multilabel
    head, and writes holdout_probs.csv / holdout_targets.csv / test_probs.csv /
    metrics.json. The launcher upload tail (added by the seam) ships them back.
    """
    return f'''
import os, json, time, subprocess, sys
t0 = time.time()
print("TEXT_TRANSFORMER_JOB_START", {backbone!r}, flush=True)

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    import transformers  # noqa
except Exception:
    _pip("transformers==4.44.2")
    import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, "torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

BACKBONE = {backbone!r}
MAX_LEN = {int(max_len)}
BS = {int(batch_size)}
INFER_BS = 256
LR = {float(lr)}
EPOCHS = {int(epochs)}
SEED = {int(seed)}
WALL_BUDGET_S = {float(wall_seconds)}
INFER_RESERVE_S = {float(infer_reserve_seconds)}
TRAIN_DEADLINE_S = WALL_BUDGET_S - INFER_RESERVE_S
HOLDOUT_FRAC = {float(holdout_fraction)}
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- schema from the staged files (general; no task-specific columns) ----
sample = pd.read_csv("sample_submission.csv")
ID = sample.columns[0]
TARGETS = list(sample.columns[1:])
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")

def _detect_text_col(train, test):
    excluded = set(TARGETS) | {{ID}}
    shared = [c for c in train.columns if c in test.columns and c not in excluded]
    obj = [c for c in shared if train[c].dtype == object or str(train[c].dtype).startswith("string")]
    if not obj:
        obj = shared
    return max(obj, key=lambda c: float(train[c].fillna("").astype(str).str.len().mean()))

TEXT = _detect_text_col(tr, te)
tr[TEXT] = tr[TEXT].fillna(" ").astype(str)
te[TEXT] = te[TEXT].fillna(" ").astype(str)
Y = tr[TARGETS].to_numpy(dtype=np.float32)
n_out = len(TARGETS)
print("id", ID, "targets", TARGETS, "text", TEXT, "train", tr.shape, "test", te.shape, flush=True)

# ---- fixed stratified holdout (never touches test); same seed across backbones
if n_out > 1:
    strat = (Y.sum(axis=1) > 0).astype(int)
else:
    strat = Y.reshape(-1).astype(int)
try:
    idx_tr, idx_val = train_test_split(np.arange(len(tr)), test_size=HOLDOUT_FRAC, random_state=SEED, stratify=strat)
except ValueError:
    idx_tr, idx_val = train_test_split(np.arange(len(tr)), test_size=HOLDOUT_FRAC, random_state=SEED)
tr_df = tr.iloc[idx_tr]; val_df = tr.iloc[idx_val]
Ytr = Y[idx_tr]; Yval = Y[idx_val]
print("train/val", tr_df.shape, val_df.shape, flush=True)

tok = AutoTokenizer.from_pretrained(BACKBONE)

class TextDS(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        return (i, self.texts[i]) if self.labels is None else (self.texts[i], self.labels[i])

def collate_train(batch):
    txt = [b[0] for b in batch]; y = np.stack([b[1] for b in batch])
    enc = tok(txt, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return enc, torch.tensor(y, dtype=torch.float32)

def collate_infer(batch):
    idx = [b[0] for b in batch]; txt = [b[1] for b in batch]
    enc = tok(txt, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return idx, enc

class ToxicModel(nn.Module):
    def __init__(self, name, n):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        h = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(h, n)
    def forward(self, **enc):
        out = self.backbone(**enc)
        cls = out.last_hidden_state[:, 0]  # CLS token; robust across bert/distilbert/roberta
        return self.head(self.drop(cls))

nworkers = min(4, os.cpu_count() or 1)
dl_tr = DataLoader(TextDS(tr_df[TEXT].values, Ytr), batch_size=BS, shuffle=True,
                   num_workers=nworkers, pin_memory=True, collate_fn=collate_train, drop_last=True)
dl_val = DataLoader(TextDS(val_df[TEXT].values, Yval), batch_size=INFER_BS, shuffle=False,
                    num_workers=nworkers, pin_memory=True, collate_fn=collate_train)

model = ToxicModel(BACKBONE, n_out).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
crit = nn.BCEWithLogitsLoss()
use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def predict(dl, has_label):
    model.eval(); probs = []; ys = []
    with torch.no_grad():
        for batch in dl:
            if has_label:
                enc, y = batch; ys.append(y.numpy())
            else:
                _, enc = batch
            enc = {{k: v.to(device, non_blocking=True) for k, v in enc.items()}}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**enc)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    P = np.concatenate(probs)
    return (P, np.concatenate(ys)) if has_label else P

def mean_col_auc(y, p):
    aucs = []
    for j in range(y.shape[1]):
        try:
            aucs.append(roc_auc_score(y[:, j], p[:, j]))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))

best_auc, best_state, best_epoch, epochs_run = -1.0, None, -1, 0
for ep in range(EPOCHS):
    if time.time() - t0 > TRAIN_DEADLINE_S:
        print("TRAIN_DEADLINE before epoch", ep, flush=True); break
    model.train(); tl = 0.0; nb = 0
    for enc, y in dl_tr:
        if time.time() - t0 > TRAIN_DEADLINE_S:
            print("TRAIN_DEADLINE mid-epoch", ep, "step", nb, flush=True); break
        enc = {{k: v.to(device, non_blocking=True) for k, v in enc.items()}}
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = crit(model(**enc), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item(); nb += 1
        if nb % 200 == 0:
            print(f"  ep{{ep}} step {{nb}} loss {{tl/nb:.4f}} elapsed {{round(time.time()-t0,1)}}s", flush=True)
    vp, vy = predict(dl_val, True)
    auc = mean_col_auc(vy, vp); epochs_run = ep + 1
    print(f"EPOCH {{ep}} loss {{tl/max(nb,1):.4f}} val_auc {{auc:.5f}} elapsed {{round(time.time()-t0,1)}}s", flush=True)
    if auc > best_auc:
        best_auc, best_epoch = auc, ep
        best_state = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
if best_state is not None:
    model.load_state_dict(best_state)
print("BEST_VAL_AUC", round(best_auc, 5), "epoch", best_epoch, flush=True)

# ---- export holdout probs+labels (best model) ----
vp_best, _ = predict(dl_val, True)
hp = pd.DataFrame({{ID: val_df[ID].values}})
ht = pd.DataFrame({{ID: val_df[ID].values}})
for j, c in enumerate(TARGETS):
    hp[c] = vp_best[:, j]
    ht[c] = Yval[:, j].astype(int)
hp.to_csv("holdout_probs.csv", index=False)
ht.to_csv("holdout_targets.csv", index=False)

# ---- test inference on ALL rows ----
dl_te = DataLoader(TextDS(te[TEXT].values, None), batch_size=INFER_BS, shuffle=False,
                   num_workers=nworkers, pin_memory=True, collate_fn=collate_infer)
ti = time.time(); Pte = predict(dl_te, False); infer_s = time.time() - ti
tp = pd.DataFrame({{ID: te[ID].values}})
for j, c in enumerate(TARGETS):
    tp[c] = Pte[:, j]
tp.to_csv("test_probs.csv", index=False)

metrics = {{
    "backbone": BACKBONE, "holdout_mean_auc": float(best_auc), "best_epoch": int(best_epoch),
    "epochs_run": int(epochs_run), "epochs_planned": EPOCHS, "id_col": ID, "target_cols": TARGETS,
    "text_col": TEXT, "n_train": int(len(tr_df)), "n_val": int(len(val_df)), "n_test": int(len(tp)),
    "max_len": MAX_LEN, "batch_size": BS, "lr": LR, "infer_seconds": round(infer_s, 1),
    "wall_seconds": round(time.time() - t0, 1),
    "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
}}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("TEXT_TRANSFORMER_JOB_DONE", json.dumps(metrics), flush=True)
'''


# ---------------------------------------------------------------------------
# Host-side blend helpers (pure; unit-tested directly)
# ---------------------------------------------------------------------------

def _mean_col_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if np.unique(col).size < 2:
            aucs.append(0.5)
            continue
        aucs.append(float(roc_auc_score(col, y_pred[:, j])))
    return float(np.mean(aucs)) if aucs else 0.5


def _simplex_grid(n_sources: int, step: float) -> list[tuple[float, ...]]:
    """Enumerate convex weights on the (n_sources-1)-simplex at `step` resolution."""
    ticks = int(round(1.0 / step))
    combos: list[tuple[float, ...]] = []
    for cut in product(range(ticks + 1), repeat=n_sources - 1):
        used = sum(cut)
        if used > ticks:
            continue
        weights = [c / ticks for c in cut]
        weights.append((ticks - used) / ticks)
        combos.append(tuple(weights))
    return combos


def select_blend_weights(
    holdout_sources: list[np.ndarray],
    y_true: np.ndarray,
    *,
    grid_step: float = 0.05,
    greedy_rounds: int = 200,
) -> tuple[list[float], float]:
    """Holdout-optimal convex weights over the sources (maximize mean-column AUC).

    A raw-space simplex grid for <= 3 sources (exactly the validated jigsaw
    recipe); greedy Caruana averaging with replacement above that so the search
    stays tractable for any number of sources.
    """
    n = len(holdout_sources)
    if n == 1:
        return [1.0], _mean_col_auc(y_true, holdout_sources[0])
    if n <= 3:
        best_w: list[float] | None = None
        best_score = -np.inf
        for weights in _simplex_grid(n, grid_step):
            blended = sum(w * src for w, src in zip(weights, holdout_sources))
            score = _mean_col_auc(y_true, blended)
            if score > best_score:
                best_score, best_w = score, list(weights)
        assert best_w is not None
        return best_w, best_score
    # Greedy Caruana averaging (general, tractable for many sources).
    counts = [0] * n
    ensemble: np.ndarray | None = None
    best_score = -np.inf
    for round_index in range(greedy_rounds):
        best_idx, round_best, round_pred = 0, -np.inf, None
        for i, src in enumerate(holdout_sources):
            cand = src if ensemble is None else (ensemble * round_index + src) / (round_index + 1)
            score = _mean_col_auc(y_true, cand)
            if score > round_best:
                best_idx, round_best, round_pred = i, score, cand
        counts[best_idx] += 1
        ensemble = round_pred
        best_score = round_best
    total = sum(counts) or 1
    return [c / total for c in counts], best_score


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def _skip(reason: str, backbones: list[str]) -> CandidateResult:
    return CandidateResult(
        candidate_id="text_transformer",
        recipe_id="text_transformer",
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source=f"ees_core:text_transformer:{','.join(backbones)}",
        no_score_reason=reason,
    )


def run_text_transformer_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    metric: str | None = None,
    tfidf_artifacts: dict[str, Any] | None = None,
    backbones: list[str] | None = None,
    seed: int = _SEED,
    holdout_fraction: float = _HOLDOUT_FRACTION,
    seam: Any = None,
) -> CandidateResult:
    """Fine-tune transformer backbone(s) on GPU and holdout-blend with TF-IDF.

    ``tfidf_artifacts`` is the sibling text_tfidf candidate's
    ``prediction_artifacts`` dict (OOF validation predictions + test submission);
    when present it joins the blend as an extra source. All GPU work goes through
    the injectable ``seam`` (default: bench.ees_core.vertex_gpu_execution) so this
    is fully unit-testable with no GPU.
    """
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbones = backbones or resolve_backbones(None)
    if seam is None:
        from bench.ees_core import vertex_gpu_execution as seam
    try:
        sample_path = _find_sample_submission(data_dir)
        if sample_path is None:
            raise FileNotFoundError("sample submission not found")
        sample = pd.read_csv(sample_path)
        id_col = str(sample.columns[0])
        target_cols = [str(c) for c in sample.columns[1:]]
        if not target_cols:
            return _skip("no_target_columns", backbones)
        train, test = _load_train_test_frames(data_dir)
        # A defensive shape self-check (the controller gate is primary): the leg
        # is only meaningful for a text classification task.
        target_spec = _target_spec(train, test, sample)
        _detect_text_columns(train, test, [*target_spec.target_cols, *( [target_spec.train_target_col] if target_spec.train_target_col else [])])

        # ---- run each backbone on a real GPU through the Cap 0 seam ----
        sources: list[_SourcePreds] = []
        holdout_targets: pd.DataFrame | None = None
        job_infos: list[dict[str, Any]] = []
        for backbone in backbones:
            recipe = _recipe_for(backbone)
            job_code = _build_transformer_job_code(
                backbone=backbone,
                max_len=recipe["max_len"],
                batch_size=recipe["batch_size"],
                lr=recipe["lr"],
                epochs=recipe["epochs"],
                wall_seconds=recipe["wall_seconds"],
                infer_reserve_seconds=recipe["infer_reserve_seconds"],
                seed=seed,
                holdout_fraction=holdout_fraction,
            )
            safe = backbone.replace("/", "_")
            job_out = output_dir / safe
            # Poll watchdog must exceed the in-container wall budget + provisioning.
            timeout_min = int(recipe["wall_seconds"] / 60.0) + 8
            job = seam.run_gpu_training_job(
                job_code,
                data_dir,
                task_id=f"{task_id or 'text'}_{safe}",
                output_dir=job_out,
                timeout_min=timeout_min,
            )
            job_infos.append({"backbone": backbone, "state": job.state, "job_id": job.job_id, "run_ts": job.run_ts})
            if not job.success:
                # A single backbone failing is not fatal: blend whatever survived.
                continue
            hp_path = job_out / "holdout_probs.csv"
            ht_path = job_out / "holdout_targets.csv"
            tp_path = job_out / "test_probs.csv"
            if not (hp_path.exists() and ht_path.exists() and tp_path.exists()):
                continue
            hp = pd.read_csv(hp_path)
            ht = pd.read_csv(ht_path)
            tp = pd.read_csv(tp_path)
            if holdout_targets is None:
                holdout_targets = ht[[id_col, *target_cols]].copy()
            # Align this backbone's holdout to the shared target row order by id.
            merged = holdout_targets[[id_col]].merge(hp, on=id_col, how="left")
            sources.append(
                _SourcePreds(
                    name=backbone,
                    holdout=merged[target_cols].to_numpy(dtype=float),
                    test=tp[[id_col, *target_cols]].copy(),
                )
            )

        if not sources or holdout_targets is None:
            return _skip(
                "no_successful_backbone:" + ";".join(f"{i['backbone']}={i['state']}" for i in job_infos),
                backbones,
            )

        # ---- add the TF-IDF leg as a blend source (holdout-aligned by id) ----
        tfidf_engaged = False
        if tfidf_artifacts:
            tf_source = _tfidf_source(tfidf_artifacts, holdout_targets, id_col, target_cols)
            if tf_source is not None:
                sources.append(tf_source)
                tfidf_engaged = True

        y_true = holdout_targets[target_cols].to_numpy(dtype=float)
        holdout_matrices = [src.holdout for src in sources]
        weights, holdout_score = select_blend_weights(holdout_matrices, y_true)

        # ---- blended holdout predictions (exported validation artifact) ----
        blended_holdout = sum(w * m for w, m in zip(weights, holdout_matrices))
        validation_predictions = holdout_targets[[id_col]].copy()
        for j, col in enumerate(target_cols):
            validation_predictions[col] = np.clip(blended_holdout[:, j], 1e-6, 1 - 1e-6)
        validation_targets = holdout_targets[[id_col, *target_cols]].copy()

        # ---- blended test submission (aligned to sample id order) ----
        submission = _blend_test(sample, sources, weights, id_col, target_cols)
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)

        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col=id_col,
            target_cols=target_cols,
            validation_kind="holdout",
        )
        blend_weights = {src.name: round(float(w), 4) for src, w in zip(sources, weights)}
        metrics = {
            "operator": "text_transformer",
            "task_id": task_id,
            "backbones": backbones,
            "successful_sources": [src.name for src in sources],
            "tfidf_engaged": tfidf_engaged,
            "blend_weights": blend_weights,
            "holdout_rows": int(len(holdout_targets)),
            "holdout_mean_auc": holdout_score,
            "metric": metric,
            "jobs": job_infos,
            "execution_lane": "vertex_gpu",
            "validation": {
                "validation_kind": "holdout",
                "score": holdout_score,
                "objective": "auc",
                "direction": "maximize",
            },
            "prediction_artifacts": artifacts,
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="text_transformer",
            recipe_id="text_transformer",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=holdout_score,
            medal="none",
            final_artifact_source=f"ees_core:text_transformer:{'+'.join(src.name for src in sources)}",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
            score_direction=internal_metric_direction(metric) or "maximize",
        )
    except Exception as exc:  # noqa: BLE001 — never crash the controller
        return _skip(f"{type(exc).__name__}: {exc}", backbones)


def _tfidf_source(
    tfidf_artifacts: dict[str, Any],
    holdout_targets: pd.DataFrame,
    id_col: str,
    target_cols: list[str],
) -> _SourcePreds | None:
    """Restrict the TF-IDF OOF to the transformer holdout ids + carry its test preds.

    The TF-IDF leg exports FULL-train OOF (leak-free for every row), so selecting
    the rows at the transformer's holdout ids yields honest, non-leaky TF-IDF
    predictions aligned to Y.
    """
    val_path = tfidf_artifacts.get("validation_prediction_path")
    test_path = tfidf_artifacts.get("test_prediction_path")
    tf_id = str(tfidf_artifacts.get("id_col") or id_col)
    if not val_path or not test_path:
        return None
    val_path = Path(val_path)
    test_path = Path(test_path)
    if not val_path.exists() or not test_path.exists():
        return None
    oof = pd.read_csv(val_path)
    test_pred = pd.read_csv(test_path)
    if tf_id not in oof.columns or any(c not in oof.columns for c in target_cols):
        return None
    if id_col not in test_pred.columns or any(c not in test_pred.columns for c in target_cols):
        return None
    # Align OOF to the shared holdout id order.
    key = holdout_targets[[id_col]].copy()
    key[id_col] = key[id_col].astype(str)
    oof = oof.rename(columns={tf_id: id_col})
    oof[id_col] = oof[id_col].astype(str)
    merged = key.merge(oof[[id_col, *target_cols]], on=id_col, how="left")
    if merged[target_cols].isna().any().any():
        return None
    return _SourcePreds(
        name="text_tfidf",
        holdout=merged[target_cols].to_numpy(dtype=float),
        test=test_pred[[id_col, *target_cols]].copy(),
    )


def _blend_test(
    sample: pd.DataFrame,
    sources: list[_SourcePreds],
    weights: list[float],
    id_col: str,
    target_cols: list[str],
) -> pd.DataFrame:
    """Weighted blend of every source's test predictions, keyed to sample id order."""
    submission = sample.copy()
    sample_ids = submission[id_col].astype(str).to_numpy()
    accum = {col: np.zeros(len(submission), dtype=float) for col in target_cols}
    for src, weight in zip(sources, weights):
        if weight == 0:
            continue
        frame = src.test.copy()
        frame[id_col] = frame[id_col].astype(str)
        frame = frame.drop_duplicates(subset=[id_col], keep="first").set_index(id_col)
        aligned = frame.reindex(sample_ids)
        for col in target_cols:
            accum[col] += weight * aligned[col].to_numpy(dtype=float)
    for col in target_cols:
        submission[col] = np.clip(accum[col], 1e-6, 1 - 1e-6)
    return submission[list(sample.columns)]
