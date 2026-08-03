"""Diverse-ensemble mode for the image fine-tune operator.

A single fine-tuned backbone plateaus below the medal bar on ordinal image
tasks (diabetic-retinopathy grading and friends) whose test set is drawn from a
DIFFERENT distribution than train: one model overfits the train distribution.
The proven fix -- earned by hand on aptos2019 -- is an ensemble of DIVERSE
backbone families at DIFFERENT resolutions (and seeds), each trained as a
regression head with k-fold OOF, then blended by **OOF-weighted averaging** so
decorrelated configs get weight and redundant ones get dropped. For ordinal
(quadratic-weighted-kappa) targets a single ``OptimizedRounder`` is fit on the
POOLED OOF and applied to the blended test predictions.

This module holds the parts that carry NO torch dependency so they unit-test
without co-loading torch (or lightgbm):

- ``OptimizedRounder`` + ``qwk_of``: ordinal threshold search on QWK.
- config generation: a general default set (efficientnet + resnet at higher
  res, distinct seeds) that is fully env-overridable -- no task-specific
  constants baked in.
- ``aggregate_ordinal_ensemble``: the pure OOF-weighted blend + pooled rounder.
- ``build_config_job_code``: the in-container k-fold regression trainer
  (Ben-Graham domain preprocessing gated by a per-config flag) with the
  torch/torchvision ABI subprocess repair baked in.

The torch-heavy orchestration (submit each config through the Vertex seam or
pull cached fold predictions, then aggregate) lives in ``image_finetune.py``.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ordinal helpers (QWK)
# ---------------------------------------------------------------------------

class OptimizedRounder:
    """Fit 4 thresholds that maximize quadratic-weighted-kappa for 5 ordinal bins.

    General over ``n_bins``: the initial coefficients and the bin labels are
    derived from ``n_bins`` so this is not hard-wired to the 0..4 aptos grade
    scale. Nelder-Mead minimizes ``-QWK`` exactly as the proven hand run did.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = int(n_bins)
        self.coef_ = [i + 0.5 for i in range(self.n_bins - 1)]

    def _labels(self) -> list[int]:
        return list(range(self.n_bins))

    def _cut(self, X, coef) -> np.ndarray:
        edges = [-np.inf] + list(np.sort(coef)) + [np.inf]
        return np.asarray(pd.cut(np.asarray(X), edges, labels=self._labels()).astype(int))

    def _loss(self, coef, X, y) -> float:
        from sklearn.metrics import cohen_kappa_score

        pred = self._cut(X, coef)
        return -cohen_kappa_score(y, pred, weights="quadratic")

    def fit(self, X, y) -> "OptimizedRounder":
        from scipy.optimize import minimize

        init = [i + 0.5 for i in range(self.n_bins - 1)]
        self.coef_ = minimize(partial(self._loss, X=np.asarray(X), y=np.asarray(y)), init, method="nelder-mead")["x"]
        return self

    def predict(self, X) -> np.ndarray:
        return self._cut(np.asarray(X), self.coef_)


def qwk_of(cont: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
    """QWK of the OptimizedRounder-thresholded continuous predictions."""
    from sklearn.metrics import cohen_kappa_score

    rounder = OptimizedRounder(n_bins=n_bins).fit(cont, labels)
    return float(cohen_kappa_score(labels, rounder.predict(cont), weights="quadratic"))


def is_ordinal_metric(metric: str | None) -> bool:
    """True for quadratic-weighted-kappa / ordinal-kappa objectives."""
    if not metric:
        return False
    norm = metric.lower().replace("_", " ").replace("-", " ")
    return "qwk" in norm or "kappa" in norm


# ---------------------------------------------------------------------------
# Config generation (general default set + env override)
# ---------------------------------------------------------------------------

# General default: a diverse trio -- two efficientnet capacities at a mid
# resolution and one resnet family at a HIGHER resolution, distinct seeds so
# even the stratified-fold split differs. Decorrelation, not any one task, is
# the design; two members measurably under-hedge the OOF->test shift (a 2-config
# blend graded 0.912 off a 0.9217 OOF where 3 hand configs graded 0.923).
_DEFAULT_CONFIGS: list[dict] = [
    {"name": "effnet_b4_380_s42", "backbone": "tf_efficientnet_b4_ns", "image_size": 380, "seed": 42, "nfolds": 5, "preprocess": "ben_graham"},
    {"name": "resnet50d_456_s123", "backbone": "resnet50d", "image_size": 456, "seed": 123, "nfolds": 5, "preprocess": "ben_graham"},
    {"name": "effnet_b5_380_s7", "backbone": "tf_efficientnet_b5_ns", "image_size": 380, "seed": 7, "nfolds": 5, "preprocess": "ben_graham"},
]


def default_ensemble_configs() -> list[dict]:
    return [dict(c) for c in _DEFAULT_CONFIGS]


def resolve_ensemble_configs(env: dict | None = None) -> list[dict]:
    """The diverse config set: ``EES_FINETUNE_ENSEMBLE_CONFIGS`` JSON or the default.

    Each config is a dict with keys: ``name``, ``backbone``, ``image_size``,
    ``seed``, ``nfolds``, optional ``preprocess`` ("ben_graham"|"plain"),
    optional ``pretrain_ckpt`` (a gs:// path to a domain-pretrained backbone
    state dict), and optional ``reuse_prefix`` (a gs:// prefix carrying
    ``fold_<f>/fold_preds.npz`` from a prior run -- when present the config's
    training is skipped and its predictions are pulled, a free warm cache).
    """
    env = os.environ if env is None else env
    raw = env.get("EES_FINETUNE_ENSEMBLE_CONFIGS")
    configs = json.loads(raw) if raw else default_ensemble_configs()
    if not isinstance(configs, list) or not configs:
        raise ValueError("EES_FINETUNE_ENSEMBLE_CONFIGS must be a non-empty JSON list")
    out = []
    for i, c in enumerate(configs):
        c = dict(c)
        c.setdefault("name", f"cfg{i}_{c.get('backbone', 'model')}_{c.get('image_size', 0)}")
        c.setdefault("image_size", 380)
        c.setdefault("seed", 42 + i)
        c.setdefault("nfolds", int(env.get("EES_FINETUNE_ENSEMBLE_NFOLDS", "5")))
        c.setdefault("preprocess", "ben_graham")
        if "backbone" not in c:
            raise ValueError(f"ensemble config {c['name']} missing 'backbone'")
        out.append(c)
    return out


def diverse_ensemble_enabled(metric: str | None, env: dict | None = None) -> bool:
    """Gate for the diverse-ensemble path (single-model stays the default).

    - ``EES_FINETUNE_DIVERSE_ENSEMBLE=1`` forces it on (explicit opt-in).
    - ``EES_FINETUNE_DIVERSE_ENSEMBLE=auto`` fires it on ORDINAL evidence
      (a QWK/kappa metric, where a single backbone plausibly underfits the
      train->test shift).
    - unset / anything else: off.

    The regression + OptimizedRounder aggregation only applies to ordinal
    targets, so even ``=1`` requires an ordinal metric; otherwise the caller
    falls back to the single-model path.
    """
    env = os.environ if env is None else env
    flag = (env.get("EES_FINETUNE_DIVERSE_ENSEMBLE") or "").strip().lower()
    if flag == "1":
        return is_ordinal_metric(metric)
    if flag == "auto":
        return is_ordinal_metric(metric)
    return False


# ---------------------------------------------------------------------------
# Aggregation (pure, unit-tested directly)
# ---------------------------------------------------------------------------

@dataclass
class ConfigPreds:
    """Pooled k-fold predictions for one ensemble config.

    ``oof`` maps train id -> continuous OOF prediction (each id appears in
    exactly one fold's validation split). ``test`` maps test id -> the
    fold-MEAN continuous prediction. ``labels`` maps train id -> int class.
    """

    name: str
    oof: dict[str, float]
    test: dict[str, float]
    labels: dict[str, int]


def config_preds_from_folds(name: str, folds: list[dict]) -> ConfigPreds:
    """Build a ``ConfigPreds`` from a list of per-fold arrays.

    Each fold dict carries ``oof_ids``, ``oof_cont``, ``oof_labels``,
    ``test_ids``, ``test_cont``. OOF is unioned across folds; test is averaged
    across folds per id -- matching the proven hand aggregator exactly.
    """
    oof: dict[str, float] = {}
    labels: dict[str, int] = {}
    test_frames: list[dict[str, float]] = []
    for fold in folds:
        for i, v, y in zip(
            [str(x) for x in fold["oof_ids"]],
            list(fold["oof_cont"]),
            list(fold["oof_labels"]),
        ):
            oof[i] = float(v)
            labels[i] = int(y)
        test_frames.append(dict(zip([str(x) for x in fold["test_ids"]], [float(x) for x in fold["test_cont"]])))
    test_ids = sorted(test_frames[0].keys()) if test_frames else []
    test = {i: float(np.mean([tf[i] for tf in test_frames if i in tf])) for i in test_ids}
    return ConfigPreds(name=name, oof=oof, test=test, labels=labels)


@dataclass
class OrdinalEnsembleResult:
    submission: pd.DataFrame          # [id_col, target_col] rounded ints
    oof_ensemble_cont: np.ndarray     # blended continuous OOF preds
    oof_labels: np.ndarray            # int labels aligned to common_oof
    common_oof_ids: list[str]
    test_ensemble_cont: np.ndarray    # blended continuous test preds
    common_test_ids: list[str]
    weights: dict[str, float]
    chosen: str                       # "weighted" | "mean"
    equal_mean_oof_qwk: float
    weighted_oof_qwk: float
    ensemble_oof_qwk: float           # score of the CHOSEN blend
    rounder_coef: list[float]
    oof_corr: dict[str, float]
    config_self_oof_qwk: dict[str, float]
    n_bins: int


def _weight_grid(n: int, step: float = 0.1):
    ticks = [round(k * step, 6) for k in range(int(round(1.0 / step)) + 1)]
    for combo in itertools.product(ticks, repeat=n):
        if abs(sum(combo) - 1.0) <= 1e-6:
            yield np.array(combo, dtype=float)


def aggregate_ordinal_ensemble(
    configs: list[ConfigPreds],
    *,
    id_col: str,
    target_col: str,
    n_bins: int | None = None,
    weight_margin: float = 1e-3,
) -> OrdinalEnsembleResult:
    """OOF-weighted diverse-ensemble blend with a pooled ordinal rounder.

    Selection is OOF-ONLY: an equal-mean blend and an OOF-weighted blend (grid
    over the simplex) are both scored on the POOLED OOF, and the weighted blend
    is chosen only when it beats the mean by ``weight_margin`` (so we don't
    chase OOF noise into overfit). One OptimizedRounder is fit on the chosen
    OOF blend and applied to the same-weighted test blend. Test predictions are
    never consulted for selection.
    """
    if not configs:
        raise ValueError("aggregate_ordinal_ensemble requires >= 1 config")
    label_map: dict[str, int] = {}
    for c in configs:
        label_map.update(c.labels)
    common_oof = sorted(set.intersection(*[set(c.oof.keys()) for c in configs]))
    common_test = sorted(set.intersection(*[set(c.test.keys()) for c in configs]))
    if not common_oof or not common_test:
        raise ValueError("configs share no common OOF/test ids; cannot blend")
    labels = np.array([label_map[i] for i in common_oof], dtype=int)
    if n_bins is None:
        n_bins = int(labels.max()) + 1
    names = [c.name for c in configs]

    OOF = np.column_stack([[c.oof[i] for i in common_oof] for c in configs])
    TST = np.column_stack([[c.test[i] for i in common_test] for c in configs])

    config_self = {c.name: qwk_of(OOF[:, j], labels, n_bins) for j, c in enumerate(configs)}

    # diversity diagnostics: pairwise OOF correlation (lower = more diverse)
    corr = np.corrcoef(OOF.T) if len(names) > 1 else np.array([[1.0]])
    oof_corr = {
        f"{names[a]}|{names[b]}": float(corr[a, b])
        for a, b in itertools.combinations(range(len(names)), 2)
    }

    n = len(configs)
    mean_qwk = qwk_of(OOF.mean(axis=1), labels, n_bins)
    best_w, best_qwk = np.ones(n) / n, mean_qwk
    if n <= 4:  # exhaustive simplex grid only tractable for a handful of configs
        for w in _weight_grid(n):
            q = qwk_of(OOF @ w, labels, n_bins)
            if q > best_qwk:
                best_qwk, best_w = q, w

    if best_qwk > mean_qwk + weight_margin:
        chosen, w, ens_qwk = "weighted", best_w, best_qwk
    else:
        chosen, w, ens_qwk = "mean", np.ones(n) / n, mean_qwk

    ens_oof = OOF @ w
    rounder = OptimizedRounder(n_bins=n_bins).fit(ens_oof, labels)
    ens_test = TST @ w
    test_pred = rounder.predict(ens_test).astype(int)
    submission = pd.DataFrame({id_col: common_test, target_col: test_pred})

    return OrdinalEnsembleResult(
        submission=submission,
        oof_ensemble_cont=ens_oof,
        oof_labels=labels,
        common_oof_ids=common_oof,
        test_ensemble_cont=ens_test,
        common_test_ids=common_test,
        weights={names[j]: float(w[j]) for j in range(n)},
        chosen=chosen,
        equal_mean_oof_qwk=float(mean_qwk),
        weighted_oof_qwk=float(best_qwk),
        ensemble_oof_qwk=float(ens_qwk),
        rounder_coef=[float(x) for x in np.sort(rounder.coef_)],
        oof_corr=oof_corr,
        config_self_oof_qwk=config_self,
        n_bins=int(n_bins),
    )


def align_submission_to_sample(
    submission: "pd.DataFrame", sample_path, id_col: str, target_col: str
) -> "pd.DataFrame":
    """Reorder a submission to the sample submission's id ORDER.

    The controller's completion gate rejects submissions whose id column is not
    in sample-submission order — `aggregate_ordinal_ensemble` emits sorted ids,
    which is a different order for most tasks. Any id the blend is missing is
    filled with the modal prediction rather than forfeiting the whole candidate.
    """
    import pandas as pd

    sample_ids = pd.read_csv(sample_path, usecols=[id_col])
    aligned = sample_ids.merge(submission, on=id_col, how="left")
    if aligned[target_col].isna().any():
        fill = int(submission[target_col].mode().iat[0]) if len(submission) else 0
        aligned[target_col] = aligned[target_col].fillna(fill)
    aligned[target_col] = aligned[target_col].astype(int)
    return aligned[[id_col, target_col]]


# ---------------------------------------------------------------------------
# In-container k-fold regression trainer (job code)
# ---------------------------------------------------------------------------

# torch/torchvision ABI subprocess repair -- the GPU base image ships a
# CUDA torch whose bundled torchvision ABI is mismatched, which breaks any timm
# job at `import torchvision` (timm imports it transitively). Repair MUST run in
# a subprocess BEFORE this process imports torch, verifying torch.ops.torchvision
# resolves; on failure it force-reinstalls a torch-version-matched torchvision
# from the correct cuXXX wheel index, then a last-resort pinned torch+tv pair.
# Ported verbatim in spirit from the proven ~/aptos_kfold.py repair.
_TORCHVISION_ABI_REPAIR = r'''
import subprocess as _sp, sys as _sys
def _pip(*a):
    _sp.run([_sys.executable, "-m", "pip", "install", "--quiet", *a], check=True)
def _run(code):
    r = _sp.run([_sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
def _tv_ok():
    rc, out, err = _run("import torch, torchvision; torch.ops.torchvision.nms; print('TVOK')")
    if rc != 0:
        print("tv_ok fail:", (err or out)[-300:], flush=True)
    return rc == 0 and "TVOK" in out
assert "torch" not in _sys.modules, "torch imported before ABI repair"
if not _tv_ok():
    rc, out, err = _run("import torch; print(torch.__version__, (torch.version.cuda or '12.1'))")
    ver, cuda = out.split()[0], out.split()[1]
    base = ".".join(ver.split(".")[:2])
    tvmap = {"2.8":"0.23.0","2.7":"0.22.0","2.6":"0.21.0","2.5":"0.20.1","2.4":"0.19.1",
             "2.3":"0.18.1","2.2":"0.17.2","2.1":"0.16.2","2.0":"0.15.2"}
    cu = "cu" + cuda.replace(".", "")
    idx = "https://download.pytorch.org/whl/" + cu
    tvver = tvmap.get(base)
    if tvver:
        print("repair: torchvision==%s (%s) for torch %s" % (tvver, cu, ver), flush=True)
        _pip("--force-reinstall", "--no-deps", "--extra-index-url", idx, "torchvision==" + tvver)
    if not _tv_ok():
        print("fallback: force torch==2.5.1 torchvision==0.20.1 cu121", flush=True)
        _pip("--force-reinstall", "--extra-index-url", "https://download.pytorch.org/whl/cu121",
             "torch==2.5.1", "torchvision==0.20.1")
    assert _tv_ok(), "torchvision still broken after repair"
print("torchvision OK (subprocess-verified)", flush=True)
for _mod, _spec in [("timm","timm==1.0.15"), ("cv2","opencv-python-headless==4.10.0.84"),
                    ("scipy","scipy"), ("sklearn","scikit-learn")]:
    if _run("import " + _mod)[0] != 0:
        print("installing", _spec, flush=True); _pip(_spec)
'''


def load_fold_arrays_from_dir(fold_root, nfolds: int) -> list[dict]:
    """Read ``fold_<f>/fold_preds.npz`` arrays produced by a config's job."""
    from pathlib import Path

    fold_root = Path(fold_root)
    folds = []
    for f in range(nfolds):
        npz = fold_root / f"fold_{f}" / "fold_preds.npz"
        if not npz.exists():
            raise FileNotFoundError(f"missing fold predictions: {npz}")
        d = np.load(npz, allow_pickle=True)
        folds.append({k: d[k] for k in ("oof_ids", "oof_cont", "oof_labels", "test_ids", "test_cont")})
    return folds


def pull_reuse_fold_preds(reuse_prefix: str, nfolds: int, *, storage_client) -> list[dict]:
    """Pull cached ``fold_<f>/fold_preds.npz`` from a gs:// prefix (free warm cache).

    Lets a config whose predictions already exist from a prior run skip training
    entirely -- the operator only aggregates. ``storage_client`` is injected so
    this unit-tests with a fake client (no network).
    """
    import io
    from pathlib import PurePosixPath

    assert reuse_prefix.startswith("gs://"), reuse_prefix
    bucket_name, _, key_prefix = reuse_prefix[5:].partition("/")
    bucket = storage_client.bucket(bucket_name)
    folds = []
    for f in range(nfolds):
        key = str(PurePosixPath(key_prefix) / f"fold_{f}" / "fold_preds.npz")
        blob = bucket.blob(key)
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        d = np.load(buf, allow_pickle=True)
        folds.append({k: d[k] for k in ("oof_ids", "oof_cont", "oof_labels", "test_ids", "test_cont")})
    return folds


def build_config_job_code(
    config: dict,
    *,
    task_id: str | None,
    id_col: str,
    label_col: str,
    train_images_dir: str,
    test_images_dir: str,
    image_ext: str,
    n_bins: int,
    max_seconds: float,
) -> str:
    """Generate the in-container k-fold regression trainer for one config.

    Runs ALL folds for the config in one job: Ben-Graham (or plain) preprocessing
    at the config resolution, a timm backbone with a single regression output,
    SmoothL1 loss, QWK-selected epoch, hflip+vflip TTA, and writes
    ``fold_<f>/fold_preds.npz`` (oof + fold-mean test continuous preds) which the
    seam's upload tail ships back. No task-specific constants: id/label column,
    image dirs, extension, resolution, backbone, seed, folds are all parameters.
    """
    params = {
        "BACKBONE": str(config["backbone"]),
        "IMG": int(config["image_size"]),
        "SEED": int(config["seed"]),
        "NFOLDS": int(config.get("nfolds", 5)),
        "PREPROCESS": str(config.get("preprocess", "ben_graham")),
        "PRETRAIN_CKPT": str(config.get("pretrain_ckpt", "") or ""),
        "BS": int(config.get("batch_size", 16)),
        "EPOCHS": int(config.get("epochs", 20)),
        "N_BINS": int(n_bins),
        "ID_COL": str(id_col),
        "LABEL_COL": str(label_col),
        "TRAIN_DIR": str(train_images_dir),
        "TEST_DIR": str(test_images_dir),
        "EXT": str(image_ext),
        "TRAIN_BUDGET_S": float(max_seconds),
    }
    header = "".join(f"{k} = {v!r}\n" for k, v in params.items())
    return _TORCHVISION_ABI_REPAIR + "\n" + header + _CONFIG_TRAINER_BODY


# The trainer body executes with the params above already bound as module
# globals. Ben-Graham preprocessing is GATED on PREPROCESS == "ben_graham";
# torchvision is intentionally never imported (transforms are reimplemented on
# PIL+numpy) so only the ABI-verified torch is used.
_CONFIG_TRAINER_BODY = r'''
import os, json, time, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from functools import partial

t0 = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("DIV_CONFIG_START backbone", BACKBONE, "img", IMG, "seed", SEED,
      "cuda", torch.cuda.is_available(), flush=True)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
tr[ID_COL] = tr[ID_COL].astype(str); te[ID_COL] = te[ID_COL].astype(str)

def _img_path(root, idc):
    p = os.path.join(root, idc + EXT)
    if os.path.exists(p):
        return p
    for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        q = os.path.join(root, idc + e)
        if os.path.exists(q):
            return q
    return p

def crop_gray(img, tol=7):
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = np.where(mask.any(1))[0]; cols = np.where(mask.any(0))[0]
    return img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

def ben_graham(path, size, sigmaX=10):
    import cv2
    bgr = cv2.imread(path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = crop_gray(img)
    img = cv2.resize(img, (size, size))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//2, 1, -1)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    img = img * mask[..., None] + 128 * (1 - mask[..., None])
    return img.astype(np.uint8)

def plain_resize(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.uint8)

def load_img(root, idc):
    path = _img_path(root, idc)
    if PREPROCESS == "ben_graham":
        return ben_graham(path, IMG)
    return plain_resize(path, IMG)

print("preprocessing", PREPROCESS, "at", IMG, "px ...", flush=True)
cache = {}
for idc in list(tr[ID_COL]):
    cache[idc] = load_img(TRAIN_DIR, idc)
for idc in list(te[ID_COL]):
    cache[idc] = load_img(TEST_DIR, idc)
print("preprocessed", len(cache), "images", round(time.time()-t0, 1), "s", flush=True)

def to_tensor_norm(pil_img):
    a = np.asarray(pil_img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))

def train_tf(pil_img):
    if random.random() < 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img = pil_img.rotate(random.uniform(-30, 30), resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    return to_tensor_norm(pil_img)

class DS(Dataset):
    def __init__(self, df, tf, has_label=True):
        self.df = df.reset_index(drop=True); self.tf = tf; self.has = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.fromarray(cache[r[ID_COL]]))
        if self.has:
            return x, float(r[LABEL_COL])
        return x, r[ID_COL]

class OptimizedRounder:
    def __init__(self): self.coef_ = [i + 0.5 for i in range(N_BINS - 1)]
    def _loss(self, coef, X, y):
        p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=list(range(N_BINS))).astype(int)
        return -cohen_kappa_score(y, p, weights="quadratic")
    def fit(self, X, y):
        self.coef_ = minimize(partial(self._loss, X=X, y=y), [i + 0.5 for i in range(N_BINS - 1)], method="nelder-mead")["x"]
    def predict(self, X):
        return pd.cut(X, [-np.inf] + list(np.sort(self.coef_)) + [np.inf], labels=list(range(N_BINS))).astype(int)

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(tr[ID_COL].values, tr[LABEL_COL].values))

def infer_cont(model, df):
    dl = DataLoader(DS(df, to_tensor_norm, has_label=False), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model.eval(); ids = []; out = []
    with torch.no_grad():
        for x, idc in dl:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                p0 = model(x).squeeze(1)
                p1 = model(torch.flip(x, dims=[3])).squeeze(1)
                p2 = model(torch.flip(x, dims=[2])).squeeze(1)
            out.append(((p0 + p1 + p2) / 3).float().cpu().numpy()); ids.extend(list(idc))
    return np.asarray(ids), np.concatenate(out)

per_fold_budget = TRAIN_BUDGET_S / max(1, NFOLDS)
for FOLD in range(NFOLDS):
    fold_start = time.time()
    tr_idx, val_idx = splits[FOLD]
    tr_df = tr.iloc[tr_idx].reset_index(drop=True); val_df = tr.iloc[val_idx].reset_index(drop=True)
    dl_tr = DataLoader(DS(tr_df, train_tf), batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dl_val = DataLoader(DS(val_df, to_tensor_norm), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=1).to(device)
    if PRETRAIN_CKPT:
        from google.cloud import storage as _st
        b, _, k = PRETRAIN_CKPT[5:].partition("/")
        _st.Client().bucket(b).blob(k).download_to_filename("/tmp/backbone.pt")
        sd = torch.load("/tmp/backbone.pt", map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, unexp = model.load_state_dict(sd, strict=False)
        print("LOADED_PRETRAIN miss", len(miss), "unexp", len(unexp), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.SmoothL1Loss(); scaler = torch.cuda.amp.GradScaler()
    best_qwk, best_state = -1.0, None
    for ep in range(EPOCHS):
        if time.time() - fold_start > per_fold_budget:
            print("FOLD_BUDGET_HIT fold", FOLD, "epoch", ep, flush=True); break
        model.train()
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).float()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); xs, ys = [], []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    xs.append(model(x).squeeze(1).float().cpu().numpy())
                ys.append(np.array(y))
        vp, vy = np.concatenate(xs), np.concatenate(ys)
        r = OptimizedRounder(); r.fit(vp, vy)
        qwk = cohen_kappa_score(vy, r.predict(vp), weights="quadratic")
        print("FT fold", FOLD, "epoch", ep, "val_qwk", round(float(qwk), 4), "elapsed", round(time.time()-t0, 1), flush=True)
        if qwk > best_qwk:
            best_qwk = qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    oof_ids, oof_cont = infer_cont(model, val_df)
    oof_labels = val_df.set_index(ID_COL).loc[oof_ids, LABEL_COL].values.astype(int)
    test_ids, test_cont = infer_cont(model, te)
    os.makedirs("fold_%d" % FOLD, exist_ok=True)
    np.savez("fold_%d/fold_preds.npz" % FOLD, oof_ids=oof_ids, oof_cont=oof_cont,
             oof_labels=oof_labels, test_ids=test_ids, test_cont=test_cont)
    with open("fold_%d/metrics.json" % FOLD, "w") as f:
        json.dump({"fold": FOLD, "backbone": BACKBONE, "img": IMG, "seed": SEED,
                   "fold_holdout_qwk": float(best_qwk)}, f)
    print("DIV_FOLD_DONE fold", FOLD, "best_qwk", round(float(best_qwk), 4), flush=True)
print("DIV_CONFIG_DONE backbone", BACKBONE, "wall", round(time.time()-t0, 1), flush=True)
'''
