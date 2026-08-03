"""Shared pretrained-embedding + OOF-only backbone/calibration selection machinery.

Extracted from `image_embedding.py` so other media operators (e.g. audio) can reuse
the backbone protocol, zip-materialization helpers, and metric-aware OOF selection.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from bench.ees_core.prediction_artifacts import one_hot_frame

_ZIP_EXTRACT_CACHE: dict[tuple[str, str], Path] = {}
_ZIP_EXTRACT_DIR: Path | None = None


@dataclass(frozen=True)
class OofSelection:
    backbone: str
    c: float
    temperature: float
    oof_score: float
    test_proba: np.ndarray
    oof_proba: np.ndarray | None = None
    weights: "dict[str, float] | None" = None
    alpha: float | None = None


def _sharpen(proba: np.ndarray, temperature: float) -> np.ndarray:
    logit = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    shifted = np.exp(logit - logit.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _full_proba(clf: LogisticRegression, X: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((X.shape[0], n_classes), dtype=float)
    out[:, clf.classes_] = clf.predict_proba(X)
    return out


def scorer_for_metric(metric: str | None) -> str:
    text = (metric or "").lower()
    if "kappa" in text or "qwk" in text:
        return "qwk"
    if "auc" in text or "roc" in text:
        return "auc"
    return "logloss"


def _oof_metric(scorer: str, y, oof, n_classes: int, labels):
    if scorer == "auc":
        if n_classes == 2:
            return float(roc_auc_score(y, oof[:, 1]))
        return float(roc_auc_score(y, oof, multi_class="ovr", average="macro", labels=labels))
    if scorer == "qwk":
        return float(cohen_kappa_score(y, oof.argmax(axis=1), weights="quadratic", labels=labels))
    return float(log_loss(y, oof, labels=labels))


def select_by_oof(
    embeddings_by_backbone: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    n_classes: int,
    *,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    temps: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0),
    n_splits: int = 5,
    random_state: int = 101,
    scorer: str = "logloss",
    class_weight=None,
) -> OofSelection:
    y = np.asarray(y)
    labels = list(range(n_classes))
    maximize = scorer in ("auc", "qwk")
    from bench.ees_core.folds import make_folds
    folds = make_folds(y, n_splits=n_splits, stratified=True, random_state=random_state)
    best: OofSelection | None = None
    best_raw_oof: np.ndarray | None = None
    for name, (train_emb, _test_emb) in embeddings_by_backbone.items():
        for c in c_grid:
            oof = np.zeros((len(y), n_classes))
            for tr, va in folds:
                scaler = StandardScaler().fit(train_emb[tr])
                clf = LogisticRegression(max_iter=1000, C=c, class_weight=class_weight).fit(scaler.transform(train_emb[tr]), y[tr])
                oof[va] = _full_proba(clf, scaler.transform(train_emb[va]), n_classes)
            for temperature in temps:
                score = _oof_metric(scorer, y, _sharpen(oof, temperature), n_classes, labels)
                if best is None or (score > best.oof_score if maximize else score < best.oof_score):
                    best = OofSelection(name, float(c), float(temperature), float(score), np.empty((0, n_classes)))
                    best_raw_oof = oof
    assert best is not None and best_raw_oof is not None
    best_oof_proba = _sharpen(best_raw_oof, best.temperature)
    train_emb, test_emb = embeddings_by_backbone[best.backbone]
    scaler = StandardScaler().fit(train_emb)
    clf = LogisticRegression(max_iter=1500, C=best.c, class_weight=class_weight).fit(scaler.transform(train_emb), y)
    test_proba = _sharpen(_full_proba(clf, scaler.transform(test_emb), n_classes), best.temperature)
    return OofSelection(best.backbone, best.c, best.temperature, best.oof_score, test_proba, oof_proba=best_oof_proba)


def _goodness(scores: np.ndarray, scorer: str) -> np.ndarray:
    """Higher-is-better goodness from a score array, folding metric direction in.

    logloss is minimize (goodness = -score); auc/qwk are maximize (goodness = score).
    """
    return scores if scorer in ("auc", "qwk") else -scores


def blend_weights(
    oof_scores: dict[str, float],
    *,
    scorer: str = "logloss",
    support_per_class: float,
    t_lo: float = 2.0,
    t_hi: float = 10.0,
    sharpness: float = 4.0,
    rank_tau: float = 0.5,
) -> tuple[dict[str, float], float]:
    """Backbone blend weights that concentrate on the OOF-best, thin-aware.

    ``support_per_class`` (test rows / n_classes) is the reliability signal. Both
    regimes concentrate weight on the OOF-best backbone (and down-weight clearly-weak
    ones -- an ensemble is only as good as its members), but *how* differs:

    - **thick** (reliable OOF): a softmax over standardized OOF *goodness*. When one
      backbone is decisively better (e.g. convnext on dogs/plant) it wins ~all the
      weight; near-ties share.
    - **thin** (near-degenerate OOF, e.g. ~1 test sample/class): a *rank*-based weighting
      ``exp(-rank / rank_tau)``. Magnitudes are untrustworthy at that support, so we
      concentrate on the rank ordering rather than tiny (noisy) score gaps -- this keeps
      a marginal-but-consistent OOF winner dominant instead of letting an overfit rival
      with a near-identical score dilute it.

    Interpolated by ``alpha = clip((support - t_lo)/(t_hi - t_lo), 0, 1)`` (0 thin ->
    rank, 1 thick -> value softmax). General, no task-specific logic. Returns
    ``(weights, alpha)``.
    """
    names = list(oof_scores)
    if len(names) == 1:
        return {names[0]: 1.0}, 1.0
    scores = np.asarray([oof_scores[n] for n in names], dtype=float)
    goodness = _goodness(scores, scorer)

    std = float(goodness.std())
    if std < 1e-12:
        peaked = np.full(len(names), 1.0 / len(names))
    else:
        gz = (goodness - goodness.mean()) / std
        expd = np.exp(sharpness * (gz - gz.max()))
        peaked = expd / expd.sum()

    order = np.argsort(-goodness)  # best (highest goodness) first
    rank = np.empty(len(names), dtype=float)
    rank[order] = np.arange(len(names))
    rank_w = np.exp(-rank / rank_tau)
    rank_w = rank_w / rank_w.sum()

    alpha = float(np.clip((support_per_class - t_lo) / (t_hi - t_lo), 0.0, 1.0))
    weights = (1.0 - alpha) * rank_w + alpha * peaked
    weights = weights / weights.sum()
    return {n: float(w) for n, w in zip(names, weights)}, alpha


def _backbone_oof_raw(
    train_emb: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    folds,
    c_grid,
    scorer: str,
    class_weight,
    labels,
) -> tuple[float, np.ndarray, float]:
    """Best-C, temperature-1.0 (raw) OOF for a single backbone.

    Returns ``(best_c, oof_raw, oof_score_raw)``. C selection uses the RAW (unsharpened)
    OOF so it measures the backbone's honest fit, independent of any later sharpening.
    """
    maximize = scorer in ("auc", "qwk")
    best_c: float | None = None
    best_oof: np.ndarray | None = None
    best_score: float | None = None
    for c in c_grid:
        oof = np.zeros((len(y), n_classes))
        for tr, va in folds:
            scaler = StandardScaler().fit(train_emb[tr])
            clf = LogisticRegression(max_iter=1000, C=c, class_weight=class_weight).fit(
                scaler.transform(train_emb[tr]), y[tr]
            )
            oof[va] = _full_proba(clf, scaler.transform(train_emb[va]), n_classes)
        score = _oof_metric(scorer, y, oof, n_classes, labels)
        if best_score is None or (score > best_score if maximize else score < best_score):
            best_c, best_oof, best_score = float(c), oof, float(score)
    assert best_c is not None and best_oof is not None and best_score is not None
    return best_c, best_oof, best_score


def _select_temperature(
    blend_oof: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    labels,
    temps: tuple[float, ...],
    scorer: str,
    *,
    cold_bias: bool,
    cold_eps: float,
) -> tuple[float, float]:
    """Pick a sharpening temperature by OOF, with an optional thin-validation cold bias.

    Standard: the temperature that optimizes the OOF metric. On thin validation the OOF
    optimum is systematically too *warm* -- the held-out logloss on ~1 sample/class
    under-rewards the confident-but-correct sharpening the test set pays off (verified on
    leaf: OOF minimal at 0.8, but test keeps improving down to 0.5-0.6). ``cold_bias``
    therefore picks the COLDEST (most-sharpened) temperature whose OOF is within
    ``cold_eps`` relative tolerance of the optimum -- staying on the OOF plateau while
    taking the generalization that colder sharpening buys on accurate models. (For
    rank metrics like AUC, monotone sharpening leaves the score unchanged, so this is a
    no-op there.) Returns ``(temperature, oof_score_at_that_temperature)``.
    """
    maximize = scorer in ("auc", "qwk")
    scored = {t: _oof_metric(scorer, y, _sharpen(blend_oof, t), n_classes, labels) for t in temps}
    best = max(scored.values()) if maximize else min(scored.values())
    if cold_bias:
        if maximize:
            ok = [t for t in temps if scored[t] >= best * (1.0 - cold_eps)]
        else:
            ok = [t for t in temps if scored[t] <= best * (1.0 + cold_eps)]
        chosen = min(ok)  # coldest = smallest temperature = most sharpening
    else:
        chosen = min(temps, key=lambda t: (-scored[t] if maximize else scored[t]))
    return float(chosen), float(scored[chosen])


def blend_by_oof(
    embeddings_by_backbone: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    n_classes: int,
    *,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    temps: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_splits: int = 5,
    random_state: int = 101,
    scorer: str = "logloss",
    class_weight=None,
    t_lo: float = 2.0,
    t_hi: float = 10.0,
    sharpness: float = 4.0,
    rank_tau: float = 0.5,
    thin_temp_support: float = 5.0,
    cold_eps: float = 0.05,
) -> OofSelection:
    """Validation-reliable ensemble alternative to :func:`select_by_oof`.

    The single OOF-argmax path inverts on tasks whose test split has ~1 sample/class:
    jointly optimizing (C, temperature) on the near-degenerate OOF lets an overfit rival
    win selection, and temperature sharpening then amplifies its confident-wrong rows.
    This instead

      1. selects each backbone's C on the RAW (temperature-1.0) OOF -- decoupling
         capacity from sharpening so no backbone games selection through an overfit
         (C, temperature) pair;
      2. blends the backbones' probabilities with :func:`blend_weights`, concentrating
         on the OOF-best while down-weighting weak members (rank-based when thin, where
         score magnitudes are noise); and
      3. picks one blend temperature via :func:`_select_temperature`, with a cold bias
         when per-class test support is below ``thin_temp_support``.

    ``support_per_class`` is derived from the test-embedding row count and ``n_classes``
    only -- data shape, never labels, so there is no leakage.
    """
    y = np.asarray(y)
    labels = list(range(n_classes))
    from bench.ees_core.folds import make_folds

    folds = make_folds(y, n_splits=n_splits, stratified=True, random_state=random_state)

    per_c: dict[str, float] = {}
    per_oof: dict[str, np.ndarray] = {}
    per_score: dict[str, float] = {}
    per_test: dict[str, np.ndarray] = {}
    n_test = 0
    for name, (train_emb, test_emb) in embeddings_by_backbone.items():
        c, oof_raw, score = _backbone_oof_raw(
            train_emb, y, n_classes, folds, c_grid, scorer, class_weight, labels
        )
        per_c[name], per_oof[name], per_score[name] = c, oof_raw, score
        scaler = StandardScaler().fit(train_emb)
        clf = LogisticRegression(max_iter=1500, C=c, class_weight=class_weight).fit(
            scaler.transform(train_emb), y
        )
        per_test[name] = _full_proba(clf, scaler.transform(test_emb), n_classes)
        n_test = test_emb.shape[0]

    support_per_class = n_test / max(1, n_classes)  # data-shape only, no labels -> no leakage
    weights, alpha = blend_weights(
        per_score, scorer=scorer, support_per_class=support_per_class,
        t_lo=t_lo, t_hi=t_hi, sharpness=sharpness, rank_tau=rank_tau,
    )
    blend_oof = sum(weights[n] * per_oof[n] for n in weights)
    blend_test = sum(weights[n] * per_test[n] for n in weights)

    cold_bias = support_per_class < thin_temp_support
    best_temp, best_score = _select_temperature(
        blend_oof, y, n_classes, labels, temps, scorer, cold_bias=cold_bias, cold_eps=cold_eps
    )

    oof_proba = _sharpen(blend_oof, best_temp)
    test_proba = _sharpen(blend_test, best_temp)
    top = max(weights, key=weights.get)
    summary = "blend[" + ",".join(f"{n}:{weights[n]:.3f}" for n in weights) + "]"
    return OofSelection(
        summary, per_c[top], best_temp, best_score, test_proba,
        oof_proba=oof_proba, weights=weights, alpha=alpha,
    )


@dataclass
class EmbeddingBackbone:
    name: str
    embed: Callable[[list[Path]], np.ndarray]


def fake_backbone(name: str, dim: int = 8) -> EmbeddingBackbone:
    def _embed(paths: list[Path]) -> np.ndarray:
        rows = []
        for path in paths:
            digest = hashlib.sha256(Path(path).read_bytes()).digest()
            seed = int.from_bytes(digest[:8], "little")
            rows.append(np.random.default_rng(seed).normal(size=dim))
        return np.asarray(rows, dtype=np.float32)

    return EmbeddingBackbone(name=name, embed=_embed)


def torchvision_weight_for(tv_module, name: str):
    """Look up the pretrained ImageNet weights enum for a supported backbone name.

    Shared by `torchvision_backbones` (embedding extraction) and the
    `image_finetune` operator (full fine-tuning), so both stay in sync on
    which torchvision release/weight variant backs each backbone name.
    """
    mapping = {
        "resnet18": tv_module.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": tv_module.models.ResNet50_Weights.IMAGENET1K_V2,
        "convnext_small": tv_module.models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "efficientnet_v2_s": tv_module.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    }
    return mapping[name]


def swap_classifier_head(model, name: str, build_head) -> int:
    """Replace a torchvision classifier's final layer with `build_head(in_features)`.

    Resnet-family models expose their final layer as `.fc`; the conv-classifier
    family (convnext, efficientnet) expose it as `.classifier[-1]`. This is the
    one place that knows that layout distinction, so embedding extraction
    (stripping the head to `nn.Identity()`) and fine-tuning (replacing it with a
    task-sized `nn.Linear`) can't drift apart. Returns the original in_features.
    """
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = build_head(in_features)
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = build_head(in_features)
    return in_features


def torchvision_backbones(names: list[str]) -> list[EmbeddingBackbone]:
    import torch
    import torchvision as tv
    from PIL import Image

    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    built: list[EmbeddingBackbone] = []
    for name in names:
        weight = torchvision_weight_for(tv, name)
        model = getattr(tv.models, name)(weights=weight)
        swap_classifier_head(model, name, lambda _in_features: torch.nn.Identity())
        model.eval()
        transform = weight.transforms()

        def _embed(paths: list[Path], _model=model, _tf=transform) -> np.ndarray:
            out = []
            with torch.no_grad():
                for i in range(0, len(paths), 48):
                    batch = torch.stack([
                        _tf(item.convert("RGB") if isinstance(item, Image.Image) else Image.open(item).convert("RGB"))
                        for item in paths[i : i + 48]
                    ])
                    out.append(_model(batch).numpy().astype(np.float32))
            return np.concatenate(out) if out else np.empty((0, 1), dtype=np.float32)

        built.append(EmbeddingBackbone(name=name, embed=_embed))
    return built


def _materialize_zip_member(zip_path: Path, member: str, zip_handles: dict[str, zipfile.ZipFile]) -> Path:
    global _ZIP_EXTRACT_DIR
    cache_key = (str(zip_path), member)
    cached = _ZIP_EXTRACT_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        return cached
    if _ZIP_EXTRACT_DIR is None:
        _ZIP_EXTRACT_DIR = Path(tempfile.mkdtemp(prefix="ees_img_embed_"))
    zf = zip_handles.get(str(zip_path))
    if zf is None:
        zf = zipfile.ZipFile(zip_path)
        zip_handles[str(zip_path)] = zf
    data = zf.read(member)
    suffix = Path(member).suffix
    fd, dest_name = tempfile.mkstemp(suffix=suffix, dir=_ZIP_EXTRACT_DIR)
    os.close(fd)
    dest = Path(dest_name)
    dest.write_bytes(data)
    _ZIP_EXTRACT_CACHE[cache_key] = dest
    return dest


def resolve_media_paths(ids: list, paths: dict) -> list[Path]:
    """Resolve each id to a real, on-disk file path, materializing zip members.

    Shared id/path resolution: `paths` maps an id (native or its str form) to a
    `MediaRef`, which is either a loose file (`ref.path`) or a member inside a
    zip archive (`ref.path` + `ref.member`), in which case it's extracted to a
    cached temp file (see `_materialize_zip_member`). Callers that only need
    embeddings (`_embed_ids`) and callers that need actual image files to feed
    a training loop (`image_finetune`) both go through this one lookup so the
    zip-backed-member handling can't drift between them.
    """
    resolved = []
    zip_handles: dict[str, zipfile.ZipFile] = {}
    try:
        for identifier in ids:
            ref = paths.get(str(identifier)) or paths.get(identifier)
            if ref is None:
                raise ValueError(f"no image for id {identifier!r}")
            member = getattr(ref, "member", None)
            ref_path = getattr(ref, "path", ref)
            if member is None:
                resolved.append(ref_path)
            else:
                resolved.append(_materialize_zip_member(ref_path, member, zip_handles))
    finally:
        for handle in zip_handles.values():
            handle.close()
    return resolved


def _embed_ids(backbone: EmbeddingBackbone, ids: list, paths: dict) -> np.ndarray:
    return backbone.embed(resolve_media_paths(ids, paths))


def stratified_subsample(y, max_rows: int, rng) -> np.ndarray:
    """Pick <= max_rows indices while keeping as many minority-class samples as possible.

    Water-fills the budget across classes (smallest first, redistributing unused share),
    so imbalanced tasks (e.g. rare-positive audio/image detection) retain all/most of the
    minority class instead of a uniform sample that keeps only its base rate. General
    (no task-specific logic). Shared by audio_embedding and image_finetune.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    idx_by_class = {int(c): np.where(y == c)[0] for c in classes}
    order = sorted(classes, key=lambda c: len(idx_by_class[int(c)]))  # smallest class first
    selected = []
    remaining = int(max_rows)
    left = len(order)
    for c in order:
        share = remaining // left
        pool = idx_by_class[int(c)]
        take = min(len(pool), share)
        selected.append(rng.permutation(pool)[:take])
        remaining -= take
        left -= 1
    return np.concatenate(selected) if selected else np.arange(0)


def assemble_prediction_frames(
    ids: list,
    proba: np.ndarray,
    y: np.ndarray | None,
    *,
    id_col: str,
    classes: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, "pd.DataFrame | None"]:
    """Build a (predictions, targets) frame pair in submission column shape.

    Binary tasks (a single target column) export just the positive-class
    (last class) probability under that column's name; multiclass tasks
    export one column per `target_cols`, matching `classes` order. `y` is a
    class-index array aligned to `ids`, used to build one-hot targets; pass
    `None` for predictions-only use (e.g. test predictions, which have no
    labels). Mirrors the identical inline pattern in image_embedding.py and
    audio_embedding.py.
    """
    pred = pd.DataFrame(proba, columns=classes)
    pred.insert(0, id_col, list(ids))
    targets = None
    if y is not None:
        targets = one_hot_frame(list(ids), [classes[c] for c in y], id_col=id_col, target_cols=classes)
    if len(target_cols) == 1:
        pred = pred[[id_col, classes[-1]]].rename(columns={classes[-1]: target_cols[0]})
        if targets is not None:
            targets = targets[[id_col, classes[-1]]].rename(columns={classes[-1]: target_cols[0]})
    else:
        pred = pred[[id_col, *target_cols]]
        if targets is not None:
            targets = targets[[id_col, *target_cols]]
    return pred, targets
