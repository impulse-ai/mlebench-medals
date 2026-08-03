"""General audio multi-label operator for composite-id long-format submissions.

Detects (from EVIDENCE, never a task id) the following contract:

  * audio recordings are present (loose WAV/AIFF or in archives);
  * the sample submission is LONG format: exactly two columns ``(Id, prob)``
    where every ``Id`` is a *composite* integer ``Id = group_id * base + class_id``
    (one row per (recording, class) pair);
  * a multi-label label file maps each recording to a *variable-length* list of
    integer class ids (with hidden ``?`` rows marking the test recordings).

When that shape is detected the operator reproduces the whale/birds audio
recipe generically:

  log-mel spectrogram + per-frequency-bin background subtraction ->
  resnet18 K-way sigmoid CNN (5-fold, best-epoch, TTA) blended with a classical
  spectrogram-band logistic head, then formatted back into the exact
  composite-id long-format submission in sample order.

Nothing here hardcodes birds' 19 species, the ``* 100`` multiplier, or the task
id: ``base`` and ``K`` are parsed from the sample submission id structure, and
the recording map / label file are discovered by structural evidence.
"""
from __future__ import annotations

import io
import json
import math
import os
import time
import wave
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.signal

from bench.adapters.task_detection import AUDIO_EXTS, _find_sample_submission
from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

# ---------------------------------------------------------------------------
# audio decoding + spectrogram features (pure numpy/scipy — no torch)
# ---------------------------------------------------------------------------


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    suf = path.suffix.lower()
    raw = path.read_bytes()
    if suf in (".aif", ".aiff"):
        import aifc  # aifc removed in py3.13 — keep the import lazy

        a = aifc.open(io.BytesIO(raw))
        data = np.frombuffer(a.readframes(a.getnframes()), dtype=">i2").astype(np.float32)
        fr, ch = a.getframerate(), a.getnchannels()
    else:
        w = wave.open(io.BytesIO(raw))
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        fr, ch = w.getframerate(), w.getnchannels()
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / 32768.0, fr


def _mel_filterbank(sr: int, nfft: int, nmel: int, fmin: float = 200.0, fmax: float = 8000.0) -> np.ndarray:
    fmax = min(fmax, sr / 2)

    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), nmel + 2)
    hz = mel2hz(mels)
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, nfft // 2)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for m in range(1, nmel + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def log_mel_bgsub(
    samples: np.ndarray,
    sr: int,
    *,
    nfft: int = 1024,
    hop: int = 256,
    nmel: int = 128,
) -> np.ndarray:
    """log-mel spectrogram with per-frequency-bin background subtraction.

    The bg-subtraction (subtract per-mel median over time, clip>=0) is the
    general whale/birds denoising lever. Returns a z-normed ``(nmel, T)`` array.
    """
    d = np.asarray(samples, dtype=np.float32)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d / (np.abs(d).max() + 1e-9)
    fb = _mel_filterbank(sr, nfft, nmel)
    noverlap = max(0, nfft - hop)
    _f, _t, Sxx = scipy.signal.spectrogram(d, fs=sr, nperseg=nfft, noverlap=noverlap, mode="magnitude")
    mel = fb @ (Sxx ** 2)
    log = np.log1p(mel * 1000.0)
    bg = np.median(log, axis=1, keepdims=True)
    log = np.clip(log - bg, 0.0, None)
    log = (log - log.mean()) / (log.std() + 1e-6)
    return log.astype(np.float32)


def spectrogram_band_features(samples: np.ndarray, sr: int, *, nfft: int = 512, nbands: int = 32) -> np.ndarray:
    """Classical log-spectrogram band statistics with per-freq bg subtraction.

    2 sources (raw log-spec + denoised) x nbands x 4 stats. Pure numpy/scipy.
    """
    d = np.asarray(samples, dtype=np.float32)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d / (np.abs(d).max() + 1e-9)
    _f, _t, Sxx = scipy.signal.spectrogram(d, fs=sr, nperseg=nfft, noverlap=nfft // 2)
    S = np.log1p(Sxx)
    bg = np.median(S, axis=1, keepdims=True)
    D = np.clip(S - bg, 0.0, None)
    bands = np.array_split(np.arange(S.shape[0]), nbands)
    feats: list[float] = []
    for src in (S, D):
        for idx in bands:
            band = src[idx].mean(axis=0)  # avg over freqs -> time series
            feats += [band.mean(), band.std(), band.max(), float(np.percentile(band, 90))]
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# composite-id parse / format  (the #1 failure mode -> generic + round-trip tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeIdScheme:
    base: int  # multiplier: Id = group_id * base + class_id
    k: int  # number of classes per group
    groups: list[int]  # ordered distinct group ids present in the submission


def parse_composite_ids(ids, *, valid_group_ids: set[int] | None = None) -> CompositeIdScheme:
    """Infer ``base`` and ``K`` from a composite long-format id column.

    Ids form contiguous runs of constant length ``K`` (one per class), one run
    per recording, with run starts equal to ``group_id * base``. We recover
    ``base`` as the gcd of the run starts and validate that, for every id,
    ``id % base`` yields exactly ``{0..K-1}`` per group. When a set of known
    recording ids is supplied we additionally require ``id // base`` to fall in
    it, and fall back to round-number bases if the gcd disagrees.
    """
    ints = sorted({int(x) for x in ids})
    if len(ints) < 2:
        raise ValueError("composite id column needs >= 2 rows")
    # contiguous runs
    runs: list[list[int]] = [[ints[0]]]
    for prev, cur in zip(ints, ints[1:]):
        if cur == prev + 1:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    lengths = {len(r) for r in runs}
    if len(runs) < 2 or len(lengths) != 1:
        raise ValueError(f"composite ids are not constant-length contiguous runs (lengths={sorted(lengths)}, groups={len(runs)})")
    k = len(runs[0])
    if k < 2:
        raise ValueError("composite groups must have K >= 2 classes")
    starts = [r[0] for r in runs]

    def _validate(base: int) -> bool:
        if base < k:
            return False
        for run in runs:
            start = run[0]
            if start % base != 0:
                return False
            if [x - start for x in run] != list(range(k)):
                return False
            if any((x % base) != (x - start) for x in run):
                return False
        if valid_group_ids is not None:
            if any((x // base) not in valid_group_ids for x in ints):
                return False
        return True

    # Preference order: the gcd of the run starts is the natural stride and is
    # correct in the common case (birds: gcd(100,200,...) == 100). Round-number
    # bases (powers of ten >= K) are only fallbacks for the pathological case
    # where every test group id shares a common factor larger than the true base
    # -- there the supplied ``valid_group_ids`` rejects the gcd and pins the base.
    candidates: list[int] = []
    gcd_base = reduce(math.gcd, starts)
    if gcd_base:
        candidates.append(gcd_base)
    power = 10
    while power <= ints[-1]:
        if power >= k and power not in candidates:
            candidates.append(power)
        power *= 10
    for base in candidates:  # gcd first, then round-number fallbacks ascending
        if _validate(base):
            groups = [s // base for s in starts]
            return CompositeIdScheme(base=int(base), k=int(k), groups=groups)
    raise ValueError(f"could not resolve a consistent composite base (tried {candidates}, K={k})")


def format_composite_submission(
    sample: pd.DataFrame,
    id_col: str,
    prob_col: str,
    scheme: CompositeIdScheme,
    group_index: dict[int, int],
    proba: np.ndarray,
) -> pd.DataFrame:
    """Map a (n_groups, K) probability matrix back into sample-order long format."""
    out = sample[[id_col]].copy()
    ids = out[id_col].astype(np.int64).to_numpy()
    values = np.empty(len(ids), dtype=np.float64)
    for i, raw in enumerate(ids):
        g = int(raw) // scheme.base
        c = int(raw) % scheme.base
        row = group_index.get(g)
        if row is None or c >= scheme.k:
            raise ValueError(f"submission id {raw} maps to unknown group/class (g={g}, c={c})")
        values[i] = proba[row, c]
    if np.isnan(values).any():
        raise ValueError("formatted submission contains NaN probabilities")
    out[prob_col] = values
    return out


# ---------------------------------------------------------------------------
# evidence-gated discovery: recording map, multilabel labels, contract
# ---------------------------------------------------------------------------


def _iter_audio_files(data_dir: Path):
    for p in Path(data_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def _read_delimited(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        text = Path(path).read_text(errors="replace")
    except Exception:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append([tok.strip() for tok in line.split(",")])
    return rows


def _is_int(tok: str) -> bool:
    try:
        int(tok)
        return True
    except (TypeError, ValueError):
        return False


def discover_recording_map(data_dir: Path, audio_stems: set[str], exclude: set[Path]) -> dict[int, str] | None:
    """Find a file mapping integer recording id -> audio filename stem."""
    data_dir = Path(data_dir)
    best: dict[int, str] | None = None
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path in exclude or path.suffix.lower() not in (".txt", ".csv"):
            continue
        rows = _read_delimited(path)
        if len(rows) < 3:
            continue
        body = rows[1:] if not _is_int(rows[0][0]) else rows  # skip header
        mapping: dict[int, str] = {}
        for row in body:
            if len(row) < 2 or not _is_int(row[0]):
                mapping = {}
                break
            stem = Path(row[1]).stem
            if stem not in audio_stems:
                mapping = {}
                break
            mapping[int(row[0])] = row[1]
        if mapping and len(mapping) >= 3:
            if best is None or len(mapping) > len(best):
                best = mapping
    return best


@dataclass
class MultilabelSet:
    train: dict[int, set[int]]  # recording id -> class ids (known/train)
    test: set[int]  # recording ids marked hidden ("?")


def discover_multilabel_labels(data_dir: Path, k: int, exclude: set[Path]) -> MultilabelSet | None:
    """Find a file mapping recording id -> variable-length integer class list.

    Rows are ``rec_id, c1, c2, ...`` (0..K classes) or ``rec_id, ?`` (hidden
    test). The variable-length integer lists + presence of ``?`` rows are the
    multi-label signal distinguishing this from single-label manifests.
    """
    data_dir = Path(data_dir)
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path in exclude or path.suffix.lower() not in (".txt", ".csv"):
            continue
        rows = _read_delimited(path)
        if len(rows) < 3:
            continue
        body = rows[1:] if not _is_int(rows[0][0]) else rows  # skip header
        train: dict[int, set[int]] = {}
        test: set[int] = set()
        has_hidden = False
        variable_len = False
        ok = True
        prev_len = None
        for row in body:
            if not row or not _is_int(row[0]):
                ok = False
                break
            rid = int(row[0])
            rest = [t for t in row[1:] if t != ""]
            if rest and rest[0] == "?":
                test.add(rid)
                has_hidden = True
                continue
            labels: set[int] = set()
            for tok in rest:
                if not _is_int(tok):
                    ok = False
                    break
                val = int(tok)
                if val < 0 or val >= k:
                    ok = False
                    break
                labels.add(val)
            if not ok:
                break
            train[rid] = labels
            if prev_len is not None and len(labels) != prev_len:
                variable_len = True
            prev_len = len(labels)
        if ok and has_hidden and variable_len and len(train) >= 3:
            return MultilabelSet(train=train, test=test)
    return None


@dataclass
class AudioMultilabelContract:
    id_col: str
    prob_col: str
    scheme: CompositeIdScheme
    rec_map: dict[int, str]  # rec id -> filename stem/name
    labels: MultilabelSet
    audio_index: dict[str, Path]  # stem -> path
    sample: pd.DataFrame
    detail: dict = field(default_factory=dict)


def detect_audio_multilabel(data_dir: Path) -> AudioMultilabelContract | None:
    """Return a contract iff the composite-id long-format multi-label AUDIO shape
    is present, else ``None``. Pure evidence — no task id."""
    data_dir = Path(data_dir)
    audio_paths = list(_iter_audio_files(data_dir))
    if len(audio_paths) < 4:
        return None
    audio_index: dict[str, Path] = {}
    for p in audio_paths:
        audio_index.setdefault(p.stem, p)
        audio_index.setdefault(p.name, p)
    audio_stems = {p.stem for p in audio_paths}

    sample_path = _find_sample_submission(data_dir)
    if sample_path is None:
        return None
    sample = pd.read_csv(sample_path)
    if sample.shape[1] != 2:
        return None
    id_col, prob_col = sample.columns[0], sample.columns[1]
    id_series = sample[id_col]
    if not np.issubdtype(id_series.dtype, np.integer):
        # tolerate ints stored as float/object as long as they are whole numbers
        try:
            as_int = id_series.astype(np.int64)
        except (TypeError, ValueError):
            return None
        if not np.allclose(as_int, id_series.astype(float)):
            return None

    exclude = {sample_path}
    # provisional parse (no group validation yet) just to learn K
    try:
        provisional = parse_composite_ids(id_series.tolist())
    except ValueError:
        return None

    rec_map = discover_recording_map(data_dir, audio_stems, exclude)
    if rec_map is None:
        return None
    labels = discover_multilabel_labels(data_dir, provisional.k, exclude)
    if labels is None:
        return None

    # authoritative parse validated against the known recording ids
    try:
        scheme = parse_composite_ids(id_series.tolist(), valid_group_ids=set(rec_map.keys()))
    except ValueError:
        scheme = provisional

    detail = {
        "n_audio_files": len(audio_paths),
        "base": scheme.base,
        "k": scheme.k,
        "n_test_groups": len(scheme.groups),
        "n_train_labeled": len(labels.train),
        "n_hidden": len(labels.test),
    }
    return AudioMultilabelContract(
        id_col=str(id_col),
        prob_col=str(prob_col),
        scheme=scheme,
        rec_map=rec_map,
        labels=labels,
        audio_index=audio_index,
        sample=sample,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# modeling
# ---------------------------------------------------------------------------


def pooled_auc(oof: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    flat_y = y.ravel()
    if len(np.unique(flat_y)) < 2:
        return float("nan")
    return float(roc_auc_score(flat_y, oof.ravel()))


def _rankavg(*mats: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    acc = np.zeros_like(mats[0], dtype=float)
    for m in mats:
        acc += rankdata(m.ravel()).reshape(m.shape)
    return acc / len(mats)


def classical_logreg_head(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    n_splits: int = 5,
    c: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class logistic regression: OOF on train + full-fit test predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n, k = Ytr.shape
    scaler = StandardScaler().fit(Xtr)
    Xs, Xt = scaler.transform(Xtr), scaler.transform(Xte)
    oof = np.zeros((n, k))
    test = np.zeros((Xte.shape[0], k))
    splits = max(2, min(n_splits, n))
    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    for tr, va in kf.split(Xs):
        for s in range(k):
            col = Ytr[tr, s]
            if len(np.unique(col)) < 2:
                oof[va, s] = float(col.mean())
                continue
            m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
            m.fit(Xs[tr], col)
            oof[va, s] = m.predict_proba(Xs[va])[:, 1]
    for s in range(k):
        col = Ytr[:, s]
        if len(np.unique(col)) < 2:
            test[:, s] = float(col.mean())
            continue
        m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        m.fit(Xs, col)
        test[:, s] = m.predict_proba(Xt)[:, 1]
    return oof, test


def torch_cnn_head(
    mels_train: list[np.ndarray],
    Ytr: np.ndarray,
    mels_test: list[np.ndarray],
    *,
    epochs: int = 30,
    n_splits: int = 5,
    backbone: str = "resnet18",
    random_state: int = 42,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """resnet K-way sigmoid CNN on log-mel bg-sub images. Returns (oof, test)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision as tv
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if threads:
        torch.set_num_threads(threads)

    k = Ytr.shape[1]
    width = int(np.median([m.shape[1] for m in mels_train + mels_test]))
    nmel = mels_train[0].shape[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def prep(L: np.ndarray, train: bool) -> torch.Tensor:
        T = L.shape[1]
        if train and T > width:
            s = np.random.randint(0, T - width)
            L = L[:, s : s + width]
        elif T > width:
            s = (T - width) // 2
            L = L[:, s : s + width]
        elif T < width:
            L = np.pad(L, ((0, 0), (0, width - T)), mode="wrap")
        if train:
            L = L.copy()
            for _ in range(2):
                fw = np.random.randint(0, max(1, nmel // 6))
                f0 = np.random.randint(0, max(1, nmel - fw))
                L[f0 : f0 + fw, :] = 0
            for _ in range(2):
                tw = np.random.randint(0, max(1, width // 5))
                t0 = np.random.randint(0, max(1, width - tw))
                L[:, t0 : t0 + tw] = 0
        t = torch.from_numpy(np.ascontiguousarray(L)).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t = t.repeat(1, 3, 1, 1)
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        t = (t - mean) / std
        return t.squeeze(0)

    def make_model():
        weights = getattr(tv.models, {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights"}[backbone]).IMAGENET1K_V1
        m = getattr(tv.models, backbone)(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, k)
        return m

    pos = Ytr.sum(0)
    neg = len(mels_train) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20)).float()

    n = len(mels_train)
    oof = np.zeros((n, k), dtype=np.float32)
    test = np.zeros((len(mels_test), k), dtype=np.float32)
    splits = max(2, min(n_splits, n))
    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    bs = 32
    for tr, va in kf.split(np.arange(n)):
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        best_va, best_state = -1.0, None
        for _ep in range(epochs):
            model.train()
            order = np.random.permutation(len(tr))
            for b in range(0, len(tr), bs):
                idx = tr[order[b : b + bs]]
                xb = torch.stack([prep(mels_train[j], True) for j in idx])
                yb = torch.from_numpy(Ytr[idx]).float()
                if np.random.rand() < 0.5:
                    lam = np.random.beta(0.4, 0.4)
                    perm = torch.randperm(xb.size(0))
                    xb = lam * xb + (1 - lam) * xb[perm]
                    yb = lam * yb + (1 - lam) * yb[perm]
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vp = np.zeros((len(va), k))
                for i, vi in enumerate(va):
                    vp[i] = torch.sigmoid(model(prep(mels_train[vi], False).unsqueeze(0))).numpy()[0]
            vauc = roc_auc_score(Ytr[va].ravel(), vp.ravel()) if len(np.unique(Ytr[va])) > 1 else 0.0
            if vauc > best_va:
                best_va = vauc
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for i, vi in enumerate(va):
                preds = [torch.sigmoid(model(prep(mels_train[vi], t > 0).unsqueeze(0))).numpy()[0] for t in range(3)]
                oof[vi] = np.mean(preds, axis=0)
            for i in range(len(mels_test)):
                preds = [torch.sigmoid(model(prep(mels_test[i], t > 0).unsqueeze(0))).numpy()[0] for t in range(3)]
                test[i] += np.mean(preds, axis=0) / splits
    return oof, test


# ---------------------------------------------------------------------------
# operator entry point
# ---------------------------------------------------------------------------


def _fail(reason: str) -> CandidateResult:
    return CandidateResult(
        candidate_id="audio_multilabel",
        recipe_id="audio_multilabel",
        success=False,
        submission_path=None,
        validation={},
        score=None,
        medal="none",
        final_artifact_source="ees_core:audio_multilabel",
        no_score_reason=reason,
    )


def run_audio_multilabel_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str,
    cnn_head: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    use_cnn: bool | None = None,
    epochs: int | None = None,
    n_splits: int | None = None,
    random_state: int = 42,
) -> CandidateResult:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        contract = detect_audio_multilabel(Path(data_dir))
        if contract is None:
            return _fail("audio_multilabel: composite-id long-format multilabel-audio shape not detected")

        scheme = contract.scheme
        k = scheme.k
        # train recordings = labeled recs that have audio; test recs = submission groups
        def _resolve_audio(rid: int) -> Path | None:
            name = contract.rec_map.get(rid)
            if name is None:
                return None
            return contract.audio_index.get(Path(name).stem) or contract.audio_index.get(name)

        train_recs, Ytr_rows = [], []
        for rid, labs in sorted(contract.labels.train.items()):
            if _resolve_audio(rid) is None:
                continue
            train_recs.append(rid)
            row = np.zeros(k, dtype=np.float32)
            for c in labs:
                row[c] = 1.0
            Ytr_rows.append(row)
        if len(train_recs) < 5:
            return _fail(f"audio_multilabel: only {len(train_recs)} usable train recordings")
        Ytr = np.vstack(Ytr_rows)

        test_recs = [g for g in scheme.groups]
        missing = [g for g in test_recs if _resolve_audio(g) is None]
        if missing:
            return _fail(f"audio_multilabel: {len(missing)} test recordings have no audio (e.g. {missing[:3]})")
        group_index = {g: i for i, g in enumerate(test_recs)}

        n_splits = n_splits or int(os.environ.get("EES_AUDIO_ML_FOLDS", "5"))
        epochs = epochs or int(os.environ.get("EES_AUDIO_ML_EPOCHS", "30"))
        if use_cnn is None:
            use_cnn = os.environ.get("EES_AUDIO_ML_DISABLE_CNN") != "1"

        # ---- decode once; build classical band features (always) + mel (for CNN) ----
        band_train, band_test = [], []
        mel_train, mel_test = [], []
        want_mel = use_cnn or cnn_head is not None
        for rid in train_recs:
            samples, sr = decode_audio(_resolve_audio(rid))
            band_train.append(spectrogram_band_features(samples, sr))
            if want_mel:
                mel_train.append(log_mel_bgsub(samples, sr))
        for rid in test_recs:
            samples, sr = decode_audio(_resolve_audio(rid))
            band_test.append(spectrogram_band_features(samples, sr))
            if want_mel:
                mel_test.append(log_mel_bgsub(samples, sr))
        Xc_tr = np.vstack(band_train)
        Xc_te = np.vstack(band_test)

        # ---- classical head (sklearn) ----
        oof_cls, test_cls = classical_logreg_head(Xc_tr, Ytr, Xc_te, n_splits=n_splits, random_state=random_state)
        heads = {"classical": (oof_cls, test_cls)}

        # ---- CNN head (torch) ----
        cnn_used = False
        if cnn_head is not None:
            heads["cnn"] = cnn_head(mel_train, Ytr, mel_test)
            cnn_used = True
        elif use_cnn:
            threads = int(os.environ.get("EES_AUDIO_ML_THREADS", str(max(1, (os.cpu_count() or 8) - 2))))
            heads["cnn"] = torch_cnn_head(
                mel_train, Ytr, mel_test,
                epochs=epochs, n_splits=n_splits, random_state=random_state, threads=threads,
            )
            cnn_used = True

        # ---- blend by OOF pooled AUC; CNN-weighted rank average when available ----
        if "cnn" in heads:
            o_cnn, t_cnn = heads["cnn"]
            o_cls, t_cls = heads["classical"]
            best_w, best_score, best_pair = 0.5, -1.0, None
            for w in (0.3, 0.4, 0.5, 0.6, 0.7):
                oof_blend = w * _rankavg(o_cnn) + (1 - w) * _rankavg(o_cls)
                score = pooled_auc(oof_blend, Ytr)
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_pair = (oof_blend, w * _rankavg(t_cnn) + (1 - w) * _rankavg(t_cls))
            # also consider each head alone
            for name, (o, t) in heads.items():
                score = pooled_auc(o, Ytr)
                if score > best_score:
                    best_score, best_w, best_pair = score, name, (o, t)
            oof_final, test_final = best_pair
            selected = f"cnn_blend_w{best_w}" if isinstance(best_w, float) else best_w
        else:
            oof_final, test_final = oof_cls, test_cls
            best_score = pooled_auc(oof_cls, Ytr)
            selected = "classical"

        # min-max scale test to valid [0,1] (rank blends are AUC-invariant)
        rng = test_final.max() - test_final.min()
        test_scaled = (test_final - test_final.min()) / (rng + 1e-12)

        # ---- format the composite-id long submission in sample order ----
        sub = format_composite_submission(
            contract.sample, contract.id_col, contract.prob_col, scheme, group_index, test_scaled
        )
        submission_path = output_dir / "submission.csv"
        sub.to_csv(submission_path, index=False)

        # ---- OOF validation artifacts in the SAME long-format contract ----
        val_ids, val_pred_vals, val_tgt_vals = [], [], []
        for i, rid in enumerate(train_recs):
            for c in range(k):
                val_ids.append(rid * scheme.base + c)
                val_pred_vals.append(float(oof_final[i, c]))
                val_tgt_vals.append(float(Ytr[i, c]))
        val_pred = pd.DataFrame({contract.id_col: val_ids, contract.prob_col: val_pred_vals})
        val_targets = pd.DataFrame({contract.id_col: val_ids, contract.prob_col: val_tgt_vals})
        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=sub,
            validation_predictions=val_pred,
            validation_targets=val_targets,
            id_col=contract.id_col,
            target_cols=[contract.prob_col],
            validation_kind="oof",
        )

        metrics = {
            "operator": "audio_multilabel",
            "task_id": task_id,
            "selected": selected,
            "oof_pooled_auc": float(best_score),
            "cnn_used": cnn_used,
            "base": scheme.base,
            "k": k,
            "n_train_recordings": len(train_recs),
            "n_test_recordings": len(test_recs),
            "contract": contract.detail,
            "prediction_artifacts": artifacts,
            "validation": {"validation_kind": "oof", "score": float(best_score), "objective": "auc", "direction": "maximize"},
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="audio_multilabel",
            recipe_id="audio_multilabel",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(sub)), "columns": list(sub.columns)},
            score=float(best_score),
            medal="none",
            final_artifact_source=f"ees_core:audio_multilabel:{selected}",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
            score_direction="maximize",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        return _fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
