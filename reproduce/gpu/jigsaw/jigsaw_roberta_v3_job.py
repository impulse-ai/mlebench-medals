import os, json, time, tarfile, subprocess, sys
t0 = time.time()
print("JIGSAW_ROBERTA_JOB_START", flush=True)

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)

import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
try:
    import transformers  # noqa
except Exception:
    print("installing transformers...", flush=True)
    pip("transformers==4.44.2")
    import transformers
print("transformers", transformers.__version__, "deps ready", round(time.time()-t0,1), "s", flush=True)

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

# ---- GCS ----
B = "engg-ai-experimental-gpu-artifacts"
from google.cloud import storage
gcs = storage.Client(); bkt = gcs.bucket(B)

DATA_ROOT = "/tmp/jigsaw"
os.makedirs(DATA_ROOT, exist_ok=True)
tar_local = "/tmp/jigsaw_prepared.tar"
print("downloading data tar...", flush=True)
bkt.blob("jigsaw/jigsaw_prepared.tar").download_to_filename(tar_local)
with tarfile.open(tar_local) as t:
    t.extractall(DATA_ROOT)
print("data extracted", round(time.time()-t0,1), "s", flush=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
tr = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
te = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
tr["comment_text"] = tr["comment_text"].fillna(" ").astype(str)
te["comment_text"] = te["comment_text"].fillna(" ").astype(str)
print("train", tr.shape, "test", te.shape, "labels", LABELS, flush=True)
print("label positive rates", tr[LABELS].mean().round(4).to_dict(), flush=True)

# ---- config ----
MODEL_NAME = "roberta-base"
MAX_LEN = 256
BS = 40
INFER_BS = 256
LR = 1e-5
EPOCHS = 3
SEED = 42
WALL_BUDGET_S = 125 * 60          # hard internal wall-clock budget (min)
INFER_RESERVE_S = 20 * 60        # reserve for 153k test inference + upload
TRAIN_DEADLINE_S = WALL_BUDGET_S - INFER_RESERVE_S
torch.manual_seed(SEED); np.random.seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def collate_test(batch):
    idx = [b[0] for b in batch]; txt = [b[1] for b in batch]
    enc = tok(txt, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return idx, enc

class ToxicModel(nn.Module):
    def __init__(self, name, n_out):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        h = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(h, n_out)
    def forward(self, **enc):
        out = self.backbone(**enc)
        # distilbert has no pooler; use CLS token (first position) of last_hidden_state
        cls = out.last_hidden_state[:, 0]
        return self.head(self.drop(cls))

# ---- split: internal holdout for AUC selection (never touches test) ----
strat = (tr[LABELS].sum(axis=1) > 0).astype(int)
tr_df, val_df = train_test_split(tr, test_size=0.05, random_state=SEED, stratify=strat)
print("train/val", tr_df.shape, val_df.shape, flush=True)

dl_tr = DataLoader(TextDS(tr_df["comment_text"].values, tr_df[LABELS].values),
                   batch_size=BS, shuffle=True, num_workers=4, pin_memory=True,
                   collate_fn=collate_train, drop_last=True)
dl_val = DataLoader(TextDS(val_df["comment_text"].values, val_df[LABELS].values),
                    batch_size=INFER_BS, shuffle=False, num_workers=4, pin_memory=True,
                    collate_fn=collate_train)

model = ToxicModel(MODEL_NAME, len(LABELS)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
from transformers import get_linear_schedule_with_warmup
total_steps = len(dl_tr) * EPOCHS
warmup_steps = int(0.06 * total_steps)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
print("SCHED total_steps", total_steps, "warmup_steps", warmup_steps, flush=True)
crit = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

def predict_probs(dl, has_label):
    model.eval(); probs = []; ys = []
    with torch.no_grad():
        for batch in dl:
            if has_label:
                enc, y = batch; ys.append(y.numpy())
            else:
                _, enc = batch
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.cuda.amp.autocast():
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
    return float(np.mean(aucs)), aucs

# ---- steps-per-epoch time estimate to enforce inference reserve ----
best_auc, best_state, best_epoch, epochs_run = -1.0, None, -1, 0
steps_total = len(dl_tr)
for ep in range(EPOCHS):
    if time.time() - t0 > TRAIN_DEADLINE_S:
        print("TRAIN_DEADLINE_HIT before epoch", ep, flush=True); break
    model.train(); tl = 0.0; nb = 0
    for enc, y in dl_tr:
        # bail mid-epoch if we hit the deadline so inference always runs
        if time.time() - t0 > TRAIN_DEADLINE_S:
            print("TRAIN_DEADLINE_HIT mid-epoch", ep, "step", nb, flush=True); break
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = crit(model(**enc), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item(); nb += 1
        if nb % 200 == 0:
            print(f"  ep{ep} step {nb}/{steps_total} loss {tl/nb:.4f} elapsed {round(time.time()-t0,1)}s", flush=True)
    vp, vy = predict_probs(dl_val, True)
    auc, aucs = mean_col_auc(vy, vp); epochs_run = ep + 1
    print(f"EPOCH {ep} loss {tl/max(nb,1):.4f} val_mean_auc {auc:.5f} per_col {[round(a,4) for a in aucs]} elapsed {round(time.time()-t0,1)}s", flush=True)
    if auc > best_auc:
        best_auc, best_epoch = auc, ep
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("BEST_VAL_MEAN_AUC", round(best_auc, 5), "at epoch", best_epoch, flush=True)
if best_state is not None:
    model.load_state_dict(best_state)

# ---- export holdout (val) probs from BEST model for honest blend selection ----
vp_best, vy_best = predict_probs(dl_val, True)
val_out = pd.DataFrame({"id": val_df["id"].values})
for j, c in enumerate(LABELS):
    val_out["p_" + c] = vp_best[:, j]
for j, c in enumerate(LABELS):
    val_out["y_" + c] = vy_best[:, j].astype(int)
val_out.to_csv("/tmp/val_probs.csv", index=False)
bkt.blob("jigsaw/out5/val_probs.csv").upload_from_filename("/tmp/val_probs.csv")
print("VAL_PROBS_UPLOADED rows", len(val_out), flush=True)

# ---- test inference on ALL 153k rows ----
print("TEST_INFERENCE_START elapsed", round(time.time()-t0,1), "s", flush=True)
dl_te = DataLoader(TextDS(te["comment_text"].values, None),
                   batch_size=INFER_BS, shuffle=False, num_workers=4, pin_memory=True,
                   collate_fn=collate_test)
ti = time.time()
Pte = predict_probs(dl_te, False)
infer_s = time.time() - ti
rate = len(te) / max(infer_s, 1e-6)
print(f"INFER_DONE {len(te)} rows in {round(infer_s,1)}s = {round(rate,1)} rows/s", flush=True)

sub = pd.DataFrame({"id": te["id"].values})
for j, c in enumerate(LABELS):
    sub[c] = Pte[:, j]
sub.to_csv("/tmp/submission.csv", index=False)
bkt.blob("jigsaw/out5/submission.csv").upload_from_filename("/tmp/submission.csv")
print("submission rows", len(sub), "uploaded", flush=True)

metrics = {
    "holdout_mean_auc": float(best_auc), "best_epoch": int(best_epoch),
    "epochs_run": int(epochs_run), "epochs_planned": EPOCHS,
    "model": MODEL_NAME, "max_len": MAX_LEN, "train_bs": BS, "infer_bs": INFER_BS,
    "lr": LR, "gpu": "T4", "n_train": int(len(tr_df)), "n_val": int(len(val_df)),
    "n_test": int(len(sub)), "infer_seconds": round(infer_s, 1),
    "infer_rows_per_s": round(rate, 1), "wall_seconds": round(time.time()-t0, 1),
    "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
}
with open("/tmp/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
bkt.blob("jigsaw/out5/metrics.json").upload_from_filename("/tmp/metrics.json")
print("JIGSAW_ROBERTA_JOB_DONE", json.dumps(metrics), flush=True)
