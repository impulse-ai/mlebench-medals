import os, io, json, time, tarfile, subprocess, sys
t0 = time.time()
print("HISTO_T4_JOB_START", flush=True)

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)
try:
    import torchvision  # noqa
except Exception:
    print("installing torchvision...", flush=True)
    pip("--extra-index-url", "https://download.pytorch.org/whl/cu121", "torchvision==0.20.1")
try:
    import timm  # noqa
except Exception:
    print("installing timm...", flush=True)
    pip("timm==1.0.15")
print("deps ready", round(time.time() - t0, 1), "s", flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

B = "engg-ai-experimental-gpu-artifacts"
from google.cloud import storage
gcs = storage.Client()
bkt = gcs.bucket(B)

# --- fresh, unique output prefix. This SA can CREATE objects but NOT delete/
# overwrite them (no storage.objects.delete), so every artifact must be a brand
# new object name. A unique per-run prefix guarantees creates, never overwrites.
RUN_ID = str(int(t0))
OUT = "histo/out/run_" + RUN_ID + "/"
print("OUTPUT_PREFIX gs://engg-ai-experimental-gpu-artifacts/" + OUT, flush=True)
import traceback as _tb
def _upload_text(name, text):
    try:
        p = "/tmp/" + name.replace("/", "_")
        with open(p, "w") as f: f.write(text)
        bkt.blob(OUT + name).upload_from_filename(p)
    except Exception as e:
        print("TELEMETRY_UPLOAD_FAILED", name, e, flush=True)
def _excepthook(et, ev, tb):
    print("UNCAUGHT_EXCEPTION", flush=True)
    _upload_text("error.txt", "".join(_tb.format_exception(et, ev, tb)))
    sys.__excepthook__(et, ev, tb)
sys.excepthook = _excepthook

DATA_ROOT = "/tmp/histo"
os.makedirs(DATA_ROOT, exist_ok=True)
print("streaming+extracting data tar (no local tar copy)...", flush=True)
# stream the composite blob straight into a streaming tar reader -> peak disk = extracted only
blob = bkt.blob("histo/histo_prepared.tar")
with blob.open("rb") as stream:
    with tarfile.open(fileobj=stream, mode="r|") as t:
        t.extractall(DATA_ROOT)
print("data extracted", round(time.time() - t0, 1), "s", flush=True)

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")
tr = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
ss = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
print("train", tr.shape, "test(sample_sub)", ss.shape, flush=True)

# ---- config ----
IMG = 96          # native patch size; label is center 32x32 -> DO NOT random-crop out the center
BACKBONE = "tf_efficientnet_b0_ns"
EPOCHS = 8
BS = 256
LR = 4e-4
WD = 1e-4
SEED = 42
# cheap-gpu profile hard-kills at 1800s from JOB start (container pull happens before t0).
# Keep the in-code ceiling well under that so inference+upload always finish before the kill.
HARD_CEILING_S = 1300      # ~21.7 min from t0
UPLOAD_MARGIN_S = 90
N_TEST = len(ss)
TTA_VIEWS = 3              # orig + hflip + vflip
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

def load_img(p):
    return Image.open(p).convert("RGB")

# center-preserving augmentations only (flips/rotation keep the informative center intact)
train_tf = T.Compose([
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(20)], p=0.5),
    T.ColorJitter(0.08, 0.08, 0.08),
    T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class HistoDS(Dataset):
    def __init__(self, df, d, tf, has_label=True):
        self.df = df.reset_index(drop=True); self.d = d; self.tf = tf; self.has = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(load_img(os.path.join(self.d, r["id"] + ".tif")))
        return (x, float(r["label"])) if self.has else (x, r["id"])

tr_df, val_df = train_test_split(tr, test_size=0.06, stratify=tr["label"], random_state=SEED)
print("train/val", tr_df.shape, val_df.shape,
      "val_pos_frac", round(val_df["label"].mean(), 4), flush=True)
dl_tr = DataLoader(HistoDS(tr_df, train_dir, train_tf), batch_size=BS, shuffle=True,
                   num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
dl_val = DataLoader(HistoDS(val_df, train_dir, eval_tf), batch_size=512, shuffle=False,
                    num_workers=4, pin_memory=True)

model = timm.create_model(BACKBONE, pretrained=True, num_classes=1).to(device)
model = model.to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

def eval_auc(dl):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast():
                logit = model(x).squeeze(1)
            ps.append(torch.sigmoid(logit).float().cpu().numpy()); ys.append(np.array(y))
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

# measure val throughput once to size the inference tail-reserve for the big 45k test set
best_auc, best_epoch, best_state, epochs_run = -1.0, -1, None, 0
tail_reserve = 600.0  # conservative default until measured
for ep in range(EPOCHS):
    # dynamic budget guard: don't start an epoch if we can't also finish inference+upload
    if ep > 0 and (time.time() - t0) + tail_reserve + UPLOAD_MARGIN_S > HARD_CEILING_S:
        print("BUDGET_GUARD stop before epoch", ep, "elapsed",
              round(time.time()-t0,1), "tail_reserve", round(tail_reserve,1), flush=True); break
    model.train(); tl = 0.0; nb = 0
    for x, y in dl_tr:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = crit(model(x).squeeze(1), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item(); nb += 1
    sched.step()
    t_eval = time.time()
    auc = eval_auc(dl_val); epochs_run = ep + 1
    val_secs = time.time() - t_eval
    # per-image single-view eval rate -> reserve for N_TEST * TTA_VIEWS * 1.5 safety + upload
    rate = max(len(val_df) / max(val_secs, 1e-3), 1.0)
    tail_reserve = (N_TEST * TTA_VIEWS / rate) * 1.5 + UPLOAD_MARGIN_S
    print(f"EPOCH {ep} loss {tl/max(nb,1):.4f} val_auc {auc:.4f} "
          f"eval_rate {rate:.0f}img/s tail_reserve {tail_reserve:.0f}s "
          f"elapsed {round(time.time()-t0,1)}s", flush=True)
    _upload_text("progress_ep%d.json" % ep, json.dumps({
        "phase": "train", "epoch": int(ep), "val_auc": float(auc),
        "best_auc": float(max(best_auc, auc)), "eval_rate": float(rate),
        "tail_reserve_s": float(tail_reserve), "elapsed_s": round(time.time()-t0, 1),
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}))
    if auc > best_auc:
        best_auc, best_epoch = auc, ep
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("BEST_AUC", round(best_auc, 4), "at epoch", best_epoch,
      "epochs_run", epochs_run, flush=True)
if best_state is not None:
    model.load_state_dict(best_state)

print("TEST_INFERENCE_START elapsed", round(time.time()-t0,1), "s reserve", round(tail_reserve,1), flush=True)
_upload_text("progress_inference.json", json.dumps({"phase": "inference_start",
             "best_auc": float(best_auc), "elapsed_s": round(time.time()-t0, 1)}))
te_df = ss[["id"]].copy()
dl_te = DataLoader(HistoDS(te_df, test_dir, eval_tf, has_label=False), batch_size=256,
                   shuffle=False, num_workers=4, pin_memory=True)
model.eval(); ids = []; probs = []
ti = time.time()
with torch.no_grad():
    for x, idc in dl_te:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast():
            p = torch.sigmoid(model(x).squeeze(1))
            p = p + torch.sigmoid(model(torch.flip(x, dims=[3])).squeeze(1))
            p = p + torch.sigmoid(model(torch.flip(x, dims=[2])).squeeze(1))
        probs.append((p / TTA_VIEWS).float().cpu().numpy()); ids.extend(list(idc))
P = np.concatenate(probs)
print("TEST_INFERENCE_DONE", round(time.time()-ti,1), "s for", len(P), "imgs "
      f"({len(P)/max(time.time()-ti,1e-3):.0f} img/s incl {TTA_VIEWS}-view TTA)", flush=True)

sub = pd.DataFrame({"id": ids, "label": P.astype(float)})
# align to sample_submission order/ids exactly
sub = ss[["id"]].merge(sub, on="id", how="left")
assert sub["label"].isna().sum() == 0, "missing predictions"
assert len(sub) == N_TEST, (len(sub), N_TEST)
sub.to_csv("/tmp/submission.csv", index=False)
bkt.blob(OUT + "submission.csv").upload_from_filename("/tmp/submission.csv")

metrics = {"holdout_auc": float(best_auc), "best_epoch": int(best_epoch),
           "epochs_run": int(epochs_run), "epochs_planned": EPOCHS, "backbone": BACKBONE,
           "img_size": IMG, "batch_size": BS, "lr": LR, "gpu": "T4",
           "n_train": int(len(tr_df)), "n_val": int(len(val_df)), "n_test": int(len(sub)),
           "pred_mean": float(P.mean()), "pred_std": float(P.std()),
           "tta": "orig+hflip+vflip", "tail_reserve_s": round(tail_reserve, 1),
           "wall_seconds": round(time.time() - t0, 1),
           "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}
with open("/tmp/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
bkt.blob(OUT + "metrics.json").upload_from_filename("/tmp/metrics.json")
print("HISTO_T4_JOB_DONE", json.dumps(metrics), flush=True)
