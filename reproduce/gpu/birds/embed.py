import os, numpy as np, pandas as pd, torch
from scipy.io import wavfile
from scipy import signal
import torchvision as tv
import warnings; warnings.filterwarnings("ignore")
torch.manual_seed(0)

BASE = os.path.expanduser("~/.cache/mle-bench/data/mlsp-2013-birds/prepared/public")
ESS = os.path.join(BASE, "essential_data")
r2f = pd.read_csv(os.path.join(ESS, "rec_id2filename.txt"))
rec2name = dict(zip(r2f.rec_id, r2f.filename))
all_recs = sorted(rec2name.keys())

torch.set_num_threads(32)
# backbone: resnet50 frozen, drop fc
w = tv.models.ResNet50_Weights.IMAGENET1K_V2
net = tv.models.resnet50(weights=w)
net.fc = torch.nn.Identity()
net.eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def logspec_img(d, sr):
    d = d.astype(np.float32)
    if d.ndim>1: d=d.mean(1)
    d = d/(np.abs(d).max()+1e-9)
    f,t,Sxx = signal.spectrogram(d, fs=sr, nperseg=512, noverlap=384)  # ~85% overlap
    S = np.log1p(Sxx)
    bg = np.median(S, axis=1, keepdims=True)
    D = np.clip(S-bg, 0, None)
    # normalize per-image to [0,1]
    D = (D - D.min())/(D.max()-D.min()+1e-9)
    return D  # (freq, time)

def to_tensor(img):
    # img (H,W) -> (1,3,224,224) normalized
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    t = torch.nn.functional.interpolate(t, size=(224,224), mode="bilinear", align_corners=False)
    t = t.repeat(1,3,1,1)
    t = (t-mean)/std
    return t

emb = {}
with torch.no_grad():
    batch=[]; ids=[]
    def flush():
        global batch, ids
        if not batch: return
        x = torch.cat(batch,0)
        out = net(x).cpu().numpy()
        for i,r in enumerate(ids): emb[r]=out[i]
        batch=[]; ids=[]
    for k,r in enumerate(all_recs):
        sr,d = wavfile.read(os.path.join(ESS,"src_wavs",rec2name[r]+".wav"))
        img = logspec_img(d, sr)
        batch.append(to_tensor(img)); ids.append(r)
        if len(batch)>=16: flush()
        if (k+1)%50==0: print("embedded",k+1,flush=True)
    flush()
E = np.array([emb[r] for r in all_recs])
print("emb shape", E.shape, flush=True)
np.save(os.path.expanduser("~/birds_emb.npy"), {"recs":all_recs, "emb":emb})
print("SAVED embeddings", flush=True)
