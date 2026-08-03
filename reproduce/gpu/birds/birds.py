import os, sys, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

BASE = os.path.expanduser("~/.cache/mle-bench/data/mlsp-2013-birds/prepared/public")
ESS = os.path.join(BASE, "essential_data")
SUP = os.path.join(BASE, "supplemental_data")
NSPEC = 19

# ---- ids / folds / filenames ----
r2f = pd.read_csv(os.path.join(ESS, "rec_id2filename.txt"))
rec2name = dict(zip(r2f.rec_id, r2f.filename))
folds = pd.read_csv(os.path.join(ESS, "CVfolds_2.txt"))
fold = dict(zip(folds.rec_id, folds.fold))
all_recs = sorted(rec2name.keys())
train_recs = [r for r in all_recs if fold[r] == 0]
test_recs  = [r for r in all_recs if fold[r] == 1]
print(f"train={len(train_recs)} test={len(test_recs)}", flush=True)

# ---- labels ----
Y = {}  # rec -> set of species (train only)
with open(os.path.join(ESS, "rec_labels_test_hidden.txt")) as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        rid = int(parts[0]); rest = parts[1:]
        if rest and rest[0] == "?":
            continue
        labs = set(int(x) for x in rest if x != "")
        Y[rid] = labs
Ytr = np.zeros((len(train_recs), NSPEC))
for i, r in enumerate(train_recs):
    for s in Y.get(r, set()):
        Ytr[i, s] = 1
print("label matrix", Ytr.shape, "pos per species:", Ytr.sum(0).astype(int), flush=True)

# ---- histogram_of_segments (100-dim, all recs) ----
hist = {}
with open(os.path.join(SUP, "histogram_of_segments.txt")) as f:
    next(f)
    for line in f:
        v = [x for x in line.strip().split(",") if x != ""]
        rid = int(v[0]); hist[rid] = np.array([float(x) for x in v[1:]])
HDIM = len(next(iter(hist.values())))
print("hist dim", HDIM, flush=True)

# ---- segment_features aggregated per rec ----
segrows = {}
with open(os.path.join(SUP, "segment_features.txt")) as f:
    next(f)
    for line in f:
        v = line.strip().split(",")
        rid = int(v[0]); feats = np.array([float(x) for x in v[2:]])  # drop rec_id, seg_id
        segrows.setdefault(rid, []).append(feats)
SFD = len(next(iter(segrows.values()))[0])
def seg_agg(rid):
    if rid in segrows:
        a = np.array(segrows[rid])
        return np.concatenate([a.mean(0), a.std(0), a.max(0), a.min(0), [len(a), 1.0]])
    return np.zeros(SFD*4 + 2)
SEGDIM = SFD*4 + 2

# ---- spectrogram features with per-freq background subtraction ----
def spec_feats(rid):
    fn = os.path.join(ESS, "src_wavs", rec2name[rid] + ".wav")
    sr, d = wavfile.read(fn)
    d = d.astype(np.float32)
    if d.ndim > 1: d = d.mean(1)
    d = d / (np.abs(d).max() + 1e-9)
    f, t, Sxx = signal.spectrogram(d, fs=sr, nperseg=512, noverlap=256)
    S = np.log1p(Sxx)
    # per-freq background subtraction (noise floor = per-freq median over time)
    bg = np.median(S, axis=1, keepdims=True)
    D = np.clip(S - bg, 0, None)
    nb = 32
    fbands = np.array_split(np.arange(S.shape[0]), nb)
    feats = []
    for src in (S, D):
        for idx in fbands:
            band = src[idx].mean(0)  # avg over freqs in band -> time series
            feats += [band.mean(), band.std(), band.max(), np.percentile(band, 90)]
    return np.array(feats)

print("computing spectrogram features...", flush=True)
specF = {}
for r in all_recs:
    specF[r] = spec_feats(r)
SPD = len(specF[all_recs[0]])
print("spec dim", SPD, flush=True)

# ---- assemble feature matrix ----
def feat(rid):
    h = hist.get(rid, np.zeros(HDIM))
    return np.concatenate([h, seg_agg(rid), specF[rid]])
X = {r: feat(r) for r in all_recs}
FD = len(X[all_recs[0]])
Xtr = np.array([X[r] for r in train_recs])
Xte = np.array([X[r] for r in test_recs])
print("feature dim", FD, "Xtr", Xtr.shape, flush=True)

# ================= MODELS =================
def cv_oof_perspecies(model_fn, X, Y, nfold=5):
    """per-species OOF; return pooled OOF preds matrix (n,19)."""
    n = X.shape[0]
    oof = np.zeros_like(Y)
    kf = KFold(n_splits=nfold, shuffle=True, random_state=42)
    for tr, va in kf.split(X):
        for s in range(NSPEC):
            m = model_fn()
            m.fit(X[tr], Y[tr, s])
            oof[va, s] = m.predict_proba(X[va])[:, 1]
    return oof

def pooled_auc(oof, Y):
    return roc_auc_score(Y.ravel(), oof.ravel())

# scaled X for logistic
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

results = {}

# A) logistic regression per species
oof_lr = cv_oof_perspecies(lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"), Xtr_s, Ytr)
results["logreg"] = pooled_auc(oof_lr, Ytr)

# B) lightgbm per species
def lgbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.6, min_child_samples=5,
                              reg_lambda=1.0, n_jobs=8, verbose=-1)
oof_lgb = cv_oof_perspecies(lgbm, Xtr, Ytr)
results["lgbm"] = pooled_auc(oof_lgb, Ytr)

# C) stacked single model (rows = rec x species, species one-hot)
def build_stacked(Xmat, recs):
    rows=[]; owner=[]; sp=[]
    for i,r in enumerate(recs):
        for s in range(NSPEC):
            oh = np.zeros(NSPEC); oh[s]=1
            rows.append(np.concatenate([Xmat[i], oh])); owner.append(i); sp.append(s)
    return np.array(rows), np.array(owner), np.array(sp)

def cv_oof_stacked(Xmat, Y, nfold=5):
    n=Xmat.shape[0]; oof=np.zeros((n,NSPEC))
    kf=KFold(n_splits=nfold, shuffle=True, random_state=42)
    for tr,va in kf.split(Xmat):
        Xs,ow,sp = build_stacked(Xmat[tr], list(range(len(tr))))
        ys = Y[tr][ow,sp]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
                               reg_lambda=1.0, n_jobs=32, verbose=-1)
        m.fit(Xs, ys)
        Xv,owv,spv = build_stacked(Xmat[va], list(range(len(va))))
        p = m.predict_proba(Xv)[:,1]
        for k in range(len(owv)):
            oof[va[owv[k]], spv[k]] = p[k]
    return oof
oof_stk = cv_oof_stacked(Xtr, Ytr)
results["stacked"] = pooled_auc(oof_stk, Ytr)

# blends
oof_blend = (oof_lr + oof_lgb + oof_stk)/3
results["blend_all"] = pooled_auc(oof_blend, Ytr)
oof_lr_lgb = (oof_lr+oof_lgb)/2
results["blend_lr_lgb"] = pooled_auc(oof_lr_lgb, Ytr)
oof_lr_stk = (oof_lr+oof_stk)/2
results["blend_lr_stk"] = pooled_auc(oof_lr_stk, Ytr)

print("=== OOF pooled AUC ===", flush=True)
for k,v in sorted(results.items(), key=lambda x:-x[1]):
    print(f"  {k}: {v:.5f}", flush=True)

best = max(results, key=results.get)
print("BEST:", best, results[best], flush=True)

# ================ FIT FULL + PREDICT TEST ================
def fit_pred_perspecies(model_fn, Xtr, Ytr, Xte):
    P=np.zeros((Xte.shape[0], NSPEC))
    for s in range(NSPEC):
        m=model_fn(); m.fit(Xtr, Ytr[:,s]); P[:,s]=m.predict_proba(Xte)[:,1]
    return P
def fit_pred_stacked(Xtr, Ytr, Xte):
    Xs,ow,sp=build_stacked(Xtr, list(range(Xtr.shape[0]))); ys=Ytr[ow,sp]
    m=lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                         subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
                         reg_lambda=1.0, n_jobs=32, verbose=-1)
    m.fit(Xs,ys)
    Xv,owv,spv=build_stacked(Xte, list(range(Xte.shape[0])))
    p=m.predict_proba(Xv)[:,1]; P=np.zeros((Xte.shape[0],NSPEC))
    for k in range(len(owv)): P[owv[k],spv[k]]=p[k]
    return P

P_lr = fit_pred_perspecies(lambda: LogisticRegression(C=0.5,max_iter=2000,class_weight="balanced"), Xtr_s, Ytr, Xte_s)
P_lgb = fit_pred_perspecies(lgbm, Xtr, Ytr, Xte)
P_stk = fit_pred_stacked(Xtr, Ytr, Xte)
Pmap = {"logreg":P_lr, "lgbm":P_lgb, "stacked":P_stk,
        "blend_all":(P_lr+P_lgb+P_stk)/3, "blend_lr_lgb":(P_lr+P_lgb)/2, "blend_lr_stk":(P_lr+P_stk)/2}
Pbest = Pmap[best]

# ================ SUBMISSION ================
sample = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
pred_map = {}
for i,r in enumerate(test_recs):
    for s in range(NSPEC):
        pred_map[r*100+s] = Pbest[i,s]
sub = sample.copy()
sub["Probability"] = sub["Id"].map(pred_map)
assert sub["Probability"].isna().sum()==0, "missing ids!"
out = os.path.expanduser("~/birds_submission.csv")
sub.to_csv(out, index=False)
print("wrote", out, "rows", len(sub), flush=True)
np.save(os.path.expanduser("~/birds_preds.npy"), {"P_lr":P_lr,"P_lgb":P_lgb,"P_stk":P_stk,"test_recs":test_recs})
print("DONE best_oof=%.5f model=%s" % (results[best], best), flush=True)
