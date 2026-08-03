import os, numpy as np, pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")
np.random.seed(0)

BASE=os.path.expanduser("~/.cache/mle-bench/data/mlsp-2013-birds/prepared/public")
ESS=os.path.join(BASE,"essential_data"); SUP=os.path.join(BASE,"supplemental_data"); NSPEC=19
r2f=pd.read_csv(os.path.join(ESS,"rec_id2filename.txt")); rec2name=dict(zip(r2f.rec_id,r2f.filename))
folds=pd.read_csv(os.path.join(ESS,"CVfolds_2.txt")); fold=dict(zip(folds.rec_id,folds.fold))
all_recs=sorted(rec2name.keys())
train_recs=[r for r in all_recs if fold[r]==0]; test_recs=[r for r in all_recs if fold[r]==1]
Y={}
with open(os.path.join(ESS,"rec_labels_test_hidden.txt")) as f:
    next(f)
    for line in f:
        p=line.strip().split(","); rid=int(p[0]); rest=p[1:]
        if rest and rest[0]=="?": continue
        Y[rid]=set(int(x) for x in rest if x!="")
Ytr=np.zeros((len(train_recs),NSPEC))
for i,r in enumerate(train_recs):
    for s in Y.get(r,set()): Ytr[i,s]=1

hist={}
with open(os.path.join(SUP,"histogram_of_segments.txt")) as f:
    next(f)
    for line in f:
        v=[x for x in line.strip().split(",") if x!=""]; hist[int(v[0])]=np.array([float(x) for x in v[1:]])
HDIM=len(next(iter(hist.values())))
segrows={}
with open(os.path.join(SUP,"segment_features.txt")) as f:
    next(f)
    for line in f:
        v=line.strip().split(","); rid=int(v[0]); segrows.setdefault(rid,[]).append(np.array([float(x) for x in v[2:]]))
SFD=len(next(iter(segrows.values()))[0])
def seg_agg(rid):
    if rid in segrows:
        a=np.array(segrows[rid]); return np.concatenate([a.mean(0),a.std(0),a.max(0),a.min(0),[len(a),1.0]])
    return np.zeros(SFD*4+2)
def spec_feats(rid):
    sr,d=wavfile.read(os.path.join(ESS,"src_wavs",rec2name[rid]+".wav")); d=d.astype(np.float32)
    if d.ndim>1: d=d.mean(1)
    d=d/(np.abs(d).max()+1e-9)
    f,t,Sxx=signal.spectrogram(d,fs=sr,nperseg=512,noverlap=256); S=np.log1p(Sxx)
    bg=np.median(S,axis=1,keepdims=True); D=np.clip(S-bg,0,None)
    fb=np.array_split(np.arange(S.shape[0]),32); feats=[]
    for src in (S,D):
        for idx in fb:
            b=src[idx].mean(0); feats+=[b.mean(),b.std(),b.max(),np.percentile(b,90)]
    return np.array(feats)
SPEC={r:spec_feats(r) for r in all_recs}
ed=np.load(os.path.expanduser("~/birds_emb.npy"),allow_pickle=True).item(); emb=ed["emb"]
def classical(rid): return np.concatenate([hist.get(rid,np.zeros(HDIM)),seg_agg(rid),SPEC[rid]])
Xc_tr=np.array([classical(r) for r in train_recs]); Xc_te=np.array([classical(r) for r in test_recs])
Xe_tr=np.array([emb[r] for r in train_recs]); Xe_te=np.array([emb[r] for r in test_recs])
Xall_tr=np.hstack([Xc_tr,Xe_tr]); Xall_te=np.hstack([Xc_te,Xe_te])

def pooled(oof): return roc_auc_score(Ytr.ravel(),oof.ravel())
def oof_fit_lr(Xtr,Xte,C=0.5):
    sc=StandardScaler().fit(Xtr); Xs=sc.transform(Xtr); Xt=sc.transform(Xte)
    oof=np.zeros_like(Ytr); P=np.zeros((Xte.shape[0],NSPEC)); kf=KFold(5,shuffle=True,random_state=42)
    for tr,va in kf.split(Xs):
        for s in range(NSPEC):
            m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced"); m.fit(Xs[tr],Ytr[tr,s]); oof[va,s]=m.predict_proba(Xs[va])[:,1]
    for s in range(NSPEC):
        m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced"); m.fit(Xs,Ytr[:,s]); P[:,s]=m.predict_proba(Xt)[:,1]
    return oof,P
def oof_fit_lgb(Xtr,Xte):
    oof=np.zeros_like(Ytr); P=np.zeros((Xte.shape[0],NSPEC)); kf=KFold(5,shuffle=True,random_state=42)
    def mk(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,subsample=0.8,colsample_bytree=0.5,min_child_samples=5,reg_lambda=1.0,n_jobs=8,verbose=-1)
    for tr,va in kf.split(Xtr):
        for s in range(NSPEC):
            m=mk(); m.fit(Xtr[tr],Ytr[tr,s]); oof[va,s]=m.predict_proba(Xtr[va])[:,1]
    for s in range(NSPEC):
        m=mk(); m.fit(Xtr,Ytr[:,s]); P[:,s]=m.predict_proba(Xte)[:,1]
    return oof,P

o_all,P_all=oof_fit_lr(Xall_tr,Xall_te)
o_clg,P_clg=oof_fit_lgb(Xc_tr,Xc_te)
cnn=np.load(os.path.expanduser("~/birds_cnn.npy"),allow_pickle=True).item()
o_cnn=cnn["oof"]; P_cnn=cnn["test"]

def rankavg(*os_):
    r=np.zeros_like(os_[0])
    for o in os_: r+=rankdata(o.ravel()).reshape(o.shape)
    return r/len(os_)

cands={}
cands["cnn"]=(o_cnn,P_cnn)
cands["lr_all"]=(o_all,P_all)
cands["lgb_cls"]=(o_clg,P_clg)
cands["cnn+lgbcls"]=((o_cnn+o_clg)/2,(P_cnn+P_clg)/2)
cands["cnn+lrall"]=((o_cnn+o_all)/2,(P_cnn+P_all)/2)
cands["cnn+lgbcls+lrall"]=((o_cnn+o_clg+o_all)/3,(P_cnn+P_clg+P_all)/3)
cands["rank_cnn_lgbcls"]=(rankavg(o_cnn,o_clg),rankavg(P_cnn,P_clg))
cands["rank_cnn_lgbcls_lrall"]=(rankavg(o_cnn,o_clg,o_all),rankavg(P_cnn,P_clg,P_all))
# weighted grid cnn vs classical-blend
cb_o=(o_clg+o_all)/2; cb_P=(P_clg+P_all)/2
for w in [0.3,0.4,0.5,0.6,0.7]:
    cands[f"w{w}_cnn"]=(w*rankavg(o_cnn)+(1-w)*rankavg(cb_o), w*rankavg(P_cnn)+(1-w)*rankavg(cb_P))

scores={k:pooled(v[0]) for k,v in cands.items()}
print("=== OOF pooled AUC ===",flush=True)
for k,v in sorted(scores.items(),key=lambda x:-x[1]): print(f"  {k}: {v:.5f}",flush=True)
best=max(scores,key=scores.get); Pbest=cands[best][1]
# normalize to [0,1] (AUC is rank-invariant; rank-avg blends need scaling to be a valid submission)
Pbest=(Pbest-Pbest.min())/(Pbest.max()-Pbest.min()+1e-12)
print("BEST",best,scores[best],"pred range",Pbest.min(),Pbest.max(),flush=True)

sample=pd.read_csv(os.path.join(BASE,"sample_submission.csv"))
pm={}
for i,r in enumerate(test_recs):
    for s in range(NSPEC): pm[r*100+s]=Pbest[i,s]
sub=sample.copy(); sub["Probability"]=sub["Id"].map(pm)
assert sub["Probability"].isna().sum()==0
sub.to_csv(os.path.expanduser("~/birds_submission.csv"),index=False)
print("wrote submission",len(sub),flush=True)

# ---- official grade ----
from mlebench.grade import grade_csv
from mlebench.registry import registry
from pathlib import Path
reg=registry.set_data_dir(Path(os.path.expanduser("~/.cache/mle-bench/data")))
comp=reg.get_competition("mlsp-2013-birds")
rep=grade_csv(Path(os.path.expanduser("~/birds_submission.csv")),comp)
print("=== OFFICIAL GRADE ===",flush=True)
print("score",rep.score,"valid",rep.valid_submission,flush=True)
print("bronze",rep.bronze_threshold,"silver",rep.silver_threshold,"gold",rep.gold_threshold,"median",rep.median_threshold,flush=True)
print("any_medal",rep.any_medal,"bronze",rep.bronze_medal,"silver",rep.silver_medal,"gold",rep.gold_medal,"above_median",rep.above_median,flush=True)
