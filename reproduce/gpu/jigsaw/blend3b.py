import os, json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

H = os.path.expanduser("~")
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# ---- holdout: distilbert val + roberta val + tfidf OOF, all on same ids ----
db = pd.read_csv(os.path.join(H,"v2_val_probs.csv"))
rb = pd.read_csv(os.path.join(H,"rob3_val_probs.csv"))
tf = pd.read_csv(os.path.join(H,"tfidf_oof.csv"))
m = db.merge(rb, on="id", suffixes=("_db","_rb")).merge(tf, on="id")
Y  = m[["y_"+c+"_db" for c in LABELS]].values.astype(int)
DB = m[["p_"+c+"_db" for c in LABELS]].values.astype(float)
RB = m[["p_"+c+"_rb" for c in LABELS]].values.astype(float)
TF = m[[c for c in LABELS]].values.astype(float)
print("holdout rows", len(m))

def rank01(a):
    out=np.empty_like(a,dtype=float)
    for j in range(a.shape[1]): out[:,j]=rankdata(a[:,j])/(len(a)+1.0)
    return out

def mauc(y,p): return float(np.mean([roc_auc_score(y[:,j],p[:,j]) for j in range(y.shape[1])]))

# grid search raw-prob weights on the 3-simplex (step 0.05)
best=None
for a in np.arange(0,1.0001,0.05):
    for b in np.arange(0,1.0001-a,0.05):
        c=1.0-a-b
        if c<-1e-9: continue
        s=mauc(Y, a*DB+b*RB+c*TF)
        if best is None or s>best[0]: best=(s,round(a,3),round(b,3),round(c,3),"raw")
# also rank-space simplex
DBr,RBr,TFr=rank01(DB),rank01(RB),rank01(TF)
for a in np.arange(0,1.0001,0.05):
    for b in np.arange(0,1.0001-a,0.05):
        c=1.0-a-b
        if c<-1e-9: continue
        s=mauc(Y, a*DBr+b*RBr+c*TFr)
        if best is None or s>best[0]: best=(s,round(a,3),round(b,3),round(c,3),"rank")
sc,a,b,c,space=best
print(f"BEST_HOLDOUT {sc:.6f} w_db={a} w_rb={b} w_tf={c} space={space}")

# ---- apply to test, grade once ----
dbt=pd.read_csv(os.path.join(H,"v2_test_db.csv"))
rbt=pd.read_csv(os.path.join(H,"rob3_test_db.csv"))
tft=pd.read_csv(os.path.join(H,"tfidf_test.csv"))
t=dbt.merge(rbt,on="id",suffixes=("_db","_rb")).merge(tft,on="id")
DBt=t[[cc+"_db" for cc in LABELS]].values.astype(float)
RBt=t[[cc+"_rb" for cc in LABELS]].values.astype(float)
TFt=t[[cc for cc in LABELS]].values.astype(float)
if space=="rank":
    DBt,RBt,TFt=rank01(DBt),rank01(RBt),rank01(TFt)
Pt=a*DBt+b*RBt+c*TFt
sub=pd.DataFrame({"id":t["id"].values})
for j,cc in enumerate(LABELS): sub[cc]=Pt[:,j]
outp=os.path.join(H,"jigsaw_blend3b_submission.csv")
sub.to_csv(outp,index=False)
print("wrote",outp,"rows",len(sub))
json.dump({"holdout":sc,"w_db":a,"w_rb":b,"w_tf":c,"space":space},open(os.path.join(H,"blend3b_sel.json"),"w"),indent=2)
