import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.io import wavfile
from scipy import signal
import torchvision as tv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(48)

BASE=os.path.expanduser("~/.cache/mle-bench/data/mlsp-2013-birds/prepared/public")
ESS=os.path.join(BASE,"essential_data"); NSPEC=19
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
Ytr=np.zeros((len(train_recs),NSPEC),dtype=np.float32)
for i,r in enumerate(train_recs):
    for s in Y.get(r,set()): Ytr[i,s]=1

# ---- log-mel with per-freq bg subtraction ----
SR=16000; NFFT=1024; HOP=256; NMEL=128
def mel_fb(sr,nfft,nmel,fmin=200,fmax=8000):
    def hz2mel(h): return 2595*np.log10(1+h/700)
    def mel2hz(m): return 700*(10**(m/2595)-1)
    mels=np.linspace(hz2mel(fmin),hz2mel(fmax),nmel+2); hz=mel2hz(mels)
    bins=np.floor((nfft+1)*hz/sr).astype(int); fb=np.zeros((nmel,nfft//2+1))
    for m in range(1,nmel+1):
        l,c,r=bins[m-1],bins[m],bins[m+1]
        for k in range(l,c):
            if c>l: fb[m-1,k]=(k-l)/(c-l)
        for k in range(c,r):
            if r>c: fb[m-1,k]=(r-k)/(r-c)
    return fb
FB=mel_fb(SR,NFFT,NMEL)
def logmel(rid):
    sr,d=wavfile.read(os.path.join(ESS,"src_wavs",rec2name[rid]+".wav")); d=d.astype(np.float32)
    if d.ndim>1: d=d.mean(1)
    d=d/(np.abs(d).max()+1e-9)
    f,t,Sxx=signal.spectrogram(d,fs=sr,nperseg=NFFT,noverlap=NFFT-HOP,mode="magnitude")
    M=FB@(Sxx**2); L=np.log1p(M*1000)
    bg=np.median(L,axis=1,keepdims=True); L=np.clip(L-bg,0,None)
    L=(L-L.mean())/(L.std()+1e-6)
    return L.astype(np.float32)  # (NMEL, T)
print("computing mel...",flush=True)
MEL={r:logmel(r) for r in all_recs}
W=int(np.median([MEL[r].shape[1] for r in all_recs]))
print("mel width median",W,flush=True)

mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def prep(L, train=False):
    T=L.shape[1]
    if train:
        # random crop width W (pad if short)
        if T>W: s=np.random.randint(0,T-W); L=L[:,s:s+W]
        elif T<W: L=np.pad(L,((0,0),(0,W-T)),mode="wrap")
        # specaugment
        Lc=L.copy()
        for _ in range(2):
            fw=np.random.randint(0,20); f0=np.random.randint(0,max(1,NMEL-fw)); Lc[f0:f0+fw,:]=0
        for _ in range(2):
            tw=np.random.randint(0,40); t0=np.random.randint(0,max(1,W-tw)); Lc[:,t0:t0+tw]=0
        L=Lc
    else:
        if T>W: s=(T-W)//2; L=L[:,s:s+W]
        elif T<W: L=np.pad(L,((0,0),(0,W-T)),mode="wrap")
    t=torch.from_numpy(L).float().unsqueeze(0).unsqueeze(0)
    t=F.interpolate(t,size=(224,224),mode="bilinear",align_corners=False)
    t=t.repeat(1,3,1,1)
    # scale to ~[0,1] then imagenet norm
    t=(t-t.min())/(t.max()-t.min()+1e-6)
    t=(t-mean)/std
    return t.squeeze(0)

def make_model():
    m=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc=nn.Linear(512,NSPEC); return m

pos=Ytr.sum(0); neg=len(train_recs)-pos; pw=torch.tensor(np.clip(neg/np.maximum(pos,1),1,20)).float()
EPOCHS=30; BS=32
kf=KFold(5,shuffle=True,random_state=42)
oof=np.zeros((len(train_recs),NSPEC),dtype=np.float32)
test_pred=np.zeros((len(test_recs),NSPEC),dtype=np.float32)
Xtest=[MEL[r] for r in test_recs]

for fi,(tr,va) in enumerate(kf.split(train_recs)):
    model=make_model(); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    trrec=[train_recs[i] for i in tr]; trY=Ytr[tr]
    best_va=0; best_state=None
    for ep in range(EPOCHS):
        model.train(); order=np.random.permutation(len(tr))
        for b in range(0,len(tr),BS):
            idx=order[b:b+BS]
            xb=torch.stack([prep(MEL[trrec[j]],train=True) for j in idx])
            yb=torch.from_numpy(trY[idx]).float()
            # mixup
            if np.random.rand()<0.5:
                lam=np.random.beta(0.4,0.4); perm=torch.randperm(xb.size(0))
                xb=lam*xb+(1-lam)*xb[perm]; yb=lam*yb+(1-lam)*yb[perm]
            opt.zero_grad(); out=model(xb); loss=lossf(out,yb); loss.backward(); opt.step()
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            vp=np.zeros((len(va),NSPEC))
            for k,vi in enumerate(va):
                xb=prep(MEL[train_recs[vi]],train=False).unsqueeze(0)
                vp[k]=torch.sigmoid(model(xb)).numpy()[0]
            vauc=roc_auc_score(Ytr[va].ravel(),vp.ravel())
        if vauc>best_va: best_va=vauc; best_state={k:v.clone() for k,v in model.state_dict().items()}
    print(f"fold{fi} best_va={best_va:.4f}",flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        for k,vi in enumerate(va):
            # TTA: center + 2 random crops
            ps=[]
            for tta in range(3):
                xb=prep(MEL[train_recs[vi]],train=(tta>0)).unsqueeze(0)
                ps.append(torch.sigmoid(model(xb)).numpy()[0])
            oof[vi]=np.mean(ps,0)
        for k,r in enumerate(test_recs):
            ps=[]
            for tta in range(3):
                xb=prep(MEL[r],train=(tta>0)).unsqueeze(0)
                ps.append(torch.sigmoid(model(xb)).numpy()[0])
            test_pred[k]+=np.mean(ps,0)/5

cnn_oof_auc=roc_auc_score(Ytr.ravel(),oof.ravel())
print("CNN OOF pooled AUC=%.5f"%cnn_oof_auc,flush=True)
np.save(os.path.expanduser("~/birds_cnn.npy"),{"oof":oof,"test":test_pred,"train_recs":train_recs,"test_recs":test_recs,"auc":cnn_oof_auc})
print("SAVED cnn preds",flush=True)
