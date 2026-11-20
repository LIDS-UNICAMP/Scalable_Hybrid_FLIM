import sys, json, hashlib, time
sys.path.insert(0,"/dados/home/moliveira/Scalable_Hybrid_FLIM")
import numpy as np, torch
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
ROOT=Path("/dados/home/moliveira/Scalable_Hybrid_FLIM")
OUT=Path(__file__).parent
torch.set_num_threads(4)
from ga_search.modehb.run_search import load_config
from ga_search.progressive_data import load_data
from core.svm import fit_svm
def data(ds):
    cfg=load_config(str(ROOT/"ga_search/config/modehb/search_v3.yaml"),ds)
    d={k:v for k,v in cfg["data"].items() if k not in ("ram_cache","val_max_samples","drop_last")}
    d.update(num_workers=4, seed=cfg["seed"])
    tr,va,_,ident=load_data(d, OUT/"data_cache")
    res={}
    for n,L in (("train",tr),("val",va)):
        X=[];Y=[]
        for x,y,*_ in L: X.append(x); Y.append(torch.as_tensor(y))
        res[n]=(torch.cat(X),torch.cat(Y).numpy())
    return res, ident
def pr(Z):
    Zc=Z-Z.mean(0); s=np.linalg.svd(Zc,compute_uv=False); l=s**2
    return float(l.sum()**2/(l**2).sum()), float(l[0]/l.sum())
@torch.no_grad()
def emb(enc,head,X):
    return torch.cat([head(enc(b)).float() for b in X.split(64)]).numpy().astype(np.float64)
for ds,ck in [(a.split(":")[0],a.split(":")[1]) for a in sys.argv[1:]]:
    D,ident=data(ds)
    p=ROOT/ck; md5=hashlib.md5(open(p,"rb").read()).hexdigest()
    c=torch.load(p,map_location="cpu",weights_only=False)
    enc,head=c["encoder"].float().eval(),c["head"].float().eval()
    t=time.time()
    Ztr=emb(enc,head,D["train"][0]); Zva=emb(enc,head,D["val"][0])
    clf=fit_svm(Ztr,D["train"][1],C=100.0,scaler=False,tag="CHECK")
    pred=clf.predict(Zva); k=cohen_kappa_score(D["val"][1],pred)
    er,top1=pr(np.vstack([Ztr,Zva]))
    r=dict(dataset=ds,ckpt=ck,md5=md5,n_train=len(Ztr),n_val=len(Zva),dim=Ztr.shape[1],kappa_orig_val=k,eff_rank_trval=er,top1=top1,
           fit=getattr(clf,"fit_diagnostics_",{}),secs=time.time()-t, data_sha=ident.get("sha256") if isinstance(ident,dict) else None)
    print(json.dumps(r)); open(OUT/"orig_results.jsonl","a").write(json.dumps(r)+"\n")
