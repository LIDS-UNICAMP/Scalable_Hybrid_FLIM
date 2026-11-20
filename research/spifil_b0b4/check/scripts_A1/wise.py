import json, glob, numpy as np, os
from scipy.stats import spearmanr
from collections import defaultdict
ROOT="/dados/home/moliveira/Scalable_Hybrid_FLIM"
B="/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena"
srcs=[(p,"jaci") for p in sorted(glob.glob(ROOT+"/artifacts/modehb_hybrid/*/state/seed42/search_state.json")+glob.glob(ROOT+"/artifacts/modehb_lemonade/*/state/seed42/search_state.json"))]+[(B+"/pnas_cs/search_state.json","ath")]
d=[];lin=defaultdict(dict); lin_all=defaultdict(dict); cut=None
for p,m in srcs:
    s=json.load(open(p)); 
    for j in s["jobs"]:
        r=j.get("result") or {}
        wk=r.get("wise_kappas") or {}
        key=(p,j.get("cand_id"))
        if r.get("wise_alpha") is not None and "0.0" in wk and "1.0" in wk:
            d.append((p.split("/")[-4] if m=="jaci" else "rs_pnas", wk["1.0"]-wk["0.0"], r.get("wise_alpha"), j.get("status")))
        if r.get("fitness_kappa") is not None and j.get("status")=="ok":
            lin_all[key][float(j["fidelity"])]=r["fitness_kappa"]
            if (j.get("op") or {}).get("type")=="calibration": lin[key][float(j["fidelity"])]=r["fitness_kappa"]
x=np.array([a[1] for a in d]); print("WiSE n",len(x),"median %.4f mean %.4f IQR %.3f %.3f pos %.2f"%(np.median(x),x.mean(),*np.percentile(x,[25,75]),(x>0).mean()))
al=np.array([a[2] for a in d]); print("alpha=0 frac %.3f alpha=1 frac %.3f"%((al==0).mean(),(al==1).mean()))
for name,L in (("calibration",lin),("all",lin_all)):
    a=[];b=[];g=[]
    for k,v in L.items():
        if 5.0 in v and max(v)>5.0 and max(v)==200.0:
            a.append(v[5.0]); b.append(v[200.0]); g.append(v[200.0]-v[5.0])
    print(name,"lineages reaching f200 with f5:",len(a),"rho %.3f"%spearmanr(a,b)[0],"gain med %.4f"%np.median(g),"|g|<=.01 %.2f"%np.mean(np.abs(g)<=.01))
    a=[];b=[]
    for k,v in L.items():
        if 5.0 in v and len(v)>=2:
            a.append(v[5.0]); b.append(v[max(v)])
    print(name,"lineages f5 + any final:",len(a),"rho %.3f"%spearmanr(a,b)[0])
