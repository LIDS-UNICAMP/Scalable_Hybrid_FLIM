import json, glob, numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
ROOT="/dados/home/moliveira/Scalable_Hybrid_FLIM"
B="/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena"
recs={}
for f,src in [(ROOT+"/ga_search/modehb/reports/test_svm_C100.jsonl","jaci"),(B+"/reports/test_svm_C100.jsonl","ath")]:
    for l in open(f):
        if not l.strip(): continue
        t=json.loads(l)
        if "test_kappa" not in t: continue
        k=(t["run"],t["dataset"],t["cand_id"],float(t["fidelity"]))
        if k not in recs or t.get("time",0)>=recs[k].get("time",0): recs[k]=dict(t,src=src)
print("unique tests",len(recs))
# fitness from search_state when available
ss={}
for p in glob.glob(ROOT+"/artifacts/*/*/state/seed42/search_state.json"):
    run="artifacts/"+p.split("/artifacts/")[1].split("/")[0]; ds=p.split("/")[-4]
    try: s=json.load(open(p))
    except Exception as e: continue
    for j in s["jobs"]:
        r=j.get("result") or {}
        ss[(run,ds,j.get("cand_id"),float(j.get("fidelity") or -1))]=(r.get("fitness_kappa"),j.get("status"))
groups=defaultdict(list); mism=0; nmatch=0
for k,t in recs.items():
    v=t.get("val_kappa")
    if k in ss and ss[k][0] is not None:
        nmatch+=1
        if v is not None and abs(ss[k][0]-v)>1e-4: mism+=1
    groups[(k[0],k[1])].append((v,t["test_kappa"],t.get("cov_kappa"),k[2],k[3]))
print("matched to search_state",nmatch,"val mismatch",mism)
for minn in (3,4):
  ro=[];rc=[];reg=[];coinc=0;neg=0;names=[]
  for g,L in sorted(groups.items()):
    L=[x for x in L if x[0] is not None and x[1] is not None]
    if len(L)<minn: continue
    v=np.array([x[0] for x in L]); o=np.array([x[1] for x in L])
    Lc=[x for x in L if x[2] is not None]
    rho_o=spearmanr(v,o)[0]
    rho_c=spearmanr([x[0] for x in Lc],[x[2] for x in Lc])[0] if len(Lc)>=3 else np.nan
    i=int(np.argmax(v)); r=o.max()-o[i]
    ro.append(rho_o); rc.append(rho_c); reg.append(r); coinc+= (r==0); neg+=(rho_o<0)
    names.append((g,len(L),round(rho_o,2),round(rho_c,2),round(r,3)))
  print(f"\n min n={minn}: groups={len(ro)} median rho ORIG={np.nanmedian(ro):.3f} IQR={np.nanpercentile(ro,25):.2f}..{np.nanpercentile(ro,75):.2f} neg={neg}; median rho COV={np.nanmedian(rc):.3f}; regret median={np.median(reg):.3f} max={max(reg):.3f} coinc={coinc}")
  for n in names: print("  ",n)

print("\n=== C100 only (exclude *_C01_*), n>=3; and dedup by cand (keep max fid) ===")
for dedup in (False,True):
  ro=[];rc=[];reg=[];neg=0
  for g,L in sorted(groups.items()):
    if "C01" in g[0]: continue
    L=[x for x in L if x[0] is not None and x[1] is not None]
    if dedup:
        d={}
        for x in L:
            if x[3] not in d or x[4]>d[x[3]][4]: d[x[3]]=x
        L=list(d.values())
    if len(L)<3: continue
    v=np.array([x[0] for x in L]); o=np.array([x[1] for x in L])
    Lc=[x for x in L if x[2] is not None]
    ro.append(spearmanr(v,o)[0]); rc.append(spearmanr([x[0] for x in Lc],[x[2] for x in Lc])[0]); reg.append(o.max()-o[int(np.argmax(v))]); neg+=ro[-1]<0
  print("dedup",dedup,"groups",len(ro),"rhoORIG med %.3f neg %d rhoCOV med %.3f regret med %.3f max %.3f"%(np.nanmedian(ro),neg,np.nanmedian(rc),np.median(reg),max(reg)))
