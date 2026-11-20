import json, glob, csv, os
ROOT="/dados/home/moliveira/Scalable_Hybrid_FLIM"
BANDS=[('B0',0,300e3),('B1',300e3,800e3),('B2',800e3,1.5e6),('B3',1.5e6,3e6),('B4',3e6,5.5e6)]
reg={(r['dataset'],r['faixa']):r for r in csv.DictReader(open(ROOT+"/research/spifil_b0b4/agent1/seed_registry.csv")) if r['regua'].startswith('COV')}
best={}
fails=[]
for p in glob.glob(ROOT+"/artifacts/modehb_hybrid/*/state/seed42/search_state.json")+glob.glob(ROOT+"/artifacts/modehb_lemonade/*/state/seed42/search_state.json"):
    run=p.split("/artifacts/")[1].split("/")[0]; ds=p.split("/")[-4]
    s=json.load(open(p)); 
    for j in s["jobs"]:
        r=j.get("result") or {}
        if j.get("status")!="ok":
            if j.get("status")=="failed" and r.get("params") or "cap" in str(j.get("error","")): fails.append((run,ds,j.get("cand_id"),j.get("fidelity"),str(j.get("error",""))[:60]))
            continue
        k=r.get("fitness_kappa"); pa=r.get("params")
        if k is None or pa is None: continue
        for B,lo,hi in BANDS:
            if lo<pa<=hi:
                key=(ds,B); cand=(k,-float(j["fidelity"]),-(r.get("flops") or 0),run,j.get("cand_id"),j["fidelity"],pa)
                if key not in best or cand[:3]>best[key][:3]: best[key]=cand
print("live best vs registry (COV) where different:")
for key in sorted(best):
    b=best[key]; r=reg.get(key)
    rc=(r['cand'],r['busca'],r['kappa_val'],r['lacuna']) if r else None
    same = r and r['cand']==b[4] and r['busca'] in b[3] and r['lacuna']=='nao'
    if not same: print(key,"live:",b[3],b[4],"f%s"%b[5],round(b[0],4),b[6]," | reg:",rc)
print("\nfailed with cap/params:",[f for f in fails if 'cap' in f[4]][:20])
