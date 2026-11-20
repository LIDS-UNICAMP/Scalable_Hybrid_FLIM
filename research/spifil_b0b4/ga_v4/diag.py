import json, glob, collections, sys, numpy as np
sys.path.insert(0, "/dados/home/moliveira/Scalable_Hybrid_FLIM")
from ga_search.modehb import strata as S
from ga_search.modehb.modehb import nondominated_fronts
DS = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
RUNS = ["modehb_hybrid", "modehb_lemonade", "modehb_seeded", "modehb_lemonade_seeded"]
B = [("B0",0,300e3),("B1",300e3,800e3),("B2",800e3,1.5e6),("B3",1.5e6,3e6),("B4",3e6,5.5e6)]
def band(p):
    if p is None: return "?"
    for n,lo,hi in B:
        if p<=hi: return n
    return ">5.5M"
out = {}
allrows = []
for run in RUNS:
    for ds in DS:
        f = f"/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/{run}/{ds}/state/seed42/search_state.json"
        try: s = json.load(open(f))
        except FileNotFoundError: continue
        cfg = s["config"]; st0 = cfg["stratified"]; st0 = {**st0, "strata": sorted([(k,v[0],v[1]) for k,v in st0["strata"].items()], key=lambda x: x[1]) if isinstance(st0["strata"], dict) else st0["strata"]}; sc = S.strat_config(st0)
        ncls = cfg.get("n_classes"); pools = cfg.get("spifil_pool_sizes")
        jobs = [j for j in s["jobs"] if j["role"] == "search"]
        key = f"{run}/{ds}"
        R = dict(bracket=s["bracket"], done=s["done"], gpuh=round(s["gpu_hours"],1), n_jobs=len(jobs),
                 mem_budget=sc.get("memory_budget_gb"), cap=sc["param_cap"], strata=[x[0] for x in sc["strata"]])
        # distribution by stratum x fidelity (unique evaluated, non-cached)
        tab = collections.defaultdict(collections.Counter); tabB = collections.defaultdict(collections.Counter)
        gen = collections.defaultdict(collections.Counter)
        proj = collections.defaultdict(collections.Counter)
        promo = collections.defaultdict(collections.Counter)
        fails = []
        memr = []
        for j in jobs:
            r = j["result"]; p = r.get("params") if j["status"]=="ok" else None
            st = j.get("stratum") or "fail"
            tab[str(j["fidelity"])][st] += 1
            tabB[str(j["fidelity"])][band(p if p else j.get("params_est"))] += 1
            op = j.get("op") or {}
            if op.get("type") in ("random","de","morph","seed","calibration") :
                gen[op["type"]][st] += 1
                if op.get("target_stratum"):
                    proj[op["target_stratum"]][st] += 1
            if op.get("type") == "promotion":
                promo[f'{op.get("from_fidelity")}->{j["fidelity"]}'][st] += 1
            if j["status"] != "ok":
                fails.append(dict(cand=j["cand_id"], fid=j["fidelity"], params_est=j.get("params_est"),
                                  err=str(r.get("error"))[:90]))
            if j["status"]=="ok" and r.get("peak_gpu_mem_gb") and not j.get("cached") and not r.get("cached_from"):
                try:
                    est = S.mem_estimate_gb(j["genome"], int(cfg["image_size"]), n_classes=ncls,
                                            seeds_per_image=int(sc.get("spifil_seeds_per_image",100)), pool_sizes=pools)
                    memr.append((est, r["peak_gpu_mem_gb"], r.get("params")))
                except Exception as e:
                    pass
        R["by_fid_stratum"] = {k: dict(v) for k,v in sorted(tab.items(), key=lambda x: float(x[0]))}
        R["by_fid_band_B0B4"] = {k: dict(v) for k,v in sorted(tabB.items(), key=lambda x: float(x[0]))}
        R["generated_by_op"] = {k: dict(v) for k,v in gen.items()}
        R["target_vs_measured"] = {k: dict(v) for k,v in proj.items()}
        R["promotions"] = {k: dict(v) for k,v in promo.items()}
        R["fails"] = fails
        ps = [j["result"]["params"] for j in jobs if j["status"]=="ok" and j["result"].get("params")]
        R["params_quantiles"] = [int(np.percentile(ps,q)) for q in (10,50,90,100)] if ps else []
        R["n_unique_cands_gt1.5M"] = len({j["cand_id"] for j in jobs if j["status"]=="ok" and (j["result"].get("params") or 0)>1.5e6})
        R["n_unique_cands"] = len({j["cand_id"] for j in jobs})
        if memr:
            a = np.array([(e,p) for e,p,_ in memr]); ratio = a[:,0]/np.maximum(a[:,1],1e-3)
            R["mem_est_over_peak"] = dict(n=len(a), median=round(float(np.median(ratio)),2), p10=round(float(np.percentile(ratio,10)),2),
                                          p90=round(float(np.percentile(ratio,90)),2), max_peak=round(float(a[:,1].max()),1),
                                          max_est=round(float(a[:,0].max()),1))
        # global Pareto front per fidelity: strata composition
        fr = {}
        for fid in cfg["fidelities"]:
            ok = [j for j in jobs if j["status"]=="ok" and float(j["fidelity"])==float(fid)]
            if len(ok) < 2: continue
            F = np.array([j["F"] for j in ok]); f0 = nondominated_fronts(F)[0]
            fr[str(fid)] = dict(collections.Counter(ok[i].get("stratum") for i in f0))
            # best quality per stratum
        R["global_front_strata"] = fr
        bq = collections.defaultdict(float)
        for j in jobs:
            if j["status"]=="ok": bq[(j.get("stratum"), j["fidelity"])] = max(bq[(j.get("stratum"), j["fidelity"])], j["result"].get("quality",0))
        R["best_quality_stratum_fid"] = {f"{k[0]}@{k[1]}": round(v,3) for k,v in sorted(bq.items(), key=lambda x:(str(x[0][0]),x[0][1]))}
        out[key] = R
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
for k,R in out.items():
    print("=====", k, {x:R[x] for x in ("bracket","done","gpuh","n_jobs","n_unique_cands","n_unique_cands_gt1.5M","mem_budget","cap","params_quantiles")})
    print("  fid x stratum", R["by_fid_stratum"])
    print("  fid x B0-B4  ", R["by_fid_band_B0B4"])
    print("  gen by op    ", R["generated_by_op"])
    print("  target->meas ", R["target_vs_measured"])
    print("  promotions   ", R["promotions"])
    print("  front        ", R["global_front_strata"])
    print("  mem est/peak ", R.get("mem_est_over_peak"))
    print("  fails        ", R["fails"])
