"""Extra: (1) cadeia de fidelidades completa dos candidatos selecionados; (2) controle random_he x spifil (f5) nas mesmas buscas."""
import glob, json, os, collections
ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"; D = f"{ROOT}/research/spifil_b0b4/wandb_curvas"
sel = json.load(open(f"{D}/models_selected.json"))
chains = {}
for m in sel:
    cdir = os.path.dirname(os.path.dirname(m["path"]))
    for f in sorted(glob.glob(f"{cdir}/f*/result.json"), key=lambda p: float(p.split('/')[-2][1:])):
        d = json.load(open(f))
        if d.get("status") == "ok" and d.get("wandb_run_id"):
            chains.setdefault(cdir, []).append(dict(fid=float(f.split('/')[-2][1:]), wandb=d["wandb_run_id"], kappa=d.get("fitness_kappa"),
                erank=(d.get("metrics") or {}).get("effective_rank"), rankme=(d.get("metrics") or {}).get("rankme"), ssl_final=d.get("ssl_final_loss")))
ctrl = collections.defaultdict(list)
for f in glob.glob(f"{ROOT}/artifacts/modehb*/remote_sensing/state/seed42/candidates/c*/f5/result.json"):
    if any(s in f for s in ["fail", "pre_aes"]): continue
    d = json.load(open(f))
    if d.get("status") != "ok" or not d.get("wandb_run_id"): continue
    o = sorted(set(L.get("prov", {}).get("origin", "?") for L in d.get("layers", [])))
    ctrl[f.split('/')[-8]].append(dict(origin="+".join(o), wandb=d["wandb_run_id"], kappa=d.get("fitness_kappa"), params=d.get("params"), path=f))
rnd_searches = {s for s, L in ctrl.items() if any(x["origin"] == "random_he" for x in L)}
control = [x for s in rnd_searches for x in ctrl[s]]
print("buscas com random_he:", rnd_searches, collections.Counter(x["origin"] for x in control))
json.dump({"chains": chains, "control": control}, open(f"{D}/models_extra.json", "w"), indent=1)
print("cadeias:", len(chains), "runs:", sum(len(v) for v in chains.values()))
