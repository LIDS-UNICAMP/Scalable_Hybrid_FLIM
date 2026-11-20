"""Baixa a historia W&B (ophira-ai/progressive_mo_dehb) dos modelos selecionados -> cache pickle."""
import json, os, pickle, sys
from concurrent.futures import ThreadPoolExecutor
import wandb
D = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(D, "cache"); os.makedirs(CACHE, exist_ok=True)
api = wandb.Api(timeout=120)
models = json.load(open(os.path.join(D, "models_selected.json")))
def get(rid):
    out = os.path.join(CACHE, f"{rid}.pkl")
    if os.path.exists(out): return rid, "cached"
    try:
        r = api.run(f"ophira-ai/progressive_mo_dehb/{rid}")
        h = r.history(samples=50000, pandas=True)
        pickle.dump({"history": h, "config": dict(r.config), "name": r.name, "state": r.state,
                     "summary": {k: v for k, v in r.summary.items() if isinstance(v, (int, float, str))}}, open(out, "wb"))
        return rid, f"ok {h.shape}"
    except Exception as e:
        return rid, f"ERR {e}"
with ThreadPoolExecutor(8) as ex:
    ids = {m["wandb"] for m in models}
    ex_ = json.load(open(os.path.join(D, "models_extra.json")))
    ids |= {r["wandb"] for v in ex_["chains"].values() for r in v} | {r["wandb"] for r in ex_["control"]}
    for rid, s in ex.map(get, sorted(ids)): print(rid, s, flush=True)
