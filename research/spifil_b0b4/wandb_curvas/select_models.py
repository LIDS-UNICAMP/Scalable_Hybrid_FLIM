"""Seleciona os melhores modelos do GA (por validacao) nas 5 bases nao-parasitas."""
import glob, json, os, collections
ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
BASES = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
STRATA = {"tiny": (0, 3e5), "small": (3e5, 8e5), "medium": (8e5, 1.5e6), "huge": (1.5e6, 3.5e6)}
def stratum(p):
    for k, (a, b) in STRATA.items():
        if a <= p < b: return k
    return "over"
rows = []
for f in glob.glob(f"{ROOT}/artifacts/modehb*/*/state/seed42/candidates/c*/f*/result.json"):
    parts = f[len(ROOT)+1:].split("/")
    search, ds, cand, fid = parts[1], parts[2], parts[6], parts[7]
    if ds not in BASES: continue
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") != "ok" or d.get("fitness_kappa") is None: continue
    rows.append(dict(search=search, dataset=ds, cand=cand, fid=int(fid[1:]), params=d.get("params"),
                     stratum=stratum(d.get("params") or 0), kappa=d["fitness_kappa"], wandb=d.get("wandb_run_id"),
                     ae_epochs=d.get("ae_epochs_per_layer"), ssl_epochs=d.get("ssl_epochs"), path=f))
print("candidatos ok:", len(rows), collections.Counter(r["dataset"] for r in rows))
skip = ("fail", "pre_aesched", "smoke")
sel = {}
for r in rows:
    if any(s in r["search"] for s in skip): continue
    k = (r["dataset"], r["search"], r["stratum"])
    if k not in sel or r["kappa"] > sel[k]["kappa"]: sel[k] = dict(r, why=f"best_val {r['search']}/{r['stratum']}")
# melhores gerais (top-3 por base, qualquer busca)
for ds in BASES:
    for r in sorted([r for r in rows if r["dataset"] == ds and not any(s in r["search"] for s in skip)], key=lambda r: -r["kappa"])[:3]:
        k = (ds, r["search"], r["stratum"])
        if sel.get(k, {}).get("path") != r["path"]: sel[(ds, r["search"], r["stratum"], r["cand"])] = dict(r, why="top3_overall")
# modelos da tabela ga_best_tabela (ga_best_list.json)
gl = json.load(open(f"{ROOT}/research/spifil_b0b4/readout_x/ga_best_list.json"))
paths = {s["path"] for s in sel.values()}
for ds, b in gl["bases"].items():
    for m in b["models"]:
        p = f"{ROOT}/{os.path.dirname(m['ckpt'])}/result.json"
        if p in paths:
            for s in sel.values():
                if s["path"] == p: s["table_id"] = m["id"]
            continue
        r = next((r for r in rows if r["path"] == p), None)
        if r: sel[("tab", m["id"])] = dict(r, why="ga_best_tabela", table_id=m["id"])
out = sorted(sel.values(), key=lambda r: (BASES.index(r["dataset"]), -r["kappa"]))
for r in out: r.setdefault("table_id", None)
json.dump(out, open(f"{ROOT}/research/spifil_b0b4/wandb_curvas/models_selected.json", "w"), indent=1)
print(len(out), "selecionados;", sum(1 for r in out if not r["wandb"]), "sem wandb id")
for r in out: print(r["dataset"], r["search"], r["stratum"], r["cand"], r["fid"], round(r["kappa"], 3), r["wandb"], r["why"])
