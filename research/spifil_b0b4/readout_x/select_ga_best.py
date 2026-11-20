"""select_ga_best.py -- escolhe, SEM olhar o teste, os modelos do GA a avaliar no teste.

Por base (flowers, remote_sensing, ham10000, aid, coconut_trees; nunca parasitas) e por busca
(artifacts/modehb_{hybrid,lemonade,seeded,lemonade_seeded}/<ds>/state/seed42, e no remote_sensing
tambem as buscas arquivadas *_C01_final), o melhor (cand, fid) por fitness_kappa (VALIDACAO) em
cada estrato de params do config_resolved.yaml da busca, mais o melhor geral por validacao.
Limite de 12 por base: se passar, as buscas arquivadas (C01) entram so com o melhor geral delas.
Saida: ga_best_list.json.
"""
import glob, json, sys
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[3]
BASES = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
SEARCHES = ["modehb_hybrid", "modehb_lemonade", "modehb_seeded", "modehb_lemonade_seeded"]
ARCHIVED = {"remote_sensing": ["modehb_hybrid_C01_final", "modehb_lemonade_C01_final"]}
MAXN = 12


def find(d, k):
    if isinstance(d, dict):
        for kk, v in d.items():
            if kk == k:
                return v
            r = find(v, k)
            if r is not None:
                return r
    return None


def results(search, ds):
    st = REPO / "artifacts" / search / ds / "state" / "seed42"
    cfg = st / "config_resolved.yaml"
    if not cfg.is_file():
        return None, []
    c = yaml.safe_load(cfg.read_text())
    strata = find(c, "strata")
    svm_c = (c.get("eval", {}).get("svm_fitness") or {}).get("C")
    rows = []
    for rj in sorted(glob.glob(str(st / "candidates" / "c*" / "*" / "result.json"))):
        rp = Path(rj)
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        enc = rp.parent / "encoder.pt"
        if r.get("status") != "ok" or r.get("fitness_kappa") is None or not enc.is_file():
            continue
        p = int(r["params"])
        stratum = next((s for s, (lo, hi) in strata.items() if lo <= p < hi), "fora")
        fid = int(rp.parent.name.lstrip("f"))
        rows.append({"search": search, "dataset": ds, "cand": rp.parent.parent.name, "fid": fid,
                     "stratum": stratum, "params": p, "kappa_val_search": float(r["fitness_kappa"]),
                     "f1_val_search": r.get("fitness_macro_f1"), "svm_C_search": svm_c,
                     "ckpt": str(enc.relative_to(REPO))})
    return strata, rows


def best(rows):
    # maior kappa de validacao; empate -> maior fidelidade, menos params
    return max(rows, key=lambda r: (r["kappa_val_search"], r["fid"], -r["params"]))


def main():
    out = {"rule": "por base: melhor por fitness_kappa (validacao, result.json) em cada estrato de "
                   "cada busca + melhor geral por validacao; teste nunca consultado; <=12 por base",
           "bases": {}}
    for ds in BASES:
        picks, info = {}, {}
        def add(r, why):
            key = (r["search"], r["cand"], r["fid"])
            if key in picks:
                picks[key]["why"].append(why)
            else:
                picks[key] = dict(r, why=[why])
        allrows = []
        for s in SEARCHES:
            strata, rows = results(s, ds)
            if strata is None:
                continue
            info[s] = {"n_results_ok": len(rows), "strata": strata}
            allrows += rows
            for st in strata:
                sr = [r for r in rows if r["stratum"] == st]
                if sr:
                    add(best(sr), f"best_val {s}/{st}")
        arch = []
        for s in ARCHIVED.get(ds, []):
            strata, rows = results(s, ds)
            if strata is None:
                continue
            info[s] = {"n_results_ok": len(rows), "strata": strata, "archived": True}
            arch.append((s, strata, rows))
        # melhor geral: so entre as buscas atuais (svm C=100 na fitness); arquivadas -> melhor delas
        if allrows:
            add(best(allrows), "best_val_overall")
        for s, strata, rows in arch:
            if rows:
                add(best(rows), f"best_val {s}/overall (arquivada, fitness com svm_C={rows[0]['svm_C_search']})")
        for s, strata, rows in arch:
            for st in strata:
                sr = [r for r in rows if r["stratum"] == st]
                if sr and len(picks) < MAXN:
                    add(best(sr), f"best_val {s}/{st}")
        lst = sorted(picks.values(), key=lambda r: -r["kappa_val_search"])
        assert len(lst) <= MAXN, (ds, len(lst))
        for r in lst:
            r["id"] = f"{ds}_{r['search'].replace('modehb_', '')}_{r['cand']}_f{r['fid']}_{r['stratum']}"
        out["bases"][ds] = {"searches": info, "models": lst}
        print(ds, len(lst), {s: v["n_results_ok"] for s, v in info.items()})
        for r in lst:
            print(f"   {r['id']:<60} params={r['params']:>8} kval={r['kappa_val_search']:.4f} {r['why']}")
    dst = Path(__file__).with_name("ga_best_list.json")
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
