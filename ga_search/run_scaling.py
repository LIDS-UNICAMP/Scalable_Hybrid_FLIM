# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Etapa 2: escalonamento composto (EfficientNet) da melhor config de cada dataset; varreduras em phi (§5)."""

import os
from pathlib import Path

import yaml

# threads antes de numpy/torch/sklearn; os workers (spawn) herdam o ambiente
_T = str(yaml.safe_load((Path(__file__).parent / "config" / "ga.yaml").read_text())["parallel"]["threads_per_worker"])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _T

import argparse
import copy
import csv
import json
import math

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import Problem
from pymoo.core.variable import Real
from pymoo.problems.static import StaticProblem
from tqdm import tqdm

from ga_search import data, derived, space
from ga_search.run_ga import _seed, evaluate_many, load_cfgs, make_pool, write_rows

COEF = ("alpha_d", "beta_w", "gamma_r", "kappa_k", "delta_gw")
DIM = {"alpha_d": "d", "beta_w": "w", "gamma_r": "r", "kappa_k": "k", "delta_gw": "gw"}
SINGLE = {"single_r": "gamma_r", "single_k": "kappa_k", "single_w": "beta_w", "single_d": "alpha_d"}

# ---------------------------------------------------------------- escala (puro)


def _up(x, n):
    """Para cima ao multiplo de n, no minimo n."""
    return max(n, math.ceil(x / n - 1e-9) * n)


def _odd(x):
    return max(1, 2 * round((x - 1) / 2) + 1)


def scale(cfg, n_eff, d=1.0, w=1.0, r=1.0, k=1.0, gw=1.0, profile=False):
    """Config ativa escalada por multiplicadores (ja elevados a phi); fatores 1 devolvem a base.

    profile=True (compound_growth): o kernel escala so o k base e reaplica kernel_growth (g_k mantido).
    """
    c = copy.deepcopy(cfg)
    L, mode, res, G = c["layers"], c["layer_mode"], c["residual"], c["growth"]
    d0, wide = len(L), w != 1 or gw != 1
    nd = max(1, round(d0 * d))
    if res:
        nd = min(nd, c["residual_d_max"])
    c["r"], c["r_factor"] = round(cfg["r"] * r), round(cfg["r_factor"] * r, 4)

    if mode == "crescimento":  # regenera a progressao (§2) com d novo, e g_w * gw
        pooled = next((l for l in L if l["pool_type"] != "none"), None)
        wg = None if res else round(G["width_growth"] * gw, 4)
        k0 = _odd(L[0]["kernel_size"] * k) if profile else L[0]["kernel_size"]
        new = []
        for l in range(1, nd + 1):
            e = l - 1
            on = pooled and {"todas": True, "primeira": l == 1, "alternadas": l % 2 == 1, "ultima": l == nd}[G["pool_schedule"]]
            N = None if res else max(n_eff, round(L[0]["out_channels"] * wg ** e))
            ks = max(1, k0 + 2 * G["kernel_growth"] * e)
            new.append({"kernel_size": ks if profile else _odd(ks * k), "out_channels": _up(N * w, n_eff) if N and wide else N,
                        "dilation": L[0]["dilation"] * G["dilation_growth"] ** e, "activation": L[0]["activation"],
                        "pool_type": pooled["pool_type"] if on else "none",
                        "pool_size": pooled["pool_size"] if on else None, "pool_stride": pooled["pool_stride"] if on else None,
                        "n_superpixels": max(1, round(c["n_superpixels"] * G["superpixel_growth"] ** e))
                        if c["spifil_in"] == "features" and l >= 2 else None})
        L = new
        if not res:
            G["width_growth"] = wg
    else:  # comum e atomico: as camadas extras repetem a ultima
        L = [dict(l) for l in L[:nd]] + [dict(L[-1]) for _ in range(nd - d0)]
        for l in L:
            l["kernel_size"] = _odd(l["kernel_size"] * k)
            if not res and wide:
                l["out_channels"] = _up(l["out_channels"] * w, n_eff)
    if c["spifil_in"] == "features":  # camada nova copiada da 1a nao herda o None da camada 0
        for l in L[1:]:
            l["n_superpixels"] = l["n_superpixels"] or c["n_superpixels"]

    if res:  # §7.7: N_1 * w; growth_l = C_l/C_(l-1) da base (as novas repetem a ultima), * gw, minimo 2
        C0 = [l["channels"] for l in cfg["layers"]]
        g = [C0[i] / C0[i - 1] for i in range(1, d0)] or [2.0]
        g = [max(2.0, x * gw) for x in g + [g[-1]] * nd]
        C = []
        for i, l in enumerate(L):
            l["out_channels"] = _up(max(cfg["layers"][0]["out_channels"] * w, n_eff), n_eff) if i == 0 \
                else _up(round(C[-1] * (g[i - 1] - 1)), n_eff)
            C.append(sum(x["out_channels"] for x in L[:i + 1]))
            l["channels"] = C[-1]
    else:
        for l in L:
            l["channels"] = l["out_channels"]
    c["layers"], c["d"] = L, nd
    return c


def flops(cfg, n_eff):
    """FLOPs de derived.derived (info so preenche o que a chave flops nao usa)."""
    return derived.derived(cfg, {"n_classes": n_eff, "n_classes_eff": n_eff, "pct": 100, "n_train_images": n_eff,
                                 "n_train_per_class": 1.0, "time_s": None, "d_max": len(cfg["layers"])})["flops"]


def match_single(base, n_eff, dim, target, f0, phi):
    """Multiplicador so de `dim` cujo FLOPs fica mais perto (em log) da razao alvo; devolve (coef, cfg)."""
    if phi == 0:
        return 1.0, base
    d0 = len(base["layers"])
    # d sem residuo: ate 8*d0 (o compound chega a 4*d0 com alpha<=2, phi<=2); alem disso dilation_growth estoura
    cands = (x / d0 for x in range(1, base["residual_d_max"] + 1 if base["residual"] else 8 * d0 + 1)) if dim == "d" \
        else np.exp(np.linspace(-3, 3, 1201))
    best, goal = (math.inf, 1.0), math.log(f0 * target)
    for m in cands:  # FLOPs crescem com o multiplicador: para assim que passa do alvo
        e = math.log(flops(scale(base, n_eff, **{dim: m}), n_eff)) - goal
        best = min(best, (abs(e), m))
        if e > 0:
            break
    m = best[1]
    return float(m) ** (1 / phi), scale(base, n_eff, **{dim: m})


def scaled(base, n_eff, coefs, phi, profile=False):
    return scale(base, n_eff, profile=profile, **{DIM[c]: coefs.get(c, 1.0) ** phi for c in COEF})

# ---------------------------------------------------------------- avaliacao


def _vctx(ds_cfg, pct, split, cfg, ga_cfg):
    return {"n_available": data.n_available(ds_cfg, pct, split, paper_classes=cfg["paper_classes"]),
            "n_classes_eff": data.n_classes_eff(ds_cfg, cfg["paper_classes"]), "max_seeds": ga_cfg["spifil"]["max_seeds"]}


def _mean_obj(rows, penalty):
    return float(np.mean([float(r["objective"]) if r["status"] == "ok" else penalty for r in rows]))


def search_coefs(pool, base, n_eff, names, kws, ga_cfg, sc, f0, colours, ds_cfg, pct):
    """GA nos coeficientes em phi=1: objetivo medio nos splits, restricao de FLOPs (violou -> penalidade sem rodar)."""
    vars_ = {n: Real(bounds=tuple(sc["coef_bounds"][n])) for n in names}
    prob = Problem(vars=vars_, n_obj=1, n_ieq_constr=1)
    algo = MixedVariableGA(pop_size=sc["ga"]["pop_size"])
    algo.setup(prob, termination=("n_gen", sc["ga"]["n_gen_max"]), seed=kws[0]["seed"], verbose=False)
    tol, best, post = math.log(1 + sc["flops_tol"]), (math.inf, None, None), {}
    bar = tqdm(total=sc["ga"]["n_gen_max"], desc=f"coef {'+'.join(names)}", colour=colours["generations"],
               position=1, leave=False, dynamic_ncols=True)
    while algo.has_next():
        pop = algo.ask()
        X = [{n: float(x[n]) for n in names} for x in pop.get("X")]
        cfgs = [scaled(base, n_eff, x, 1) for x in X]
        G = [abs(math.log(flops(c, n_eff) / f0) - math.log(sc["flops_target"])) - tol for c in cfgs]
        run = [i for i, (c, g) in enumerate(zip(cfgs, G)) if g <= 0
               and all(space.is_valid(c, _vctx(ds_cfg, pct, kw["split"], c, ga_cfg))[0] for kw in kws)]
        rows = evaluate_many(pool, [("config", cfgs[i], kw) for i in run for kw in kws],
                             desc="avaliacoes coef", colour=colours["evals"], position=2)
        F = [ga_cfg["penalty"]] * len(X)
        for j, i in enumerate(run):
            rs = rows[j * len(kws):(j + 1) * len(kws)]
            F[i] = _mean_obj(rs, ga_cfg["penalty"])
            if F[i] < best[0]:
                best = (F[i], X[i], flops(cfgs[i], n_eff) / f0)
                post = {m: f"{np.mean([float(r[k]) for r in rs]):.4g}" for m, k in
                        (("CH", "ch"), ("DBI", "dbi"), ("Silhouette", "silhouette"))}
        Evaluator().eval(StaticProblem(prob, F=np.array(F)[:, None], G=np.array(G)[:, None]), pop)
        algo.tell(infills=pop)
        bar.set_postfix(**post, refresh=False)
        bar.update()
    bar.close()
    return best


def pick_base(pool, ga_cfg, ds_cfg, sp, dataset, pct, colours, log):
    """best_config.yaml de cada split; vence a de melhor objetivo medio nos splits com run (consenso)."""
    root = Path(ga_cfg["results_dir"]) / dataset / f"pct_{pct}"
    found = {s: yaml.safe_load((root / f"split_{s}" / "best_config.yaml").read_text())["cfg"]
             for s in ds_cfg["splits"] if (root / f"split_{s}" / "best_config.yaml").exists()}
    if not found:
        return None, None, None
    if len(found) < len(ds_cfg["splits"]):
        log.append(f"faltam splits {sorted(set(ds_cfg['splits']) - set(found))} em pct {pct}; usando {sorted(found)}")
    kws = [{"dataset": dataset, "pct": pct, "split": s, "ds_cfg": ds_cfg, "space": sp, "ga_cfg": ga_cfg,
            "seed": _seed(ga_cfg, dataset, pct, s)} for s in found]
    cands = list(found.values())
    rows = evaluate_many(pool, [("config", c, kw) for c in cands for kw in kws],
                         desc="base", colour=colours["evals"], position=1)
    score = [_mean_obj(rows[i * len(kws):(i + 1) * len(kws)], ga_cfg["penalty"]) for i in range(len(cands))]
    i = int(np.argmin(score))
    log.append(f"base: split {list(found)[i]} (objetivo medio {score[i]:.6g}; candidatos {dict(zip(found, score))})")
    return cands[i], kws, list(found)

# ---------------------------------------------------------------- CLI


def run_dataset(pool, dataset, args, colours):
    ga_cfg, ds_cfg, sp = load_cfgs(args.config_dir, dataset, False)
    sc, pct = ga_cfg["scaling"], ga_cfg["scaling"]["pct"]
    if args.smoke:  # pasta propria, pct do smoke, phi {0,1} e GA do smoke
        ga_cfg["results_dir"] = ga_cfg["smoke_results_dir"]
        ga_cfg["cache"]["path"] = str(Path(ga_cfg["smoke_results_dir"]) / "cache.sqlite")
        pct = ga_cfg["smoke"]["pct"]
        sc["phi"] = [p for p in sc["phi"] if p in (0, 1)]
        sc["ga"] = {"pop_size": ga_cfg["smoke"]["pop_size"], "n_gen_max": ga_cfg["smoke"]["n_gen_max"]}
    out = Path(ga_cfg["results_dir"]) / dataset / "scaling"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    base, kws, splits = pick_base(pool, ga_cfg, ds_cfg, sp, dataset, pct, colours, log)
    if base is None:
        return tqdm.write(f"{dataset}: nenhum best_config.yaml em pct {pct}; pulando")
    n_eff, bh = data.n_classes_eff(ds_cfg, base["paper_classes"]), space.config_hash(base)
    f0 = flops(base, n_eff)
    growth = base["residual"] or base["layer_mode"] == "crescimento"

    cf = out / "coefficients.yaml"
    co = yaml.safe_load(cf.read_text()) if args.resume and cf.exists() else None
    if co and co["base"]["cfg_hash"] != bh:
        log.append(f"coefficients.yaml de outra base ({co['base']['cfg_hash']}); refazendo")
        co = None
    if co is None:
        co = {"base": {"cfg_hash": bh, "cfg": base, "flops": f0}, "pct": pct, "splits": splits,
              "flops_target": sc["flops_target"], "flops_tol": sc["flops_tol"], "objective": ga_cfg["objective"]}
        for mode, names in (("compound", COEF[:4]), ("compound_gw", COEF) if growth else (None, None)):
            if mode is None:
                continue
            obj, x, fr = search_coefs(pool, base, n_eff, names, kws, ga_cfg, sc, f0, colours, ds_cfg, pct)
            co[mode] = {"coefs": x, "flops_ratio": fr, "objective": None if x is None else obj}
            if x is None:
                log.append(f"{mode}: nenhum coeficiente viavel na restricao de FLOPs")
        co["log"] = log
        cf.write_text(yaml.safe_dump(json.loads(json.dumps(co, default=float)), sort_keys=False))
    else:
        co["log"] = co.get("log", []) + log

    res_csv, done = out / "results.csv", set()
    if args.resume and res_csv.exists():
        with open(res_csv, newline="") as f:
            done = {(r["scale_mode"], float(r["phi"]), int(r["split"])) for r in csv.DictReader(f)}
    comp = (co.get("compound") or {}).get("coefs")
    modes = ["compound", *SINGLE, "compound_growth", "compound_gw"]
    bar = tqdm(modes, desc=f"{dataset} scaling", colour=colours["generations"], position=1, leave=False,
               dynamic_ncols=True)
    for mode in bar:
        coefs = (co.get("compound_gw") or {}).get("coefs") if mode == "compound_gw" else comp
        if coefs is None or (mode in ("compound_growth", "compound_gw") and not growth):
            msg = f"{mode}: pulado ({'sem coeficientes' if growth or mode in modes[:5] else 'base nao e crescimento nem residual'})"
            if msg not in co["log"]:
                co["log"].append(msg)
            continue
        for phi in sc["phi"]:
            todo = [kw for kw in kws if (mode, float(phi), kw["split"]) not in done]
            if not todo:
                continue
            target = flops(scaled(base, n_eff, comp, phi), n_eff) / f0
            if mode in SINGLE:
                m, cfg = match_single(base, n_eff, DIM[SINGLE[mode]], target, f0, phi)
                cvals = {c: (m if c == SINGLE[mode] else 1.0) for c in COEF}
            else:
                cfg = scaled(base, n_eff, coefs, phi, profile=mode == "compound_growth")
                cvals = {c: coefs.get(c, 1.0) for c in COEF}
            fr = flops(cfg, n_eff) / f0
            rows = evaluate_many(pool, [("config", cfg, kw) for kw in todo], desc=f"{mode} phi={phi}",
                                 colour=colours["evals"], position=2)
            ok = [r for r in rows if r["status"] == "ok"]
            bar.set_postfix(phi=phi, flops=f"x{fr:.3g}", **{m: f"{np.mean([float(r[k]) for r in ok]):.4g}" if ok else "nan"
                                                            for m, k in (("CH", "ch"), ("DBI", "dbi"), ("Silhouette", "silhouette"))})
            write_rows(res_csv, [{**r, "stage": "scaling", "scale_mode": mode, "phi": phi, **cvals, "flops_ratio": fr}
                                 for r in rows])
    cf.write_text(yaml.safe_dump(json.loads(json.dumps(co, default=float)), sort_keys=False))
    for line in co["log"]:
        tqdm.write(f"{dataset}: {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default=str(Path(__file__).parent / "config"))
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--gpus", nargs="*", type=int)
    args = ap.parse_args()
    ga_cfg = yaml.safe_load((Path(args.config_dir) / "ga.yaml").read_text())
    par, colours = ga_cfg["parallel"], ga_cfg["tqdm"]["colours"]
    datasets = [ga_cfg["smoke"]["dataset"]] if args.smoke else args.datasets or ga_cfg["default_datasets"]
    with make_pool(args.n_jobs or par["n_jobs"], par["gpus"] if args.gpus is None else args.gpus, ga_cfg) as pool:
        for ds in tqdm(datasets, desc="datasets", colour=colours["runs"], position=0, dynamic_ncols=True):
            run_dataset(pool, ds, args, colours)


if __name__ == "__main__":
    main()
