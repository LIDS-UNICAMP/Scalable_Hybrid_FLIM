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
"""Etapa 1: GA (pymoo MixedVariableGA) por dataset x split x pct, avaliacao paralela num Pool, saidas da §5."""

import os
from pathlib import Path

import yaml

# threads antes de numpy/torch/sklearn; os workers (spawn) herdam o ambiente
_T = str(yaml.safe_load((Path(__file__).parent / "config" / "ga.yaml").read_text())["parallel"]["threads_per_worker"])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _T

import argparse
import csv
import json
import multiprocessing as mp
import resource
import platform
import subprocess
import sys
import tempfile
import time
import warnings
import zlib
from datetime import datetime
from importlib.metadata import version

import numpy as np
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableGA, MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Binary, Choice, Integer, Real
from tqdm import tqdm

from ga_search import data, space

REPO = Path(__file__).resolve().parent.parent
SPIFIL_REPO = Path("/dados/home/moliveira/SPiFiL")
OUTS = ("results.csv", "generations.csv", "top_k.csv", "best_config.yaml", "run_meta.json")
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------- workers

_DEVICE = "cpu"


def _init(counter, gpus, threads, max_mem_gb):
    """Initializer: indice do worker -> device cuda:<gpus[i % len]> ou cpu; warnings em silencio."""
    global _DEVICE
    warnings.filterwarnings("ignore")
    # teto de RAM: uma config gulosa leva MemoryError (vira linha de erro) em vez de o OOM do Linux
    # matar o worker (13/10: um worker pediu 471 GB e a geracao travou)
    lim = int(max_mem_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    _DEVICE = f"cuda:{gpus[i % len(gpus)]}" if gpus else "cpu"
    import torch
    torch.set_num_threads(threads)


def _work(task):
    from ga_search import evaluate
    kind, payload, kw = task
    ctx = evaluate.EvalContext(**kw, device=_DEVICE)
    return (evaluate.evaluate_genome if kind == "genome" else evaluate.evaluate_config)(payload, ctx)


def make_pool(n_jobs: int, gpus: list[int], ga_cfg: dict):
    c = mp.get_context(ga_cfg["parallel"]["start_method"])
    return c.Pool(n_jobs, initializer=_init, initargs=(c.Value("i", 0), list(gpus), ga_cfg["parallel"]["threads_per_worker"],
                                                            ga_cfg["parallel"]["max_mem_gb_per_worker"]))


def evaluate_many(pool, tasks, *, desc, colour, position):
    """Avalia em paralelo, na ordem das tasks; barra com cache hits.

    Cada task tem prazo (parallel.task_timeout_s): um worker morto (ex.: OOM) some com a task
    e o Pool esperaria para sempre; estourou o prazo, a task vira linha de erro e a geracao segue.
    """
    rows, hits = [], 0
    pending = [pool.apply_async(_work, (t,)) for t in tasks]
    with tqdm(total=len(tasks), desc=desc, colour=colour, position=position, leave=False, dynamic_ncols=True) as bar:
        for t, p in zip(tasks, pending):
            g = t[2]["ga_cfg"]
            try:
                r = p.get(timeout=g["parallel"]["task_timeout_s"])
            except mp.TimeoutError:
                r = {"status": "error", "error": "timeout: worker morto ou travado", "objective": g["penalty"],
                     "cache_hit": False, "cfg_hash": None}
            rows.append(r)
            hits += bool(r.get("cache_hit"))
            bar.set_postfix(hits=hits, refresh=False)
            bar.update()
    return rows


def write_rows(path, rows):
    """Append no CSV; cabecalho fixado na criacao (se surgir coluna nova, o arquivo e reescrito com ela)."""
    path, rows = Path(path), list(rows)
    if not rows:
        return
    header, old = [], []
    if path.exists():
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
    new = list(dict.fromkeys(k for r in sorted(rows, key=len, reverse=True) for k in r if k not in header))
    if new and header:
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    header += new
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if old or not path.exists() else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore")
        if old or f.tell() == 0:
            w.writeheader()
        w.writerows(old + rows)
        f.flush()


def load_cfgs(config_dir, dataset, residual: bool):
    """(ga_cfg, ds_cfg, space) do dataset; residual troca o YAML do espaco."""
    config_dir = Path(config_dir)
    ga_cfg = yaml.safe_load((config_dir / "ga.yaml").read_text())
    ds_cfg = data.load_dataset_cfg(dataset, config_dir)
    sp = space.load_space(config_dir / ("search_space_residual.yaml" if residual else "search_space.yaml"), ds_cfg)
    return ga_cfg, ds_cfg, sp

# ---------------------------------------------------------------- GA


class _Problem(ElementwiseProblem):
    def __init__(self, sp, runner):
        super().__init__(vars=space.pymoo_vars(sp), n_obj=1, elementwise_runner=runner)


class _Runner:
    """elementwise_runner do pymoo: manda a geracao para o Pool e guarda as linhas (results.csv)."""

    def __init__(self, pool, kw, path, colour):
        self.pool, self.kw, self.path, self.colour = pool, kw, path, colour
        self.gen, self.idx, self.rows, self.by_x = 0, 0, [], {}

    def __call__(self, f, X):
        rows = evaluate_many(self.pool, [("genome", x, self.kw) for x in X],
                             desc=f"avaliacoes g{self.gen}", colour=self.colour, position=2)
        for x, r in zip(X, rows):
            r.update(stage="ga", gen=self.gen, eval_idx=self.idx)
            self.idx += 1
            self.by_x[id(x)] = (x, r)
        self.rows += rows
        write_rows(self.path, rows)
        return [{"F": float(r["objective"])} for r in rows]


def _seed(ga_cfg, dataset, pct, split):
    return ga_cfg["seeds"]["base"] + zlib.crc32(f"{dataset}|{pct}|{split}".encode()) % 2**31


def _ok(rows):
    return [r for r in rows if r["status"] == "ok"]


def _gen_stats(rows, ga_cfg):
    """{metrica}_{best,mean,worst,std} sobre as linhas ok (best: objetivo min; as outras pelo sentido de ga.yaml metrics)."""
    ok, out = _ok(rows), {}
    for m, lo in (("objective", True), *((k, d == "min") for k, d in ga_cfg["metrics"].items())):
        v = np.array([float(r[m]) for r in ok])
        b, w = (v.min(), v.max()) if lo else (v.max(), v.min())
        out.update({f"{m}_best": b, f"{m}_mean": v.mean(), f"{m}_worst": w, f"{m}_std": v.std()} if len(v)
                   else {f"{m}_{s}": None for s in ("best", "mean", "worst", "std")})
    return out


def _stop(hist, ga_cfg):
    """Criterio de parada (§6.2) sobre o historico de geracoes; None = continua."""
    st, p, last = ga_cfg["stop"], ga_cfg["stop"]["patience"], hist[-1]
    if len(hist) > p and hist[-1 - p]["objective_best"] is not None and last["objective_best"] is not None:
        win = hist[-1 - p:]
        if len({h["best_hash"] for h in win}) == 1 and min(h["dbi_best"] for h in win[1:]) >= win[0]["dbi_best"]:
            return "cross_stability"
        a, b = win[0]["objective_best"], last["objective_best"]
        if (a - b) / max(abs(a), 1e-12) < st["tol"]:
            return "stagnation"
    if last["n_ok_pop"] >= 2 and last["objective_std"] / max(abs(last["objective_mean"]), 1e-12) < st["pop_tol"]:
        return "population"
    return None


def _cast_genome(row, vars_, sp):
    """Linha do top_k.csv -> genoma com os tipos do pymoo; gene ausente fica no default."""
    X = space.default_genome(sp)
    for n, v in vars_.items():
        s = row.get(f"g_{n}", "")
        if s in ("", None):
            continue
        X[n] = (float(s) if isinstance(v, Real) else int(float(s)) if isinstance(v, Integer)
                else s == "True" if isinstance(v, Binary) else next(o for o in v.options if str(o) == s))
    return X


def _git(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _versions():
    v = {"python": platform.python_version()}
    for k, pkg in (("torch", "torch"), ("pymoo", "pymoo"), ("sklearn", "scikit-learn"), ("spifil", "spifil")):
        try:
            v[k] = version(pkg)
        except Exception:
            v[k] = None
    return v


def _kw(dataset, pct, split, ds_cfg, sp, ga_cfg, seed):
    return {"dataset": dataset, "pct": pct, "split": split, "ds_cfg": ds_cfg, "space": sp, "ga_cfg": ga_cfg, "seed": seed}


def run_one(pool, dataset, pct, split, ga_cfg, ds_cfg, sp, run_dir, warm_dir, pop_size, n_gen_max, colours):
    """Um run do GA; escreve os 5 arquivos da §5 em run_dir."""
    for n in OUTS:
        (run_dir / n).unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed, t0 = _seed(ga_cfg, dataset, pct, split), time.time()
    meta = {"seeds": {"base": ga_cfg["seeds"]["base"], "run": seed}, "ga_cfg": ga_cfg, "space_cfg": sp, "ds_cfg": ds_cfg,
            "versions": _versions(), "git_commit": {"repo": _git(REPO), "spifil": _git(SPIFIL_REPO)},
            "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "duration_s": None,
            "n_gen": 0, "n_evals": 0, "stop_reason": None, "warm_start_from": None,
            "dataset": dataset, "pct": pct, "split": split, "pop_size": pop_size, "n_gen_max": n_gen_max}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    runner = _Runner(pool, _kw(dataset, pct, split, ds_cfg, sp, ga_cfg, seed), run_dir / "results.csv", colours["evals"])
    problem = _Problem(sp, runner)
    warm = []
    if warm_dir is not None and (warm_dir / "top_k.csv").exists():
        with open(warm_dir / "top_k.csv", newline="") as f:
            warm = [_cast_genome(r, problem.vars, sp) for r in list(csv.DictReader(f))[:min(ga_cfg["ga"]["warm_start_k"], pop_size)]]
        meta["warm_start_from"] = str(warm_dir)
    rnd = MixedVariableSampling().do(problem, max(pop_size - len(warm), 0), random_state=np.random.default_rng(seed)).get("X")
    X0 = np.empty(len(warm) + len(rnd), dtype=object)
    X0[:] = warm + list(rnd)
    algo = MixedVariableGA(pop_size=pop_size, sampling=Population.new(X=X0),
                           eliminate_duplicates=MixedVariableDuplicateElimination() if ga_cfg["ga"]["eliminate_duplicates"] else None)
    algo.setup(problem, termination=("n_gen", n_gen_max), seed=seed, verbose=False)

    hist, stop = [], None
    gbar = tqdm(total=n_gen_max, desc=f"{dataset} p{pct} s{split}", colour=colours["generations"], position=1,
                leave=False, dynamic_ncols=True)
    while stop is None:
        runner.gen = len(hist) + 1
        n0 = len(runner.rows)
        algo.next()
        new = runner.rows[n0:]
        pop = [runner.by_x[id(x)][1] for x in algo.pop.get("X")]
        st = _gen_stats(pop, ga_cfg)
        best = min(_ok(pop), key=lambda r: float(r["objective"]), default=None)
        h = {"dataset": dataset, "pct": pct, "split": split, "gen": runner.gen, **st,
             "n_evals_new": sum(not r.get("cache_hit") for r in new), "n_cache_hits": sum(bool(r.get("cache_hit")) for r in new),
             "n_invalid": sum(r["status"] == "invalid" for r in new), "best_hash": best and best["cfg_hash"],
             "n_ok_pop": len(_ok(pop))}
        hist.append(h)
        stop = _stop(hist, ga_cfg) or ("n_gen_max" if runner.gen >= n_gen_max else None)
        ok = _ok(runner.rows)
        best_of = {k: (min if d == "min" else max) for k, d in ga_cfg["metrics"].items()}
        gbar.set_postfix(**{k: f"{best_of[k]((float(r[k]) for r in ok), default=float('nan')):.4g}" for k in best_of},
                         hits=sum(bool(r.get("cache_hit")) for r in runner.rows), refresh=False)
        gbar.update()
        write_rows(run_dir / "generations.csv", [{**{k: v for k, v in h.items() if k != "n_ok_pop"}, "stop_reason": stop or ""}])
    gbar.close()

    seen, top = set(), []
    for r in sorted(_ok(runner.rows), key=lambda r: float(r["objective"])):
        if r["cfg_hash"] not in seen and len(top) < ga_cfg["ga"]["top_k"]:
            seen.add(r["cfg_hash"])
            top.append({**r, "rank": len(top) + 1})
    write_rows(run_dir / "top_k.csv", top)
    if top:
        cfg = json.loads(top[0]["cfg_json"])
        out = {**space.to_spifil_yaml(cfg, ds_cfg, pct, split),
               "metrics": {m: top[0][m] for m in ("ch", "dbi", "silhouette", "objective")},
               "cfg_hash": top[0]["cfg_hash"], "cfg": cfg}
        (run_dir / "best_config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(out, default=float)), sort_keys=False))
    meta.update(finished=datetime.now().isoformat(timespec="seconds"), duration_s=time.time() - t0,
                n_gen=len(hist), n_evals=len(runner.rows), stop_reason=stop)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta

# ---------------------------------------------------------------- CLI


def _estimate(pool, args, ga_cfg, datasets):
    """Tempo medio de avaliacao de genomas validos aleatorios (pct 100, split 1, cache temporario)."""
    e, total = ga_cfg["estimate"], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for ds in datasets:
            g, ds_cfg, sp = load_cfgs(args.config_dir, ds, args.residual)
            g["cache"]["path"] = str(Path(tmp) / f"{ds}.sqlite")
            pct, split, seed = 100, 1, _seed(g, ds, 100, 1)
            prob, rng, cfgs = _Problem(sp, None), np.random.default_rng(seed), []
            for _ in range(1000):
                for X in MixedVariableSampling().do(prob, 64, random_state=rng).get("X"):
                    cfg = space.active_config(X, sp, ds_cfg)
                    ctx = {"n_available": data.n_available(ds_cfg, pct, split, paper_classes=cfg["paper_classes"]),
                           "n_classes_eff": data.n_classes_eff(ds_cfg, cfg["paper_classes"]),
                           "max_seeds": g["spifil"]["max_seeds"]}
                    if space.is_valid(cfg, ctx)[0] and len(cfgs) < e["n_evals_per_dataset"]:
                        cfgs.append(cfg)
                if len(cfgs) >= e["n_evals_per_dataset"]:
                    break
            rows = evaluate_many(pool, [("config", c, _kw(ds, pct, split, ds_cfg, sp, g, seed)) for c in cfgs],
                                 desc=f"estimate {ds}", colour=g["tqdm"]["colours"]["evals"], position=0)
            t = float(np.mean([r["time_s"] for r in rows]))
            n_runs = len(args.pcts or ds_cfg["pcts"]) * len(args.splits or ds_cfg["splits"])
            pred = t * g["ga"]["pop_size"] * e["assumed_n_gen"] * n_runs / args.n_jobs
            total += pred
            print(f"{ds}: {t:.1f} s/aval (status {[r['status'] for r in rows]}); "
                  f"{n_runs} runs x {g['ga']['pop_size']} x {e['assumed_n_gen']} ger / {args.n_jobs} jobs "
                  f"= {pred / 3600:.1f} h")
    print(f"TOTAL Etapa 1: {total / 3600:.1f} h ({total / 86400:.2f} dias)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", default=str(Path(__file__).parent / "config"))
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--pcts", nargs="+", type=int)
    ap.add_argument("--splits", nargs="+", type=int)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--gpus", nargs="*", type=int)
    args = ap.parse_args()
    ga_cfg = yaml.safe_load((Path(args.config_dir) / "ga.yaml").read_text())
    par, sm = ga_cfg["parallel"], ga_cfg["smoke"]
    args.n_jobs = args.n_jobs or par["n_jobs"]
    gpus = par["gpus"] if args.gpus is None else args.gpus
    if args.smoke:
        args.datasets, args.pcts, args.splits = [sm["dataset"]], [sm["pct"]], [sm["split"]]
    datasets = args.datasets or ga_cfg["default_datasets"]

    with make_pool(args.n_jobs, gpus, ga_cfg) as pool:
        if args.estimate:
            return _estimate(pool, args, ga_cfg, datasets)
        plan = []
        for ds in datasets:
            g, ds_cfg, sp = load_cfgs(args.config_dir, ds, args.residual)
            if args.smoke:
                g["results_dir"] = g["smoke_results_dir"]
                g["cache"]["path"] = str(Path(g["smoke_results_dir"]) / "cache.sqlite")
                g["ga"].update(pop_size=sm["pop_size"], n_gen_max=sm["n_gen_max"])
            Path(g["cache"]["path"]).parent.mkdir(parents=True, exist_ok=True)
            base = Path(g["results_dir"]) / ds / ("residual" if args.residual else "")
            for s in args.splits or ds_cfg["splits"]:
                prev = None
                for p in sorted(args.pcts or ds_cfg["pcts"]):
                    plan.append((ds, p, s, g, ds_cfg, sp, base / f"pct_{p}" / f"split_{s}", prev))
                    prev = base / f"pct_{p}" / f"split_{s}"
        for ds, p, s, g, ds_cfg, sp, run_dir, prev in tqdm(plan, desc="runs", colour=ga_cfg["tqdm"]["colours"]["runs"],
                                                           position=0, dynamic_ncols=True):
            mf = run_dir / "run_meta.json"
            if mf.exists():
                if not args.resume:
                    sys.exit(f"{run_dir} ja existe; use --resume")
                if json.loads(mf.read_text()).get("finished"):
                    tqdm.write(f"pulando {run_dir} (finished)")
                    continue
            m = run_one(pool, ds, p, s, g, ds_cfg, sp, run_dir, prev, g["ga"]["pop_size"], g["ga"]["n_gen_max"],
                        g["tqdm"]["colours"])
            tqdm.write(f"{run_dir}: {m['n_gen']} ger, {m['n_evals']} aval, stop={m['stop_reason']}, {m['duration_s']:.0f} s")


if __name__ == "__main__":
    main()
