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
"""FMs congelados nas 11 bases, para a curva kappa x % (3 splits) e o grafico a 100% (split da busca).

So CPU: os embeddings ja existem (benchmark/common.py: emb/<FM>/<D>.npz = raw/orig; run_modehb_pre.py:
emb_modehb_pre/<FM>/<D>.npz = pre/orig). SVM = core.svm.fit_svm (linear, C=100, ovo, max_iter=-1, sem scaler),
um por vez sob /exps/moliveira/queues/svm.lock. Linha que ja existe nos CSVs do benchmark e copiada, nao refeita.
Grava all11/<FM>.csv = schema de benchmark/common.FIELDS + base, protocol (incremental|busca), pct_source.
- incremental, bases orig e *_raw: splits_incremental split1-3 (treino = descritor p%, val/teste do split).
- incremental, eggs/larvae/protozoan (pre): os nomes de pre/orig nao batem com os descritores (<95% exato e
  sem prefixo) -> ADAPTACAO do runner (eval_backbone_head.extract_pct): p% estratificado Random(42+split) do
  treino da busca (split1_modehb), min. 1 por classe; val/teste da busca. pct_source=fallback_estratificado.
- busca (split=1, pct=100): split1.json (ham10000: split1_modehb.json); parasitas = modehb_pre / modehb_raw.
Uso: PYTHONPATH=<discovery> python run_all11.py   (relancar retoma; so acrescenta o que falta)
"""
import csv
import fcntl
import json
import os
import random
import time

import numpy as np

from benchmark.common import FIELDS, SVM_LOCK, _cell, _metrics
from core.svm import fit_svm

ORG = "/dados/home/moliveira/datasets/organized"
BENCH = "/exps/moliveira/results/benchmark"
EMB = "/exps/moliveira/benchmark"
OUT = os.environ.get("ALL11_OUT", os.path.dirname(os.path.abspath(__file__)))
FMS = ["resnet50.a1_in1k", "dinov2-giant", "ijepa_vitg16_22k", "aimv2-3B-patch14-448", "convnextv2-pico-1k-224"]
ORIG = {b: b for b in ("flowers", "remote_sensing", "ham10000", "aid", "coconut_trees")}
RAW = {"eggs_raw": "Eggs", "larvae_raw": "Larvae", "protozoan_raw": "Protozoan"}
PRE = {"eggs": "Eggs", "larvae": "Larvae", "protozoan": "Protozoan"}
PCTS, SPLITS = [1, 5, 10, 25, 50, 75, 100], [1, 2, 3]
COLS = FIELDS + ["base", "protocol", "pct_source"]


def _csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def _fit(model, X, y, sets, nc, tag):
    t0 = time.perf_counter()
    with open(SVM_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait = time.perf_counter() - t0
        clf = fit_svm(X[sets["train"]], y[sets["train"]] + 1, tag=tag)
        t1, row = time.perf_counter(), {}
        for k, r in sets.items():
            for name, v in _metrics(clf, X[r], y[r], nc).items():
                row[f"{k}_{name}"] = v
    return {**row, **clf.fit_diagnostics_, "status": "ok", "lock_wait_s": round(wait, 1),
            "predict_s": round(time.perf_counter() - t1, 1)}


def _head(model, emb, D, split, pct, sets, nc):
    ytr = emb["y"][sets["train"]]
    return {"model": model, "source": str(emb["source"]), "preprocess": str(emb["preprocess"]), "dataset": D,
            "split": split, "pct": pct, "params": int(emb["params"]), "dim": emb["X"].shape[1],
            "image_size": int(emb["image_size"]), "crop_pct": float(emb["crop_pct"]), "n_train": len(ytr),
            "n_val": len(sets["val"]), "n_test": len(sets["test"]),
            "classes_missing_train": json.dumps(sorted(set(range(nc)) - set(ytr.tolist())))}


def _names(L):
    return [os.path.basename(n) for n in (L if isinstance(L, list) else list(L))]


def main():
    jobs = []
    for fm in FMS:
        path = f"{OUT}/{fm}.csv"
        done = {(r["base"], r["protocol"], r["split"], r["pct"]) for r in _csv(path)}
        old = {(r["dataset"], r["split"], r["pct"]): r for r in _csv(f"{BENCH}/{fm}.csv")}
        model = next(iter(old.values()))["model"]
        mh = {r["base"]: r for sub in ("modehb_pre", "modehb_raw") for r in _csv(f"{BENCH}/{sub}/{fm}.csv")}
        rows = []
        for base, D in {**ORIG, **RAW}.items():
            for s in SPLITS:
                for p in PCTS:
                    k = (base, "incremental", str(s), str(p))
                    if k in done:
                        continue
                    if (D, str(s), str(p)) in old:
                        rows.append({**old[(D, str(s), str(p))], "base": base, "protocol": "incremental",
                                     "pct_source": "descritor"})
                    elif os.path.exists(f"{ORG}/{D}/splits_incremental/split{s}/data_descriptor_perc{p}.json"):
                        jobs.append((fm, model, base, D, "incremental", s, p))
            k, o = (base, "busca", "1", "100"), old.get((D, "1", "100"))
            if k in done:
                continue
            if base in RAW:
                r = mh[base]
                rows.append({**{c: r[c] for c in COLS if c in r}, "model": model, "source": o["source"],
                             "preprocess": o["preprocess"], "params": o["params"], "split": 1, "pct": 100,
                             "status": "ok", "base": base, "protocol": "busca", "pct_source": "split1_modehb_raw.json"})
            elif base == "ham10000":
                jobs.append((fm, model, base, D, "busca", 1, 100))
            else:  # split1.json == descritor perc100 do split1 (conferido em 19/11)
                rows.append({**o, "base": base, "protocol": "busca", "pct_source": "split1.json"})
        for base, D in PRE.items():
            r, o = mh[base], old[(D, "1", "100")]
            k = (base, "busca", "1", "100")
            busca = {**{c: r[c] for c in COLS if c in r}, "model": model, "source": f"{o['source']} (pre/orig)",
                     "preprocess": o["preprocess"], "split": 1, "pct": 100, "status": "ok", "base": base,
                     "protocol": "busca", "pct_source": "split1_modehb.json"}
            if k not in done:
                rows.append(busca)
            for s in SPLITS:
                for p in PCTS:
                    if (base, "incremental", str(s), str(p)) in done:
                        continue
                    if p == 100:   # 100% estratificado = treino inteiro da busca, identico a linha busca
                        rows.append({**busca, "split": s, "protocol": "incremental", "pct_source": "fallback_estratificado"})
                    else:
                        jobs.append((fm, model, base, D, "pre", s, p))
        _append(path, rows)
    heavy = lambda j: (j[0].startswith("convnext") and j[2] in ("coconut_trees", "ham10000"), j[6], j[0])
    jobs.sort(key=heavy)
    print(f"[all11] {len(jobs)} SVMs na fila", flush=True)
    cache = {}
    for i, (fm, model, base, D, kind, s, p) in enumerate(jobs):
        src = f"{EMB}/emb_modehb_pre/{fm}/{D}.npz" if kind == "pre" else f"{EMB}/emb/{fm}/{D}.npz"
        if src not in cache:
            cache.clear()
            with np.load(src) as z:
                cache[src] = {k: z[k] for k in z.files}
        emb = cache[src]
        if kind == "incremental":
            row = {**_cell(model, D, s, p, emb), "base": base, "protocol": "incremental", "pct_source": "descritor"}
        else:
            sj = "split1_modehb.json"
            sp = json.load(open(f"{ORG}/{D}/splits/{sj}"))
            index = {n: j for j, n in enumerate(emb["names"])}
            sets = {k: [index[n] for n in sorted(_names(sp[v]))] for k, v in
                    (("train", "train"), ("val", "validation"), ("test", "test"))}
            nc = int(emb["y"].max()) + 1
            if kind == "pre":   # = eval_backbone_head.extract_pct, ADAPTACAO
                by, g, keep = {}, random.Random(42 + s), []
                for j in sets["train"]:
                    by.setdefault(int(emb["y"][j]), []).append((str(emb["names"][j]), j))
                for lab in sorted(by):
                    xs = sorted(by[lab]); g.shuffle(xs)
                    keep += [j for _n, j in xs[:max(1, round(len(xs) * p / 100))]]
                sets["train"] = sorted(keep)
                emb.setdefault("source", f"{model} (pre/orig)")  # o npz do pre nao grava source
                src_tag = "fallback_estratificado"
            else:
                src_tag = sj
            row = {**_head(model, emb, D, s, p, sets, nc),
                   "extract_ms_per_img": round(1000 * float(emb["seconds"]) / len(emb["y"]), 2),
                   **_fit(model, emb["X"], emb["y"], sets, nc, f"{model} {base} {kind} s{s} p{p}"),
                   "base": base, "protocol": "incremental" if kind == "pre" else "busca", "pct_source": src_tag}
        _append(f"{OUT}/{fm}.csv", [row])
        print(f"[all11] {i + 1}/{len(jobs)} {fm} {base} {row['protocol']} s{s} p{p}: test kappa {row.get('test_kappa')}"
              f" fit {row.get('svm_fit_seconds')}s", flush=True)
    print("[all11] FIM", flush=True)


def _append(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
