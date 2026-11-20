"""Monitor MO-DEHB runs: TEST SVM for EVERY finished candidate job (all fidelities) + Markdown
report vs the foundation-model baselines. Loop: python -m ga_search.modehb.monitor_report

Two rulers per job, both reporting-only (never read by the search):
  * ORIGINAL = the repo's canonical SVM, ``core.svm.fit_svm`` with its defaults (linear SVC, C=100,
    ovo, unbounded solver, NO scaler) on the model embedding ``head(encoder(x))`` -- the same
    protocol as the frozen foundation-model baselines. This is the number that is reported.
  * COV = the search's fitness readout (cov_head, C=100, StandardScaler), for reference.
Order: highest fidelity first, then newest job. Archived searches (*_C01_*) are evaluated too.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import torch

ROOT_GLOBS = ["artifacts/modehb_hybrid", "artifacts/modehb_lemonade",
              "artifacts/modehb_hybrid_C01_*", "artifacts/modehb_lemonade_C01_*",
              "artifacts/modehb_seeded", "artifacts/modehb_lemonade_seeded",   # external-seed searches
              "artifacts/pnas_cs"]                                           # PNAS-CS (pnas_cs.py)
# roles reported on TEST (never fed back): search jobs + the PNAS-CS Phase-0 AE ablation arms
REPORT_ROLES = ("search", "phase0_ae", "phase0_noae")
OUT_JSONL = Path("ga_search/modehb/reports/test_svm_C100.jsonl")   # C=100 sempre; o C=0.1 antigo fica em test_svm.jsonl
OUT_MD = Path("ga_search/modehb/reports/HYBRID_PROGRESS.md")
# test kappa (split 1) of frozen foundation models + linear SVM C=100 (claude_explore_models, 14/11)
FM = {"remote_sensing": {"AIMv2": 0.746, "DINOv2": 0.719, "ResNet-50": 0.704, "I-JEPA": 0.698, "ConvNeXtV2-pico": 0.655},
      "coconut_trees": {"ConvNeXtV2-pico": 0.792, "I-JEPA": 0.785, "AIMv2": 0.737, "DINOv2": 0.685, "ResNet-50": 0.671},
      "ham10000": {"AIMv2": 0.724, "DINOv2": 0.701, "I-JEPA": 0.595, "ResNet-50": 0.492, "ConvNeXtV2-pico": 0.485},
      # athena benchmark, test split1 (original split1.json), 100% train
      "flowers": {"AIMv2": 0.979, "DINOv2": 0.976, "I-JEPA": 0.930, "ConvNeXtV2-pico": 0.878, "ResNet-50": 0.872},
      "larvae": {"AIMv2": 1.000, "DINOv2": 1.000, "ConvNeXtV2-pico": 0.933, "I-JEPA": 0.919, "ResNet-50": 0.906},
      "eggs": {"DINOv2": 0.988, "AIMv2": 0.985, "I-JEPA": 0.976, "ResNet-50": 0.948, "ConvNeXtV2-pico": 0.944},
      "aid": {"AIMv2": 0.979, "DINOv2": 0.970, "I-JEPA": 0.916, "ResNet-50": 0.898, "ConvNeXtV2-pico": 0.888},
      "protozoan": {"AIMv2": 0.968, "DINOv2": 0.956, "I-JEPA": 0.938, "ConvNeXtV2-pico": 0.853, "ResNet-50": 0.822}}


def _kind(root):
    if "pnas_cs" in Path(root).name:
        return "pnas_cs"
    kind = "lemonade" if "lemonade" in Path(root).name else "hybrid"
    return kind + "_seeded" if "seeded" in Path(root).name else kind   # own keys: cand ids restart at c00001


def _fit_sig(state, root=None):
    """Fitness signature of a search. search_state.json does not store the eval block, so fall back to the
    directory convention: archives named *_C01_* hold the old C=0.1/kw=0.75 runs; every live / seeded / pnas
    directory was (re)started after 15/11 20:50 with C=100, kappa_weight=1.0."""
    sv = ((state.get("config") or {}).get("eval") or {}).get("svm_fitness") or {}
    if sv:
        return f"C{float(sv.get('C', 100.0)):g}_kw{float(sv.get('kappa_weight', 1.0)):g}"
    if root is not None and "_C01" in Path(root).name:
        return "C0.1_kw0.75"
    return "C100_kw1"


def _key(kind, ds, sig, cand, fid):
    return f"{kind}|{ds}|{sig}|{cand}|{float(fid):g}"


MAX_ERRORS = 3            # a job whose evaluation errored is retried on later passes, up to this many times
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CUDA_VISIBLE_DEVICES= -> CPU (18/11)
CAP_GB = 6.0              # GPU memory cap of the monitor process, absolute (any GPU size)


def done_keys():
    """Keys that need no more work: evaluated OK, or errored MAX_ERRORS times."""
    if not OUT_JSONL.exists():
        return set()
    ok, errs = set(), {}
    for r in map(json.loads, OUT_JSONL.read_text().splitlines()):
        if "key" not in r:
            continue
        if "test_kappa" in r:
            ok.add(r["key"])
        elif "out of memory" not in str(r.get("error", "")).lower():   # OOM on a full shared GPU: retry forever
            errs[r["key"]] = errs.get(r["key"], 0) + 1
    return ok | {k for k, n in errs.items() if n >= MAX_ERRORS}


def _ckpt(st, job):
    """Checkpoint path; if the search dir was moved to an archive, rebase it onto the new dir."""
    p = Path(job["result"]["checkpoint"])
    if p.exists():
        return str(p)
    parts = p.parts
    if "candidates" in parts:
        q = st.parent.joinpath(*parts[parts.index("candidates"):])
        if q.exists():
            return str(q)
    return None


TOP_KEYS: set = set()   # best-by-validation key of each (search, dataset, stratum), refreshed by pending()


def pending():
    seen = done_keys()
    todo = []
    for g in ROOT_GLOBS:
        for root in sorted(Path(".").glob(g)):
            for st in root.glob("*/state/seed*/search_state.json"):
                ds = st.parts[-4]
                if ds in {x.strip() for x in os.environ.get("MONITOR_SKIP_DS", "").split(",") if x.strip()}:
                    continue                      # bases fora do escopo (usuário 18/11: nenhum parasita)
                try:
                    s = json.loads(st.read_text())
                except Exception:
                    continue
                sig, kind = _fit_sig(s, root), _kind(root)
                if st.parts[-2] != "seed42":          # other search seeds get their own keys (cand ids restart)
                    kind = f"{kind}_{st.parts[-2]}"
                for k, j in enumerate(s["jobs"]):
                    if j["status"] != "ok" or j.get("role") not in REPORT_ROLES or not j["result"].get("checkpoint"):
                        continue
                    key = _key(kind, ds, sig, j["cand_id"], j["fidelity"])
                    if key in seen:
                        continue
                    seen.add(key)
                    todo.append((float(j["fidelity"]), k, str(root), ds, st, sig, kind, j, key))
    # priority: (1) new fitness (C=100) before archived C=0.1; (2) the best-by-VALIDATION job of each
    # (search, dataset, stratum) first, so every size class (tiny..huge, P0..P4) gets a test number early;
    # (3) high fidelity; (4) newest job.
    best = {}
    for t in todo:
        j = t[7]
        g = (t[6], t[3], t[5], j.get("stratum"))
        q = j["result"].get("fitness_kappa", j["result"].get("quality_raw", -1.0)) or -1.0
        if g not in best or (t[0], q) > best[g][0]:
            best[g] = ((t[0], q), t[8])
    top = {v[1] for v in best.values()}
    TOP_KEYS.clear(); TOP_KEYS.update(top)
    todo.sort(key=lambda t: (t[5] == "C0.1_kw0.75", t[8] not in top, -t[0], -t[1]))
    shard = os.environ.get("MONITOR_SHARD")          # "i/n": several monitors split the work by key hash
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        todo = [t for t in todo if int(hashlib.md5(t[8].encode()).hexdigest(), 16) % n == i]
    return todo


def evaluate_new(max_jobs=None):
    from ga_search.modehb.run_search import load_config
    from ga_search.modehb.svm_on_demand import _test_loader, evaluate_checkpoint
    from ga_search.progressive_data import load_data
    data_cache = {}
    for n, (fid, k, root, ds, st, sig, kind, j, key) in enumerate(pending()[:max_jobs] if max_jobs else pending()):
        if max_jobs is not None and n >= max_jobs:
            break
        if ds not in data_cache:
            data_cache.clear()                     # one dataset in RAM at a time
            cfg = load_config("ga_search/config/modehb/search_v3.yaml", ds)
            d = {kk: v for kk, v in cfg["data"].items() if kk not in ("ram_cache", "val_max_samples", "drop_last")}
            d.update(num_workers=8, seed=cfg["seed"])
            tr, va, _, _ = load_data(d, Path(cfg["output_dir"]) / "svm_data")
            data_cache[ds] = (cfg, {"train": tr, "validation": _test_loader(d)})
        cfg, data = data_cache[ds]
        rec = {"key": key, "run": root, "kind": kind, "fitness_sig": sig, "dataset": ds, "cand_id": j["cand_id"],
               "fidelity": fid, "job": k, "stratum": j.get("stratum"), "op": j.get("op"), "role": j.get("role"),
               "val_kappa": j["result"].get("fitness_kappa"), "params": j["result"].get("params"),
               "flops": j["result"].get("flops"), "split": "test"}
        ck = _ckpt(st, j)
        res = {}
        try:
            if ck is None:
                raise FileNotFoundError(j["result"]["checkpoint"])
            t0 = time.time()
            res["orig"] = evaluate_checkpoint(ck, data, DEV, scaler=False, readout="head", C=100.0)
            rec["orig_seconds"] = round(time.time() - t0, 1)
            # cov_head column (ablation, same protocol: C=100 + StandardScaler, full train) is slow on big train
            # sets (libsvm C=100 on 9.5k x 1.4-8k dims: ~1 h). MONITOR_COV=top computes it only for the
            # best-by-validation job of each (search, dataset, stratum); the SVM-original column is always computed.
            if os.environ.get("MONITOR_COV", "all") == "all" or key in TOP_KEYS:
                t0 = time.time()
                res["cov"] = evaluate_checkpoint(ck, data, DEV, scaler=True, readout="cov_head", C=100.0)
                rec["cov_seconds"] = round(time.time() - t0, 1)
            o = res["orig"]
            rec.update(test_kappa=o["kappa"], test_f1=o["macro_f1"], orig_protocol=o["svm_protocol"],
                       orig_dim=o["embed_dim"], n_train=o["n_train"], n_test=o["n_validation"])
            if "cov" in res:
                c = res["cov"]
                rec.update(cov_kappa=c["kappa"], cov_f1=c["macro_f1"], cov_protocol=c["svm_protocol"], cov_dim=c["embed_dim"])
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["time"] = time.time()
        with OUT_JSONL.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)              # several monitor shards append to the same file
            f.write(json.dumps(rec) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"[monitor] {key} orig={rec.get('test_kappa')} cov={rec.get('cov_kappa')} {rec.get('error', '')}", flush=True)
        if "test_kappa" in rec:   # every test result also goes to W&B (external_eval run)
            try:
                from ga_search.modehb import wandb_logger as WL
                o, c = res["orig"], res.get("cov")
                run = WL.ExternalEvalRun(f"{kind}-{sig}-{j['cand_id']}-f{int(fid)}", dataset=ds,
                                         project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"],
                                         config={"split_eval": "test", "ruler": "original: core.svm.fit_svm C=100 no scaler, head embedding",
                                                 "C": 100.0, "scaler": False, "readout": "head", "search_run": root,
                                                 "fitness_sig": sig, "checkpoint": ck, "val_kappa": rec["val_kappa"],
                                                 "params": rec["params"], "flops": rec["flops"], "stratum": rec["stratum"],
                                                 "never_feeds_fitness": True})
                run.add_result(macro_f1=o["macro_f1"], kappa=o["kappa"], balanced_acc=o["balanced_acc"],
                               per_class=o["per_class"], fidelity=fid)
                if c is not None:
                    run.run.log({"svm_cov_head/kappa": c["kappa"], "svm_cov_head/macro_f1": c["macro_f1"]})
                run.run.summary.update({"test_kappa": o["kappa"], "test_kappa_original": o["kappa"],
                                        "test_kappa_cov_head": c["kappa"] if c is not None else None, "fidelity": fid})
                run.finish()
            except Exception as exc:
                print(f"[monitor] wandb upload failed: {exc}", flush=True)
        res.clear()
        if torch.cuda.is_available(): torch.cuda.empty_cache()   # shared GPU: give the reserved memory back between evaluations

OUT_CSV = Path("ga_search/modehb/reports/ranking_ga_C100.csv")   # export p/ a tabela de resultados (results_grids)
BASE_TYPE = {"remote_sensing": "natural", "flowers": "natural", "aid": "natural", "ham10000": "natural",
             "coconut_trees": "natural", "larvae": "fundo preto (pre/orig)", "eggs": "fundo preto (pre/orig)",
             "protozoan": "fundo preto (pre/orig)", "larvae_raw": "imagem original (raw/orig)",
             "eggs_raw": "imagem original (raw/orig)", "protozoan_raw": "imagem original (raw/orig)"}


def _genome_summary(g):
    if not g:
        return ""
    L = [l for l in g.get("layers", []) if l.get("active")]
    arch = "-".join(f"{l['conv'][:3]}{l['kernel']}x{l['channels']}" + (f"p{l['pool'][0]}" if l.get("pool") not in (None, "none") else "")
                    for l in L)
    return (f"{len(L)}L {arch} | init={g.get('init')} ssl={g.get('ssl')} reg={g.get('reg')} protect={g.get('protect')} "
            f"pool={g.get('pooling')}")


def write_csv():
    """One row per (job, ruler): SVM original (headline) and cov_head, test split1, C=100."""
    import csv
    lines = OUT_JSONL.read_text().splitlines() if OUT_JSONL.exists() else []
    genomes = {}
    for g in ROOT_GLOBS:
        for root in Path(".").glob(g):
            for st in root.glob("*/state/seed*/search_state.json"):
                try:
                    s = json.loads(st.read_text())
                except Exception:
                    continue
                sig, kind = _fit_sig(s, root), _kind(root)
                if st.parts[-2] != "seed42":
                    kind = f"{kind}_{st.parts[-2]}"
                for j in s["jobs"]:
                    genomes[_key(kind, st.parts[-4], sig, j["cand_id"], j["fidelity"])] = j.get("genome")
    best = {}
    for i, l in enumerate(lines, 1):
        r = json.loads(l)
        if "test_kappa" in r and "key" in r:
            best[r["key"]] = (i, r)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tecnica", "base", "tipo", "pct", "split", "conjunto", "metrica", "valor", "params", "flops",
                    "fidelidade_epocas", "estrato", "cand_id", "fitness", "regua", "genoma", "fonte"])
        for key, (i, r) in sorted(best.items(), key=lambda kv: (kv[1][1]["dataset"], -kv[1][1]["test_kappa"])):
            tec = ("Hybrid SPiFiL PNAS-CS" if r["kind"].startswith("pnas_cs") else
                   ("Hybrid SPiFiL MO-DEHB" if r["kind"].startswith("hybrid") else "Hybrid SPiFiL LEMONADE") +
                   (" semeada (SPiFiL pequeno + LeJEPA)" if "_seeded" in r["kind"] else "")) + \
                (f" ({r['kind'].rsplit('_', 1)[-1]})" if "_seed4" in r["kind"] else "")
            g = _genome_summary(genomes.get(key))
            common = [tec, r["dataset"], BASE_TYPE.get(r["dataset"], "?"), 100, "split1 (seed42, n=1)", "test", "svm_kappa"]
            src = f"jaci:ga_search/modehb/reports/test_svm_C100.jsonl:{i}"
            tail = [r.get("params"), r.get("flops"), r["fidelity"], r.get("stratum"), r["cand_id"], r["fitness_sig"]]
            w.writerow(common[:-1] + ["svm_kappa", round(r["test_kappa"], 4)] + tail +
                       ["SVM original: core.svm.fit_svm C=100 sem scaler, embedding head 1280-d", g, src])
            if "cov_kappa" in r:
                w.writerow(common[:-1] + ["svm_kappa", round(r["cov_kappa"], 4)] + tail +
                           ["cov_head (2a ordem por bloco) + StandardScaler, SVM C=100 (ablação)", g, src])


def write_md():
    rows = [json.loads(l) for l in OUT_JSONL.read_text().splitlines()] if OUT_JSONL.exists() else []
    rows = list({r["key"]: r for r in rows if "test_kappa" in r and "key" in r}.values())
    out = ["# Hybrid SPiFiL — κ de TESTE com o SVM ORIGINAL do repositório vs modelos grandes",
           f"_Atualizado: {time.strftime('%d/%m %H:%M')}. SVM original = core.svm.fit_svm (SVC linear, C=100, "
           "sem scaler) no embedding do modelo, mesmo protocolo dos FMs. Coluna cov_head = leitura do fitness "
           "(C=100). Seleção na validação; teste só reportado._", ""]
    for ds in sorted({r["dataset"] for r in rows}):
        rr = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: -r["test_kappa"])[:12]
        out += [f"## {ds}", "", "| modelo | κ teste SVM original | κ teste cov_head | κ val (fitness) | fid | params | estrato | busca |",
                "|---|---|---|---|---|---|---|---|"]
        for name, k in sorted(FM.get(ds, {}).items(), key=lambda x: -x[1]):
            out.append(f"| **{name}** (congelado) | {k:.3f} | — | — | — | — | FM | baseline |")
        for r in rr:
            out.append(f"| {r['cand_id']} | {r['test_kappa']:.3f} | {r.get('cov_kappa', float('nan')):.3f} | "
                       f"{r['val_kappa'] or 0:.3f} | {r['fidelity']:g} | {(r['params'] or 0)/1e3:.0f}k | {r['stratum']} | "
                       f"{r['kind']} {r['fitness_sig']} ({Path(r['run']).name}) |")
        out.append("")
    OUT_MD.write_text("\n".join(out))


if __name__ == "__main__":
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, CAP_GB / total_gb))
    while True:
        busy = False
        try:
            busy = bool(pending())
            if busy:
                evaluate_new(max_jobs=6)      # re-plan every 6 evaluations (new jobs / new top candidates)
            write_md()
            write_csv()
        except Exception as exc:
            print(f"[monitor] {exc}", flush=True)
        if not busy:
            time.sleep(300)
