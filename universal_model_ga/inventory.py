"""inventory.py -- inventário centralizado das CNNs produzidas pelo GA / MO-DEHB / LEMONADE / crescimento SPiFiL.

Um registro por (busca, base, candidato, fidelidade) com os campos pedidos:
  architecture_id, dataset, architecture_config, graph, kappa_val, kappa_test, params, flops, latency,
  checkpoint_path, seed, generation, readout_type
e extras (source, search, cand_id, fidelity, arch_hash, procedure, kappa_by_readout, focus, missing).
Campo ausente = None e listado em `missing` (nunca inventado).

kappa_val = κ de VALIDAÇÃO da própria busca (fitness, SVM original na leitura da busca) -- é o que seleciona.
kappa_by_readout[head] = {val, test} das reavaliações oficiais (readout_x: gap1280, so2_4096_pca, *_tta8);
kappa_test = teste da SVM original na leitura da busca (test_svm_C100) ou, se não houver, o gap1280 do readout_x.
O TESTE nunca entra em seleção (só relatório).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import append_jsonl, load_config, md_table, read_jsonl, write_json

FIELDS = ["architecture_id", "dataset", "architecture_config", "graph", "kappa_val", "kappa_test", "params", "flops",
          "latency", "checkpoint_path", "seed", "generation", "readout_type"]


def _search_name(path: Path) -> str:
    # artifacts/<search>/<dataset>/state/seed42/candidates/<cid>/f<fid>/result.json
    return path.parts[path.parts.index("artifacts") + 1]


def _jobs_index(state_json: Path) -> dict:
    """(cand_id, fidelity) -> job (genome, generation, arch_hash, F) do search_state.json."""
    try:
        d = json.loads(state_json.read_text())
    except Exception:
        return {}
    idx = {}
    for j in d.get("jobs", []) or []:
        idx[(j.get("cand_id"), float(j.get("fidelity", -1)))] = j
    return idx


def _modehb_records(repo: Path, glob_pat: str) -> list[dict]:
    recs, jobs_cache = [], {}
    for rp in sorted(glob.glob(str(repo / glob_pat))):
        rp = Path(rp)
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        fid = float(rp.parent.name[1:])
        cid, ds, search = rp.parent.parent.name, rp.parents[5].name, _search_name(rp)
        st = rp.parents[3] / "search_state.json"
        if st not in jobs_cache:
            jobs_cache[st] = _jobs_index(st)
        job = jobs_cache[st].get((cid, fid)) or {}
        g = job.get("genome")
        ck = rp.parent / "encoder.pt"
        glob_keys = ["init", "ssl", "reg", "pooling", "ae_target", "ae_schedule", "embed_dim", "projector", "protect",
                     "spifil_images_per_class", "seed", "depth"]
        rec = {
            "architecture_id": f"{search}/{ds}/{cid}/f{fid:g}",
            "source": "lemonade" if "lemonade" in search else "modehb",
            "search": search, "dataset": ds, "cand_id": cid, "fidelity": fid, "status": r.get("status"),
            "architecture_config": {"effective_specs": r.get("effective_specs"),
                                    "genome_layers": [L for L in (g or {}).get("layers", []) if L.get("active", True)] if g else None,
                                    "globals": {k: (g or {}).get(k) for k in glob_keys} if g else None},
            "graph": None,
            "kappa_val": r.get("fitness_kappa", r.get("quality")) if r.get("status") == "ok" else None,
            "fitness_mode": r.get("fitness_mode"),
            "kappa_test": None, "kappa_by_readout": {},
            "params": r.get("params"), "flops": r.get("flops"), "latency": r.get("latency_ms"),
            "checkpoint_path": str(ck) if ck.is_file() else None,
            "seed": (g or {}).get("seed", 42), "generation": job.get("generation"),
            "readout_type": f"search:{(g or {}).get('pooling')}|fitness:{r.get('fitness_mode')}" if g else f"fitness:{r.get('fitness_mode')}",
            "arch_hash": job.get("arch_hash"),
            "procedure": {"ae_epochs_per_layer": r.get("ae_epochs_per_layer"), "ssl": (g or {}).get("ssl"),
                          "ssl_epochs": r.get("ssl_epochs"), "reg": (g or {}).get("reg"), "init": (g or {}).get("init")},
        }
        recs.append(rec)
    return recs


def _growth_records(repo: Path, glob_pat: str) -> list[dict]:
    recs = []
    for rp in sorted(glob.glob(str(repo / glob_pat), recursive=True)):
        rp = Path(rp)
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        parts = rp.parts
        ds = next((d for p in parts for d in _ALL_DS if p == d or p.startswith(d + "_fix") or p.startswith(d + "_B")), None) or "AUSENTE"
        node, fid = rp.parent.parent.name, rp.parent.name
        rel = "/".join(parts[parts.index("artifacts") + 1:-1])
        ck = rp.parent / "backbone.pt"
        recs.append({
            "architecture_id": rel, "source": "growth", "search": parts[parts.index("artifacts") + 2],
            "dataset": ds, "cand_id": node, "fidelity": float(fid[1:]) if fid[1:].replace(".", "").isdigit() else None,
            "status": r.get("status"),
            "architecture_config": {"arch": r.get("arch"), "ops": r.get("ops"), "genes": r.get("genes")},
            "graph": None, "kappa_val": r.get("kappa_val_orig", r.get("quality")), "fitness_mode": r.get("readout"),
            "kappa_test": None, "kappa_by_readout": {},
            "params": r.get("params"), "flops": r.get("flops"), "latency": None,
            "checkpoint_path": str(ck) if ck.is_file() else None, "seed": r.get("seed"),
            "generation": None, "readout_type": f"growth:{r.get('readout')}", "arch_hash": r.get("growth_spec_id"),
            "procedure": {"ssl": r.get("ssl_method"), "ae_epochs": r.get("ae_epochs"), "init": r.get("init_method")},
        })
    return recs


_ALL_DS = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees", "eggs_raw", "larvae_raw", "protozoan_raw",
           "eggs", "larvae", "protozoan"]


def _attach_tests(recs: list[dict], repo: Path, patterns: list[str]):
    by_ck = {str(Path(r["checkpoint_path"]).resolve()): r for r in recs if r.get("checkpoint_path")}
    by_key = {(r["search"], r["dataset"], r["cand_id"], r["fidelity"]): r for r in recs}
    n_attached = 0
    for pat in patterns:
        for f in sorted(glob.glob(str(repo / pat))):
            for t in read_jsonl(f):
                if "test_kappa" in t and "run" in t:  # test_svm_C100 (leitura da busca)
                    k = (Path(t["run"]).name, t.get("dataset"), t.get("cand_id"), float(t.get("fidelity", -1)))
                    r = by_key.get(k)
                    if r is not None:
                        r["kappa_test"] = t["test_kappa"]; r.setdefault("_test_src", "test_svm_C100"); n_attached += 1
                    continue
                ck = t.get("ckpt")
                if not ck or "head" not in t:
                    continue
                ck = ck if Path(ck).is_absolute() else str(repo / ck)
                r = by_ck.get(str(Path(ck).resolve()))
                if r is None:
                    continue
                if t.get("split_name") and "/pct100" not in t["split_name"]:
                    continue  # só 100% do treino
                h = t["head"] + ("_tta8" if t.get("tta") and "tta" not in t["head"] else "")
                r["kappa_by_readout"][h] = {"val": t.get("kappa_val"), "test": t.get("kappa_test")}
                n_attached += 1
    for r in recs:
        kb = {h: v for h, v in r["kappa_by_readout"].items() if v.get("val") is not None}
        if kb:  # leitura escolhida pela VALIDAÇÃO (regra do projeto); o teste dela só vai para relatório
            h = max(kb, key=lambda x: kb[x]["val"]); r["best_val_readout"] = h; r["kappa_test_best_val_readout"] = kb[h]["test"]
        if r["kappa_test"] is None and "gap1280" in r["kappa_by_readout"]:
            r["kappa_test"] = r["kappa_by_readout"]["gap1280"]["test"]; r["_test_src"] = "readout_x:gap1280"
    return n_attached


def _mark_selection(recs, fractions, bg_n, seed, focus: set):
    """top-f por κ de VALIDAÇÃO dentro de cada base (registros ok, maior fidelidade por candidato)."""
    rng = np.random.default_rng(seed)
    by_ds = defaultdict(list)
    for r in recs:
        r["focus"] = r["architecture_id"] in focus or (r.get("checkpoint_path") and r["checkpoint_path"] in focus)
        r["top_frac"] = None
        if r["kappa_val"] is not None and r.get("status") in ("ok", None):
            by_ds[r["dataset"]].append(r)
    for ds, rs in by_ds.items():
        rs.sort(key=lambda x: -x["kappa_val"])
        n = len(rs)
        for i, r in enumerate(rs):
            r["rank_val_in_dataset"] = i + 1; r["n_in_dataset"] = n
            for f in sorted(fractions):
                if i < max(1, int(np.ceil(f * n))):
                    r["top_frac"] = f; break
        rest = [r for r in rs if r["top_frac"] is None]
        for j in rng.choice(len(rest), size=min(bg_n, len(rest)), replace=False) if rest else []:
            rest[j]["background"] = True


def _missing(r):
    m = [k for k in FIELDS if r.get(k) in (None, {}, [])]
    return [k for k in m if k != "graph"]  # graph é preenchido pela etapa seguinte


def build(cfg) -> list[dict]:
    repo = Path(cfg["repo_root"]); inv = cfg["inventory"]
    recs = _modehb_records(repo, inv["modehb_glob"]) + _growth_records(repo, inv["growth_glob"])
    recs = [r for r in recs if r["dataset"] in cfg["inventory_datasets"]]
    n_t = _attach_tests(recs, repo, inv["test_jsonl"])
    fp = repo / inv["focus_list"]
    focus = {l.strip() for l in fp.read_text().splitlines() if l.strip() and not l.startswith("#")} if fp.is_file() else set()
    _mark_selection(recs, inv["top_fractions"], inv["background_per_dataset"], cfg["seed"], focus)
    for r in recs:
        r["missing"] = _missing(r)
        r["selected"] = bool(r["focus"] or r["top_frac"] is not None or r.get("background"))
    out = Path(cfg["out_dir"]) / "inventory.jsonl"
    if out.exists():
        out.rename(out.with_suffix(".jsonl.prev"))  # arquiva a versão anterior (nunca apaga)
    for r in recs:
        append_jsonl(out, r)
    write_json(Path(cfg["out_dir"]) / "inventory_summary.json", summary(recs, n_t))
    report(cfg, recs)
    return recs


def summary(recs, n_tests=0):
    s = {"n_records": len(recs), "n_test_rows_attached": n_tests, "by_dataset": {}, "missing_counts": {}}
    for r in recs:
        d = s["by_dataset"].setdefault(r["dataset"], {"n": 0, "ok": 0, "with_ckpt": 0, "with_test": 0, "focus": 0,
                                                      "top30": 0, "sources": {}})
        d["n"] += 1; d["ok"] += r.get("kappa_val") is not None; d["with_ckpt"] += bool(r.get("checkpoint_path"))
        d["with_test"] += r.get("kappa_test") is not None; d["focus"] += bool(r.get("focus"))
        d["top30"] += r.get("top_frac") is not None
        d["sources"][r["search"]] = d["sources"].get(r["search"], 0) + 1
        for m in r.get("missing", []):
            s["missing_counts"][m] = s["missing_counts"].get(m, 0) + 1
    return s


def load(cfg, selected_only=False) -> list[dict]:
    recs = read_jsonl(Path(cfg["out_dir"]) / "inventory.jsonl")
    return [r for r in recs if r.get("selected")] if selected_only else recs


def report(cfg, recs):
    s = summary(recs)
    rows = [dict(base=k, **{x: v[x] for x in ("n", "ok", "with_ckpt", "with_test", "top30", "focus")})
            for k, v in sorted(s["by_dataset"].items())]
    top = sorted([r for r in recs if r.get("focus")], key=lambda r: (r["dataset"], -(r["kappa_val"] or 0)))
    md = ["# Inventário das CNNs (GA / MO-DEHB / LEMONADE / crescimento)", "",
          f"{len(recs)} registros (busca × base × candidato × fidelidade). κ_val = validação da própria busca (seleção); "
          "κ_teste só para relatório.", "", md_table(rows, ["base", "n", "ok", "with_ckpt", "with_test", "top30", "focus"]), "",
          "## Campos ausentes (contagem)", "", md_table([{"campo": k, "n": v} for k, v in sorted(s["missing_counts"].items())], ["campo", "n"]),
          "", "## Modelos em foco (draw_io)", "",
          md_table([{"id": r["architecture_id"], "base": r["dataset"], "κ_val": r["kappa_val"], "κ_teste": r["kappa_test"],
                     "leitura(val)": r.get("best_val_readout"), "κ_teste(leitura val)": r.get("kappa_test_best_val_readout"),
                     "params": r["params"], "ckpt": "sim" if r["checkpoint_path"] else "AUSENTE"} for r in top],
                   ["id", "base", "κ_val", "κ_teste", "leitura(val)", "κ_teste(leitura val)", "params", "ckpt"])]
    Path(cfg["report_dir"], "01_inventario.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    recs = build(load_config(a.config)); print(json.dumps(summary(recs), indent=1)[:3000])
