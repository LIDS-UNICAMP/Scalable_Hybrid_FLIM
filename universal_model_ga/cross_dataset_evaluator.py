"""cross_dataset_evaluator.py -- avaliação cruzada da "rede universal" (etapa 5 da missão).

Protocolo (spifil_only, config cross_eval):
  para cada arquitetura-fonte A (genome de um candidato MO-DEHB) e cada base-alvo D das 5 naturais:
    1. BUILD: mesma topologia de A; cada bloco ajustado pelo SPiFiL SÓ com o treino de D (Evaluator._grow,
       ae_off=True, nada herdado, sem SSL, sem backprop) -- o mesmo procedimento de research/tools/spifil_twin.py,
       com o bloco `data` do config trocado pelo da busca de D (mesmo split da busca, ham10000 = split1_modehb).
    2. EVAL: research/spifil_b0b4/readout_x/eval_backbone_head.py (SVM original C=100), cabeças gap1280 +
       so2_4096_pca + so2_4096_pca_tta8 (TTA8 no teste).
    3. REPORT: matriz arquitetura x base; leitura escolhida pela VALIDAÇÃO; ranking universal por κ_val médio
       e arrependimento máximo (regret de validação). O TESTE só aparece como coluna, nunca seleciona.
Os checkpoints originais não são tocados (só lidos). Tudo é retomável (pula o que já existe).

Uso:
  python -m universal_model_ga.cross_dataset_evaluator --stage build  [--workers 4 --threads 8]
  python -m universal_model_ga.cross_dataset_evaluator --stage eval   --device cuda:2 [--shard 0/2]
  python -m universal_model_ga.cross_dataset_evaluator --stage report
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from universal_model_ga.common import append_jsonl, load_config, read_jsonl, write_json, fmt


def md_table(hdr, rows):
    return "\n".join(["| " + " | ".join(map(str, hdr)) + " |", "|" + "---|" * len(hdr)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])

READOUTS = ("gap1280", "so2_4096_pca", "so2_4096_pca_tta8")
SELECT_READOUTS = ("gap1280", "so2_4096_pca_tta8")   # a leitura de teste é escolhida pela val entre estas


# ─────────────────────────────────────────────────────────── plano
def focus_sources(cfg) -> list[dict]:
    """Arquiteturas-fonte: os candidatos MO-DEHB da lista foco (os de crescimento não têm genome -> ficam de fora)."""
    repo = Path(cfg["repo_root"])
    out, skipped = [], []
    for line in (repo / cfg["inventory"]["focus_list"]).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("/")
        if parts[0].startswith("modehb") and len(parts) == 4:
            search, ds, cand, fid = parts
            enc = repo / "artifacts" / search / ds / "state/seed42/candidates" / cand / fid / "encoder.pt"
            sid = f"{search}__{ds}__{cand}_{fid}"
            out.append({"source_id": sid, "search": search, "source_dataset": ds, "cand": cand, "fid": fid,
                        "encoder": str(enc), "exists": enc.is_file()})
        else:
            skipped.append(line)
    return out, skipped


def target_cfg_path(cfg, dataset: str) -> Path:
    """config_resolved da busca MO-DEHB da base-alvo (fonte do bloco `data`)."""
    repo = Path(cfg["repo_root"])
    for search in ("modehb_hybrid", "modehb_lemonade", "modehb_lemonade_seeded"):
        p = repo / "artifacts" / search / dataset / "state/seed42/config_resolved.yaml"
        if p.is_file():
            return p
    raise FileNotFoundError(f"sem config_resolved.yaml de busca para {dataset}")


def plan(cfg) -> list[dict]:
    srcs, _ = focus_sources(cfg)
    root = Path(cfg["repo_root"]) / cfg["out_dir"] / "cross"
    jobs = []
    for s in srcs:
        for d in cfg["datasets"]:
            jid = f"{s['source_id']}__on__{d}"
            jobs.append({**s, "target": d, "job_id": jid, "out": str(root / jid)})
    return jobs


# ─────────────────────────────────────────────────────────── build (CPU)
def build_one(job: dict, cfg: dict, threads: int, device: str = "cpu") -> dict:
    out = Path(job["out"])
    if (out / "encoder.pt").is_file():
        return {"job_id": job["job_id"], "status": "cached"}
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    repo = Path(cfg["repo_root"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from ga_search.modehb.run_search import Evaluator, _NullRun

    src = Path(job["encoder"])
    src_cfg_p = next(p / "config_resolved.yaml" for p in src.parents if (p / "config_resolved.yaml").is_file())
    tgt_cfg_p = target_cfg_path(cfg, job["target"])
    c = yaml.safe_load(src_cfg_p.read_text())
    tgt = yaml.safe_load(tgt_cfg_p.read_text())
    # só os dados mudam: topologia, SPiFiL e o resto da receita ficam os da fonte
    c["data"] = copy.deepcopy(tgt["data"])
    c["dataset_name"] = tgt.get("dataset_name", job["target"])
    c["device"] = device
    c["output_dir"] = str(out / "_eval_tmp")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()
    d = torch.load(src, map_location="cpu", weights_only=False)     # só leitura do original
    g = dict(d["genome"])
    ev = Evaluator(c, tmp / "_eval_tmp")
    enc, specs, _ = ev._grow(g, {}, 0, int(g.get("seed", c.get("seed", 42))), _NullRun(), inherit=None, ae_off=True)
    enc = enc.cpu().eval()
    with torch.no_grad():
        c_out = enc(torch.zeros(1, 3, 224, 224)).shape[1]
    n_par = sum(p.numel() for p in enc.parameters())
    # cabeça = a da fonte reinicializada (igual ao spifil_twin), 1ª conv ajustada à largura efetiva do SPiFiL
    head = copy.deepcopy(d["head"])
    torch.manual_seed(int(c.get("seed", 42)))
    for m in head.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for name, m in list(head.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            if m.in_channels != c_out:
                parent = head
                *path, last = name.split(".")
                for pp in path:
                    parent = getattr(parent, pp)
                setattr(parent, last, torch.nn.Conv2d(c_out, m.out_channels, m.kernel_size, m.stride, m.padding,
                                                      bias=m.bias is not None))
            break
    torch.save({"encoder": enc, "head": head.cpu().eval(), "genome": g, "twin_of": str(src),
                "procedure": f"rede universal: topologia de {job['source_id']}; SPiFiL por bloco só no treino de "
                             f"{job['target']}; sem AE local, sem SSL, sem backprop (ae_off=True)",
                "effective_specs": specs, "fidelity": 0.0, "cand_id": d.get("cand_id"),
                "source_dataset": job["source_dataset"], "target_dataset": job["target"]}, tmp / "encoder.pt")
    c.pop("output_dir", None)
    (tmp / "config_resolved.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    shutil.rmtree(tmp / "_eval_tmp", ignore_errors=True)
    if out.exists():
        shutil.move(str(out), str(out) + f".old_{int(time.time())}")   # arquiva, nunca apaga
    tmp.rename(out)
    return {"job_id": job["job_id"], "status": "built", "params": n_par, "c_out": c_out,
            "build_s": round(time.time() - t0, 1)}


def stage_build(cfg, args):
    jobs = [j for j in plan(cfg) if j["exists"]]
    if args.only:
        jobs = [j for j in jobs if any(o in j["job_id"] for o in args.only)]
    log = Path(cfg["repo_root"]) / cfg["out_dir"] / "cross_build.jsonl"
    todo = [j for j in jobs if not (Path(j["out"]) / "encoder.pt").is_file()]
    print(f"[build] {len(jobs)} jobs, {len(todo)} a construir, workers={args.workers} threads={args.threads}")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(build_one, j, cfg, args.threads, args.build_device): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"job_id": j["job_id"], "status": "erro", "erro": repr(e)[:500]}
            r["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
            append_jsonl(log, r)
            print("[build]", json.dumps(r, ensure_ascii=False), flush=True)


# ─────────────────────────────────────────────────────────── eval (GPU)
def stage_eval(cfg, args):
    repo = Path(cfg["repo_root"])
    jobs = [j for j in plan(cfg) if (Path(j["out"]) / "encoder.pt").is_file()]
    if args.only:
        jobs = [j for j in jobs if any(o in j["job_id"] for o in args.only)]
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        jobs = jobs[k::n]
    out = repo / cfg["out_dir"] / "cross_eval.jsonl"
    eb = repo / "research/spifil_b0b4/readout_x/eval_backbone_head.py"
    print(f"[eval] {len(jobs)} jobs -> {out} ({args.device})")
    for j in jobs:
        cmd = [sys.executable, str(eb), "--item", f"id={j['job_id']},kind=modehb,ckpt={j['out']}/encoder.pt",
               "--heads", "gap1280", "so2_4096_pca", "--tta8", "--code-root", str(repo),
               "--device", args.device, "--out", str(out),
               "--table-md", str(repo / cfg["out_dir"] / "cross_eval_tabela.md")]
        if args.max_images:
            cmd += ["--max-images", str(args.max_images)]
        print("[eval]", j["job_id"], flush=True)
        r = subprocess.run(cmd, cwd=str(repo))
        if r.returncode != 0:
            append_jsonl(repo / cfg["out_dir"] / "cross_eval_errors.jsonl",
                         {"job_id": j["job_id"], "rc": r.returncode, "t": time.strftime("%Y-%m-%d %H:%M:%S")})


# ─────────────────────────────────────────────────────────── report
def stage_report(cfg, args):
    repo = Path(cfg["repo_root"])
    rows = read_jsonl(repo / cfg["out_dir"] / "cross_eval.jsonl")
    if args.max_images is None:
        rows = [r for r in rows if r.get("max_images") is None]
    jobs = {j["job_id"]: j for j in plan(cfg)}
    cell = {}   # (source, target) -> {head: row}
    for r in rows:
        j = jobs.get(r["backbone_id"])
        if j:
            cell.setdefault((j["source_id"], j["target"]), {})[r["head"]] = r
    dsets = cfg["datasets"]
    srcs = sorted({s for s, _ in cell})
    # por célula: leitura escolhida pela VALIDAÇÃO
    pick = {}
    for key, hs in cell.items():
        cands = [hs[h] for h in SELECT_READOUTS if h in hs]
        if cands:
            b = max(cands, key=lambda r: r["kappa_val"])
            pick[key] = b
    best_val = {d: max([pick[(s, d)]["kappa_val"] for s in srcs if (s, d) in pick] or [None]) for d in dsets}
    summ = []
    for s in srcs:
        kv = [pick[(s, d)]["kappa_val"] for d in dsets if (s, d) in pick]
        kt = [pick[(s, d)]["kappa_test"] for d in dsets if (s, d) in pick]
        reg = [best_val[d] - pick[(s, d)]["kappa_val"] for d in dsets if (s, d) in pick]
        summ.append({"source_id": s, "n_bases": len(kv),
                     "mean_kappa_val": sum(kv) / len(kv) if kv else None,
                     "max_regret_val": max(reg) if reg else None,
                     "mean_kappa_test": sum(kt) / len(kt) if kt else None,
                     **{f"val_{d}": pick[(s, d)]["kappa_val"] if (s, d) in pick else None for d in dsets},
                     **{f"test_{d}": pick[(s, d)]["kappa_test"] if (s, d) in pick else None for d in dsets},
                     **{f"head_{d}": pick[(s, d)]["head"] if (s, d) in pick else None for d in dsets}})
    full = [x for x in summ if x["n_bases"] == len(dsets)]
    full.sort(key=lambda x: (x["max_regret_val"], -x["mean_kappa_val"]))
    part = [x for x in summ if x["n_bases"] < len(dsets)]
    rep = repo / cfg["report_dir"]
    rep.mkdir(parents=True, exist_ok=True)
    write_json(repo / cfg["out_dir"] / "cross_summary.json", {"full": full, "partial": part, "best_val": best_val})
    hdr = ["#", "arquitetura (fonte)", "κ_val médio", "regret máx (val)"] + [f"val {d}" for d in dsets] + ["κ_teste médio"]
    tab = [[i + 1, x["source_id"], fmt(x["mean_kappa_val"]), fmt(x["max_regret_val"])]
           + [fmt(x[f"val_{d}"]) for d in dsets] + [fmt(x["mean_kappa_test"])] for i, x in enumerate(full)]
    hdr_t = ["arquitetura (fonte)"] + [f"teste {d}" for d in dsets] + ["leituras (val)"]
    tab_t = [[x["source_id"]] + [fmt(x[f"test_{d}"]) for d in dsets]
             + [", ".join(sorted({x[f'head_{d}'] for d in dsets if x[f'head_{d}']}))] for x in full]
    md = ["# 05 — Avaliação cruzada (rede universal, só SPiFiL)", "",
          "Mesma topologia em todas as bases; SPiFiL ajustado só no treino de cada base; SVM original (C=100) por base.",
          "Sem backprop/SSL. Leitura por célula escolhida pela **validação** entre gap1280 e so2_4096_pca_tta8.",
          "Ranking: menor arrependimento máximo de validação, depois maior κ_val médio. **O teste não seleciona.**", "",
          f"Completas (5 bases): {len(full)}; parciais: {len(part)}.", "",
          "## Ranking (validação)", "", md_table(hdr, tab), "",
          "## Teste (só reportado)", "", md_table(hdr_t, tab_t), ""]
    if part:
        md += ["## Parciais (faltam bases)", "",
               md_table(["arquitetura", "bases"], [[x["source_id"], x["n_bases"]] for x in part]), ""]
    (rep / "05_cross_eval.md").write_text("\n".join(md))
    print("\n".join(md))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="universal_model_ga/config/default.yaml")
    ap.add_argument("--stage", choices=["plan", "build", "eval", "report"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--build-device", default="cpu")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None, help="k/n: só os jobs k, k+n, ... (2 GPUs = 0/2 e 1/2)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="SÓ smoke")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.stage == "plan":
        srcs, skipped = focus_sources(cfg)
        for s in srcs:
            print(("OK   " if s["exists"] else "FALTA"), s["source_id"])
        for s in skipped:
            print("FORA (sem genome):", s)
        print(len(plan(cfg)), "jobs")
    elif args.stage == "build":
        stage_build(cfg, args)
    elif args.stage == "eval":
        stage_eval(cfg, args)
    else:
        stage_report(cfg, args)


if __name__ == "__main__":
    main()
