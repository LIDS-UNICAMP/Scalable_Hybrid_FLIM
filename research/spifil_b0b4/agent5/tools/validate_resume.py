#!/usr/bin/env python
"""Valida o RESUME de uma busca MO-DEHB/LEMONADE migrada (staging) SEM treinar e SEM GPU.

Uso: python validate_resume.py --src artifacts/_staging_gpu3/<kind>/<base> --kind <kind> --base <base>
         [--seed 42] [--json-out arquivo.json]

Checagens (todas só leitura sobre --src; o que precisa escrever vai para um diretório temporário):
  1. search_state.json carrega; nº de jobs, status, brackets, último job;
  2. manifesto (gpu3_migration_manifest.csv / conteúdo do marcador .ok) -> nº de jobs (e sha256 se houver);
  3. checkpoints dos jobs ok existem; torch.load em CPU (weights_only=False) de ≤5 amostras + TODOS os de
     fidelidade máxima; encoder presente;
  4. configs: yaml vivo da Jaci (search_v3*.yaml + datasets/<base>.yaml) reconstruído com os mesmos --set e
     comparado com state["config"] (seção search) e config_resolved.yaml (eval/train/data);
  5. dataset: imagens/split existem; sha256 do split = split_sha256; identidade de dados (hash de TODAS as
     imagens de treino/val, o mesmo do Evaluator) = data_sha256 gravado nos jobs; spifil_pool_sizes iguais;
  6. semente externa (buscas semeadas): arquivos presentes na Jaci; sha do encoder convertido = o do staging;
  7. controller: assinatura do fitness (C, kappa_weight) e id do controller W&B que a Jaci vai usar;
  8. REPLAY a seco: MODEHB.run(resume=True) com avaliador-stub sobre uma CÓPIA do state; tem de reproduzir
     todos os jobs (sem ReplayMismatch) e parar exatamente no próximo job novo.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU only (regra: sem GPU)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("MODEHB_THREADS", "2")
os.environ["WANDB_MODE"] = "disabled"

import argparse
import copy
import csv
import functools
import glob
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
AG5 = REPO / "research/spifil_b0b4/agent5"
SCRATCH = Path(os.environ.get("AG5_SCRATCH", "/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/"
                              "05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad"))
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"

import torch  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(2)

KIND_CONFIG = {
    "modehb_hybrid": "ga_search/config/modehb/search_v3.yaml",
    "modehb_lemonade": "ga_search/config/modehb/search_v3_lemonade.yaml",
    "modehb_seeded": "ga_search/config/modehb/search_v3_seeded.yaml",
    "modehb_lemonade_seeded": "ga_search/config/modehb/search_v3_lemonade_seeded.yaml",
    "pnas_cs": "ga_search/config/modehb/pnas_cs.yaml",
}
KIND_ALGO = {"modehb_hybrid": "MO-DEHB hybrid (SPiFiL)", "modehb_lemonade": "LEMONADE",
             "modehb_seeded": "MO-DEHB hybrid SEMEADA (ssl_pequeno_lejepa)",
             "modehb_lemonade_seeded": "LEMONADE SEMEADA (ssl_pequeno_lejepa)", "pnas_cs": "PNAS-CS"}
# chaves de state["config"] que dependem do host/arquivo e são checadas à parte
IGNORE_SEARCH_KEYS = {"spifil_pool_sizes", "external_seed"}


class StopReplay(BaseException):
    """Levantada pelo avaliador-stub no 1º job NÃO registrado (BaseException: o _call do MODEHB
    captura Exception como candidato infeasible)."""

    def __init__(self, genome, fid, ctx):
        super().__init__("stop")
        self.genome, self.fid, self.ctx = genome, fid, ctx


def sha256_file(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def diff_dict(a, b, prefix="", out=None, ignore=()):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if prefix == "" and k in ignore:
                continue
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a:
                out.append((p, "<ausente>", b[k]))
            elif k not in b:
                out.append((p, a[k], "<ausente>"))
            else:
                diff_dict(a[k], b[k], p, out)
    else:
        na = json.loads(json.dumps(a, default=str))
        nb = json.loads(json.dumps(b, default=str))
        if isinstance(na, list) and isinstance(nb, list) and len(na) == len(nb):
            na = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in na]
            nb = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in nb]
        if isinstance(na, (int, float)) and isinstance(nb, (int, float)) and not isinstance(na, bool):
            if float(na) != float(nb):
                out.append((prefix, a, b))
        elif na != nb:
            out.append((prefix, a, b))
    return out


def find_manifest(kind, base, seed, src, marker=None):
    """Procura o nº de jobs esperado (e sha256 do search_state) em: marcador .ok, manifest no staging,
    gpu3_migration_manifest.csv (agent5)."""
    found = {"sources": []}
    marker = Path(marker) if marker else REPO / f"artifacts/_staging_gpu3/{kind}_{base}.ok"
    cands = [marker] + sorted(Path(src).glob("*manifest*")) + sorted(Path(src).glob("MANIFEST*")) + [
        REPO / "artifacts/_staging_gpu3/gpu3_migration_manifest.csv", AG5 / "gpu3_migration_manifest.csv"]
    for p in cands:
        if not p.exists() or p.is_dir():
            continue
        txt = p.read_text(errors="replace")
        found["sources"].append(str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p))
        # linhas tipo sha256sum
        for m in re.finditer(r"^([0-9a-f]{64})\s+\*?(\S+)\s*$", txt, re.M):
            found.setdefault("sha256", {})[m.group(2)] = m.group(1)
        # key=value / json
        try:
            js = json.loads(txt)
            if isinstance(js, dict):
                for k in ("n_jobs", "jobs", "njobs"):
                    if k in js:
                        found["n_jobs"] = int(js[k])
                for k, v in js.items():
                    if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                        found.setdefault("sha256", {})[k] = v
        except Exception:
            pass
        m = re.search(r"\bn_?jobs\s*[=:]\s*(\d+)", txt)
        if m and "n_jobs" not in found:
            found["n_jobs"] = int(m.group(1))
        if p.suffix == ".csv":
            try:
                rows = list(csv.DictReader(txt.splitlines()))
            except Exception:
                rows = []
            for r in rows:
                vals = " ".join(str(v) for v in r.values())
                if "kind" in r and "base" in r:          # manifesto do athena: casamento exato
                    if str(r["base"]).strip() != base or str(r["kind"]).strip() not in (kind, f"{kind}_seed{seed}"):
                        continue
                    if str(r.get("seed", "")).strip() not in ("", f"seed{seed}", str(seed)):
                        continue
                else:
                    if base not in [str(v).strip() for v in r.values()] and f"/{base}" not in vals:
                        continue
                    if kind not in vals and kind.replace("modehb_", "") not in vals:
                        continue
                found["csv_row"] = r
                for k, v in r.items():
                    kl = (k or "").lower()
                    if kl in ("md5_search_state_jaci", "md5_search_state_athena") and re.fullmatch(r"[0-9a-f]{32}", str(v).strip()):
                        found.setdefault("md5", {})[k] = str(v).strip()
                    if kl in ("n_jobs", "jobs", "njobs", "num_jobs", "n_jobs_state", "jobs_jaci", "jobs_athena") \
                            and str(v).strip().isdigit():
                        found["n_jobs"] = int(v)
                    if "sha" in kl and re.fullmatch(r"[0-9a-f]{64}", str(v).strip()):
                        found.setdefault("sha256", {})[k] = str(v).strip()
    return found


def validate(src, kind, base, seed, n_sample=5, do_replay=True, do_data=True, marker=None, budget_override=None, cap_override=None):
    t0 = time.time()
    src = Path(src)
    rel_live = f"artifacts/{kind}/{base}"
    sd = src / "state" / f"seed{seed}"
    R = {"kind": kind, "base": base, "seed": seed, "src": str(src), "dest": rel_live,
         "state_dir_dest": f"{rel_live}/state/seed{seed}", "checks": {}, "errors": [], "warnings": [],
         "validated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    C = R["checks"]

    def err(m):
        R["errors"].append(m)

    def warn(m):
        R["warnings"].append(m)

    # ---- 1. state
    sp = sd / "search_state.json"
    try:
        st = json.loads(sp.read_text())
        jobs = st["jobs"]
        stat = {}
        for j in jobs:
            stat[j["status"]] = stat.get(j["status"], 0) + 1
        fids = sorted({float(j["fidelity"]) for j in jobs})
        last = jobs[-1] if jobs else {}
        C["state"] = {"ok": True, "n_jobs": len(jobs), "status": stat, "done": st.get("done"),
                      "brackets_done": len(st.get("brackets_log") or []), "bracket": st.get("bracket"),
                      "gpu_hours": round(float(st.get("gpu_hours", 0)), 2),
                      "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.get("updated", 0))),
                      "fidelities_cfg": st["config"].get("fidelities"), "max_brackets": st["config"].get("max_brackets"),
                      "last_job": {"job": last.get("job"), "cand_id": last.get("cand_id"),
                                   "fidelity": last.get("fidelity"), "status": last.get("status"),
                                   "role": last.get("role")},
                      "sha256": sha256_file(sp)}
        failed = [(j["job"], j["cand_id"], j["fidelity"], str((j.get("result") or {}).get("error", ""))[:160])
                  for j in jobs if j["status"] != "ok"]
        C["state"]["failed_jobs"] = failed
        if any("params" in f[3] and "> cap" in f[3] for f in failed):
            pass   # infeasível por teto de parâmetros: comportamento esperado da busca
    except Exception as exc:
        C["state"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        err(f"search_state.json não carrega: {exc}")
        R["verdict"] = "BLOQUEADA"
        return R

    # ---- 2. manifesto
    man = find_manifest(kind, base, seed, src, marker)
    C["manifest"] = man
    if "n_jobs" in man:
        if int(man["n_jobs"]) != len(jobs):
            err(f"nº de jobs do state ({len(jobs)}) != manifesto ({man['n_jobs']})")
    else:
        warn("manifesto sem nº de jobs (marcador/CSV não informam): conferir com o athena")
    for k, h in (man.get("sha256") or {}).items():
        if "search_state" in k:
            if h != C["state"]["sha256"]:
                err(f"sha256 do search_state.json difere do manifesto ({k})")
            else:
                C["manifest"]["state_sha_ok"] = True
    md5_state = hashlib.md5(sp.read_bytes()).hexdigest()
    C["state"]["md5"] = md5_state
    for k, h in (man.get("md5") or {}).items():
        if h != md5_state:
            err(f"md5 do search_state.json ({md5_state}) != manifesto {k} ({h})")
        else:
            C["manifest"]["state_md5_ok"] = True
    # demais hashes de arquivos críticos listados (sha256sum)
    hv = {"ok": 0, "bad": [], "missing": []}
    for k, h in (man.get("sha256") or {}).items():
        if "/" not in k and "." not in k:
            continue
        cand = [src / k, REPO / k, sd / Path(k).name] if not k.startswith("/") else [Path(k)]
        if k.startswith(rel_live + "/"):
            cand.insert(0, src / k[len(rel_live) + 1:])
        p = next((c for c in cand if c.exists()), None)
        if p is None:
            hv["missing"].append(k)
        elif sha256_file(p) != h:
            hv["bad"].append(k)
        else:
            hv["ok"] += 1
    if hv["ok"] or hv["bad"] or hv["missing"]:
        C["manifest_hashes"] = hv
        if hv["bad"]:
            err(f"hash diferente em {len(hv['bad'])} arquivo(s) do manifesto: {hv['bad'][:5]}")
        if hv["missing"]:
            warn(f"{len(hv['missing'])} arquivo(s) do manifesto não achados no staging: {hv['missing'][:5]}")

    # ---- 3. checkpoints
    def map_path(p):
        p = str(p)
        if p.startswith(rel_live + "/"):
            return src / p[len(rel_live) + 1:]
        if "/" + rel_live + "/" in p:          # absoluto do athena
            return src / p.split("/" + rel_live + "/", 1)[1]
        return Path(p)

    ok_jobs = [j for j in jobs if j["status"] == "ok" and (j.get("result") or {}).get("checkpoint")]
    missing = [j["result"]["checkpoint"] for j in ok_jobs if not map_path(j["result"]["checkpoint"]).exists()]
    emb_missing = [j["result"]["embedding_path"] for j in ok_jobs
                   if j["result"].get("embedding_path") and not map_path(j["result"]["embedding_path"]).exists()]
    absolute = [j["result"]["checkpoint"] for j in ok_jobs if str(j["result"]["checkpoint"]).startswith("/")]
    fmax = max((float(j["fidelity"]) for j in ok_jobs), default=None)
    top = [j for j in ok_jobs if fmax is not None and float(j["fidelity"]) == fmax]
    rng = random.Random(f"{kind}/{base}/{seed}")
    rest = [j for j in ok_jobs if j not in top]
    sample = rng.sample(rest, min(n_sample, len(rest))) + top
    loaded, bad = [], []
    for j in sample:
        p = map_path(j["result"]["checkpoint"])
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            enc = ck.get("encoder") if isinstance(ck, dict) else None
            nb = len(enc) if enc is not None and hasattr(enc, "__len__") else None
            if enc is None or not nb:
                bad.append((str(p), "sem encoder"))
            else:
                loaded.append({"job": j["job"], "cand_id": j["cand_id"], "fid": j["fidelity"], "blocks": nb,
                               "MB": round(p.stat().st_size / 1e6, 1)})
        except Exception as exc:
            bad.append((str(p), f"{type(exc).__name__}: {str(exc)[:120]}"))
    C["checkpoints"] = {"ok_jobs": len(ok_jobs), "missing": len(missing), "missing_examples": missing[:5],
                        "embeddings_missing": len(emb_missing), "absolute_paths": len(absolute),
                        "max_fidelity": fmax, "n_max_fidelity": len(top), "loaded": len(loaded),
                        "load_failed": bad, "loaded_detail": loaded}
    if missing:
        err(f"{len(missing)} checkpoint(s) de jobs ok ausentes no staging (ex.: {missing[:2]})")
    if emb_missing:
        warn(f"{len(emb_missing)} val_embedding.pt ausentes")
    if absolute:
        warn(f"{len(absolute)} checkpoints com caminho ABSOLUTO no state (o resume na Jaci pode não achar)")
    if bad:
        err(f"{len(bad)} checkpoint(s) não carregam em CPU: {bad[:2]}")

    # ---- 4. configs
    cfg_path = KIND_CONFIG.get(kind)
    resolved_p = sd / "config_resolved.yaml"
    resolved = yaml.safe_load(resolved_p.read_text()) if resolved_p.exists() else None
    if resolved is None:
        warn("config_resolved.yaml ausente no staging")
    if cfg_path is None or not (REPO / cfg_path).exists():
        err(f"config da Jaci para kind={kind} não existe ({cfg_path})")
        R["verdict"] = "BLOQUEADA"
        return R
    ds_yaml = REPO / f"ga_search/config/modehb/datasets/{base}.yaml"
    if not ds_yaml.exists():
        err(f"dataset yaml ausente na Jaci: {ds_yaml.relative_to(REPO)}")
        R["verdict"] = "BLOQUEADA"
        return R
    scfg = st["config"]
    budget = ((scfg.get("stratified") or {}).get("memory_budget_gb"))
    cap = (resolved or {}).get("gpu_process_cap_gb")
    if budget_override is not None:
        budget = budget_override
        R["what_if_budget"] = budget_override
    if cap_override is not None:
        cap = cap_override
    sets = [f"output_dir={rel_live}/state/seed{seed}"]
    if budget is not None:
        sets.append(f"search.stratified.memory_budget_gb={budget:g}" if isinstance(budget, float)
                    else f"search.stratified.memory_budget_gb={budget}")
    if cap is not None:
        sets.append(f"gpu_process_cap_gb={cap}")
    from ga_search.modehb import run_search as RS
    from ga_search.modehb import modehb as M
    cfg = RS.load_config(cfg_path, base, seed, sets)
    cfg["device"] = "cpu"
    R["run_cmd"] = {"config": cfg_path, "dataset": base, "seed": seed, "sets": sets}
    d_search = diff_dict(scfg, {**M.DEFAULTS, **cfg["search"]}, ignore=IGNORE_SEARCH_KEYS)
    d_search = [x for x in d_search if x[0] not in ("seed",)]
    C["config_search_diff"] = [(p, str(a)[:120], str(b)[:120]) for p, a, b in d_search]
    if d_search:
        err(f"seção search do state difere do yaml vivo da Jaci em {len(d_search)} chave(s): "
            f"{[x[0] for x in d_search][:8]}")
    if resolved is not None:
        dr = []
        for sec in ("eval", "train", "ssl", "spifil", "data"):
            if sec in resolved or sec in cfg:
                dr += diff_dict(resolved.get(sec), cfg.get(sec), prefix=sec)
        # caminhos de dados podem diferir só pela raiz; registrar mas não bloquear se forem paths
        dr_paths = [x for x in dr if any(t in x[0] for t in ("root", "dir", "json", "csv", "path"))]
        dr_other = [x for x in dr if x not in dr_paths]
        C["config_resolved_diff"] = [(p, str(a)[:120], str(b)[:120]) for p, a, b in dr]
        if dr_other:
            warn(f"config_resolved (athena, 1º start) difere do yaml vivo em {[x[0] for x in dr_other][:8]} "
                 "(o resume usa o yaml VIVO; o REPLAY abaixo é o juiz)")
        if dr_paths:
            warn(f"caminhos diferentes no config_resolved: {[x[0] for x in dr_paths][:5]}")

    # ---- 7. controller / fitness
    sv = (cfg.get("eval") or {}).get("svm_fitness") or {}
    fit_sig = f"C{float(sv.get('C', 100.0)):g}_kw{float(sv.get('kappa_weight', 0.75)):g}"
    sv_r = ((resolved or {}).get("eval") or {}).get("svm_fitness") or {}
    fit_sig_r = f"C{float(sv_r.get('C', 100.0)):g}_kw{float(sv_r.get('kappa_weight', 0.75)):g}" if resolved else None
    key = str((REPO / rel_live / "state" / f"seed{seed}").resolve()) + ("" if fit_sig == "C0.1_kw0.75" else "|" + fit_sig)
    ctrl = "ctrl-" + hashlib.sha1(key.encode()).hexdigest()[:16]
    modes = sorted({str((j.get("result") or {}).get("fitness_mode")) for j in ok_jobs})
    athena_ctrl = set()
    for lp in list(src.glob("logs/*.log")) + list(sd.glob("*.log")):
        try:
            athena_ctrl |= set(re.findall(r"ctrl-[0-9a-f]{16}", lp.read_text(errors="replace")))
        except Exception:
            pass
    C["controller"] = {"fit_sig_jaci": fit_sig, "fit_sig_resolved": fit_sig_r, "controller_id_jaci": ctrl,
                       "controller_ids_in_logs": sorted(athena_ctrl), "fitness_modes_jobs": modes}
    if fit_sig != "C100_kw1":
        err(f"assinatura do fitness na Jaci = {fit_sig} (esperado C100_kw1)")
    if fit_sig_r and fit_sig_r != fit_sig:
        err(f"assinatura do fitness do athena ({fit_sig_r}) != Jaci ({fit_sig}): misturaria curvas/jobs")
    if len(modes) > 1:
        err(f"jobs com fitness_mode misturado: {modes}")
    if ctrl in athena_ctrl:
        C["controller"]["note"] = "mesmo id do athena (o resolve() coincide)"
    else:
        C["controller"]["note"] = ("id NOVO na Jaci (esperado: o artifacts do athena é symlink, o resolve() muda); "
                                   "o state/jobs continuam; só o painel W&B do controller recomeça")

    # ---- 5. dataset
    ds = yaml.safe_load(ds_yaml.read_text())
    root = ds.get("root", "")
    sj = str(ds.get("split_json", "")).format(root=root, split=ds.get("split", 1))
    imgs = str(ds.get("images_dir", "")).format(root=root)
    dsc = {"yaml": str(ds_yaml.relative_to(REPO)), "root_exists": Path(root).exists(),
           "images_dir": imgs, "images_dir_exists": Path(imgs).exists(), "split_json": sj,
           "split_json_exists": Path(sj).exists()}
    if Path(sj).exists() and ds.get("split_sha256"):
        dsc["split_sha256_ok"] = sha256_file(sj) == ds["split_sha256"]
        if not dsc["split_sha256_ok"]:
            err(f"sha256 do split {sj} != split_sha256 do yaml")
    for k in ("root_exists", "images_dir_exists", "split_json_exists"):
        if not dsc[k]:
            err(f"dataset {base}: {k} = False")
    job_sha = sorted({(j.get("result") or {}).get("data_sha256") for j in ok_jobs} - {None})
    dsc["data_sha256_jobs"] = job_sha
    tmp = Path(tempfile.mkdtemp(prefix=f"resumeval_{kind}_{base}_s{seed}_", dir=SCRATCH))
    pool_sizes = None
    if do_data:
        try:
            from ga_search.progressive_data import load_data
            d = dict(cfg["data"])
            d.setdefault("seed", int(cfg["seed"]))
            d["num_workers"] = 0
            tl, vl, selected, identity = load_data(
                {k: v for k, v in d.items() if k not in ("ram_cache", "val_max_samples", "drop_last")}, tmp / "data")
            choices = cfg.get("search", {}).get("spifil_images_choices") or RS.SPIFIL_IMAGES_CHOICES
            pool = RS.build_spifil_pool(selected, tl.dataset.samples, max(int(k) for k in choices), int(d["seed"]))
            pool_sizes = [len(v) for _, v in sorted(pool.items())]
            dsc.update({"identity_sha256_jaci": identity["sha256"], "n_train": len(tl.dataset),
                        "n_val": len(vl.dataset), "spifil_pool_sizes_jaci": pool_sizes,
                        "spifil_pool_sizes_state": scfg.get("spifil_pool_sizes")})
            if job_sha and identity["sha256"] not in job_sha:
                err(f"identidade dos dados na Jaci ({identity['sha256'][:12]}) != data_sha256 dos jobs "
                    f"({[s[:12] for s in job_sha]}): imagens/split diferentes do athena")
            if len(job_sha) > 1:
                warn(f"jobs com mais de um data_sha256: {[s[:12] for s in job_sha]}")
            if scfg.get("spifil_pool_sizes") is not None and list(scfg["spifil_pool_sizes"]) != pool_sizes:
                err(f"spifil_pool_sizes Jaci {pool_sizes} != state {scfg['spifil_pool_sizes']}")
        except Exception as exc:
            err(f"load_data na Jaci falhou: {type(exc).__name__}: {str(exc)[:200]}")
    C["dataset"] = dsc
    cfg["search"]["spifil_pool_sizes"] = pool_sizes if pool_sizes is not None else scfg.get("spifil_pool_sizes")

    # ---- 6. semente externa
    ec = cfg["search"].get("external_seed")
    if ec:
        es = {"name": ec.get("name"), "arch": ec.get("arch"), "checkpoint": ec.get("checkpoint")}
        es["arch_exists"] = Path(ec["arch"]).exists()
        es["checkpoint_exists"] = Path(ec["checkpoint"]).exists()
        es["checkpoint_realpath"] = os.path.realpath(ec["checkpoint"])
        from ga_search.modehb.external_seed import prepare
        try:
            prep = prepare(copy.deepcopy(ec), tmp / "state")
            es["sha_jaci"] = prep.get("sha256")
            es["missing"] = prep.get("missing")
            cfg["search"]["external_seed"] = prep
        except Exception as exc:
            err(f"semente externa não converte na Jaci: {exc}")
            prep = None
        staged = sd / "external_seed" / f"{ec.get('name')}.pt"
        if staged.exists():
            try:
                es["sha_staging"] = torch.load(staged, map_location="cpu", weights_only=False).get("sha256")
            except Exception as exc:
                es["sha_staging"] = f"erro: {exc}"
        es["sha_state_config"] = (scfg.get("external_seed") or {}).get("sha256")
        ref = es.get("sha_staging") or es.get("sha_state_config")
        if not (es["arch_exists"] and es["checkpoint_exists"]):
            err("semente externa AUSENTE na Jaci (a busca rodaria sem semente: inconsistente com o athena)")
        elif ref and es.get("sha_jaci") != ref:
            err(f"semente externa na Jaci (sha {es.get('sha_jaci')}) != a usada no athena ({ref}): os symlinks "
                "lejepa da Jaci apontam para outro L6 (ex.: L6_kappa); repor o encoder do athena no staging/"
                "external_seed ou decidir com o coordenador")
        elif not ref:
            warn("sem sha de referência da semente no staging (external_seed/*.pt ausente)")
        C["external_seed"] = es

    # ---- 8. replay a seco
    if do_replay:
        (tmp / "state").mkdir(exist_ok=True)
        shutil.copy2(sp, tmp / "state" / "search_state.json")

        def stub(genome, fid, ctx):
            raise StopReplay(genome, fid, ctx)

        spifil_repair = None
        if cfg.get("spifil", {}).get("spifil_ops", False):
            from ga_search.modehb.init import repair_for_spifil
            spifil_repair = functools.partial(repair_for_spifil, spifil_ops=True)
        rp = {}
        try:
            search = M.MODEHB(cfg["search"], stub, tmp / "state", logger=None, spifil_repair=spifil_repair)
            t1 = time.time()
            try:
                search.run(resume=True)
                rp = {"ok": True, "outcome": "busca termina sem jobs novos (done)", "replayed": len(search.state["jobs"])}
            except StopReplay as s:
                inh = (s.ctx or {}).get("inherit")
                sj = search.state["jobs"]
                same = all(a["arch_hash"] == b["arch_hash"] and float(a["fidelity"]) == float(b["fidelity"])
                           and a["role"] == b["role"] for a, b in zip(sj, jobs))
                extra = sj[len(jobs):]
                rp_ok = same and len(sj) >= len(jobs) and all(j.get("cached") for j in extra)
                rp = {"ok": rp_ok, "replayed": len(sj), "logged_identical": same,
                      "extra_cached_jobs": [(j["job"], j["cand_id"], j["fidelity"]) for j in extra],
                      "next_job": len(search.state["jobs"]), "next_fidelity": float(s.fid),
                      "next_role": (s.ctx or {}).get("role"), "next_cand": (s.ctx or {}).get("cand_id"),
                      "next_inherit_ckpt": inh.get("checkpoint") if inh else None,
                      "outcome": "replay reproduziu todos os jobs e parou no próximo job novo"}
                if inh and inh.get("checkpoint"):
                    ip = map_path(inh["checkpoint"])
                    rp["next_inherit_ckpt_exists"] = ip.exists()
                    if not ip.exists():
                        err(f"o próximo job herda de {inh['checkpoint']}, ausente no staging")
            rp["seconds"] = round(time.time() - t1, 1)
            if not rp["ok"]:
                err(f"replay inconsistente: parou no job {rp.get('replayed')} (registrados {len(jobs)}; "
                    f"extras não-cached ou jobs diferentes)")
        except M.ReplayMismatch as exc:
            rp = {"ok": False, "outcome": f"ReplayMismatch: {exc}"}
            err(f"REPLAY divergiu na Jaci: {exc}")
        except Exception as exc:
            rp = {"ok": False, "outcome": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1500:]}
            err(f"replay falhou: {type(exc).__name__}: {str(exc)[:200]}")
        C["replay"] = rp

    # ---- recursos
    peaks = [float((j.get("result") or {}).get("peak_gpu_reserved_gb") or (j.get("result") or {}).get("peak_gpu_mem_gb") or 0)
             for j in ok_jobs]
    capv = float(cap) if cap is not None else (float(budget) + 4 if budget is not None else 14.0)
    R["resources"] = {"memory_budget_gb": budget, "gpu_process_cap_gb": cap,
                      "peak_gpu_reserved_gb_max": round(max(peaks), 2) if peaks else None,
                      # gate = pico real observado + 8 GB (limitado a cap + 4), mínimo 22 GB (padrão dos migr_*)
                      "gate_free_mib": int(max(22.0, min(capv + 4.0, (max(peaks) if peaks else capv) + 8.0)) * 1024)}
    R["tmpdir"] = str(tmp)
    R["seconds"] = round(time.time() - t0, 1)
    R["verdict"] = "VALIDADA" if not R["errors"] else "BLOQUEADA"
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: todos os state/seed*")
    ap.add_argument("--json-out")
    ap.add_argument("--marker")
    ap.add_argument("--budget", type=float, help="what-if: outro memory_budget_gb (testa se o replay ainda bate)")
    ap.add_argument("--cap", type=float)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--no-data", action="store_true")
    a = ap.parse_args()
    seeds = [a.seed] if a.seed is not None else sorted(
        int(Path(p).name[4:]) for p in glob.glob(f"{a.src}/state/seed*") if Path(p).name[4:].isdigit())
    out = []
    for s in seeds:
        try:
            r = validate(a.src, a.kind, a.base, s, do_replay=not a.no_replay, do_data=not a.no_data, marker=a.marker, budget_override=a.budget, cap_override=a.cap)
        except Exception as exc:
            r = {"kind": a.kind, "base": a.base, "seed": s, "verdict": "BLOQUEADA",
                 "errors": [f"validador quebrou: {type(exc).__name__}: {exc}"], "tb": traceback.format_exc()[-2000:]}
        out.append(r)
        print(json.dumps({k: r.get(k) for k in ("kind", "base", "seed", "verdict", "errors", "warnings")},
                         ensure_ascii=False), flush=True)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
