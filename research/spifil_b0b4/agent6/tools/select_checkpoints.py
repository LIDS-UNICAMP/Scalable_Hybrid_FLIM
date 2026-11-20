"""Melhor checkpoint por VALIDACAO de cada (dataset, kind, estrato B0–B4).

FONTE DE VERDADE = ``search_state.json`` (auditoria CHECK C4): so entram jobs com
``status == "ok"`` no search_state E ``result.status == "ok"``, com ``encoder.pt`` e
``result.json`` (status ok) presentes.  Um job ``failed`` no search_state (ex.: coconut hybrid
B4 c00012, ``params 3541364 > cap 3500000``) nunca e escolhido, mesmo com ``result.json`` ok.

Criterio: maior ``fitness_kappa`` (kappa de VALIDACAO da propria busca, regua cov_head).
Desempate DECLARADO (CHECK, nao-bloqueante): (1) maior kappa; (2) MAIOR fidelidade (mais
treino); (3) menor ``cand_id``; (4) menor numero do job.  Nunca le teste.
Estratos da missao por ``params`` (encoder + cabeca): B0 <=300k, B1 <=800k, B2 <=1,5M,
B3 <=3M, B4 <=5,5M.

Uso como modulo: ``candidates(art, cutoff=None)`` devolve todos os candidatos validos (com
``job_time``), ``select(cands)`` o melhor por estrato.  CLI: ``python select_checkpoints.py
<artifacts> <out.csv>`` (para congelar use ``freeze_E0_E8.py``).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

KINDS = ("modehb_hybrid", "modehb_lemonade")
BOUNDS = (("B0", 3e5), ("B1", 8e5), ("B2", 1.5e6), ("B3", 3e6), ("B4", 5.5e6))
FIELDS = ["dataset", "kind", "stratum", "cand", "fid", "params", "flops_mac", "n_blocks",
          "kappa_val_search", "search_ruler", "wise_alpha", "job", "job_time", "checkpoint",
          "config_resolved", "md5"]


def stratum(p):
    for name, ub in BOUNDS:
        if p <= ub:
            return name
    return None


def md5(path, bs=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def _fid_num(fid: str) -> float:
    return float(str(fid).lstrip("f"))


def rank_key(r: dict):
    """Ordenacao total: maior kappa, maior fidelidade, menor cand, menor job."""
    return (-float(r["kappa_val_search"]), -_fid_num(r["fid"]), r["cand"], int(r.get("job") or 0))


def candidates(art: Path, cutoff: float | None = None) -> list[dict]:
    """Todos os jobs ok (search_state E result.json) com checkpoint; ``cutoff`` = epoch maximo
    do job (congelamento)."""
    out = []
    root = art.parent
    for kind in KINDS:
        for ssp in sorted((art / kind).glob("*/state/seed42/search_state.json")):
            state = ssp.parent
            ds = state.parts[-3]
            cfgp = state / "config_resolved.yaml"
            if not cfgp.exists():
                continue
            cfg = yaml.safe_load(cfgp.read_text())
            sf = (cfg.get("eval") or {}).get("svm_fitness") or {}
            ruler = f"{sf.get('readout')}|C={sf.get('C')}|scaler={sf.get('scaler')}|tta={sf.get('tta')}"
            for j in json.loads(ssp.read_text()).get("jobs", []):
                r = j.get("result") or {}
                if j.get("status") != "ok" or r.get("status", "ok") != "ok":
                    continue
                if r.get("fitness_kappa") is None or r.get("params") is None or not r.get("checkpoint"):
                    continue
                if cutoff is not None and float(j.get("time") or 0) > cutoff:
                    continue
                ck = root / r["checkpoint"]
                rj = ck.parent / "result.json"
                if not ck.exists() or not rj.exists():
                    continue
                try:
                    if json.loads(rj.read_text()).get("status") != "ok":
                        continue
                except Exception:
                    continue
                s = stratum(float(r["params"]))
                if s is None:
                    continue
                out.append({"dataset": ds, "kind": kind, "stratum": s, "cand": ck.parts[-3],
                            "fid": ck.parts[-2], "params": int(r["params"]), "flops_mac": r.get("flops"),
                            "n_blocks": len(r.get("effective_specs") or []),
                            "kappa_val_search": float(r["fitness_kappa"]), "search_ruler": ruler,
                            "wise_alpha": r.get("wise_alpha"), "job": j.get("job"),
                            "job_time": j.get("time"), "checkpoint": r["checkpoint"],
                            "config_resolved": str(cfgp.relative_to(root)), "md5": None})
    return out


def job_status(art: Path, checkpoint: str) -> tuple[str | None, str | None]:
    """(status, erro) do job do search_state cujo result aponta para ``checkpoint``."""
    ck = Path(checkpoint)
    ssp = art.parent / Path(*ck.parts[:5]) / "search_state.json"   # artifacts/<kind>/<ds>/state/seed42
    if not ssp.exists():
        return None, "search_state ausente"
    found = None
    for j in json.loads(ssp.read_text()).get("jobs", []):
        r = j.get("result") or {}
        cand, fid = ck.parts[-3], _fid_num(ck.parts[-2])
        if r.get("checkpoint") == checkpoint or (j.get("cand_id") == cand and float(j.get("fidelity") or -1) == fid):
            found = (j.get("status"), r.get("error"))
            if j.get("status") == "ok":
                return found
    return found or (None, "job nao encontrado")


def select(cands: list[dict]) -> list[dict]:
    best: dict = {}
    for r in cands:
        key = (r["dataset"], r["kind"], r["stratum"])
        if key not in best or rank_key(r) < rank_key(best[key]):
            best[key] = r
    return [best[k] for k in sorted(best)]


def write_csv(rows: list[dict], path: Path, root: Path) -> None:
    for r in rows:
        r["md5"] = r.get("md5") or md5(root / r["checkpoint"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    art = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "checkpoints_E0_E8.csv")
    rows = select(candidates(art))
    write_csv(rows, out, art.parent)
    print(f"{len(rows)} checkpoints -> {out}")
