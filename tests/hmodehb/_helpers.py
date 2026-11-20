"""Utilitarios dos testes do H-MODEHB: avaliador sintetico (sem GPU, sem dados) que se comporta como o
``HEvaluator`` no que importa para a integridade (result.json + checkpoint + sha256 + chaves)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import fitness_v2 as FV  # noqa: E402
from ga_search.hmodehb import integrity as I  # noqa: E402
from ga_search.hmodehb import run_hsearch as R  # noqa: E402
from ga_search.modehb import strata as S  # noqa: E402

SPEC = FV.spec(FV.validate_eval({"fitness_mode": "svm_kappa", "svm_fitness": {"C": 100, "scaler": "none"}}),
               val_sha256="synthetic-val", n_val=60, dataset="remote_sensing", split=1)


def search_cfg(arm="A", fids=(5, 15), brackets=2, overrides=(), base=None):
    ov = [f"search.fidelities={list(fids)}", f"search.max_brackets={brackets}", "wandb.enabled=false",
          "device=cpu", *overrides]
    cfg = R.load_hconfig(arm, "remote_sensing", 42, ov, **({"base": base} if base else {}))
    cfg["search"]["spifil_pool_sizes"] = [48] * 14
    return cfg


class FakeEval:
    """``quality`` deterministica do genoma e da origem dos pesos; escreve checkpoint e result.json
    como o ``run_search.Evaluator`` (crashes por variavel de ambiente ``H4_CRASH=job:fase``)."""

    fitness_spec = SPEC
    fitness_sig = FV.signature(SPEC)
    ruler_stamp = FV.ruler_sig(SPEC)            # a regua de cada decisao (ORIG-val)

    def __init__(self, out, oom_times=0, nan_jobs=(), fatal_jobs=(), missing_jobs=()):
        self.out = Path(out)
        self.oom_left = {}
        self.oom_times = int(oom_times)
        self.nan_jobs, self.fatal_jobs, self.missing_jobs = set(nan_jobs), set(fatal_jobs), set(missing_jobs)
        self.calls = []

    def _crash(self, k, phase):
        want = os.environ.get("H4_CRASH", "")
        if want == f"{k}:{phase}":
            os.kill(os.getpid(), signal.SIGKILL)

    def __call__(self, g, fid, ctx):
        import torch
        k = int(ctx["job_index"])
        self.calls.append((k, ctx["cand_id"], float(fid)))
        self._crash(k, "start")
        if self.oom_times:
            n = self.oom_left.setdefault(k, self.oom_times)
            if n > 0:
                self.oom_left[k] = n - 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if k in self.fatal_jobs:
            self.fatal_jobs.discard(k)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if k in self.nan_jobs:
            raise FV.NonFiniteFitness("loss SSL final nao finita: nan")
        if k in self.missing_jobs and ctx.get("inherit"):
            raise FileNotFoundError(ctx["inherit"]["checkpoint"])
        p = S.param_estimate(g)
        bonus = 0.01 if (ctx.get("weight_origin") or {}).get("kind") == "inherit" else 0.0
        kappa = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3)) + bonus) * (1 - math.exp(-float(fid) / 30))
        d = self.out / "candidates" / ctx["cand_id"] / f"f{int(fid)}"
        d.mkdir(parents=True, exist_ok=True)
        ck = d / "encoder.pt"
        ck.write_bytes(json.dumps({"g": I._plain(g), "fid": float(fid), "key": ctx["cand_key"]}, sort_keys=True).encode())
        self._crash(k, "before_result")
        res = {"status": "ok", "quality": kappa, "quality_raw": kappa, "fitness_kappa": kappa,
               "kappa_val_orig": kappa, "flops": p * 3000, "params": int(p), "gpu_hours": 0.001,
               "checkpoint": str(ck), "checkpoint_sha256": I.file_sha256(ck),
               "cand_key": ctx["cand_key"], "fitness_sig": ctx["fitness_sig"], "config_sha": ctx["config_sha"],
               "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]], "aux": {},
               "decision_rulers": [self.ruler_stamp]}
        (d / "result.json").write_text(json.dumps(res, indent=1))
        self._crash(k, "after_result")
        return res


def fingerprint(state):
    """O que tem de ser identico entre uma execucao continua e uma interrompida/retomada."""
    return [(j["job"], j["cand_id"], j["cand_key"], float(j["fidelity"]), j["role"], j["status"],
             [round(x, 12) for x in j["F"]], (j.get("op") or {}).get("type")) for j in state["jobs"]]


def digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]
