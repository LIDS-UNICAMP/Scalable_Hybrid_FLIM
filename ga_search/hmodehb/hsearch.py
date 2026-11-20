"""Controller do H-MODEHB: subclasse de ``modehb.MODEHB`` com a camada de integridade.

O que muda em relacao ao MODEHB (que continua intocado e e reusado por heranca):

* faixas B0-B4 fixas e teto duro de 5,5M (``bk.py``); pre-checagem estrutural antes de gastar GPU;
* fitness v2 (``fitness_v2.py``): F = [1 - kappa_val_ORIG, MACs log-normalizados]; ``fitness_sig``
  gravada no estado e em cada job; o resume recusa outra regua (``FitnessSigMismatch``);
* ``cand_key`` = genoma + origem dos pesos (ancestral + sha256 do checkpoint) + fidelidade +
  assinatura. O cache e o resume de promocao usam essa chave (``lineage_key`` sem a fidelidade),
  nunca so o ``arch_hash``: um filho herdado com a mesma arquitetura de um candidato ja avaliado nao
  recebe o resultado em cache (a heranca nao e descartada);
* ``config_sha`` do config completo e ``config_full`` no estado; deriva no resume -> ``ConfigDrift``
  (salvo ``allow_drift``, registrado em ``drift_log``);
* trava ``flock`` contra dois controladores no mesmo diretorio;
* adocao, no resume, de um resultado ja gravado (``hjobs/<job>.json`` do controller ou
  ``candidates/<id>/f<F>/result.json`` do avaliador) com a mesma ``cand_key`` e checkpoint integro;
* OOM com tentativas parametrizadas e motivo de falha explicito (oom, nan, missing_ancestor, ...);
* ids ``h4-<braco>-c#####`` e ``kind = h4_<braco>`` no estado (monitor_report).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from ga_search.hmodehb import bk as BK
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import integrity as I
from ga_search.modehb import modehb as M

STATE_VERSION = 1


class HMODEHB(M.MODEHB):
    def __init__(self, config: dict, evaluate: Callable, output_dir, *, arm: str, fitness_spec: dict,
                 full_config: dict, allow_drift: Iterable[str] = (), oom_retries: int = 4,
                 oom_sleep_s: float = 180.0, lock: Optional[I.ControllerLock] = None, logger=None,
                 spifil_repair: Optional[Callable] = None, param_fn: Optional[Callable] = None):
        cfg = dict(config or {})
        if (cfg.get("pnas_cs") or {}).get("enabled") or cfg.get("pnas_cs"):
            raise NotImplementedError("braco C (PNAS-CS) ainda nao roda no H-MODEHB: os niveis do pnas_cs.py "
                                      "(pontos medios geometricos, cap x1,05) nao sao as faixas B0-B4")
        if cfg.get("use_latency"):
            raise FV.FitnessConfigError("fitness v2: latencia nao e objetivo (use_latency: false)")
        ob = cfg.get("objective_bounds") or {}
        want = {"flops_lo": FV.FLOPS_BOUNDS[0], "flops_hi": FV.FLOPS_BOUNDS[1]}
        if any(float(ob[k]) != v for k, v in want.items() if k in ob) or set(ob) - set(want):
            raise FV.FitnessConfigError(f"objective_bounds deve ser {want}, veio {ob}")
        cfg["objective_bounds"] = want
        cfg["stratified"] = BK.stratified_block(cfg.get("stratified"))
        super().__init__(cfg, evaluate, output_dir, logger=logger, spifil_repair=spifil_repair, param_fn=param_fn)
        BK.check_strat(self.strat)
        self.arm = I.check_arm(arm)
        self.kind = f"h4_{self.arm}"
        self.fitness_spec = dict(fitness_spec)
        self.fitness_sig = FV.signature(self.fitness_spec)
        self.full_config = I._plain(full_config)
        self.config_sha = I.config_sha(self.full_config)
        self.allow_drift = tuple(allow_drift or ())
        self.oom_retries, self.oom_sleep_s = int(oom_retries), float(oom_sleep_s)
        self._lock = lock
        self._carry = {}

    # ---- estado
    def _fresh_state(self):
        st = super()._fresh_state()
        st.update({
            "hmodehb_version": STATE_VERSION, "arm": self.arm, "kind": self.kind,
            "id_prefix": f"{I.ID_PREFIX}-{self.arm}-",
            "fitness_sig": self.fitness_sig, "fitness_spec": self.fitness_spec,
            "config_sha": self.config_sha, "config_full": self.full_config,
            "config_sha_initial": self._carry.get("config_sha_initial", self.config_sha),
            "drift_log": list(self._carry.get("drift_log", [])),
            "created": self._carry.get("created") or time.time(),
            "bk_bands": BK.bands_dict(), "param_cap": BK.HARD_CAP,
            "oom": {"retries": self.oom_retries, "sleep_s": self.oom_sleep_s}})
        return st

    def _check_resume(self):
        if not self.state_path.exists():
            return
        old = json.loads(self.state_path.read_text())
        if old.get("fitness_sig") != self.fitness_sig:
            raise FV.FitnessSigMismatch(
                f"{self.state_path}: regua gravada {old.get('fitness_sig')} != atual {self.fitness_sig}. "
                f"Uma busca nunca mistura reguas (ex.: C=0.1 x C=100, cov_head x head): use outro diretorio.")
        if old.get("arm") != self.arm:
            raise I.ConfigDrift(f"{self.state_path}: braco gravado {old.get('arm')} != {self.arm}")
        drift = I.check_drift(old, self.full_config, self.allow_drift)
        self._carry = {"drift_log": list(old.get("drift_log") or []) + ([drift] if drift else []),
                       "config_sha_initial": old.get("config_sha_initial", old.get("config_sha")),
                       "created": old.get("created")}

    def run(self, resume: bool = False) -> dict:
        own = self._lock is None
        lock = I.ControllerLock(self.out).acquire() if own else self._lock
        try:
            self._carry = {}
            if self.state_path.exists() and resume:
                self._check_resume()
            return super().run(resume=resume)
        finally:
            if own:
                lock.release()

    # ---- ids
    def _new_id(self):
        self.state["n_candidates"] += 1
        return I.cand_id(self.arm, self.state["n_candidates"])

    # ---- identidade do candidato
    def _weight_origin(self, genome, cand_id, role, inherit):
        """Origem dos pesos: a do candidato (promocao/extensao/gemea seguinte) ou uma nova."""
        if cand_id is not None:
            for j in self.state["jobs"]:
                if j["cand_id"] == cand_id and j["role"] == role and j.get("weight_origin") is not None:
                    return j["weight_origin"]
        if inherit and role == "search":
            jobs, pj, sha = self.state["jobs"], inherit.get("parent_job"), None
            if pj is not None and 0 <= int(pj) < len(jobs):
                sha = (jobs[int(pj)].get("result") or {}).get("checkpoint_sha256")
            if sha is None:
                sha = I.file_sha256(inherit.get("checkpoint"))
            return {"kind": "inherit", "parent_cand_id": inherit.get("parent_cand_id"),
                    "parent_fidelity": inherit.get("parent_fidelity"), "parent_ckpt_sha256": sha,
                    "n_blocks": inherit.get("n_blocks"), "layer_map": inherit.get("layer_map")}
        if genome.get("external"):
            return {"kind": "external", "external": genome["external"]}
        return {"kind": "scratch", "init": genome.get("init")}

    def _keys(self, genome, fid, cand_id, role, inherit):
        origin = self._weight_origin(genome, cand_id, role, inherit)
        lkey = I.lineage_key(M.canonical(genome), genome.get("seed"), origin, self.fitness_sig)
        return origin, lkey, I.cand_key(lkey, fid)

    # ---- avaliacao (mesma estrutura de MODEHB._evaluate; cache/resume por cand_key/lineage_key)
    def _evaluate(self, genome, fid, *, parent_id, generation, rung, cand_id=None, vector=None,
                  role="search", twin_of=None, twin_source=None, inherit=None, extra_ctx=None):
        genome = M.repair(genome, **self.repair_kw)
        origin, lkey, ckey = self._keys(genome, fid, cand_id, role, inherit)
        jobs = self.state["jobs"]
        cached = [j for j in jobs if j.get("cand_key") == ckey and j["role"] == role]
        lower = [j for j in jobs if j.get("lineage_key") == lkey and j["role"] == role and j["fidelity"] < fid
                 and j["status"] == "ok" and j["result"].get("checkpoint")]
        resume = max(lower, key=lambda j: j["fidelity"]) if lower else None
        if cand_id is None:
            cand_id = resume["cand_id"] if resume else (cached[0]["cand_id"] if cached else self._new_id())
        ctx = {"cand_id": cand_id, "parent_id": parent_id, "generation": generation, "bracket": generation,
               "rung": rung, "role": role, "twin_of": twin_of, "resume": resume, "twin_source": twin_source,
               "front_rank": lambda result, _f=fid: self.rank_of(result, _f),
               "cand_key": ckey, "lineage_key": lkey, "fitness_sig": self.fitness_sig,
               "config_sha": self.config_sha, "job_index": len(jobs), "weight_origin": origin, "arm": self.arm}
        if extra_ctx:
            ctx.update(extra_ctx)
        if inherit and resume is None and not cached and role == "search":
            ctx["inherit"] = inherit
        p_est = int(self.param_fn(genome))
        if cached:
            result = {k: v for k, v in cached[0]["result"].items()}
            result["cached_from"] = cached[0]["job"]
            cost_free = True
        elif p_est > BK.HARD_CAP:         # teto duro: nunca gasta GPU num genoma acima de 5,5M
            result = {"status": "failed", "infeasible": "param_cap", "reason": "param_cap",
                      "error": f"params estimados {p_est} > teto {BK.HARD_CAP}"}
            cost_free = True
        else:
            result = self._call(genome, fid, ctx)
            cost_free = False
        FV.check_decisions(result, self.fitness_spec)   # toda decisao pela regua gravada, senao para tudo
        from ga_search.modehb import strata as S
        p_meas = result.get("params") if result.get("status", "ok") == "ok" else None
        stratum = S.stratum_of(p_meas if p_meas is not None else p_est, self.strat)
        if result.get("status", "ok") == "ok" and p_meas is not None and int(p_meas) > BK.HARD_CAP:
            result = {**result, "status": "failed", "infeasible": "param_cap", "reason": "param_cap",
                      "error": f"params {p_meas} > teto {BK.HARD_CAP}"}
        F = FV.objectives(result)
        rec = {"job": len(jobs), "cand_id": cand_id, "parent_id": parent_id,
               "generation": generation, "rung": rung, "fidelity": fid, "role": role, "twin_of": twin_of,
               "genome": genome, "arch_hash": M.arch_hash(genome),
               "twin_key": M.arch_hash(genome, include_init=False),
               "vector": list(vector) if vector is not None else self._encode(genome).tolist(),
               "status": result.get("status", "ok"), "F": F, "result": result,
               "resumed_from": resume["job"] if resume else None, "cached": cost_free,
               "time": time.time(), "cand_key": ckey, "lineage_key": lkey, "weight_origin": origin,
               "fitness_sig": self.fitness_sig, "config_sha": self.config_sha,
               "stratum": stratum, "bk": stratum, "params_est": p_est,
               **({"inherit": {k: v for k, v in ctx["inherit"].items() if k != "effective_specs"}}
                  if ctx.get("inherit") else {})}
        jobs.append(rec)
        if not cost_free:
            self.state["gpu_hours"] += float(result.get("gpu_hours", 0.0) or 0.0)
        self._save()
        if role == "search" and genome["init"] == "spifil" and fid == self.fidelities[-1] and rec["status"] == "ok" \
                and self.random_baseline:
            self._evaluate_twin(rec)
        return rec

    def _evaluate_twin(self, rec):
        """Gemea random ao longo do caminho de fidelidades DESTE candidato (cand_id, nao arch_hash)."""
        twin = M.make_twin(rec["genome"])
        mine = [j for j in self.state["jobs"] if j["cand_id"] == rec["cand_id"] and j["role"] == "search"]
        path = sorted({j["fidelity"] for j in mine if j["status"] == "ok"})
        twin_id = None
        for f in path:
            done = [j for j in self.state["jobs"] if j["role"] == "twin" and j["twin_of"] == rec["cand_id"]
                    and j["fidelity"] == f]
            if done:
                twin_id = done[0]["cand_id"]
                continue
            src = next(j for j in mine if j["fidelity"] == f)
            t = self._evaluate(twin, f, parent_id=rec["cand_id"], generation=rec["generation"],
                               rung=src["rung"], cand_id=twin_id, role="twin",
                               twin_of=rec["cand_id"], twin_source=src)
            twin_id = t["cand_id"]

    # ---- chamada do avaliador: replay, adocao, OOM, motivo de falha
    def _hjob_path(self, k):
        return self.out / "hjobs" / f"{int(k):06d}.json"

    def _adopt(self, k, ctx, fid):
        """Resultado ja gravado deste job (crash entre a escrita e o _save) ou None."""
        srcs = [(self._hjob_path(k), lambda d: (d or {}).get("result")),
                (self.out / "candidates" / ctx["cand_id"] / f"f{int(fid)}" / "result.json", lambda d: d)]
        for path, get in srcs:
            res = get(I.read_json(path)) if path.exists() else None
            if res is None:
                continue
            why = I.adoptable(res, ckey=ctx["cand_key"], fitness_sig=self.fitness_sig)
            if why is None:
                res = dict(res)
                res["adopted"] = {"from": str(path), "job": int(k), "time": time.time()}
                print(f"[hmodehb] job {k} ({ctx['cand_id']}@{fid:g}) adotado de {path} (sem re-treino)", flush=True)
                return res
            print(f"[hmodehb] job {k}: {path} nao adotado ({why})", flush=True)
        return None

    def _call(self, genome, fid, ctx):
        k = len(self.state["jobs"])
        if k < len(self._replay):
            old = self._replay[k]
            if old["arch_hash"] != M.arch_hash(genome) or float(old["fidelity"]) != float(fid) \
                    or old["role"] != ctx["role"] or old.get("cand_key") != ctx["cand_key"] \
                    or old.get("fitness_sig") != self.fitness_sig:
                raise M.ReplayMismatch(
                    f"job {k}: gravado {old['arch_hash']}@{old['fidelity']} key={old.get('cand_key')} "
                    f"sig={old.get('fitness_sig')} vs {M.arch_hash(genome)}@{fid} key={ctx['cand_key']} "
                    f"sig={self.fitness_sig}")
            return old["result"]
        result = self._adopt(k, ctx, fid)
        if result is None:
            I.atomic_write_json(self.out / "pending_job.json",
                                {"job": k, "cand_id": ctx["cand_id"], "cand_key": ctx["cand_key"], "fidelity": fid,
                                 "role": ctx["role"], "config_sha": self.config_sha,
                                 "fitness_sig": self.fitness_sig, "time": time.time()})
            result = self._run_with_retries(genome, fid, ctx, k)
        FV.check_decisions(result, self.fitness_spec)       # antes de gravar qualquer coisa
        result.setdefault("status", "ok")
        result.setdefault("cand_key", ctx["cand_key"])
        result.setdefault("fitness_sig", self.fitness_sig)
        result.setdefault("config_sha", self.config_sha)
        if result["status"] == "ok" and result.get("checkpoint") and not result.get("checkpoint_sha256"):
            result["checkpoint_sha256"] = I.file_sha256(result["checkpoint"])
        I.atomic_write_json(self._hjob_path(k), {"job": k, "cand_id": ctx["cand_id"], "cand_key": ctx["cand_key"],
                                                 "fidelity": fid, "role": ctx["role"], "result": result})
        return result

    def _run_with_retries(self, genome, fid, ctx, k):
        attempt = 0
        while True:
            try:
                res = dict(self.evaluate_fn(genome, fid, ctx))
                if attempt:
                    res["oom_retries"] = attempt
                return res
            except Exception as exc:       # falha de avaliacao = candidato inviavel, nao crash
                if M._fatal_cuda(exc):        # contexto CUDA envenenado: sem registro, o resume refaz
                    raise
                if isinstance(exc, FV.FitnessSigMismatch):   # regua errada (inclui RulerViolation): fatal
                    raise
                if M._is_oom(exc) and attempt < self.oom_retries:
                    attempt += 1
                    print(f"[hmodehb] OOM no job {k}; tentativa {attempt}/{self.oom_retries} em {self.oom_sleep_s:g}s",
                          flush=True)
                    time.sleep(self.oom_sleep_s)
                    continue
                if M._is_oom(exc):
                    reason = "oom"
                elif isinstance(exc, FV.NonFiniteFitness):
                    reason = "nan"
                elif type(exc).__name__ == "EmbedDimError":
                    reason = "embed_dim"
                elif isinstance(exc, FV.FitnessConfigError):
                    reason = "fitness_config"
                elif isinstance(exc, FileNotFoundError) and ctx.get("inherit"):
                    reason = "missing_ancestor"
                else:
                    reason = "error"
                res = {"status": "failed", "reason": reason, "error": f"{type(exc).__name__}: {exc}"}
                if attempt:
                    res["oom_retries"] = attempt
                return res

    # ---- exportacao: melhor por B_k (validacao, fidelidade maxima)
    def best_by_bk(self, fidelity=None) -> dict:
        fid = float(fidelity if fidelity is not None else self.fidelities[-1])
        out = {}
        for j in self.state["jobs"]:
            if j["role"] != "search" or j["status"] != "ok" or float(j["fidelity"]) != fid or not j.get("bk"):
                continue
            cur = out.get(j["bk"])
            key = (j["F"][0], j["F"][1])
            if cur is None or key < (cur["F"][0], cur["F"][1]):
                out[j["bk"]] = {"cand_id": j["cand_id"], "cand_key": j["cand_key"], "F": j["F"],
                                "kappa_val_orig": j["result"].get("kappa_val_orig", j["result"].get("fitness_kappa")),
                                "params": j["result"].get("params"), "flops": j["result"].get("flops"),
                                "checkpoint": j["result"].get("checkpoint"),
                                "checkpoint_sha256": j["result"].get("checkpoint_sha256"), "fidelity": fid}
        return {"fitness_sig": self.fitness_sig, "fidelity": fid, "best": out}
