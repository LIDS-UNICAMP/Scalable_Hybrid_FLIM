"""W&B logging for Progressive MO-DEHB (prompt §10, §17, §17.1).

Three run kinds share one project (default ``progressive_mo_dehb``):

* ``CandidateRun``  (job_type="candidate")     -- one per evaluated genome x fidelity.
* ``ControllerRun`` (job_type="controller")    -- the search loop: HV curves, Pareto, cost.
* ``ExternalEvalRun`` (job_type="external_eval") -- on-demand SVM probe, never read by search.

Hypervolume math is NOT implemented here: ``hypervolume`` / ``hv_contributions`` come
from ``ga_search.modehb.modehb`` (lazy import, so this module imports without it).
Objectives are assumed in *minimization* form (as passed to the HV functions).

Without network the runs fall back to ``WANDB_MODE=offline`` (sync later with
``wandb sync``). Tests use ``WANDB_MODE=disabled``/``offline``.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.constants import WANDB_ENTITY

WANDB_PROJECT_MODEHB: str = "progressive_mo_dehb"
DASHBOARD_NAME: str = "Progressive MO-DEHB: SPiFiL Evolution & SSL Representation Learning"
EXTERNAL_TABLE_NAME: str = "SVM sob demanda"
LOWFID_PREFIX: str = "lowfid/"
GROUP_KEYS: tuple[str, ...] = ("init", "ssl", "reg", "depth")
DEFAULT_ALERTS_JSONL: str = "artifacts/modehb/alerts.jsonl"
DEFAULT_EXTERNAL_JSONL: str = "artifacts/modehb/svm_sob_demanda.jsonl"
STAGNATION_WINDOW: int = 5
HV_GAIN_EPS: float = 1e-12
_NET_HOST, _NET_PORT, _NET_TIMEOUT = "api.wandb.ai", 443, 2.0


# ─── helpers ──────────────────────────────────────────────────────────────────

def _wandb():
    import wandb  # lazy: heavy import

    return wandb


def _hv_funcs():
    """Lazy import of I9's HV functions (kept separate so tests can monkeypatch)."""
    from ga_search.modehb.modehb import hv_contributions, hypervolume

    return hypervolume, hv_contributions


def _hv(objs: np.ndarray, ref: Sequence[float]) -> float:
    if len(objs) == 0:
        return 0.0
    hypervolume, _ = _hv_funcs()
    return float(hypervolume(np.asarray(objs, dtype=float), np.asarray(ref, dtype=float)))


def _hv_contrib(objs: np.ndarray, ref: Sequence[float]) -> np.ndarray:
    if len(objs) == 0:
        return np.zeros(0)
    _, hv_contributions = _hv_funcs()
    return np.asarray(hv_contributions(np.asarray(objs, dtype=float), np.asarray(ref, dtype=float)), dtype=float)


def resolve_mode() -> str:
    """Honour WANDB_MODE; otherwise 'online' if api.wandb.ai is reachable else 'offline'."""
    env = os.environ.get("WANDB_MODE")
    if env:
        return env
    try:
        with socket.create_connection((_NET_HOST, _NET_PORT), timeout=_NET_TIMEOUT):
            return "online"
    except OSError:
        return "offline"


def genome_to_dict(genome: Any) -> dict:
    if genome is None:
        return {}
    if isinstance(genome, Mapping):
        return dict(genome)
    if dataclasses.is_dataclass(genome):
        return dataclasses.asdict(genome)
    for meth in ("to_dict", "as_dict"):
        if callable(getattr(genome, meth, None)):
            return dict(getattr(genome, meth)())
    return dict(vars(genome))


def _tag(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = "+".join(map(str, value))
    return f"{key}:{value}"[:64]  # W&B tag length limit


def candidate_tags(genome: Mapping, dataset: str, fidelity: Any) -> list[str]:
    tags = [_tag("dataset", dataset), _tag("fidelity", fidelity)]
    for key in ("init", "ssl", "depth"):
        if key in genome:
            tags.append(_tag(key, genome[key]))
    ops = genome.get("operators", genome.get("ops"))
    if ops is not None:
        tags.append(_tag("operators", ops))
    group = genome.get("experimental_group", genome.get("group"))
    if group is not None:
        tags.append(_tag("group", group))
    return tags


def _init_run(project: str, entity: str, mode: str | None, **kwargs):
    wandb = _wandb()
    mode = mode or resolve_mode()
    try:
        run = wandb.init(project=project, entity=entity, mode=mode, reinit="create_new", **kwargs)
    except Exception:  # network/auth failure -> offline fallback
        if mode != "online":
            raise
        run = wandb.init(project=project, entity=entity, mode="offline", reinit="create_new", **kwargs)
    return run


class _BaseRun:
    job_type: str = ""

    def __init__(self, *, project: str = WANDB_PROJECT_MODEHB, entity: str = WANDB_ENTITY,
                 mode: str | None = None, **init_kwargs):
        self.init_kwargs = dict(project=project, entity=entity, job_type=self.job_type, **init_kwargs)
        self.run = _init_run(project, entity, mode, job_type=self.job_type, **init_kwargs)

    @property
    def id(self) -> str:
        return getattr(self.run, "id", "") or ""

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        self.run.log(dict(data), step=step)

    def log_table(self, name: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> Any:
        table = _wandb().Table(columns=list(columns), data=[list(r) for r in rows])
        self.run.log({name: table})
        return table

    def finish(self, exit_code: int | None = None) -> None:
        self.run.finish(exit_code=exit_code)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish(exit_code=1 if exc_type else 0)


# ─── candidate ────────────────────────────────────────────────────────────────

class CandidateRun(_BaseRun):
    job_type = "candidate"

    def __init__(self, *, alerts_path: str | os.PathLike = DEFAULT_ALERTS_JSONL, **kw):
        super().__init__(**kw)
        self.alerts_path = Path(alerts_path)
        cfg = self.init_kwargs.get("config", {})
        self.generation = cfg.get("generation")
        self.fidelity = cfg.get("fidelity")

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        """wandb.alert + local JSONL copy (alerts survive offline mode)."""
        wandb = _wandb()
        record = {"time": time.time(), "run_id": self.id, "title": title, "text": text,
                  "level": str(level).upper(), "generation": self.generation, "fidelity": self.fidelity}
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alerts_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        try:
            self.run.alert(title=title, text=text, level=getattr(wandb.AlertLevel, record["level"], wandb.AlertLevel.WARN))
        except Exception:  # offline/disabled: the JSONL copy is the source of truth
            pass

    def log_manual_metrics(self, *, dataloader_wait: float | None = None, latency_per_img: float | None = None,
                           flops: float | None = None, gpu_hours_cum: float | None = None,
                           step: int | None = None) -> None:
        """Metrics W&B does not collect itself (GPU/CPU/mem come from W&B system metrics)."""
        data = {"perf/dataloader_wait": dataloader_wait, "perf/latency_per_img": latency_per_img,
                "perf/flops": flops, "cost/gpu_hours_cum": gpu_hours_cum}
        data = {k: v for k, v in data.items() if v is not None}
        if data:
            self.log(data, step=step)

    def log_checkpoint(self, path: str | os.PathLike, meta: Mapping[str, Any] | None = None) -> Any:
        """Log an encoder checkpoint as Artifact type 'encoder' with gen-/fid-/front-rank- aliases."""
        meta = dict(meta or {})
        gen = meta.get("generation", self.generation)
        fid = meta.get("fidelity", self.fidelity)
        rank = meta.get("front_rank")
        metadata = {"run_id": self.id, "generation": gen, "fidelity": fid, "front_rank": rank, **meta}
        art = _wandb().Artifact(name=f"encoder-{self.id or 'local'}", type="encoder", metadata=metadata)
        art.add_file(str(path))
        aliases = ["latest"]
        if gen is not None:
            aliases.append(f"gen-{gen}")
        if fid is not None:
            aliases.append(f"fid-{fid}")
        if rank is not None:
            aliases.append(f"front-rank-{rank}")
        self.run.log_artifact(art, aliases=aliases)
        return art


def start_candidate(genome: Any, dataset: str, fidelity: Any, parent_id: str | None, generation: int,
                    seed: int, split_hash: str, arch_hash: str, git_commit: str, *,
                    project: str = WANDB_PROJECT_MODEHB, entity: str = WANDB_ENTITY, mode: str | None = None,
                    alerts_path: str | os.PathLike = DEFAULT_ALERTS_JSONL, **extra_config) -> CandidateRun:
    g = genome_to_dict(genome)
    config = {"genome": g, **g, "dataset": dataset, "fidelity": fidelity, "parent_id": parent_id,
              "generation": generation, "seed": seed, "split_hash": split_hash, "arch_hash": arch_hash,
              "git_commit": git_commit, **extra_config}
    name = f"cand-{dataset}-g{generation}-f{fidelity}-{str(arch_hash)[:8]}"
    return CandidateRun(project=project, entity=entity, mode=mode, alerts_path=alerts_path,
                        group=f"{dataset}", tags=candidate_tags(g, dataset, fidelity), config=config, name=name)


# ─── controller ───────────────────────────────────────────────────────────────

def pareto_mask(objs: np.ndarray) -> np.ndarray:
    """True for non-dominated rows (minimization)."""
    objs = np.asarray(objs, dtype=float)
    n = len(objs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        mask[i] = not dominated.any()
    return mask


def front_spacing(front: np.ndarray) -> float:
    """Schott spacing: std of nearest-neighbour L1 distances on the front (0 = uniform)."""
    front = np.asarray(front, dtype=float)
    if len(front) < 2:
        return 0.0
    d = np.abs(front[:, None, :] - front[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    return float(np.std(d.min(axis=1), ddof=1)) if len(front) > 2 else 0.0


def front_spread(front: np.ndarray) -> float:
    """Extent of the front: Euclidean norm of per-objective ranges."""
    front = np.asarray(front, dtype=float)
    if len(front) < 2:
        return 0.0
    return float(np.linalg.norm(front.max(axis=0) - front.min(axis=0)))


def hv_ci(series_list: Sequence[Sequence[float]], confidence: float = 0.95) -> dict[str, np.ndarray]:
    """Mean and CI of HV curves across search seeds (aligned by index, truncated to shortest)."""
    if not series_list:
        return {"mean": np.zeros(0), "lo": np.zeros(0), "hi": np.zeros(0), "n": 0}
    length = min(len(s) for s in series_list)
    arr = np.asarray([list(s)[:length] for s in series_list], dtype=float)
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    if n < 2:
        return {"mean": mean, "lo": mean.copy(), "hi": mean.copy(), "n": n}
    try:
        from scipy.stats import t

        crit = float(t.ppf(0.5 + confidence / 2, df=n - 1))
    except Exception:
        crit = 1.959963984540054
    half = crit * arr.std(axis=0, ddof=1) / math.sqrt(n)
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n}


class ControllerRun(_BaseRun):
    """Search-loop run. ``points`` passed to ``log_generation`` are dicts with keys:
    ``objectives`` (minimization vector, same order as ``ref_point``), ``fidelity``,
    optional ``id``, ``quality``, ``flops``, ``init``, ``ssl``, ``reg``, ``depth``.
    Main HV uses ONLY points at ``max_fidelity``; the rest go under ``lowfid/``."""

    job_type = "controller"
    X_AXES = ("generation", "gpu_hours", "wall_time", "n_evaluations")

    def __init__(self, *, ref_point: Sequence[float], max_fidelity: float, name: str | None = None,
                 config: Mapping[str, Any] | None = None, tags: Sequence[str] | None = None,
                 group: str | None = None, stagnation_window: int = STAGNATION_WINDOW, **kw):
        cfg = {"ref_point": list(ref_point), "max_fidelity": max_fidelity, "dashboard": DASHBOARD_NAME, **(config or {})}
        super().__init__(name=name, config=cfg, tags=list(tags or ["controller"]), group=group or "controller", **kw)
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.max_fidelity = max_fidelity
        self.window = stagnation_window
        self._archive: list[np.ndarray] = []          # cumulative max-fidelity objectives
        self._hv_best = 0.0
        self._hv_first: float | None = None
        self._gains: list[float] = []
        self._evals_at_last_gain = 0
        self._gpu_at_last = 0.0
        self.history: list[dict] = []
        self._define_metrics()

    def _define_metrics(self) -> None:
        dm = getattr(self.run, "define_metric", None)
        if dm is None:
            return
        for x in self.X_AXES:
            dm(x)
        for prefix in ("hv/*", "front/*", "stagnation/*", "cost/*", "promotion/*", "hv_strata/*", "strata/*", "ops/*",
                       "hv_by_init/*", "hv_by_ssl/*", "hv_by_reg/*", "hv_by_depth/*", f"{LOWFID_PREFIX}*", "pnas/*"):
            dm(prefix, step_metric="generation")

    def _is_max(self, p: Mapping, default_fid: Any) -> bool:
        fid = p.get("fidelity", default_fid)
        return fid is not None and float(fid) >= float(self.max_fidelity)

    def compute_generation_metrics(self, points: Sequence[Mapping], fidelity: Any, n_evals: int,
                                   gpu_hours: float, meta: Mapping[str, Any] | None = None) -> dict:
        meta = dict(meta or {})
        hi = [p for p in points if self._is_max(p, fidelity)]
        lo = [p for p in points if not self._is_max(p, fidelity)]
        out: dict[str, Any] = {}

        m = len(self.ref_point)
        objs = np.asarray([p["objectives"] for p in hi], dtype=float).reshape(len(hi), m)
        self._archive.extend(objs)
        arch = np.asarray(self._archive, dtype=float).reshape(len(self._archive), m)
        hv_cum = _hv(arch[pareto_mask(arch)] if len(arch) else arch, self.ref_point)
        hv_pop = _hv(objs, self.ref_point)
        gain = max(0.0, hv_cum - self._hv_best)
        if self._hv_first is None and len(arch):
            self._hv_first = hv_cum
        if gain > HV_GAIN_EPS:
            self._evals_at_last_gain = n_evals
            delta_gpu = gpu_hours - self._gpu_at_last
            out["cost/gpu_hours_per_hv_gain"] = delta_gpu / gain
            self._gpu_at_last = gpu_hours
        self._hv_best = max(self._hv_best, hv_cum)
        self._gains.append(gain)
        total_gain = self._hv_best - (self._hv_first or 0.0)
        out.update({
            "hv/best_cumulative": self._hv_best,
            "hv/population": hv_pop,
            "hv/marginal": gain,
            "hv/n_max_fidelity_points": len(hi),
            "stagnation/moving_improvement": float(np.mean(self._gains[-self.window:])),
            "stagnation/evals_since_last_gain": n_evals - self._evals_at_last_gain,
            "cost/gpu_hours_per_hv_unit": (gpu_hours / total_gain) if total_gain > HV_GAIN_EPS else None,
        })
        n_prom = meta.get("n_promotions")
        if n_prom:
            out["promotion/delta_hv_per_promotion"] = gain / float(n_prom)
            out["promotion/n_promotions"] = n_prom

        mask = pareto_mask(objs) if len(objs) else np.zeros(0, dtype=bool)
        front = objs[mask] if len(objs) else objs
        out["front/size"] = int(mask.sum())
        out["front/spacing"] = front_spacing(front)
        out["front/spread"] = front_spread(front)

        for key in GROUP_KEYS:
            for val in sorted({str(p[key]) for p in hi if key in p}):
                sub = np.asarray([p["objectives"] for p in hi if str(p.get(key)) == val], dtype=float)
                out[f"hv_by_{key}/{val}"] = _hv(sub, self.ref_point)

        if lo:  # low-fidelity HV: never mixed with the main curve
            by_fid: dict[str, list] = {}
            for p in lo:
                by_fid.setdefault(str(p.get("fidelity", fidelity)), []).append(p["objectives"])
            for fid, lst in by_fid.items():
                out[f"{LOWFID_PREFIX}hv_population/fid-{fid}"] = _hv(np.asarray(lst, dtype=float), self.ref_point)
            out[f"{LOWFID_PREFIX}n_points"] = len(lo)

        contrib = _hv_contrib(objs, self.ref_point) if len(objs) else np.zeros(0)
        return {"scalars": {k: v for k, v in out.items() if v is not None}, "hi": hi, "mask": mask,
                "contrib": contrib}

    def log_generation(self, points: Sequence[Mapping], fidelity: Any, generation: int, gpu_hours: float,
                       wall_time: float, n_evals: int, meta: Mapping[str, Any] | None = None,
                       dry: bool = False) -> dict:
        """``dry=True`` (search replaying its state after a restart): update the cumulative
        archive / Pareto points but log nothing, so a resumed run does not duplicate history."""
        res = self.compute_generation_metrics(points, fidelity, n_evals, gpu_hours, meta)
        self._all_pts = getattr(self, "_all_pts", {})
        for p in points:                                  # every fidelity, latest result per (id, fid)
            self._all_pts[(str(p.get("id")), float(p.get("fidelity", fidelity) or 0))] = p
        self._scatter_pts = getattr(self, "_scatter_pts", {})
        for p in res["hi"]:
            self._scatter_pts[str(p.get("id", len(self._scatter_pts)))] = p
        if dry:
            return {}
        data: dict[str, Any] = {"generation": generation, "gpu_hours": gpu_hours, "wall_time": wall_time,
                                "n_evaluations": n_evals, "fidelity": fidelity, **res["scalars"]}
        wandb = _wandb()
        hi, mask, contrib = res["hi"], res["mask"], res["contrib"]
        if hi:
            cols = ["id", "fidelity", "init", "ssl", "reg", "depth", "quality", "flops", "hv_contribution", "front"]
            rows = [[str(p.get("id", i)), p.get("fidelity", fidelity), str(p.get("init")), str(p.get("ssl")),
                     str(p.get("reg")), str(p.get("depth")), p.get("quality"), p.get("flops"),
                     float(contrib[i]) if i < len(contrib) else None,
                     "non-dominated" if mask[i] else "dominated"] for i, p in enumerate(hi)]
            contrib_table = wandb.Table(columns=cols, data=rows)
            data["tables/hv_contribution"] = contrib_table
            # Pareto scatter = ALL max-fidelity points seen so far, front recomputed on the archive
            allp = list(self._scatter_pts.values())
            amask = pareto_mask(np.asarray([p["objectives"] for p in allp], dtype=float))
            scat_rows = [[pid, p.get("fidelity", fidelity), str(p.get("init")), str(p.get("ssl")),
                          str(p.get("reg")), str(p.get("depth")), p.get("quality"), p.get("flops"), None,
                          "non-dominated" if amask[i] else "dominated"]
                         for i, (pid, p) in enumerate(self._scatter_pts.items())]
            scat_rows = [r for r in scat_rows if r[6] is not None and r[7] is not None]
            if scat_rows:
                scat = wandb.Table(columns=cols, data=scat_rows)
                data["pareto/table"] = scat
                data["pareto/quality_vs_flops"] = wandb.plot.scatter(
                    scat, x="flops", y="quality", title=f"Pareto quality x FLOPs (gen {generation})")
        # live Pareto over EVERY evaluation (all fidelities): moves after each candidate.
        # front = non-dominated within the same fidelity (fidelities are not comparable).
        allp = [p for p in self._all_pts.values() if p.get("quality") is not None and p.get("flops") is not None]
        if allp:
            front = set()
            for f in {float(p.get("fidelity") or 0) for p in allp}:
                sub = [i for i, p in enumerate(allp) if float(p.get("fidelity") or 0) == f]
                m = pareto_mask(np.asarray([allp[i]["objectives"] for i in sub], dtype=float))
                front |= {sub[k] for k in range(len(sub)) if m[k]}
            cols2 = ["id", "fidelity", "stratum", "params", "flops", "gflops", "quality", "kappa", "init", "front"]
            rows2 = [[str(p.get("id")), float(p.get("fidelity") or 0), str(p.get("stratum")), p.get("params"),
                      p.get("flops"), float(p["flops"]) / 1e9, p.get("quality"), p.get("kappa"), str(p.get("init")),
                      "non-dominated" if i in front else "dominated"] for i, p in enumerate(allp)]
            t2 = wandb.Table(columns=cols2, data=rows2)
            data["pareto/table_all_fidelities"] = t2
            data["pareto/all_fidelities"] = wandb.plot.scatter(
                t2, x="gflops", y="quality", title=f"Pareto ao vivo: todas as avaliações ({len(rows2)})")
            # κ-only Pareto over EVERYTHING evaluated (all fidelities, all categories):
            # objectives (1 - κ, GFLOPs); the front line = best κ reachable at each cost.
            kp = [p for p in allp if p.get("kappa") is not None]
            if kp:
                K = np.asarray([[1.0 - float(p["kappa"]), float(p["flops"])] for p in kp], dtype=float)
                km = pareto_mask(K)
                rows3 = [[str(p.get("id")), float(p.get("fidelity") or 0), str(p.get("stratum")), p.get("params"),
                          float(p["flops"]) / 1e9, float(p["kappa"]), "pareto" if km[i] else "dominado"]
                         for i, p in enumerate(kp)]
                cols3 = ["id", "fidelity", "stratum", "params", "gflops", "kappa", "front"]
                t3 = wandb.Table(columns=cols3, data=rows3)
                data["pareto_kappa/table"] = t3
                data["pareto_kappa/kappa_vs_gflops"] = wandb.plot.scatter(
                    t3, x="gflops", y="kappa", title=f"κ × GFLOPs: todas as avaliações ({len(rows3)})")
                fr = sorted((r for r in rows3 if r[-1] == "pareto"), key=lambda r: r[4])
                t4 = wandb.Table(columns=cols3, data=fr)
                data["pareto_kappa/front_table"] = t4
                data["pareto_kappa/front"] = wandb.plot.line(
                    t4, x="gflops", y="kappa", title=f"Frente de Pareto de κ ({len(fr)} redes)")
                data["pareto_kappa/best_kappa"] = float(max(r[5] for r in rows3))
        self.run.log(data)
        self.history.append({k: v for k, v in data.items() if isinstance(v, (int, float, str)) or v is None})
        return data


    def log_strata(self, generation: int, gpu_hours: float, n_evals: int, scalars: Mapping[str, Any],
                   tables: Mapping[str, tuple]) -> dict:
        """Parameter-stratified MO-DEHB panels: per-category HV/rates (scalars, x = generation,
        gpu_hours or n_evaluations) + tables with line/scatter/bar plots."""
        return self._log_panels("strata", generation, gpu_hours, n_evals, scalars, tables)

    def log_pnas(self, generation: int, gpu_hours: float, n_evals: int, scalars: Mapping[str, Any],
                 tables: Mapping[str, tuple]) -> dict:
        """PNAS-CS panels (pnas_cs.report): HV x GPU-hours, kappa x params, live Pareto kappa x FLOPs and
        its evolution, growth curves P0-P4 per family, depth x width, marginal gain per expansion,
        operator/mutation survival, scaling laws, Phase-0 AE ablation. Tables under ``pnas_tables/``."""
        return self._log_panels("pnas", generation, gpu_hours, n_evals, scalars, tables)

    def _log_panels(self, prefix: str, generation: int, gpu_hours: float, n_evals: int,
                    scalars: Mapping[str, Any], tables: Mapping[str, tuple]) -> dict:
        wandb = _wandb()
        data: dict[str, Any] = {"generation": generation, "gpu_hours": gpu_hours, "n_evaluations": n_evals,
                                **{k: v for k, v in scalars.items() if v is not None}}
        for name, (cols, rows, plot) in tables.items():
            rows = [r for r in rows if all(v is not None for v in r[:len(cols)])] if plot else rows
            t = wandb.Table(columns=list(cols), data=[list(r) for r in rows])
            data[f"{prefix}_tables/{name}"] = t
            if not plot or not rows:
                continue
            kind, x, y, extra = plot
            title = f"{name} (gen {generation})"
            try:
                if kind == "scatter":
                    data[f"{prefix}_plots/{name}"] = wandb.plot.scatter(t, x=x, y=y, title=title)
                elif kind == "bar":
                    data[f"{prefix}_plots/{name}"] = wandb.plot.bar(t, label=x, value=y, title=title)
                elif kind == "line":
                    stroke = extra.split(":", 1)[1] if extra and extra.startswith("stroke:") else None
                    data[f"{prefix}_plots/{name}"] = wandb.plot.line(t, x=x, y=y, stroke=stroke, title=title)
            except Exception as exc:   # a plot helper failing never drops the scalars/tables
                print(f"[wandb_logger] {prefix} plot {name} failed: {exc}")
        self.run.log(data)
        return data


# ─── external eval ────────────────────────────────────────────────────────────

class ExternalEvalRun(_BaseRun):
    """On-demand SVM probe. Results go to the persistent table "SVM sob demanda" (W&B
    INCREMENTAL table + local JSONL). The search NEVER reads these back."""

    job_type = "external_eval"
    COLUMNS = ["parent_id", "dataset", "fidelity", "generation", "macro_f1", "kappa", "balanced_acc",
               "per_class", "time"]

    def __init__(self, parent_id: str, *, dataset: str | None = None,
                 local_path: str | os.PathLike = DEFAULT_EXTERNAL_JSONL, config: Mapping | None = None, **kw):
        cfg = {"parent_id": parent_id, "dataset": dataset, **(config or {})}
        tags = ["external_eval"] + ([_tag("dataset", dataset)] if dataset else [])
        super().__init__(config=cfg, tags=tags, group=dataset or "external_eval",
                         name=f"svm-{parent_id}", **kw)
        self.parent_id = parent_id
        self.dataset = dataset
        self.local_path = Path(local_path)
        self.table = _wandb().Table(columns=self.COLUMNS, log_mode="INCREMENTAL")

    def add_result(self, *, macro_f1: float, kappa: float, balanced_acc: float,
                   per_class: Mapping[str, float] | None = None, fidelity: Any = None,
                   generation: int | None = None) -> list:
        row = [self.parent_id, self.dataset, fidelity, generation, macro_f1, kappa, balanced_acc,
               json.dumps(dict(per_class or {})), time.time()]
        self.table.add_data(*row)
        self.run.log({EXTERNAL_TABLE_NAME: self.table, "svm/macro_f1": macro_f1, "svm/kappa": kappa,
                      "svm/balanced_acc": balanced_acc,
                      **{f"svm/per_class/{k}": v for k, v in (per_class or {}).items()}})
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_path.open("a") as f:
            f.write(json.dumps(dict(zip(self.COLUMNS, row)), default=str) + "\n")
        return row


# ─── dashboard spec ───────────────────────────────────────────────────────────

DASHBOARD_PANELS: dict[str, list[dict[str, Any]]] = {
    "10.1 Search progress (hypervolume)": [
        {"panel": "HV best cumulative (max fidelity)", "type": "line", "x": "generation|gpu_hours|wall_time|n_evaluations", "keys": ["hv/best_cumulative"]},
        {"panel": "HV population (max fidelity)", "type": "line", "x": "generation", "keys": ["hv/population"]},
        {"panel": "HV marginal gain", "type": "bar", "x": "generation", "keys": ["hv/marginal"]},
        {"panel": "dHV per promotion", "type": "line", "x": "generation", "keys": ["promotion/delta_hv_per_promotion", "promotion/n_promotions"]},
        {"panel": "Low-fidelity HV (separate)", "type": "line", "x": "generation", "keys": [f"{LOWFID_PREFIX}hv_population/fid-*", f"{LOWFID_PREFIX}n_points"]},
        {"panel": "HV with CI across search seeds", "type": "line+band", "x": "generation", "keys": ["hv/best_cumulative"], "note": "group controller runs by seed; offline: hv_ci()"},
    ],
    "10.2 Pareto front & diversity": [
        {"panel": "Pareto quality x FLOPs", "type": "scatter", "keys": ["pareto/quality_vs_flops", "pareto/table"], "note": "color by init, marker by 'front'"},
        {"panel": "Per-candidate HV contribution", "type": "table", "keys": ["tables/hv_contribution"]},
        {"panel": "Front diversity", "type": "line", "x": "generation", "keys": ["front/spacing", "front/spread", "front/size"]},
        {"panel": "HV by group", "type": "line", "x": "generation", "keys": ["hv_by_init/*", "hv_by_ssl/*", "hv_by_reg/*", "hv_by_depth/*"]},
    ],
    "10.3 Cost, stagnation & candidates": [
        {"panel": "Stagnation", "type": "line", "x": "n_evaluations", "keys": ["stagnation/moving_improvement", "stagnation/evals_since_last_gain"]},
        {"panel": "Cost per HV unit", "type": "line", "x": "gpu_hours", "keys": ["cost/gpu_hours_per_hv_unit", "cost/gpu_hours_per_hv_gain"]},
        {"panel": "Candidate performance", "type": "line", "filter": "job_type=candidate", "keys": ["perf/dataloader_wait", "perf/latency_per_img", "perf/flops", "cost/gpu_hours_cum"]},
        {"panel": "System metrics (W&B native)", "type": "system", "filter": "job_type=candidate", "keys": ["system/gpu.*", "system/cpu", "system/memory"]},
        {"panel": "Alerts", "type": "run-alerts", "keys": [], "note": f"local copy in {DEFAULT_ALERTS_JSONL}"},
        {"panel": "Encoder checkpoints", "type": "artifacts", "keys": ["artifact type=encoder, aliases gen-N/fid-F/front-rank-R"]},
    ],
    "10.4 External evaluation (SVM sob demanda)": [
        {"panel": EXTERNAL_TABLE_NAME, "type": "table", "filter": "job_type=external_eval", "keys": [EXTERNAL_TABLE_NAME]},
        {"panel": "SVM metrics", "type": "bar", "filter": "job_type=external_eval", "keys": ["svm/macro_f1", "svm/kappa", "svm/balanced_acc", "svm/per_class/*"]},
    ],
}


def export_dashboard_spec(out_dir: str | os.PathLike = "artifacts/modehb/dashboard",
                          project: str = WANDB_PROJECT_MODEHB, entity: str = WANDB_ENTITY,
                          create_report: bool = False) -> dict[str, str]:
    """Write dashboard_spec.json + dashboard_spec.md (every §10.1-10.4 panel with its keys).
    If ``create_report`` and ``wandb_workspaces`` is installed, also save a W&B report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": DASHBOARD_NAME, "entity": entity, "project": project,
            "x_axes": list(ControllerRun.X_AXES), "sections": DASHBOARD_PANELS}
    json_path, md_path = out / "dashboard_spec.json", out / "dashboard_spec.md"
    json_path.write_text(json.dumps(spec, indent=2))
    lines = [f"# {DASHBOARD_NAME}", "", f"Project: `{entity}/{project}`. "
             "Main HV uses only max-fidelity points; low-fidelity HV lives under `lowfid/`.", ""]
    for section, panels in DASHBOARD_PANELS.items():
        lines += [f"## {section}", "", "| Panel | Type | Keys | Notes |", "|---|---|---|---|"]
        for p in panels:
            notes = "; ".join(str(p[k]) for k in ("x", "filter", "note") if k in p)
            lines.append(f"| {p['panel']} | {p['type']} | {', '.join(f'`{k}`' for k in p['keys'])} | {notes} |")
        lines.append("")
    result = {"json": str(json_path), "md": str(md_path)}
    try:
        import wandb_workspaces.reports.v2 as wr  # optional dependency
    except ImportError:
        lines += ["## Manual setup (wandb-workspaces not installed)", "",
                  f"1. Open https://wandb.ai/{entity}/{project} -> Workspace.",
                  "2. For each section above create a panel section with the same name.",
                  "3. Add each panel with the listed keys; set the x-axis as listed (controller metrics are"
                  " defined with step_metric=generation; switch to gpu_hours/wall_time/n_evaluations as needed).",
                  "4. Use run filters `job_type=controller|candidate|external_eval`.",
                  "5. Save as a report named as the title. Or `pip install wandb-workspaces` and call"
                  " `export_dashboard_spec(create_report=True)`."]
    else:
        if create_report:
            blocks = [wr.H1(DASHBOARD_NAME)]
            for section, panels in DASHBOARD_PANELS.items():
                plots = [wr.LinePlot(title=p["panel"], y=[k for k in p["keys"] if "*" not in k])
                         for p in panels if p["type"].startswith("line") and any("*" not in k for k in p["keys"])]
                blocks += [wr.H2(section), wr.PanelGrid(runsets=[wr.Runset(entity=entity, project=project)], panels=plots)]
            report = wr.Report(entity=entity, project=project, title=DASHBOARD_NAME, blocks=blocks)
            report.save()
            result["report_url"] = report.url
    md_path.write_text("\n".join(lines) + "\n")
    return result
