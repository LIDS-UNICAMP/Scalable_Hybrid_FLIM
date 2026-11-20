"""Entrada do H-MODEHB B0-B4.

    python -m ga_search.hmodehb.run_hsearch --arm A --dataset remote_sensing [--seed 42]
        [--config ga_search/config/hmodehb_b0b4/base.yaml] [--set a.b=VALOR] [--allow-drift chave]

Config = ``base.yaml`` (schema I11 do search_v3, autocontido) + ``arms/<braco>.yaml`` (merge profundo)
+ ``--set``. O bloco ``hmodehb:`` do YAML (oom, allow_drift) vai para ``cfg["hmodehb"]``. Diretorio:
``artifacts/hmodehb_b0b4/arm_<braco>/<dataset>/state/seed<k>`` (sem as substrings lemonade/seeded/pnas_cs,
que o monitor_report usaria para classificar o run).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ga_search.hmodehb import bk as BK
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import integrity as I
from ga_search.modehb import run_search as RS

CONFIG_DIR = Path("ga_search/config/hmodehb_b0b4")
BASE = CONFIG_DIR / "base.yaml"
H_DEFAULTS = {"oom": {"retries": 4, "sleep_s": 180}, "allow_drift": []}   # R5: 4 x 180 s


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def load_raw(arm: str, base=BASE, arms_dir=None) -> dict:
    I.check_arm(arm)
    raw = yaml.safe_load(Path(base).read_text()) or {}
    arm_path = Path(arms_dir or Path(base).parent / "arms") / f"{arm}.yaml"
    if not arm_path.exists():
        raise FileNotFoundError(f"braco {arm}: {arm_path} nao existe (bracos D0/D/E dependem dos gates A2/A3)")
    return deep_merge(raw, yaml.safe_load(arm_path.read_text()) or {})


def load_hconfig(arm: str, dataset: str, seed=None, overrides=(), base=BASE, arms_dir=None) -> dict:
    """Config resolvido (mesmo schema do run_search.load_config) + ``cfg["hmodehb"]``."""
    raw = load_raw(arm, base, arms_dir)
    h = deep_merge(H_DEFAULTS, raw.pop("hmodehb", None) or {})
    if (raw.get("pnas_cs") or {}).get("enabled"):
        raise NotImplementedError("braco C (PNAS-CS) ainda nao suportado no H-MODEHB (niveis != B0-B4)")
    paths = raw.setdefault("paths", {})
    paths["state_dir"] = str(paths.get("state_dir", "artifacts/hmodehb_b0b4/arm_{arm}/{dataset}/state")).replace(
        "{arm}", arm)
    if "dataset_config_dir" in raw:            # mesma traducao do run_search.load_config
        raw = RS.from_search_yaml(raw, dataset, seed)
        raw["eval"] = {**raw.get("eval", {}), **raw.pop("eval_fitness", {})}
    elif seed is not None:
        raw["seed"] = int(seed)
    for item in overrides or ():
        if item.startswith("hmodehb."):
            RS._set_dotted(h, item[len("hmodehb."):])
        else:
            RS._set_dotted(raw, item)
    cfg = copy.deepcopy(RS.DEFAULTS)
    for k, v in raw.items():
        cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["search"] = {**cfg["search"], "seed": int(cfg["seed"]),
                     "image_size": int(cfg["data"].get("image_size", 200))}
    cfg["search"]["stratified"] = BK.stratified_block(cfg["search"].get("stratified"))
    cfg["search"]["objective_bounds"] = {"flops_lo": FV.FLOPS_BOUNDS[0], "flops_hi": FV.FLOPS_BOUNDS[1]}
    cfg["search"]["use_latency"] = False
    FV.validate_eval(cfg["eval"])
    h.update(arm=arm, kind=f"h4_{arm}", id_prefix=f"{I.ID_PREFIX}-{arm}-")
    h["allow_drift"] = list(h.get("allow_drift") or [])
    cfg["hmodehb"] = h
    return cfg


def controller_id(cfg: dict, fitness_sig: str) -> str:
    """Id do controller W&B: diretorio + braco + REGUA (outra regua = outro controller, nunca curvas misturadas)."""
    key = f"{Path(cfg['output_dir']).resolve()}|{cfg['hmodehb']['arm']}|{fitness_sig}"
    return "hctrl-" + hashlib.sha1(key.encode()).hexdigest()[:16]


def hcontroller(cfg: dict, fitness_sig: str):
    w = cfg.get("wandb") or {}
    if not w.get("enabled"):
        return None
    from ga_search.modehb import wandb_logger as WL
    arm, ds = cfg["hmodehb"]["arm"], cfg.get("dataset_name") or ""
    kw = {k: w[k] for k in ("project", "entity", "mode") if w.get(k)}
    return WL.ControllerRun(ref_point=[float(cfg["search"].get("ref_point", 1.1))] * 2,
                            max_fidelity=float(cfg["search"]["fidelities"][-1]),
                            name=f"h4-controller-{ds}-{arm}-v2_{fitness_sig}",
                            group=f"hmodehb_b0b4/{ds}",
                            tags=["controller", "hmodehb_b0b4", "fitness_v2", f"arm_{arm}", f"fsig_{fitness_sig}"],
                            config={**cfg, "fitness_sig": fitness_sig}, id=controller_id(cfg, fitness_sig),
                            resume="allow", **kw)


def build(cfg: dict, evaluator, *, allow_drift=(), lock=None, logger=None, param_fn=None):
    """HMODEHB pronto para ``run`` (sem efeitos alem do diretorio). Separado do main para os testes."""
    from ga_search.hmodehb.hsearch import HMODEHB
    spifil_repair = None
    if (cfg.get("spifil") or {}).get("spifil_ops", False):
        import functools
        from ga_search.modehb.init import repair_for_spifil
        spifil_repair = functools.partial(repair_for_spifil, spifil_ops=True)
    h = cfg["hmodehb"]
    return HMODEHB(cfg["search"], evaluator, cfg["output_dir"], arm=h["arm"], fitness_spec=evaluator.fitness_spec,
                   full_config=cfg, allow_drift=list(h.get("allow_drift") or []) + list(allow_drift or []),
                   oom_retries=int(h["oom"]["retries"]), oom_sleep_s=float(h["oom"]["sleep_s"]), lock=lock,
                   logger=logger, spifil_repair=spifil_repair, param_fn=param_fn)


def write_outputs(search, state, out: Path):
    top = float(search.fidelities[-1])
    summary = {"n_jobs": len(state["jobs"]), "gpu_hours": state["gpu_hours"], "fitness_sig": search.fitness_sig,
               "config_sha": search.config_sha, "arm": search.arm,
               "pareto_max_fidelity": [{"cand_id": j["cand_id"], "bk": j.get("bk"), "F": j["F"],
                                        "checkpoint": j["result"].get("checkpoint")} for j in search.front(top)],
               "brackets": state["brackets_log"]}
    I.atomic_write_json(out / "summary.json", summary)
    I.atomic_write_json(out / "best_by_bk.json", search.best_by_bk(top))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=I.ARMS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config", default=str(BASE))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=VALOR")
    ap.add_argument("--allow-drift", action="append", default=[], metavar="chave[,chave]",
                    help="chaves de config que podem mudar no resume (registradas em drift_log)")
    args = ap.parse_args(argv)
    torch.multiprocessing.set_sharing_strategy("file_system")
    RS._cap_cpu_threads()
    cfg = load_hconfig(args.arm, args.dataset, args.seed, args.set, base=args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lock = I.ControllerLock(out).acquire()      # antes de carregar dados: o 2o controlador sai na hora
    try:
        resolved = out / "config_resolved.yaml"
        if not resolved.exists():
            resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        torch.manual_seed(int(cfg["seed"]))
        np.random.seed(int(cfg["seed"]))
        from ga_search.hmodehb.hevaluate import HEvaluator
        evaluator = HEvaluator(cfg, out)
        if cfg["search"].get("external_seed"):
            from ga_search.modehb.external_seed import prepare
            cfg["search"]["external_seed"] = prepare(cfg["search"]["external_seed"], out)
        cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(evaluator.spifil_pool.items())]
        controller = hcontroller(cfg, evaluator.fitness_sig)
        allow = [k for a in args.allow_drift for k in a.split(",") if k]
        search = build(cfg, evaluator, allow_drift=allow, lock=lock, logger=controller)
        resume = args.resume or search.state_path.exists()
        state = search.run(resume=resume)
        summary = write_outputs(search, state, out)
        if controller is not None:
            controller.finish()
        print(json.dumps({k: summary[k] for k in ("n_jobs", "gpu_hours", "fitness_sig")}))
    finally:
        lock.release()


if __name__ == "__main__":
    main()
