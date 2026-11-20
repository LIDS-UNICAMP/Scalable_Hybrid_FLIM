"""Synthetic GA simulation: v3 vs v4. kappa grows with params (+noise), big models OOM in fp32."""
import sys, json, math, time, tempfile, collections, warnings
warnings.filterwarnings("ignore")
REPO = sys.argv[1]; sys.path.insert(0, REPO)
import numpy as np, yaml
from ga_search.modehb import modehb as M, strata as S, run_search as RS
M.OOM_RETRY_SLEEP_S = 0
B = [("B0", 3e5), ("B1", 8e5), ("B2", 1.5e6), ("B3", 3e6), ("B4", 5.5e6)]
def band(p): return next((n for n, hi in B if p <= hi), ">cap")
class OOM(RuntimeError): pass
OOM.__name__ = "OutOfMemoryError"
GPU_REAL = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
def make_eval(seed):
    rng = np.random.default_rng(seed)
    def ev(g, fid, ctx):
        p = S.param_estimate(g) * float(rng.uniform(0.92, 1.0))
        real = 0.62 * S.mem_estimate_gb(g, 224, n_classes=21)   # measured median peak/est
        fb = ctx.get("mem_fallback") or {}
        if fb.get("amp_dtype"): real *= 0.6
        real /= max(1, int(fb.get("batch_div", 1) or 1)) ** 0.7
        if real > GPU_REAL:
            raise OOM("CUDA out of memory (synthetic)")
        q = 0.40 + 0.06 * math.log10(max(p, 1e4) / 1e4) + 0.02 * math.log2(fid / 5) / 5 + rng.normal(0, 0.02)
        return {"status": "ok", "quality": float(min(max(q, 0), 1)), "flops": p * 400.0, "params": int(p),
                "latency_ms": 1.0, "gpu_hours": 0.01, "checkpoint": None}
    return ev
def run(cfg_path, label, brackets, scale):
    raw = yaml.safe_load(open(cfg_path))
    raw.pop("lemonade", None); raw.pop("external_seed", None)
    cfg = RS.from_search_yaml(raw, "remote_sensing", 42)["search"]
    cfg.update(max_brackets=brackets, n_configs_scale=scale)
    cfg["stratified"] = {**cfg["stratified"], "memory_budget_gb": 24, "calibration": False, "oom_retry_sleep_s": 0}
    out = tempfile.mkdtemp()
    m = M.MODEHB(cfg, make_eval(0), out)
    t = time.time(); st = m.run()
    jobs = [j for j in st["jobs"] if j["role"] == "search"]
    gen = collections.defaultdict(collections.Counter); pro = collections.defaultdict(collections.Counter)
    top = collections.Counter(); fails = collections.Counter(); fb = 0
    for j in jobs:
        p = j["result"].get("params") or j.get("params_est") or 0
        op = (j.get("op") or {}).get("type")
        if j["status"] != "ok": fails[band(j.get("params_est") or 0)] += 1; continue
        if j["result"].get("mem_fallback"): fb += 1
        if op == "promotion": pro[f'{j["fidelity"]:g}'][band(p)] += 1
        else: gen[f'b{j["generation"]}'][band(p)] += 1
        if j["fidelity"] == m.fidelities[-1]: top[band(p)] += 1
    res = {"label": label, "secs": round(time.time() - t), "n_jobs": len(jobs),
           "generated_by_bracket": {k: dict(sorted(v.items())) for k, v in gen.items()},
           "promoted_into_fid": {k: dict(sorted(v.items())) for k, v in pro.items()},
           "at_max_fidelity": dict(sorted(top.items())), "failed_by_band": dict(fails),
           "ok_after_oom_fallback": fb,
           "best_q_max_fid_band": max(((j["result"]["quality"], band(j["result"]["params"])) for j in jobs
                                       if j["status"] == "ok" and j["fidelity"] == m.fidelities[-1]), default=None)}
    print(json.dumps(res), flush=True)
    return res
r = [run(f"{REPO}/ga_search/config/modehb/search_v3.yaml", "v3", 3, 1.0),
     run(f"{REPO}/ga_search/config/modehb/search_v4.yaml", "v4", 3, 1.0)]
json.dump(r, open(sys.argv[2], "w"), indent=1)
