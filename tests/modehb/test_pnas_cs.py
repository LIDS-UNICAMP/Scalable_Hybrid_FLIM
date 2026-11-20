"""PNAS-CS (ga_search/modehb/pnas_cs.py): compound scaling, growth through the external_seed morphisms,
mutations, P-level promotions, Phase 0, bf16 path, W&B panels, resume, features-off == original behaviour."""
import copy
import functools
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from ga_search.modehb import external_seed as ES
from ga_search.modehb import modehb as M
from ga_search.modehb import pnas_cs as PC
from ga_search.modehb import run_search as RS
from ga_search.modehb import strata as S
from ga_search.modehb.init import repair_for_spifil
from ga_search.modehb.spifil_ops import random_twin_ops, to_ops_spec

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "ga_search" / "config" / "modehb"
SPIFIL_REPAIR = functools.partial(repair_for_spifil, spifil_ops=True)
PC_CFG = PC.pnas_config({"enabled": True})
CAP = PC.cap(PC_CFG)


def _cfg(overrides=()):
    cfg = RS.load_config(CFG_DIR / "pnas_cs.yaml", "remote_sensing", None,
                         ["search.fidelities=[1,3,9]", "search.max_brackets=3", "search.initial_size=8",
                          "search.pnas_cs.phase0.fidelity=3", "search.pnas_cs.seeds.external.enabled=false",
                          "search.pnas_cs.seeds.checkpoints={}", "search.pnas_cs.seeds.best_from=[]",
                          "search.stratified.memory_budget_gb=400", "search.external_seed.enabled=false",
                          *overrides])
    c = cfg["search"]
    c["spifil_pool_sizes"] = [32] * 14
    return c


def _rep():
    c = _cfg()
    return functools.partial(M.repair, image_size=224, channel_space=c["channel_space"],
                             search_space=M.normalize_search_space(c["search_space"]), spifil_repair=SPIFIL_REPAIR)


def _genome(widths, pools, conv="conv", kernel=3, act="relu", norm="none", residual=None, pooling="gem"):
    layers = [{"channels": w, "kernel": kernel, "conv": conv if i else "conv", "residual": bool(residual and residual[i]),
               "activation": act, "norm": norm, "pool": p, "softhebb": False} for i, (w, p) in enumerate(zip(widths, pools))]
    while len(layers) < M.N_LAYERS:
        layers.append({**layers[-1], "pool": "none"})
    return _rep()({"init": "spifil", "seed": 42, "depth": len(widths), "layers": layers, "ssl": "lejepa", "reg": "sigreg",
                   "embed_dim": 1280, "pooling": pooling, "ae_target": "pixel", "projector": False, "protect": "lowlr",
                   "ae_schedule": "decoder_first", "spifil_images_per_class": 16})


SEED_B = dict(widths=[40, 80, 152, 152, 152, 312], pools=["max"] * 4 + ["none"] * 2)


class _Eval:
    """Synthetic evaluator: quality from the parameter count, fake checkpoints, records ctx."""

    def __init__(self):
        self.ctxs = []

    def __call__(self, g, fid, ctx):
        self.ctxs.append((copy.deepcopy(g), fid, {k: v for k, v in ctx.items() if k != "front_rank"}))
        p = S.param_estimate(g)
        q = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3))) * (1 - math.exp(-fid / 3))
        q += 0.001 * (int(M.arch_hash(g)[:4], 16) % 97) / 97
        return {"status": "ok", "quality": q, "fitness_kappa": q, "flops": p * 3000, "params": int(p),
                "gpu_hours": 0.01, "checkpoint": f"ck/{ctx['cand_id']}_f{int(fid)}.pt",
                "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]]}


class _Logger:
    def __init__(self):
        self.pnas, self.gen, self.strata = [], [], []

    def log_generation(self, pts, *a, **k):
        self.gen.append((pts, k))

    def log_strata(self, *a, **k):
        self.strata.append(a)

    def log_pnas(self, generation, gpu_hours, n_evals, scalars, tables):
        self.pnas.append((generation, n_evals, dict(scalars), dict(tables)))


# ───────────────────────────────────────────────────────────── config / levels
def test_config_absent_disabled_invalid():
    assert PC.pnas_config(None) is None and PC.pnas_config({"enabled": False}) is None
    for bad in ({"budgets": [2, 1]}, {"deepen": "x"}, {"mutations": ["prune"]}, {"betas": [1.0]},
                {"grow_prob": 0.8, "mutate_prob": 0.4}):
        with pytest.raises(ValueError):
            PC.pnas_config({"enabled": True, **bad})


def test_levels_contiguous_cap():
    st = PC.level_strata(PC_CFG)
    assert list(st) == ["P0", "P1", "P2", "P3", "P4"] and CAP == 5_775_000
    edges = [v for lohi in st.values() for v in lohi]
    assert edges[0] == 0 and edges[-1] == CAP and all(a == b for a, b in zip(edges[1::2], edges[2::2]))
    for b, (lo, hi) in zip(PC_CFG["budgets"], st.values()):
        assert lo < b <= hi
    sc = S.strat_config({"strata": st, "param_cap": CAP})
    assert S.stratum_of(1_383_536, sc) == "P1" and S.stratum_of(5_500_000, sc) == "P4" and S.stratum_of(CAP + 1, sc) is None


def test_pnas_yaml_is_search_v3_plus_documented_changes():
    a = yaml.safe_load((CFG_DIR / "search_v3.yaml").read_text())
    b = yaml.safe_load((CFG_DIR / "pnas_cs.yaml").read_text())
    changed = {"datasets", "max_brackets", "pnas_cs", "paths", "resources", "stratified", "training", "stages",
               "external_seed"}
    assert set(b) - set(a) == {"pnas_cs", "external_seed"}
    seeded = yaml.safe_load((CFG_DIR / "search_v3_seeded.yaml").read_text())["external_seed"]
    assert {k: v for k, v in b["external_seed"].items() if k not in ("grid", "genes")} == \
        {k: v for k, v in seeded.items() if k not in ("grid", "genes")}
    assert b["external_seed"]["grid"]["width_mults"] == [1.0] and b["external_seed"]["grid"]["extra_depths"] == [0]
    for k in set(a) - changed:
        assert a[k] == b[k], k
    sa, sb = a["stratified"], b["stratified"]
    assert {k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)} == {
        "strata", "param_cap", "explore_strata", "projection_prob", "memory_budget_gb"}
    assert {k for k in a["training"] if a["training"][k] != b["training"].get(k)} == set()
    assert set(b["training"]) - set(a["training"]) == {"amp_dtype"} and b["training"]["amp_dtype"] == "bf16"
    s2a, s2b = a["stages"]["stage2"]["space"], b["stages"]["stage2"]["space"]
    assert {k: v for k, v in s2b.items() if k not in ("ssl", "reg")} == s2a
    assert s2b["ssl"] == ["lejepa"] and s2b["reg"] == ["sigreg"]
    assert b["resources"]["gpu_process_cap_gb"] == 44 and b["paths"]["state_dir"].startswith("artifacts/pnas_cs/")
    c = RS.load_config(CFG_DIR / "pnas_cs.yaml", "remote_sensing", None, [])
    assert c["output_dir"] == "artifacts/pnas_cs/remote_sensing/state/seed42"
    assert c["train"]["amp_dtype"] == "bf16" and c["search"]["pnas_cs"]["dataset"] == "remote_sensing"
    for name in ("search_v3.yaml", "search_v3_lemonade.yaml"):    # features off in the existing configs
        c = RS.load_config(CFG_DIR / name, "remote_sensing", None, [])
        assert "pnas_cs" not in c["search"] and "amp_dtype" not in c["train"]


# ───────────────────────────────────────────────────────────── compound scaling
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.1), (1.2, 1.1), (1.4, 1.3), (1.1, 1.05)])
def test_fit_level_hits_every_budget_within_tolerance(alpha, beta):
    rep = _rep()
    base = _genome(**SEED_B)
    for name, budget in zip(PC_CFG["level_names"], PC_CFG["budgets"]):
        g, info = PC.fit_level(base, alpha, beta, budget, S.param_estimate, rep, cap_params=CAP)
        p = S.param_estimate(g)
        assert abs(p / budget - 1) <= 0.10, (name, p, info)
        assert info["within_tol"] and p <= CAP and int(g["depth"]) <= 8 and g == rep(g)
        if alpha == 1.0:
            assert int(g["depth"]) == int(base["depth"])


def test_compound_chain_grows_monotonically_and_is_morphable():
    rep = _rep()
    base = _genome([32, 64, 128], ["max", "max", "none"])
    prev, prev_info = None, None
    for budget in PC_CFG["budgets"]:
        mm = (prev_info["d_mult"], prev_info["w_mult"]) if prev_info else (None, None)
        g, info = PC.fit_level(base, 1.2, 1.1, budget, S.param_estimate, rep, cap_params=CAP,
                               min_params=S.param_estimate(prev) if prev else None, min_mults=mm)
        assert abs(S.param_estimate(g) / budget - 1) <= 0.10
        if prev is not None:
            lm = PC.layer_map(prev, g)
            assert lm is not None, (PC.active(prev), PC.active(g))
            assert len(lm) == g["depth"] and sum(isinstance(q, int) for q in lm) == prev["depth"]
            assert all(q in ("identity", "residual_zero") for q in lm if not isinstance(q, int))
        prev, prev_info = g, info


def test_stage_additions_monotone_under_depth_cap():
    base = _genome([16, 32, 64, 64], ["max", "max", "max", "none"])
    last = None
    for d in np.linspace(1.0, 6.0, 60):
        a = PC.stage_additions(base, d, max_depth=8)
        assert sum(a) + base["depth"] <= 8
        if last is not None:
            assert all(x >= y for x, y in zip(a, last))
        last = a
    assert PC.stage_additions(base, 0.4) == [0, 0, 0, 0]          # pooled single-layer stages keep 1


def test_fit_level_respects_cap_and_feasibility():
    rep = _rep()
    base = _genome(**SEED_B)
    g, info = PC.fit_level(base, 1.2, 1.1, 50_000_000, S.param_estimate, rep, cap_params=CAP)
    assert S.param_estimate(g) <= CAP and not info["within_tol"]
    small = lambda x: S.param_estimate(x) < 900_000                                  # noqa: E731
    g, _ = PC.fit_level(base, 1.2, 1.1, 2_000_000, S.param_estimate, rep, cap_params=CAP, feasible=small)
    assert S.param_estimate(g) < 900_000


# ───────────────────────────────────────────────────────────── growth through external_seed (function preservation)
def _plain_encoder(layers, seed=0, flim_pool=False):
    torch.manual_seed(seed)
    blocks, cin = [], 3
    for i, L in enumerate(layers):
        spec = to_ops_spec({k: v for k, v in L.items() if k != "active"}, cin)
        if flim_pool and L["pool"] == "max":
            spec.update(pool_size=3, pool_padding=0)
        b = random_twin_ops(spec, seed + i, in_channels=cin)
        blocks.append(b)
        cin = b.out_channels
    return nn.Sequential(*blocks).eval()


def _rand_filters(i, prefix_blocks, L, cin, n_new, kernel):
    g = torch.Generator().manual_seed(100 + i)
    return torch.randn(n_new, cin, kernel, kernel, generator=g) * 0.1, torch.randn(n_new, generator=g) * 0.1


@pytest.mark.parametrize("flim_pool", [False, True])
def test_compound_growth_chain_is_function_preserving(flim_pool):
    """P0 -> P1 -> P2 -> P3 of a plain relu family through external_seed.grow_blocks (SPiFiL new filters
    replaced by random ones): every grown child reproduces its parent's features on the old channels."""
    rep = _rep()
    base = _genome([16, 32, 48], ["max", "max", "none"], act="relu")
    g0, info = PC.fit_level(base, 1.3, 1.1, 300_000, S.param_estimate, rep, cap_params=CAP)
    parent, enc = g0, _plain_encoder(PC.active(g0), flim_pool=flim_pool)
    x = torch.rand(2, 3, 64, 64)
    for budget in (600_000, 1_200_000, 2_400_000):
        child, info2 = PC.fit_level(base, 1.3, 1.1, budget, S.param_estimate, rep, cap_params=CAP,
                                    min_params=S.param_estimate(parent), min_mults=(info["d_mult"], info["w_mult"]))
        lm = PC.layer_map(parent, child)
        assert lm is not None
        grown = ES.grow_blocks(list(enc), ES.active_layers(parent), ES.active_layers(child), lm,
                               new_filters=_rand_filters)
        assert len(grown.blocks) == child["depth"], [lg["prov"]["origin"] for lg in grown.logs]
        err = ES.preservation_error(list(enc), grown, x)
        assert err is not None and err < 1e-5, err
        parent, info, enc = child, info2, nn.Sequential(*grown.blocks).eval()


def test_zero_residual_growth_step_is_function_preserving():
    """silu family: depth growth inserts zero-init residual blocks (x + silu(0*x)) -> exact at step 0."""
    rep = _rep()
    parent = _genome([16, 32, 48], ["max", "max", "none"], act="silu")
    enc = _plain_encoder(PC.active(parent))
    child = rep(PC.scale_genome(parent, 2.0, 1.2))
    lm = PC.layer_map(parent, child)
    assert lm is not None and "residual_zero" in lm
    grown = ES.grow_blocks(list(enc), ES.active_layers(parent), ES.active_layers(child), lm, new_filters=_rand_filters)
    assert len(grown.blocks) == child["depth"]
    assert ES.preservation_error(list(enc), grown, torch.rand(2, 3, 64, 64)) < 1e-5


# ───────────────────────────────────────────────────────────── plans and mutations
def test_layer_map_cases():
    a = _genome([32, 64, 128], ["max", "max", "none"])
    assert PC.layer_map(a, a) is None
    w = copy.deepcopy(a)
    w["layers"][1]["channels"] = 96
    w = _rep()(w)
    assert PC.layer_map(a, w) == [0, 1, 2]
    assert PC.layer_map(a, _rep()({**copy.deepcopy(a), "pooling": "gap"})) == [0, 1, 2]   # head-only change
    s = copy.deepcopy(a)
    s["layers"][0]["pool"] = "avg"
    assert PC.layer_map(a, _rep()(s)) is None                   # stage pool change: not a pure growth
    d = PC.insert_layer(a, PC.slots(a)[1], "identity")
    assert PC.layer_map(a, _rep()(d)) == [0, "identity", 1, 2]
    r = PC.insert_layer(a, PC.slots(a)[-1], "residual")
    assert PC.layer_map(a, _rep()(r)) == [0, 1, 2, "residual_zero"]


@pytest.mark.parametrize("op", PC.MUTATIONS)
def test_mutations_valid_under_cap(op):
    rng = np.random.default_rng(3)
    rep = _rep()
    n_ok = 0
    for t in range(30):
        base = _genome([16 + 8 * (t % 5), 48, 96, 96], ["max", "max", "none", "none"],
                       act=["relu", "gelu", "silu"][t % 3])
        m = PC.mutate(base, op, rng, PC_CFG, poolings=("gap", "gem", "moments"), pools=("max", "avg"))
        if m is None:
            continue
        child = rep(m[0])
        assert child == rep(child) and int(child["depth"]) <= 8 and M.arch_hash(child) != M.arch_hash(base)
        assert S.param_estimate(child) <= CAP or S.param_estimate(base) > CAP
        lm = PC.layer_map(base, child)
        if op in ("add_layer", "add_residual", "widen"):
            assert lm is not None
        if op == "pooling" and m[1]["kind"] == "final":
            assert lm == list(range(int(base["depth"])))
        if op == "add_layer":
            assert "identity" in lm
        if op == "add_residual":
            assert "residual_zero" in lm
        n_ok += 1
    assert n_ok >= 5


# ───────────────────────────────────────────────────────────── MO-DEHB integration
def _run(tmp, overrides=(), logger=None, ev=None, **kw):
    ev = ev or _Eval()
    s = M.MODEHB({**_cfg(overrides), **kw}, ev, tmp, spifil_repair=SPIFIL_REPAIR, logger=logger)
    return s, s.run(resume=(Path(tmp) / "search_state.json").exists()), ev


def test_search_covers_ops_levels_promotions_and_phase0(tmp_path):
    log = _Logger()
    s, st, ev = _run(tmp_path, ["search.max_brackets=4", "search.pnas_cs.grow_prob=0.35",
                                "search.pnas_cs.mutate_prob=0.45"], logger=log)
    assert s.dim == M.DIM + 2 and list(s.strat["strata"][i][0] for i in range(5)) == ["P0", "P1", "P2", "P3", "P4"]
    ops = {(j.get("op") or {}).get("type") for j in st["jobs"]}
    assert {"phase0", "calibration", "seed", "grow", "mutation", "promotion", "de"} <= ops
    muts = {(j.get("op") or {}).get("mutation") for j in st["jobs"] if (j.get("op") or {}).get("type") == "mutation"}
    assert {"add_residual", "widen", "pooling"} <= muts
    assert {j.get("stratum") for j in st["jobs"] if j["role"] == "search"} >= {"P0", "P1", "P2", "P3", "P4"}
    assert all(S.param_estimate(j["genome"]) <= CAP for j in st["jobs"])
    # Phase 0: 3 arms x n seeds, AE off only there, schedule fixed afterwards
    p0 = [j for j in st["jobs"] if j["role"].startswith("phase0")]
    assert len(p0) == 6 and st["phase0"]["chosen_ae_schedule"] in ("decoder_first", "joint")
    off = [ctx for g, f, ctx in ev.ctxs if ctx.get("ae_mode") == "off"]
    assert len(off) == 2 and all(ctx["role"] == "phase0_noae" for ctx in off)
    later = [g for g, f, ctx in ev.ctxs if not ctx["role"].startswith("phase0")]
    assert {g["ae_schedule"] for g in later} == {st["phase0"]["chosen_ae_schedule"]}
    # growth inherits a net2net plan from a trained parent of the same family, one level below
    grows = [(g, ctx) for g, f, ctx in ev.ctxs if (ctx.get("inherit") or {}).get("layer_map") is not None]
    assert grows
    for g, ctx in grows:
        inh = ctx["inherit"]
        assert inh["checkpoint"].startswith("ck/") and len(inh["layer_map"]) == g["depth"]
        assert inh["parent_layers"] and not g.get("external")
    gj = [j for j in st["jobs"] if (j.get("op") or {}).get("type") == "grow"]
    assert any(j["op"]["morph"] == "net2net" for j in gj)
    for j in gj:
        assert j["genome"]["pnas"]["family"] == j["op"]["family"]
        lv = PC.level_index(PC_CFG, j["op"]["to_level"])
        assert lv == PC.level_index(PC_CFG, j["op"]["from_level"]) + 1
    # protected promotions by level: in the first protected bracket every level present is promoted
    b0 = [j for j in st["jobs"] if j["generation"] == 0]
    rung0 = {j.get("stratum") for j in b0 if j["rung"] == 0 and j["status"] == "ok"}
    prom = {(j.get("op") or {}).get("stratum") for j in b0 if (j.get("op") or {}).get("type") == "promotion"}
    assert len(prom & rung0) >= min(len(rung0), 3)
    # W&B panels (fake logger) + local files
    assert log.pnas, "log_pnas never called"
    names = set().union(*(set(t) for _, _, _, t in log.pnas))
    assert names >= {"hv_vs_gpu_hours", "kappa_vs_params", "pareto_kappa_flops", "pareto_evolution", "growth_curves",
                     "depth_x_width", "marginal_gain", "mutation_survival", "scaling_law", "phase0_ae_ablation"}
    assert all("pareto_kappa_flops" in t for _, _, _, t in log.pnas)        # Pareto: every evaluation
    last = log.pnas[-1][2]
    assert "pnas/hv_highest_reached" in last and any(k.startswith("pnas/survival/") for k in last)
    rep = json.loads((tmp_path / "pnas" / "report.json").read_text())
    assert rep["tables"]["pareto_kappa_flops"][1] and (tmp_path / "pnas" / "pareto_front.json").exists()
    assert len(st["pnas_hv_history"]) == len(st["jobs"])


def test_resume_identical_no_replay_mismatch(tmp_path):
    full_dir, part_dir = tmp_path / "full", tmp_path / "part"
    _, full, _ = _run(full_dir)
    _, part, _ = _run(part_dir, max_evaluations=40)
    assert len(part["jobs"]) == 40
    s, res, ev = _run(part_dir)                          # restart = replay + continue
    assert [j["arch_hash"] for j in res["jobs"]] == [j["arch_hash"] for j in full["jobs"]]
    assert [j.get("op") for j in res["jobs"]] == [j.get("op") for j in full["jobs"]]
    assert len(ev.ctxs) == len(full["jobs"]) - 40        # only the new jobs were evaluated


def test_existing_configs_identical_with_features_off(tmp_path):
    """search_v3 / LEMONADE / plain MO-DEHB produce the same job sequence as the ORIGINAL code
    (golden recorded before the PNAS-CS patch, same fake evaluator)."""
    gold = json.loads((Path(__file__).parent / "data" / "golden_features_off.json").read_text())

    def fake(g, fid, ctx):
        p = S.param_estimate(g)
        q = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3))) * (1 - math.exp(-fid / 3)) \
            + 0.001 * (int(M.arch_hash(g)[:4], 16) % 97) / 97
        return {"status": "ok", "quality": q, "flops": p * 3000, "params": int(p), "gpu_hours": 0.0,
                "checkpoint": f"ck/{ctx['cand_id']}_f{int(fid)}.pt",
                "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]]}

    for name, path in (("search_v3", "search_v3.yaml"), ("lemonade", "search_v3_lemonade.yaml"),
                       ("seeded_off", "search_v3_seeded.yaml"), ("lemonade_seeded_off", "search_v3_lemonade_seeded.yaml")):
        cfg = RS.load_config(CFG_DIR / path, "remote_sensing", None,
                             ["search.fidelities=[1,3,9]", "search.max_brackets=4", "search.initial_size=8"])
        c = cfg["search"]
        c["spifil_pool_sizes"] = [32] * 14
        c.pop("external_seed", None)
        s = M.MODEHB(c, fake, tmp_path / name, spifil_repair=SPIFIL_REPAIR)
        assert s.pnas is None and s.dim == M.DIM
        got = [[j["arch_hash"], j["fidelity"], j["role"], j["cand_id"]] for j in s.run()["jobs"]]
        assert got == gold[name], name
    s = M.MODEHB(dict(fidelities=[1, 3, 9], eta=3, max_brackets=3, seed=7, image_size=64), fake, tmp_path / "plain")
    assert [[j["arch_hash"], j["fidelity"], j["role"], j["cand_id"]] for j in s.run()["jobs"]] == gold["plain"]


# ───────────────────────────────────────────────────────────── seeds
def test_discover_seeds_missing_files_and_frozen_manifest(tmp_path, capsys):
    pc = PC.pnas_config({"enabled": True, "seeds": {
        "external": {"dir": str(tmp_path / "nope" / "{dataset}")},
        "checkpoints": {"remote_sensing": [str(tmp_path / "missing.pt")]}, "best_from": [str(tmp_path / "x.json")],
        "best_k": 1}})
    out = PC.discover_seeds(pc, "remote_sensing", tmp_path / "o", 42)
    assert out == [] and "WARNING" in capsys.readouterr().out
    man = tmp_path / "o" / "pnas_seeds.json"
    assert json.loads(man.read_text()) == []
    man.write_text(json.dumps([{"kind": "external", "name": "frozen"}]))
    assert PC.discover_seeds(pc, "remote_sensing", tmp_path / "o", 42)[0]["name"] == "frozen"


# ───────────────────────────────────────────────────────────── bf16 path
def test_amp_dtype_resolution():
    assert RS._amp_dtype(None, torch.device("cpu")) is None and RS._amp_dtype("none", torch.device("cuda")) is None
    assert RS._amp_dtype("bf16", torch.device("cpu")) is None                 # CPU runs stay fp32
    with pytest.raises(ValueError):
        RS._amp_dtype("fp16", torch.device("cpu"))


def test_whitening_runs_fp32_under_bf16_autocast():
    from ga_search.modehb.ops_norm_act import DecorrelatedBatchNorm, ZCAWhitening
    torch.manual_seed(0)
    x = torch.randn(8, 6, 5, 5)
    z = ZCAWhitening(6).fit(x)
    d = DecorrelatedBatchNorm(6).eval()
    ref_z, ref_d = z(x), d(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        oz, od = z(x.bfloat16()), d(x.bfloat16())
    assert oz.dtype == torch.float32 and od.dtype == torch.float32
    assert float((oz - ref_z).abs().max()) < 0.05 and float((od - ref_d).abs().max()) < 0.05
    with torch.autocast("cpu", dtype=torch.float16, enabled=False):            # unchanged without bf16
        assert z(x).dtype == torch.float32


def _cpu_autocast_on():
    try:
        return torch.is_autocast_enabled("cpu")       # torch >= 2.4
    except TypeError:
        return torch.is_autocast_cpu_enabled()        # torch 2.2 (athena)


def test_ssl_bf16_losses_fp32(monkeypatch):
    from ga_search.modehb import ssl_trainers as T
    from ga_search.modehb.pooling_head import EmbeddingHead
    seen = []
    real = T.RealSIGReg

    class Spy(real):
        def forward(self, H, *a, **k):
            seen.append((H.dtype, _cpu_autocast_on()))
            return super().forward(H, *a, **k)
    monkeypatch.setattr(T, "RealSIGReg", Spy)
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    head = EmbeddingHead(8, 1280, "gem")
    loader = [(torch.rand(4, 3, 16, 16),) for _ in range(2)]
    out = T.train_global_ssl(enc, head, loader, "lejepa", "sigreg", 1, 1e-4, 1e-3, ["0."], "cpu",
                             amp_dtype=torch.bfloat16)
    assert seen and all(dt == torch.float32 and not ac for dt, ac in seen)
    assert math.isfinite(out["final_loss_total"]) and out["skipped_steps"] == 0
    assert all(p.dtype == torch.float32 for p in enc.parameters())


def test_local_ae_bf16_runs_and_loss_fp32():
    from ga_search.modehb.local_ae import train_local_ae
    torch.manual_seed(0)
    prefix = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    blk = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU())
    loader = [(torch.rand(4, 3, 16, 16),) for _ in range(2)]
    r = train_local_ae(prefix, blk, loader, "pixel", 1, 1e-3, "cpu", amp_dtype=torch.bfloat16)
    assert math.isfinite(r["final_recon_loss"]) and r["skipped_steps"] == 0


# ───────────────────────────────────────────────────────────── W&B logger / monitor
def test_controller_log_pnas_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    from ga_search.modehb import wandb_logger as WL
    run = WL.ControllerRun(ref_point=[1.1, 1.1], max_fidelity=9.0, name="t", project="pnas_cs_test")
    tabs = {"kappa_vs_params": (["cand_id", "params", "kappa"], [["c1", 1e5, 0.5], ["c2", 2e5, 0.6]],
                                ("scatter", "params", "kappa", None)),
            "growth_curves": (["family", "params", "kappa"], [["F", 1e5, 0.5]], ("line", "params", "kappa", "stroke:family")),
            "mutation_survival": (["operator", "promoted_rate"], [["widen", 0.5]], ("bar", "operator", "promoted_rate", None))}
    data = run.log_pnas(0, 0.1, 3, {"pnas/hv_max_fidelity": 0.2}, tabs)
    assert "pnas_tables/kappa_vs_params" in data and "pnas_plots/growth_curves" in data
    assert data["pnas/hv_max_fidelity"] == 0.2
    run.finish()


def test_monitor_report_covers_pnas_and_phase0(tmp_path, monkeypatch):
    from ga_search.modehb import monitor_report as MR
    assert "artifacts/pnas_cs" in MR.ROOT_GLOBS and MR._kind("artifacts/pnas_cs") == "pnas_cs"
    st = tmp_path / "artifacts" / "pnas_cs" / "remote_sensing" / "state" / "seed42"
    st.mkdir(parents=True)
    jobs = [{"status": "ok", "role": r, "cand_id": f"c{i}", "fidelity": 3.0, "result": {"checkpoint": "x.pt"}}
            for i, r in enumerate(("search", "phase0_ae", "phase0_noae", "twin"))]
    (st / "search_state.json").write_text(json.dumps({"jobs": jobs, "config": {}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MR, "OUT_JSONL", tmp_path / "none.jsonl")
    todo = [t for t in MR.pending() if t[6] == "pnas_cs"]
    assert sorted(t[7]["role"] for t in todo) == ["phase0_ae", "phase0_noae", "search"]


# ───────────────────────────────────────────────────────────── external seed + GA checkpoints in the search
EXT = ROOT / "artifacts" / "external_ssl_pequeno" / "remote_sensing_split1_pct100"


@pytest.mark.skipif(not (EXT / "lejepa" / "checkpoints" / "best_kappa.ckpt").exists(), reason="external seed absent")
def test_external_seed_is_a_family_seed_and_grows(tmp_path):
    c = _cfg(["search.external_seed.enabled=true"])
    c["external_seed"] = ES.prepare({**c["external_seed"], "enabled": True}, tmp_path)
    ev = _Eval()
    s = M.MODEHB(c, ev, tmp_path, spifil_repair=SPIFIL_REPAIR)
    init = s._initial()
    assert init[0].get("external") and init[0]["pnas"]["origin"] == "seed_external"
    st = s.run()
    ext_jobs = [j for j in st["jobs"] if j["genome"].get("external")]
    assert ext_jobs and all(j["status"] == "ok" for j in ext_jobs)
    fam = init[0]["pnas"]["family"]
    kids = [ctx for g, f, ctx in ev.ctxs if (ctx.get("inherit") or {}).get("parent_cand_id") == ext_jobs[0]["cand_id"]]
    assert all(not ctx.get("external") for ctx in kids)
    assert all(not g.get("external") for g, f, ctx in ev.ctxs if g["pnas"].get("family") == fam
               and g["pnas"].get("origin") == "grow")


def test_checkpoint_seed_imports_through_grow_fp(tmp_path, monkeypatch):
    ev = _cpu_evaluator(tmp_path, monkeypatch)
    g = _genome([8, 12], ["max", "none"], pooling="gem")
    enc = _plain_encoder(PC.active(g))
    ck = tmp_path / "seed.pt"
    torch.save({"encoder": enc, "head": None, "genome": g, "effective_specs": [b.spec for b in enc]}, ck)
    inh = PC.checkpoint_inherit({"checkpoint": str(ck), "genome": g, "name": "ckpt:t"})
    from ga_search.modehb.run_search import _NullRun
    e2, specs, logs = ev._grow_fp(g, {}, 1, 0, _NullRun(), inh)
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(e2.eval()(x), enc(x))
    assert all(lg["prov"]["origin"] == "inherited" for lg in logs)
    assert ev._fp_info["n_fp_layers"] == 2 and ev._fp_info["rel_err"] < 1e-6
    assert all(bool(m.all()) for m in ev._fp_masks.values())          # everything inherited -> protected


def test_evaluator_grow_fp_compound_child(tmp_path, monkeypatch):
    ev = _cpu_evaluator(tmp_path, monkeypatch)
    monkeypatch.setattr(ES, "fit_new_filters", lambda prefix, samples, L, cin, n, k, cfg: _rand_filters(0, None, L, cin, n, k))
    rep = _rep()
    parent = _genome([8, 12, 12], ["max", "none", "none"], act="silu")
    enc = _plain_encoder(PC.active(parent))
    ck = tmp_path / "parent.pt"
    torch.save({"encoder": enc, "head": None, "genome": parent, "effective_specs": [b.spec for b in enc]}, ck)
    child = rep(PC.scale_genome(parent, 1.6, 1.3))
    lm = PC.layer_map(parent, child)
    assert lm is not None and child["depth"] > parent["depth"]
    inh = {"checkpoint": str(ck), "layer_map": lm, "parent_layers": ES.active_layers(parent),
           "parent_cand_id": "c1", "parent_fidelity": 2.0}
    from ga_search.modehb.run_search import _NullRun
    e2, specs, logs = ev._grow_fp(child, {}, 1, 0, _NullRun(), inh)
    assert len(e2) == child["depth"] and ev._fp_info["n_fp_layers"] == child["depth"]
    assert ev._fp_info["rel_err"] is not None and ev._fp_info["rel_err"] < 1e-5
    assert "fp_widen" in ev._fp_info["origins"] and "fp_residual_zero" in ev._fp_info["origins"]
    inherited = sum(int(m.sum()) for m in ev._fp_masks.values())
    assert 0 < inherited < sum(p.numel() for p in e2.parameters())   # new filters / layers are free


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, n, off=0):
        g = torch.Generator().manual_seed(off)
        self.x = torch.rand(n, 3, 32, 32, generator=g)
        self.samples = [type("S", (), {"name": f"s{off + i}", "label": i % 3})() for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.x[i], self.samples[i].label, self.samples[i].name


def _cpu_evaluator(tmp_path, monkeypatch):
    import ga_search.progressive_data as PD
    from torch.utils.data import DataLoader
    monkeypatch.setattr(PD, "load_data", lambda cfg, out: (DataLoader(_ImgDS(24), batch_size=8),
                                                            DataLoader(_ImgDS(12, 100), batch_size=8),
                                                            _ImgDS(24).samples[:3], {"sha256": "x"}))
    monkeypatch.setattr(RS, "build_spifil_pool", lambda sel, train, k, seed: {s.label: [s] for s in sel})
    cfg = RS.load_config(CFG_DIR / "pnas_cs.yaml", "remote_sensing", None,
                         ["device=cpu", "data.batch_size=8", "data.val_max_samples=null", "wandb.enabled=false"])
    ev = RS.Evaluator.__new__(RS.Evaluator)
    ev.cfg, ev.out, ev.device, ev.seed = cfg, tmp_path, torch.device("cpu"), 0
    ev._load_data()
    ev._spifil_fit = {"seconds": 0.0, "peak_gb": 0.0}
    return ev


def test_checkpoint_seed_memory_guard(tmp_path):
    g = _genome([8, 12], ["max", "none"])
    for name, peak, ssl in (("ok", 9.0, "lejepa"), ("big", 30.0, "simclr")):
        d = tmp_path / name / "f5"
        d.mkdir(parents=True)
        torch.save({"encoder": _plain_encoder(PC.active(g)), "genome": {**g, "ssl": ssl}}, d / "encoder.pt")
        (d / "result.json").write_text(json.dumps({"peak_gpu_mem_gb": peak}))
    c = _cfg(["search.stratified.memory_budget_gb=40"])
    c["pnas_cs"]["seeds"] = {"checkpoints": {"remote_sensing": [str(tmp_path / n / "f5" / "encoder.pt") for n in ("ok", "big")]}}
    s = M.MODEHB(c, _Eval(), tmp_path / "out", spifil_repair=SPIFIL_REPAIR)
    names = [g["pnas"].get("name") for g in s._initial() if g["pnas"]["origin"] == "seed_checkpoint"]
    assert names == ["ckpt:ok/f5"]                     # 30 GB x 4/2 views = 60 GB > 40 -> skipped


def test_cov_readout_eigh_fallback(monkeypatch):
    """CovReadout eigh: identical when LAPACK converges; on non-convergence (dead-ReLU, repeated zero
    eigenvalues) the jittered retry is used instead of failing the whole job."""
    from ga_search.modehb import pooling_head as PH
    torch.manual_seed(0)
    x = torch.randn(64, 6, dtype=torch.float64)
    x[:, 3:] = 0.0                                        # dead channels: repeated zero eigenvalues
    cov = torch.cov(x.T)
    w0, v0 = torch.linalg.eigh(cov)
    w1, v1 = PH._eigh(cov)
    assert torch.equal(w0, w1) and torch.equal(v0, v1)
    real, calls = torch.linalg.eigh, []

    def flaky(a, *k, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("linalg.eigh: The algorithm failed to converge")
        return real(a, *k, **kw)
    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    w2, v2 = PH._eigh(cov)
    assert len(calls) == 2 and torch.allclose(w2, w0, atol=1e-8)
    rec = v2 @ torch.diag(w2) @ v2.T
    assert torch.allclose(rec, cov, atol=1e-8)
