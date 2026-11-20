"""Parameter-stratified MO-DEHB (ga_search/modehb/strata.py + MODEHB stratified mode)."""
import collections
import functools
import math

import numpy as np
import pytest

from ga_search.modehb import modehb as M
from ga_search.modehb import strata as S
from ga_search.modehb.init import repair_for_spifil

SC = S.strat_config({"enabled": True})
SPACE = {"operator": ["conv", "dwsep", "inception"], "softhebb": [True, False], "kernel": [3, 5, 7],
         "activation": ["relu", "gelu", "silu", "selu"],
         "norm": ["none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"], "pool": ["none", "max", "avg"],
         "residual": [True, False], "pooling": ["gap", "gem", "moments"]}


def _cfg(**kw):
    c = dict(fidelities=[1, 3, 9], eta=3, max_brackets=4, seed=42, image_size=224,
             channel_space={"type": "int_range", "min": 1, "max": 1024, "scale": "log", "monotonic": True},
             spifil_images_choices=[1, 2, 4, 8, 16, 32], initial_pyramid_growth=2.0, random_baseline=False,
             initial_size=16, initial_operators="sampled", stage1_brackets=0, search_space=SPACE,
             stratified={"enabled": True, "calibration": True, "extension_fidelities": [27],
                         "protected_brackets": 3})
    c.update(kw)
    return c


def slow_big_eval(g, fid, ctx):
    """Big nets start worse and keep improving (the case the stratification protects)."""
    p = S.param_estimate(g)
    tau = 1 + 6 * math.log10(max(p, 1e4) / 1e4)
    q = min(0.95, 0.4 + 0.1 * math.log10(max(p, 1e4) / 1e4)) * (1 - math.exp(-fid / tau))
    return {"status": "ok", "quality": q, "flops": p * 3000, "params": int(p), "gpu_hours": 0.0,
            "checkpoint": "ck"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("strat")
    s = M.MODEHB(_cfg(), slow_big_eval, out, spifil_repair=functools.partial(repair_for_spifil, spifil_ops=True))
    return s, s.run(), out


def test_classification_bounds():
    assert S.stratum_of(1, SC) == "tiny" and S.stratum_of(300_000, SC) == "tiny"
    assert S.stratum_of(300_001, SC) == "small" and S.stratum_of(800_000, SC) == "small"
    assert S.stratum_of(800_001, SC) == "medium" and S.stratum_of(1_500_000, SC) == "medium"
    assert S.stratum_of(1_500_001, SC) == "huge" and S.stratum_of(3_500_000, SC) == "huge"
    assert S.stratum_of(3_500_001, SC) is None


def test_strata_must_be_contiguous():
    with pytest.raises(ValueError):
        S.strat_config({"strata": {"a": [0, 10], "b": [11, 20]}})


def test_fit_to_stratum_hits_each_category():
    rep = functools.partial(M.repair, image_size=224, channel_space=_cfg()["channel_space"])
    base = M.initial_population(42, size=4, twins=False, pyramid_growth=2.0, image_size=224,
                                channel_space=_cfg()["channel_space"])[0]
    base = {**base, "depth": 3}
    for j, s in enumerate(S.names(SC)):
        g = S.fit_to_stratum(base, s, SC, rep, np.random.default_rng(j))
        assert S.stratum_of(S.param_estimate(g), SC) == s
        w = [L["channels"] for L in g["layers"][:g["depth"]]]
        assert w == sorted(w)                      # monotonic widths preserved


def test_cap_never_exceeded(run):
    s, st, _ = run
    assert all(j["params_est"] <= SC["param_cap"] for j in st["jobs"])


def test_all_categories_explored_and_reach_max_fidelity(run):
    s, st, _ = run
    top = s.fidelities[-1]
    for f in s.fidelities:
        got = {j["stratum"] for j in st["jobs"] if j["fidelity"] == f and j["status"] == "ok"}
        assert got == set(S.names(SC)), (f, got)
    assert {j["stratum"] for j in st["jobs"] if j["fidelity"] == top} == set(S.names(SC))


def test_mutation_within_and_across_categories(run):
    _, st, _ = run
    de = [j["op"] for j in st["jobs"] if (j.get("op") or {}).get("type") == "de"]
    assert de and {o["pool"] for o in de} == {"same", "cross"}
    assert any(o["parent_stratum"] != o["child_stratum"] for o in de)   # offspring reclassified
    assert any(o["parent_stratum"] == o["child_stratum"] for o in de)


def test_protected_promotion_quota_and_exploration():
    cands = ([{"F": [0.1, 0.1], "stratum": "tiny", "slope": 0.0}] * 8 +
             [{"F": [0.6, 0.5], "stratum": "small", "slope": 0.0}] * 2 +
             [{"F": [0.7, 0.6], "stratum": "medium", "slope": 0.05}] * 3 +
             [{"F": [0.8, 0.9], "stratum": "huge", "slope": 0.2}] * 3)
    sel = lambda F, k: M.select_promotions(F, k)
    idx = S.stratified_promotions(cands, 8, SC, sel)
    base = collections.Counter(cands[i]["stratum"] for i in idx[:8])
    assert all(base[s] >= 2 for s in S.names(SC))   # >= 25% of 8 slots each
    assert len(idx) == 9 and cands[idx[-1]]["stratum"] == "huge"   # exploratory: best slope
    glob = collections.Counter(cands[i]["stratum"] for i in M.select_promotions(np.array([c["F"] for c in cands]), 8))
    assert glob["tiny"] == 8                       # what the unstratified MO-DEHB would do


def test_promotion_counts_rotate_when_few_slots():
    cands = [{"F": [0.1, 0.1], "stratum": s, "slope": None} for s in S.names(SC)]
    sel = lambda F, k: M.select_promotions(F, k)
    first = S.stratified_promotions(cands, 1, SC, sel, counts={"tiny": 3, "small": 1, "medium": 0, "huge": 2})
    assert cands[first[0]]["stratum"] == "medium"


def test_insert_keeps_categories(run):
    s, st, _ = run
    for f, sp in st["subpops"].items():
        cnt = collections.Counter(m["stratum"] for m in sp)
        assert max(cnt.values()) <= len(sp) - (len(cnt) - 1)   # no category takes the whole subpop
        assert len(cnt) >= min(4, len(sp))


def test_calibration_and_extension(run):
    s, st, _ = run
    assert set(st["calibration"]) == set(S.names(SC))
    for c in st["calibration"].values():
        assert {"1", "3", "9"} <= set(c["quality_by_epochs"])
    ext = [j for j in st["jobs"] if j["fidelity"] == 27.0]
    assert ext and all(j["op"]["type"] == "extension" for j in ext)
    assert all(j["resumed_from"] is not None for j in ext)          # checkpoints reused


def test_extension_excluded_from_hv_and_front(run):
    s, st, _ = run
    top = s.fidelities[-1]
    assert all(j["fidelity"] == top for j in s.front(top))
    rep = s.strata_report(0)
    hv = M.hypervolume([j["F"] for j in st["jobs"] if j["fidelity"] == top and j["status"] == "ok"], s.ref)
    assert rep["scalars"]["hv_strata/global"] == pytest.approx(hv)
    assert max(rep["scalars"][f"hv_strata/{n}"] for n in S.names(SC)) <= hv + 1e-12
    for name, (cols, rows, plot) in rep["tables"].items():
        assert all(len(r) == len(cols) for r in rows), name


def test_resume_replays_identically(run):
    s, st, out = run
    s2 = M.MODEHB(_cfg(), slow_big_eval, out, spifil_repair=functools.partial(repair_for_spifil, spifil_ops=True))
    st2 = s2.run(resume=True)
    assert [j["arch_hash"] for j in st2["jobs"]] == [j["arch_hash"] for j in st["jobs"]]
    assert [j["stratum"] for j in st2["jobs"]] == [j["stratum"] for j in st["jobs"]]


def test_measured_above_cap_is_infeasible(tmp_path):
    def ev(g, fid, ctx):
        return {**slow_big_eval(g, fid, ctx), "params": 4_000_000}
    s = M.MODEHB(_cfg(max_brackets=1, stratified={"enabled": True, "calibration": False}), ev, tmp_path)
    st = s.run()
    assert all(j["status"] == "failed" and j["result"]["infeasible"] == "param_cap" for j in st["jobs"])


def test_disabled_is_original_behaviour(tmp_path):
    cfg = _cfg(max_brackets=1)
    cfg.pop("stratified")
    s = M.MODEHB(cfg, slow_big_eval, tmp_path)
    st = s.run()
    assert s.strat is None and all("stratum" not in j for j in st["jobs"])
