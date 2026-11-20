"""GA v4 (search_v4.yaml): bands B0-B4, directed quotas, per-band promotion, OOM ladder, mem scale, so2 readout."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ga_search.modehb import modehb as M
from ga_search.modehb import strata as S

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "ga_search/config/modehb/search_v4.yaml"


def sc():
    return S.strat_config(yaml.safe_load(V4.read_text())["stratified"])


def test_v4_bands_and_cap():
    c = sc()
    assert [n for n, _, _ in c["strata"]] == ["B0", "B1", "B2", "B3", "B4"]
    assert c["param_cap"] == 5_500_000 and c["strata"][-1][2] == 5_500_000
    w = S.band_weights(c)
    assert w["B4"] >= w["B3"] >= w["B2"] >= w["B1"] >= w["B0"]
    raw = yaml.safe_load(V4.read_text())
    assert raw["svm_fitness"]["C"] == 100 and raw["svm_fitness"]["scaler"] == "none"
    assert raw["svm_fitness"]["readout"] == "so2_4096"


def test_strata_sorted_from_json_order():
    c = S.strat_config({"strata": {"huge": [1500000, 3500000], "medium": [800000, 1500000],
                                   "small": [300000, 800000], "tiny": [0, 300000]}})
    assert S.names(c) == ["tiny", "small", "medium", "huge"]


def test_next_band_targets_underrepresented_large_first():
    c = sc()
    assert S.next_band({}, c) == "B4"                      # tie -> largest band
    counts = {"B0": 2, "B1": 2, "B2": 2, "B3": 3, "B4": 1}
    assert S.next_band(counts, c) == "B4"
    # a failed big model is not counted -> the band stays the target
    assert S.next_band({"B0": 5, "B1": 5, "B2": 5, "B3": 5, "B4": 0}, c) == "B4"


def _sel(F, k):
    return [int(i) for i in M.select_promotions(F, k)]


def test_band_promotions_inside_band_and_min_per_band():
    c = {**sc(), "promote_min_per_band": 1, "explore_extra": 0}
    rng = np.random.default_rng(0)
    cands = []
    for b, q0, fl in [("B0", .60, .1), ("B1", .62, .3), ("B2", .64, .5), ("B3", .66, .7), ("B4", .68, .9)]:
        for _ in range(4):
            cands.append({"F": [1 - (q0 + rng.normal(0, .01)), fl + rng.uniform(0, .05)], "stratum": b})
    # global Pareto would pick only small/cheap ones; per band every band gets >= 1 even with n=1
    got = S.band_promotions(cands, 1, c, _sel)
    assert {cands[i]["stratum"] for i in got} == {"B0", "B1", "B2", "B3", "B4"}
    got = S.band_promotions(cands, 12, c, _sel)
    per = {b: sum(cands[i]["stratum"] == b for i in got) for b in S.names(c)}
    assert per["B4"] >= per["B3"] >= per["B0"] and per["B4"] >= per["B1"]
    assert len(set(got)) == len(got)


def test_mem_scale_admits_bigger_models():
    g = M.initial_population(0, (64,), (4,), image_size=224, min_spatial=6)[0]
    e1 = S.mem_estimate_gb(g, 224)
    assert S.mem_estimate_gb(g, 224, scale=0.8) == pytest.approx(0.8 * e1)
    rep = lambda x: M.repair(x, image_size=224, min_spatial=6)
    big = rep({**g, "layers": [{**L, "channels": 512} for L in g["layers"]], "depth": 4})
    budget = 0.9 * S.mem_estimate_gb(big, 224)
    a = S.mem_repair(big, budget, rep, scale=1.0)
    b = S.mem_repair(big, budget, rep, scale=0.8)
    assert S.mem_estimate_gb(a, 224) <= budget + 1e-6
    assert S.mem_estimate_gb(b, 224, scale=0.8) <= budget + 1e-6
    assert b == big and a != big          # calibrated: the big model fits untouched


class _OOM(RuntimeError):
    pass


_OOM.__name__ = "OutOfMemoryError"


def _bare(ladder):
    m = M.MODEHB.__new__(M.MODEHB)
    m.strat = {"oom_fallback": ladder, "oom_fast_retries": 1, "oom_retry_sleep_s": 0}
    m.state = {"jobs": []}
    m._replay = []
    return m


def test_oom_ladder_bf16_then_ok():
    calls = []

    def ev(g, f, ctx):
        calls.append(ctx.get("mem_fallback"))
        if (ctx.get("mem_fallback") or {}).get("batch_div", 1) < 2:
            raise _OOM("CUDA out of memory")
        return {"quality": .7}
    m = _bare([{"amp_dtype": "bf16"}, {"amp_dtype": "bf16", "batch_div": 2}])
    m.evaluate_fn = ev
    r = m._call({}, 5.0, {"cand_id": "c1", "role": "search"})
    assert r["status"] == "ok" and r["mem_fallback"] == {"amp_dtype": "bf16", "batch_div": 2}
    assert len(r["oom_attempts"]) == 3 and calls[:2] == [None, None]


def test_oom_ladder_exhausted_is_not_penalized():
    def ev(g, f, ctx):
        raise _OOM("CUDA out of memory")
    m = _bare([{"amp_dtype": "bf16"}])
    m.evaluate_fn = ev
    r = m._call({}, 5.0, {"cand_id": "c1", "role": "search"})
    assert r["status"] == "failed" and r["infeasible"] == "oom" and r["penalize"] is False


def test_failed_never_enters_subpop_v4():
    m = M.MODEHB.__new__(M.MODEHB)
    m.strat = {"skip_failed_in_subpop": True}
    m.state = {"subpops": {"5.0": []}}
    m._max_pop = {5.0: 10}
    m.fidelities = [5.0]
    m._insert(5.0, {"role": "search", "status": "failed", "cand_id": "c9", "vector": [], "F": [2, 2],
                    "genome": {}, "job": 0, "stratum": "B4"}, None)
    assert m.state["subpops"]["5.0"] == []


def test_split_loader():
    from ga_search.modehb.run_search import _SplitLoader
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(16.).view(16, 1)), batch_size=8)
    out = list(_SplitLoader(dl, 2))
    assert len(out) == 4 and all(b[0].shape[0] == 4 for b in out)
    assert torch.cat([b[0] for b in out]).flatten().tolist() == list(range(16))


def test_so2_readout_matches_reported_reference():
    ref_p = ROOT / "research/spifil_b0b4/readout_x/eval_backbone_head.py"
    if not ref_p.exists():
        pytest.skip("readout_x not present")
    spec = importlib.util.spec_from_file_location("ebh", ref_p)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    from ga_search.modehb import so2_readout as R
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(3, 96, 3, padding=1), torch.nn.ReLU()),
                              torch.nn.Sequential(torch.nn.Conv2d(96, 120, 3, stride=2, padding=1), torch.nn.ReLU()))
    xtr, xva = torch.randn(12, 3, 16, 16), torch.randn(5, 3, 16, 16)
    r = R.So2Readout(D=4096, n_px=16).fit(enc, xtr, "cpu", bs=4)
    mom = ref.PixelMoments(16, 0)
    with torch.no_grad():
        for i0 in range(0, 12, 4):
            mom.add(enc(xtr[i0:i0 + 4]), list(range(i0, i0 + 4)))
        mu, V = mom.pca(88)
        z_ref = ref.so2_vector(enc(xva), mu, V, 4096)
    z = r.embed(enc, xva, "cpu", bs=2)
    assert z.shape == (5, 4096) and torch.allclose(z, z_ref, atol=1e-6)


C00015 = [dict(conv="inception", kernel=7, channels=15, activation="relu", norm="batch_zca", pool="avg", residual=False),
          dict(conv="conv", kernel=5, channels=62, activation="gelu", norm="batch", pool="max", residual=True),
          dict(conv="conv", kernel=5, channels=192, activation="silu", norm="dbn", pool="max", residual=False),
          dict(conv="dwsep", kernel=7, channels=586, activation="silu", norm="dbn", pool="avg", residual=True),
          dict(conv="conv", kernel=3, channels=733, activation="silu", norm="dbn", pool="none", residual=True)]


def test_cap_checked_before_training(tmp_path, monkeypatch):
    """remote_sensing v4 c00015: ConvBlock estimate 5,251,641 (under 5.5M), measured 5,612,872 (bias +
    1x1 residual projections). cap_count: spifil_ops counts the built blocks (>= measured) and no genome
    above the cap is trained: reprojected, or rejected untrained; a job logged raw replays raw."""
    g15 = {"depth": 5, "embed_dim": 1280, "pooling": "gem", "layers": C00015}
    assert S.param_estimate(g15) == 5_251_641 and S.param_count_built(g15) == 5_797_782 > sc()["param_cap"]
    assert sc()["cap_count"] == "spifil_ops"
    trained = []

    def ev(g, fid, ctx):
        trained.append(S.param_count_built(g))
        return {"status": "ok", "quality": .5, "flops": 1e9, "params": trained[-1], "gpu_hours": 0.0}
    s = M.MODEHB({"fidelities": [1, 3], "seed": 42, "image_size": 224, "random_baseline": False,
                  "stratified": {"enabled": True, "calibration": False, "cap_count": "spifil_ops"}}, ev, tmp_path)
    g = M.initial_population(0, (64,), (4,), image_size=224, min_spatial=6)[0]
    raw = M.repair({**g, "layers": [{**L, "residual": True} for L in g["layers"]]}, **s.repair_kw)
    est, built = S.param_estimate(raw), S.param_count_built(raw)
    cap = (est + built) // 2                      # the c00015 case: estimate under the cap, built count above
    s.strat = S.strat_config({**s.strat, "strata": {"lo": [0, cap // 2], "hi": [cap // 2, cap]}, "param_cap": cap})
    s.state, s._replay = s._fresh_state(), []
    rec = s._evaluate(raw, 1.0, parent_id=None, generation=0, rung=0)
    assert rec["status"] == "ok" and trained and max(trained) <= cap and rec["arch_hash"] != M.arch_hash(raw)
    monkeypatch.setattr(S, "enforce_cap", lambda x, *a, **k: x)          # cannot shrink -> rejected untrained
    rec = s._evaluate(raw, 3.0, parent_id=None, generation=0, rung=1)
    assert len(trained) == 1 and rec["result"]["not_trained"] and rec["result"]["infeasible"] == "param_cap"
    assert rec["result"]["params"] == built and rec["stratum"] is None
    monkeypatch.undo()
    old = {"status": "failed", "infeasible": "param_cap", "params": built + 1}
    s.state, s._replay = s._fresh_state(), [{"arch_hash": M.arch_hash(raw), "fidelity": 1.0, "role": "search",
                                             "result": old}]
    rec = s._evaluate(raw, 1.0, parent_id=None, generation=0, rung=0)
    assert rec["arch_hash"] == M.arch_hash(raw) and rec["result"] == old and len(trained) == 1


def test_pnas_v4_cap_from_config():
    from ga_search.modehb import pnas_cs as PC
    p4 = ROOT / "ga_search/config/modehb/pnas_cs_v4.yaml"
    if not p4.exists():
        pytest.skip("pnas_cs_v4.yaml not present")
    v4, old = yaml.safe_load(p4.read_text()), yaml.safe_load((ROOT / "ga_search/config/modehb/pnas_cs.yaml").read_text())
    pc4 = PC.pnas_config(v4["pnas_cs"])
    assert PC.cap(pc4) == 5_500_000 and PC.level_strata(pc4)["P4"][1] == 5_500_000
    assert PC.level_strata(pc4)["P3"] == PC.level_strata(PC.pnas_config(old["pnas_cs"]))["P3"]   # levels unchanged
    assert PC.cap(PC.pnas_config(old["pnas_cs"])) == 5_775_000           # old PNAS (replay): 5.5M x 1.05
    assert v4["stratified"]["cap_count"] == "spifil_ops"
