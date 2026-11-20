"""LEMONADE warm start + network morphisms + Lamarckian inheritance (ga_search/modehb/lemonade.py)."""
import functools
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from ga_search.modehb import lemonade as LM
from ga_search.modehb import modehb as M
from ga_search.modehb import run_search as RS
from ga_search.modehb import strata as S
from ga_search.modehb.init import repair_for_spifil

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "ga_search" / "config" / "modehb"
SPACE = {"operator": ["conv", "dwsep", "inception"], "softhebb": [True, False], "kernel": [3, 5, 7],
         "activation": ["relu", "gelu", "silu", "selu"],
         "norm": ["none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"], "pool": ["none", "max", "avg"],
         "residual": [True, False], "pooling": ["gap", "gem", "moments"]}
CS = {"type": "int_range", "min": 1, "max": 1024, "scale": "log", "monotonic": True}
SEED_B = [40, 80, 152, 152, 152, 312]
LC = LM.lemonade_config({"enabled": True})
SPIFIL_REPAIR = functools.partial(repair_for_spifil, spifil_ops=True)


def _rep():
    return functools.partial(M.repair, image_size=224, channel_space=CS, search_space=SPACE,
                             spifil_repair=SPIFIL_REPAIR)


def _cfg(lemonade=True, **kw):
    c = dict(fidelities=[1, 3, 9], eta=3, max_brackets=3, seed=42, image_size=224, channel_space=CS,
             spifil_images_choices=[1, 2, 4, 8, 16, 32], initial_pyramid_growth=2.0, random_baseline=False,
             initial_size=8, initial_operators="sampled", stage1_brackets=0, search_space=SPACE, n_classes=14,
             spifil_pool_sizes=[32] * 14,
             stratified={"enabled": True, "calibration": False, "memory_budget_gb": 20, "protected_brackets": 2})
    if lemonade:
        c["lemonade"] = {"enabled": True, "n_descendants": 4, "morph_prob": 0.6}
    c.update(kw)
    return c


class _Eval:
    """Synthetic evaluator that records every ctx and fakes a checkpoint per job."""

    def __init__(self):
        self.ctxs = []

    def __call__(self, g, fid, ctx):
        self.ctxs.append((g, fid, dict(ctx)))
        p = S.param_estimate(g)
        q = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3))) * (1 - math.exp(-fid / 3))
        return {"status": "ok", "quality": q, "flops": p * 3000, "params": int(p), "gpu_hours": 0.0,
                "checkpoint": f"ck/{ctx['cand_id']}_f{int(fid)}.pt",
                "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]]}


# ---------------------------------------------------------------- config
def test_absent_or_disabled_is_none():
    assert LM.lemonade_config(None) is None and LM.lemonade_config({"enabled": False}) is None
    with pytest.raises(ValueError):
        LM.lemonade_config({"enabled": True, "operators": ["prune"]})


def test_disabled_is_original_behaviour(tmp_path):
    ev0, ev1 = _Eval(), _Eval()
    a = M.MODEHB(_cfg(lemonade=False), ev0, tmp_path / "a", spifil_repair=SPIFIL_REPAIR)
    b = M.MODEHB({**_cfg(lemonade=False), "lemonade": {"enabled": False}}, ev1, tmp_path / "b",
                 spifil_repair=SPIFIL_REPAIR)
    sa, sb = a.run(), b.run()
    assert a.lemonade is None and b.lemonade is None
    assert [j["arch_hash"] for j in sa["jobs"]] == [j["arch_hash"] for j in sb["jobs"]]
    assert all("inherit" not in ctx for _, _, ctx in ev0.ctxs)
    assert all("inherit" not in j and (j.get("op") or {}).get("type") != "morph" for j in sa["jobs"])


def test_lemonade_yaml_equals_search_v3_except_lemonade():
    v3 = yaml.safe_load((CFG_DIR / "search_v3.yaml").read_text())
    lm = yaml.safe_load((CFG_DIR / "search_v3_lemonade.yaml").read_text())
    assert lm.pop("lemonade")["enabled"] is True
    assert lm.pop("max_brackets") == 4 and v3.pop("max_brackets")
    lm.pop("paths"), v3.pop("paths")
    assert lm == v3                                        # same fitness / readout / space as run A
    c = RS.load_config(CFG_DIR / "search_v3_lemonade.yaml", "remote_sensing", 42, [])
    assert c["search"]["lemonade"]["enabled"] and "modehb_lemonade" in c["output_dir"]
    assert c["eval"]["svm_fitness"]["readout"] == "cov_head" and c["eval"]["svm_fitness"]["wise_alphas"]
    assert "lemonade" not in RS.load_config(CFG_DIR / "search_v3.yaml", "remote_sensing", 42, [])["search"]


# ---------------------------------------------------------------- warm start
def test_seed_survives_memory_repair(tmp_path):
    c = RS.load_config(CFG_DIR / "search_v3_lemonade.yaml", "remote_sensing", 42,
                       ["search.stratified.memory_budget_gb=20"])
    sc = {**c["search"], "spifil_pool_sizes": [32] * 14}
    m = M.MODEHB(sc, _Eval(), tmp_path)
    pop = m._initial()
    seed = pop[0]
    assert LM.widths(seed) == SEED_B and seed["depth"] == 6
    L = seed["layers"][:6]
    assert all(x["kernel"] == 3 and x["conv"] == "conv" and x["activation"] == "relu" for x in L)
    assert [x["pool"] for x in L] == ["max"] * 4 + ["none"] * 2      # 224 -> 14 px >= 7
    assert m._rep(seed) == seed                                        # fixed point of the memory repair
    assert S.mem_estimate_gb(seed, 224, n_classes=14, pool_sizes=[32] * 14) <= 20
    assert m.param_fn(seed) <= 3_500_000
    kids = [g for g in pop if M.arch_hash(g) in m._lineage]
    assert len(kids) == int(m.lemonade["n_descendants"])
    assert pop[-len(kids):] == kids                                    # descendants after the seeds
    for k in kids:
        parent, info = m._lineage[M.arch_hash(k)]
        assert info["morphism"] in LM.MORPHISMS and M.arch_hash(k) != M.arch_hash(parent)


# ---------------------------------------------------------------- morphisms
def _seed():
    return LM.seed_genome(LC, 42, _rep())


def test_each_morphism():
    g, rng = _seed(), np.random.default_rng(0)
    d, info = LM.morph(g, "deepen", rng, LC)
    p = info["layer"]
    assert d["depth"] == 7 and d["layers"][p]["channels"] == g["layers"][p - 1]["channels"]
    assert d["layers"][p]["pool"] == "none" and LM.reusable_prefix(g, _rep()(d)) == p
    w, info = LM.morph(g, "widen", rng, LC)
    assert w["layers"][info["layer"]]["channels"] > g["layers"][info["layer"]]["channels"]
    w = _rep()(w)
    assert LM.widths(w) == sorted(LM.widths(w))                    # monotonic after repair
    r, info = LM.morph(g, "residual", rng, LC)
    assert info["layer"] >= 1 and r["layers"][info["layer"]]["residual"]
    k, info = LM.morph(g, "kernel", rng, LC)
    assert k["layers"][info["layer"]]["kernel"] != g["layers"][info["layer"]]["kernel"]
    assert LM.reusable_prefix(g, _rep()(k)) == info["layer"]


def test_deepen_at_max_depth_not_applicable():
    g = {**_seed(), "depth": 8}
    assert LM.morph(g, "deepen", np.random.default_rng(0), LC) is None


def test_random_morph_changes_architecture_and_prefix():
    g, rng = _seed(), np.random.default_rng(1)
    for _ in range(20):
        c, info = LM.random_morph(g, rng, LC, _rep())
        assert c is not None and M.arch_hash(c) != M.arch_hash(g)
        assert 0 <= info["n_reuse"] < c["depth"]
        assert all(LM._layer(a) == LM._layer(b) for a, b in zip(g["layers"][:info["n_reuse"]],
                                                                 c["layers"][:info["n_reuse"]]))


def test_reusable_prefix_capped_below_child_depth():
    g = _seed()
    assert LM.reusable_prefix(g, g) == g["depth"] - 1


def test_pick_parent_checkpoint():
    jobs = [{"status": "ok", "fidelity": f, "cand_id": "c1", "job": i, "result": {"checkpoint": f"p{f}"}}
            for i, f in enumerate([5.0, 15.0, 45.0])] + [{"status": "failed", "fidelity": 1.0, "result": {}}]
    assert LM.pick_parent_checkpoint(jobs, 20)["fidelity"] == 15.0
    assert LM.pick_parent_checkpoint(jobs, 1)["fidelity"] == 5.0            # none <= fid: lowest
    assert LM.pick_parent_checkpoint(jobs, 1, "max")["fidelity"] == 45.0
    assert LM.pick_parent_checkpoint([], 5) is None
    spec = LM.inherit_spec(jobs[1], 3)
    assert spec["checkpoint"] == "p15.0" and spec["n_blocks"] == 3 and spec["parent_cand_id"] == "c1"
    assert LM.inherit_spec(jobs[1], 0) is None


# ---------------------------------------------------------------- search integration
@pytest.fixture(scope="module")
def lem_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("lem")
    ev = _Eval()
    s = M.MODEHB(_cfg(), ev, out, spifil_repair=SPIFIL_REPAIR)
    return s, s.run(), ev, out


def test_search_uses_morphisms_and_inheritance(lem_run):
    s, st, ev, _ = lem_run
    assert st["jobs"][0]["genome"]["depth"] == 6 and LM.widths(st["jobs"][0]["genome"]) == SEED_B
    morphs = [j for j in st["jobs"] if (j.get("op") or {}).get("type") == "morph"]
    assert morphs, "no morphism offspring"
    inh = [ctx["inherit"] for _, _, ctx in ev.ctxs if "inherit" in ctx]
    assert inh, "no evaluation inherited weights"
    by_ck = {j["result"]["checkpoint"]: j for j in st["jobs"] if j["status"] == "ok"}
    for (g, fid, ctx) in ev.ctxs:
        if "inherit" not in ctx:
            continue
        i = ctx["inherit"]
        parent = by_ck[i["checkpoint"]]
        assert parent["cand_id"] == i["parent_cand_id"] and parent["fidelity"] <= fid or \
            parent["fidelity"] == min(j["fidelity"] for j in st["jobs"] if j["cand_id"] == parent["cand_id"])
        assert 1 <= i["n_blocks"] == LM.reusable_prefix(parent["genome"], g) < g["depth"]
        assert ctx["resume"] is None                    # promotions resume their own checkpoint
    # initial descendants of the seed inherit the seed's trained blocks
    seed_id = st["jobs"][0]["cand_id"]
    assert any(c["inherit"]["parent_cand_id"] == seed_id for _, _, c in ev.ctxs if "inherit" in c)
    assert all(j["params_est"] <= 3_500_000 for j in st["jobs"])


def test_lemonade_resume_replays_identically(lem_run):
    s, st, _, out = lem_run
    s2 = M.MODEHB(_cfg(), _Eval(), out, spifil_repair=SPIFIL_REPAIR)
    st2 = s2.run(resume=True)
    assert [j["arch_hash"] for j in st2["jobs"]] == [j["arch_hash"] for j in st["jobs"]]
    assert [j.get("inherit") for j in st2["jobs"]] == [j.get("inherit") for j in st["jobs"]]


# ---------------------------------------------------------------- evaluator hook
class _DS(torch.utils.data.Dataset):
    def __init__(self, n, n_cls=3, off=0):
        self.samples = [types.SimpleNamespace(name=f"s{off + i}", label=i % n_cls) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.rand(3, 8, 8), self.samples[i].label, self.samples[i].name


def _evaluator(tmp_path, monkeypatch):
    import ga_search.progressive_data as PD

    def fake(cfg, out):
        tr, va = _DS(12), _DS(6, off=100)
        return (torch.utils.data.DataLoader(tr, batch_size=4), torch.utils.data.DataLoader(va, batch_size=4),
                tr.samples[:3], {"sha256": "x"})
    monkeypatch.setattr(PD, "load_data", fake)
    monkeypatch.setattr(RS, "build_spifil_pool", lambda sel, train, k, seed: {s.label: [s] for s in sel})
    cfg = RS.load_config(CFG_DIR / "search_v3_lemonade.yaml", "remote_sensing", None,
                         ["device=cpu", "data.ram_cache=true", "data.batch_size=4", "wandb.enabled=false"])
    ev = RS.Evaluator.__new__(RS.Evaluator)
    ev.cfg, ev.out, ev.device, ev.seed = cfg, tmp_path, torch.device("cpu"), 0
    ev._load_data()
    return ev


def _blk(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU())


def test_grow_inherits_prefix_and_fits_only_new_layers(tmp_path, monkeypatch):
    from ga_search.modehb import init as I
    from ga_search.modehb import local_ae
    ev = _evaluator(tmp_path, monkeypatch)
    torch.manual_seed(0)
    parent = nn.Sequential(_blk(3, 4), _blk(4, 6), _blk(6, 8))
    ck = tmp_path / "parent.pt"
    torch.save({"encoder": parent, "effective_specs": [{"channels": 4}, {"channels": 6}, {"channels": 8}]}, ck)
    fitted, ae_calls = [], []

    def fake_init(genome, i, prefix, donors, cfg, prefix_id):
        fitted.append((i, prefix_id))
        cin = ev._out_channels(prefix) if len(prefix) else 3
        return _blk(cin, int(genome["layers"][i]["channels"])), {"origin": "spifil",
                                                                  "effective_spec": {"channels": 9}}
    monkeypatch.setattr(I, "init_block", fake_init)
    monkeypatch.setattr(local_ae, "train_local_ae", lambda prefix, block, *a, **k: ae_calls.append(block) or {})
    g = M.repair({"depth": 4, "init": "spifil", "seed": 0, "ssl": "lejepa", "reg": "sigreg",
                  "layers": [{"channels": c, "norm": "none", "conv": "conv", "kernel": 3, "softhebb": False}
                             for c in (4, 6, 9, 9)]}, image_size=8, min_spatial=1, channel_space=CS)
    inherit = {"checkpoint": str(ck), "n_blocks": 2, "parent_cand_id": "c7", "parent_fidelity": 5.0}
    enc, specs, logs = ev._grow(g, {}, 1, 0, RS._NullRun(), inherit=inherit)
    assert [i for i, _ in fitted] == [2, 3] and len(ae_calls) == 2       # only the new layers
    assert all("inherit:c7@5.0" in pid for _, pid in fitted)
    for i in range(2):                                                    # trained weights reused
        assert torch.equal(enc[i][0].weight, parent[i][0].weight)
        assert enc[i] is not parent[i]                                    # a copy, parent untouched
    assert [l["prov"]["origin"] for l in logs[:2]] == ["inherited", "inherited"]
    assert specs[:2] == [{"channels": 4}, {"channels": 6}]
    assert all(p.requires_grad for p in enc.parameters())
    assert enc(torch.rand(2, 3, 8, 8)).shape[1] == 9
    # without inherit: every layer is fitted (original path)
    fitted.clear(), ae_calls.clear()
    ev._grow(g, {}, 1, 0, RS._NullRun())
    assert [i for i, _ in fitted] == [0, 1, 2, 3] and len(ae_calls) == 4
