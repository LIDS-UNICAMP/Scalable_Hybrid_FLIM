"""CPU tests for ga_search.modehb.modehb (evaluation is mocked; no training)."""
import hashlib
import json

import numpy as np
import pytest

from ga_search.modehb import modehb as M
from ga_search.modehb.init import spifil_supported


# ─── encoding / repair ────────────────────────────────────────────────────────

def _check_valid(g):
    assert g["init"] in M.INITS and 1 <= g["depth"] <= M.N_LAYERS
    assert g["embed_dim"] in M.EMBED_DIMS and g["embed_dim"] >= 1280
    assert g["spifil_images_per_class"] in M.SPIFIL_IMAGES
    assert len(g["layers"]) == M.N_LAYERS
    spatial, cin = 200, 3
    for i, L in enumerate(g["layers"]):
        assert L["active"] == (i < g["depth"])
        for k, ch in M.LAYER_GENES.items():
            assert L[k] in ch
        if not L["active"]:
            continue
        if g["init"] == "spifil":
            assert spifil_supported(L), L
        if L["conv"] == "pointwise":
            assert L["kernel"] == 1
        if i == 0:
            assert L["conv"] not in ("depthwise", "dwsep") and not L["residual"]
        if L["residual"]:
            assert cin == L["channels"] and L["pool"] == "none"
        if L["pool"] != "none":
            spatial //= 2
        cin = L["channels"]
    assert spatial >= 6


def test_decode_repair_always_valid():
    rng = np.random.default_rng(0)
    for _ in range(400):
        v = rng.uniform(-0.2, 1.2, size=M.DIM)     # out-of-box values are clipped
        g = M.decode(v, seed=3)
        _check_valid(g)
        assert M.repair(g) == g                     # idempotent
        assert M.decode(M.encode(g), seed=3) == g   # bin-centre round trip
        for init in M.INITS:                        # never switches init
            assert M.repair({**g, "init": init})["init"] == init
            _check_valid(M.repair({**g, "init": init}))


def test_spifil_images_gene():
    assert M.SPIFIL_IMAGES == (1, 2, 4, 8, 16)
    assert M.DEFAULTS["spifil_images_choices"] == [1, 2, 4, 8, 16]
    assert M.DIM == 11 + M.N_LAYERS * M.N_LAYER_GENES   # + projector, protect (Hybrid SPiFiL)
    i = M.gene_index("spifil_images_per_class")
    seen = set()
    for x in np.linspace(0, 1, 11):
        v = np.full(M.DIM, 0.5)
        v[i] = x
        g = M.decode(v)
        seen.add(g["spifil_images_per_class"])
        assert M.encode(g)[i] == pytest.approx((M.SPIFIL_IMAGES.index(g["spifil_images_per_class"]) + .5) / 5)
    assert seen == set(M.SPIFIL_IMAGES)
    # custom choices from the search config: decode/repair stay in the grid
    ch = (2, 32)
    for x in (0.0, 0.49, 0.51, 1.0):
        v = np.full(M.DIM, 0.5)
        v[i] = x
        g = M.decode(v, spifil_images_choices=ch)
        assert g["spifil_images_per_class"] == (2 if x < .5 else 32)
        assert M.repair(g, spifil_images_choices=ch) == g
    g0 = M.initial_population(0)[0]
    assert M.repair({**g0, "spifil_images_per_class": 5})["spifil_images_per_class"] == 4   # snapped
    legacy = {k: v for k, v in g0.items() if k != "spifil_images_per_class"}
    assert M.repair(legacy)["spifil_images_per_class"] == 1                                # old default
    assert M.arch_hash(g0) != M.arch_hash({**g0, "spifil_images_per_class": 16})
    # initial population spreads the gene; twins share it
    pop = M.initial_population(0)
    assert {g["spifil_images_per_class"] for g in pop} == set(M.SPIFIL_IMAGES)
    popc = M.initial_population(0, spifil_images_choices=ch)
    assert {g["spifil_images_per_class"] for g in popc} == set(ch)


def test_spifil_repair_applied_and_init_kept():
    g = M.initial_population(0)[0]
    g["layers"][0].update(conv="dwsep", kernel=7, residual=True, activation="gelu", pool="avg")
    r = M.repair(g)
    assert r["init"] == "spifil" and spifil_supported(r["layers"][0])
    assert r["layers"][0]["kernel"] == 5 and r["layers"][0]["activation"] == "silu"


def test_twin_differs_only_in_init():
    pop = M.initial_population(5)
    assert len(pop) >= 16
    assert {g["layers"][0]["channels"] for g in pop} == {16, 32, 64, 128}
    for sp, rd in zip(pop[::2], pop[1::2]):
        assert sp["init"] == "spifil" and rd["init"] == "random"
        assert {k: v for k, v in sp.items() if k != "init"} == {k: v for k, v in rd.items() if k != "init"}
        assert M.arch_hash(sp, include_init=False) == M.arch_hash(rd, include_init=False)
        assert M.arch_hash(sp) != M.arch_hash(rd)
        assert sp["spifil_images_per_class"] == rd["spifil_images_per_class"]
    g = M.decode(np.random.default_rng(1).uniform(size=M.DIM))
    g = M.repair({**g, "init": "spifil"})
    t = M.make_twin(g)
    diff = [k for k in g if g[k] != t[k]]
    assert diff == ["init"]
    g16 = M.repair({**g, "spifil_images_per_class": 16})
    assert M.make_twin(g16)["spifil_images_per_class"] == 16


def test_gene_mask_stages():
    g = M.initial_population(0)[0]
    m1, m2 = M.gene_mask(g, 1), M.gene_mask(g, 2)
    assert not m1[M.gene_index("init")] and not m2[M.gene_index("init")]
    assert m1[M.gene_index("depth")] and m1[M.gene_index("embed_dim")] and m1[M.gene_index("channels", 3)]
    assert m1[M.gene_index("spifil_images_per_class")] and m2[M.gene_index("spifil_images_per_class")]
    assert not m1[M.gene_index("conv", 0)] and m2[M.gene_index("conv", 0)]
    assert m1.sum() == 3 + M.N_LAYERS and m2.sum() == M.DIM - 1


# ─── multi-objective primitives ───────────────────────────────────────────────

F4 = np.array([[1, 4], [2, 2], [4, 1], [3, 3]], float)


def test_nondominated_sort_and_hv_hand_example():
    fronts = M.nondominated_fronts(F4)
    assert sorted(fronts[0].tolist()) == [0, 1, 2] and fronts[1].tolist() == [3]
    # HV w.r.t. (5,5): 4*1 + 3*2 + 1*1 = 11 (slabs sorted by f0)
    assert M.hypervolume(F4, [5, 5]) == pytest.approx(11.0)
    # exclusive contributions: (1,4)->1, (2,2)->3, (4,1)->1, dominated (3,3)->0
    assert M.hv_contributions(F4, [5, 5]) == pytest.approx([1, 3, 1, 0])
    # 3-D path goes through pymoo; unit cube corner point
    assert M.hypervolume([[0, 0, 0]], [1, 1, 1]) == pytest.approx(1.0)
    assert M.hypervolume(F4) == pytest.approx(0.0)   # nothing inside default ref 1.1


def test_dominance_based_selection():
    pop = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.9, 0.9]])
    assert M.mo_select(pop, 1, [0.4, 0.4]) == 1          # child dominates target -> replace target
    assert M.mo_select(pop, 1, [0.6, 0.6]) is None       # target dominates child -> no update
    # mutually non-dominated with target: last front of pop+child is {3}; it is removed
    assert M.mo_select(pop, 1, [0.45, 0.55]) == 3
    # child ND with target 0 but dominated by member 1: fronts {0,1,2},{child},{3} -> 3 removed
    assert M.mo_select(pop, 0, [0.55, 0.6]) == 3
    # same child without the dominated member: the child is the last front -> no update
    assert M.mo_select(pop[:3], 0, [0.55, 0.6]) is None
    # last front with several members: least HV contribution is replaced
    pop2 = np.array([[0.1, 0.1], [0.3, 0.9], [0.5, 0.5], [0.9, 0.3]])
    # last front {1,2,3,child}; exclusive HV: child .05*.15=.0075 < (.3,.9) .2*.05=.01 -> child loses
    assert M.mo_select(pop2, 1, [0.25, 0.95]) is None
    # child (.2,.85) vs last front {1:(.3,.9),2,3}: now (.3,.9) is dominated -> fronts {0},{child,2,3},{1}
    assert M.mo_select(pop2, 2, [0.2, 0.85]) == 1


def test_promotion_takes_whole_fronts():
    F = np.array([[0, 1], [1, 0],                       # front 0
                  [0.1, 1.5], [0.6, 1.2], [1.5, 0.1],   # front 1 ((0.6,1.2) is the middle)
                  [2, 2]])                              # front 2
    assert sorted(M.select_promotions(F, 2)) == [0, 1]
    assert sorted(M.select_promotions(F, 5)) == [0, 1, 2, 3, 4]
    p4 = M.select_promotions(F, 4)                      # last front by crowding: extremes kept
    assert sorted(p4) == [0, 1, 2, 4]
    pe = M.select_promotions(F, 3, tie_breaker="epsnet")
    assert set(pe[:2]) == {0, 1} and pe[2] in (2, 3, 4)


# ─── full loop with mocked evaluate ───────────────────────────────────────────

def fake_eval(genome, fidelity, ctx):
    h = int(hashlib.sha256(M.arch_hash(genome).encode()).hexdigest()[:8], 16)
    flops = sum(L["channels"] ** 2 for L in genome["layers"][:genome["depth"]]) * 9 * 1e4
    q = ((h % 1000) / 1000.0) * (1 - 1 / (1 + fidelity)) * (1.1 if genome["init"] == "spifil" else 1.0)
    return {"quality": min(q, 1.0), "flops": flops, "latency_ms": 1.0, "gpu_hours": 0.01,
            "checkpoint": f"ckpt/{ctx['cand_id']}_{fidelity}.pt",
            "resumed": ctx["resume"]["cand_id"] if ctx["resume"] else None,
            "twin_src": ctx["twin_source"]["cand_id"] if ctx["twin_source"] else None}


CFG = dict(seed=11, fidelities=[5, 15, 45], eta=3, max_brackets=4, stage1_brackets=2)


def _strip(jobs):
    return [{k: v for k, v in j.items() if k != "time"} for j in jobs]


def test_search_twins_promotions_and_restart(tmp_path):
    calls = []

    def ev(g, f, c):
        calls.append((M.arch_hash(g), f))
        return fake_eval(g, f, c)

    full = M.MODEHB(CFG, ev, tmp_path / "full").run()
    jobs = full["jobs"]
    n_calls_full = len(calls)
    assert full["done"] and len(jobs) > 20
    # first bracket: 16 seeds evaluated at the lowest fidelity
    assert [j["fidelity"] for j in jobs[:16]] == [5.0] * 16
    # promotions in the first HB iteration resume lower-fidelity checkpoints and take whole fronts
    b0 = [j for j in jobs if j["generation"] == 0 and j["role"] == "search"]
    low = [j for j in b0 if j["fidelity"] == 5.0]
    mid = [j for j in b0 if j["fidelity"] == 15.0]
    assert all(j["resumed_from"] is not None for j in mid)
    expected = [low[i]["cand_id"] for i in M.select_promotions(np.array([j["F"] for j in low]), len(mid))]
    assert sorted(j["cand_id"] for j in mid) == sorted(expected)
    # every SPiFiL candidate at max fidelity has a random twin along the same fidelity path
    top = [j for j in jobs if j["role"] == "search" and j["fidelity"] == 45.0
           and j["genome"]["init"] == "spifil" and j["status"] == "ok"]
    assert top
    for j in top:
        twins = [t for t in jobs if t["role"] == "twin" and t["twin_of"] == j["cand_id"]]
        path = sorted(x["fidelity"] for x in jobs if x["role"] == "search" and x["arch_hash"] == j["arch_hash"])
        assert sorted(t["fidelity"] for t in twins) == sorted(set(path))   # cached duplicates share a twin
        for t in twins:
            assert M.canonical(t["genome"]) == M.canonical(M.make_twin(j["genome"])) and t["genome"]["seed"] == j["genome"]["seed"]
            assert t["genome"]["spifil_images_per_class"] == j["genome"]["spifil_images_per_class"]
            assert t["result"]["twin_src"] == j["cand_id"] if t["fidelity"] == j["fidelity"] else True
    # stage 1 offspring only differ from their target in channels/depth/embed_dim (+eps exploration)
    st = json.loads((tmp_path / "full" / "search_state.json").read_text())
    assert st["done"] and len(st["jobs"]) == len(jobs)
    assert len({j["genome"]["spifil_images_per_class"] for j in jobs}) > 1

    # ---- crash after 25 real evaluations, then resume
    class Crash(BaseException):
        pass

    n = {"k": 0}

    def crashing(g, f, c):
        n["k"] += 1
        if n["k"] > 25:
            raise Crash()
        return fake_eval(g, f, c)

    with pytest.raises(Crash):
        M.MODEHB(CFG, crashing, tmp_path / "rs").run()
    with pytest.raises(FileExistsError):
        M.MODEHB(CFG, fake_eval, tmp_path / "rs").run(resume=False)   # never overwritten
    calls.clear()
    resumed = M.MODEHB(CFG, ev, tmp_path / "rs").run(resume=True)
    assert _strip(resumed["jobs"]) == _strip(jobs)
    assert resumed["subpops"] == full["subpops"]
    assert len(calls) == n_calls_full - 25             # logged jobs are replayed, not re-trained


def test_failed_evaluation_is_infeasible(tmp_path):
    def ev(g, f, c):
        if g["layers"][0]["channels"] == 128:
            raise RuntimeError("OOM")
        return fake_eval(g, f, c)

    st = M.MODEHB({**CFG, "max_brackets": 1}, ev, tmp_path).run()
    bad = [j for j in st["jobs"] if j["status"] != "ok"]
    assert bad and all(j["F"] == [M.FAIL_OBJ] * 2 for j in bad)


def test_spifil_allocation_fraction(tmp_path):
    st = M.MODEHB({**CFG, "max_brackets": 6}, fake_eval, tmp_path).run()
    new = [j for j in st["jobs"][16:] if j["role"] == "search" and j["resumed_from"] is None]
    frac = np.mean([j["genome"]["init"] == "spifil" for j in new])
    assert 0.5 < frac <= 1.0


# ─── opt-in integer channel space (channel_space, search_v3.yaml) ────────────

CS = {"type": "int_range", "min": 1, "max": 1024, "scale": "log", "monotonic": True}


def _check_channels(g, cs=CS):
    act = [L["channels"] for L in g["layers"] if L["active"]]
    for L in g["layers"]:
        c = L["channels"]
        assert isinstance(c, int) and not isinstance(c, bool)
        assert cs["min"] <= c <= cs["max"]
    assert all(a <= b for a, b in zip(act, act[1:])), act
    cin = 3
    for i, L in enumerate(g["layers"][:g["depth"]]):
        if L["conv"] == "depthwise":
            assert i > 0 and L["channels"] % cin == 0
        if L["residual"]:
            assert cin == L["channels"] and L["pool"] == "none"
        cin = L["channels"]


def test_channel_space_property_1000():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(1000):
        g = M.decode(rng.uniform(size=M.DIM), 7, channel_space=CS)
        _check_channels(g)
        assert g["embed_dim"] >= 1280
        if g["init"] == "spifil":
            assert all(spifil_supported(L) for L in g["layers"][:g["depth"]])
        assert M.repair(g, channel_space=CS) == g                           # idempotent
        assert M.decode(M.encode(g, channel_space=CS), 7, channel_space=CS) == g   # encode inverse
        tw = M.make_twin(g)
        assert [L["channels"] for L in tw["layers"]] == [L["channels"] for L in g["layers"]]
        assert M.arch_hash(tw, include_init=False) == M.arch_hash(g, include_init=False)
        seen.update(L["channels"] for L in g["layers"])
    assert min(seen) <= 4 and max(seen) >= 900 and len(seen) > 200   # whole integer range explored


def test_channel_space_repair_running_max_and_clamp():
    g = M.initial_population(0, channels=(64,), depths=(4,), channel_space=CS)[0]
    for L, c in zip(g["layers"], (0, 2000, 8, 300)):
        L["channels"] = c
    r = M.repair(g, channel_space=CS)
    assert [L["channels"] for L in r["layers"][:4]] == [1, 1024, 1024, 1024]
    assert M.repair(r, channel_space=CS) == r


def test_channel_space_decode_encode_scalars():
    for c in (1, 2, 3, 17, 24, 48, 100, 513, 1023, 1024):
        assert M._dec_channels(M._enc_channels(c, M._channel_space(CS)), M._channel_space(CS)) == c
    assert M._dec_channels(0.0, M._channel_space(CS)) == 1
    assert M._dec_channels(1.0, M._channel_space(CS)) == 1024
    assert M._dec_channels(0.5, M._channel_space(CS)) == 32    # log scale: geometric midpoint


def test_channel_space_initial_population():
    pop = M.initial_population(0, channel_space=CS)
    assert sorted({g["layers"][0]["channels"] for g in pop}) == [16, 32, 64, 128]
    for g in pop:
        _check_channels(g)


def test_channel_space_absent_is_default():
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.uniform(size=M.DIM)
        assert M.decode(v, 3) == M.decode(v, 3, channel_space=None)
        assert all(L["channels"] in M.CHANNELS for L in M.decode(v, 3)["layers"])
    with pytest.raises(ValueError):
        M._channel_space({"type": "choice"})


def test_widths_monotonic():
    assert M.widths_monotonic([16, 16, 32]) and not M.widths_monotonic([32, 24])


def test_modehb_channel_space_end_to_end(tmp_path):
    def ev(genome, fid, ctx):
        _check_channels(genome)
        return {"status": "ok", "quality": float(np.mean([L["channels"] for L in genome["layers"]])) / 1024,
                "flops": 1e8, "gpu_hours": 0.0}
    cfg = {"fidelities": [1, 3], "max_brackets": 3, "seed": 0, "channel_space": CS}
    st = M.MODEHB(cfg, ev, tmp_path).run()
    assert st["jobs"] and all(j["status"] == "ok" for j in st["jobs"])


# ─── search_v3: SPiFiL-only, operators from start, YAML space, sampled seeds ─

import functools

from ga_search.modehb.init import repair_for_spifil

SPACE = {"operator": ["conv", "dwsep", "inception"], "softhebb": [True, False], "kernel": [3, 5, 7],
         "activation": ["relu", "gelu", "silu", "selu"],
         "norm": ["none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"],
         "pool": ["none", "max", "avg"], "residual": [True, False], "pooling": ["gap", "gem", "moments"]}
NSS = M.normalize_search_space(SPACE)
OPS_REPAIR = functools.partial(repair_for_spifil, spifil_ops=True)
V3 = dict(seed=3, fidelities=[1, 3, 9], eta=3, max_brackets=4, stage1_brackets=0, random_baseline=False,
          initial_size=16, initial_operators="sampled", search_space=SPACE)


def _in_space(g):
    assert g["pooling"] in NSS["pooling"]
    for L in g["layers"]:
        for k in ("conv", "kernel", "activation", "norm", "pool", "residual", "softhebb"):
            assert L[k] in NSS[k], (k, L[k])


def test_search_space_normalize_and_reject():
    assert NSS["conv"] == ("conv", "dwsep", "inception") and NSS["softhebb"] == (True, False)
    assert M.normalize_search_space(None) is None
    assert M.normalize_search_space({"batchnorm": [True], "whitening": ["stat"]}) is None   # legacy keys ignored
    with pytest.raises(ValueError):
        M.normalize_search_space({"operator": ["softhebb_conv"]})


def test_decode_restricted_to_yaml_space():
    rng = np.random.default_rng(0)
    for init in ("spifil", "random"):
        for _ in range(300):
            v = rng.uniform(size=M.DIM)
            v[M.gene_index("init")] = 0.25 if init == "spifil" else 0.75
            g = M.decode(v, 0, search_space=SPACE, spifil_repair=OPS_REPAIR)
            _in_space(g)
            assert g["embed_dim"] >= 1280
            assert M.repair(g, search_space=SPACE, spifil_repair=OPS_REPAIR) == g     # idempotent
            enc = M.encode(g, search_space=SPACE)
            assert M.decode(enc, 0, search_space=SPACE, spifil_repair=OPS_REPAIR) == g


def test_search_space_absent_is_default():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = rng.uniform(size=M.DIM)
        assert M.decode(v, 0) == M.decode(v, 0, search_space=None)
    a = M.initial_population(5)
    assert len(a) == 16 and sum(g["init"] == "random" for g in a) == 8
    assert all(L["conv"] == "conv" and L["activation"] == "relu" for g in a for L in g["layers"])


def test_initial_population_spifil_only_sampled_diverse():
    pop = M.initial_population(7, pyramid_growth=2.0, twins=False, size=16, sample_operators=True,
                               search_space=SPACE, spifil_repair=OPS_REPAIR)
    assert len(pop) == 16 and all(g["init"] == "spifil" for g in pop)
    assert len({M.arch_hash(g) for g in pop}) == 16
    for g in pop:
        _in_space(g)
        w = [L["channels"] for L in g["layers"][:g["depth"]]]
        assert w == sorted(w)                                   # pyramid widths kept
    acts = {L["activation"] for g in pop for L in g["layers"][:g["depth"]]}
    kers = {L["kernel"] for g in pop for L in g["layers"][:g["depth"]]}
    assert acts == set(NSS["activation"]) and kers == set(NSS["kernel"])
    assert {g["pooling"] for g in pop} == set(NSS["pooling"])
    assert len({L["norm"] for g in pop for L in g["layers"][:g["depth"]]}) >= 3
    assert pop == M.initial_population(7, pyramid_growth=2.0, twins=False, size=16, sample_operators=True,
                                       search_space=SPACE, spifil_repair=OPS_REPAIR)   # seeded


def test_all_genes_mutable_at_bracket0():
    g = M.initial_population(0)[0]
    m = M.gene_mask(g, 2)
    assert m.sum() == M.DIM - 1 and not m[M.gene_index("init")]


def test_v3_search_no_random_and_operators_from_start(tmp_path):
    seen = []

    def ev(genome, fid, ctx):
        seen.append((genome, ctx, s.state["bracket"], s.state["stage"]))
        return fake_eval(genome, fid, ctx)

    s = M.MODEHB(V3, ev, tmp_path, spifil_repair=OPS_REPAIR)
    assert all(g["init"] == "spifil" for g in s._initial()) and len(s._initial()) == 16
    st = s.run()
    assert st["jobs"] and all(j["genome"]["init"] == "spifil" and j["role"] == "search" for j in st["jobs"])
    assert all(ctx["role"] == "search" and ctx["twin_source"] is None for _, ctx, _, _ in seen)
    assert all(stage == 2 for _, _, _, stage in seen)          # every gene unlocked from bracket 0
    for j in st["jobs"]:
        _in_space(j["genome"])
    # offspring in bracket 0 (after the 16 seeds) already vary operator genes
    off0 = [g for g, ctx, b, _ in seen if b == 0][16:]
    assert off0 and len({(L["activation"], L["norm"], L["conv"]) for g in off0 for L in g["layers"]}) > 4


def test_v3_default_keeps_twins(tmp_path):
    st = M.MODEHB({**CFG, "max_brackets": 3}, fake_eval, tmp_path).run()
    assert any(j["role"] == "twin" for j in st["jobs"])
