"""External pretrained seed (ga_search/modehb/external_seed.py): converter equality, function-preserving
growth, per-element protection, MO-DEHB seed injection, LEMONADE inheritance, missing-file fallback and
neutrality (seed off = the exact genome sequence of before the patch)."""
import functools
import hashlib
import json
import math
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from ga_search.modehb import external_seed as ES
from ga_search.modehb import lemonade as LM
from ga_search.modehb import modehb as M
from ga_search.modehb import run_search as RS
from ga_search.modehb import strata as S
from ga_search.modehb.init import repair_for_spifil

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "ga_search" / "config" / "modehb"
REAL = ROOT / "artifacts" / "external_ssl_pequeno" / "remote_sensing_split1_pct100"
DISCOVERY = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery")
SPIFIL_REPAIR = functools.partial(repair_for_spifil, spifil_ops=True)
CS = {"type": "int_range", "min": 1, "max": 1024, "scale": "log", "monotonic": True}
SPACE = {"operator": ["conv", "dwsep", "inception"], "softhebb": [True, False], "kernel": [3, 5, 7],
         "activation": ["relu", "gelu", "silu", "selu"],
         "norm": ["none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"], "pool": ["none", "max", "avg"],
         "residual": [True, False], "pooling": ["gap", "gem", "moments"],
         "protect": ["full", "lowlr", "freeze"], "projector": [True, False], "ae_schedule": ["decoder_first", "joint"]}
GRID = {"width_mults": [0.75, 1.0, 1.5, 2.0], "extra_depths": [0, 1, 2], "residual_extra": True}


# ---------------------------------------------------------------- synthetic FLIM encoder (their format)
def _flim_files(tmp, widths=(4, 6, 8), pools=("max_pool", "none", "none"), extra=None, seed=0):
    torch.manual_seed(seed)
    arch = {"stdev_factor": 0.01, "nlayers": len(widths), "apply_intrinsic_atrous": False}
    sd, cin = {}, 3
    for n, (c, p) in enumerate(zip(widths, pools), 1):
        arch[f"layer{n}"] = {"conv": {"kernel_size": [3, 3, 0], "dilation_rate": [1, 1, 0], "noutput_channels": c,
                                      "nkernels_per_marker": c, "nkernels_per_image": c},
                             "relu": True, "pooling": {"type": p, "size": [3, 3, 0], "stride": 2 if p != "none" else 1}}
        w, b = torch.randn(c, cin, 3, 3) * 0.3, torch.randn(c) * 0.1
        for pre in ("model.encoder.", "model.encoder.blocks."):
            sd[f"{pre}conv{n}.0.weight"], sd[f"{pre}conv{n}.0.bias"] = w, b
        cin = c
    sd["model.decoder.blocks.0.conv1.weight"] = torch.randn(2, 2)          # discarded
    sd["model.projector.0.weight"] = torch.randn(2, 2)                     # discarded
    sd.update(extra or {})
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "architecture.json").write_text(json.dumps(arch))
    torch.save({"state_dict": sd, "epoch": 3}, tmp / "best_kappa.ckpt")
    return {"enabled": True, "name": "ext_test", "arch": str(tmp / "architecture.json"),
            "checkpoint": str(tmp / "best_kappa.ckpt"), "grid": GRID,
            "genes": {"protect": ["lowlr", "freeze"], "pooling": "gem"}}


def _their_encoder(arch_json, ckpt):
    """Re-implementation of discovery ``flim.arch.build_encoder_from_arch`` + the LeJEPA state dict."""
    arch = json.loads(Path(arch_json).read_text())
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    blocks, cin = [], 3
    for n in range(1, arch["nlayers"] + 1):
        d = arch[f"layer{n}"]
        k = d["conv"]["kernel_size"][0]
        conv = nn.Conv2d(cin, d["conv"]["noutput_channels"], k, padding=k // 2)
        conv.weight.data, conv.bias.data = sd[f"model.encoder.conv{n}.0.weight"], sd[f"model.encoder.conv{n}.0.bias"]
        mods = [conv, nn.ReLU(inplace=True)]
        if d["pooling"]["type"] == "max_pool":
            mods.append(nn.MaxPool2d(tuple(d["pooling"]["size"][:2]), stride=d["pooling"]["stride"]))
        blocks.append(nn.Sequential(*mods))
        cin = d["conv"]["noutput_channels"]
    return nn.Sequential(*blocks).eval()


def _feats(enc, x):
    out, h = [], x
    with torch.no_grad():
        for b in enc:
            h = b(h)
            out.append(h)
    return out


# ---------------------------------------------------------------- 1. converter equality
def test_converter_equals_their_encoder_synthetic(tmp_path):
    ec = _flim_files(tmp_path / "ext")
    ec = ES.resolve_config(ec, "x")
    g, states = ES.load_external_seed("x", ec)
    assert g["depth"] == 3 and [L["channels"] for L in g["layers"][:3]] == [4, 6, 8]
    assert [L["pool"] for L in g["layers"][:3]] == ["max", "none", "none"]
    L0 = g["layers"][0]
    assert (L0["conv"], L0["kernel"], L0["activation"], L0["norm"], L0["residual"]) == ("conv", 3, "relu", "none", False)
    assert g["external"] == {"name": "ext_test", "width": 1.0, "extra_depth": 0, "residual": False, "n_imported": 3}
    ours, theirs = ES.build_blocks(states).eval(), _their_encoder(ec["arch"], ec["checkpoint"])
    x = torch.rand(3, 3, 33, 33)
    for a, b in zip(_feats(ours, x), _feats(theirs, x)):
        assert a.shape == b.shape and float((a - b).abs().max()) < 1e-6
    assert ours[0].pool.padding == 0 and ours[0].pool.kernel_size == 3 and ours[0].pool.stride == 2


def test_converter_rejects_unmapped_tensors(tmp_path):
    ec = _flim_files(tmp_path / "ext", extra={"model.encoder.conv2.2.shift": torch.zeros(6)})
    with pytest.raises(NotImplementedError):
        ES.load_external_seed("x", ES.resolve_config(ec, "x"))


@pytest.mark.skipif(not (REAL / "lejepa" / "checkpoints" / "best_kappa.ckpt").exists(), reason="no external encoder")
def test_converter_equals_real_encoder():
    g, states = ES.load_external_seed("remote_sensing", {"arch": str(REAL / "L0" / "architecture.json"),
                                                          "checkpoint": str(REAL / "lejepa/checkpoints/best_kappa.ckpt")})
    assert [L["channels"] for L in g["layers"][:6]] == [40, 80, 152, 152, 152, 312]
    ours = ES.build_blocks(states).eval()
    theirs = _their_encoder(REAL / "L0" / "architecture.json", REAL / "lejepa/checkpoints/best_kappa.ckpt")
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    fa, fb = _feats(ours, x), _feats(theirs, x)
    assert [tuple(f.shape[1:]) for f in fa] == [(40, 111, 111), (80, 55, 55), (152, 27, 27), (152, 13, 13),
                                                 (152, 13, 13), (312, 13, 13)]
    assert max(float((a - b).abs().max()) for a, b in zip(fa, fb)) < 1e-5
    assert sum(p.numel() for p in ours.parameters()) == 982_896


DISCOVERY_SCRIPT = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, os.getcwd())
from flim.build import build
arch, l0, ckpt, xin, out = sys.argv[1:6]
enc = build(arch, "flim", l0)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
sub = {k[len("model.encoder."):]: v for k, v in sd.items() if k.startswith("model.encoder.conv")}
missing, unexpected = enc.load_state_dict(sub, strict=False)
assert not unexpected and all(k.startswith("blocks.") for k in missing), (missing, unexpected)
x = torch.from_numpy(np.load(xin))
with torch.no_grad():
    h, feats = x, []
    for n in range(1, enc.n_layers + 1):
        h = enc.blocks[f"conv{n}"](h)
        feats.append(h.numpy())
np.savez(out, *feats)
'''


@pytest.mark.skipif(not (DISCOVERY / "flim" / "build.py").exists()
                    or not (REAL / "lejepa" / "checkpoints" / "best_kappa.ckpt").exists(),
                    reason="discovery repo / external encoder not available")
def test_converter_equals_discovery_code(tmp_path):
    """Their ORIGINAL code (discovery flim.build + LeJEPA state dict), run in a subprocess."""
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    np.save(tmp_path / "x.npy", x.numpy())
    (tmp_path / "ref.py").write_text(DISCOVERY_SCRIPT)
    subprocess.run([sys.executable, str(tmp_path / "ref.py"), str(REAL / "L0" / "architecture.json"), str(REAL / "L0"),
                    str(REAL / "lejepa/checkpoints/best_kappa.ckpt"), str(tmp_path / "x.npy"), str(tmp_path / "f.npz")],
                   cwd=DISCOVERY, check=True, capture_output=True, env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                                                                         "PATH": "/usr/bin:/bin"})
    ref = np.load(tmp_path / "f.npz")
    _, states = ES.load_external_seed("remote_sensing", {"arch": str(REAL / "L0" / "architecture.json"),
                                                          "checkpoint": str(REAL / "lejepa/checkpoints/best_kappa.ckpt")})
    ours = _feats(ES.build_blocks(states).eval(), x)
    assert len(ours) == len(ref.files) == 6
    assert max(float(np.abs(a.numpy() - ref[f"arr_{i}"]).max()) for i, a in enumerate(ours)) < 1e-5


# ---------------------------------------------------------------- 2. function-preserving growth
@pytest.fixture()
def ext(tmp_path):
    ec = ES.resolve_config(_flim_files(tmp_path / "ext", widths=(4, 6, 8), pools=("none", "none", "none")), "x")
    info = ES.prepare(ec, tmp_path / "out")
    return ec, info


def _rnd_filters(i, prefix, L, cin, n, k):
    g = torch.Generator().manual_seed(100 + i)
    return torch.randn(n, cin, k, k, generator=g), torch.randn(n, generator=g)


def test_growth_grid_preserves_seed_features(ext):
    _, info = ext
    parent = ES.load_parent_blocks(info["converted"])[0]
    x = torch.rand(2, 3, 12, 12)
    seeds = ES.seed_genomes(info, 42)
    assert {(g["external"]["width"], g["external"]["extra_depth"], g["external"]["residual"]) for g in seeds} == \
        {(m, d, False) for m in GRID["width_mults"] for d in GRID["extra_depths"]} | {(m, 1, True) for m in GRID["width_mults"]}
    assert seeds[0]["external"]["width"] == 1.0 and seeds[0]["external"]["extra_depth"] == 0   # pure seed first
    for g in seeds:
        lm = ES.external_layer_map(g)
        grown = ES.grow_blocks(parent, info["layers"], ES.active_layers(g), lm, new_filters=_rnd_filters)
        assert len(grown.blocks) == g["depth"]                                  # every layer mapped
        out = nn.Sequential(*grown.blocks).eval()(x)
        assert out.shape[1] == g["layers"][g["depth"] - 1]["channels"]
        err = ES.preservation_error(parent, grown, x)
        if g["external"]["width"] >= 1.0:
            assert err is not None and err < 1e-6, (g["external"], err)        # step 0 == seed (old channels)
        else:
            assert grown.cut and err is None                                    # cut: only removes filters
        for i, (blk, mk, L, pL, pb) in enumerate(zip(grown.blocks, grown.masks, ES.active_layers(g), info["layers"],
                                                    parent)):
            w, k, ci = blk.conv.weight, pb.conv.out_channels, pb.conv.in_channels
            if L["channels"] >= pL["channels"] and (i == 0 or g["external"]["width"] >= 1.0):
                assert torch.equal(w[:k, :ci], pb.conv.weight)                   # kept filters = parent's
                assert torch.equal(blk.conv.bias[:k], pb.conv.bias)
                assert int(mk["conv.weight"].sum()) == pb.conv.weight.numel()   # exactly the parent's elements
                assert bool(mk["conv.weight"][:k, :ci].all())
                if blk.conv.in_channels > ci:                                   # consumer columns of NEW inputs = 0
                    assert float(w[:k, ci:].abs().max()) == 0.0
            if L["channels"] > pL["channels"]:                                   # NEW filters = SPiFiL callback
                nw, nb = _rnd_filters(i, None, L, blk.conv.in_channels, L["channels"] - pL["channels"], 3)
                assert torch.equal(w[k:], nw) and torch.equal(blk.conv.bias[k:], nb)
                assert not mk["conv.weight"][k:].any() and not mk["conv.bias"][k:].any()
        for blk, mk, kind in zip(grown.blocks[3:], grown.masks[3:], lm[3:]):   # extra layers: identity / zero res
            assert sum(int(m.sum()) for m in mk.values()) == 0
            if kind == "residual_zero":
                assert blk.skip is not None and float(blk.conv.weight.abs().max()) == 0.0


def test_cut_keeps_most_used_filters_unchanged(ext):
    _, info = ext
    parent = ES.load_parent_blocks(info["converted"])[0]
    g = ES.variant_genome(info, 0.5, 0, False, 42, "freeze")
    grown = ES.grow_blocks(parent, info["layers"], ES.active_layers(g), ES.external_layer_map(g))
    assert [b.conv.out_channels for b in grown.blocks] == [2, 3, 4]
    keep0 = ES.keep_filters(parent[0], 2, parent[1])
    assert torch.equal(grown.blocks[0].conv.weight, parent[0].conv.weight[keep0])
    assert all(m.all() for mk in grown.masks for m in mk.values())             # nothing new: no retraining


def test_lemonade_morph_maps_are_function_preserving(ext):
    _, info = ext
    parent = ES.load_parent_blocks(info["converted"])[0]
    pg = M.repair(ES.variant_genome(info, 1.0, 0, False, 42, "lowlr"), image_size=12, min_spatial=1, channel_space=CS)
    x = torch.rand(2, 3, 12, 12)
    lc = LM.lemonade_config({"enabled": True, "function_preserving": True})
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        op = LM.MORPHISMS[int(rng.integers(4))]
        m = LM.morph(pg, op, rng, lc, max_depth=8)
        if m is None:
            continue
        child, info_m = M.repair(m[0], image_size=12, min_spatial=1, channel_space=CS), m[1]
        assert "external" not in child                                           # morph child = normal genome
        lm = ES.morph_layer_map(pg, child, info_m)
        grown = ES.grow_blocks(parent, info["layers"], ES.active_layers(child), lm, new_filters=_rnd_filters)
        seen.add(op)
        if op in ("deepen", "widen") or (op == "kernel" and info_m["to"] > info_m["from"]):
            assert len(grown.blocks) == child["depth"], (op, info_m)
            assert ES.preservation_error(parent, grown, x) < 1e-5, (op, info_m)
        if op == "residual" and child["layers"][info_m["layer"]]["residual"]:
            assert len(grown.blocks) == info_m["layer"]                          # fresh from the changed layer
    assert {"deepen", "widen", "residual", "kernel"} <= seen


# ---------------------------------------------------------------- 3. per-element protection
@pytest.mark.parametrize("opt_cls", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("scale", [0.0, 0.01])
def test_element_protect_equals_lr_scaled_group(opt_cls, scale):
    torch.manual_seed(0)
    x, y = torch.randn(16, 5), torch.randn(16, 3)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:2, :4] = True
    w0 = torch.randn(3, 5)
    # reference: masked elements as a separate parameter with lr * scale
    a = nn.Parameter(w0.clone())
    ref_old, ref_new = nn.Parameter(w0[mask].clone()), nn.Parameter(w0[~mask].clone())
    o1 = opt_cls([{"params": [ref_old], "lr": 1e-2 * scale}, {"params": [ref_new], "lr": 1e-2}], weight_decay=0.01)
    lin = nn.Linear(5, 3, bias=False)
    lin.weight.data = w0.clone()
    o2 = opt_cls(lin.parameters(), lr=1e-2, weight_decay=0.01)
    for _ in range(25):
        W = torch.zeros(3, 5)
        W = W.masked_scatter(mask, ref_old).masked_scatter(~mask, ref_new)
        o1.zero_grad(), ((x @ W.T - y) ** 2).mean().backward(), o1.step()
        with ES.ElementProtect(lin, {"weight": mask}, scale):
            o2.zero_grad(), ((lin(x) - y) ** 2).mean().backward(), o2.step()
    assert torch.allclose(lin.weight.data[mask], ref_old.data, atol=1e-6)
    assert torch.allclose(lin.weight.data[~mask], ref_new.data, atol=1e-6)
    if scale == 0.0:
        assert torch.equal(lin.weight.data[mask], w0[mask])
    assert not torch.equal(lin.weight.data[~mask], w0[~mask])
    del a


def test_element_protect_noop_and_hooks_removed():
    lin = nn.Linear(3, 2)
    opt = torch.optim.Adam(lin.parameters(), lr=0.1)
    w0 = lin.weight.detach().clone()
    with ES.ElementProtect(lin, None, 0.0):
        lin(torch.ones(1, 3)).sum().backward(), opt.step()
    assert not torch.equal(lin.weight, w0)                                       # no masks: plain training
    with ES.ElementProtect(lin, {"weight": torch.ones(2, 3, dtype=torch.bool)}, 0.0):
        w1 = lin.weight.detach().clone()
        opt.zero_grad(), lin(torch.ones(1, 3)).sum().backward(), opt.step()
        assert torch.equal(lin.weight, w1)
    opt.zero_grad(), lin(torch.ones(1, 3)).sum().backward(), opt.step()
    assert not torch.equal(lin.weight, w1)                                       # hooks gone after exit


# ---------------------------------------------------------------- 4. MO-DEHB seed injection
class _Eval:
    def __init__(self):
        self.ctxs = []

    def __call__(self, g, fid, ctx):
        self.ctxs.append((g, fid, dict(ctx)))
        p = S.param_estimate(g)
        q = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3))) * (1 - math.exp(-fid / 3)) + (0.1 if g.get("external") else 0)
        return {"status": "ok", "quality": q, "flops": p * 3000, "params": int(p), "gpu_hours": 0.0,
                "checkpoint": f"ck/{ctx['cand_id']}_f{int(fid)}.pt",
                "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]]}


def _cfg(info=None, lemonade=False, **kw):
    c = dict(fidelities=[1, 3, 9], eta=3, max_brackets=2, seed=42, image_size=224, channel_space=CS,
             spifil_images_choices=[1, 2, 4, 8, 16, 32], initial_pyramid_growth=2.0, random_baseline=False,
             initial_size=8, initial_operators="sampled", stage1_brackets=0, search_space=SPACE, n_classes=14,
             spifil_pool_sizes=[32] * 14,
             stratified={"enabled": True, "calibration": True, "memory_budget_gb": 60, "protected_brackets": 2,
                         "strata": {"small": [0, 800000], "medium": [800000, 1500000], "large": [1500000, 3000000],
                                    "xl": [3000000, 5500000]}, "param_cap": 5500000, "explore_strata": ["large", "xl"]})
    if info is not None:
        c["external_seed"] = info
    if lemonade:
        c["lemonade"] = {"enabled": True, "n_descendants": 6, "morph_prob": 0.6, "function_preserving": True}
    c.update(kw)
    return c


@pytest.fixture(scope="module")
def real_like(tmp_path_factory):
    """Synthetic encoder with the REAL widths/pools (40-80-152-152-152-312) for the search tests."""
    d = tmp_path_factory.mktemp("rl")
    ec = ES.resolve_config({**_flim_files(d / "ext", widths=(40, 80, 152, 152, 152, 312),
                                          pools=("max_pool",) * 4 + ("none",) * 2),
                            "grid": {"width_mults": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5], "extra_depths": [0, 1, 2],
                                     "residual_extra": True}}, "x")
    return ES.prepare(ec, d / "out")


def test_seeds_first_in_initial_population(tmp_path, real_like):
    s = M.MODEHB(_cfg(real_like), _Eval(), tmp_path / "a", spifil_repair=SPIFIL_REPAIR)
    pop = s._initial()
    ext = [g for g in pop if g.get("external")]
    assert pop[:len(ext)] == ext and len(ext) == 16                           # 24 variants, 8 above the 5.5M cap
    assert ext[0]["external"]["width"] == 1.0 and ext[0]["external"]["extra_depth"] == 0
    assert [L["channels"] for L in ext[0]["layers"][:6]] == [40, 80, 152, 152, 152, 312]
    assert all(s.param_fn(g) <= 5_500_000 for g in ext)
    strata = [s._stratum_est(g) for g in ext]
    assert set(strata) == {"medium", "large", "xl"} and s._stratum_est(ext[0]) == "medium"
    assert sum(st in ("large", "xl") for st in strata) > len(strata) / 2        # most seeds >= the seed
    assert all(M.repair(g, **s.repair_kw) == g for g in ext)                    # fixed points of the repair
    assert {g["protect"] for g in ext} == {"lowlr", "freeze"}
    plain = {k: v for k, v in ext[0].items() if k != "external"}
    assert M.arch_hash(plain) != M.arch_hash(ext[0])                           # never aliases a SPiFiL genome
    no = M.MODEHB(_cfg(None), _Eval(), tmp_path / "b", spifil_repair=SPIFIL_REPAIR)._initial()
    assert pop[len(ext):] == no                                                  # rest = the usual seeds


def test_seeded_search_runs_seeds_promotions_and_de(tmp_path, real_like):
    ev = _Eval()
    s = M.MODEHB(_cfg(real_like), ev, tmp_path / "a", spifil_repair=SPIFIL_REPAIR)
    st = s.run()
    ext_jobs = [j for j in st["jobs"] if j["genome"].get("external")]
    calib = [j for j in st["jobs"] if (j.get("op") or {}).get("type") == "calibration" and j["fidelity"] == 1.0]
    assert [j["stratum"] for j in calib] == ["small", "medium", "large", "xl"]
    assert all(j["genome"].get("external") for j in calib[1:])                  # seeds = calibration reps
    assert calib[1]["genome"]["external"]["width"] == 1.0                       # medium rep = the pure seed
    assert {j["fidelity"] for j in ext_jobs} >= {1.0, 3.0, 9.0}                 # seeds promoted / calibrated
    proms = [j for j in ext_jobs if j["resumed_from"] is not None]
    assert proms and all(st["jobs"][j["resumed_from"]]["genome"] == j["genome"] for j in proms)
    seeds = [j for j in st["jobs"] if (j.get("op") or {}).get("type") == "external_seed"]
    assert seeds and all(j["genome"].get("external") and j["op"]["external"] == j["genome"]["external"] for j in seeds)
    assert all((j.get("op") or {}).get("type") != "seed" for j in ext_jobs)
    de = [j for j in st["jobs"] if (j.get("op") or {}).get("type") in ("de", "random")]
    assert de and not any(j["genome"].get("external") for j in de)              # offspring = normal genomes
    assert all(j["status"] == "ok" for j in st["jobs"])
    s2 = M.MODEHB(_cfg(real_like), _Eval(), tmp_path / "a", spifil_repair=SPIFIL_REPAIR)
    assert [j["arch_hash"] for j in s2.run(resume=True)["jobs"]] == [j["arch_hash"] for j in st["jobs"]]


# ---------------------------------------------------------------- 5. LEMONADE inherits from the external parent
def test_lemonade_descendants_inherit_from_external_seed(tmp_path, real_like):
    ev = _Eval()
    s = M.MODEHB(_cfg(real_like, lemonade=True), ev, tmp_path / "l", spifil_repair=SPIFIL_REPAIR)
    pop = s._initial()
    kids = [g for g in pop if M.arch_hash(g) in s._lineage]
    assert len(kids) == 6 and all(s._lineage[M.arch_hash(k)][0].get("external") for k in kids)
    assert not any(k.get("external") for k in kids)
    assert {s._lineage[M.arch_hash(k)][1]["morphism"] for k in kids} >= {"deepen", "widen"}
    st = s.run()
    by_ck = {j["result"]["checkpoint"]: j for j in st["jobs"]}
    inh = [(g, ctx["inherit"]) for g, _, ctx in ev.ctxs if "inherit" in ctx]
    from_ext = [(g, i) for g, i in inh if by_ck[i["checkpoint"]]["genome"].get("external")]
    assert from_ext, "no morphism inherited from the external seed"
    for g, i in from_ext:
        parent = by_ck[i["checkpoint"]]
        assert i["parent_cand_id"] == parent["cand_id"] and i["layer_map"] and i["parent_layers"] == \
            ES.active_layers(parent["genome"])
        assert len(i["layer_map"]) == g["depth"]


def test_lemonade_without_function_preserving_unchanged(tmp_path, real_like):
    """function_preserving off: inherit specs have no layer_map (original LEMONADE semantics)."""
    cfg = _cfg(real_like, lemonade=True)
    cfg["lemonade"]["function_preserving"] = False
    ev = _Eval()
    M.MODEHB(cfg, ev, tmp_path / "l", spifil_repair=SPIFIL_REPAIR).run()
    inh = [ctx["inherit"] for _, _, ctx in ev.ctxs if "inherit" in ctx]
    assert inh and all("layer_map" not in i for i in inh)


# ---------------------------------------------------------------- 6. missing file -> warning, no seed
def test_missing_file_runs_as_without_seed(tmp_path):
    ec = ES.resolve_config({"enabled": True, "arch": str(tmp_path / "nope/{dataset}/a.json"),
                            "checkpoint": str(tmp_path / "nope/{dataset}/b.ckpt")}, "flowers")
    with pytest.warns(UserWarning, match="WITHOUT the seed"):
        info = ES.prepare(ec, tmp_path / "out")
    assert info["missing"] and not info.get("layers")
    a = M.MODEHB(_cfg(info), _Eval(), tmp_path / "a", spifil_repair=SPIFIL_REPAIR)
    b = M.MODEHB(_cfg(None), _Eval(), tmp_path / "b", spifil_repair=SPIFIL_REPAIR)
    assert a._initial() == b._initial()
    assert [j["arch_hash"] for j in a.run()["jobs"]] == [j["arch_hash"] for j in b.run()["jobs"]]


def test_seeded_yaml_resolves_per_dataset_paths():
    c = RS.load_config(CFG_DIR / "search_v3_seeded.yaml", "flowers", 42, [])
    es = c["search"]["external_seed"]
    assert es["arch"] == "artifacts/external_ssl_pequeno/flowers_split1_pct100/L0/architecture.json"
    assert es["checkpoint"].endswith("flowers_split1_pct100/lejepa/checkpoints/best_kappa.ckpt")
    assert "modehb_seeded/flowers" in c["output_dir"]
    lm = RS.load_config(CFG_DIR / "search_v3_lemonade_seeded.yaml", "remote_sensing", 42, [])
    assert "modehb_lemonade_seeded/remote_sensing" in lm["output_dir"]
    assert lm["search"]["lemonade"]["function_preserving"] is True
    for y in ("search_v3.yaml", "search_v3_lemonade.yaml"):
        assert "external_seed" not in RS.load_config(CFG_DIR / y, "remote_sensing", 42, [])["search"]


def test_seeded_yamls_equal_base_except_seed_and_strata():
    for base, seeded in (("search_v3.yaml", "search_v3_seeded.yaml"),
                         ("search_v3_lemonade.yaml", "search_v3_lemonade_seeded.yaml")):
        b = yaml.safe_load((CFG_DIR / base).read_text())
        s = yaml.safe_load((CFG_DIR / seeded).read_text())
        assert s.pop("external_seed")["enabled"] is True
        for k in ("strata", "param_cap", "explore_strata"):
            s["stratified"].pop(k), b["stratified"].pop(k)
        s.pop("paths"), b.pop("paths")
        if "lemonade" in base:
            assert s["lemonade"].pop("function_preserving") is True
            s["lemonade"].pop("n_descendants"), b["lemonade"].pop("n_descendants")
        assert s == b                                       # same fitness (svm kappa, cov_head, C=100), space...
        st = yaml.safe_load((CFG_DIR / seeded).read_text())["stratified"]
        assert list(st["strata"]) == ["small", "medium", "large", "xl"] and st["param_cap"] == 5_500_000
        sv = s["svm_fitness"]
        assert (sv["C"], sv["readout"], sv["kappa_weight"]) == (100, "cov_head", 1.0)


# ---------------------------------------------------------------- 7. neutrality: seed off = before the patch
# Fingerprints computed with the code BEFORE this patch (synthetic evaluator, remote_sensing, seed 42,
# fidelities [5, 15], 2 brackets): initial population + every job (arch_hash, fidelity, role, op type).
GOLDEN = {"search_v3.yaml": {"initial": "f3ca1439f8fabdc9", "n_initial": 16, "jobs": "db6688e654935c96", "n_jobs": 52},
          "search_v3_lemonade.yaml": {"initial": "7002d6cecd14127f", "n_initial": 24, "jobs": "e6d24dc3b213bef1",
                                      "n_jobs": 42}}


class _GoldenEval:
    def __call__(self, g, fid, ctx):
        p = S.param_estimate(g)
        q = min(0.95, 0.3 + 0.05 * math.log10(max(p, 1e3))) * (1 - math.exp(-fid / 30))
        return {"status": "ok", "quality": q, "flops": p * 3000, "params": int(p), "gpu_hours": 0.0,
                "checkpoint": f"ck/{ctx['cand_id']}_f{int(fid)}.pt",
                "effective_specs": [{"channels": L["channels"]} for L in g["layers"][:g["depth"]]]}


@pytest.mark.parametrize("yaml_name", sorted(GOLDEN))
def test_seed_off_genome_sequence_identical_to_before_patch(tmp_path, yaml_name):
    cfg = RS.load_config(CFG_DIR / yaml_name, "remote_sensing", 42, ["max_brackets=2", "search.fidelities=[5,15]"])
    sc = {**cfg["search"], "spifil_pool_sizes": [48] * 14}
    s = M.MODEHB(sc, _GoldenEval(), tmp_path, spifil_repair=SPIFIL_REPAIR)
    init = [M.arch_hash(g) for g in s._initial()]
    st = s.run()
    seq = [(j["arch_hash"], j["fidelity"], j["role"], (j.get("op") or {}).get("type")) for j in st["jobs"]]
    h = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]
    assert {"initial": h(init), "n_initial": len(init), "jobs": h(seq), "n_jobs": len(seq)} == GOLDEN[yaml_name]


# ---------------------------------------------------------------- 8. evaluator: function-preserving growth
class _DS(torch.utils.data.Dataset):
    def __init__(self, n, n_cls=3, off=0):
        self.samples = [types.SimpleNamespace(name=f"s{off + i}", label=i % n_cls) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.rand(3, 12, 12), self.samples[i].label, self.samples[i].name


def _evaluator(tmp_path, monkeypatch):
    import ga_search.progressive_data as PD

    def fake(cfg, out):
        tr, va = _DS(12), _DS(6, off=100)
        return (torch.utils.data.DataLoader(tr, batch_size=4), torch.utils.data.DataLoader(va, batch_size=4),
                tr.samples[:3], {"sha256": "x"})
    monkeypatch.setattr(PD, "load_data", fake)
    monkeypatch.setattr(RS, "build_spifil_pool", lambda sel, train, k, seed: {s.label: [s] for s in sel})
    cfg = RS.load_config(CFG_DIR / "search_v3_seeded.yaml", "remote_sensing", None,
                         ["device=cpu", "data.ram_cache=true", "data.batch_size=4", "wandb.enabled=false"])
    ev = RS.Evaluator.__new__(RS.Evaluator)
    ev.cfg, ev.out, ev.device, ev.seed = cfg, tmp_path, torch.device("cpu"), 0
    ev._load_data()
    return ev


def test_evaluator_grow_fp_masks_ae_and_protection(tmp_path, monkeypatch, ext):
    from ga_search.modehb import local_ae
    from ga_search.modehb.ssl_trainers import train_global_ssl
    ec, info = ext
    ev = _evaluator(tmp_path, monkeypatch)
    ev.cfg["search"]["external_seed"] = info
    monkeypatch.setattr(ES, "fit_new_filters", lambda prefix, donors, L, cin, n, k, cfg: _rnd_filters(0, None, L, cin, n, k))
    ae_blocks = []
    real_ae = local_ae.train_local_ae
    monkeypatch.setattr(local_ae, "train_local_ae",
                        lambda prefix, block, *a, **k: ae_blocks.append(block) or real_ae(prefix, block, *a, **k))
    g = M.repair(ES.variant_genome(info, 1.5, 1, False, 0, "freeze"), image_size=12, min_spatial=1, channel_space=CS)
    inh = ES.external_inherit(info, g)
    enc, specs, logs = ev._grow_fp(g, {}, 1, 0, RS._NullRun(), inh)
    assert [l["prov"]["origin"] for l in logs] == ["fp_widen", "fp_widen", "fp_widen", "fp_identity"]
    assert len(ae_blocks) == 4 and all(p.requires_grad for p in enc.parameters())
    masks = ev._fp_masks
    parent = ES.load_parent_blocks(info["converted"])[0]
    for i, pb in enumerate(parent):                       # freeze: AE never moved an inherited element
        w = enc[i].conv.weight
        m = masks[f"{i}.conv.weight"]
        assert int(m.sum()) == pb.conv.weight.numel()
        assert torch.equal(w[:pb.conv.out_channels, :pb.conv.in_channels], pb.conv.weight)
        assert torch.equal(enc[i].conv.bias[:pb.conv.out_channels], pb.conv.bias)
    assert "3.conv.weight" not in masks                                       # identity layer: all new
    before = {n: p.detach().clone() for n, p in enc.named_parameters()}
    from ga_search.modehb.pooling_head import EmbeddingHead
    head = EmbeddingHead(enc[-1].conv.out_channels, 1280, "gap")
    with ES.ElementProtect(enc, masks, 0.0):
        train_global_ssl(enc, head, ev.train_loader, "lejepa", "sigreg", 1, 1e-3, 1e-3, ["3."], "cpu")
    after = dict(enc.named_parameters())
    for n, m in masks.items():
        assert torch.equal(after[n].detach()[m], before[n][m])          # inherited: frozen through SSL
        if (~m).any():
            assert not torch.equal(after[n].detach()[~m], before[n][~m]) or n.endswith("bias")


def test_evaluator_grow_fp_stops_at_incompatible_layer(tmp_path, monkeypatch, ext):
    """A layer whose genes changed (e.g. residual added on an imported layer) and every later one go
    through the normal SPiFiL path (fresh, free); the previous ones are inherited."""
    from ga_search.modehb import init as I
    from ga_search.modehb import local_ae
    ec, info = ext
    ev = _evaluator(tmp_path, monkeypatch)
    fitted = []

    def fake_init(genome, i, prefix, donors, cfg, prefix_id):
        fitted.append(i)
        cin = ev._out_channels(prefix) if len(prefix) else 3
        return nn.Sequential(nn.Conv2d(cin, int(genome["layers"][i]["channels"]), 3, padding=1), nn.ReLU()), \
            {"origin": "spifil", "effective_spec": {"channels": int(genome["layers"][i]["channels"])}}
    monkeypatch.setattr(I, "init_block", fake_init)
    scales = []
    monkeypatch.setattr(local_ae, "train_local_ae", lambda prefix, block, *a, **k: scales.append(k["conv_lr_scale"]) or {})
    g = M.repair(ES.variant_genome(info, 1.0, 0, False, 0, "lowlr"), image_size=12, min_spatial=1, channel_space=CS)
    child = {**g, "layers": [dict(L) for L in g["layers"]]}
    child.pop("external")
    child["layers"][1]["kernel"] = 1                                          # kernel shrink: not preservable
    inh = {"checkpoint": info["converted"], "layer_map": [0, 1, 2], "parent_layers": info["layers"],
           "parent_cand_id": "c1", "parent_fidelity": 1.0}
    enc, specs, logs = ev._grow_fp(child, {}, 1, 0, RS._NullRun(), inh)
    assert fitted == [1, 2] and [l["prov"]["origin"] for l in logs][:1] == ["inherited"]
    assert scales == [1.0, 1.0]                                                # fresh layers train free
    assert set(ev._fp_masks) == {"0.conv.weight", "0.conv.bias"}
