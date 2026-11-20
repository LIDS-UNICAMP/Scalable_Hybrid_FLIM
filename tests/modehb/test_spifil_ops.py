"""Tiny CPU tests for ga_search.modehb.spifil_ops (REAL SPiFiL fits on synthetic PNGs)."""
import numpy as np
import pytest
import torch
from torch import nn

import flim.progressive as progressive
from ga_search.modehb import init as mi
from ga_search.modehb import spifil_ops as so

SIZE = 32
CFG = {"device": "cpu", "image_size": SIZE, "n_superpixels": 30, "impurities": True}  # N = 150 seeds


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    from skimage.io import imsave
    from spifil.data import Sample
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    out = []
    for i in range(6):
        lab = i % 2 + 1
        img = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
        if lab == 2:
            img[:, :, 0] = np.linspace(0, 255, 40)[None].astype(np.uint8)
        p = d / f"{lab:06d}_{i:08d}.png"
        imsave(p, img, check_contrast=False)
        out.append(Sample(p.stem, p, None, lab))
    return out


def _prefix(cout=4):
    torch.manual_seed(0)
    p = nn.Sequential(nn.Conv2d(3, cout, 3, padding=1), nn.Tanh()).eval()   # no dead channels
    p.out_channels = cout
    return p


def _layer(**kw):
    base = dict(active=True, channels=8, kernel=3, conv="conv", residual=False,
                activation="relu", norm="none", pool="none", softhebb=False)
    base.update(kw)
    return base


def _genome(init="spifil", **kw):
    return dict(init=init, seed=7, depth=1, layers=[_layer(**kw)] + [_layer()] * 7)


def _same_structure(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)


def _twin_check(block, prov, cin, x):
    twin = mi.random_twin_block(prov["effective_spec"], 11, in_channels=cin)
    _same_structure(block, twin)
    assert twin(x).shape == block(x).shape


# ─── support / repair / default-off ──────────────────────────────────────────

def test_ops_support_and_default_unchanged():
    for L in (_layer(kernel=7), _layer(kernel=1), _layer(conv="dwsep"), _layer(conv="depthwise"),
              _layer(residual=True), _layer(activation="gelu"), _layer(pool="avg"), _layer(norm="layer")):
        assert mi.spifil_supported(L, spifil_ops=True)
        assert mi.repair_for_spifil(L, spifil_ops=True)["spifil_repairs"] == []
    assert not mi.spifil_supported(_layer(conv="dwsep"))           # default: unchanged
    assert mi.repair_for_spifil(_layer(conv="dwsep"))["conv"] == "conv"
    assert mi.repair_for_spifil(_layer(kernel=9), spifil_ops=True)["kernel"] == 7


def test_default_off_uses_fit_new_block(monkeypatch):
    calls = []

    def _fit(prefix, samples, block_spec, config, *, prefix_id=None):
        calls.append(config)
        return mi.random_twin_block(block_spec, 1, in_channels=3), {"effective_spec": block_spec}

    def _boom(*a, **k):
        raise AssertionError("spifil_ops must not run by default")

    monkeypatch.setattr(progressive, "fit_new_block", _fit)
    monkeypatch.setattr(so, "fit_ops_block", _boom)
    for cfg in ({"device": "cpu"}, {"device": "cpu", "spifil_ops": False}):
        _, prov = mi.init_block(_genome(conv="dwsep", kernel=7), 0, None, ["s"], cfg)
        assert "spifil_ops" not in prov and prov["spifil_repaired"]
    assert len(calls) == 2


def test_exact_allocator():
    from spifil.config import LayerSpec
    from spifil.types import PatchSet
    labels = torch.tensor([1, 1, 2, 2, 3])
    ps = PatchSet(feats=torch.zeros(5, 2), labels=labels, seed_rows=torch.arange(5),
                  image_ids=torch.zeros(5, dtype=torch.int64))
    scores = torch.tensor([0.1, 0.2, 0.9, 0.3, 0.5])
    alloc = so.ExactBudgetAllocator()
    assert alloc(1, LayerSpec(out_channels=1), ps, scores) == {1: 0, 2: 1, 3: 0}
    assert alloc(1, LayerSpec(out_channels=5), ps, scores) == {1: 1, 2: 2, 3: 2}
    assert alloc(1, LayerSpec(out_channels=6), ps, scores) == {1: 2, 2: 2, 3: 2}


# ─── real fits ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kernel", [1, 7])
def test_dense_kernels_faithful(samples, kernel):
    spec = so.to_ops_spec(_layer(kernel=kernel, channels=8, pool="avg", activation="gelu"), 3)
    block, prov = so.fit_ops_block(None, samples, spec, CFG)
    assert block.conv.weight.shape == (8, 3, kernel, kernel)
    assert prov["patch_dimension"] == 3 * kernel * kernel and prov["patch_count"] == 150
    assert prov["spifil_variant"]["faithful"] and isinstance(block.activation, nn.GELU)
    x = torch.rand(2, 3, SIZE, SIZE)
    assert block(x).shape == (2, 8, SIZE // 2, SIZE // 2)
    _twin_check(block, prov, 3, x)


def test_dense_k3_matches_fit_new_block(samples):
    """Same library path: identical filters to flim.progressive.fit_new_block."""
    ref, _ = progressive.fit_new_block(None, samples, {"kernel": 3, "channels": 8}, {**CFG, "seed": 5})
    block, _ = so.fit_ops_block(None, samples, so.to_ops_spec(_layer(), 3), {**CFG, "seed": 5})
    assert torch.allclose(ref.conv.weight, block.conv.weight)
    assert torch.allclose(ref.conv.bias, block.conv.bias)


def test_pointwise_depthwise_dwsep(samples):
    prefix, x = _prefix(4), torch.rand(2, 4, SIZE, SIZE)
    for L, wshape in ((_layer(conv="pointwise", channels=6), (6, 4, 1, 1)),
                      (_layer(conv="depthwise", channels=8, activation="prelu", norm="group"), (8, 1, 3, 3)),
                      (_layer(conv="dwsep", kernel=5, channels=6, activation="selu", norm="layer"), None)):
        block, prov = mi.init_block(_genome(**L), 0, prefix, samples, {**CFG, "spifil_ops": True})
        assert prov["spifil_ops"] and prov["builder"] == "spifil_ops"
        if wshape:
            assert block.conv.weight.shape == wshape
        else:
            assert block.conv.depthwise.weight.shape == (4, 1, 5, 5)
            assert block.conv.pointwise.weight.shape == (6, 4, 1, 1)
            assert prov["stages"]["depthwise"]["D"] == 25 and prov["stages"]["pointwise"]["D"] == 4
        if L["conv"] != "pointwise":
            assert not prov["spifil_variant"]["faithful"]
            dw = block.conv if L["conv"] == "depthwise" else block.conv.depthwise
            assert dw.groups == 4 and dw.weight.flatten(1).norm(dim=1).min() > 0   # no zero filters
        assert all(torch.isfinite(p).all() for p in block.parameters())
        _twin_check(block, prov, 4, x)


def test_residual_identity_and_projection(samples):
    prefix, x = _prefix(4), torch.rand(2, 4, SIZE, SIZE)
    b, p = mi.init_block(_genome(residual=True, channels=4), 0, prefix, samples, {**CFG, "spifil_ops": True})
    assert p["effective_spec"]["skip"] == "identity" and isinstance(b.skip, nn.Identity)
    b, p = mi.init_block(_genome(residual=True, channels=6), 0, prefix, samples, {**CFG, "spifil_ops": True})
    assert p["effective_spec"]["skip"] == "project" and b.skip.weight.shape == (6, 4, 1, 1)
    assert "skip" in p["stages"] and not p["spifil_variant"]["faithful"]
    _twin_check(b, p, 4, x)


def test_next_layer_sees_true_features(samples):
    """Layer 2 is fitted through layer 1 incl. its activation/pool (grid halves)."""
    b1, _ = mi.init_block(_genome(activation="gelu", pool="avg", channels=4), 0, None, samples,
                          {**CFG, "spifil_ops": True})
    prefix = nn.Sequential(b1).eval()
    prefix.out_channels = 4
    b2, p2 = mi.init_block(_genome(conv="depthwise", channels=8), 0, prefix, samples,
                           {**CFG, "spifil_ops": True})
    assert p2["in_channels"] == 4 and p2["patch_count"] > 0
    assert b2(prefix(torch.rand(1, 3, SIZE, SIZE))).shape == (1, 8, SIZE // 2, SIZE // 2)


def test_budget_exhausted_is_infeasible(samples):
    spec = so.to_ops_spec(_layer(kernel=7), 4)                    # D = 4*49 = 196 > N = 150
    with pytest.raises(progressive.InfeasibleCandidate, match="BudgetExhausted"):
        so.fit_ops_block(_prefix(4), samples, spec, CFG)


def test_random_mode_ops_twin():
    g = _genome("random", conv="dwsep", kernel=7, norm="batch", residual=True, channels=4)
    prefix = _prefix(4)
    b, p = mi.init_block(g, 0, prefix, [], {"device": "cpu", "spifil_ops": True})
    b2, _ = mi.init_block(g, 0, prefix, [], {"device": "cpu", "spifil_ops": True})
    assert p["origin"] == "random_he" and p["effective_spec"]["skip"] == "identity"
    assert all(torch.equal(v, b2.state_dict()[k]) for k, v in b.state_dict().items())
    _same_structure(b, mi.random_twin_block(p["effective_spec"], 0, in_channels=4))


# ─── inception / small widths / combined norms ───────────────────────────────

@pytest.mark.parametrize("pool,residual", [("none", False), ("max", True), ("avg", True)])
def test_inception_per_branch(samples, pool, residual):
    prefix, x = _prefix(4), torch.rand(2, 4, SIZE, SIZE)
    g = _genome(conv="inception", channels=8, pool=pool, residual=residual, norm="batch_zca")
    block, prov = mi.init_block(g, 0, prefix, samples, {**CFG, "spifil_ops": True})
    assert prov["spifil_variant"]["name"] == "inception_per_branch"
    br = prov["effective_spec"]["inception_branches"]
    assert [b[0] for b in br] == ["b1", "b3", "b5", "pool"]
    assert sum(b[2] for b in br) == prov["actual_channels"] == block.out_channels
    assert block.conv.branches["b5"].weight.shape[1:] == (4, 5, 5)
    assert block.conv.branches["pool"].proj.weight.shape[1:] == (4, 1, 1)
    assert prov["stages"]["inception_b5"]["D"] == 4 * 25
    assert all(torch.isfinite(p).all() for p in block.parameters())
    s = SIZE // 2 if pool != "none" else SIZE
    assert block(x).shape == (2, prov["actual_channels"], s, s)
    _twin_check(block, prov, 4, x)
    # next layer is fitted through the real inception block
    pre = nn.Sequential(prefix, block).eval()
    pre.out_channels = prov["actual_channels"]
    b2, p2 = mi.init_block(_genome(channels=4), 0, pre, samples, {**CFG, "spifil_ops": True})
    assert p2["in_channels"] == prov["actual_channels"]


def test_small_width_exact(samples):
    """channels (1) < n_classes (2): exact width instead of 'needs at least one selected'."""
    prefix, x = _prefix(4), torch.rand(2, 4, SIZE, SIZE)
    for L in (_layer(channels=1), _layer(conv="dwsep", channels=1), _layer(conv="inception", channels=3)):
        b, p = mi.init_block(_genome(**L), 0, prefix, samples, {**CFG, "spifil_ops": True})
        assert p["actual_channels"] == L["channels"] and p["effective_spec"].get("exact_small_width")
        assert any("exact_small_width" in s for s in p["spifil_variant"]["parts"])
        _twin_check(b, p, 4, x)
    # default (non-ops) path is routed through spifil_ops for this case
    b, p = mi.init_block(_genome(channels=1), 0, prefix, samples, CFG)
    assert p["small_width_route"] and p["actual_channels"] == 1
    _twin_check(b, p, 4, x)


def test_random_twin_inception():
    g = _genome("random", conv="inception", channels=7, norm="batch_dbn", pool="max")
    b, p = mi.init_block(g, 0, _prefix(4), [], {"device": "cpu", "spifil_ops": True})
    assert b(torch.rand(1, 4, SIZE, SIZE)).shape == (1, 7, SIZE // 2, SIZE // 2)
    _same_structure(b, mi.random_twin_block(p["effective_spec"], 0, in_channels=4))
