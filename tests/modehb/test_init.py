"""Tiny CPU tests for ga_search.modehb.init (SPiFiL fit is mocked)."""
import math

import pytest
import torch
from torch import nn

import flim.progressive as progressive
from ga_search.modehb import init as mi


def _layer(**kw):
    base = dict(active=True, channels=8, kernel=3, conv="conv", residual=False,
                activation="relu", norm="none", pool="max", softhebb=False)
    base.update(kw)
    return base


def _genome(init="spifil", **layer_kw):
    return dict(init=init, seed=7, depth=1, layers=[_layer(**layer_kw)] + [_layer()] * 7,
                ssl="lejepa", reg="sigreg", embed_dim=1280, pooling="gap", ae_target="pixel")


@pytest.fixture
def fake_fit(monkeypatch):
    """Mock fit_new_block: returns a twin with fewer channels, like the allocator."""
    calls = []

    def _fit(prefix, samples, block_spec, config, *, prefix_id=None):
        calls.append((block_spec, config, prefix_id))
        eff = dict(block_spec, channels=max(1, block_spec["channels"] - 2))
        block = mi.random_twin_block(eff, 999, in_channels=3)
        return block, {"origin": "spifil", "effective_spec": eff,
                       "requested_spec": block_spec, "seed": config["seed"]}

    monkeypatch.setattr(progressive, "fit_new_block", _fit)
    return calls


def test_supported_and_repair():
    assert mi.spifil_supported(_layer())
    for bad in (dict(kernel=7), dict(kernel=1), dict(conv="dwsep"), dict(residual=True),
                dict(activation="gelu"), dict(pool="avg")):
        assert not mi.spifil_supported(_layer(**bad))
        fixed = mi.repair_for_spifil(_layer(**bad))
        assert fixed["spifil_repairs"]
        assert mi.spifil_supported(fixed)
    r = mi.repair_for_spifil(_layer(kernel=7, conv="depthwise", activation="gelu", pool="avg"))
    assert (r["kernel"], r["conv"], r["activation"], r["pool"]) == (5, "conv", "silu", "max")
    assert mi.repair_for_spifil(_layer(kernel=1))["kernel"] == 3
    assert mi.repair_for_spifil(_layer())["spifil_repairs"] == []


def test_random_twin_matches_spifil_topology_and_is_seeded():
    eff = dict(op="conv", connection="sequential", kernel=5, channels=6, activation="silu",
               pool_stride=2, bn=True)
    a = mi.random_twin_block(eff, 3, in_channels=4)
    b = mi.random_twin_block(eff, 3, in_channels=4)
    c = mi.random_twin_block(eff, 4, in_channels=4)
    assert list(dict(a.named_children())) == ["conv", "bn", "activation", "pool"]
    assert a.conv.weight.shape == (6, 4, 5, 5) and a.out_channels == 6
    assert isinstance(a.activation, nn.SiLU) and isinstance(a.pool, nn.MaxPool2d)
    assert torch.equal(a.conv.weight, b.conv.weight)
    assert not torch.equal(a.conv.weight, c.conv.weight)
    assert torch.all(a.conv.bias == 0)
    assert a(torch.randn(1, 4, 16, 16)).shape == (1, 6, 8, 8)
    # forked RNG: global stream untouched
    torch.manual_seed(0); x = torch.rand(1)
    torch.manual_seed(0); mi.random_twin_block(eff, 3, in_channels=4); y = torch.rand(1)
    assert torch.equal(x, y)


def test_init_block_spifil_uses_fit_new_block(fake_fit):
    g = _genome("spifil", kernel=7, conv="dwsep")
    block, prov = mi.init_block(g, 0, None, ["s"], {"device": "cpu"}, prefix_id="p0")
    spec, cfg, pid = fake_fit[0]
    assert spec["kernel"] == 5 and spec["op"] == "conv" and spec["pool_stride"] == 2
    assert cfg["seed"] == 7 and pid == "p0"
    assert prov["spifil_repaired"] and prov["init"] == "spifil" and prov["layer_idx"] == 0
    # random twin of the SPiFiL effective spec has identical shapes
    twin = mi.random_twin_block(prov["effective_spec"], 1, in_channels=3)
    assert {k: v.shape for k, v in twin.state_dict().items()} == \
           {k: v.shape for k, v in block.state_dict().items()}


def test_init_block_random_seed_and_prefix_channels():
    g = _genome("random", channels=16, kernel=3, pool="none")
    prefix = nn.Sequential(nn.Conv2d(3, 5, 3))
    b1, p1 = mi.init_block(g, 1, prefix, [], {"device": "cpu"})
    b2, _ = mi.init_block(g, 1, prefix, [], {"device": "cpu"})
    b3, _ = mi.init_block(g, 2, prefix, [], {"device": "cpu"})
    assert b1.conv.in_channels == 5 and p1["seed"] == 8 and p1["origin"] == "random_he"
    assert torch.equal(b1.conv.weight, b2.conv.weight)
    assert b3.conv.weight.shape != b1.conv.weight.shape or \
        not torch.equal(b1.conv.weight, b3.conv.weight)
    with pytest.raises(ValueError):
        mi.init_block(dict(g, init="bogus"), 0, None, [], {})


def test_kernel_diagnostics():
    torch.manual_seed(0)
    w = torch.randn(8, 3, 3, 3)
    d = mi.kernel_diagnostics(w)
    assert all(isinstance(v, float) for v in d.values())
    assert 0 < d["filter_diversity"] < 2 and math.isnan(d["featmap_similarity"])
    dup = torch.cat([w[:1]] * 4)
    dd = mi.kernel_diagnostics(dup)
    assert dd["filter_redundancy"] == 1.0 and dd["filter_diversity"] < 1e-5
    feats = torch.relu(nn.functional.conv2d(torch.randn(2, 3, 12, 12), w))
    df = mi.kernel_diagnostics(w, feats)
    assert 0 <= df["featmap_similarity"] <= 1 and 0 < df["pattern_coverage"] <= 1
    assert 1 <= df["activation_effective_rank"] <= 8 and df["dead_fraction"] == 0
    flat = torch.zeros(2, 4, 5, 5); flat[:, 0] = torch.randn(2, 5, 5)
    dz = mi.kernel_diagnostics(torch.randn(4, 3, 3, 3), flat)
    assert dz["dead_fraction"] == 0.75 and dz["pattern_coverage"] == 0.25
    assert math.isnan(mi.kernel_diagnostics(torch.randn(1, 3, 3, 3))["filter_diversity"])


def test_small_width_routes_to_spifil_ops(monkeypatch, fake_fit):
    from types import SimpleNamespace
    from ga_search.modehb import spifil_ops as so
    seen = []

    def _ops(prefix, samples, spec, cfg, *, prefix_id=None):
        seen.append(spec)
        return nn.Identity(), {"effective_spec": spec}

    monkeypatch.setattr(so, "fit_ops_block", _ops)
    smp = [SimpleNamespace(label=c) for c in (1, 2, 3)]
    _, prov = mi.init_block(_genome(channels=2, kernel=7), 0, None, smp, {"device": "cpu"})
    assert prov["small_width_route"] and seen[0]["channels"] == 2 and seen[0]["kernel"] == 5
    assert not fake_fit                                            # fit_new_block not called
    mi.init_block(_genome(channels=3), 0, None, smp, {"device": "cpu"})
    assert len(fake_fit) == 1
