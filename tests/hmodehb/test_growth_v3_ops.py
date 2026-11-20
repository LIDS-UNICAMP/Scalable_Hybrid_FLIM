"""Cadeia de crescimento v3, morfismos (ga_search/backbones/morphisms_v3.py + whitening no deepen_residual_zero):
preservação EXATA de inception_branch_add / dense_concat, checkpoint, hash antigo, bad_filters/replace_filters."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones import checkpoint as CK  # noqa: E402
from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones import morphisms_v3 as M3  # noqa: E402
from ga_search.backbones.spec import BackboneSpec, HeadSpec, InputSpec, LayerSpec, StageSpec  # noqa: E402

RES = 32


def _tiny(flavor="spifil_ops", seed=0, last_conv="conv", norm0="batch", norm1="batch"):
    L = [LayerSpec(channels=12, kernel=3, conv="conv", norm=norm0, activation="relu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=20, kernel=3, conv="conv", norm=norm1, activation="silu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=16, kernel=3, conv=last_conv, norm="batch", activation="gelu", pool="none",
                   flavor=flavor, init="inherited")]
    spec = BackboneSpec(stem=L[0], stages=[StageSpec("local", L[1:])],
                        head=HeadSpec(embed_dim=1280, pooling="meanstd"), input=InputSpec(resolution=RES),
                        dataset="synthetic")
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():                      # estatísticas não triviais (BN) para o teste valer
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, RES, RES))
    return m.eval()


def _x(n=4, seed=1):
    return torch.rand(n, 3, RES, RES, generator=torch.Generator().manual_seed(seed))


def _assert_exact(res, x=None):
    x = _x() if x is None else x
    P, C = res.parent.eval(), res.child.eval()
    with torch.no_grad():
        a, b = P(x), C(x)
    d = float((a - b).abs().max())
    assert d <= 1e-6 * max(1.0, float(a.abs().max())), d
    g = MO.finalize(res, x, data={"split": "synthetic"})
    assert g.preservation.verdict == "exact", g.preservation
    return d


def _new_mask_false(res, prefix):
    ks = [k for k in res.masks if k.startswith(prefix)]
    assert ks
    for k in ks:
        assert not res.masks[k].any(), k


# ───────────────────────────────────────────────────────────────── (a) inception_branch_add
@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("layer", [1, 2])           # meio e último (cabeça)
@pytest.mark.parametrize("kernel", [1, 5])
def test_branch_add_conv_to_inception_exact(flavor, layer, kernel):
    m = _tiny(flavor)
    res = M3.inception_branch_add(MO.start(m), layer, kernel, 6, seed=3)
    L = res.child.spec.layers()[layer]
    assert L.conv == "inception" and L.channels == m.spec.layers()[layer].channels + 6
    assert L.branches[-1][1] == kernel
    _assert_exact(res)
    _new_mask_false(res, f"blocks.{layer}.conv.branches.{L.branches[-1][0]}.")
    old = L.branches[0][0]
    with torch.no_grad():                  # ramo herdado bit a bit
        assert torch.equal(res.child.blocks[layer].conv.branches[old].weight, m.blocks[layer].conv.weight)


@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("layer", [1, 2])
def test_branch_add_to_inception_exact(flavor, layer):
    m = _tiny(flavor)
    r1 = M3.inception_branch_add(MO.start(m), layer, 3, 4, seed=1)
    r2 = M3.inception_branch_add(r1, layer, 3, 5, seed=2)       # b3 (herdado), b3_2, b3_3
    names = [b[0] for b in r2.child.spec.layers()[layer].branches]
    assert names == ["b3", "b3_2", "b3_3"] and len(set(names)) == len(names)
    _assert_exact(r2)


def test_branch_add_new_filters_hook_and_refusals():
    m = _tiny()
    seen = {}

    def hook(j, conv, rows):
        seen.update(j=j, rows=list(rows), shape=tuple(conv.weight.shape))
        with torch.no_grad():
            conv.weight.fill_(0.5)
    res = M3.inception_branch_add(MO.start(m), 1, 3, 4, new_filters=hook)
    assert seen == {"j": 1, "rows": [0, 1, 2, 3], "shape": (4, 12, 3, 3)}
    _assert_exact(res)
    with pytest.raises(MO.MorphismError):
        M3.inception_branch_add(MO.start(m), 1, 7, 4)
    dw = _tiny(last_conv="dwsep")
    with pytest.raises(MO.MorphismError):
        M3.inception_branch_add(MO.start(dw), 2, 3, 4)


def test_branch_add_then_widen_and_checkpoint(tmp_path):
    m = _tiny()
    res = M3.inception_branch_add(MO.start(m), 1, 5, 6, seed=4)
    res = MO.widen(res, {1: res.child.spec.layers()[1].channels + 4, 2: 20}, seed=5)
    _assert_exact(res)
    MO.finalize(res, None)
    ck = CK.save_backbone(res.child, tmp_path / "b.pt", torchscript=False)
    m2, _ = CK.load_backbone(ck["path"], expected_sha256=ck["file_sha256"])
    x = _x()
    with torch.no_grad():
        assert torch.equal(res.child.eval()(x), m2(x))


# ───────────────────────────────────────────────────────────────── (b) dense_concat
@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("layer", [0, 1])
def test_dense_concat_exact_and_checkpoint(flavor, layer, tmp_path):
    m = _tiny(flavor)
    res = M3.dense_concat(MO.start(m), layer)
    sp = res.child.spec
    assert sp.layers()[layer].dense_out
    assert BB.block_out_channels(sp, layer) == sp.layers()[layer].channels + BB.block_in_channels(sp, layer)
    assert res.child.blocks[layer + 1].conv.weight.shape[1] == BB.block_out_channels(sp, layer)
    _assert_exact(res)
    ck = CK.save_backbone(res.child, tmp_path / "d.pt", torchscript=True)
    m2, spec2 = CK.load_backbone(ck["path"], expected_sha256=ck["file_sha256"])
    assert spec2.layers()[layer].dense_out and spec2.arch_hash() == sp.arch_hash()
    x = _x()
    with torch.no_grad():
        assert torch.equal(res.child.eval()(x), m2(x))
    mask = res.masks[f"blocks.{layer + 1}.conv.weight"]
    c_old = m.spec.block_out_channels(layer)
    assert mask[:, :c_old].all() and not mask[:, c_old:].any()


def test_spec_hash_without_dense_unchanged():
    m = _tiny()
    d = m.spec.arch_dict()
    assert all("dense_out" not in L for st in d["stages"] for L in st["layers"]) and "dense_out" not in d["stem"]
    s2 = BackboneSpec.from_json(m.spec.to_json())
    assert s2.arch_hash() == m.spec.arch_hash()
    legacy = m.spec.to_dict()                # spec JSON antigo, sem a chave
    legacy["stem"].pop("dense_out", None)
    for st in legacy["stages"]:
        for L in st["layers"]:
            L.pop("dense_out", None)
    assert BackboneSpec.from_dict(legacy).arch_hash() == m.spec.arch_hash()


def test_dense_refusals_and_followups():
    m = _tiny()
    with pytest.raises(MO.MorphismError):
        M3.dense_concat(MO.start(m), 2)      # último bloco
    res = M3.dense_concat(MO.start(m), 0)
    with pytest.raises(MO.MorphismError):
        M3.dense_concat(res, 0)
    # widen do bloco dense e do consumidor: exato
    r2 = MO.widen(res, {0: 16, 1: 24}, seed=2)
    _assert_exact(r2)
    # widen da última com o bloco dense no meio + head
    r3 = MO.widen(M3.dense_concat(MO.start(m), 1), {1: 26, 2: 18}, seed=3)
    _assert_exact(r3)
    # deepen depois do bloco dense (largura = saída com passagem)
    r4 = MO.deepen_residual_zero(res, 1)
    assert r4.child.spec.layers()[1].channels == res.child.spec.block_out_channels(0)
    _assert_exact(r4)
    with pytest.raises(MO.NotExactError):
        MO.deepen_identity(res, 1)
    r5 = MO.add_skip_zero(res, 1)
    _assert_exact(r5)
    r6 = M3.inception_branch_add(res, 0, 3, 4)      # ramo no bloco dense
    _assert_exact(r6)
    r7 = M3.dense_concat(res, 1)                    # dense encadeado
    _assert_exact(r7)


# ───────────────────────────────────────────────────────────────── (c) deepen_residual_zero com whitening
@pytest.mark.parametrize("norm", ["zca", "dbn", "batch_dbn", "batch", "none"])
@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
def test_deepen_residual_zero_whitening_exact(norm, flavor):
    m = _tiny(flavor)
    res = MO.deepen_residual_zero(MO.start(m), 2, norm=norm)
    blk = res.child.blocks[2]
    assert res.child.spec.layers()[2].norm == norm
    with torch.no_grad():                    # estatísticas não nulas não podem quebrar a exatidão em eval
        for mod in blk.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_var.fill_(2.0)
    _assert_exact(res)
    if norm in ("zca", "dbn", "batch_dbn"):
        assert "eval" in res.ops[-1].condition
    with pytest.raises(MO.MorphismError):
        MO.deepen_residual_zero(MO.start(m), 2, norm="group")


# ───────────────────────────────────────────────────────────────── (d) bad_filters / replace_filters
def test_bad_filters_dead_and_dup():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(6, 8, 5, 5, generator=g)
    a[:, 3] = 0.7                                   # morto (constante)
    a[:, 5] = 2.0 * a[:, 1] + 0.3 + 1e-4 * torch.randn(6, 5, 5, generator=g)   # duplicado de 1 (maior var)
    r = M3.bad_filters(a, corr_thr=0.95)
    assert r["dead"] == [3] and abs(r["mean"][3] - 0.7) < 1e-6
    assert len(r["dup"]) == 1
    c, keep, aa, bb = r["dup"][0]
    assert (c, keep) == (1, 5)                      # cai o de MENOR variância
    assert abs(aa - 0.5) < 1e-3 and abs(bb + 0.15) < 1e-3
    assert r == M3.bad_filters(a, corr_thr=0.95)    # determinístico
    assert M3.bad_filters(a, n_conv=3)["dead"] == []


@pytest.mark.parametrize("layer", [1, 2])
def test_replace_filters_shapes_masks_deterministic(layer):
    m = _tiny()
    rows = [0, 3]
    r1 = M3.replace_filters(MO.start(m), layer, rows, merge={3: (1, 0.5, 0.1)}, seed=9)
    r2 = M3.replace_filters(MO.start(m), layer, rows, merge={3: (1, 0.5, 0.1)}, seed=9)
    for (k, v), (k2, v2) in zip(r1.child.state_dict().items(), r2.child.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
    for k, v in m.state_dict().items():
        assert r1.child.state_dict()[k].shape == v.shape
    mk = r1.masks[f"blocks.{layer}.conv.weight"]
    assert not mk[rows].any() and mk[[1, 2]].all()
    cons = "head.proj.0.weight" if layer == 2 else f"blocks.{layer + 1}.conv.weight"
    cm = r1.masks[cons]
    assert not cm[:, rows].any() and cm[:, [1, 2]].all()
    W = dict(r1.child.named_parameters())[cons]
    assert float(W[:, rows].abs().max()) == 0.0
    assert r1.ops[-1].op == "replace_bad_filters" and not r1.ops[-1].exact_claimed
    g = MO.finalize(r1, _x(), data={"split": "synthetic"})
    assert g.preservation.verdict in ("approx", "not_preserving")
    with torch.no_grad():                         # parent intacto
        assert torch.equal(r1.parent.blocks[layer].conv.weight, m.blocks[layer].conv.weight)


def test_replace_filters_hook_and_refusals():
    m = _tiny(last_conv="dwsep")
    called = {}
    r = M3.replace_filters(MO.start(m), 1, [2], new_filters=lambda j, c, rows: called.update(j=j, rows=rows))
    assert called == {"j": 1, "rows": [2]}
    k = "blocks.2.conv.depthwise.weight" if "blocks.2.conv.depthwise.weight" in r.masks else "blocks.2.conv.0.weight"
    assert not r.masks[k][2].any()
    with pytest.raises(MO.MorphismError):
        M3.replace_filters(MO.start(m), 2, [0])   # dwsep não é densa
