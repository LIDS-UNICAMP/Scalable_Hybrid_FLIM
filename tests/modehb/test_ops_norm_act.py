import itertools

import pytest
import torch
from torch import nn

from ga_search.modehb.ops_norm_act import (
    ACTIVATIONS, CONVS, NORMS, ConvBlock, DecorrelatedBatchNorm, ZCAWhitening, make_act,
    make_conv, make_norm, selu_compat_check,
)


def _cov(x):  # (N,C,H,W) -> (C,C)
    f = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return torch.cov(f.T)


def _correlated(n=64, C=8, hw=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, C, hw, hw, generator=g)
    mix = torch.randn(C, C, generator=g)
    return torch.einsum("dc,nchw->ndhw", mix, z) + 3.0


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_make_act(name):
    x = torch.randn(2, 4, 5, 5)
    assert make_act(name)(x).shape == x.shape


@pytest.mark.parametrize("name", NORMS)
def test_make_norm_shapes(name):
    x = torch.randn(4, 12, 6, 6)
    assert make_norm(name, 12)(x).shape == x.shape


def test_layer_norm_is_channelwise():
    x = torch.randn(2, 6, 4, 4) * 5 + 2
    y = make_norm("layer", 6)(x)
    assert torch.allclose(y.mean(1), torch.zeros(2, 4, 4), atol=1e-5)


@pytest.mark.parametrize("kind,cin,cout,k,stride",
                         [(kd, ci, co, k, s) for kd in CONVS for ci, co in [(4, 8), (6, 8), (8, 8)]
                          for k in (1, 3, 5) for s in (1, 2)])
def test_make_conv_shapes(kind, cin, cout, k, stride):
    y = make_conv(kind, cin, cout, k, stride)(torch.randn(2, cin, 16, 16))
    assert y.shape == (2, cout, 16 // stride, 16 // stride)


def test_depthwise_repair():
    assert make_conv("depthwise", 4, 8, 3).groups == 4 and not make_conv("depthwise", 4, 8, 3).repaired
    m = make_conv("depthwise", 6, 8, 3)
    assert m.repaired and m.groups == 2


def test_zca_identity_cov_on_fit_data():
    x = _correlated()
    z = ZCAWhitening(8, eps=1e-6)
    assert torch.equal(z(x), x)  # unfitted -> identity
    z.fit(x)
    assert not any(p.requires_grad for p in z.parameters())
    assert torch.allclose(_cov(z(x)), torch.eye(8), atol=1e-3)
    # default relative eps (as flim/spifil.py): covariance = U diag(s/(s+eps*mean s)) U^T,
    # a shrunk identity -> eigenvalues in (0, 1], dominant directions ~1
    ev = torch.linalg.eigvalsh(_cov(ZCAWhitening(8).fit(x)(x)))
    assert ev.max() <= 1 + 1e-4 and ev.min() > 0 and ev[-4:].min() > 0.9


def test_zca_fit_has_no_grad():
    x = _correlated().requires_grad_(True)
    z = ZCAWhitening(8).fit(x)
    assert not z.W.requires_grad and not z.mu.requires_grad


def test_dbn_grad_finite_and_decorrelated():
    torch.manual_seed(0)
    x = _correlated(C=12).requires_grad_(True)
    dbn = DecorrelatedBatchNorm(12, group_size=4)
    dbn.train()
    y = dbn(x)
    (y ** 3).mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(dbn.weight.grad).all()
    c = _cov(y.detach())
    for a in range(0, 12, 4):
        assert torch.allclose(c[a:a + 4, a:a + 4], torch.eye(4), atol=1e-2)
    dbn.eval()
    assert torch.isfinite(dbn(x.detach())).all()


def test_dbn_group_size_cap():
    dbn = DecorrelatedBatchNorm(64, group_size=128)
    assert dbn.group_size == 16 and max(b - a for a, b in dbn.bounds) <= 16
    assert len(DecorrelatedBatchNorm(20, group_size=8).bounds) == 3


@pytest.mark.parametrize("conv,norm,act,pool,residual", list(itertools.product(
    CONVS, NORMS, ACTIVATIONS, ("none", "max", "avg"), (False, True))))
def test_convblock_all_combos(conv, norm, act, pool, residual):
    spec = dict(channels=8, kernel=3, conv=conv, residual=residual, activation=act,
                norm=norm, pool=pool)
    b = ConvBlock(spec, cin=8)
    y = b(torch.randn(4, 8, 8, 8))
    s = 4 if pool != "none" else 8
    assert y.shape == (4, 8, s, s) and torch.isfinite(y).all()
    assert b.has_residual == residual


def test_residual_only_when_compatible():
    base = dict(channels=16, kernel=3, conv="conv", residual=True, activation="relu", norm="batch")
    assert ConvBlock(base, cin=16).residual_mode == "identity"
    assert not ConvBlock(base, cin=8).has_residual
    assert not ConvBlock({**base, "stride": 2}, cin=16).has_residual
    p = ConvBlock({**base, "residual_mismatch": "project"}, cin=8)
    assert p.residual_mode == "project" and p(torch.randn(2, 8, 8, 8)).shape == (2, 16, 8, 8)
    assert not ConvBlock({**base, "residual": False}, cin=16).has_residual


def test_selu_compat_check():
    ok = ConvBlock(dict(channels=8, activation="selu", norm="none", dropout=0.1), cin=4)
    r = selu_compat_check(ok)
    assert r["compatible"] and r["init_lecun_normal"] and r["dropout_alpha"]
    bad = ConvBlock(dict(channels=8, activation="selu", norm="batch"), cin=4)
    assert not selu_compat_check(bad)["norm_none"] and not selu_compat_check(bad)["compatible"]
    relu = ConvBlock(dict(channels=8, activation="relu", dropout=0.1), cin=4)
    r = selu_compat_check(relu)
    assert not r["uses_selu"] and not r["dropout_alpha"]


# ─── inception + combined norms ─────────────────────────────────────────────

def test_inception_split_and_degrade():
    from ga_search.modehb.ops_norm_act import InceptionConv, inception_split
    assert inception_split(10) == [["b1", 1, 4], ["b3", 3, 2], ["b5", 5, 2], ["pool", 1, 2]]
    assert [b[0] for b in inception_split(3)] == ["b1", "b3", "b5"]
    assert inception_split(2) == [["b1", 1, 1], ["b3", 3, 1]]
    assert inception_split(1) == [["b1", 1, 1]]
    for cout in range(1, 12):
        m = make_conv("inception", 5, cout, 3, stride=2)
        assert isinstance(m, InceptionConv) and min(b[2] for b in m.branch_spec) >= 1
        assert m(torch.randn(2, 5, 8, 8)).shape == (2, cout, 4, 4)


@pytest.mark.parametrize("name", ["batch_zca", "batch_dbn"])
def test_batch_plus_whitening(name):
    m = make_norm(name, 8, zca_eps=1e-6)
    assert isinstance(m[0], nn.BatchNorm2d)
    x = _correlated()
    if name == "batch_zca":
        assert isinstance(m[1], ZCAWhitening)
        m.eval()
        m[1].fit(m[0](x))                          # fitted on the BN output (train features)
        c = _cov(m(x)).detach()
        off = c - torch.diag(torch.diag(c))
        assert off.abs().max() < 0.1 and torch.diag(c).min() > 0.8   # relative eps shrinks diag
    else:
        assert isinstance(m[1], DecorrelatedBatchNorm)
        y = m.train()(x)
        y.sum().backward()
        assert torch.isfinite(y).all()
