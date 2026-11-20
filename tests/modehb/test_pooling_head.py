"""Testes de ga_search/modehb/pooling_head.py."""

import itertools

import pytest
import torch
import torch.nn as nn

from ga_search.modehb.pooling_head import (
    EMBED_DIMS, POOLINGS, EmbeddingHead, encode_tta, gem_pool, tta_cost, tta_transforms,
)

IN_CH = 48


def _encoder():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, IN_CH, 3, padding=1), nn.GELU())


@pytest.mark.parametrize("pooling,dim", list(itertools.product(POOLINGS, EMBED_DIMS)))
def test_output_dim_exact(pooling, dim):
    head = EmbeddingHead(IN_CH, dim, pooling)
    y = head(torch.randn(2, IN_CH, 7, 9))
    assert y.shape == (2, dim) and head.out_dim == head.nominal_dim == dim
    assert torch.isfinite(y).all()
    assert head.proj_flops(7, 9) == 2 * IN_CH * head.width * 63
    assert head.proj_params > 0


@pytest.mark.parametrize("dim", EMBED_DIMS)
def test_moments_without_kurtosis_exact(dim):
    head = EmbeddingHead(IN_CH, dim, "moments", kurtosis=False)
    assert head(torch.randn(2, IN_CH, 5, 5)).shape == (2, dim)


def test_rejects_small_dim():
    with pytest.raises(AssertionError):
        EmbeddingHead(IN_CH, 1024, "gap")


def test_gem_learn_grad_and_p_gt_1():
    head = EmbeddingHead(IN_CH, 1280, "gem_learn")
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        head(torch.randn(4, IN_CH, 6, 6)).sum().backward()
        assert head.p_raw.grad is not None and head.p_raw.grad.abs() > 0
        opt.step()
        assert head.p.item() > 1.0
    with torch.no_grad():
        head.p_raw.fill_(-50.0)
    assert head.p.item() > 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gem_finite_on_negative(dtype):
    x = -torch.rand(2, 8, 5, 5).to(dtype) * 100
    assert torch.isfinite(gem_pool(x)).all()
    for pooling in ("gem", "gem_learn", "combo", "moments"):
        head = EmbeddingHead(8, 1280, pooling).to(dtype)
        assert torch.isfinite(head(x.to(dtype) * 50).float()).all()


def test_tta_no_params_and_n1_plain():
    enc, head = _encoder(), EmbeddingHead(IN_CH, 1536, "meanstd")
    n_params = sum(p.numel() for p in [*enc.parameters(), *head.parameters()])
    x = torch.randn(2, 3, 16, 16)
    for n in (1, 4, 8):
        assert encode_tta(enc, head, x, n).shape == (2, 1536)
    assert sum(p.numel() for p in [*enc.parameters(), *head.parameters()]) == n_params
    assert torch.equal(encode_tta(enc, head, x, 1), head(enc(x)))
    assert [tta_cost(n) for n in (1, 4, 8)] == [1.0, 4.0, 8.0]


def test_tta_views():
    x = torch.randn(1, 3, 6, 8)  # nao quadrado: flip_only n=4 preserva shape
    assert all(t(x).shape == x.shape for t in tta_transforms(4, "flip_only"))
    sq = torch.randn(1, 3, 6, 6)
    views = [t(sq) for t in tta_transforms(8)]
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8
    with pytest.raises(ValueError):
        tta_transforms(8, "flip_only")
    with pytest.raises(ValueError):
        tta_transforms(2)


def test_tta8_matches_eval_svm_gem_tta():
    from eval.svm import _encode_pooled

    class Toy(nn.Module):
        n_layers = 1

        def __init__(self):
            super().__init__()
            self.conv1 = _encoder()

    toy = Toy().eval()
    x = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        ref = _encode_pooled(toy, x, "gem_tta")
        ours = encode_tta(toy.conv1, gem_pool, x, 8)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)
