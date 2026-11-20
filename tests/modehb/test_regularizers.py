"""Tests for ga_search/modehb/regularizers.py.

Run: python tests/modehb/test_regularizers.py   (pytest optional: python -m pytest ...)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ga_search.modehb.regularizers import SIGReg, WeakSIGReg, build_regularizer

N, D = 512, 128


def _gauss(seed=0, D=D):
    g = torch.Generator().manual_seed(10_000 + seed)  # decoupled from global RNG (sketch)
    return torch.randn(N, D, generator=g)


def _rank1(seed=0, D=D):
    g = torch.Generator().manual_seed(10_000 + seed)  # decoupled from global RNG (sketch)
    u = torch.randn(N, 1, generator=g)
    v = torch.randn(1, D, generator=g)
    return u @ v / v.norm()


def _collapsed(seed=0, D=D):
    return torch.full((N, D), 0.3) + 1e-4 * _gauss(seed, D)


NAMES = ("sigreg", "weak_sigreg")


def test_sigreg_gaussian_low_collapsed_high():
    torch.manual_seed(0)
    s = SIGReg()
    lg, lr, lc = s(_gauss()), s(_rank1()), s(_collapsed())
    assert lg < 0.05 * lr and lg < 0.05 * lc, (lg, lr, lc)


def test_weak_gaussian_low_collapsed_high_sketched():
    # Paper regime C=1024 > K=64. Sketch noise floor ||S S^T - I||_F ~ K/sqrt(C) = 2,
    # collapse ~ sqrt(K) = 8. (At C=128 the floor ~5.7 nearly erases the contrast.)
    w = WeakSIGReg()
    for seed in range(5):
        torch.manual_seed(seed)
        lg = w(_gauss(seed, 1024))
        lr = w(_rank1(seed, 1024))
        lc = w(_collapsed(seed, 1024))
        assert lg < 0.6 * lr and lg < 0.6 * lc, (seed, lg, lr, lc)


def test_weak_contrast_unsketched():
    # D <= sketch_dim: no sketch noise, Gaussian loss ~ D/sqrt(N) << collapse loss ~ sqrt(D).
    w = WeakSIGReg(sketch_dim=64)
    lg, lr, lc = w(_gauss()[:, :32]), w(_rank1()[:, :32]), w(_collapsed()[:, :32])
    assert lg < 0.4 * lr and lg < 0.4 * lc, (lg, lr, lc)


def test_gradients_finite():
    for name in NAMES:
        torch.manual_seed(0)
        reg = build_regularizer(name)
        for z0 in (_gauss(), _rank1(), _collapsed()):
            z = z0.clone().requires_grad_(True)
            loss = reg(z)
            assert loss.dim() == 0 and torch.isfinite(loss)
            loss.backward()
            assert z.grad is not None and torch.isfinite(z.grad).all()


def test_deterministic_under_seed():
    for name in NAMES:
        reg = build_regularizer(name)
        z = _gauss(1)
        torch.manual_seed(123)
        a = reg(z)
        torch.manual_seed(123)
        b = reg(z)
        assert torch.equal(a, b), name


def test_not_aliases():
    s, w = build_regularizer("sigreg"), build_regularizer("weak_sigreg")
    assert type(s) is not type(w) and isinstance(s, SIGReg) and isinstance(w, WeakSIGReg)
    for z in (_gauss(), _rank1()):
        torch.manual_seed(0)
        a = s(z)
        torch.manual_seed(0)
        b = w(z)
        assert not torch.allclose(a, b, rtol=1e-3), (a, b)


def test_weak_matches_paper_listing_small_dim():
    # C <= sketch_dim: no sketch, ||Cov(z) - I||_F exactly (Fig. 3).
    z = _gauss()[:, :16]
    zc = z - z.mean(0, keepdim=True)
    ref = torch.norm(zc.T @ zc / (N - 1 + 1e-6) - torch.eye(16), p="fro")
    assert torch.allclose(WeakSIGReg(64)(z), ref, atol=1e-6)


def test_weak_ignores_mean_sigreg_does_not():
    torch.manual_seed(0)
    z = _gauss()
    shifted = z + 5.0
    w = WeakSIGReg()
    torch.manual_seed(0)
    a = w(z)
    torch.manual_seed(0)
    b = w(shifted)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)
    s = SIGReg()
    torch.manual_seed(0)
    c = s(z)
    torch.manual_seed(0)
    d = s(shifted)
    assert d > 10 * c


def test_factory():
    assert build_regularizer("none") is None
    assert isinstance(build_regularizer("weak-sigreg", sketch_dim=32), WeakSIGReg)
    for bad in (lambda: build_regularizer("vicreg"), lambda: SIGReg()(torch.randn(3, 4, 5))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for k, fn in tests:
        fn()
        print(f"PASS {k}")
    print(f"{len(tests)} passed")
