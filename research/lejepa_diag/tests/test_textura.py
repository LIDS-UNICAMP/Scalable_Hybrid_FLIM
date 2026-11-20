import torch

from research.lejepa_diag import textura as T


def test_blur_sigma0_identity_and_shapes():
    x = torch.randn(2, 3, 20, 20)
    assert torch.equal(T.gaussian_blur(x, 0.0), x)
    assert torch.equal(T.down_up(x, 0.0), x)
    assert torch.equal(T.matched_noise(x, 0.0), x)
    for s in T.SIGMAS[1:]:
        assert T.gaussian_blur(x, s).shape == x.shape and T.down_up(x, s).shape == x.shape


def test_blur_properties():
    c = torch.full((1, 3, 16, 16), 2.5)
    assert torch.allclose(T.gaussian_blur(c, 2.0), c, atol=1e-5)          # preserva constante
    assert T.gaussian_kernel1d(1.0).sum().item() == torch.tensor(1.0).item() or \
        abs(T.gaussian_kernel1d(1.0).sum().item() - 1) < 1e-6
    x = torch.randn(4, 3, 32, 32)
    e = [T.removed_energy_frac(x, s) for s in (0.5, 1, 2, 4)]
    assert all(a < b for a, b in zip(e, e[1:]))                           # mais sigma, mais energia removida


def test_matched_noise_energy():
    x = torch.randn(3, 3, 24, 24, dtype=torch.float64)
    y = T.matched_noise(x, 1.0, seed=5)
    rem = (x - T.gaussian_blur(x, 1.0)).flatten(1).pow(2).sum(1)
    add = (y - x).flatten(1).pow(2).sum(1)
    assert torch.allclose(rem, add, rtol=1e-6)
    assert torch.equal(y, T.matched_noise(x, 1.0, seed=5))               # deterministico


def test_cosine():
    a = torch.randn(5, 7)
    assert torch.allclose(T.cosine_rows(a, a), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(T.cosine_rows(a, -a), -torch.ones(5, dtype=torch.float64))
