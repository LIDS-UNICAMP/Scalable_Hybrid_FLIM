import torch
import torch.nn as nn
import torch.nn.functional as F

from ga_search.modehb.softhebb import fit_softhebb, soft_wta, softhebb_update


def _data(seed=0, n=40, b=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, 3, 16, 16, generator=g) for _ in range(n)]


def _conv(seed=0, scale=1.0, **kw):
    torch.manual_seed(seed)
    c = nn.Conv2d(3, 8, 5, bias=True, **kw)
    with torch.no_grad():
        c.weight.mul_(scale / c.weight.flatten(1).norm(dim=1).view(-1, 1, 1, 1))
        c.weight.mul_(torch.linspace(0.3, 3.0, 8).view(-1, 1, 1, 1))  # radii far from 1
    return c


def test_delta_matches_unfold_reference():
    """Conv trick for sum y x equals the explicit patch formula (Eq. 4), with stride/padding."""
    torch.manual_seed(1)
    conv = nn.Conv2d(3, 6, 3, stride=2, padding=1, bias=False)
    x = torch.randn(4, 3, 9, 9)
    w0 = conv.weight.detach().clone()
    softhebb_update(conv, x, lr=1.0, temperature=0.5, adaptive_lr=False, normalize_delta=False)
    got = conv.weight.detach() - w0
    p = F.unfold(x, 3, padding=1, stride=2)  # (B, C*9, L)
    u = torch.einsum("kc,bcl->bkl", w0.flatten(1), p)
    y, _ = soft_wta(u.view(4, 6, 5, 5), 0.5)
    y = y.view(4, 6, -1)
    ref = torch.einsum("bkl,bcl->kc", y, p) - (y * u).sum((0, 2)).view(-1, 1) * w0.flatten(1)
    assert torch.allclose(got.flatten(1), ref, atol=1e-4)


def test_norms_converge_to_unit():
    """Pure SoftHebb (Moraitis et al. 2022): every filter converges to ||w||=1."""
    conv = _conv()
    r0 = (conv.weight.flatten(1).norm(dim=1) - 1).abs().mean().item()
    out = fit_softhebb(conv, _data(), epochs=5, lr=0.05, temperature=1.0, anti_hebbian=False)
    assert out["final_weight_norm_dev"] < 0.01 * r0, (r0, out["final_weight_norm_dev"])
    assert (conv.weight.flatten(1).norm(dim=1) - 1).abs().max() < 0.01


def test_anti_hebbian_winners_converge_to_unit():
    """Soft anti-Hebbian (Journe et al. 2023): filters that win converge to unit norm;
    neurons that never win are only pushed anti-Hebbian and may shrink (known dead-unit case)."""
    conv = _conv()
    fit_softhebb(conv, _data(), epochs=5, lr=0.05, temperature=1.0, anti_hebbian=True)
    r = conv.weight.flatten(1).norm(dim=1)
    assert ((r - 1).abs() < 0.01).float().mean() >= 0.75, r


def test_no_grad_tracking_and_requires_grad_restored():
    conv = _conv()
    seen = []

    def logger(m, step):
        seen.append((conv.weight.requires_grad, conv.weight.grad, m))

    out = fit_softhebb(conv, _data(n=5), epochs=1, lr=0.05, temperature=1.0, logger=logger)
    assert out["steps"] == 5 and len(seen) == 5
    assert all(rg is False and g is None for rg, g, _ in seen)
    for key in ("softhebb/weight_norm", "softhebb/delta_norm", "softhebb/winner_entropy"):
        assert key in seen[0][2]
    assert conv.weight.requires_grad and conv.bias.requires_grad and conv.weight.grad is None
    assert conv.weight.grad_fn is None


def test_requires_grad_restored_when_false_and_on_error():
    conv = _conv()
    conv.weight.requires_grad_(False)
    fit_softhebb(conv, _data(n=2), epochs=1, lr=0.05, temperature=1.0)
    assert conv.weight.requires_grad is False and conv.bias.requires_grad is True

    conv2 = _conv()
    try:
        fit_softhebb(conv2, [torch.randn(2, 5, 8, 8)], epochs=1, lr=0.05, temperature=1.0)
    except RuntimeError:
        pass
    assert conv2.weight.requires_grad is True


def test_bias_untouched_and_callable_loader():
    conv = _conv()
    b0 = conv.bias.detach().clone()
    data = _data(n=3)
    out = fit_softhebb(conv, lambda: iter(data), epochs=2, lr=0.05, temperature=1.0)
    assert out["steps"] == 6
    assert torch.equal(conv.bias.detach(), b0)


def test_deterministic_with_seed():
    a, b = _conv(seed=3), _conv(seed=3)
    fit_softhebb(a, _data(seed=7), epochs=2, lr=0.05, temperature=0.7)
    fit_softhebb(b, _data(seed=7), epochs=2, lr=0.05, temperature=0.7)
    assert torch.equal(a.weight, b.weight)
    c = _conv(seed=4)
    fit_softhebb(c, _data(seed=7), epochs=2, lr=0.05, temperature=0.7)
    assert not torch.equal(a.weight, c.weight)


def test_grouped_and_depthwise_do_not_raise_and_match_per_group():
    torch.manual_seed(3)
    x = torch.randn(4, 4, 9, 9)
    g = nn.Conv2d(4, 8, 3, padding=1, groups=2, bias=False)
    ref = [nn.Conv2d(2, 4, 3, padding=1, bias=False) for _ in range(2)]
    for i, r in enumerate(ref):
        r.weight.data = g.weight.data[i * 4:(i + 1) * 4].clone()
    m = softhebb_update(g, x, lr=0.1, temperature=0.5)
    assert m["softhebb/groups"] == 2.0
    for i, r in enumerate(ref):
        softhebb_update(r, x[:, i * 2:(i + 1) * 2], lr=0.1, temperature=0.5)
        assert torch.allclose(g.weight[i * 4:(i + 1) * 4], r.weight, atol=1e-6)
    dw = nn.Conv2d(4, 4, 3, padding=1, groups=4)
    w0 = dw.weight.detach().clone()
    out = fit_softhebb(dw, [x, x], 1, 0.1, 1.0)
    assert out["steps"] == 2 and not torch.equal(w0, dw.weight)
    assert torch.isfinite(dw.weight).all()
