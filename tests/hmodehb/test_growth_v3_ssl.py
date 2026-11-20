"""growth_v3: SSL como gene (VICReg), cabeça antiga congelada no SSL e força da DA (CPU, minúsculo)."""
import json
import math

import pytest
import torch
import torch.nn as nn

from ga_search.modehb import ssl_trainers as ST
from ga_search.modehb.ssl_trainers import (AUG_PRESETS, AugConfig, aug_config, aug_record, make_views,
                                           method_name, train_global_ssl, vicreg_loss)


def _enc():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())


class _Head(nn.Module):
    """GAP -> Linear -> BN1d (BN para checar que a cabeça congelada não atualiza running stats)."""

    def __init__(self, c=8, d=16):
        super().__init__()
        self.fc = nn.Linear(c, d)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, f):
        return self.bn(self.fc(f.mean(dim=(2, 3))))


def _loader(n=16, bs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 16, 16, generator=g)
    return [(x[i:i + bs], torch.zeros(len(x[i:i + bs]), dtype=torch.long)) for i in range(0, n, bs)]


# ───────────────────────────────────────────── VICReg
def test_vicreg_loss_known_values():
    torch.manual_seed(0)
    z = torch.randn(64, 8) * 3.0
    tot, p = vicreg_loss(z, z.clone())
    assert float(p["inv"]) == 0.0
    # std ~3 > 1 -> hinge zero
    assert float(p["var"]) == 0.0
    # cov igual à fórmula direta
    x = z - z.mean(0)
    c = x.T @ x / 63
    off = (c.pow(2).sum() - c.diagonal().pow(2).sum()) / 8
    assert torch.allclose(p["cov"], 2 * off)
    assert torch.allclose(tot, 25 * p["inv"] + 25 * p["var"] + p["cov"])
    # colapso: z constante -> var = relu(1 - sqrt(eps)) por dim, média/2 por ramo -> soma = 1 - 0.01
    zc = torch.ones(32, 4)
    _, pc = vicreg_loss(zc, zc)
    assert math.isclose(float(pc["var"]), 1 - math.sqrt(1e-4), rel_tol=1e-5)
    assert float(pc["cov"]) == 0.0
    # diagonal-only cov -> cov 0
    zd = torch.tensor([[1., -1.], [-1., 1.], [1., 1.], [-1., -1.]])
    _, pd = vicreg_loss(zd, zd)
    assert abs(float(pd["cov"])) < 1e-12
    # inv = MSE
    a, b = torch.randn(10, 5), torch.randn(10, 5)
    _, pab = vicreg_loss(a, b, sim_coeff=1, std_coeff=0, cov_coeff=0)
    assert torch.allclose(pab["inv"], ((a - b) ** 2).mean())


def test_method_name_vicreg():
    assert method_name("vicreg", "none") == "vicreg"
    assert method_name("vicreg", "sigreg") == "vicreg+sigreg"
    with pytest.raises(ValueError):
        method_name("swav", "none")          # growth v3.1: byol passou a existir (bloco 7)


def test_expander_is_paper_shape():
    e = ST._make_expander(16, (32, 32, 24))
    kinds = [type(m).__name__ for m in e]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Linear", "BatchNorm1d", "ReLU", "Linear"]
    assert e[-1].out_features == 24


def test_vicreg_trains_finite_and_state_has_expander():
    enc, head = _enc(), _Head()
    w0 = enc[0].weight.detach().clone()
    out = train_global_ssl(enc, head, _loader(), "vicreg", "none", 1, 1e-3, 1e-3, [], "cpu", seed=1,
                           vicreg_kwargs={"expander_dims": (32, 32, 32)}, return_state=True)
    assert out["method"] == "vicreg" and out["n_views"] == 2 and out["steps"] == 2
    assert math.isfinite(out["final_loss_total"])
    for k in ("ssl/loss_vicreg_inv", "ssl/loss_vicreg_var", "ssl/loss_vicreg_cov"):
        assert k in out["final"] and math.isfinite(out["final"][k])
    assert "expander" in out["state"]["extra"]
    assert not torch.equal(w0, enc[0].weight)
    with pytest.raises(ValueError):
        train_global_ssl(_enc(), _Head(), _loader(), "vicreg", "none", 1, 1e-3, 1e-3, [], "cpu", n_views=4)


def test_vicreg_resume_equals_continuous():
    kw = dict(vicreg_kwargs={"expander_dims": (16, 16, 16)})
    e1, h1 = _enc(), _Head()
    train_global_ssl(e1, h1, _loader(), "vicreg", "none", 2, 1e-3, 1e-3, [], "cpu", seed=3, **kw)
    e2, h2 = _enc(), _Head()
    st = train_global_ssl(e2, h2, _loader(), "vicreg", "none", 1, 1e-3, 1e-3, [], "cpu", seed=3,
                          return_state=True, **kw)["state"]
    train_global_ssl(e2, h2, _loader(), "vicreg", "none", 1, 1e-3, 1e-3, [], "cpu", seed=3, resume=st, **kw)
    for a, b in zip(e1.state_dict().values(), e2.state_dict().values()):
        assert torch.equal(a, b)


# ───────────────────────────────────────────── freeze_head
@pytest.mark.parametrize("ssl", ["lejepa", "vicreg", "simclr"])
def test_freeze_head_keeps_head_bitwise(ssl):
    enc, head = _enc(), _Head()
    head.fc.weight.requires_grad_(True)
    head.bn.bias.requires_grad_(False)          # requires_grad original misto: restaurado no fim
    rg0 = [p.requires_grad for p in head.parameters()]
    sd0 = {k: v.clone() for k, v in head.state_dict().items()}
    w0 = enc[0].weight.detach().clone()
    out = train_global_ssl(enc, head, _loader(), ssl, "sigreg" if ssl == "lejepa" else "none", 1, 1e-2, 1e-2,
                           [], "cpu", seed=0, freeze_head=True, vicreg_kwargs={"expander_dims": (16, 16, 16)})
    assert out.get("freeze_head") is True
    for k, v in head.state_dict().items():      # pesos E running stats do BN da cabeça
        assert torch.equal(v, sd0[k]), k
    assert not torch.equal(w0, enc[0].weight)   # o gradiente chegou ao encoder
    assert [p.requires_grad for p in head.parameters()] == rg0


def test_freeze_head_false_is_historical():
    a_enc, a_head = _enc(), _Head()
    b_enc, b_head = _enc(), _Head()
    b_head.load_state_dict(a_head.state_dict())
    ra = train_global_ssl(a_enc, a_head, _loader(), "lejepa", "sigreg", 1, 1e-3, 1e-3, [], "cpu", seed=2)
    rb = train_global_ssl(b_enc, b_head, _loader(), "lejepa", "sigreg", 1, 1e-3, 1e-3, [], "cpu", seed=2,
                          freeze_head=False)
    assert ra["final_loss_total"] == rb["final_loss_total"]
    for m1, m2 in ((a_enc, b_enc), (a_head, b_head)):
        for x, y in zip(m1.state_dict().values(), m2.state_dict().values()):
            assert torch.equal(x, y)
    assert "freeze_head" not in ra and "aug" not in ra


# ───────────────────────────────────────────── DA
def _hist_make_views(x, V, cfg=None, seed=None):
    """Cópia literal do make_views do fix8 (CPU), referência bit a bit."""
    import torch.nn.functional as F
    cfg = cfg or AugConfig()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    N, C, H, W = x.shape
    S = cfg.out_size or H
    dev, dt = x.device, x.dtype
    shapes = ((N,),) * 6 + ((N, 1, 1, 1), (N, 1, 1, 1), (N, C, 1, 1))
    cpu = [torch.rand(*s, generator=generator) for _ in range(V) for s in shapes]
    flat = torch.cat([t.reshape(-1) for t in cpu]).to(dev).to(dt)
    draws, off = [], 0
    for t in cpu:
        draws.append(flat[off:off + t.numel()].view(t.shape))
        off += t.numel()
    it = iter(draws)
    rnd = lambda *s: next(it)                    # noqa: E731
    views = []
    for _ in range(V):
        area = cfg.crop_scale[0] + rnd(N) * (cfg.crop_scale[1] - cfg.crop_scale[0])
        lr0, lr1 = math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])
        ratio = torch.exp(lr0 + rnd(N) * (lr1 - lr0))
        sw = torch.sqrt(area * ratio).clamp(max=1.0)
        sh = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = (rnd(N) * 2 - 1) * (1 - sw)
        cy = (rnd(N) * 2 - 1) * (1 - sh)
        fx = torch.where(rnd(N) < cfg.hflip_p, -1.0, 1.0).to(dt)
        fy = torch.where(rnd(N) < cfg.vflip_p, -1.0, 1.0).to(dt)
        theta = torch.zeros(N, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (N, C, S, S), align_corners=False)
        v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        mean = v.mean(dim=(1, 2, 3), keepdim=True)
        con = 1 + (rnd(N, 1, 1, 1) * 2 - 1) * cfg.contrast
        bri = (rnd(N, 1, 1, 1) * 2 - 1) * cfg.brightness
        ch = (rnd(N, C, 1, 1) * 2 - 1) * cfg.channel_jitter
        v = (v - mean) * con + mean + bri + ch
        views.append(v.clamp(0.0, 1.0))
    return views, generator.get_state()


@pytest.mark.parametrize("V", [1, 2, 4])
def test_make_views_default_bitwise_historical(V):
    x = torch.rand(5, 3, 12, 12, generator=torch.Generator().manual_seed(9))
    ref, gstate = _hist_make_views(x, V, seed=11)
    for cfg in (None, AugConfig(), AugConfig(blur_p=0.0, rot90_p=0.0), aug_config("default"), aug_config(None)):
        g = torch.Generator().manual_seed(11)
        got = make_views(x, V, cfg, generator=g)
        assert all(torch.equal(a, b) for a, b in zip(ref, got))
        assert torch.equal(g.get_state(), gstate)


def test_presets_valid_deterministic_and_distinct():
    x = torch.rand(6, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    outs = {}
    for name in AUG_PRESETS:
        cfg = aug_config(name)
        assert isinstance(cfg, AugConfig)
        a = make_views(x, 2, cfg, seed=5)
        b = make_views(x, 2, cfg, seed=5)
        assert all(torch.equal(p, q) for p, q in zip(a, b))
        assert all(v.shape == x.shape and float(v.min()) >= 0 and float(v.max()) <= 1 for v in a)
        outs[name] = a
    assert not torch.equal(outs["weak"][0], outs["strong"][0])
    assert aug_config("strong").crop_scale[0] < aug_config("weak").crop_scale[0]
    with pytest.raises(ValueError):
        aug_config("ultra")
    d = aug_config({"preset": "weak", "blur_p": 0.3})
    assert d.blur_p == 0.3 and d.crop_scale == (0.6, 1.0)


def test_blur_and_rot90_do_something():
    x = torch.rand(8, 3, 16, 16, generator=torch.Generator().manual_seed(2))
    base = dict(crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), hflip_p=0.0, vflip_p=0.0, brightness=0.0,
                contrast=0.0, channel_jitter=0.0)
    plain = make_views(x, 1, AugConfig(**base), seed=0)[0]
    blur = make_views(x, 1, AugConfig(**base, blur_p=1.0, blur_sigma=(1.5, 1.5)), seed=0)[0]
    assert blur.std() < plain.std()                       # blur suaviza
    rot = make_views(x, 1, AugConfig(**base, rot90_p=1.0), seed=0)[0]
    for i in range(8):                                    # cada amostra é uma rotação k em {1,2,3} da original
        assert any(torch.allclose(rot[i], torch.rot90(plain[i], k, dims=(1, 2)), atol=1e-6) for k in (1, 2, 3))
    # sigma constante: blur == conv gaussiana separável direta (amostra 0, centro)
    g = torch.exp(-(torch.arange(5.) - 2) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    k2 = (g[:, None] * g[None, :])
    ref = (plain[0, 0, 5:10, 5:10] * k2).sum()
    assert torch.allclose(blur[0, 0, 7, 7], ref, atol=1e-5)


def test_aug_record_json_and_ssl_result_records_aug():
    for name in list(AUG_PRESETS) + [None]:
        r = aug_record(name)
        json.dumps(r)
        assert set(r) >= {"crop_scale", "blur_p", "rot90_p"}
    out = train_global_ssl(_enc(), _Head(), _loader(), "lejepa", "sigreg", 1, 1e-3, 1e-3, [], "cpu", seed=0,
                           aug_cfg="strong")
    assert out["aug"] == aug_record("strong")
    json.dumps(out["aug"])
