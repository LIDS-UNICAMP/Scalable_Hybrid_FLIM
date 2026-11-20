"""growth v3.1 bloco 7: Barlow Twins e BYOL conforme os artigos (articles/barlow_twins.md, byol.md), vistas
assimétricas com solarização (LAB), retomada contínua; receitas registradas."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb import ssl_trainers as T  # noqa: E402


def _official_barlow(z1, z2, lambd=0.0051):
    bn = nn.BatchNorm1d(z1.shape[1], affine=False).train()
    c = bn(z1).T @ bn(z2)
    c.div_(z1.shape[0])
    on = torch.diagonal(c).add_(-1).pow_(2).sum()
    n = c.shape[0]
    off = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow_(2).sum()
    return on + lambd * off


def test_barlow_loss_equals_official():
    g = torch.Generator().manual_seed(0)
    z1, z2 = torch.randn(32, 16, generator=g), torch.randn(32, 16, generator=g)
    a, _ = T.barlow_loss(z1, z2)
    assert torch.allclose(a, _official_barlow(z1, z2), rtol=1e-5)


def test_byol_tau_and_loss():
    assert T.byol_tau(0, 10) == pytest.approx(0.996) and T.byol_tau(10, 10) == pytest.approx(1.0)
    p = torch.randn(4, 8)
    assert float(T.byol_loss(p, p)) == pytest.approx(0.0, abs=1e-6)
    assert float(T.byol_loss(p, -p)) == pytest.approx(4.0, abs=1e-5)


def test_asym_views_solarize():
    x = torch.rand(4, 3, 16, 16)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a = T.make_views_asym(x, T.ASYM_VIEWS, g1)
    b = T.make_views_asym(x, T.ASYM_VIEWS, g2)
    assert len(a) == 2 and all(torch.equal(u, v) for u, v in zip(a, b))
    cfg = T.AugConfig(crop_scale=(1.0, 1.0), hflip_p=0.0, vflip_p=0.0, brightness=0.0, contrast=0.0,
                      channel_jitter=0.0, solarize_p=1.0)
    v = T.make_views(x, 1, cfg, seed=0)[0]
    ref = T.make_views(x, 1, T.AugConfig(crop_scale=(1.0, 1.0), hflip_p=0.0, vflip_p=0.0, brightness=0.0,
                                          contrast=0.0, channel_jitter=0.0), seed=0)[0]
    assert torch.allclose(v, torch.where(ref >= 0.5, 1 - ref, ref))


def _toy():
    torch.manual_seed(1)
    enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 16))
    return enc, head


def _loader():
    from torch.utils.data import DataLoader, TensorDataset
    x = torch.rand(32, 3, 16, 16, generator=torch.Generator().manual_seed(2))
    return DataLoader(TensorDataset(x), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(3),
                      drop_last=True)


KW = {"barlow": dict(barlow_kwargs={"projector_dims": [32, 32, 32]}, optimizer="lars", asym_views=True,
                     lr_schedule={"total_steps": 12, "warmup_steps": 3, "end_ratio": 1e-3}, weight_decay=1e-6),
      "byol": dict(byol_kwargs={"projector_dims": [32, 16], "predictor_dims": [32, 16], "tau_base": 0.99},
                   optimizer="lars", asym_views=True,
                   lr_schedule={"total_steps": 12, "warmup_steps": 3, "end_ratio": 0.0}, weight_decay=1.5e-6)}


@pytest.mark.parametrize("ssl", ["barlow", "byol"])
def test_train_and_resume_equals_continuous(ssl):
    e1, h1 = _toy()
    r = T.train_global_ssl(e1, h1, _loader(), ssl, "none", 3, 0.1, 0.1, [], "cpu", None, 0, 0.05, **KW[ssl])
    assert r["steps"] == 12 and r["method"] == ssl
    e2, h2 = _toy()
    ld = _loader()
    st = T.train_global_ssl(e2, h2, ld, ssl, "none", 1, 0.1, 0.1, [], "cpu", None, 0, 0.05, return_state=True,
                            **KW[ssl])["state"]
    if ssl == "byol":
        assert set(st["byol_target"]) == {"encoder", "head", "projector"}
    T.train_global_ssl(e2, h2, ld, ssl, "none", 2, 0.1, 0.1, [], "cpu", None, 0, 0.05, resume=st, **KW[ssl])
    for a, b in zip(list(e1.state_dict().values()) + list(h1.state_dict().values()),
                    list(e2.state_dict().values()) + list(h2.state_dict().values())):
        assert torch.equal(a, b)


def test_barlow_projector_no_bias_and_lr_groups():
    proj = T._make_barlow_projector(8, (16, 16, 4))
    assert all(m.bias is None for m in proj if isinstance(m, nn.Linear))
    e, h = _toy()
    seen = {}
    orig = T.LARS.step

    def spy(self, closure=None):
        seen.update({g["name"]: g["base_lr"] for g in self.param_groups})
        return orig(self, closure)
    T.LARS.step = spy
    try:
        T.train_global_ssl(e, h, _loader(), "barlow", "none", 1, 1.0, 1.0, [], "cpu", None, 0, 0.05, **KW["barlow"])
    finally:
        T.LARS.step = orig
    assert any(abs(v - 0.2) < 1e-12 for v in seen.values()) and any(abs(v - 0.0048) < 1e-12 for v in seen.values())


def test_chain_recipes_barlow_byol():
    g = G.load_growth_config(None, [])
    g["v3"] = {"ssl_recipe": "paper"}
    b = G.ssl_recipe(G.apply_genes(g, {"ssl": "barlow"}))
    assert b["kwargs"]["asym_views"] and b["kwargs"]["barlow_kwargs"]["lambd"] == 0.0051
    o = G.recipe_optim(b, 256, 10, 100)
    assert o["optimizer"] == "lars" and o["lr"] == pytest.approx(1.0)
    y = G.ssl_recipe(G.apply_genes(g, {"ssl": "byol"}))
    assert G.recipe_optim(y, 512, 10, 100)["lr"] == pytest.approx(0.4)
    assert G.apply_genes(g, {"ssl": "byol"})["reg"] == "none"


def test_dino_head_and_loss():
    h = T.DINOHead(16, K=64, hidden=32, bottleneck=8)
    assert [n for n, _ in h.last_layer.named_parameters()] == ["weight_v"] and torch.all(h.last_layer.weight_g == 1)
    import copy
    copy.deepcopy(h)                                              # a professora é uma cópia
    o = h(torch.randn(4, 16))
    assert o.shape == (4, 64)
    t = [torch.randn(4, 64), torch.randn(4, 64)]
    s = [t[0].clone(), t[1].clone(), torch.randn(4, 64)]
    l_ = T.dino_loss(s, t, torch.zeros(1, 64), 0.1, 0.04)
    assert torch.isfinite(l_)


def test_dino_trains_returns_teacher_and_resumes():
    kw = dict(dino_kwargs={"K": 32, "hidden": 16, "bottleneck": 8, "n_local": 2, "local_frac": 0.5,
                           "tau_t_warmup_epochs": 1, "steps_per_epoch": 4},
              optimizer="lars", lr_schedule={"total_steps": 12, "warmup_steps": 2, "end_ratio": 0.016},
              weight_decay=1e-6)
    e1, h1 = _toy()
    r = T.train_global_ssl(e1, h1, _loader(), "dino", "none", 3, 0.1, 0.1, [], "cpu", None, 0, 0.05, **kw)
    assert r["n_views"] == 4 and r["method"] == "dino" and r["steps"] == 12
    e2, h2 = _toy()
    ld = _loader()
    st = T.train_global_ssl(e2, h2, ld, "dino", "none", 1, 0.1, 0.1, [], "cpu", None, 0, 0.05, return_state=True,
                            **kw)["state"]
    assert "center" in st["byol_target"] and "dino_head" in st["byol_target"]
    T.train_global_ssl(e2, h2, ld, "dino", "none", 2, 0.1, 0.1, [], "cpu", None, 0, 0.05, resume=st, **kw)
    for a, b in zip(list(e1.state_dict().values()) + list(h1.state_dict().values()),
                    list(e2.state_dict().values()) + list(h2.state_dict().values())):
        assert torch.equal(a, b)


FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_B0 = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_B0.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v31_barlow_byol_dino_flowers(tmp_path):
    import json
    from ga_search.hmodehb import fitness_v2 as FV
    out = tmp_path / "fl"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=1"]
    cfg = G.build_cfg("flowers", 42, ov, out)
    g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml",
                             ["seeds.top_k=1", "n_parents=1", "n_children=3", "generations=1", "fid_low=1",
                              "fid_high=1", "end_band=B1", "spifil_images_per_class=1", "debug_train=48",
                              "debug_val=32", "test_channel=false", "seeds.external_small=false",
                              f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}",
                              "v3.ssl_recipe=paper", "v3.ssl_batch_size=16", "v3.so2.seed_mode=fixed",
                              "v3.barlow_kwargs={projector_dims: [64, 64, 64]}",
                              "v3.byol_kwargs={projector_dims: [64, 32], predictor_dims: [64, 32]}",
                              "v3.dino_kwargs={K: 64, hidden: 32, bottleneck: 16, n_local: 2}"])
    g["v3"]["genes"] = {"ssl": ["barlow", "byol", "dino"]}
    g["v3"]["families"] = None
    g["v3"]["ops"] = []
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"], v3=g["v3"])
    G.GrowthChain(cfg, g, gev, out).run()
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    assert res
    seen = set()
    for r in res:
        assert r["status"] == "ok", r.get("error")
        FV.check_decisions(r, gev.fitness_spec)
        seen.add(r["ssl_method"])
        assert r["ssl_recipe"]["fiel"] and r["ssl_recipe"]["batch"]["optim"]["optimizer"] == "lars"
    assert seen == {"barlow", "byol", "dino"}, seen
