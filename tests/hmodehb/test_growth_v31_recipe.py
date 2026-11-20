"""growth v3.1 bloco 2: receita fiel com batch maior (LARS oficial, AdamW + warmup + cosseno, lr = f(B), época =
passada completa, batch por OOM -> metade registrado), retomada contínua da agenda pelo passo global."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb import ssl_trainers as T  # noqa: E402


def _ref_lars_step(p, grad, mu, lr, wd, m=0.9, eta=1e-3):
    dp = grad.clone()
    if p.ndim != 1:
        dp = dp + wd * p
        pn, un = p.norm(), dp.norm()
        q = eta * pn / un if (pn > 0 and un > 0) else torch.tensor(1.0)
        dp = dp * q
    mu = m * mu + dp
    return p - lr * mu, mu


def test_lars_matches_official_formula():
    torch.manual_seed(0)
    w, b = nn.Parameter(torch.randn(4, 3)), nn.Parameter(torch.randn(4))
    opt = T.LARS([w, b], lr=0.5, weight_decay=1e-4, momentum=0.9, eta=1e-3,
                 weight_decay_filter=T.exclude_bias_and_norm, lars_adaptation_filter=T.exclude_bias_and_norm)
    refs = {"w": (w.detach().clone(), torch.zeros(4, 3)), "b": (b.detach().clone(), torch.zeros(4))}
    for it in range(3):
        gw, gb = torch.randn(4, 3), torch.randn(4)
        w.grad, b.grad = gw.clone(), gb.clone()
        opt.step()
        refs["w"] = _ref_lars_step(refs["w"][0], gw, refs["w"][1], 0.5, 1e-4)
        refs["b"] = _ref_lars_step(refs["b"][0], gb, refs["b"][1], 0.5, 1e-4)
        assert torch.allclose(w, refs["w"][0], atol=1e-6) and torch.allclose(b, refs["b"][0], atol=1e-6)


def test_schedule_factor_shape():
    f = lambda s: T.schedule_factor(s, 100, 10, 1e-3, 0.0)        # noqa: E731
    assert f(0) == 0.0 and f(10) == pytest.approx(1.0) and f(100) == pytest.approx(1e-3)
    vals = [f(s) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert T.schedule_factor(0, 100, 10, 1e-3, 0.01) == pytest.approx(0.01)


def test_recipe_optim_values():
    g = G.load_growth_config(None, [])
    g["v3"] = {"ssl_recipe": "paper"}
    s = G.recipe_optim(G.ssl_recipe(G.apply_genes(g, {"ssl": "simclr"})), 256, 10, 100)
    assert s["optimizer"] == "lars" and s["lr"] == pytest.approx(0.075 * 16)
    assert s["lr_schedule"]["warmup_steps"] == 100 and s["lr_schedule"]["end_ratio"] == 0.0
    v = G.recipe_optim(G.ssl_recipe(G.apply_genes(g, {"ssl": "vicreg"})), 256, 10, 100)
    assert v["optimizer"] == "lars" and v["lr"] == pytest.approx(0.2) and v["lr_schedule"]["warmup_steps"] == 100
    l_ = G.recipe_optim(G.ssl_recipe(G.apply_genes(g, {"ssl": "lejepa"})), 512, 10, 100)
    assert l_["optimizer"] == "adamw" and l_["lr"] == pytest.approx(5e-4) and l_["lr_schedule"]["warmup_steps"] == 10
    assert l_["lr_schedule"]["start_ratio"] == 0.01 and not l_["wd_exclude_1d"]   # AdamW oficial: wd em tudo


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


@pytest.mark.parametrize("ssl,optim", [("vicreg", "lars"), ("simclr", "lars"), ("lejepa", "adamw")])
def test_schedule_resume_equals_continuous(ssl, optim):
    kw = dict(optimizer=optim, lr_schedule={"total_steps": 12, "warmup_steps": 3, "end_ratio": 1e-3},
              weight_decay=1e-4, wd_exclude_1d=True, vicreg_kwargs={"expander_dims": [16, 16, 16]},
              reg_kwargs={"n_proj": 16}, n_views=2)
    if ssl != "vicreg":
        kw.pop("vicreg_kwargs")
    if ssl != "lejepa":
        kw.pop("reg_kwargs")
    reg = "sigreg" if ssl == "lejepa" else "none"
    e1, h1 = _toy()
    T.train_global_ssl(e1, h1, _loader(), ssl, reg, 3, 0.1, 0.1, [], "cpu", None, 0, 0.05, **kw)
    e2, h2 = _toy()
    ld = _loader()
    st = T.train_global_ssl(e2, h2, ld, ssl, reg, 1, 0.1, 0.1, [], "cpu", None, 0, 0.05, return_state=True, **kw)["state"]
    T.train_global_ssl(e2, h2, ld, ssl, reg, 2, 0.1, 0.1, [], "cpu", None, 0, 0.05, resume=st, **kw)
    for a, b in zip(list(e1.state_dict().values()) + list(h1.state_dict().values()),
                    list(e2.state_dict().values()) + list(h2.state_dict().values())):
        assert torch.equal(a, b)


def test_lr_is_scheduled_per_step():
    e, h = _toy()
    seen = []
    orig = T.LARS.step

    def spy(self, closure=None):
        seen.append(self.param_groups[0]["lr"])
        return orig(self, closure)
    T.LARS.step = spy
    try:
        T.train_global_ssl(e, h, _loader(), "vicreg", "none", 2, 0.2, 0.2, [], "cpu", None, 0, 0.05,
                           optimizer="lars", lr_schedule={"total_steps": 8, "warmup_steps": 2, "end_ratio": 1e-3},
                           vicreg_kwargs={"expander_dims": [16, 16, 16]})
    finally:
        T.LARS.step = orig
    assert seen[0] == 0.0 and seen[2] == pytest.approx(0.2) and seen[-1] < 0.2


# ───────────────────────────────────────────────────────────────── e2e mini (flowers, CPU): receita fiel + OOM
FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_B0 = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_B0.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v31_recipe_batch_flowers(tmp_path, monkeypatch):
    from ga_search.hmodehb import fitness_v2 as FV
    calls = {"n": 0}
    real = T.train_global_ssl

    def flaky(enc, head, loader, *a, **k):                    # 1ª tentativa com batch > 16: OOM simulado
        calls["n"] += 1
        if getattr(loader, "batch_size", 0) > 16 and k.get("resume") is None and calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 9.00 GiB")
        return real(enc, head, loader, *a, **k)
    monkeypatch.setattr(T, "train_global_ssl", flaky)
    out = tmp_path / "fl"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=1"]
    cfg = G.build_cfg("flowers", 42, ov, out)
    g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml",
                             ["seeds.top_k=1", "n_parents=1", "n_children=3", "generations=1", "fid_low=1",
                              "fid_high=2", "eta=1", "end_band=B1", "spifil_images_per_class=1", "debug_train=48",
                              "debug_val=32", "test_channel=false", "seeds.external_small=false",
                              f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}",
                              "v3.ssl_recipe=paper", "v3.ssl_batch_size=32", "v3.ssl_batch_min=8",
                              "v3.so2.seed_mode=fixed", "v3.recipe_small_views=true"])
    g["v3"]["genes"] = {"ssl": ["simclr", "vicreg", "lejepa"]}
    g["v3"]["vicreg_kwargs"] = {"expander_dims": [64, 64, 64]}
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
        b = r["ssl_recipe"]["batch"]
        seen.add(b["optim"]["optimizer"])
        assert b["epoch"].startswith("passada completa") and b["optim"]["lr_schedule"]["total_steps"] >= 1
        if b.get("oom_halvings"):
            assert b["used"] == 16 and any("por memória" in a for a in r["ssl_recipe"]["adaptacao"])
    assert seen == {"lars", "adamw"}
    assert any(r["ssl_recipe"]["batch"].get("oom_halvings") for r in res)
    hi = [json.loads(p.read_text()) for p in out.glob("nodes/*/f2/result.json")]
    for r in hi:                                               # retomada f1 -> f2 com o mesmo batch
        assert r["status"] == "ok", r.get("error")
        if r.get("ssl_recipe"):
            assert r["ssl_recipe"]["batch"]["used"] in (16, 32)
