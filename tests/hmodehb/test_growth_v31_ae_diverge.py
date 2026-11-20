"""growth_v31_hotfix_ae_diverge: AE local divergindo nos blocos alargados.

- desligado (sem kwargs novos) = chamada histórica: bit a bit;
- a guarda (diverge_ratio) só observa: sem divergência, pesos bit a bit iguais aos sem guarda;
- dec_in_norm corrige o caso "z grande x fan-in grande" (a perda cai);
- train_child: bloco divergente volta bit a bit ao morfismo (pré-AE), "ae_diverged" registrado, o resto segue;
- smoke estrito: recon_ratio > 1 -> ChainError; fora do smoke só alerta.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb.local_ae import train_local_ae  # noqa: E402

SYN = ["data.synthetic=true", "data.image_size=32", "data.synthetic_train_samples=16", "data.synthetic_val_samples=12",
       "data.batch_size=4", "data.num_workers=0", "data.ram_cache=true", "train.max_steps_per_epoch=2",
       "search.n_classes=2", "search.spifil_images_choices=[1,2]", "device=cpu", "wandb.enabled=false"]
OPS = [{"op": "widen", "widths": {"0": 16}, "seed": 1}]


def _loader(n=32, hw=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, hw, hw, generator=g)
    return DataLoader(TensorDataset(x), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))


def _wide_block(C=256, gain=40.0, seed=0):
    """Bloco 'alargado' com saída grande (RMS ~ dezenas) e muitos canais: o caso que diverge."""
    torch.manual_seed(seed)
    blk = nn.Sequential(nn.Conv2d(3, C, 3, padding=1), nn.ReLU())
    with torch.no_grad():
        blk[0].weight.mul_(gain)
    return blk


def _sd(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _run(blk, **kw):
    blk = copy.deepcopy(blk)
    out = train_local_ae(nn.Identity(), blk, _loader(), "pixel", 2, 1e-3, "cpu", None, 3, **kw)
    return blk, out


def test_defaults_and_guard_observe_only_bitwise():
    blk = _wide_block(C=16, gain=1.0)
    a, oa = _run(blk)
    b, ob = _run(blk, dec_in_norm=False, grad_clip=None, lr_scale_fanin=False, warmup_steps=0)
    c, oc = _run(blk, diverge_ratio=1e9)                    # guarda ligada, nunca dispara
    for m in (b, c):
        for k, v in _sd(a).items():
            assert torch.equal(v, _sd(m)[k]), k
    assert oa["final_recon_loss"] == ob["final_recon_loss"] == oc["final_recon_loss"]
    assert "diverged" not in oa and oc["diverged"] is False
    assert G.ae_extra_kw({}) == {}                           # sem chaves -> chamada histórica


def test_guard_verdict_ratio_and_trend_not_absolute():
    from ga_search.modehb.local_ae import ae_guard_verdict as V
    # athena superpixels_prune: começa alto e cai (escala) -> ok
    assert V([10.85, 8.0, 6.1, 5.0], None, 10.85, 1.0, False)[0] is False
    assert V([4.1, 3.6, 3.4, 3.28], 7.5, 10.85, 1.0, True)[0] is False
    # widen RS 0,626 -> 1,846 e tiny 86,4 -> 369,5 (sobe na época) -> diverge
    assert V([0.626, 2.9, 2.1, 1.9, 1.846], None, 0.626, 1.0, True)[0] is True
    assert V([86.4, 150.0, 250.0, 369.5], None, 86.4, 1.0, True)[0] is True
    # época 2 mais alta que a 1 e acima do inicial -> diverge antes do fim
    assert V([3.0, 3.2, 3.4], 2.0, 1.0, 1.0, False)[0] is True
    assert V([float("nan")], None, 1.0, 1.0, False)[0] is True


def test_wide_block_diverges_and_dec_in_norm_fixes():
    blk = _wide_block()
    _, bad = _run(blk, diverge_ratio=1.0)
    _, good = _run(blk, diverge_ratio=1.0, dec_in_norm=True)
    assert bad["recon_ratio"] > 1.0 and bad["diverged"] is True
    assert good["recon_ratio"] < 1.0 and good["diverged"] is False and good["dec_in_norm"] is True
    _, both = _run(blk, diverge_ratio=1.0, dec_in_norm=True, lr_scale_fanin=True)     # padrão do growth.yaml
    assert both["recon_ratio"] < 1.0 and both["diverged"] is False
    _, clip = _run(blk, diverge_ratio=1.0, dec_in_norm=True, grad_clip=1.0)
    assert clip["recon_ratio"] < 1.0 and clip["grad_clip"] == 1.0


def _tiny(seed=0, widths=(12, 24), res=32):
    L = [LayerSpec(channels=widths[0], kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=widths[1], kernel=3, conv="conv", norm="batch", activation="silu", pool="max",
                   flavor="spifil_ops", init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


@pytest.fixture(scope="module")
def gev(tmp_path_factory):
    out = tmp_path_factory.mktemp("gev_aed")
    cfg = G.build_cfg("remote_sensing", 42, SYN, out)
    return G.GrowthEvaluator(cfg, out / "eval")


def _g(**kw):
    return {**G.GROWTH_DEFAULTS, "wise_alphas": [0.0, 1.0], "ssl": "none", "fid_high": 15, **kw}


def test_train_child_guard_restores_block_bitwise(gev):
    m = _tiny()
    res = G.realize_plan(copy.deepcopy(m), OPS, None)
    pre = _sd(res.child.blocks[0])
    # lr absurdo -> o AE do bloco 0 diverge -> o bloco volta bit a bit ao morfismo
    cap = {}
    model, out = G.train_child(gev, res, 2, _g(lr_ae=50.0, ae_dec_in_norm=False), 7, 0.5, capture=cap)
    r0 = next(r for r in out["ae_logs"] if r["block"] == 0)
    assert r0["ae_diverged"] is True and r0["ae_restored"] is True
    assert 0 in out["ae_diverged_blocks"] and out["ae_alerts"]
    post = {k[len("blocks.0."):]: v for k, v in cap["trained"].items() if k.startswith("blocks.0.")}   # pré-WiSE
    assert set(post) == set(pre) and all(torch.equal(pre[k], post[k]) for k in pre)
    # mesma semente, guarda desligada: o bloco muda (o AE treinou) -> a guarda é quem restaurou
    res2 = G.realize_plan(copy.deepcopy(m), OPS, None)
    cap2 = {}
    model2, out2 = G.train_child(gev, res2, 2, _g(lr_ae=50.0, ae_dec_in_norm=False, ae_diverge_guard=None), 7, 0.5,
                                 capture=cap2)
    assert "ae_diverged" not in out2["ae_logs"][0] and not out2["ae_diverged_blocks"]
    post2 = {k[len("blocks.0."):]: v for k, v in cap2["trained"].items() if k.startswith("blocks.0.")}
    assert any(not torch.equal(pre[k], post2[k]) for k in pre if pre[k].is_floating_point())


def test_train_child_off_equals_historic_bitwise(gev):
    """Opções desligadas (= histórico) e guarda ligada sem disparar dão o MESMO filho bit a bit."""
    m = _tiny()
    off = dict(ae_dec_in_norm=False, ae_diverge_guard=None, ae_grad_clip=None)
    a, oa = G.train_child(gev, G.realize_plan(copy.deepcopy(m), OPS, None), 2, _g(**off), 7, 0.5)
    b, ob = G.train_child(gev, G.realize_plan(copy.deepcopy(m), OPS, None), 2,
                          _g(**{**off, "ae_diverge_guard": 1e9}), 7, 0.5)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert oa["kappa_val_orig"] == ob["kappa_val_orig"]
    assert ob["ae_logs"][0]["ae_diverged"] is False


def test_smoke_strict_fails_on_ratio_gt1(gev):
    m = _tiny()
    g = _g(lr_ae=50.0, ae_dec_in_norm=False, ae_smoke_strict=True)   # smoke: --gset ae_smoke_strict=true
    assert not G.ae_smoke_strict(_g()) and not G.ae_smoke_strict(_g(fid_high=1))      # padrão: só alerta
    assert G.ae_smoke_strict(_g(ae_smoke_strict=None, fid_high=1))                     # null = automático
    with pytest.raises(G.ChainError, match="recon_ratio > 1"):
        G.train_child(gev, G.realize_plan(copy.deepcopy(m), OPS, None), 2, g, 7, 0.5)
