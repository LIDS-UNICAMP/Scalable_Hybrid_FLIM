"""growth v3.1 bloco 4: eps relativo na so2, ZCA re-ajustada depois do AE/SSL, SoftHebb com tau/lr por camada,
BN sem treino e raio R (norma SPiFiL devolvida)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones import morphisms_v3 as MV3  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.hmodehb import so2_head as SO2  # noqa: E402
from test_growth_v31_covgate import _rand_filters, _tiny, _x  # noqa: E402


def test_so2_eps_rel():
    g = torch.Generator().manual_seed(0)
    tr = [torch.randn(4, 30, 6, 6, generator=g) for _ in range(3)]
    va = torch.randn(3, 30, 6, 6, generator=g)
    heads = []
    for eps in (0.0, 1e-3):
        h = SO2.So2Head(4096, 64, 0, eps)
        for i, f in enumerate(tr):
            h.add_train(f, list(range(4 * i, 4 * i + 4)))
        heads.append(h.finish_fit())
    a, b = heads[0](va), heads[1](va)
    assert torch.isfinite(b).all() and not torch.equal(a, b)
    assert heads[1].meta()["eps_rel"] == 1e-3 and "eps_rel" not in heads[0].meta()
    assert torch.equal(SO2.So2Head.from_payload(heads[1].payload())(va), b)


def _gev(n=12, res=32):
    return SimpleNamespace(ev=SimpleNamespace(svm_train_x=_x(n, res, seed=4)), device=torch.device("cpu"))


def test_zca_refit_after_training():
    m = _tiny()
    out = MV3.add_gated_residual(MO.start(m), 2, norm="zca", new_filters=_rand_filters, fit_x=_x(8, seed=3))
    blk = out.child.blocks[2]
    W0 = blk.norm.W.clone()
    with torch.no_grad():
        blk.conv.weight.mul_(2.0)                       # "treino" mudou a conv
    log = G._zca_refit(_gev(), out.child, [2], {"v3": {"zca_refit_images": 12}})
    assert log and log[0]["block"] == 2 and log[0]["split"] == "train" and log[0]["W_rel_change"] > 0
    assert not torch.equal(blk.norm.W, W0)


def test_softhebb_paper_keeps_spifil_norm_and_layer_params():
    m = _tiny()
    res = G.realize_plan(m, [{"op": "widen", "widths": {"1": 30}, "seed": 1}], None, {})
    conv = res.child.blocks[1].conv
    key = "blocks.1.conv.weight"
    rows = [r for r in range(conv.weight.shape[0]) if not bool(res.masks[key][r].any())]
    with torch.no_grad():
        conv.weight[rows] = torch.randn(conv.weight[rows].shape, generator=torch.Generator().manual_seed(5)) * 0.5
    n0 = conv.weight[rows].flatten(1).norm(dim=1).clone()
    W0 = conv.weight[rows].clone()
    log = G.softhebb_new_filters(res, _gev(), {"paper": True, "radius": 3.0, "n_images": 12, "epochs": 1, "batch": 4})
    rec = [r for r in log if r["param"] == key]
    assert rec and rec[0]["status"] == "ok" and rec[0]["paper"] and rec[0]["temperature"] == 1 / 0.65
    assert rec[0]["lr"] == 0.005
    n1 = conv.weight[rows].flatten(1).norm(dim=1)
    assert torch.allclose(n0, n1, rtol=1e-5)                     # norma SPiFiL devolvida
    assert not torch.allclose(W0, conv.weight[rows])             # a direção mudou
    gs = MO.finalize(res, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"                    # colunas do consumidor seguem 0
