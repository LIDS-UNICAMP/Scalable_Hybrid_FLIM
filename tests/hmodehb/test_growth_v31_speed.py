"""growth v3.1 (velocidade do AE): sem checkpoint no AE local o resultado é bit a bit igual (pesos e state);
ElementProtect funciona também com o LARS (hooks globais do otimizador)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb import ssl_trainers as T  # noqa: E402
from ga_search.modehb.external_seed import ElementProtect  # noqa: E402
from ga_search.modehb.local_ae import train_local_ae  # noqa: E402
from ga_search.modehb.run_search import _NullRun  # noqa: E402
from test_growth_v31_covgate import _tiny, _x  # noqa: E402


def test_ae_without_checkpoint_is_bitwise_equal():
    m = _tiny()
    res = MO.widen(MO.start(m), {1: 30})
    loader = [(_x(8, seed=s),) for s in range(3)]
    outs = []
    for ck in (True, False):
        child = copy.deepcopy(res.child).train()
        prefix = nn.Sequential(*child.blocks[:1]).eval()
        prefix.out_channels = 12
        blk = child.blocks[1]
        torch.manual_seed(0)
        with ElementProtect(blk, G._masks_under(res.masks, "blocks.1."), 0.01):
            train_local_ae(prefix, G._CkptSequential(blk) if ck else blk, loader, "pixel", 1, 1e-3,
                           torch.device("cpu"), _NullRun(), 0)
        outs.append({k: v.clone() for k, v in child.state_dict().items()})
    assert all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0])


def test_element_protect_with_lars():
    w = nn.Parameter(torch.randn(4, 3))
    opt = T.LARS([w], lr=0.5, weight_decay=0.0, weight_decay_filter=T.exclude_bias_and_norm,
                 lars_adaptation_filter=T.exclude_bias_and_norm)
    mod = nn.Module()
    mod.w = w
    mask = torch.zeros(4, 3, dtype=torch.bool)
    mask[0] = True                                      # linha 0 herdada (congelada com scale 0)
    w0 = w.detach().clone()
    with ElementProtect(mod, {"w": mask}, 0.0):
        w.grad = torch.ones_like(w)
        opt.step()
    assert torch.equal(w[0], w0[0]) and not torch.equal(w[1], w0[1])
