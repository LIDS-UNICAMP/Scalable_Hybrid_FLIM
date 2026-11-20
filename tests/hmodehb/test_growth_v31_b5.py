"""growth v3.1 bloco 5: ReDo (tau-dormente) e RePr (O_f, p=30%) no replace_bad_filters; poda EXATA dos canais
novos redundantes (mais superpixels -> redundância eliminada dentro da rede)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones import morphisms_v3 as MV3  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from test_growth_v31_covgate import _tiny, _x  # noqa: E402


def test_redo_scores_and_dormant():
    a = torch.rand(4, 5, 3, 3) + 0.5
    a[:, 2] *= 0.01                                   # quase dormente
    s = MV3.redo_scores(a)
    assert s.mean().item() == pytest.approx(1.0)
    assert MV3.redo_dormant(a, 0.1) == [2] and MV3.redo_dormant(a, 0.0) == []


def test_repr_overlap_and_select():
    torch.manual_seed(0)
    W = torch.randn(10, 4, 3, 3)
    W[3] = W[1] * 2.0                                 # duplicado (cos = 1)
    O = MV3.repr_overlap(W)
    sel = MV3.repr_select(W, 0.3)
    assert len(sel) == 3 and 1 in sel and 3 in sel
    assert sel == MV3.repr_select(W, 0.3)             # determinístico


def test_prune_new_channels_exact_and_refuses_inherited():
    m = _tiny()
    r = MO.widen(MO.start(m), {1: 32})
    new = [c for c, p in enumerate(r.out_maps[1]) if p < 0]
    r2 = MV3.prune_new_channels(r, 1, new[:5])
    x = _x()
    with torch.no_grad():
        assert torch.allclose(r2.child.eval()(x), m(x), atol=1e-6, rtol=0)
    assert int(r2.child.spec.layers()[1].channels) == 27
    gs = MO.finalize(r2, x, data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
    with pytest.raises(MO.MorphismError):
        MV3.prune_new_channels(r, 1, [0])                # herdado: nunca


def test_prune_redundant_new_in_chain_keeps_band():
    m = _tiny()
    r = MO.widen(MO.start(m), {1: 60})
    blk = r.child.blocks[1]
    new = [c for c, p in enumerate(r.out_maps[1]) if p < 0]
    with torch.no_grad():                             # metade dos canais novos = cópias exatas de um herdado
        for c in new[: len(new) // 2]:
            blk.conv.weight[c] = blk.conv.weight[0]
            blk.conv.bias[c] = blk.conv.bias[0]
            blk.norm.weight[c], blk.norm.bias[c] = blk.norm.weight[0], blk.norm.bias[0]
            blk.norm.running_mean[c], blk.norm.running_var[c] = blk.norm.running_mean[0], blk.norm.running_var[0]
    gev = SimpleNamespace(ev=SimpleNamespace(svm_train_x=_x(12, seed=7)), device=torch.device("cpu"))
    band0 = G.band_of(G._params(r))
    r2, log = G.prune_redundant_new(r, gev, {"prune": {"n_images": 12}}, {})
    assert log and log[0]["n_pruned"] > 0
    assert G.band_of(G._params(r2)) == band0
    gs = MO.finalize(r2, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
