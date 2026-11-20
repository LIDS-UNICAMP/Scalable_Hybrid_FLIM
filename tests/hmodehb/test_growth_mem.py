"""growth_mem: as reduções de memória da cadeia de crescimento são bit a bit iguais (CPU) e reduzem o pico.

Proxy de pico em CPU: bytes de tensores salvos pelo autograd (saved_tensors_hooks) -- é o que domina a GPU
no SSL (V*B = 256 imagens 224 px) -- e tracemalloc não vê o alocador do torch.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb.ops_norm_act import DecorrelatedBatchNorm  # noqa: E402


def _rs_like(widths=(24, 24), res=32, seed=0):
    """Forma do B0 de RS (c00306): conv3+BN+max, dwsep7+batch_dbn+avg+identity."""
    L = [LayerSpec(channels=widths[0], kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=widths[1], kernel=7, conv="dwsep", norm="batch_dbn", activation="silu", pool="avg",
                   residual="identity", flavor="spifil_ops", init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


def _dbn_ref_eval(m: DecorrelatedBatchNorm, x):
    """Implementação ANTERIOR (eval) copiada literalmente."""
    N, C, H, W = x.shape
    flat = x.permute(1, 0, 2, 3).reshape(C, -1)
    xc = flat - m.running_mean[:, None].to(x.dtype)
    out = torch.cat([getattr(m, f"running_W{i}").to(x.dtype) @ xc[a:b] for i, (a, b) in enumerate(m.bounds)], 0)
    out = out.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return out * m.weight.view(1, -1, 1, 1) + m.bias.view(1, -1, 1, 1)


@pytest.mark.parametrize("C", [16, 37, 210])
def test_dbn_eval_bitwise(C):
    torch.manual_seed(C)
    m = DecorrelatedBatchNorm(C)
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(4, C, 9, 9) * 3 + 1)
        m.weight.uniform_(0.5, 2), m.bias.normal_()
    m.eval()
    x = torch.randn(5, C, 11, 11)
    with torch.no_grad():
        assert torch.equal(m(x), _dbn_ref_eval(m, x))
    xg = x.clone().requires_grad_(True)               # com autograd: caminho antigo, idem
    assert torch.equal(m(xg).detach(), _dbn_ref_eval(m, x))


def _ssl_step(model, ckpt: bool, x, seed=0):
    torch.manual_seed(seed)
    enc = (G._CkptSequential if ckpt else nn.Sequential)(*model.blocks)
    enc.train(), model.head.train()
    saved = [0, 0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        saved[1] = max(saved[1], saved[0])
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = model.head(enc(x))
    loss = h.pow(2).mean() + h.std(0).mean()
    loss.backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    return h.detach(), grads, {k: v.clone() for k, v in model.state_dict().items()}, saved[1]


def test_grad_checkpoint_bitwise_and_smaller():
    base = _rs_like()
    x = torch.rand(16, 3, 32, 32)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    ha, ga, sa, peak_a = _ssl_step(a, False, x)
    hb, gb, sb, peak_b = _ssl_step(b, True, x)
    assert torch.equal(ha, hb)
    assert ga.keys() == gb.keys() and all(torch.equal(ga[k], gb[k]) for k in ga)
    # running stats (BN e DBN running_W/mean, num_batches_tracked) atualizadas UMA vez, igual ao sem checkpoint
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), \
        [k for k in sa if not torch.equal(sa[k], sb[k])]
    print(f"\n[growth_mem] saved-for-backward: sem ckpt {peak_a / 1e6:.2f} MB, com ckpt {peak_b / 1e6:.2f} MB")
    assert peak_b < 0.5 * peak_a


def test_ckpt_names_unchanged():
    m = _rs_like()
    assert [k for k, _ in nn.Sequential(*m.blocks).named_parameters()] == \
        [k for k, _ in G._CkptSequential(*m.blocks).named_parameters()]


def test_blend_cpu_anchor_bitwise():
    m = _rs_like()
    anchor = {k: v.detach().clone() for k, v in m.state_dict().items()}
    trained = {k: (v + 0.1 * torch.randn_like(v)) if v.is_floating_point() else v for k, v in anchor.items()}
    cpu = {k: v.to("cpu", copy=True) for k, v in anchor.items()}
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        m1, m2 = copy.deepcopy(m), copy.deepcopy(m)
        G._blend_into(m1, anchor, trained, a)
        G._blend_into(m2, cpu, {k: v.cpu() for k, v in trained.items()}, a)
        ref = {k: ((1 - a) * anchor[k] + a * trained[k]) if trained[k].is_floating_point() else trained[k]
               for k in trained} if a else anchor
        for k, v in m1.state_dict().items():
            assert torch.equal(v, m2.state_dict()[k]) and torch.equal(v, ref[k])


def test_embed_bitwise_eval():
    """Forward de avaliação inteiro (com DBN novo) == implementação antiga do DBN, no B0-like."""
    m = _rs_like()
    x = torch.rand(6, 3, 32, 32)
    with torch.no_grad():
        new = m(x)
        orig = DecorrelatedBatchNorm.forward
        try:
            DecorrelatedBatchNorm.forward = lambda self, z: _dbn_ref_eval(self, z) if not self.training else orig(self, z)
            old = m(x)
        finally:
            DecorrelatedBatchNorm.forward = orig
    assert torch.equal(new, old)


def test_mem_defaults_declared():
    mem = G._mem({})
    assert mem["grad_checkpoint"] and mem["wise_on_cpu"] and mem["empty_cache"]
    assert mem["embed_batch_size"] is None and mem["ssl_amp_bf16"] is False     # mudam a ciência: desligados


@pytest.mark.parametrize("chunk", [1, 3, 10**9])
def test_actpool_fused_bitwise(chunk, monkeypatch):
    """growth_mem3: act+skip+pool em micro-lotes (_ActPool) == forward/backward original, bit a bit."""
    monkeypatch.setattr(G, "ACTPOOL_CHUNK", chunk)
    base = _rs_like()
    assert all(G._fusable(b) for b in base.blocks)
    x = torch.rand(7, 3, 32, 32)
    ha, ga, sa, _ = _ssl_step(copy.deepcopy(base), False, x)
    hb, gb, sb, _ = _ssl_step(copy.deepcopy(base), True, x)
    assert torch.equal(ha, hb)
    assert all(torch.equal(ga[k], gb[k]) for k in ga), [k for k in ga if not torch.equal(ga[k], gb[k])]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
