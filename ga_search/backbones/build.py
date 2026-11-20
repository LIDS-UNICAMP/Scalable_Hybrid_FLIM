"""Construtor: ``BackboneSpec`` -> ``nn.Module`` autocontido (encoder + cabeça >= 1280) + custo medido.

Reaproveita os operadores do repo sem modificá-los:
* cada camada é um ``ga_search.modehb.ops_norm_act.ConvBlock`` (conv -> norm -> act (+ skip) -> pool);
  ``inception`` usa ``InceptionConv`` com a partição explícita do spec; residual ``project`` força um
  1x1 sem bias mesmo quando cin == cout; residual ``gated`` = ``GatedSkip`` (y + lambda*x, lambda=0 no
  morfismo ``add_skip_zero``).
* a cabeça é ``ga_search.modehb.pooling_head.EmbeddingHead`` (1x1 -> pooling -> ``embed_dim``); a
  variante ``lowrank`` antepõe um 1x1 C -> rank sem bias (``LowRankHead``), mantendo ``embed_dim``.

Custo: params do encoder e da cabeça contados no módulo construído (decoders/projetores à parte, em
``extras_not_counted``, nunca somados); MACs = termos de kernel de Conv2d/Linear (mesma metodologia de
``run_search.inference_macs``), FLOPs = 2 x MACs, explícito no spec.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ga_search.backbones.spec import (BUDGETS, HARD_CAP, BackboneSpec, CostSpec, HeadSpec, LayerSpec,
                                      SpecError, band_of)
from ga_search.modehb.ops_norm_act import ConvBlock, InceptionConv
from ga_search.modehb.pooling_head import EmbeddingHead


class BudgetError(SpecError):
    pass


class GatedSkip(nn.Module):
    """skip = lambda * x, lambda escalar aprendível (residual de contribuição zero numa camada existente)."""

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return self.gate * x


class LowRankHead(nn.Module):
    """1x1 C -> rank (sem bias) -> EmbeddingHead(rank, embed_dim, pooling). Saída = embed_dim (>= 1280)."""

    def __init__(self, in_ch: int, rank: int, embed_dim: int, pooling: str, **kw):
        super().__init__()
        self.reduce = nn.Conv2d(int(in_ch), int(rank), 1, bias=False)
        self.head = EmbeddingHead(int(rank), int(embed_dim), pooling, **kw)
        self.in_ch, self.out_dim, self.nominal_dim = int(in_ch), self.head.out_dim, self.head.nominal_dim

    def forward(self, x):
        return self.head(self.reduce(x))


# ─────────────────────────────────────────────────────────────────── growth v3: dense_out (DenseNet)
from ga_search.modehb.spifil_ops import SpifilOpsBlock as _SpifilOpsBlock  # noqa: E402


def _dense_cat(y, x):
    return torch.cat([y, F.adaptive_avg_pool2d(x, tuple(y.shape[-2:])).to(y.dtype)], 1)


class DenseOutSpifilOpsBlock(_SpifilOpsBlock):
    """``SpifilOpsBlock`` com saída cat(bloco(x), adaptive_avg_pool(x)) (mesmos nomes de parâmetros)."""

    def forward(self, x):
        return _dense_cat(super().forward(x), x)


class DenseOutConvBlock(ConvBlock):
    """``ConvBlock`` com saída cat(bloco(x), adaptive_avg_pool(x)) (mesmos nomes de parâmetros)."""

    def forward(self, x):
        return _dense_cat(super().forward(x), x)


class CovGatedSpifilOpsBlock(_SpifilOpsBlock):
    """growth v3.1: y = x + g * bloco(x) (ramo de covariância gated, g = 0 no início -> função preservada)."""
    is_cov_gated = True

    def forward(self, x):
        b = super().forward(x)
        return x + self.gate.view(1, -1, 1, 1).to(b.dtype) * b


class CovGatedConvBlock(ConvBlock):
    is_cov_gated = True

    def forward(self, x):
        b = super().forward(x)
        return x + self.gate.view(1, -1, 1, 1).to(b.dtype) * b


def _as_cov_gated(blk: nn.Module, cout: int, mode: str) -> nn.Module:
    blk.__class__ = CovGatedSpifilOpsBlock if isinstance(blk, _SpifilOpsBlock) else CovGatedConvBlock
    blk.gate = nn.Parameter(torch.zeros(int(cout) if mode == "channel" else 1))
    blk.cov_gate_mode = str(mode)
    return blk


def cov_gate_values(model: nn.Module) -> dict:
    """{bloco: valores de g} dos blocos cov_gate (diagnóstico para o result.json)."""
    out = {}
    for j, b in enumerate(getattr(model, "blocks", [])):
        if getattr(b, "is_cov_gated", False):
            g = b.gate.detach().float().cpu()
            out[str(j)] = {"mode": getattr(b, "cov_gate_mode", "?"), "mean": float(g.mean()),
                           "mean_abs": float(g.abs().mean()), "max_abs": float(g.abs().max()),
                           "values": [round(float(v), 6) for v in g[:64]]}
    return out


def block_in_channels(spec: BackboneSpec, j: int) -> int:
    return spec.block_in_channels(j)


def block_out_channels(spec: BackboneSpec, j: int) -> int:
    return spec.block_out_channels(j)


def _as_dense(blk: nn.Module, cin: int, cout: int) -> nn.Module:
    blk.__class__ = DenseOutSpifilOpsBlock if isinstance(blk, _SpifilOpsBlock) else DenseOutConvBlock
    blk.dense_out, blk.conv_out_channels = True, int(cout)
    blk.out_channels = int(cout) + int(cin)
    return blk


def make_spifil_block(L: LayerSpec, cin: int) -> nn.Module:
    """Bloco ``flavor=spifil_ops``: exatamente o ``SpifilOpsBlock`` dos encoders de produção (mesmos nomes
    de parâmetros, bias em toda conv, pool da biblioteca), com os residuais do A3 (gated/project)."""
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    ops = {"builder": "spifil_ops", "norm_in_block": True, "conv": L.conv, "kernel": int(L.kernel),
           "in_channels": int(cin), "channels": int(L.channels), "requested_channels": int(L.channels),
           "dw_multiplier": None, "activation": L.activation, "norm": L.norm, "pool": L.pool,
           "pool_stride": 2 if L.pool != "none" else 1, "residual": L.residual != "none",
           "skip": {"identity": "identity", "project": "project"}.get(L.residual), "dropout": 0.0}
    if L.conv == "inception":
        ops["inception_branches"] = [list(b) for b in L.branches]
    if L.pool_padding is not None and L.pool != "none":
        ops.update(pool_size=int(L.pool_size), pool_padding=int(L.pool_padding))
    blk = SpifilOpsBlock(ops, int(cin))
    blk.residual_mode = {"none": "off"}.get(L.residual, L.residual)
    if L.residual == "gated":
        blk.skip = GatedSkip(0.0)
    return blk


def make_block(L: LayerSpec, cin: int) -> ConvBlock:
    blk = _make_block(L, cin)
    if getattr(L, "dense_out", False):
        blk = _as_dense(blk, cin, int(L.channels))
    if getattr(L, "cov_gate", "none") != "none":
        blk = _as_cov_gated(blk, int(L.channels), L.cov_gate)
    return blk


def _make_block(L: LayerSpec, cin: int) -> ConvBlock:
    if L.flavor == "spifil_ops":
        return make_spifil_block(L, cin)
    d = {"channels": int(L.channels), "kernel": int(L.kernel), "conv": L.conv, "norm": L.norm,
         "activation": L.activation, "pool": L.pool, "pool_size": int(L.pool_size),
         "dbn_group_size": int(L.dbn_group_size), "residual": L.residual == "identity",
         "residual_mismatch": "disable"}
    blk = ConvBlock(d, int(cin))
    if L.conv == "inception":
        blk.conv = InceptionConv(int(cin), int(L.channels), 1, bias=L.norm == "none",
                                 branches=[list(b) for b in L.branches])
    if L.residual == "identity" and blk.residual_mode != "identity":
        raise SpecError("residual identity não montado (cin != cout?)")
    if L.residual == "project":
        blk.skip, blk.residual_mode = nn.Conv2d(int(cin), int(L.channels), 1, bias=False), "project"
    elif L.residual == "gated":
        blk.skip, blk.residual_mode = GatedSkip(0.0), "gated"
    return blk


def make_head(H: HeadSpec, cin: int) -> nn.Module:
    kw = dict(kurtosis=bool(H.kurtosis), act=H.act, bn=bool(H.bn))
    if H.proj == "lowrank":
        return LowRankHead(cin, int(H.rank), int(H.embed_dim), H.pooling, **kw)
    return EmbeddingHead(int(cin), int(H.embed_dim), H.pooling, **kw)


class Backbone(nn.Module):
    """Encoder (``blocks``: stem + estágios, planos em ordem) + ``head`` (embedding >= 1280)."""

    def __init__(self, spec: BackboneSpec):
        super().__init__()
        spec = spec.copy().normalize()
        spec.validate()
        self.spec = spec
        cin = int(spec.input.in_channels)
        self.blocks = nn.ModuleList()
        for L in spec.layers():
            self.blocks.append(make_block(L, cin))
            cin = int(L.channels) + (cin if getattr(L, "dense_out", False) else 0)
        self.head = make_head(spec.head, cin)
        self.embed_dim = int(spec.head.embed_dim)

    def features(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, x):
        return self.head(self.features(x))

    def forward_all(self, x) -> Tuple[List[torch.Tensor], torch.Tensor]:
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return outs, self.head(x)


# ─────────────────────────────────────────────────────────────────── custo
def count_params(model: Backbone) -> Tuple[int, int]:
    enc = sum(p.numel() for p in model.blocks.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return int(enc), int(head)


def count_macs(model: Backbone, resolution: int) -> Tuple[int, int]:
    """(macs_encoder, macs_head) por imagem; hooks em Conv2d/Linear (``run_search.inference_macs``)."""
    acc = {"enc": 0, "head": 0}
    head_mods = {id(m) for m in model.head.modules()}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            n = out.numel() // out.shape[0] * mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
        else:
            n = out.numel() // out.shape[0] * mod.in_features
        acc["head" if id(mod) in head_mods else "enc"] += int(n)

    def run(m, dev):
        hs = [x.register_forward_hook(hook) for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
        try:
            was = m.training
            m.eval()
            with torch.no_grad():
                m(torch.zeros(1, int(m.spec.input.in_channels), resolution, resolution, device=dev))
            m.train(was)
        finally:
            for h in hs:
                h.remove()

    try:                                   # meta: sem alocar memória (ZCA não roda em meta)
        meta = copy.deepcopy(model).to("meta")
        head_mods = {id(m) for m in meta.head.modules()}
        run(meta, "meta")
    except Exception:
        acc = {"enc": 0, "head": 0}
        head_mods = {id(m) for m in model.head.modules()}
        run(model, next(model.parameters()).device)
    return acc["enc"], acc["head"]


def measure_cost(model: Backbone, resolution: Optional[int] = None) -> CostSpec:
    res = int(resolution or model.spec.input.resolution)
    pe, ph = count_params(model)
    me, mh = count_macs(model, res)
    return CostSpec(params_encoder=pe, params_head=ph, params_total=pe + ph,
                    params_extras=dict(model.spec.extras_not_counted), macs_encoder=me, macs_head=mh,
                    macs=me + mh, flops=2 * (me + mh), resolution=res, measured=True)


def check_budget(spec: BackboneSpec) -> None:
    """Teto duro 5,5M + teto da classe declarada (sem tolerância, sem reparo silencioso)."""
    n = int(spec.cost.params_total)
    if n > HARD_CAP:
        raise BudgetError(f"{spec.hash12}: params_total={n:,} > teto duro {HARD_CAP:,}")
    if spec.budget_class is not None and n > BUDGETS[spec.budget_class]:
        raise BudgetError(f"{spec.hash12}: params_total={n:,} > {spec.budget_class}={BUDGETS[spec.budget_class]:,}")


def param_count_spec(spec: BackboneSpec) -> Tuple[int, int]:
    """(encoder, head) sem alocar pesos (meta device)."""
    with torch.device("meta"):
        m = Backbone(spec)
    return count_params(m)


def build_backbone(spec: BackboneSpec, *, seed: Optional[int] = None, enforce_budget: bool = True,
                   measure: bool = True) -> Backbone:
    """Monta o módulo, mede o custo e grava em ``model.spec.cost``; ``budget_class`` vazio -> menor Bk
    que comporta. ``enforce_budget``: viola teto -> ``BudgetError`` (o teto duro de 5,5M vale sempre)."""
    if seed is not None:
        torch.manual_seed(int(seed))
    model = Backbone(spec)
    if measure:
        model.spec.cost = measure_cost(model)
        if model.spec.cost.params_total > HARD_CAP:
            raise BudgetError(f"params_total={model.spec.cost.params_total:,} > teto duro {HARD_CAP:,}")
        if model.spec.budget_class is None:
            model.spec.budget_class = band_of(model.spec.cost.params_total)
        if enforce_budget:
            check_budget(model.spec)
        elif model.spec.cost.params_total > HARD_CAP:
            raise BudgetError(f"params_total={model.spec.cost.params_total:,} > teto duro {HARD_CAP:,}")
    return model


def refresh_spec(model: Backbone) -> BackboneSpec:
    """Re-mede custo depois de um morfismo; mantém budget_class se ainda couber, senão sobe de faixa."""
    model.spec.cost = measure_cost(model)
    if model.spec.cost.params_total > HARD_CAP:
        raise BudgetError(f"params_total={model.spec.cost.params_total:,} > teto duro {HARD_CAP:,}")
    band = band_of(model.spec.cost.params_total)
    if model.spec.budget_class is None or BUDGETS[band] > BUDGETS[model.spec.budget_class]:
        model.spec.budget_class = band
    return model.spec
