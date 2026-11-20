"""Morfismos que preservam a função (Net2Net / LEMONADE) sobre ``Backbone`` (ConvBlock + EmbeddingHead).

Cada operação recebe e devolve um ``GrowthResult`` acumulado em relação ao pai ORIGINAL: mapa de
camadas (filho -> pai | "identity" | "residual_zero"), mapas de canais por camada (canal do filho ->
canal do pai, -1 = novo), máscaras por elemento (True = peso herdado do pai) e a lista de operações.
``finalize`` mede o custo, a preservação em ``eval()`` e gera o ``GrowthSpec``.

Operações e condição de exatidão (verificadas numericamente em tests/backbones):

* ``widen`` (zero-pad do consumidor): filtros novos com init livre (aleatória ou ``new_filters``); as
  colunas de ENTRADA novas de toda conv consumidora = 0. Exato para:
    - conv densa / pointwise, com ``kernel_grow`` (k -> k+2m, zero-pad central);
    - BN (gamma, beta, running_mean, running_var, num_batches_tracked copiados por canal): exato em
      eval() e também em treino (estatísticas por canal);
    - InstanceNorm afim (por canal);
    - residual ``identity``/``gated`` (o mapa de saída É o de entrada, mesma ordem: o skip leva os
      canais novos aos canais novos), ``project`` (1x1 com colunas novas = 0);
    - dwsep (depthwise novo para canal novo, pointwise com colunas novas = 0) e depthwise com cin == cout;
    - inception (widen por ramo, colunas novas = 0 em cada ramo; a partição ``branches`` é explícita no
      spec e os canais novos ficam intercalados por ramo; o mapa de canais acompanha);
    - ZCA estatístico: W_filho = [[W_pai, 0], [0, I]] e mu_filho = [mu_pai, 0] (na ordem do mapa).
      Exato SÓ com o branqueamento CONGELADO (buffers do pai, nunca re-``fit`` no filho).
    - DBN: running_W por grupo estendido em bloco-diagonal com I nos canais novos. Exato SÓ em eval()
      (branqueamento congelado = running stats). Em treino o DBN recalcula o branqueamento sobre todos
      os canais do grupo e a exatidão se perde. Exige que cada grupo do filho contenha os canais de no
      máximo um grupo inteiro do pai (verificado; senão ``NotExactError``).
    - GroupNorm / LayerNorm (canal): NÃO exatos (as estatísticas misturam canais); só com
      ``allow_approx=True`` e declarados ``exact_claimed=False``.
* ``head_widen`` (sempre junto do widen da última camada): 1x1 da cabeça [D, C+n] com colunas novas = 0
  (ou o 1x1 ``reduce`` da cabeça lowrank). Exato para qualquer pooling (o 1x1 vem antes do pooling).
* ``deepen_identity``: conv k (centro = I, bias 0) + ReLU [+ BN com rm=0, rv=1, gamma=sqrt(1+eps),
  beta=0], inserida antes da camada ``at``. Exato se a entrada for >= 0 (camada anterior com ReLU/ReLU6
  e sem skip); com BN, exato só em eval().
* ``deepen_residual_zero``: y = act(conv(x)) + x com conv = 0 (act(0) = 0: silu/gelu/relu). Exato para
  qualquer entrada; norm none ou batch (BN(0) = 0 com rm = 0, beta = 0).
* ``add_skip_zero``: liga um skip de contribuição zero numa camada existente: ``gated`` (y + lambda*x,
  lambda = 0, cin == cout) ou ``project`` (1x1 = 0). Exato.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from ga_search.backbones.build import Backbone, GatedSkip, LowRankHead, refresh_spec
from ga_search.backbones.spec import (GrowthSpec, LayerInheritance, LayerSpec, OpRecord,
                                      band_of)
from ga_search.modehb.ops_norm_act import DecorrelatedBatchNorm, InceptionConv, ZCAWhitening, inception_split


class MorphismError(ValueError):
    pass


class NotExactError(MorphismError):
    """A operação pedida não preserva a função nas condições dadas (use allow_approx para declarar)."""


Src = Union[int, str]


@dataclass
class GrowthResult:
    parent: Backbone                                   # pai original (não é modificado)
    child: Backbone
    layer_map: List[Src]                               # camada do filho -> camada do pai | "identity" | "residual_zero"
    out_maps: List[Optional[List[int]]]                # canal do filho -> canal do pai (-1 = novo); None = camada nova
    masks: Dict[str, torch.Tensor]                     # param do filho -> bool (True = herdado do pai)
    ops: List[OpRecord] = field(default_factory=list)
    new_init: Dict[int, str] = field(default_factory=dict)
    skip_added: set = field(default_factory=set)
    head_mode: str = "inherit_zero_cols"

    @property
    def exact_claimed(self) -> bool:
        return all(o.exact_claimed for o in self.ops)


def start(parent: Backbone) -> GrowthResult:
    """Resultado identidade (filho = cópia literal do pai)."""
    child = copy.deepcopy(parent)
    n = len(parent.blocks)
    chans = [int(parent.spec.block_out_channels(j)) for j in range(n)]      # growth v3: conta dense_out
    masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in child.named_parameters()}
    return GrowthResult(parent=parent, child=child, layer_map=list(range(n)),
                        out_maps=[list(range(c)) for c in chans], masks=masks)


# ─────────────────────────────────────────────────────────────────── utilidades
def _ix(lst):
    return torch.as_tensor(lst, dtype=torch.long)


def _old(map_):
    pos = [i for i, m in enumerate(map_) if m >= 0]
    return pos, [int(map_[i]) for i in pos]


def _remap_vec(dst: torch.Tensor, src: torch.Tensor, rows: Sequence[int]) -> None:
    pos, par = _old(rows)
    if pos:
        dst[_ix(pos)] = src[_ix(par)].to(dst.dtype)


def _remap_conv(dst: torch.Tensor, src: torch.Tensor, rows: Sequence[int], cols: Optional[Sequence[int]]) -> None:
    """Linhas (filtros) herdadas: zera a linha inteira (colunas novas e borda do kernel = 0) e copia o
    bloco [linhas herdadas x colunas herdadas] no centro do kernel. ``cols=None``: conv por canal."""
    pos, par = _old(rows)
    if not pos:
        return
    kc, kp = dst.shape[-1], src.shape[-1]
    if kc < kp or (kc - kp) % 2:
        raise NotExactError(f"kernel {kp} -> {kc} não preserva a função (só k -> k+2m)")
    o = (kc - kp) // 2
    r = _ix(pos)
    dst[r] = torch.zeros_like(dst[r])
    if cols is None:
        dst[r, :, o:o + kp, o:o + kp] = src[_ix(par)].to(dst.dtype)
        return
    cpos, cpar = _old(cols)
    if cpos:
        dst[r[:, None], _ix(cpos)[None, :], o:o + kp, o:o + kp] = src[_ix(par)][:, _ix(cpar)].to(dst.dtype)


def _branch_maps(parent_conv: InceptionConv, child_conv: InceptionConv, local: bool = False) -> Dict[str, List[int]]:
    """Mapa de canais por ramo: ramo b do filho = canais herdados do ramo b do pai + novos ao fim.
    ``local=True``: índices dentro do ramo (linhas do conv do ramo); senão, na concatenação do pai."""
    pb = {n: int(c) for n, _, c in parent_conv.branch_spec}
    cb = [(n, int(c)) for n, _, c in child_conv.branch_spec]
    if [n for n, _ in cb] != [n for n, _, _ in parent_conv.branch_spec]:
        raise NotExactError("inception: ramos do filho diferem dos do pai")
    off_p, maps = 0, {}
    for n, c in cb:
        if c < pb[n]:
            raise NotExactError(f"inception: ramo {n} encolheu ({pb[n]} -> {c})")
        base = 0 if local else off_p
        maps[n] = list(range(base, base + pb[n])) + [-1] * (c - pb[n])
        off_p += pb[n]
    return maps


def _inception_out_map(bmaps: Dict[str, List[int]], order: Sequence[str]) -> List[int]:
    out = []
    for n in order:
        out += bmaps[n]
    return out


def _dbn_extend(dst_mod: DecorrelatedBatchNorm, src_mod: DecorrelatedBatchNorm, rows: Sequence[int],
                put: Callable[[str, torch.Tensor], None]) -> None:
    pgroup = {}
    for i, (a, b) in enumerate(src_mod.bounds):
        for q in range(a, b):
            pgroup[q] = i
    for gi, (a, b) in enumerate(dst_mod.bounds):
        W = torch.eye(b - a, dtype=getattr(dst_mod, f"running_W{gi}").dtype)
        loc = [(p - a, rows[p]) for p in range(a, b) if rows[p] >= 0]
        if loc:
            gs = {pgroup[q] for _, q in loc}
            if len(gs) != 1:
                raise NotExactError(f"DBN: grupo {gi} do filho mistura grupos {sorted(gs)} do pai")
            pg = gs.pop()
            pa, pb_ = src_mod.bounds[pg]
            if sorted(q for _, q in loc) != list(range(pa, pb_)):
                raise NotExactError(f"DBN: grupo {pg} do pai foi partido entre grupos do filho")
            Wp = getattr(src_mod, f"running_W{pg}")
            li = _ix([p for p, _ in loc])
            qi = _ix([q - pa for _, q in loc])
            W[li[:, None], li[None, :]] = Wp[qi[:, None], qi[None, :]].to(W.dtype)
        put(f"running_W{gi}", W)


# ─────────────────────────────────────────────────────────────────── preenchimento de um bloco
def _fill_block(child_blk: nn.Module, parent_blk: nn.Module, in_map: Sequence[int], out_map: Sequence[int],
                prior_masks: Optional[Dict[str, torch.Tensor]] = None, *, allow_approx: bool = False):
    """Copia os pesos herdados de ``parent_blk`` para ``child_blk`` (in place) segundo os mapas.
    Devolve (máscaras do bloco filho: True = herdado do pai ORIGINAL, exato?, notas)."""
    exact, notes = True, []
    psd = parent_blk.state_dict()
    pmods = dict(parent_blk.named_modules())
    child_params = dict(child_blk.named_parameters())
    masks = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in child_params.items()}
    pm = prior_masks if prior_masks is not None else {k: torch.ones_like(psd[k], dtype=torch.bool)
                                                      for k, _ in parent_blk.named_parameters()}
    bmaps = None
    if isinstance(getattr(child_blk, "conv", None), InceptionConv):
        bmaps = _branch_maps(parent_blk.conv, child_blk.conv, local=True)

    with torch.no_grad():
        for mname, mod in child_blk.named_modules():
            pmod = pmods.get(mname)
            pre = f"{mname}." if mname else ""
            if pmod is None:
                if any(True for _ in mod.parameters(recurse=False)):
                    raise MorphismError(f"módulo {mname!r} sem correspondente no pai")
                continue

            def key(n):
                return pre + n

            def put_param(n, fn_val, fn_mask):
                t = getattr(mod, n)
                fn_val(t.data, psd[key(n)])
                if key(n) in masks:
                    fn_mask(masks[key(n)], pm[key(n)])

            if isinstance(mod, nn.Conv2d):
                if mname.startswith("conv.branches."):
                    rows, cols = bmaps[mname.split(".")[2]], in_map
                elif mname in ("conv.0", "conv.depthwise") and mod.groups > 1:   # depthwise do dwsep (ConvBlock | SpifilOpsBlock)
                    rows, cols = in_map, None
                elif mod.groups > 1:                                 # depthwise com cin == cout
                    if mod.groups != mod.in_channels or mod.in_channels != mod.out_channels:
                        raise NotExactError("depthwise com multiplicador > 1: widen não implementado")
                    rows, cols = out_map, None
                else:
                    rows, cols = out_map, in_map
                put_param("weight", lambda d, s: _remap_conv(d, s, rows, cols),
                          lambda d, s: _remap_conv(d, s, rows, cols))
                if mod.bias is not None:
                    put_param("bias", lambda d, s: _remap_vec(d, s, rows), lambda d, s: _remap_vec(d, s, rows))
            elif getattr(mod, "is_cov_gated", False) and mname == "" and "gate" in dict(mod.named_parameters(recurse=False)):
                gmode = getattr(mod, "cov_gate_mode", "channel")          # growth v3.1: canal novo -> g = 0
                if gmode == "channel":
                    put_param("gate", lambda d, s: (d.zero_(), _remap_vec(d, s, out_map)),
                              lambda d, s: _remap_vec(d, s, out_map))
                else:
                    put_param("gate", lambda d, s: d.copy_(s), lambda d, s: d.copy_(s))
            elif isinstance(mod, GatedSkip):
                put_param("gate", lambda d, s: d.copy_(s), lambda d, s: d.copy_(s))
            elif isinstance(mod, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if isinstance(mod, (nn.GroupNorm, nn.LayerNorm)):
                    if not allow_approx:
                        raise NotExactError(f"{type(mod).__name__}: estatísticas misturam canais; widen não exato")
                    exact = False
                    notes.append(f"{mname}: {type(mod).__name__} NÃO exato (afim copiado)")
                for n in ("weight", "bias"):
                    if getattr(mod, n, None) is not None:
                        put_param(n, lambda d, s: _remap_vec(d, s, out_map), lambda d, s: _remap_vec(d, s, out_map))
                for n in ("running_mean", "running_var"):
                    if getattr(mod, n, None) is not None:
                        _remap_vec(getattr(mod, n), psd[key(n)], out_map)
                if getattr(mod, "num_batches_tracked", None) is not None:
                    mod.num_batches_tracked.copy_(psd[key("num_batches_tracked")])
            elif isinstance(mod, ZCAWhitening):
                pos, par = _old(out_map)
                mod.mu.zero_()
                _remap_vec(mod.mu, psd[key("mu")], out_map)
                W = torch.eye(mod.W.shape[0], dtype=mod.W.dtype)
                if pos:
                    W[_ix(pos)[:, None], _ix(pos)[None, :]] = psd[key("W")][_ix(par)[:, None], _ix(par)[None, :]].to(W.dtype)
                mod.W.copy_(W)
                mod.fitted.copy_(psd[key("fitted")])
                notes.append(f"{mname}: ZCA exato só com branqueamento congelado (não re-fit no filho)")
            elif isinstance(mod, DecorrelatedBatchNorm):
                mod.running_mean.zero_()
                _remap_vec(mod.running_mean, psd[key("running_mean")], out_map)
                _dbn_extend(mod, pmod, out_map, lambda n, W: getattr(mod, n).copy_(W))
                if mod.weight is not None:
                    put_param("weight", lambda d, s: _remap_vec(d, s, out_map), lambda d, s: _remap_vec(d, s, out_map))
                    put_param("bias", lambda d, s: _remap_vec(d, s, out_map), lambda d, s: _remap_vec(d, s, out_map))
                notes.append(f"{mname}: DBN exato só em eval() (running_W congelado)")
            elif isinstance(mod, nn.PReLU):
                if mod.weight.numel() != 1:
                    raise NotExactError("PReLU por canal: não implementado")
                put_param("weight", lambda d, s: d.copy_(s), lambda d, s: d.copy_(s))
            else:
                own = list(mod.named_parameters(recurse=False))
                if own:
                    raise MorphismError(f"módulo {type(mod).__name__} em {mname!r} sem regra de widen")
    return masks, exact, notes


def _fill_head(child_head: nn.Module, parent_head: nn.Module, in_map: Sequence[int],
               prior_masks: Dict[str, torch.Tensor]):
    """head_widen: o 1x1 de entrada da cabeça recebe colunas novas = 0; o resto é copiado inteiro."""
    first = "reduce" if isinstance(child_head, LowRankHead) else "proj.0"
    psd = parent_head.state_dict()
    masks = {}
    with torch.no_grad():
        for k, t in child_head.state_dict().items():
            if k == f"{first}.weight":
                rows = list(range(t.shape[0]))
                _remap_conv(t, psd[k], rows, in_map)
                m = torch.zeros_like(t, dtype=torch.bool)
                _remap_conv(m, prior_masks[k], rows, in_map)
            else:
                if t.shape != psd[k].shape:
                    raise MorphismError(f"cabeça: {k} mudou de forma {tuple(psd[k].shape)} -> {tuple(t.shape)}")
                t.copy_(psd[k])
                m = prior_masks[k].clone() if k in prior_masks else None
            if m is not None:
                masks[k] = m
    return masks


def _prefixed(masks: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    n = len(prefix)
    return {k[n:]: v for k, v in masks.items() if k.startswith(prefix)}


def _compose(prev: Optional[List[int]], local: Sequence[int]) -> Optional[List[int]]:
    if prev is None:
        return None
    return [int(prev[m]) if m >= 0 else -1 for m in local]


# ─────────────────────────────────────────────────────────────────── widen (+ head_widen)
NewFilters = Callable[[int, nn.Module, List[int]], None]


def widen(res: GrowthResult, widths: Dict[int, int], *, kernels: Optional[Dict[int, int]] = None,
          branches: Optional[Dict[int, List[List]]] = None, new_filters: Optional[NewFilters] = None,
          new_init_label: str = "random:pytorch_default", seed: int = 0, allow_approx: bool = False) -> GrowthResult:
    """Alarga as camadas ``widths`` (índice plano -> nova largura >= atual) e, se mudar a última, a cabeça.

    ``new_filters(j, child_block, out_map_local)``: opcional, inicializa os filtros NOVOS (ex.: SPiFiL do
    Agente 2) antes da cópia dos herdados (a cópia sobrescreve as linhas herdadas e zera as colunas novas).
    """
    kernels, branches = dict(kernels or {}), dict(branches or {})
    M = res.child
    spec = M.spec.copy()
    layers = spec.layers()
    n = len(layers)
    for j in list(widths) + list(kernels) + list(branches):
        if not 0 <= int(j) < n:
            raise MorphismError(f"camada {j} fora de 0..{n - 1}")
    in_count = int(spec.input.in_channels)
    for j, L in enumerate(layers):
        c_old = int(L.channels)
        c_new = int(widths.get(j, c_old))
        if c_new < c_old:
            raise MorphismError(f"camada {j}: widen não encolhe ({c_old} -> {c_new})")
        if (L.residual in ("identity", "gated") or L.conv == "depthwise"
                or getattr(L, "cov_gate", "none") != "none") and c_new != in_count:
            raise NotExactError(f"camada {j} ({L.residual}/{L.conv}) exige cout == cin = {in_count}")
        cin_j = in_count
        L.channels = c_new
        L.kernel = int(kernels.get(j, L.kernel))
        if L.conv == "inception":
            if j in branches:
                L.branches = [list(b) for b in branches[j]]
            elif c_new != c_old:                 # canais extras repartidos como inception_split(extra)
                add = {nm: c for nm, _, c in inception_split(c_new - c_old)}
                L.branches = [[nm, k, int(c) + int(add.get(nm, 0))] for nm, k, c in L.branches]
                left = c_new - sum(int(b[2]) for b in L.branches)    # growth v3: ramos com nomes fora do
                if left > 0:                                          # inception_split (ramo novo b3_2...): o resto
                    L.branches[0][2] = int(L.branches[0][2]) + left   # vai para o 1º ramo (antes: SpecError)
        in_count = c_new + (cin_j if getattr(L, "dense_out", False) else 0)      # growth v3: passagem dense
    torch.manual_seed(int(seed))
    child = Backbone(spec)                          # pesos novos = init PyTorch com semente registrada
    masks: Dict[str, torch.Tensor] = {}
    ops_notes, exact = [], True
    in_local = list(range(int(spec.input.in_channels)))
    out_maps = []
    for j in range(n):
        cb, pb = child.blocks[j], M.blocks[j]
        Lc = spec.layers()[j]
        if isinstance(cb.conv, InceptionConv):
            bm = _branch_maps(pb.conv, cb.conv)
            out_local = _inception_out_map(bm, [b[0] for b in cb.conv.branch_spec])
        elif Lc.residual in ("identity", "gated") or Lc.conv == "depthwise" or getattr(Lc, "cov_gate", "none") != "none":
            out_local = list(in_local)
        else:
            c_par = int(pb.conv.out_channels if not isinstance(pb.conv, nn.Sequential) else pb.conv[1].out_channels)
            out_local = list(range(c_par)) + [-1] * (int(Lc.channels) - c_par)
        if new_filters is not None and any(m < 0 for m in out_local):
            new_filters(j, cb, list(out_local))
        bm_masks, ex, nt = _fill_block(cb, pb, in_local, out_local, _prefixed(res.masks, f"blocks.{j}."),
                                       allow_approx=allow_approx)
        exact &= ex
        ops_notes += [f"blocks.{j}: {x}" for x in nt]
        masks.update({f"blocks.{j}.{k}": v for k, v in bm_masks.items()})
        if getattr(Lc, "dense_out", False):        # growth v3: saída = conv (out_local) + passagem da entrada
            c_par_conv = int(M.spec.layers()[j].channels)
            out_local = list(out_local) + [c_par_conv + m if m >= 0 else -1 for m in in_local]
        out_maps.append(_compose(res.out_maps[j], out_local))
        in_local = out_local
    masks.update({f"head.{k}": v for k, v in _fill_head(child.head, M.head, in_local,
                                                         _prefixed(res.masks, "head.")).items()})
    widened = {int(j): int(w) for j, w in widths.items() if int(w) != int(M.spec.layers()[int(j)].channels)}
    new_res = GrowthResult(parent=res.parent, child=child, layer_map=list(res.layer_map), out_maps=out_maps,
                           masks=masks, ops=list(res.ops), new_init=dict(res.new_init),
                           skip_added=set(res.skip_added), head_mode=res.head_mode)
    for j in widened:
        new_res.new_init[j] = new_init_label + f"(seed={seed})"
    cond = "eval(); " + "; ".join(sorted(set(x.split(": ", 1)[1] for x in ops_notes))) if ops_notes else "sempre"
    new_res.ops.append(OpRecord(op="widen", where={"layers": sorted(widened)},
                                params={"widths": {str(k): v for k, v in sorted(widened.items())},
                                        "kernels": {str(k): int(v) for k, v in kernels.items()},
                                        "new_init": new_init_label, "seed": int(seed), "variant": "widen_zero"},
                                exact_claimed=exact, condition=cond))
    if n - 1 in widened:
        new_res.ops.append(OpRecord(op="head_widen", where={"head": spec.head.label},
                                    params={"in_channels": [int(M.spec.layers()[-1].channels), int(spec.layers()[-1].channels)],
                                            "rule": "inherit_zero_cols"},
                                    exact_claimed=True, condition="qualquer pooling (1x1 antes do pooling)"))
    return new_res


# ─────────────────────────────────────────────────────────────────── inserção de camadas
def _nonneg_output(L: LayerSpec) -> bool:
    return (L.activation in ("relu", "relu6") and L.residual == "none" and not getattr(L, "dense_out", False)
            and getattr(L, "cov_gate", "none") == "none")


def _insert(res: GrowthResult, at: int, newL: LayerSpec, tag: str, seed: int) -> GrowthResult:
    M = res.child
    spec = M.spec.copy()
    n = len(spec.layers())
    if not 1 <= at <= n:
        raise MorphismError(f"inserção em {at}: use 1..{n} (antes do stem não)")
    if at < n:
        s, pos = spec.locate(at)
        spec.stages[s].layers.insert(pos, newL)
    else:
        spec.stages[-1].layers.append(newL)
    torch.manual_seed(int(seed))
    child = Backbone(spec)
    src = list(range(at)) + [None] + list(range(at, n))
    masks = {}
    with torch.no_grad():
        for j, s_ in enumerate(src):
            if s_ is None:
                for k, v in child.blocks[j].named_parameters():
                    masks[f"blocks.{j}.{k}"] = torch.zeros_like(v, dtype=torch.bool)
                continue
            child.blocks[j].load_state_dict(M.blocks[s_].state_dict())
            for k, v in _prefixed(res.masks, f"blocks.{s_}.").items():
                masks[f"blocks.{j}.{k}"] = v.clone()
        child.head.load_state_dict(M.head.state_dict())
    masks.update({k: v.clone() for k, v in res.masks.items() if k.startswith("head.")})
    shift = lambda d: {(k if k < at else k + 1): v for k, v in d.items()}  # noqa: E731
    return GrowthResult(parent=res.parent, child=child,
                        layer_map=list(res.layer_map[:at]) + [tag] + list(res.layer_map[at:]),
                        out_maps=list(res.out_maps[:at]) + [None] + list(res.out_maps[at:]), masks=masks,
                        ops=list(res.ops), new_init={**shift(res.new_init), at: tag},
                        skip_added={k if k < at else k + 1 for k in res.skip_added}, head_mode=res.head_mode)


def deepen_identity(res: GrowthResult, at: int, *, kernel: int = 3, norm: str = "none",
                    allow_approx: bool = False) -> GrowthResult:
    """Net2Deeper: insere conv k identidade + ReLU antes da camada ``at`` (largura = a da entrada)."""
    if norm not in ("none", "batch"):
        raise MorphismError("deepen_identity: norm none|batch")
    layers = res.child.spec.layers()
    prevL = layers[at - 1]
    exact = _nonneg_output(prevL)
    if not exact and not allow_approx:
        raise NotExactError(f"deepen_identity antes de {at}: entrada não é >= 0 "
                            f"(camada anterior act={prevL.activation}, residual={prevL.residual})")
    C = int(res.child.spec.block_out_channels(at - 1))
    newL = LayerSpec(channels=C, kernel=int(kernel), conv="conv", norm=norm, activation="relu",
                     residual="none", pool="none", init="identity")
    out = _insert(res, at, newL, "identity", seed=0)
    blk = out.child.blocks[at]
    with torch.no_grad():
        blk.conv.weight.zero_()
        blk.conv.weight[torch.arange(C), torch.arange(C), kernel // 2, kernel // 2] = 1.0
        if blk.conv.bias is not None:
            blk.conv.bias.zero_()
        if norm == "batch":
            bn = blk.norm
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.weight.fill_(math.sqrt(1.0 + bn.eps))
            bn.bias.zero_()
    cond = "entrada >= 0 (ReLU anterior, sem skip)" + ("; BN: só eval()" if norm == "batch" else "")
    out.ops.append(OpRecord(op="deepen_identity", where={"before": int(at)},
                            params={"kernel": int(kernel), "norm": norm, "width": C},
                            exact_claimed=exact, condition=cond if exact else "NÃO exato: entrada pode ser < 0"))
    return out


RESIDUAL_ZERO_NORMS = ("none", "batch", "zca", "dbn", "batch_dbn", "batch_zca")


@torch.no_grad()
def _zero_norm_of_zero(norm: nn.Module) -> None:
    """Garante norm(0) = 0 em eval(): BN (running_mean = 0, beta = 0), ZCA (mu = 0; W qualquer), DBN
    (running_mean = 0, bias = 0); Sequential (batch_dbn/batch_zca) recursivo."""
    for m in norm.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, ZCAWhitening):
            m.mu.zero_()
        elif isinstance(m, DecorrelatedBatchNorm):
            m.running_mean.zero_()
            if m.bias is not None:
                m.bias.zero_()


def deepen_residual_zero(res: GrowthResult, at: int, *, act: str = "silu", kernel: int = 3, conv: str = "conv",
                         norm: str = "none") -> GrowthResult:
    """Insere y = act(conv(x)) + x com conv = 0 (act(0) = 0) antes da camada ``at``."""
    if act not in ("silu", "gelu", "relu"):
        raise MorphismError("deepen_residual_zero: act com act(0) = 0 (silu|gelu|relu)")
    if norm not in RESIDUAL_ZERO_NORMS:
        raise MorphismError(f"deepen_residual_zero: norm {RESIDUAL_ZERO_NORMS}")
    C = int(res.child.spec.block_out_channels(at - 1))
    newL = LayerSpec(channels=C, kernel=int(kernel), conv=conv, norm=norm, activation=act, residual="identity",
                     pool="none", init="residual_zero")
    out = _insert(res, at, newL, "residual_zero", seed=0)
    blk = out.child.blocks[at]
    with torch.no_grad():
        for m in blk.conv.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
        _zero_norm_of_zero(blk.norm)            # growth v3: norm(0) = 0 (BN, ZCA, DBN, batch_dbn)
    cond = "act(0)=0; qualquer entrada"
    if norm in ("zca", "dbn", "batch_dbn", "batch_zca"):
        cond += "; whitening: exato só em eval()/branqueamento congelado (running_mean/mu = 0, bias = 0)"
    elif norm == "batch":
        cond += "; BN: running_mean = 0, beta = 0"
    out.ops.append(OpRecord(op="deepen_residual_zero", where={"before": int(at)},
                            params={"kernel": int(kernel), "conv": conv, "norm": norm, "act": act, "width": C,
                                    "trainable_at_zero": act != "relu"},
                            exact_claimed=True, condition=cond))
    return out


def add_skip_zero(res: GrowthResult, layer: int) -> GrowthResult:
    """Liga um residual de contribuição zero numa camada existente (gated lambda=0 ou project 1x1=0)."""
    M = res.child
    spec = M.spec.copy()
    layers = spec.layers()
    L = layers[layer]
    if L.residual != "none":
        raise MorphismError(f"camada {layer} já tem residual {L.residual}")
    cin = int(spec.block_in_channels(layer))         # growth v3: conta a passagem dense_out
    L.residual = "gated" if cin == int(L.channels) else "project"
    child = Backbone(spec)
    masks = {k: v.clone() for k, v in res.masks.items()}
    with torch.no_grad():
        for j in range(len(layers)):
            missing, unexpected = child.blocks[j].load_state_dict(M.blocks[j].state_dict(), strict=False)
            if unexpected or (j != layer and missing) or any(not k.startswith("skip.") for k in missing):
                raise MorphismError(f"add_skip_zero: chaves inesperadas {missing} {unexpected}")
        child.head.load_state_dict(M.head.state_dict())
        for k, p in child.blocks[layer].skip.named_parameters():
            p.zero_()
            masks[f"blocks.{layer}.skip.{k}"] = torch.zeros_like(p, dtype=torch.bool)
    out = GrowthResult(parent=res.parent, child=child, layer_map=list(res.layer_map), out_maps=list(res.out_maps),
                       masks=masks, ops=list(res.ops), new_init=dict(res.new_init),
                       skip_added=set(res.skip_added) | {layer}, head_mode=res.head_mode)
    out.ops.append(OpRecord(op="add_skip_zero", where={"layer": int(layer)}, params={"skip": L.residual},
                            exact_claimed=True, condition="skip inicial = 0"))
    return out


# ─────────────────────────────────────────────────────────────────── aplicação / GrowthSpec
OPS = {"widen": widen, "deepen_identity": deepen_identity, "deepen_residual_zero": deepen_residual_zero,
       "add_skip_zero": add_skip_zero}


def apply_ops(parent: Backbone, ops: Sequence[dict]) -> GrowthResult:
    """``ops``: lista de ``{"op": nome, **kwargs}`` aplicadas em ordem (índices referem-se ao filho corrente)."""
    res = start(parent)
    for o in ops:
        o = dict(o)
        fn = OPS[o.pop("op")]
        res = fn(res, **o)
    return res


def _shapes(mod: nn.Module) -> Dict[str, List[int]]:
    return {k: list(v.shape) for k, v in mod.named_parameters()}


def finalize(res: GrowthResult, x: Optional[torch.Tensor] = None, *, data: Optional[dict] = None,
             tol: float = 1e-5, child_budget_class: Optional[str] = None, protect: str = "full") -> GrowthSpec:
    """Mede custo do filho, herança por camada e preservação (se ``x`` for dado) e monta o ``GrowthSpec``.
    ``x``: imagens de TREINO/VALIDAÇÃO (nunca teste); ``data`` descreve a origem (registrada)."""
    from ga_search.backbones.preservation import measure_preservation
    P, C = res.parent, res.child
    if child_budget_class is not None:
        C.spec.budget_class = child_budget_class
    refresh_spec(C)
    from ga_search.backbones.build import check_budget
    check_budget(C.spec)
    C.spec.provenance.parent = P.spec.id
    pnames = P.spec.layer_names()
    cnames = C.spec.layer_names()
    per_layer = []
    for j, src in enumerate(res.layer_map):
        blk = C.blocks[j]
        tot = sum(p.numel() for p in blk.parameters())
        inh = int(sum(int(res.masks[f"blocks.{j}.{k}"].sum()) for k, _ in blk.named_parameters()))
        if isinstance(src, int):
            origin = "inherited" if inh == tot else "fp_widen"
            pidx, pname, pshape = src, pnames[src], _shapes(P.blocks[src])
        else:
            origin, pidx, pname, pshape = f"fp_{src}", None, None, {}
        per_layer.append(LayerInheritance(child_layer=cnames[j], child_index=j, parent_layer=pname, parent_index=pidx,
                                          origin=origin, shape_parent=pshape, shape_child=_shapes(blk),
                                          out_map=res.out_maps[j],
                                          in_map=(list(range(int(C.spec.input.in_channels))) if j == 0
                                                  else res.out_maps[j - 1]),
                                          n_inherited_elements=inh,
                                          n_new_elements=tot - inh, new_init=res.new_init.get(j, ""),
                                          skip_added=j in res.skip_added))
    htot = sum(p.numel() for p in C.head.parameters())
    hinh = int(sum(int(res.masks[f"head.{k}"].sum()) for k, _ in C.head.named_parameters()))
    from dataclasses import asdict
    inheritance = {"head": res.head_mode, "per_layer": [asdict(r) for r in per_layer],
                   "head_detail": {"label": C.spec.head.label, "shape_parent": _shapes(P.head),
                                   "shape_child": _shapes(C.head), "n_inherited_elements": hinh,
                                   "n_new_elements": htot - hinh},
                   "totals": {"inherited": int(sum(r.n_inherited_elements for r in per_layer) + hinh),
                              "new": int(sum(r.n_new_elements for r in per_layer) + htot - hinh)}}
    if x is not None:
        pres = measure_preservation(P, C, res.layer_map, res.out_maps, x, tol=tol, exact_claimed=res.exact_claimed,
                                    data=data)
    else:
        from ga_search.backbones.spec import PreservationRecord
        pres = PreservationRecord(exact_claimed=res.exact_claimed, tolerance_rel=tol)
    from ga_search.backbones.build import measure_cost
    ps, cs = (P.spec.cost if P.spec.cost.measured else measure_cost(P)), C.spec.cost
    g = GrowthSpec(parent=P.spec.id, child=C.spec.id, parent_hash=P.spec.arch_hash(), child_hash=C.spec.arch_hash(),
                   operations=list(res.ops), inheritance=inheritance,
                   params={"parent_total": int(ps.params_total), "child_total": int(cs.params_total),
                           "delta": int(cs.params_total - ps.params_total),
                           "child_budget_class": C.spec.budget_class, "child_band": band_of(cs.params_total)},
                   macs={"parent": int(ps.macs), "child": int(cs.macs)},
                   flops={"parent": int(ps.flops), "child": int(cs.flops)}, preservation=pres,
                   protect={"mode": protect, "scope": "inherited_elements"})
    C.spec.provenance.growth_spec = g.id
    return g
