"""Morfismos da cadeia de crescimento v3 (sobre o ``GrowthResult`` de ``morphisms``).

* ``inception_branch_add(res, layer, kernel, channels)``: acrescenta um ramo (k1/k3/k5) ao bloco ``layer``.
  Conv densa (``conv``/``pointwise``) vira ``inception`` [[b{k0}, k0, C], [novo, k, n]] com os pesos herdados
  copiados para o ramo antigo; bloco já inception ganha um ramo novo (nome único) no fim. Os canais novos
  ficam no FIM da concatenação; as colunas do consumidor (bloco seguinte ou 1x1 da cabeça) são 0 ->
  exato em eval() (mesmas condições de norma do ``widen``).
* ``dense_concat(res, layer)``: DenseNet. O bloco ``layer`` passa a ter ``dense_out`` (saída =
  cat(bloco(x), adaptive_avg_pool(x))); as colunas novas do bloco seguinte (conv e skip project) = 0 -> exato.
* ``bad_filters(acts)`` / ``replace_filters(res, layer, rows, merge=...)``: acha filtros mortos (variância ~0)
  ou quase duplicados (|corr| > thr) pela covariância das respostas e troca as linhas da conv por filtros
  novos; colunas do consumidor = 0 (+ merge Net2Net). NÃO preserva a função exatamente (exact_claimed=False):
  o chamador decide pelo WiSE com âncora = pai e só aceita se kappa >= pai.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from ga_search.backbones.build import Backbone, LowRankHead
from ga_search.backbones.morphisms import (GrowthResult, MorphismError, NotExactError, _compose, _fill_block,
                                          _fill_head, _prefixed)
from ga_search.backbones.spec import OpRecord
from ga_search.modehb.ops_norm_act import DecorrelatedBatchNorm, InceptionConv, ZCAWhitening

BranchInit = Callable[[int, nn.Module, List[int]], None]


# ─────────────────────────────────────────────────────────────────── utilidades
def _conv_rows(blk: nn.Module) -> int:
    conv = blk.conv
    if isinstance(conv, nn.Sequential):
        return int(conv[-1].out_channels)
    return int(conv.out_channels)


def _full_out_local(L, Lp, conv_local: List[int], in_local: List[int], c_par_conv: int) -> List[int]:
    """Mapa de saída do bloco (local ao pai do passo): conv + passagem dense_out (se houver). Passagem que o
    pai do passo não tinha (``Lp`` sem dense_out) é nova: -1."""
    if not getattr(L, "dense_out", False):
        return list(conv_local)
    if not getattr(Lp, "dense_out", False):
        return list(conv_local) + [-1] * len(in_local)
    return list(conv_local) + [c_par_conv + m if m >= 0 else -1 for m in in_local]


def _copy_block(child_blk: nn.Module, parent_blk: nn.Module, prior: Dict[str, torch.Tensor]):
    child_blk.load_state_dict(parent_blk.state_dict())
    return {k: v.clone() for k, v in prior.items()}


def _rebuild(res: GrowthResult, spec, j: int, fill_j: Callable, seed: int, op: OpRecord,
             new_init_label: Optional[str] = None) -> GrowthResult:
    """Monta o filho de ``spec`` (só o bloco ``j`` e o consumidor mudam de forma). ``fill_j(cb, pb, prior)``
    preenche o bloco j e devolve (masks_do_bloco, conv_local). O consumidor (bloco j+1 ou a cabeça) recebe
    colunas novas = 0 via ``_fill_block``/``_fill_head``."""
    M = res.child
    torch.manual_seed(int(seed))
    child = Backbone(spec)
    masks: Dict[str, torch.Tensor] = {}
    out_maps = list(res.out_maps)
    n = len(spec.layers())
    exact, notes = True, []
    in_local = list(range(int(spec.input.in_channels)))
    with torch.no_grad():
        for k in range(n):
            cb, pb = child.blocks[k], M.blocks[k]
            prior = _prefixed(res.masks, f"blocks.{k}.")
            Lc = spec.layers()[k]
            c_par_conv = int(M.spec.layers()[k].channels)
            if k == j:
                bm, conv_local = fill_j(cb, pb, prior, in_local)
            elif k == j + 1:
                conv_local = list(range(_conv_rows(pb)))
                if Lc.residual in ("identity", "gated") or Lc.conv == "depthwise":
                    conv_local = list(in_local)
                bm, ex, nt = _fill_block(cb, pb, in_local, conv_local, prior)
                exact &= ex
                notes += nt
            else:
                bm = _copy_block(cb, pb, prior)
                conv_local = list(range(c_par_conv)) if Lc.residual not in ("identity", "gated") else list(in_local)
            masks.update({f"blocks.{k}.{q}": v for q, v in bm.items()})
            full = _full_out_local(Lc, M.spec.layers()[k], conv_local, in_local, c_par_conv)
            if k in (j, j + 1):
                out_maps[k] = _compose(res.out_maps[k], full)
            in_local = full
        if j == n - 1:
            masks.update({f"head.{q}": v for q, v in _fill_head(child.head, M.head, in_local,
                                                                _prefixed(res.masks, "head.")).items()})
        else:
            child.head.load_state_dict(M.head.state_dict())
            masks.update({q: v.clone() for q, v in res.masks.items() if q.startswith("head.")})
    out = GrowthResult(parent=res.parent, child=child, layer_map=list(res.layer_map), out_maps=out_maps,
                       masks=masks, ops=list(res.ops), new_init=dict(res.new_init),
                       skip_added=set(res.skip_added), head_mode=res.head_mode)
    if new_init_label:
        out.new_init[j] = new_init_label
    if notes:
        op.condition = (op.condition + "; " if op.condition else "") + "; ".join(sorted(set(notes)))
    op.exact_claimed = bool(op.exact_claimed and exact)
    out.ops.append(op)
    return out


# ─────────────────────────────────────────────────────────────────── (a) ramo inception
def _branch_name(existing: Sequence[str], kernel: int) -> str:
    base = f"b{int(kernel)}"
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def inception_branch_add(res: GrowthResult, layer: int, kernel: int, channels: int, *,
                         new_filters: Optional[BranchInit] = None, seed: int = 0) -> GrowthResult:
    """Acrescenta um ramo ``kernel`` x ``kernel`` com ``channels`` filtros ao bloco ``layer`` (saída nova no fim,
    colunas do consumidor = 0). ``new_filters(j, conv_do_ramo_novo, rows)`` inicializa os filtros novos (SPiFiL);
    senão init PyTorch com ``seed``."""
    M = res.child
    spec = M.spec.copy()
    layers = spec.layers()
    n = len(layers)
    j, k, c = int(layer), int(kernel), int(channels)
    if not 0 <= j < n:
        raise MorphismError(f"camada {j} fora de 0..{n - 1}")
    if k not in (1, 3, 5) or c < 1:
        raise MorphismError(f"inception_branch_add: kernel {k} (1|3|5), canais {c} >= 1")
    L = layers[j]
    if L.residual in ("identity", "gated"):
        raise MorphismError(f"inception_branch_add: bloco {j} com residual {L.residual} (cout == cin)")
    if L.conv not in ("conv", "pointwise", "inception"):
        raise MorphismError(f"inception_branch_add: conv {L.conv!r} não suportada (só conv/pointwise/inception)")
    if j + 1 < n and (layers[j + 1].residual in ("identity", "gated") or layers[j + 1].conv == "depthwise"):
        raise MorphismError(f"inception_branch_add: consumidor {j + 1} exige cin fixo "
                            f"({layers[j + 1].residual}/{layers[j + 1].conv})")
    pconv = M.blocks[j].conv
    if L.conv == "inception":
        old = [list(b) for b in L.branches]
    else:
        if not (isinstance(pconv, nn.Conv2d) and pconv.groups == 1):
            raise MorphismError("inception_branch_add: conv do bloco não é densa")
        old = [[f"b{int(pconv.kernel_size[0])}", int(pconv.kernel_size[0]), int(L.channels)]]
    name = _branch_name([b[0] for b in old], k)
    C = int(L.channels)
    L.conv = "inception"
    L.branches = old + [[name, k, c]]
    L.channels = C + c
    conv_local = list(range(C)) + [-1] * c

    def fill_j(cb, pb, prior, in_local):
        if in_local != list(range(len(in_local))):
            raise MorphismError("inception_branch_add: entrada do bloco remapeada (use um passo de cada vez)")
        new_conv = cb.conv
        cb.conv = copy.deepcopy(pb.conv)        # nomes iguais aos do pai -> _fill_block cuida da norma/skip
        bm, ex, nt = _fill_block(cb, pb, in_local, conv_local, prior)
        if not ex:
            raise NotExactError(f"inception_branch_add: norma do bloco não exata ({nt})")
        cb.conv = new_conv
        bm = {q: v for q, v in bm.items() if not q.startswith("conv.")}
        if isinstance(pb.conv, InceptionConv):
            for bn, _, _ in old:
                new_conv.branches[bn].load_state_dict(pb.conv.branches[bn].state_dict())
                for q, _p in new_conv.branches[bn].named_parameters():
                    bm[f"conv.branches.{bn}.{q}"] = prior[f"conv.branches.{bn}.{q}"].clone()
        else:
            ob = new_conv.branches[old[0][0]]
            ob.weight.copy_(pb.conv.weight)
            bm[f"conv.branches.{old[0][0]}.weight"] = prior["conv.weight"].clone()
            if ob.bias is not None:
                if pb.conv.bias is not None:
                    ob.bias.copy_(pb.conv.bias)
                    bm[f"conv.branches.{old[0][0]}.bias"] = prior["conv.bias"].clone()
                else:                            # pai sem bias (norm != none no ConvBlock): bias 0 herdado
                    ob.bias.zero_()
                    bm[f"conv.branches.{old[0][0]}.bias"] = torch.ones_like(ob.bias, dtype=torch.bool)
        nb = new_conv.branches[name]
        if new_filters is not None:
            new_filters(j, nb, list(range(c)))
        for q, p in nb.named_parameters():
            bm[f"conv.branches.{name}.{q}"] = torch.zeros_like(p, dtype=torch.bool)
        return bm, conv_local

    op = OpRecord(op="inception_branch_add", where={"layer": j},
                  params={"kernel": k, "channels": c, "branch": name, "from_conv": "inception" if old and
                          isinstance(pconv, InceptionConv) else "conv", "seed": int(seed),
                          "new_init": "new_filters" if new_filters is not None else "random:pytorch_default"},
                  exact_claimed=True, condition="eval(); colunas do consumidor = 0")
    return _rebuild(res, spec, j, fill_j, seed, op,
                    new_init_label=f"branch:{name}:" + ("new_filters" if new_filters else "pytorch") + f"(seed={seed})")


# ─────────────────────────────────────────────────────────────────── (b) dense_concat
def dense_concat(res: GrowthResult, layer: int) -> GrowthResult:
    """DenseNet: saída do bloco ``layer`` = cat(bloco(x), adaptive_avg_pool(x)); o bloco seguinte recebe as
    colunas novas = 0 (conv e skip project). Exato."""
    M = res.child
    spec = M.spec.copy()
    layers = spec.layers()
    n = len(layers)
    j = int(layer)
    if not 0 <= j < n - 1:
        raise MorphismError(f"dense_concat: camada {j} (proibido no último bloco; use 0..{n - 2})")
    L, Ln = layers[j], layers[j + 1]
    if getattr(L, "dense_out", False):
        raise MorphismError(f"dense_concat: bloco {j} já é dense_out")
    if Ln.residual in ("identity", "gated") or Ln.conv == "depthwise":
        raise MorphismError(f"dense_concat: consumidor {j + 1} exige cin fixo ({Ln.residual}/{Ln.conv})")
    cin = int(spec.block_in_channels(j))
    L.dense_out = True

    def fill_j(cb, pb, prior, in_local):
        bm = _copy_block(cb, pb, prior)
        rows = _conv_rows(pb)
        conv_local = list(range(rows))
        # passagem nova: -1 (não vem do pai deste passo) -> o consumidor zera as colunas
        return bm, conv_local

    op = OpRecord(op="dense_concat", where={"layer": j}, params={"passthrough_channels": cin,
                                                                 "pool": "adaptive_avg"},
                  exact_claimed=True, condition="colunas novas do consumidor = 0")
    return _rebuild(res, spec, j, fill_j, 0, op)


# ─────────────────────────────────────────────────────────────────── (d) filtros ruins
@torch.no_grad()
def bad_filters(acts: torch.Tensor, corr_thr: float = 0.95, var_rel_eps: float = 1e-6,
                n_conv: Optional[int] = None) -> dict:
    """Filtros mortos (var <= var_rel_eps*mediana(var) ou var == 0) e quase duplicados (|corr de Pearson| > thr)
    nas respostas ``acts`` [N, C, H, W] (saída do bloco = o que o consumidor vê). Determinístico.
    ``n_conv``: só os canais < n_conv são candidatos (bloco dense_out: a passagem não é filtro).
    -> {"dead": [c], "mean": {c: média}, "dup": [[c, keep, a, b]] (y_c ~ a*y_keep + b), "corr_max": [C],
        "var": [C], "corr_thr", "var_rel_eps"}"""
    C = int(acts.shape[1])
    m = C if n_conv is None else min(int(n_conv), C)
    Y = acts.detach().double().cpu().transpose(0, 1).reshape(C, -1)[:m]
    mean = Y.mean(1)
    Yc = Y - mean[:, None]
    var = (Yc * Yc).mean(1)
    med = float(var.median()) if m else 0.0
    dead = [int(i) for i in range(m) if float(var[i]) == 0.0 or float(var[i]) <= var_rel_eps * med]
    alive = [i for i in range(m) if i not in set(dead)]
    corr_max = [0.0] * m
    dup: List[list] = []
    if len(alive) > 1:
        A = torch.as_tensor(alive)
        Z = Yc[A] / var[A].sqrt()[:, None]
        R = (Z @ Z.T) / Z.shape[1]
        R.fill_diagonal_(0.0)
        for ii, c in enumerate(alive):
            corr_max[c] = float(R[ii].abs().max())
        pairs = [(-abs(float(R[a, b])), alive[a], alive[b]) for a in range(len(alive)) for b in range(a + 1, len(alive))
                 if abs(float(R[a, b])) > corr_thr]
        pairs.sort()
        dropped, kept = set(), set()
        for _r, a, b in pairs:
            if a in dropped or b in dropped:
                continue
            drop, keep = (a, b) if (float(var[a]), -a) < (float(var[b]), -b) else (b, a)
            if drop in kept:
                drop, keep = keep, drop
                if drop in kept:
                    continue
            dropped.add(drop)
            kept.add(keep)
            cov = float((Yc[drop] * Yc[keep]).mean())
            aa = cov / float(var[keep])
            bb = float(mean[drop]) - aa * float(mean[keep])
            dup.append([int(drop), int(keep), aa, bb])
    return {"dead": dead, "mean": {int(c): float(mean[c]) for c in dead}, "dup": dup, "corr_max": corr_max,
            "var": [float(v) for v in var], "corr_thr": float(corr_thr), "var_rel_eps": float(var_rel_eps)}


def _consumer_convs(child: Backbone, j: int) -> List[Tuple[str, nn.Conv2d, bool]]:
    """Convs que LEEM a saída do bloco j: (nome, conv, linear?) — linear=False para o proj do ramo pool
    (MaxPool antes: o merge não vale) e para o depthwise do dwsep (zera só as linhas)."""
    n = len(child.blocks)
    if j == n - 1:
        h = child.head
        first = h.reduce if isinstance(h, LowRankHead) else h.proj[0]
        return [("head." + ("reduce" if isinstance(h, LowRankHead) else "proj.0"), first, True)]
    blk = child.blocks[j + 1]
    out = []
    for name, mod in blk.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        full = f"blocks.{j + 1}.{name}"
        if name.startswith("conv.branches.pool"):
            out.append((full, mod, False))
        elif name.startswith("conv.branches.") or name == "conv" or name.startswith("skip"):
            if mod.groups != 1:
                raise MorphismError(f"replace_filters: consumidor {name} agrupado")
            out.append((full, mod, True))
        elif name in ("conv.0", "conv.depthwise"):          # dwsep: depthwise lê canal a canal
            out.append((full, mod, False))
    return out


def replace_filters(res: GrowthResult, layer: int, rows: Sequence[int], *,
                    merge: Optional[Dict[int, Tuple[int, float, float]]] = None,
                    new_filters: Optional[BranchInit] = None, seed: int = 0) -> GrowthResult:
    """Troca as linhas ``rows`` da conv densa do bloco ``layer`` por filtros novos (``new_filters(j, conv, rows)``
    = SPiFiL; senão init PyTorch com ``seed``), reseta a norma desses canais e zera as colunas do consumidor.
    ``merge`` {c: (keep, a, b)}: Net2Net, coluna_keep += a*coluna_c e bias += b*sum(coluna_c) (keep < 0: só o
    termo constante b, ex. canal morto com média b). NÃO exato (exact_claimed=False)."""
    M = res.child
    j = int(layer)
    rows = sorted({int(r) for r in rows})
    blk = M.blocks[j]
    conv = blk.conv
    if not (isinstance(conv, nn.Conv2d) and conv.groups == 1):
        raise MorphismError("replace_filters: só conv densa (groups=1)")
    if not rows or rows[-1] >= conv.out_channels or rows[0] < 0:
        raise MorphismError(f"replace_filters: linhas {rows} fora de 0..{conv.out_channels - 1}")
    L = M.spec.layers()[j]
    if L.residual in ("identity", "gated"):
        raise MorphismError("replace_filters: bloco com residual identity/gated (o skip mantém o canal)")
    child = copy.deepcopy(M)
    masks = {k: v.clone() for k, v in res.masks.items()}
    cb = child.blocks[j]
    r = torch.as_tensor(rows, dtype=torch.long)
    merge = {int(k): (int(v[0]), float(v[1]), float(v[2])) for k, v in (merge or {}).items()}
    with torch.no_grad():
        g = torch.Generator().manual_seed(int(seed))
        tmp = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding,
                        bias=conv.bias is not None)
        bound = 1.0 / max(1, conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]) ** 0.5
        tmp.weight.uniform_(-bound, bound, generator=g)
        if tmp.bias is not None:
            tmp.bias.uniform_(-bound, bound, generator=g)
        cb.conv.weight[r] = tmp.weight[r]
        if cb.conv.bias is not None:
            cb.conv.bias[r] = tmp.bias[r]
        if new_filters is not None:
            new_filters(j, cb.conv, list(rows))
        masks[f"blocks.{j}.conv.weight"][r] = False
        if cb.conv.bias is not None:
            masks[f"blocks.{j}.conv.bias"][r] = False
        for name, m in cb.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean[r], m.running_var[r] = 0.0, 1.0
                if m.weight is not None:
                    m.weight[r], m.bias[r] = 1.0, 0.0
                    masks[f"blocks.{j}.{name}.weight"][r] = False
                    masks[f"blocks.{j}.{name}.bias"][r] = False
        for full, cm, linear in _consumer_convs(child, j):
            key = full + ".weight"
            if linear:
                W = cm.weight
                for c in rows:
                    if c in merge:
                        keep, a, b = merge[c]
                        if 0 <= keep < W.shape[1] and keep not in rows:
                            W[:, keep] += a * W[:, c]
                        if cm.bias is not None and b != 0.0:
                            cm.bias += b * W[:, c].sum(dim=(1, 2))
                    W[:, c] = 0.0
                    masks[key][:, c] = False
            else:                                  # depthwise (linhas = canais) ou proj do ramo pool (colunas)
                if cm.groups > 1:
                    cm.weight[r] = 0.0
                    masks[key][r] = False
                    if cm.bias is not None:
                        cm.bias[r] = 0.0
                        masks[full + ".bias"][r] = False
                else:
                    cm.weight[:, r] = 0.0
                    masks[key][:, r] = False
    out = GrowthResult(parent=res.parent, child=child, layer_map=list(res.layer_map), out_maps=list(res.out_maps),
                       masks=masks, ops=list(res.ops), new_init=dict(res.new_init),
                       skip_added=set(res.skip_added), head_mode=res.head_mode)
    out.new_init[j] = f"replace:{'new_filters' if new_filters else 'pytorch'}(seed={seed})"
    out.ops.append(OpRecord(op="replace_bad_filters", where={"layer": j},
                            params={"rows": rows, "merge": {str(k): list(v) for k, v in merge.items()},
                                    "seed": int(seed),
                                    "new_init": "new_filters" if new_filters is not None else "random:pytorch_default"},
                            exact_claimed=False,
                            condition="NÃO preserva a função: decidir por WiSE com âncora = pai (kappa >= pai)"))
    return out


# ─────────────────────────────────────────────────────────────────── v3.1: residual GATED (cov_gate / ReZero)
GATED_NORMS = ("none", "batch", "zca", "dbn", "batch_dbn", "batch_zca")


@torch.no_grad()
def add_gated_residual(res: GrowthResult, at: int, *, kernel: int = 3, norm: str = "zca", act: str = "relu",
                       gate: str = "channel", tag: str = "cov_gate", new_filters: Optional[BranchInit] = None,
                       seed: int = 0, fit_x: Optional[torch.Tensor] = None) -> GrowthResult:
    """Insere antes da camada ``at`` o bloco y = x + g * act(norm(conv(x))) com g = 0 treinável (ReZero).

    ``tag="cov_gate"`` (pedido do usuário): norm de covariância (zca/dbn) -> o ramo usa o branqueamento pela
    covariância das respostas; g decide no SSL/AE se o mecanismo é usado. ``tag="rezero"`` (deepen_rezero):
    mesmo bloco, norm none, g escalar. A conv recebe filtros SPiFiL (``new_filters(at, conv, rows)``) -> o
    gradiente de g é NÃO nulo desde o passo 0 (ao contrário do deepen_residual_zero, que zera a conv).
    ``fit_x`` (imagens de TREINO): ajusta a ZCA nas respostas da conv nova (branqueamento congelado depois).
    Exato para qualquer entrada: g = 0 => y = x (o ramo é finito)."""
    from ga_search.backbones.morphisms import _insert
    from ga_search.backbones.spec import LayerSpec
    if norm not in GATED_NORMS:
        raise MorphismError(f"gated residual: norm {norm!r} fora de {GATED_NORMS}")
    if gate not in ("scalar", "channel"):
        raise MorphismError("gate: scalar | channel")
    spec = res.child.spec
    C = int(spec.block_out_channels(at - 1))
    flavor = spec.layers()[at - 1].flavor
    newL = LayerSpec(channels=C, kernel=int(kernel), conv="conv", norm=norm, activation=act, residual="none",
                     pool="none", init=tag, flavor=flavor, cov_gate=gate)
    out = _insert(res, at, newL, tag, seed=int(seed))
    blk = out.child.blocks[at]
    blk.gate.zero_()
    spifil = None
    if new_filters is not None:
        n_log = None
        new_filters(at, blk.conv, list(range(C)))
        spifil = "new_filters"
    zfit = None
    if fit_x is not None and isinstance(getattr(blk, "norm", None), ZCAWhitening):
        h = fit_x.float()
        was = out.child.training
        out.child.eval()
        for b in out.child.blocks[:at]:
            h = b(h)
        blk.norm.fit(blk.conv(h))
        out.child.train(was)
        zfit = {"n_images": int(fit_x.shape[0]), "split": "train"}
    out.ops.append(OpRecord(op=tag, where={"before": int(at)},
                            params={"kernel": int(kernel), "norm": norm, "act": act, "gate": gate, "width": C,
                                    "init": spifil or "pytorch", "zca_fit": zfit, "gate_init": 0.0},
                            exact_claimed=True, condition="g = 0 (ReZero); qualquer entrada; grad de g != 0"))
    return out


# ─────────────────────────────────────────────────────────────────── v3.1 bloco 5: critérios ReDo / RePr
def redo_scores(acts: torch.Tensor, n_conv: Optional[int] = None) -> torch.Tensor:
    """ReDo (Sokar et al. 2023, Eq. 1): s_i = E|h_i| / mean_k E|h_k| na saída pós-ativação [N,C,H,W]."""
    a = acts.detach().double()
    if n_conv is not None:
        a = a[:, : int(n_conv)]
    m = a.abs().mean(dim=(0, 2, 3))
    return m / m.mean().clamp_min(1e-12)


def redo_dormant(acts: torch.Tensor, tau: float = 0.1, n_conv: Optional[int] = None) -> List[int]:
    s = redo_scores(acts, n_conv)
    return [int(i) for i in torch.nonzero(s <= float(tau)).flatten().tolist()]


def repr_overlap(weight: torch.Tensor) -> torch.Tensor:
    """RePr (Prakash et al. 2019, Eqs. 1-2): O_f = (1/J) sum_j |cos(w_f, w_j) - delta_fj| (filtros achatados)."""
    W = weight.detach().double().flatten(1)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    P = (W @ W.T).abs()
    P.fill_diagonal_(0.0)
    return P.sum(1) / max(P.shape[0], 1)


def repr_select(weight: torch.Tensor, p: float = 0.30) -> List[int]:
    """Os ``p`` filtros mais redundantes (maior O_f), desempate pelo índice (determinístico)."""
    O = repr_overlap(weight)
    k = int(round(float(p) * O.numel()))
    order = sorted(range(O.numel()), key=lambda i: (-float(O[i]), i))
    return sorted(order[:k])


# ─────────────────────────────────────────────────────────────────── v3.1 bloco 5: poda EXATA de canais novos
@torch.no_grad()
def prune_new_channels(res: GrowthResult, layer: int, rows: Sequence[int]) -> GrowthResult:
    """Remove canais de saída do bloco ``layer`` cujas colunas no consumidor são 0 (canais NOVOS do crescimento):
    a função não muda. Recusa canal herdado, bloco amarrado (residual/depthwise/cov_gate), inception e dense."""
    spec = res.child.spec.copy()
    L = spec.layers()[layer]
    n = len(spec.layers())
    if L.conv not in ("conv", "pointwise", "dwsep") or getattr(L, "dense_out", False) or L.residual != "none" \
            or getattr(L, "cov_gate", "none") != "none":
        raise MorphismError(f"prune: bloco {layer} ({L.conv}/{L.residual}) não suportado")
    if layer + 1 < n:
        Ln = spec.layers()[layer + 1]
        if Ln.residual in ("identity", "gated") or Ln.conv == "depthwise" or getattr(Ln, "cov_gate", "none") != "none":
            raise MorphismError("prune: consumidor amarrado (cin == cout)")
    rows = sorted({int(r) for r in rows})
    om = res.out_maps[layer]
    if om is None or any(om[r] >= 0 for r in rows):
        raise MorphismError("prune: só canais NOVOS (herdados nunca são podados)")
    C = int(L.channels)
    kept = [c for c in range(C) if c not in set(rows)]
    if not kept:
        raise MorphismError("prune: bloco ficaria vazio")
    L.channels = len(kept)
    op = OpRecord(op="prune_new_channels", where={"layer": int(layer)},
                  params={"removed": rows, "n_removed": len(rows)}, exact_claimed=True,
                  condition="canais novos com colunas 0 no consumidor")

    def fill_j(cb, pb, prior, in_local):
        bm, ex, nt = _fill_block(cb, pb, in_local, kept, prior)
        return bm, list(kept)
    return _rebuild(res, spec, layer, fill_j, 0, op)


# ─────────────────────────────────────────────────────────────────── v3.1 bloco 3: add_dense_layer (DenseNet)
@torch.no_grad()
def add_dense_layer(res: GrowthResult, at: int, growth: int, *, kernel: int = 3, norm: str = "batch",
                    act: str = "relu", new_filters: Optional[BranchInit] = None, seed: int = 0) -> GrowthResult:
    """Insere antes da camada ``at`` uma camada DenseNet H (conv k -> ``growth`` canais, norm, act, sem pool) cuja
    saída é cat(H(x), x): o consumidor (bloco ``at`` antigo ou o 1x1 da cabeça) passa a ver growth + C canais, com
    as colunas de H = 0 e as de x = as antigas (mapa [-1]*growth + 0..C-1) -> exato para qualquer entrada.
    ``new_filters(at, conv, rows)``: filtros SPiFiL de H sobre o prefixo herdado."""
    from ga_search.backbones.spec import LayerSpec
    M = res.child
    spec = M.spec.copy()
    layers = spec.layers()
    n = len(layers)
    if not 1 <= at < n:                          # no fim a cabeça leria a passagem (spec proíbe dense_out no último)
        raise MorphismError(f"add_dense_layer em {at}: use 1..{n - 1}")
    if at < n:
        Lc = layers[at]
        if Lc.residual in ("identity", "gated") or Lc.conv == "depthwise" or getattr(Lc, "cov_gate", "none") != "none":
            raise MorphismError(f"add_dense_layer: consumidor {at} exige cin == cout (residual/depthwise/cov_gate)")
    C = int(spec.block_out_channels(at - 1))
    newL = LayerSpec(channels=int(growth), kernel=int(kernel), conv="conv", norm=norm, activation=act, residual="none",
                     pool="none", init="dense_layer", flavor=layers[at - 1].flavor, dense_out=True)
    if at < n:
        s_, pos = spec.locate(at)
        spec.stages[s_].layers.insert(pos, newL)
    else:
        spec.stages[-1].layers.append(newL)
    torch.manual_seed(int(seed))
    child = Backbone(spec)
    masks: Dict[str, torch.Tensor] = {}
    in_cons = [-1] * int(growth) + list(range(C))
    exact, notes = True, []
    for j in range(n + 1):
        cb = child.blocks[j]
        if j < at:
            masks.update({f"blocks.{j}.{q}": v for q, v in _copy_block(cb, M.blocks[j], _prefixed(res.masks, f"blocks.{j}.")).items()})
        elif j == at:
            if new_filters is not None:
                new_filters(at, cb.conv, list(range(int(growth))))
            for q, v in cb.named_parameters():
                masks[f"blocks.{j}.{q}"] = torch.zeros_like(v, dtype=torch.bool)
        elif j == at + 1:
            pb = M.blocks[at]
            conv_local = list(range(_conv_rows(pb)))
            bm, ex, nt = _fill_block(cb, pb, in_cons, conv_local, _prefixed(res.masks, f"blocks.{at}."))
            exact &= ex
            notes += nt
            masks.update({f"blocks.{j}.{q}": v for q, v in bm.items()})
        else:
            masks.update({f"blocks.{j}.{q}": v for q, v in _copy_block(cb, M.blocks[j - 1], _prefixed(res.masks, f"blocks.{j - 1}.")).items()})
    if at == n:
        masks.update({f"head.{q}": v for q, v in _fill_head(child.head, M.head, in_cons, _prefixed(res.masks, "head.")).items()})
    else:
        child.head.load_state_dict(M.head.state_dict())
        masks.update({q: v.clone() for q, v in res.masks.items() if q.startswith("head.")})
    shift = lambda d: {(k if k < at else k + 1): v for k, v in d.items()}          # noqa: E731
    out = GrowthResult(parent=res.parent, child=child,
                       layer_map=list(res.layer_map[:at]) + ["dense_layer"] + list(res.layer_map[at:]),
                       out_maps=list(res.out_maps[:at]) + [None] + list(res.out_maps[at:]), masks=masks,
                       ops=list(res.ops), new_init={**shift(res.new_init), at: "dense_layer"},
                       skip_added={k if k < at else k + 1 for k in res.skip_added}, head_mode=res.head_mode)
    out.ops.append(OpRecord(op="add_dense_layer", where={"before": int(at)},
                            params={"growth": int(growth), "kernel": int(kernel), "norm": norm, "act": act,
                                    "init": "new_filters" if new_filters is not None else "pytorch"},
                            exact_claimed=bool(exact),
                            condition="saída = cat(H(x), x); colunas de H no consumidor = 0" +
                                      ("; " + "; ".join(sorted(set(notes))) if notes else "")))
    return out


OPS_V3 = {"inception_branch_add": inception_branch_add, "dense_concat": dense_concat,
          "replace_filters": replace_filters, "add_gated_residual": add_gated_residual,
          "add_dense_layer": add_dense_layer}
