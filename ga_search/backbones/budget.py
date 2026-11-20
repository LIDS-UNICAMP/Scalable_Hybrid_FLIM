"""Faixas B0–B4, teto duro de 5,5M e reparo da FORMA da cabeça (nunca da dimensão).

Regra (MISSION.md, decisão do usuário 17/11): o embedding tem dimensão >= 1280 e a cabeça conta no
orçamento. Quando encoder + cabeça não cabem em Bk, o reparo troca só a forma da cabeça, numa ordem
fixa e registrada, mantendo ``embed_dim``:

    forma atual -> meanstd (1x1 -> D/2, mu||sigma) -> moments (1x1 -> D/4, mu||sigma||skew||kurt)
    -> lowrank (1x1 C->r sem bias, então 1x1 r->largura do pooling) com r decrescente.

O reparo gera um NOVO spec (novo hash) e um log; nunca altera o spec de entrada. Se nenhuma forma
cabe, levanta ``BudgetError``: o encoder precisa encolher, e isso não é decisão do reparo.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

from ga_search.backbones.build import BudgetError, param_count_spec
from ga_search.backbones.spec import BUDGETS, HARD_CAP, BackboneSpec, HeadSpec, band_of

LOWRANK_RANKS = (512, 384, 256, 192, 128, 96, 64, 48, 32)


def cap_of(budget_class: str) -> int:
    return min(int(BUDGETS[budget_class]), HARD_CAP)


def head_candidates(head: HeadSpec, ranks: Sequence[int] = LOWRANK_RANKS) -> List[HeadSpec]:
    """Formas na ordem de preferência; todas com o mesmo ``embed_dim`` (>= 1280)."""
    D = int(head.embed_dim)
    out = [copy.deepcopy(head)]
    for pooling in ("meanstd", "moments"):
        if head.proj == "full" and head.pooling == pooling:
            continue
        out.append(HeadSpec(embed_dim=D, pooling=pooling, proj="full", kurtosis=True, act=head.act, bn=head.bn))
    for pooling in ("gem", "meanstd", "moments"):
        for r in ranks:
            out.append(HeadSpec(embed_dim=D, pooling=pooling, proj="lowrank", rank=int(r), kurtosis=True,
                                act=head.act, bn=head.bn))
    seen, uniq = set(), []
    for h in out:
        key = (h.pooling, h.proj, h.rank, h.kurtosis)
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def fits(spec: BackboneSpec, budget_class: str) -> Tuple[bool, int]:
    e, h = param_count_spec(spec)
    return e + h <= cap_of(budget_class), e + h


def repair_head(spec: BackboneSpec, budget_class: Optional[str] = None) -> Tuple[BackboneSpec, Dict]:
    """Primeira forma de cabeça (ordem de ``head_candidates``) que faz o spec caber em ``budget_class``."""
    bc = budget_class or spec.budget_class
    if bc is None:
        raise ValueError("informe budget_class")
    tried = []
    for h in head_candidates(spec.head):
        cand = spec.copy()
        cand.head = h
        ok, n = fits(cand, bc)
        tried.append({"head": h.label, "proj": h.proj, "params_total": n, "fits": ok})
        if ok:
            cand.budget_class = bc
            changed = h.label != spec.head.label or h.proj != spec.head.proj
            log = {"budget_class": bc, "cap": cap_of(bc), "repaired": changed, "from_head": spec.head.label,
                   "to_head": h.label, "embed_dim": int(h.embed_dim), "params_total": n, "tried": tried}
            if changed:
                cand.provenance.notes = list(cand.provenance.notes) + [
                    f"head_repair {spec.head.label}->{h.label} para caber em {bc} (embed_dim mantido)"]
            return cand, log
    raise BudgetError(f"nenhuma forma de cabeça com embed_dim={spec.head.embed_dim} cabe em {bc}; "
                      f"o encoder precisa encolher (tentativas: {[t['head'] for t in tried]})")


def repair_family_head(specs: Sequence[BackboneSpec], budget_classes: Sequence[str]) -> Tuple[List[BackboneSpec], Dict]:
    """Uma única forma de cabeça para toda a cadeia (necessário para ``head_widen`` exato entre Bk e Bk+1):
    a primeira de ``head_candidates`` que faz TODOS os membros caberem em suas classes."""
    tried = []
    for h in head_candidates(specs[0].head):
        out, totals, ok_all = [], [], True
        for s, bc in zip(specs, budget_classes):
            c = s.copy()
            c.head = h
            ok, n = fits(c, bc)
            totals.append(n)
            ok_all &= ok
            c.budget_class = bc
            out.append(c)
        tried.append({"head": h.label, "totals": totals, "fits_all": ok_all})
        if ok_all:
            return out, {"head": h.label, "from_head": specs[0].head.label, "repaired": h.label != specs[0].head.label,
                         "tried": tried}
    raise BudgetError("nenhuma forma de cabeça comum cabe em toda a família")


def describe_bands() -> List[Dict]:
    lo, rows = 0, []
    for k, cap in BUDGETS.items():
        rows.append({"class": k, "min_exclusive": lo, "max_inclusive": cap})
        lo = cap
    return rows


__all__ = ["cap_of", "head_candidates", "fits", "repair_head", "repair_family_head", "describe_bands", "band_of",
           "BudgetError", "LOWRANK_RANKS"]
