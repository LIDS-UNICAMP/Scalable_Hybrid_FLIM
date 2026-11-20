"""Família de referência PLANA e CONTIDA B0 -> B4 (prep_literatura_e_specs.md §4), construída por morfismo exato.

Template: stem conv k5 + max-pool 2 (224 -> 112); 3 estágios conv k3 + ReLU, cada um fechado por
max-pool 2 (112 -> 56 -> 28 -> 14); norm none; cabeça 1280 no mapa 14x14.
Cada Bk+1 contém Bk (larguras e nº de camadas por estágio não decrescem), então cada aresta é
``deepen_identity`` (camadas novas inseridas ANTES da camada que faz pool do estágio, como no PNAS-CS)
seguido de ``widen`` (zero-pad do consumidor) + ``head_widen`` (colunas zero): toda a cadeia é exata.

Diferença registrada em relação à tabela §4: o pool do stem é MaxPool(2, 2) do ``ConvBlock`` (não 3x3 s2
pad 1); mesmas resoluções, params iguais, MACs iguais (o pooling não entra nos MACs).

Cabeça: com GeM 1280 o B1 da tabela (0,882M) passa de 800k. ``repair_family_head`` escolhe UMA forma de
cabeça para a cadeia toda (necessário para o ``head_widen`` exato entre Bk e Bk+1): a primeira, na ordem
gem -> meanstd -> moments -> lowrank, em que os cinco membros cabem. Resultado: meanstd 1280.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ga_search.backbones.budget import repair_family_head
from ga_search.backbones.build import Backbone, build_backbone
from ga_search.backbones.morphisms import GrowthResult, deepen_identity, finalize, start, widen
from ga_search.backbones.spec import (BUDGETS, BackboneSpec, GrowthSpec, HeadSpec, InputSpec, LayerSpec,
                                      StageSpec)

REF_FLAT: Dict[str, Dict[str, object]] = {
    "B0": {"stem": 24, "local": [48], "intermediate": [80], "abstract": [112]},
    "B1": {"stem": 32, "local": [64], "intermediate": [112, 112], "abstract": [176, 176]},
    "B2": {"stem": 32, "local": [64, 64], "intermediate": [128, 128], "abstract": [256, 256]},
    "B3": {"stem": 40, "local": [80, 80], "intermediate": [160, 160, 160], "abstract": [288, 288, 288]},
    "B4": {"stem": 48, "local": [96, 96], "intermediate": [208, 208, 208], "abstract": [400, 400, 400]},
}


def flat_spec(widths: Dict[str, object], *, head: Optional[HeadSpec] = None, dataset: str = "reference",
              budget_class: Optional[str] = None, init: str = "spifil0") -> BackboneSpec:
    stem = LayerSpec(channels=int(widths["stem"]), kernel=5, pool="max", init=init)
    stages = []
    for name in ("local", "intermediate", "abstract"):
        ws = list(widths.get(name, []))
        if ws:
            stages.append(StageSpec(name=name, layers=[
                LayerSpec(channels=int(w), kernel=3, pool="max" if i == len(ws) - 1 else "none", init=init)
                for i, w in enumerate(ws)]))
    return BackboneSpec(stem=stem, stages=stages, head=head or HeadSpec(embed_dim=1280, pooling="gem"),
                        input=InputSpec(), dataset=dataset, budget_class=budget_class).normalize()


def reference_family_specs(*, head: Optional[HeadSpec] = None, dataset: str = "reference",
                           classes: Sequence[str] = tuple(REF_FLAT)) -> Tuple[List[BackboneSpec], Dict]:
    specs = [flat_spec(REF_FLAT[k], head=head, dataset=dataset) for k in classes]
    return repair_family_head(specs, list(classes))


def edge_plan(parent: BackboneSpec, child: BackboneSpec) -> List[dict]:
    """Operações (em ordem) que levam o pai plano ao filho plano contido: deepen_identity e depois widen."""
    if parent.head.label != child.head.label or parent.head.proj != child.head.proj:
        raise ValueError("cadeia exata exige a mesma forma de cabeça")
    pst = {s.name: len(s.layers) for s in parent.stages}
    cst = {s.name: len(s.layers) for s in child.stages}
    if list(pst) != list(cst) or any(cst[k] < pst[k] for k in pst):
        raise ValueError("filho não contém o pai (estágios/profundidade)")
    ops, depth = [], dict(pst)
    for name in pst:
        while depth[name] < cst[name]:
            # índice plano da camada que faz pool do estágio ``name`` no modelo corrente
            flat = 1
            for nm in pst:
                if nm == name:
                    break
                flat += depth[nm]
            at = flat + depth[name] - 1
            ops.append({"op": "deepen_identity", "at": at})
            depth[name] += 1
    widths = {j: int(L.channels) for j, L in enumerate(child.layers())}
    ops.append({"op": "widen", "widths": widths})
    return ops


def grow_edge(parent: Backbone, child_target: BackboneSpec, x: Optional[torch.Tensor] = None, *, seed: int = 0,
              data: Optional[dict] = None, new_filters=None) -> Tuple[Backbone, GrowthSpec, GrowthResult]:
    res = start(parent)
    for o in edge_plan(parent.spec, child_target):
        if o["op"] == "deepen_identity":
            res = deepen_identity(res, o["at"])
        else:
            res = widen(res, o["widths"], seed=seed, new_filters=new_filters)
    if res.child.spec.arch_hash() != child_target.arch_hash():
        raise AssertionError(f"filho {res.child.spec.hash12} != alvo {child_target.hash12}")
    res.child.spec.dataset = child_target.dataset
    g = finalize(res, x, data=data, child_budget_class=child_target.budget_class)
    return res.child, g, res


def build_reference_chain(x: Optional[torch.Tensor] = None, *, seed: int = 0, dataset: str = "reference",
                          data: Optional[dict] = None, classes: Sequence[str] = tuple(REF_FLAT)):
    """B0 (init aleatória registrada) -> B1 -> ... por morfismo exato. Devolve (modelos, growth_specs, log_reparo)."""
    specs, repair_log = reference_family_specs(dataset=dataset, classes=classes)
    torch.manual_seed(int(seed))
    m0 = build_backbone(specs[0], seed=seed)
    m0.spec.provenance.lineage_id = f"L-size/{dataset}/ref_flat"
    models, edges = [m0], []
    for k in range(1, len(specs)):
        child, g, _ = grow_edge(models[-1], specs[k], x, seed=seed + k, data=data)
        child.spec.provenance.lineage_id = m0.spec.provenance.lineage_id
        models.append(child)
        edges.append(g)
    return models, edges, repair_log


__all__ = ["REF_FLAT", "flat_spec", "reference_family_specs", "edge_plan", "grow_edge", "build_reference_chain",
           "BUDGETS"]
