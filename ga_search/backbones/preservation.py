"""Erro de preservação pai x filho ANTES de qualquer treino: ``eval()``, fp32, sem gradiente.

Métricas (por camada herdada, no mapa final do encoder e no embedding >= 1280):
* ``max_abs``: max |filho - pai| nos canais herdados;
* ``rel_l2``: ||filho - pai||_2 / ||pai||_2 (lote inteiro);
* ``cos_min``: menor cosseno por imagem entre os vetores achatados.

Veredito: ``exact`` se a operação foi declarada exata e max(rel_l2) <= tol; ``violated`` se foi
declarada exata e passou de tol (bug); se NÃO foi declarada exata: ``approx`` (rel_l2 <= 1e-2) ou
``not_preserving``. Os dados devem ser de treino/validação (nunca teste) e ficam descritos em ``data``.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ga_search.backbones.spec import PreservationRecord

APPROX_REL = 1e-2


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a, b = a.double(), b.double()
    d = (b - a)
    den = a.norm().item()
    rel = d.norm().item() / den if den > 0 else (0.0 if d.norm().item() == 0 else float("inf"))
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1, eps=1e-30)
    return {"max_abs": float(d.abs().max().item()) if d.numel() else 0.0, "rel_l2": float(rel),
            "cos_min": float(cos.min().item())}


def tensor_sha256(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def measure_preservation(parent, child, layer_map: Sequence[Union[int, str]], out_maps: Sequence[Optional[List[int]]],
                         x: torch.Tensor, *, tol: float = 1e-5, exact_claimed: bool = True,
                         data: Optional[dict] = None, compare_embedding: bool = True) -> PreservationRecord:
    pt, ct = parent.training, child.training
    parent.eval()
    child.eval()
    try:
        x = x.float()
        po, pe = parent.forward_all(x)
        co, ce = child.forward_all(x)
    finally:
        parent.train(pt)
        child.train(ct)
    per_layer, last = [], None
    n_parent = len(po)
    for c, p in enumerate(layer_map):
        if not isinstance(p, int) or out_maps[c] is None:
            continue
        pos = [j for j, m in enumerate(out_maps[c]) if m >= 0]
        src = [out_maps[c][j] for j in pos]
        if sorted(src) != list(range(po[p].shape[1])):
            raise ValueError(f"camada {c}: nem todos os canais do pai foram herdados (cut não suportado aqui)")
        m = _metrics(po[p][:, src], co[c][:, pos])
        rec = {"parent_index": int(p), "child_index": int(c), **m}
        per_layer.append(rec)
        if p == n_parent - 1:
            last = rec
    rec = PreservationRecord(data=dict(data or {}), exact_claimed=bool(exact_claimed), tolerance_rel=float(tol),
                             per_layer=per_layer)
    rec.data.setdefault("n_images", int(x.shape[0]))
    rec.data.setdefault("shape", list(x.shape))
    rec.data.setdefault("sha256", tensor_sha256(x))
    rec.data.setdefault("mode", "eval(), fp32, no_grad, antes de treino")
    if last is not None:
        rec.map_max_abs, rec.map_rel_l2, rec.map_cos_min = last["max_abs"], last["rel_l2"], last["cos_min"]
    if compare_embedding and pe.shape == ce.shape:
        e = _metrics(pe, ce)
        rec.emb_max_abs, rec.emb_rel_l2, rec.emb_cos_min = e["max_abs"], e["rel_l2"], e["cos_min"]
    rels = [r["rel_l2"] for r in per_layer] + ([rec.emb_rel_l2] if rec.emb_rel_l2 is not None else [])
    worst = max(rels) if rels else float("nan")
    if exact_claimed:
        rec.verdict = "exact" if worst <= tol else "violated"
    else:
        rec.verdict = "approx" if worst <= APPROX_REL else "not_preserving"
    return rec
