"""Readouts oficiais (sem redefinir) + leituras por camada.

* GAP = embedding oficial ``head(encoder(x))`` (extract.official_embedding).
* COV = ``ga_search.modehb.pooling_head.CovReadout`` OFICIAL: PCA k=64 do canal-covariancia
  ajustada SO nos pixels do treino (``fit`` recebe apenas X_train), sqrtm por eigh fp64,
  triangulo superior, signed-sqrt + L2 por grupo, concatenado com o head. O fitness historico
  (svm_on_demand --readout cov_head) usa StandardScaler: o caller registra COV (sem scaler) e
  COV+scaler como colunas separadas (ver svm_eval / run).
* Por camada: GAP do mapa do bloco e covariancia entre canais do bloco = o MESMO grupo do
  CovReadout (``CovReadout._finish`` aplicado a um bloco so), com k fixo (k_eff = min(k, C_l)).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ga_search.modehb.pooling_head import CovReadout

from .extract import fp32_strict

COV_K = 64           # run_search: svm_fitness.cov_k default
COV_FIT_BS = 32      # CovReadout.fit bs historico (define o sorteio de pixels)


class GAPHead(nn.Module):
    """Head substituto quando o checkpoint nao tem head: GAP do ultimo mapa."""

    def forward(self, z: Tensor) -> Tensor:
        return z.float().mean(dim=(2, 3))


def _prep(encoder, head, device):
    encoder = encoder.to(device).float().eval()
    head = (head if head is not None else GAPHead()).to(device).float().eval()
    return encoder, head


@torch.no_grad()
def fit_cov(encoder: nn.Module, x_train: Tensor, device="cpu", *, k: int = COV_K, seed: int = 0,
            micro: int | None = None, n_img: int = 160, n_px: int = 256) -> CovReadout:
    """CovReadout oficial ajustado SO em ``x_train`` (label-free). ``micro`` so muda memoria."""
    device = torch.device(device)
    encoder = encoder.to(device).float().eval()
    with fp32_strict():
        return CovReadout(k=k, n_img=n_img, n_px=n_px, seed=seed).fit(
            encoder, x_train, device, bs=COV_FIT_BS, micro=micro)


@torch.no_grad()
def cov_features(cr: CovReadout, encoder: nn.Module, head: nn.Module | None, X: Tensor,
                 device="cpu", batch_size: int = 8) -> Tensor:
    """Features COV oficiais (``CovReadout.embed``) em fp32 estrito -> [N, D] (cpu)."""
    device = torch.device(device)
    encoder, head = _prep(encoder, head, device)
    with fp32_strict():
        return cr.embed(encoder, head, X, batch_size, device).float()


def cov_readout(encoder, head, x_train: Tensor, x_val: Tensor, device="cpu", *, k: int = COV_K,
                seed: int = 0, batch_size: int = 8) -> tuple[Tensor, Tensor, CovReadout]:
    """Atalho: ajusta no treino e devolve (F_train, F_val, readout)."""
    cr = fit_cov(encoder, x_train, device, k=k, seed=seed, micro=batch_size)
    return (cov_features(cr, encoder, head, x_train, device, batch_size),
            cov_features(cr, encoder, head, x_val, device, batch_size), cr)


@torch.no_grad()
def layer_features(cr: CovReadout, encoder: nn.Module, X: Tensor, device="cpu",
                   batch_size: int = 8) -> dict[str, dict[str, Tensor]]:
    """Por bloco L1..Ln: {"gap": [N,C_l], "cov": [N, k_eff(k_eff+1)/2]}.

    ``cov`` = grupo do bloco no CovReadout oficial (PCA do treino em ``cr``, sqrtm fp64, triu,
    signed-sqrt + L2), calculado por ``cr._finish`` com head vazio (mesma definicao, bit a bit)."""
    device = torch.device(device)
    encoder = encoder.to(device).float().eval()
    out: dict[str, dict[str, list]] = {}
    with fp32_strict():
        for b in X.split(batch_size):
            outs = cr._blocks(encoder, b.to(device))
            covs = cr._covs(outs)
            eig = cr._eig(covs)
            empty = torch.empty(len(b), 0, device=device)
            for i, (o, C, e) in enumerate(zip(outs, covs, eig)):
                d = out.setdefault(f"L{i + 1}", {"gap": [], "cov": []})
                d["gap"].append(o.mean(dim=(2, 3)).float().cpu())
                d["cov"].append(cr._finish([C], [e], empty, cr._pn_l2).float().cpu())
    return {n: {k: torch.cat(v) for k, v in d.items()} for n, d in out.items()}


def layer_k_eff(cr: CovReadout) -> dict[str, int]:
    """k efetivo por bloco (min(k, C_l)); deve coincidir entre P0 e P1."""
    return {f"L{i + 1}": int(V.shape[1]) for i, (_mu, V) in enumerate(cr.proj)}
