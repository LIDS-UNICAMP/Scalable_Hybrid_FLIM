"""growth v3 (1): cabeça de 2ª ordem FIXA (sem parâmetros treináveis) para a fitness e o teste.

Mesma conta de ``research/spifil_b0b4/readout_x/eval_backbone_head.py`` (cabeça ``so2_4096_pca``):

  z = Norm(pad_D([mu(Y), sigma(Y), triu(sqrtm(Cov(Y)))])),  Y = (X - mu_pca) @ V,

X = mapa do ÚLTIMO bloco como (posições x canais), V = os k primeiros componentes da PCA dos canais
ajustada SÓ no TREINO (pixels amostrados com semente por imagem: ``PixelMoments``, n_px=64, semente
PCA_SEED + índice do bloco), D = 4096 -> k = 88 (88+88+88*89/2 = 4092 + 4 zeros); Norm = signed-sqrt + L2.
Momentos / cov / sqrtm em fp64 na CPU (eigh). A SVM original (core.svm.fit_svm, C=100, sem scaler) roda
sobre z. Nada é ajustado em validação/teste.

A cabeça "faz parte do modelo": é um ``nn.Module`` só com BUFFERS (mu, V), ajustado por backbone (cada
alpha do WiSE tem o seu) e gravado ao lado do checkpoint (``so2_head.pt``, sha256 no result.json).
O SSL nunca a vê (não tem parâmetros).
"""
from __future__ import annotations

import hashlib
import io
import math
from typing import Iterable, Optional, Sequence

import torch
from torch import nn

SO2_K = {1280: 48, 4096: 88}
PCA_SEED = 0
READOUT = "so2_4096_pca"
VERSION = "so2_head/v1"


def so2_dim(k: int) -> int:
    return 2 * k + k * (k + 1) // 2


def k_for(D: int) -> int:
    if D in SO2_K:
        return SO2_K[D]
    k = 1
    while so2_dim(k + 1) <= D:
        k += 1
    return k


def pad_to(z, D: int):
    if z.shape[1] > D:
        raise ValueError(f"vetor {z.shape[1]} > {D}")
    return torch.cat([z, z.new_zeros(z.shape[0], D - z.shape[1])], 1)


def norm_ssqrt_l2(z):
    z = z.sign() * z.abs().sqrt()
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _sqrtm_psd(C):
    C = (C + C.transpose(1, 2)) / 2
    w, U = torch.linalg.eigh(C)
    return (U * w.clamp(min=0).sqrt()[:, None, :]) @ U.transpose(1, 2)


def _flat(x):
    """[B,C,H,W] -> [B,HW,C] fp64 CPU."""
    return x.detach().double().cpu().permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def so2_vector(fmap, mu, V, D: int, eps_rel: float = 0.0):
    """Leitura de 2a ordem de UM mapa: Norm(pad([mu, sigma, triu(sqrtm(Cov))])) -> [B, D] fp32.
    ``eps_rel`` > 0 (v3.1, second_order_pooling.md): Cov + eps*(tr Cov / k)*I antes do sqrtm (0 = readout_x)."""
    Y = (_flat(fmap) - mu) @ V
    m = Y.mean(1)
    Yc = Y - m[:, None, :]
    cov = Yc.transpose(1, 2) @ Yc / Y.shape[1]
    sd = cov.diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
    if eps_rel:
        k_ = cov.shape[-1]
        tr = cov.diagonal(dim1=1, dim2=2).sum(-1).clamp(min=0) / k_
        cov = cov + float(eps_rel) * tr[:, None, None] * torch.eye(k_, dtype=cov.dtype)
    S = _sqrtm_psd(cov)
    k = V.shape[1]
    iu = torch.triu_indices(k, k)
    z = torch.cat([m, sd, S[:, iu[0], iu[1]]], 1)
    return norm_ssqrt_l2(pad_to(z, D)).float()


class PixelMoments:
    """Momentos de canal em fp64 sobre pixels amostrados (semente por IMAGEM). Só imagens de TREINO."""

    def __init__(self, n_px: int = 64, seed: int = PCA_SEED):
        self.n_px, self.seed = int(n_px), int(seed)
        self.s1 = self.s2 = None
        self.n = 0

    def add(self, fmap, img_ids):
        f = _flat(fmap)
        rows = []
        for b, gid in enumerate(img_ids):
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(gid))
            hw = f.shape[1]
            idx = torch.randperm(hw, generator=g)[: min(self.n_px, hw)]
            rows.append(f[b, idx])
        P = torch.cat(rows)
        if self.s1 is None:
            self.s1 = torch.zeros(P.shape[1], dtype=torch.float64)
            self.s2 = torch.zeros(P.shape[1], P.shape[1], dtype=torch.float64)
        self.s1 += P.sum(0)
        self.s2 += P.T @ P
        self.n += P.shape[0]

    def pca(self, k: int):
        mu = self.s1 / self.n
        cov = (self.s2 - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        cov = (cov + cov.T) / 2
        _, V = torch.linalg.eigh(cov)
        V = V.flip(1)[:, : min(k, V.shape[1])]
        sgn = V.gather(0, V.abs().argmax(0, keepdim=True)).sign()
        return mu, (V * torch.where(sgn == 0, torch.ones_like(sgn), sgn)).contiguous()


class So2Head(nn.Module):
    """Cabeça de 2ª ordem fixa: buffers (mu [C], V [C, k]); ``forward(fmap) -> [B, D]``. Sem parâmetros."""

    def __init__(self, dim: int = 4096, n_px: int = 64, block_index: int = 0, eps_rel: float = 0.0):
        super().__init__()
        self.dim, self.n_px, self.block_index = int(dim), int(n_px), int(block_index)
        self.eps_rel = float(eps_rel)
        self.k = k_for(self.dim)
        self.register_buffer("mu", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("V", torch.zeros(0, 0, dtype=torch.float64))
        self._mom: Optional[PixelMoments] = None
        self.fitted = False

    # ---- ajuste (SÓ treino)
    def add_train(self, fmap, img_ids: Sequence[int]):
        if self._mom is None:
            self._mom = PixelMoments(self.n_px, PCA_SEED + self.block_index)
        self._mom.add(fmap, img_ids)

    def finish_fit(self):
        mu, V = self._mom.pca(self.k)
        self.mu, self.V = mu, V
        self.n_fit_pixels = int(self._mom.n)
        self._mom = None
        self.fitted = True
        return self

    def forward(self, fmap):
        if not self.fitted:
            raise RuntimeError("So2Head.finish_fit (treino) antes de ler")
        return so2_vector(fmap, self.mu, self.V, self.dim, self.eps_rel)

    def meta(self) -> dict:
        return {"version": VERSION, "readout": READOUT if self.dim == 4096 else f"so2_{self.dim}_pca",
                "dim": self.dim, "k": self.k, "n_px": self.n_px, "pca_seed": PCA_SEED + self.block_index,
                "block_index": self.block_index, "norm": "signed_sqrt+l2", "fit": "train_only",
                "trainable_params": 0, **({"eps_rel": self.eps_rel} if self.eps_rel else {})}

    def payload(self) -> dict:
        return {"meta": self.meta(), "mu": self.mu.clone(), "V": self.V.clone()}

    def sha256(self) -> str:
        buf = io.BytesIO()
        torch.save({"mu": self.mu, "V": self.V}, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @classmethod
    def from_payload(cls, p: dict) -> "So2Head":
        m = p["meta"]
        h = cls(m["dim"], m["n_px"], m["block_index"], float(m.get("eps_rel", 0.0)))
        h.mu, h.V = p["mu"], p["V"]
        h.fitted = True
        return h


def spec_entry(dim: int = 4096, n_px: int = 64) -> dict:
    """O que entra na fitness_spec (muda a fitness_sig)."""
    return {"readout": READOUT if dim == 4096 else f"so2_{dim}_pca", "dim": int(dim), "k": k_for(dim),
            "n_px": int(n_px), "pca_seed": "PCA_SEED+bloco", "pca_fit": "train_only", "norm": "signed_sqrt+l2",
            "map": "ultimo_bloco", "version": VERSION, "precision": "fp64_cpu_moments"}


@torch.no_grad()
def fit_and_read(encoder: nn.Module, train_x: torch.Tensor, others: Iterable[torch.Tensor], device, *,
                 batch_size: int, dim: int = 4096, n_px: int = 64, block_index: int = 0, tta: int = 1,
                 gap_head: Optional[nn.Module] = None, head: Optional["So2Head"] = None, eps_rel: float = 0.0):
    """Passe 1: PCA só no treino (ids = posição em ``train_x``). Passe 2: z do treino e de cada ``others``.
    ``tta`` > 1: média de z sobre as vistas de ``pooling_head.tta_transforms`` (só canal de teste).
    ``gap_head``: também devolve head(encoder(x)) (coluna aux GAP, tta 1).
    -> (So2Head, [z_train, z_other...], [gap_train, gap_other...] | None)."""
    from ga_search.modehb.pooling_head import tta_transforms
    encoder.eval()
    if gap_head is not None:
        gap_head.eval()
    bs = int(batch_size)
    if head is None:                     # ``head`` dado (cabeça salva com o checkpoint): sem re-ajuste
        head = So2Head(dim, n_px, block_index, eps_rel)
        for i0 in range(0, len(train_x), bs):
            xb = train_x[i0:i0 + bs].to(device).float()
            head.add_train(encoder(xb), list(range(i0, i0 + xb.shape[0])))
        head.finish_fit()
    views = tta_transforms(int(tta))
    outs, gaps = [], ([] if gap_head is not None else None)
    for X in [train_x] + list(others):
        acc, gacc = [], []
        for i0 in range(0, len(X), bs):
            xb = X[i0:i0 + bs].to(device).float()
            if len(views) == 1:
                f = encoder(xb)
                acc.append(head(f))
                if gap_head is not None:
                    gacc.append(gap_head(f).float().cpu())
            else:
                acc.append(torch.stack([head(encoder(t(xb))) for t in views]).mean(0))
                if gap_head is not None:
                    gacc.append(gap_head(encoder(xb)).float().cpu())
        outs.append(torch.cat(acc))
        if gaps is not None:
            gaps.append(torch.cat(gacc))
    return head, outs, gaps
