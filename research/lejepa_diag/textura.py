"""Sensibilidade a textura: blur gaussiano sobre a imagem JA normalizada do pipeline, com dois
controles e cosseno embedding(original) vs embedding(perturbado).

* ``blur``      : gaussiano separavel, raio ceil(3 sigma), padding reflect; sigma=0 = identidade.
* ``noise``     : x + ruido gaussiano com a MESMA energia que o blur removeu,
                  ||n||^2 = ||x - blur_sigma(x)||^2 por imagem (degradacao geral, nao-textural).
* ``downup``    : area-downsample por f = 1 + 2 sigma e upsample bilinear de volta
                  (preserva estrutura global; sigma=0 = identidade).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
KINDS = ("blur", "noise", "downup")


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> Tensor:
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float64)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum()).to(dtype)


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Blur gaussiano depthwise em [B,C,H,W]; sigma <= 0 devolve x inalterado (identidade)."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r, C = (len(k) - 1) // 2, x.shape[1]
    H, W = x.shape[-2:]
    mode = "reflect" if r < min(H, W) else "replicate"
    y = F.pad(x, (r, r, 0, 0), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    y = F.pad(y, (0, 0, r, r), mode=mode)
    return F.conv2d(y, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def matched_noise(x: Tensor, sigma: float, seed: int = 0) -> Tensor:
    """x + ruido gaussiano com energia por imagem = energia removida pelo blur(sigma)."""
    if sigma <= 0:
        return x
    removed = (x - gaussian_blur(x, sigma)).flatten(1).pow(2).sum(1)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device, x.dtype)
    n = n * (removed / n.flatten(1).pow(2).sum(1).clamp_min(1e-12)).sqrt().view(-1, 1, 1, 1)
    return x + n


def downup_factor(sigma: float) -> float:
    return 1.0 + 2.0 * float(sigma)


def down_up(x: Tensor, sigma: float) -> Tensor:
    """Area-downsample por f = 1 + 2 sigma e bilinear de volta ao tamanho original."""
    if sigma <= 0:
        return x
    H, W = x.shape[-2:]
    f = downup_factor(sigma)
    small = F.interpolate(x, size=(max(1, round(H / f)), max(1, round(W / f))), mode="area")
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)


def perturb(kind: str, sigma: float, seed: int = 0):
    """Callable x -> x' para extract(transform=...). O seed do ruido muda por lote (contador)."""
    if kind not in KINDS:
        raise ValueError(f"kind deve ser um de {KINDS}")
    if kind == "blur":
        return lambda x: gaussian_blur(x, sigma)
    if kind == "downup":
        return lambda x: down_up(x, sigma)
    state = {"i": 0}

    def _noise(x):
        state["i"] += 1
        return matched_noise(x, sigma, seed * 1_000_003 + state["i"])
    return _noise


def cosine_rows(A, B) -> Tensor:
    """Cosseno linha a linha entre embeddings pareados [N,D] -> [N]."""
    A, B = torch.as_tensor(A).double(), torch.as_tensor(B).double()
    return F.cosine_similarity(A, B, dim=1, eps=1e-12)


def removed_energy_frac(x: Tensor, sigma: float) -> float:
    """Fracao media de energia removida pelo blur (||x - blur||^2 / ||x||^2)."""
    if sigma <= 0:
        return 0.0
    num = (x - gaussian_blur(x, sigma)).flatten(1).pow(2).sum(1)
    return float((num / x.flatten(1).pow(2).sum(1).clamp_min(1e-12)).mean())
