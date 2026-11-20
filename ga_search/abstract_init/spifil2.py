"""SPiFiL-2 — inicialização de kernels invariante entre vistas (PROPOSTA NÃO TESTADA cientificamente).

Desenho: research/spifil_b0b4/agent2/prep_literatura_e_desenho.md §5.2 (e §3 interface, §7, §8).
Status: implementação isolada. Nada aqui afirma que o método funciona; é um braço de ablação.

Ideia
-----
Entre K candidatos (protótipos de patches densos de TREINO em espaço branqueado), escolher os
C_out filtros cuja resposta é estável e repetível entre duas vistas aumentadas da mesma imagem,
em posições correspondentes conhecidas pela geometria θ da augmentation (à la VICRegL, casamento
por posição), com consistência de atribuição a protótipos sob equipartição de Sinkhorn (à la
SwAV) e diversidade (NMS + termo de distância). Refino local opcional (VICRegL/SwAV local) que
só mexe numa cópia temporária dos pesos do bloco novo; o prefixo fica congelado e nenhum
parâmetro persistente é criado.

API pública
-----------
- ``AUG_POLICIES`` / ``AugPolicy``          políticas de augmentation por domínio (variante registrada).
- ``make_views_with_theta(x, V, cfg, seed)``  réplica bit a bit de ``ssl_trainers.make_views`` que
                                              devolve também θ (N,2,3) por vista.
- ``correspondences(theta_a, theta_b, grid)`` pares (imagem, posição em a, posição em b) pela geometria.
- ``sinkhorn(scores, eps, iters)``           códigos equipartidos (SwAV).
- ``Spifil2Invariant().fit(prefix, train_data, block_spec, context)`` -> (weights, bias, provenance).
- ``VARIANTS``                                braços registrados (s2, s2_noinv, s2_sup, s2_ref_*...).

Convenções de geometria
-----------------------
``F.affine_grid(theta, align_corners=False)``: a célula (i, j) de uma grade H×W tem centro
normalizado u = ((2j+1)/W − 1, (2i+1)/H − 1); a coordenada na imagem é u_img = A·u + t, com
A = θ[:, :, :2], t = θ[:, :, 2]. A grade de features do prefixo é tratada como cobrindo [-1, 1]
uniformemente (aproximação registrada ``feature_grid_uniform``; exata quando o prefixo preserva
H×W, aproximada com pooling).
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, fields
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Interface publicada pelo Agente 2a (research/spifil_b0b4/agent2/interface/base.py,
# instalada como ga_search/abstract_init/base.py). Dependência dura: o patch do 2a vem antes.
from ga_search.abstract_init.base import (
    InfeasibleCandidate, InitBudget, InitContext, KernelInitializer, TrainData, check_prefix_unchanged,
    check_train_only, make_provenance, prefix_fingerprint, register)

__all__ = [
    "AugPolicy", "AUG_POLICIES", "make_views_with_theta", "Correspondences", "correspondences",
    "sinkhorn", "Spifil2Invariant", "VARIANTS", "DEFAULT_CONFIG", "InfeasibleCandidate",
    "TrainData", "InitBudget", "InitContext", "dc_fraction",
]

METHOD = "spifil2"
VERSION = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vistas com θ (réplica de ga_search.modehb.ssl_trainers.make_views)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AugPolicy:
    """Superconjunto de ``ssl_trainers.AugConfig``. Com ``rot90=False`` a amostragem é idêntica
    (mesma ordem de sorteios) à de ``make_views``; ``rot90`` sorteia k·90° DEPOIS de tudo.
    ``max_bg_frac``: descarta (só na estatística) imagens cuja vista tem fração de fundo
    (canal 0 < ``bg_thr``) acima do limite — parasitas "pre" com fundo preto."""
    crop_scale: tuple = (0.3, 1.0)
    crop_ratio: tuple = (3 / 4, 4 / 3)
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    channel_jitter: float = 0.05
    out_size: Optional[int] = None
    rot90: bool = False
    max_bg_frac: Optional[float] = None
    bg_thr: float = 0.02

    def aug_config_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("crop_scale", "crop_ratio", "hflip_p", "vflip_p",
                                             "brightness", "contrast", "channel_jitter", "out_size")}


# Política por domínio = VARIANTE REGISTRADA (hipóteses do desenho §5.2, não verdades).
# "aug_default" é exatamente a política de produção (AugConfig()).
AUG_POLICIES: dict[str, AugPolicy] = {
    "aug_default": AugPolicy(),
    "none": AugPolicy(crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), hflip_p=0.0, vflip_p=0.0,
                      brightness=0.0, contrast=0.0, channel_jitter=0.0),
    "rs": AugPolicy(crop_scale=(0.5, 1.0), hflip_p=0.5, vflip_p=0.5, rot90=True,
                    brightness=0.2, contrast=0.2, channel_jitter=0.05),
    "aid": AugPolicy(crop_scale=(0.5, 1.0), hflip_p=0.5, vflip_p=0.5, rot90=True,
                     brightness=0.2, contrast=0.2, channel_jitter=0.05),
    "flowers": AugPolicy(crop_scale=(0.3, 1.0), hflip_p=0.5, vflip_p=0.0, rot90=False,
                         brightness=0.1, contrast=0.1, channel_jitter=0.0),
    "ham10000": AugPolicy(crop_scale=(0.6, 1.0), hflip_p=0.5, vflip_p=0.5, rot90=True,
                          brightness=0.1, contrast=0.1, channel_jitter=0.02),
    "parasites_raw": AugPolicy(crop_scale=(0.7, 1.0), hflip_p=0.5, vflip_p=0.5, rot90=True,
                               brightness=0.1, contrast=0.1, channel_jitter=0.1),
    "parasites_pre": AugPolicy(crop_scale=(0.7, 1.0), hflip_p=0.5, vflip_p=0.5, rot90=True,
                               brightness=0.1, contrast=0.1, channel_jitter=0.1, max_bg_frac=0.9),
}


def _as_policy(cfg) -> AugPolicy:
    if cfg is None:
        return AugPolicy()
    if isinstance(cfg, AugPolicy):
        return cfg
    if isinstance(cfg, str):
        if cfg not in AUG_POLICIES:
            raise ValueError(f"aug policy desconhecida {cfg!r}; registradas: {sorted(AUG_POLICIES)}")
        return AUG_POLICIES[cfg]
    if isinstance(cfg, dict):
        return AugPolicy(**cfg)
    # ssl_trainers.AugConfig (ou similar)
    return AugPolicy(**{f.name: getattr(cfg, f.name) for f in fields(AugPolicy) if hasattr(cfg, f.name)})


def make_views_with_theta(x: Tensor, V: int, cfg=None, seed: Optional[int] = None,
                          generator: Optional[torch.Generator] = None):
    """Mesmo contrato de ``make_views`` + θ: devolve (views, thetas, photometric).

    ``views[v]``: [N,C,S,S] em [0,1]; ``thetas[v]``: [N,2,3] (saída normalizada -> entrada);
    ``photometric[v]``: dict con/bri/ch. Com política sem ``rot90`` (ex.: ``aug_default``) as vistas
    são iguais bit a bit às de ``ssl_trainers.make_views`` com a mesma seed (teste §8.1-9).
    """
    cfg = _as_policy(cfg)
    if generator is None:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()
    N, C, H, W = x.shape
    S = cfg.out_size or H
    dev, dt = x.device, x.dtype

    def rnd(*shape):
        return torch.rand(*shape, generator=generator).to(dev, dt)

    views, thetas, photo = [], [], []
    for _ in range(V):
        area = cfg.crop_scale[0] + rnd(N) * (cfg.crop_scale[1] - cfg.crop_scale[0])
        lr0, lr1 = math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])
        ratio = torch.exp(lr0 + rnd(N) * (lr1 - lr0))
        sw = torch.sqrt(area * ratio).clamp(max=1.0)
        sh = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = (rnd(N) * 2 - 1) * (1 - sw)
        cy = (rnd(N) * 2 - 1) * (1 - sh)
        fx = torch.where(rnd(N) < cfg.hflip_p, -1.0, 1.0).to(dt)
        fy = torch.where(rnd(N) < cfg.vflip_p, -1.0, 1.0).to(dt)
        theta = torch.zeros(N, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (N, C, S, S), align_corners=False)
        v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        mean = v.mean(dim=(1, 2, 3), keepdim=True)
        con = 1 + (rnd(N, 1, 1, 1) * 2 - 1) * cfg.contrast
        bri = (rnd(N, 1, 1, 1) * 2 - 1) * cfg.brightness
        ch = (rnd(N, C, 1, 1) * 2 - 1) * cfg.channel_jitter
        v = (v - mean) * con + mean + bri + ch
        v = v.clamp(0.0, 1.0)
        if cfg.rot90:
            # Extensão registrada: rotação k·90° da grade de SAÍDA (A = D·R), sorteada após os
            # demais sorteios (não altera a sequência de make_views para rot90=False). Exata na
            # grade: rot90 da vista == reamostrar com θ' = θ·R.
            kr = torch.floor(rnd(N) * 4).clamp(max=3).to(torch.long)
            v = torch.stack([torch.rot90(v[i], int(kr[i]), dims=(1, 2)) for i in range(N)])
            theta = _compose_rot90(theta, kr)
        views.append(v)
        thetas.append(theta)
        photo.append({"contrast": con.flatten(), "brightness": bri.flatten(), "channel": ch.flatten(1)})
    return views, thetas, photo


def _compose_rot90(theta: Tensor, k: Tensor) -> Tensor:
    """θ' tal que amostrar com θ' == torch.rot90(amostra com θ, k, dims=(H, W)).

    rot90 com k=1 (dims=(1,2), sentido anti-horário na exibição): out[i', j'] = in[j', W-1-i'].
    Em coordenadas normalizadas (x=col, y=lin): (x_in, y_in) = (−y_out, x_out) para k=1.
    """
    th = theta.clone()
    A, t = theta[:, :, :2], theta[:, :, 2:]
    Rs = []
    for kk in k.tolist():
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][kk % 4]
        # mapeamento saída->entrada do rot90: u_in = R u_out
        Rs.append(torch.tensor([[c, -s], [s, c]], dtype=theta.dtype))
    R = torch.stack(Rs).to(theta.device)
    th[:, :, :2] = A @ R
    th[:, :, 2:] = t
    return th


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correspondências locais pela geometria
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Correspondences:
    img: Tensor      # (n,) índice da imagem no lote
    pa: Tensor       # (n,) posição achatada na grade de a
    pb: Tensor       # (n,) posição achatada na grade de b (célula mais próxima)
    ib: Tensor       # (n,) linha em b
    jb: Tensor       # (n,) coluna em b
    err: Tensor      # (n,) distância (em células) entre o ponto exato e a célula escolhida
    grid_a: tuple
    grid_b: tuple

    def __len__(self):
        return int(self.img.numel())

    def subset(self, mask: Tensor) -> "Correspondences":
        return Correspondences(self.img[mask], self.pa[mask], self.pb[mask], self.ib[mask],
                               self.jb[mask], self.err[mask], self.grid_a, self.grid_b)


def _cell_centers(H: int, W: int) -> Tensor:
    ys = (2 * torch.arange(H, dtype=torch.float64) + 1) / H - 1
    xs = (2 * torch.arange(W, dtype=torch.float64) + 1) / W - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2) como (x, y)


def correspondences(theta_a: Tensor, theta_b: Tensor, grid_a: Sequence[int],
                    grid_b: Optional[Sequence[int]] = None, keep: Optional[Tensor] = None,
                    margin_cells: float = 1.0) -> Correspondences:
    """Pares de posições correspondentes entre a vista a e a vista b da MESMA imagem.

    u_img = A_a·u_a + t_a;  u_b = A_b⁻¹(u_img − t_b). Mantém o par se u_img ∈ [-1,1]² e
    ‖u_b‖∞ ≤ 1 − margin_cells/H_b (por eixo); q = célula mais próxima em b (tolerância 1 célula).
    ``keep`` (N,) bool descarta imagens (ex.: vista com fundo demais).
    """
    Ha, Wa = int(grid_a[0]), int(grid_a[1])
    Hb, Wb = (Ha, Wa) if grid_b is None else (int(grid_b[0]), int(grid_b[1]))
    ta, tb = theta_a.detach().to("cpu", torch.float64), theta_b.detach().to("cpu", torch.float64)
    N = ta.shape[0]
    ua = _cell_centers(Ha, Wa)                                        # (P, 2)
    u = torch.einsum("nij,pj->npi", ta[:, :, :2], ua) + ta[:, None, :, 2]
    Ab_inv = torch.linalg.inv(tb[:, :, :2])
    ub = torch.einsum("nij,npj->npi", Ab_inv, u - tb[:, None, :, 2])  # (N, P, 2)
    tol = 1e-9
    lim = torch.tensor([1 - margin_cells / Wb, 1 - margin_cells / Hb], dtype=torch.float64)
    valid = (ub.abs() <= lim + tol).all(-1) & (u.abs() <= 1 + tol).all(-1)
    if keep is not None:
        valid &= keep.to("cpu", torch.bool)[:, None]
    jf = ((ub[..., 0] + 1) * Wb - 1) / 2
    if_ = ((ub[..., 1] + 1) * Hb - 1) / 2
    jb = torch.round(jf).clamp(0, Wb - 1)
    ib = torch.round(if_).clamp(0, Hb - 1)
    err = torch.sqrt((jf - jb) ** 2 + (if_ - ib) ** 2)
    n_idx, p_idx = torch.nonzero(valid, as_tuple=True)
    ib_s, jb_s = ib[n_idx, p_idx].long(), jb[n_idx, p_idx].long()
    return Correspondences(img=n_idx, pa=p_idx, pb=ib_s * Wb + jb_s, ib=ib_s, jb=jb_s,
                           err=err[n_idx, p_idx].float(), grid_a=(Ha, Wa), grid_b=(Hb, Wb))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sinkhorn (SwAV) e utilidades
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sinkhorn(scores: Tensor, eps: float = 0.05, iters: int = 3) -> Tensor:
    """Códigos equipartidos q (n, K) como no SwAV (log-domínio, estável). Cada linha soma 1;
    as colunas tendem a massa n/K (exata no limite de iterações)."""
    s = scores.double() / eps
    n, K = s.shape
    logQ = s - torch.logsumexp(s.reshape(-1), 0)
    for _ in range(int(iters)):
        logQ = logQ - torch.logsumexp(logQ, dim=0, keepdim=True) - math.log(K)   # colunas: 1/K
        logQ = logQ - torch.logsumexp(logQ, dim=1, keepdim=True) - math.log(n)   # linhas: 1/n
    return (logQ + math.log(n)).exp().to(scores.dtype)


def _z(v: Tensor) -> Tensor:
    v = v.double()
    sd = v.std(unbiased=False)
    if not torch.isfinite(sd) or sd < 1e-12:
        return torch.zeros_like(v)
    return (v - v.mean()) / sd


def _sha256_tensor(t: Tensor) -> str:
    return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()


def dc_fraction(w: Tensor, k: int) -> Tensor:
    """Fração DC por filtro: k²·‖média espacial de w‖² / ‖w‖² (§7). w: (n, C·k·k)."""
    n = w.shape[0]
    ws = w.reshape(n, -1, k * k)
    num = (k * k) * (ws.mean(-1) ** 2).sum(-1)
    return num / (w.pow(2).sum(-1) + 1e-20)


def _erank_from_eig(eig: Tensor) -> float:
    s = eig.clamp(min=0).sqrt()
    p = s / (s.sum() + 1e-20)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuração e variantes registradas
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    "aug_policy": "aug_default",   # nome em AUG_POLICIES ou dict AugPolicy
    "views": 2,                    # V (só o par (0,1) entra nas estatísticas nesta versão)
    "patches_per_image": 400,
    "patch_norm": "none",          # none | mean_only
    "whiten": "zca",               # zca | none
    "eps_zca": 0.01,               # × média dos autovalores (igual a GrowOneLayer)
    "cand": "kmeans",              # kmeans | patches (FPS em patches branqueados)
    "candidate_fn": None,          # gancho: fn(X_fit, K, gen) -> W (K', D) no espaço de entrada
    "cand_mult": 3,                # K = cand_mult·C_out
    "kmeans_iters": 30,
    "holdout_frac": 0.25,          # patches para unit_response_std (fora do ajuste)
    "unit_std": True,              # divide cada filtro pelo desvio da resposta (registrado)
    "tau": 0.1, "eps_sk": 0.05, "sk_iters": 3,
    "rep_m": 8, "rep_tol": 1,
    "lam_stab": 1.0, "lam_rep": 0.5, "lam_agree": 1.0, "lam_mass": 0.5,
    "lam_fisher": 1.0,             # só com label_policy == "supervised"
    "lam_dc": 0.0,                 # penalidade de energia DC (0 = só medir)
    "lam_div": 0.5,                # bônus de diversidade (1 − max|cos| com escolhidos)
    "red_max": 0.9, "stab_floor": 0.2,
    "chunk_images": 16,
    "refine": {"steps": 0, "lr": 1e-3, "loss": "vicregl_local", "batch_images": 8,
               "inv": 25.0, "var": 25.0, "cov": 1.0},
}

# Braços registrados (nome -> overrides). "s2_noinv" é o CONTROLE: mesma pipeline sem nenhum
# termo de invariância (só massa + diversidade) — invariância não é suposta boa a priori.
VARIANTS: dict[str, dict] = {
    "s2": {},
    "s2_noinv": {"lam_stab": 0.0, "lam_rep": 0.0, "lam_agree": 0.0, "stab_floor": -1.0},
    "s2_aug_domain": {"aug_policy": "<domínio>"},   # preencher com a chave de AUG_POLICIES
    "s2_nounitstd": {"unit_std": False},
    "s2_patches": {"cand": "patches"},
    "s2_dcpen": {"lam_dc": 0.5},
    "s2_sup": {"label_policy": "supervised"},
    "s2_ref_vicregl": {"refine": {"steps": 300, "loss": "vicregl_local"}},
    "s2_ref_swav": {"refine": {"steps": 300, "loss": "swav_local"}},
}


def _merge(base: dict, over: Mapping) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _jsonable(o):
    if isinstance(o, AugPolicy):
        return asdict(o)
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if callable(o):
        return getattr(o, "__name__", repr(o))
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 5. Núcleo: candidatos, estatísticas entre vistas, seleção, refino
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Cache:
    F: Tensor            # (M, C, H, W) features do prefixo nas imagens originais
    Fa: Tensor           # (M, C, H, W) vista a
    Fb: Tensor           # (M, C, H, W) vista b
    corr: Correspondences
    y: Optional[Tensor]


def _unfold_rows(Fsrc: Tensor, k: int, pos: Tensor) -> Tensor:
    """Patches (n, C·k·k) channel-major (= ordem de Conv2d.weight.reshape) nas posições pos (M,P)."""
    rows = []
    for i in range(Fsrc.shape[0]):
        u = F.unfold(Fsrc[i:i + 1], k, padding=k // 2)[0]            # (D, H*W)
        rows.append(u[:, pos[i]].t())
    return torch.cat(rows, 0)


def _spherical_kmeans(Xn: Tensor, K: int, iters: int, gen: torch.Generator) -> Tensor:
    """k-means esférico, semeado por k-means++ (gerador CPU). Xn linhas unitárias. -> (K, D)."""
    N = Xn.shape[0]
    first = int(torch.randint(N, (1,), generator=gen))
    C = [Xn[first]]
    d2 = (1 - Xn @ Xn[first]).clamp(min=0)
    for _ in range(1, K):
        p = d2.double()
        if p.sum() <= 0:
            idx = int(torch.randint(N, (1,), generator=gen))
        else:
            idx = int(torch.multinomial(p / p.sum(), 1, generator=gen))
        C.append(Xn[idx])
        d2 = torch.minimum(d2, (1 - Xn @ Xn[idx]).clamp(min=0))
    C = torch.stack(C)
    for _ in range(int(iters)):
        a = (Xn @ C.t()).argmax(1)
        newC = torch.zeros_like(C).index_add_(0, a, Xn)
        empty = newC.norm(dim=1) < 1e-12
        newC[empty] = C[empty]
        C = F.normalize(newC, dim=1)
    return C


def _fps(Xn: Tensor, K: int, gen: torch.Generator) -> Tensor:
    N = Xn.shape[0]
    idx = [int(torch.randint(N, (1,), generator=gen))]
    d = 1 - Xn @ Xn[idx[0]]
    for _ in range(1, K):
        j = int(d.argmax())
        idx.append(j)
        d = torch.minimum(d, 1 - Xn @ Xn[j])
    return Xn[torch.tensor(idx)]


def _candidates(Fsrc: Tensor, k: int, K: int, cfg: dict, gen: torch.Generator, info: dict,
                max_patches: int):
    """Candidatos no espaço de entrada (W (K,D), b (K,)) + versão branqueada unitária (diversidade)."""
    M, C, H, W = Fsrc.shape
    D = C * k * k
    P = max(1, min(int(cfg["patches_per_image"]), H * W, max_patches // max(M, 1)))
    pos = torch.stack([torch.randperm(H * W, generator=gen)[:P] for _ in range(M)])
    X = _unfold_rows(Fsrc, k, pos).double()                            # (M·P, D)
    n_hold = int(round(X.shape[0] * float(cfg["holdout_frac"])))
    perm = torch.randperm(X.shape[0], generator=gen)
    Xh, Xf = X[perm[:n_hold]], X[perm[n_hold:]]
    N = Xf.shape[0]
    if K > N:
        raise InfeasibleCandidate("budget: candidatos K > patches N",
                                  details={"K": K, "N": N, "D": D})
    mu = Xf.mean(0)
    Xc = Xf - mu
    Pm = (torch.eye(D, dtype=X.dtype) - 1.0 / D) if cfg["patch_norm"] == "mean_only" else torch.eye(D, dtype=X.dtype)
    Xp = Xc @ Pm
    cov = Xp.t() @ Xp / max(N - 1, 1)
    eig, vec = torch.linalg.eigh(cov)
    eig = eig.clamp(min=0)
    if cfg["whiten"] == "zca":
        eps = float(cfg["eps_zca"]) * float(eig.mean().clamp(min=1e-12))
        Z = vec @ torch.diag(1.0 / torch.sqrt(eig + eps)) @ vec.t()
        Zinv = vec @ torch.diag(torch.sqrt(eig + eps)) @ vec.t()
    elif cfg["whiten"] == "none":
        Z = Zinv = torch.eye(D, dtype=X.dtype)
    else:
        raise ValueError(f"whiten {cfg['whiten']!r}")
    Xw = Xp @ Z
    Xn = F.normalize(Xw, dim=1)
    keep = Xw.norm(dim=1) > 1e-9
    Xn = Xn[keep]
    if Xn.shape[0] < K:
        raise InfeasibleCandidate("budget: patches não degenerados < K",
                                  details={"K": K, "N_nondeg": int(Xn.shape[0])})
    fn = cfg.get("candidate_fn")
    if fn is not None:
        # Gancho para candidatos externos (protótipos SPiFiL-1, patches SPiFiL-0, banco controlado):
        # fn(X_fit (N,D) espaço de entrada, K, gen) -> W (K', D) espaço de entrada, K' ≥ n_out.
        Wc = torch.as_tensor(fn(Xf, K, gen), dtype=X.dtype)
        if Wc.ndim != 2 or Wc.shape[1] != D:
            raise ValueError(f"candidate_fn devolveu {tuple(Wc.shape)}, esperado (K', {D})")
        Cw = F.normalize(Wc @ Zinv, dim=1)       # c = w·Z⁻¹ (diversidade no espaço branqueado)
        info["candidate_fn"] = getattr(fn, "__name__", repr(fn))
    elif cfg["cand"] == "kmeans":
        Cw = _spherical_kmeans(Xn, K, cfg["kmeans_iters"], gen)
        Wc = Cw @ Z @ Pm                                               # dobra exata (Z, P simétricas)
    elif cfg["cand"] == "patches":
        Cw = _fps(Xn, K, gen)
        Wc = Cw @ Z @ Pm
    else:
        raise ValueError(f"cand {cfg['cand']!r}")
    bc = -(Wc @ mu)
    # escala: desvio da resposta em patches separados do ajuste
    r = Xh @ Wc.t() + bc if Xh.shape[0] > 1 else Xf @ Wc.t() + bc
    sd = r.std(0, unbiased=False).clamp(min=1e-12)
    Wn, bn = Wc / sd[:, None], bc / sd
    info.update({"N": int(N), "D": int(D), "N_over_D": float(N) / D, "K": int(Wc.shape[0]),
                 "n_holdout": int(n_hold), "patches_per_image": int(P),
                 "erank_patches": _erank_from_eig(eig), "cov_trace": float(eig.sum())})
    return {"W": Wn.float(), "b": bn.float(), "W_raw": Wc.float(), "b_raw": bc.float(),
            "Cw": Cw.float(), "mu": mu, "Pm": Pm, "Z": Z, "Xh": Xh, "sd": sd}


def _gather(R: Tensor, img: Tensor, pos: Tensor) -> Tensor:
    """R (b, K, H·W) -> (n, K) nas posições (img, pos)."""
    return R[img, :, pos]


def _view_stats(Wk: Tensor, bk: Tensor, k: int, cache: _Cache, cfg: dict) -> dict:
    """Estabilidade (Pearson casado), repetibilidade (top-m), concordância Sinkhorn e massa."""
    K = Wk.shape[0]
    C = cache.Fa.shape[1]
    weight = Wk.reshape(K, C, k, k)
    acc = torch.zeros(6, K, dtype=torch.float64)       # n, sa, sb, saa, sbb, sab
    rep_sum = torch.zeros(K, dtype=torch.float64)
    rep_cnt = 0
    agree_num = torch.zeros(K, dtype=torch.float64)
    agree_den = torch.zeros(K, dtype=torch.float64)
    mass = torch.zeros(K, dtype=torch.float64)
    n_tot = 0
    M = cache.Fa.shape[0]
    ch = int(cfg["chunk_images"])
    m, tol, tau = int(cfg["rep_m"]), int(cfg["rep_tol"]), float(cfg["tau"])
    corr = cache.corr
    for s in range(0, M, ch):
        e = min(M, s + ch)
        sel = (corr.img >= s) & (corr.img < e)
        if not bool(sel.any()):
            continue
        c = corr.subset(sel)
        Ra = F.conv2d(cache.Fa[s:e], weight, bk, padding=k // 2).flatten(2)
        Rb = F.conv2d(cache.Fb[s:e], weight, bk, padding=k // 2).flatten(2)
        ii = c.img - s
        ra, rb = _gather(Ra, ii, c.pa).double(), _gather(Rb, ii, c.pb).double()
        acc[0] += ra.shape[0]
        acc[1] += ra.sum(0); acc[2] += rb.sum(0)
        acc[3] += (ra * ra).sum(0); acc[4] += (rb * rb).sum(0); acc[5] += (ra * rb).sum(0)
        # repetibilidade: top-m picos de a mapeados para b caem a ≤ tol células de um top-m de b
        for i in torch.unique(ii).tolist():
            msk = ii == i
            n_i = int(msk.sum())
            if n_i < 2 * m:
                continue
            rai, rbi = ra[msk], rb[msk]
            cb = torch.stack([c.ib[msk], c.jb[msk]], -1)              # (n_i, 2) coords em b
            ta = rai.topk(m, dim=0).indices.t()                       # (K, m)
            tb = rbi.topk(m, dim=0).indices.t()
            pa_c, pb_c = cb[ta], cb[tb]                               # (K, m, 2)
            d = (pa_c[:, :, None, :] - pb_c[:, None, :, :]).abs().amax(-1)   # (K, m, m)
            rep_sum += (d <= tol).any(-1).double().mean(-1)
            rep_cnt += 1
        # consistência de atribuição (SwAV, troca de predição; protótipos = candidatos)
        qa, qb = sinkhorn(ra, cfg["eps_sk"], cfg["sk_iters"]), sinkhorn(rb, cfg["eps_sk"], cfg["sk_iters"])
        lpa, lpb = F.log_softmax(ra / tau, 1), F.log_softmax(rb / tau, 1)
        agree_num += (qa * lpb).sum(0) + (qb * lpa).sum(0)
        agree_den += qa.sum(0) + qb.sum(0)
        mass += qa.sum(0) + qb.sum(0)
        n_tot += 2 * ra.shape[0]
    n = acc[0].clamp(min=1)
    ma, mb = acc[1] / n, acc[2] / n
    va = (acc[3] / n - ma ** 2).clamp(min=0)
    vb = (acc[4] / n - mb ** 2).clamp(min=0)
    cv = acc[5] / n - ma * mb
    den = torch.sqrt(va * vb)
    stab = torch.where(den > 1e-12, cv / den.clamp(min=1e-12), torch.zeros_like(cv))
    rep = rep_sum / max(rep_cnt, 1)
    agree = torch.where(agree_den > 0, agree_num / agree_den.clamp(min=1e-30),
                        torch.full_like(agree_num, -math.log(K)))
    mass = mass / max(n_tot, 1)
    return {"stab": stab, "rep": rep, "agree": agree, "mass": mass,
            "n_pairs": int(acc[0][0]) if K else 0, "rep_images": rep_cnt}


def _fisher_gap(Wk: Tensor, bk: Tensor, k: int, Fsrc: Tensor, y: Tensor, cfg: dict) -> Tensor:
    """Razão de Fisher (entre/intra classes) da resposta GAP(ReLU) por filtro. Só S2-sup."""
    K = Wk.shape[0]
    weight = Wk.reshape(K, Fsrc.shape[1], k, k)
    g = []
    for s in range(0, Fsrc.shape[0], int(cfg["chunk_images"])):
        g.append(F.relu(F.conv2d(Fsrc[s:s + cfg["chunk_images"]], weight, bk, padding=k // 2)).mean((2, 3)))
    g = torch.cat(g).double()
    mu = g.mean(0)
    between = torch.zeros(K, dtype=torch.float64)
    within = torch.zeros(K, dtype=torch.float64)
    for c in torch.unique(y).tolist():
        gc = g[y == c]
        between += gc.shape[0] * (gc.mean(0) - mu) ** 2
        within += ((gc - gc.mean(0)) ** 2).sum(0)
    return between / within.clamp(min=1e-12)


def _greedy_select(score: Tensor, Cw: Tensor, n: int, stab: Tensor, cfg: dict) -> tuple[list, dict]:
    """Seleção gulosa: max score + lam_div·(1 − max|cos|), com NMS (|cos| ≤ red_max) e piso de
    estabilidade; se faltar, relaxa o piso e depois o NMS (completa por score; registrado)."""
    K = score.shape[0]
    Cn = F.normalize(Cw.double(), dim=1)
    sim = (Cn @ Cn.t()).abs()
    chosen: list[int] = []
    taken = torch.zeros(K, dtype=torch.bool)
    maxsim = torch.zeros(K, dtype=torch.float64)
    relax = {"floor_relaxed": 0, "nms_relaxed": 0}
    floor_ok = stab >= float(cfg["stab_floor"])
    lam_div, red = float(cfg["lam_div"]), float(cfg["red_max"])
    for _ in range(n):
        val = score + lam_div * (1 - maxsim)
        stage = None
        for st, ok in (("strict", floor_ok & (maxsim <= red)),
                       ("floor_relaxed", maxsim <= red),
                       ("nms_relaxed", torch.ones(K, dtype=torch.bool))):
            ok = ok & ~taken
            if bool(ok.any()):
                stage = st
                v = torch.where(ok, val, torch.full_like(val, -math.inf))
                j = int(v.argmax())
                break
        if stage is None:
            raise InfeasibleCandidate("spifil2: candidatos insuficientes", details={"K": K, "n": n})
        if stage != "strict":
            relax[stage] += 1
        chosen.append(j)
        taken[j] = True
        maxsim = torch.maximum(maxsim, sim[j])
    return chosen, relax


def _refine_local(w: Tensor, b: Tensor, k: int, cache: _Cache, cfg: dict, gen: torch.Generator):
    """Refino local (S2-ref) numa CÓPIA TEMPORÁRIA dos pesos do bloco; prefixo congelado (features
    em cache, sem grafo); nenhum parâmetro persistente. Loss vicregl_local ou swav_local."""
    rc = cfg["refine"]
    steps = int(rc.get("steps", 0))
    if steps <= 0:
        return w, b, None
    C = cache.Fa.shape[1]
    n = w.shape[0]
    wp = torch.nn.Parameter(w.detach().clone().reshape(n, C, k, k))
    bp = torch.nn.Parameter(b.detach().clone())
    opt = torch.optim.Adam([wp, bp], lr=float(rc.get("lr", 1e-3)))
    M = cache.Fa.shape[0]
    bs = max(1, min(int(rc.get("batch_images", 8)), M))
    corr = cache.corr
    hist = []
    tau = float(cfg["tau"])
    for _ in range(steps):
        idx = torch.randperm(M, generator=gen)[:bs].sort().values
        sel = torch.isin(corr.img, idx)
        if not bool(sel.any()):
            continue
        c = corr.subset(sel)
        remap = torch.full((M,), -1, dtype=torch.long)
        remap[idx] = torch.arange(idx.numel())
        ii = remap[c.img]
        za = F.conv2d(cache.Fa[idx], wp, bp, padding=k // 2).flatten(2)
        zb = F.conv2d(cache.Fb[idx], wp, bp, padding=k // 2).flatten(2)
        ra, rb = _gather(za, ii, c.pa), _gather(zb, ii, c.pb)
        if rc.get("loss", "vicregl_local") == "vicregl_local":
            inv = F.mse_loss(ra, rb)
            sa = torch.sqrt(ra.var(0) + 1e-4)
            sb = torch.sqrt(rb.var(0) + 1e-4)
            var = F.relu(1 - sa).mean() + F.relu(1 - sb).mean()

            def _cov(z):
                zc = z - z.mean(0)
                cm = zc.t() @ zc / max(z.shape[0] - 1, 1)
                off = cm - torch.diag(torch.diag(cm))
                return off.pow(2).sum() / z.shape[1]
            loss = rc.get("inv", 25.0) * inv + rc.get("var", 25.0) * var + rc.get("cov", 1.0) * (_cov(ra) + _cov(rb))
        elif rc["loss"] == "swav_local":
            qa = sinkhorn(ra.detach(), cfg["eps_sk"], cfg["sk_iters"])
            qb = sinkhorn(rb.detach(), cfg["eps_sk"], cfg["sk_iters"])
            loss = -(qa * F.log_softmax(rb / tau, 1)).sum(1).mean() - (qb * F.log_softmax(ra / tau, 1)).sum(1).mean()
        else:
            raise ValueError(f"refine.loss {rc['loss']!r}")
        if not torch.isfinite(loss):
            raise InfeasibleCandidate("spifil2: refino não finito", details={"step": len(hist)})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    w2, b2 = wp.detach().reshape(n, -1), bp.detach()
    cos = F.cosine_similarity(w2.double(), w.detach().double(), dim=1)
    return w2, b2, {"steps": steps, "loss_first": hist[0] if hist else None,
                    "loss_last": hist[-1] if hist else None, "cos_to_init_mean": float(cos.mean()),
                    "cos_to_init_min": float(cos.min())}


def _fit_bank(Fsrc: Tensor, Fa: Tensor, Fb: Tensor, corr: Correspondences, y: Optional[Tensor],
              k: int, n_out: int, cfg: dict, gen: torch.Generator, supervised: bool,
              max_patches: int, tag: str) -> tuple[Tensor, Tensor, dict]:
    """Ajusta um banco denso (n_out, C·k·k) sobre um tensor de entrada já transformado."""
    info: dict = {"tag": tag, "kernel": k, "in_channels": int(Fsrc.shape[1]), "n_out": int(n_out)}
    K = max(n_out, int(cfg["cand_mult"]) * n_out)
    cand = _candidates(Fsrc, k, K, cfg, gen, info, max_patches)
    cache = _Cache(Fsrc, Fa, Fb, corr, y)
    st = _view_stats(cand["W"], cand["b"], k, cache, cfg)
    dc = dc_fraction(cand["W_raw"].double(), k)
    score = (cfg["lam_stab"] * _z(st["stab"]) + cfg["lam_rep"] * _z(st["rep"])
             + cfg["lam_agree"] * _z(st["agree"])
             + cfg["lam_mass"] * _z(torch.log(st["mass"].clamp(min=1e-30)))
             - cfg["lam_dc"] * _z(dc))
    fisher = None
    if supervised:
        fisher = _fisher_gap(cand["W"], cand["b"], k, Fsrc, y, cfg)
        score = score + cfg["lam_fisher"] * _z(fisher)
    sel, relax = _greedy_select(score, cand["Cw"], n_out, st["stab"], cfg)
    sel_t = torch.tensor(sel, dtype=torch.long)
    if cfg["unit_std"]:
        w, b = cand["W"][sel_t].clone(), cand["b"][sel_t].clone()
    else:
        w, b = cand["W_raw"][sel_t].clone(), cand["b_raw"][sel_t].clone()
    w, b, ref = _refine_local(w, b, k, cache, cfg, gen)
    if ref is not None:
        # recentra (b = −w·μ) e recalcula a escala nos patches de holdout (registrado)
        mu = cand["mu"]
        b = -(w.double() @ mu)
        if cfg["unit_std"]:
            r = cand["Xh"] @ w.double().t() + b
            sd = r.std(0, unbiased=False).clamp(min=1e-12)
            w, b = (w.double() / sd[:, None]).float(), (b / sd).float()
        else:
            w, b = w.float(), b.float()
        st_after = _view_stats(w, b, k, cache, cfg)
        ref["stab_after_mean"] = float(st_after["stab"].mean())
    Cn = F.normalize(cand["Cw"][sel_t].double(), dim=1)
    simsel = (Cn @ Cn.t()).abs() - torch.eye(len(sel), dtype=torch.float64)
    info.update({
        "n_pairs": st["n_pairs"], "rep_images": st["rep_images"], "relaxations": relax,
        "selected_idx": sel,
        "selected": {key: st[key][sel_t].tolist() for key in ("stab", "rep", "agree", "mass")},
        "selected_score": score[sel_t].tolist(),
        "candidates_mean": {key: float(st[key].mean()) for key in ("stab", "rep", "agree")},
        "selected_mean": {key: float(st[key][sel_t].mean()) for key in ("stab", "rep", "agree")},
        "dc_fraction": {"candidates_mean": float(dc.mean()), "selected_mean": float(dc[sel_t].mean())},
        "redundancy_selected": float((simsel > 0.9).double().sum() / max(len(sel) * (len(sel) - 1), 1)),
        "fisher_selected": fisher[sel_t].tolist() if fisher is not None else None,
        "refine": ref,
    })
    return w.float().contiguous(), b.float().contiguous(), info


# ─────────────────────────────────────────────────────────────────────────────
# 6. O inicializador
# ─────────────────────────────────────────────────────────────────────────────
def _bg_keep(views: Sequence[Tensor], pol: AugPolicy) -> Optional[Tensor]:
    if pol.max_bg_frac is None:
        return None
    keep = None
    for v in views:
        frac = (v[:, 0] < pol.bg_thr).float().mean((1, 2))
        ok = frac <= pol.max_bg_frac
        keep = ok if keep is None else keep & ok
    return keep


@register(METHOD)
class Spifil2Invariant(KernelInitializer):
    """KernelInitializer SPiFiL-2. ``fit`` devolve (weights, bias, provenance) com as chaves do
    state_dict de ``SpifilOpsBlock(effective_spec)`` (conv.* e skip.*).

    ``Spifil2Invariant(variant="s2_noinv", lam_div=0.3)``: variante registrada + overrides; o
    ``context.config`` (chave ``"spifil2"`` ou chaves de ``DEFAULT_CONFIG``) sobrepõe por último.
    """
    name = METHOD
    version = VERSION
    label_policy = "label_free"

    def __init__(self, variant: Optional[str] = None, **config):
        super().__init__(**config)
        over = {}
        if variant is not None:
            if variant not in VARIANTS:
                raise ValueError(f"variante desconhecida {variant!r}; registradas: {sorted(VARIANTS)}")
            over = VARIANTS[variant]
        self.variant = variant or "s2"
        self.base_config = _merge(_merge(DEFAULT_CONFIG, over), config)

    @staticmethod
    def _load_images(td, seed: int, max_images: int, need_labels: bool):
        xs, ys, n = [], [], 0
        for batch in td.loader_factory(int(seed)):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(x.shape[0], max_images - n)
            xs.append(x[:take].detach().float().cpu())
            if need_labels:
                ys.append(torch.as_tensor(batch[1][:take]).long().cpu())
            n += take
            if n >= max_images:
                break
        if n == 0:
            raise InfeasibleCandidate("spifil2: loader de treino vazio")
        return torch.cat(xs), (torch.cat(ys) if need_labels else None)

    # -- fit -------------------------------------------------------------------------------
    def fit(self, prefix: Optional[nn.Module], train_data, block_spec: dict, context):
        t0 = time.time()
        check_train_only(train_data)                                           # I5
        ccfg = context.config if isinstance(getattr(context, "config", None), Mapping) else {}
        if "spifil2" in ccfg:
            over = ccfg["spifil2"]
        else:   # só as chaves deste método (não polui o hash com as de outros métodos)
            over = {k: v for k, v in ccfg.items() if k in DEFAULT_CONFIG or k == "label_policy"}
        cfg = _merge(self.base_config, over)
        # Label-free é o padrão do SPiFiL-2; supervisionado só pela variante/config explícita.
        policy = cfg.get("label_policy", "label_free")
        if policy not in ("supervised", "label_free"):
            raise ValueError(f"label_policy {policy!r}")
        if policy == "supervised" and getattr(train_data, "label_policy", "supervised") == "label_free":
            raise ValueError("label_policy supervisionada pedida, mas TrainData é label_free (I6)")
        cfg["label_policy"] = policy
        supervised = policy == "supervised"
        pol = _as_policy(cfg["aug_policy"])
        budget = getattr(context, "budget", None) or InitBudget()
        dev = torch.device(getattr(context, "device", "cpu") or "cpu")
        seed = int(context.seed)
        spec = dict(block_spec)
        kind = spec.get("conv", "conv")
        cin, cout = int(spec["in_channels"]), int(spec["channels"])

        # prefixo congelado (I1): eval, sem grad, fingerprint antes/depois
        pre_sha = prefix_fingerprint(prefix)
        saved = None
        if prefix is not None:
            saved = (prefix.training, [p.requires_grad for p in prefix.parameters()])
            prefix.eval()
            for p in prefix.parameters():
                p.requires_grad_(False)
        try:
            X, y = self._load_images(train_data, seed, int(budget.max_images), supervised)
            gen_v = torch.Generator(device="cpu").manual_seed(seed * 7919 + 1)
            feats, fa, fb, th_a, th_b, keeps = [], [], [], [], [], []
            ch = int(cfg["chunk_images"])

            def _pf(t):
                t = t.to(dev)
                with torch.no_grad():
                    if prefix is None:
                        out = t
                    elif getattr(context, "amp_dtype", None) is not None and dev.type == "cuda":
                        with torch.autocast("cuda", dtype=context.amp_dtype):
                            out = prefix(t)
                    else:
                        out = prefix(t)
                return out.float().cpu()

            for s in range(0, X.shape[0], ch):
                xb = X[s:s + ch]
                views, thetas, _ = make_views_with_theta(xb, int(cfg["views"]), pol, generator=gen_v)
                k_ = _bg_keep(views[:2], pol)
                keeps.append(k_ if k_ is not None else torch.ones(xb.shape[0], dtype=torch.bool))
                feats.append(_pf(xb)); fa.append(_pf(views[0])); fb.append(_pf(views[1]))
                th_a.append(thetas[0]); th_b.append(thetas[1])
            Fx, Fa, Fb = torch.cat(feats), torch.cat(fa), torch.cat(fb)
        finally:
            if prefix is not None and saved is not None:
                prefix.train(saved[0])
                for p, rg in zip(prefix.parameters(), saved[1]):
                    p.requires_grad_(rg)
        check_prefix_unchanged(prefix, pre_sha)                               # I1
        if Fx.shape[1] != cin:
            raise ValueError(f"in_channels do spec ({cin}) != canais do prefixo ({Fx.shape[1]})")
        mem_gb = 3 * Fx.numel() * 4 / 1e9
        if mem_gb > float(budget.max_gpu_gb):
            raise InfeasibleCandidate(f"budget: cache de features {mem_gb:.2f} GB > {budget.max_gpu_gb}",
                                      details={"mem_gb": mem_gb})
        keep = torch.cat(keeps)
        corr = correspondences(torch.cat(th_a), torch.cat(th_b), Fa.shape[-2:], keep=keep)
        if len(corr) == 0:
            raise InfeasibleCandidate("spifil2: nenhuma correspondência entre vistas")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        mp = int(budget.max_patches)
        banks: list[dict] = []

        def bank(Fs, Fa_, Fb_, k, n, tag):
            w, b, info = _fit_bank(Fs, Fa_, Fb_, corr, y, k, n, cfg, gen, supervised, mp, tag)
            banks.append(info)
            return w, b

        weights, bias = {}, {}
        eff = dict(spec)
        approx = ["feature_grid_uniform", "views_pair_0_1"]
        if cfg["unit_std"]:
            approx.append("unit_response_std")
        if kind in ("conv", "pointwise"):
            k = 1 if kind == "pointwise" else int(spec["kernel"])
            w, b = bank(Fx, Fa, Fb, k, cout, kind)
            weights["conv.weight"], bias["conv.bias"] = w.reshape(cout, cin, k, k), b
        elif kind == "depthwise":
            k = int(spec["kernel"])
            if cout % cin:
                raise InfeasibleCandidate("depthwise: channels % in_channels != 0")
            m = cout // cin
            ws, bs_ = [], []
            for c in range(cin):
                w, b = bank(Fx[:, c:c + 1], Fa[:, c:c + 1], Fb[:, c:c + 1], k, m, f"depthwise.c{c}")
                ws.append(w.reshape(m, 1, k, k)); bs_.append(b)
            weights["conv.weight"], bias["conv.bias"] = torch.cat(ws), torch.cat(bs_)
        elif kind == "dwsep":
            k = int(spec["kernel"])
            ws, bs_ = [], []
            for c in range(cin):
                w, b = bank(Fx[:, c:c + 1], Fa[:, c:c + 1], Fb[:, c:c + 1], k, 1, f"dwsep.dw.c{c}")
                ws.append(w.reshape(1, 1, k, k)); bs_.append(b)
            wd, bd = torch.cat(ws), torch.cat(bs_)
            weights["conv.depthwise.weight"], bias["conv.depthwise.bias"] = wd, bd

            def dw(t):
                return F.conv2d(t, wd, bd, padding=k // 2, groups=cin)
            w, b = bank(dw(Fx), dw(Fa), dw(Fb), 1, cout, "dwsep.pw")
            weights["conv.pointwise.weight"], bias["conv.pointwise.bias"] = w.reshape(cout, cin, 1, 1), b
        elif kind == "inception":
            from ga_search.modehb.ops_norm_act import inception_split
            branches = spec.get("inception_branches") or inception_split(cout)
            eff["inception_branches"] = [list(br) for br in branches]
            approx.append("inception_no_cross_branch_diversity")
            for name, kb, nb in branches:
                kb, nb = int(kb), int(nb)
                if name == "pool":
                    mp3 = lambda t: F.max_pool2d(t, 3, 1, 1)  # noqa: E731
                    w, b = bank(mp3(Fx), mp3(Fa), mp3(Fb), 1, nb, "inception.pool")
                    weights["conv.branches.pool.proj.weight"] = w.reshape(nb, cin, 1, 1)
                    bias["conv.branches.pool.proj.bias"] = b
                else:
                    w, b = bank(Fx, Fa, Fb, kb, nb, f"inception.{name}")
                    weights[f"conv.branches.{name}.weight"] = w.reshape(nb, cin, kb, kb)
                    bias[f"conv.branches.{name}.bias"] = b
        else:
            raise InfeasibleCandidate(f"spifil2: conv {kind!r} não suportado")

        skip = spec.get("skip") if spec.get("residual") else None
        if skip == "project":
            # PCA label-free do 1×1 de entrada: preserva a variância máxima no atalho (registrado)
            pos = torch.stack([torch.randperm(Fx.shape[2] * Fx.shape[3], generator=gen)[:64]
                               for _ in range(Fx.shape[0])])
            X1 = _unfold_rows(Fx, 1, pos).double()
            mu1 = X1.mean(0)
            cov1 = (X1 - mu1).t() @ (X1 - mu1) / max(X1.shape[0] - 1, 1)
            _, vec = torch.linalg.eigh(cov1)
            r = min(cout, cin)
            ws = torch.zeros(cout, cin, dtype=torch.float64)
            ws[:r] = vec[:, -r:].flip(1).t()
            weights["skip.weight"] = ws.float().reshape(cout, cin, 1, 1).contiguous()
            bias["skip.bias"] = (-(ws @ mu1)).float().contiguous()
            approx.append("skip_pca_label_free" + ("_zero_rows" if cout > cin else ""))
        elif skip == "identity" and cin != cout:
            raise InfeasibleCandidate("identity skip exige in_channels == channels")

        for d in (weights, bias):
            for kk, v in d.items():
                if not torch.isfinite(v).all():
                    raise InfeasibleCandidate(f"spifil2: tensor não finito em {kk}")
        cfg_json = json.dumps(_jsonable(cfg), sort_keys=True, default=str)
        n_patches = sum(int(bk["N"]) + int(bk["n_holdout"]) for bk in banks)
        prov = make_provenance(
            METHOD, VERSION, context=context, train_data=train_data, requested_spec=spec,
            effective_spec=eff, labels_used=supervised,
            label_source="image_labels" if supervised else "none", mask_used=False,
            n_images=int(X.shape[0]), n_patches=n_patches,
            patch_dim=int(banks[0]["D"]) if banks else None, approximations=approx,
            fit_seconds=time.time() - t0,
            extra={
                "variant": self.variant, "faithful_to": None,
                "method_config": json.loads(cfg_json),
                "method_config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
                "prefix_state_sha256": pre_sha,
                "aug_policy": {"name": cfg["aug_policy"] if isinstance(cfg["aug_policy"], str) else "custom",
                               **_jsonable(asdict(pol))},
                "banks": _jsonable(banks),
            })
        prov["data"].update({"images_sha256": _sha256_tensor(X), "n_images_kept_bg": int(keep.sum()),
                             "n_pairs": len(corr), "mean_match_err_cells": float(corr.err.mean())})
        prov["env"]["python"] = platform.python_version()
        if prov["timing"]["fit_seconds"] > float(budget.max_seconds):
            raise InfeasibleCandidate("budget: tempo", details={"seconds": prov["timing"]["fit_seconds"]})
        return weights, bias, prov
