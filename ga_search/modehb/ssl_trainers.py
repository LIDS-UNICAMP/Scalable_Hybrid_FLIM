"""Global SSL training of the full (progressively grown) encoder — MO-DEHB Fase C.

Public API
----------
- ``nt_xent(z1, z2, tau)``      SimCLR NT-Xent (L2-normalised, 2N-2 negatives).
- ``make_views(x, V, cfg, seed)`` multi-view augmentation on [0,1] tensors.
- ``train_global_ssl(...)``    unfreezes everything, AdamW with two fixed-lr
  param groups (old layers / new layer + head), trains with SimCLR, LeJEPA
  (invariance + official Epps–Pulley SIGReg) or an AE objective, logs separate
  loss series + diagnostics, raises on NaN/Inf, discards the projector/decoder.

Method naming
-------------
- ``simclr`` / ``simclr+sigreg`` / ``simclr+weak_sigreg``
- ``lejepa``       = invariance + RealSIGReg (the original LeJEPA objective)
- ``lejepa_weak``  = invariance + weak SIGReg (NOT LeJEPA; never named as such)
- ``ae`` / ``ae+sigreg`` / ``ae+weak_sigreg``

Encoder contract: ``encoder(x) -> feature map [N,C,h,w]``; ``head(feat) -> [N,D]``
(``EmbeddingHead`` from ``pooling_head.py``). ``new_param_names`` are encoder
parameter names (or name prefixes) belonging to the newly added layer; all head
params are always "new".
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.lejepa_loss import RealSIGReg, invariance_loss

__all__ = [
    "AugConfig",
    "AUG_PRESETS",
    "aug_config",
    "aug_record",
    "nt_xent",
    "vicreg_loss",
    "make_views",
    "method_name",
    "split_param_groups",
    "train_global_ssl",
]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def nt_xent(z1: Tensor, z2: Tensor, tau: float = 0.2) -> Tensor:
    """SimCLR NT-Xent. z1, z2: [N, D] (unnormalised). Each of the 2N anchors has
    1 positive (its pair) and 2N-2 negatives; mean over the 2N anchors."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)          # [2N, D]
    sim = z @ z.t() / tau                                        # [2N, 2N]
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def vicreg_loss(z1: Tensor, z2: Tensor, sim_coeff: float = 25.0, std_coeff: float = 25.0,
                cov_coeff: float = 1.0, eps: float = 1e-4):
    """growth_v3: VICReg (Bardes, Ponce & LeCun, ICLR 2022), como no código oficial (facebookresearch/vicreg):
    inv = MSE(z1, z2); var = mean(relu(1 - std(z1)))/2 + mean(relu(1 - std(z2)))/2 com std = sqrt(var + eps);
    cov = soma dos off-diagonais^2 / D de cada ramo (z centralizado, cov = z^T z / (N - 1)).
    total = sim_coeff * inv + std_coeff * var + cov_coeff * cov. -> (total, {"inv", "var", "cov"})."""
    n, d = z1.shape
    inv = F.mse_loss(z1, z2)
    x = z1 - z1.mean(dim=0)
    y = z2 - z2.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + eps)
    std_y = torch.sqrt(y.var(dim=0) + eps)
    var = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    cov_x = (x.T @ x) / max(n - 1, 1)
    cov_y = (y.T @ y) / max(n - 1, 1)
    off = lambda c: c.pow(2).sum() - c.diagonal().pow(2).sum()          # noqa: E731
    cov = off(cov_x) / d + off(cov_y) / d
    total = sim_coeff * inv + std_coeff * var + cov_coeff * cov
    return total, {"inv": inv, "var": var, "cov": cov}


@torch.no_grad()
def _pos_neg_sim(z1: Tensor, z2: Tensor) -> tuple[float, float]:
    a, b = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    s = a @ b.t()
    n = s.shape[0]
    pos = s.diagonal().mean()
    neg = (s.sum() - s.diagonal().sum()) / max(n * n - n, 1)
    return float(pos), float(neg)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    crop_scale: tuple = (0.3, 1.0)
    crop_ratio: tuple = (3 / 4, 4 / 3)
    hflip_p: float = 0.5
    vflip_p: float = 0.5          # microscopy: no canonical orientation
    brightness: float = 0.2       # additive per-sample shift (all channels)
    contrast: float = 0.2         # multiplicative around per-image mean
    channel_jitter: float = 0.05  # small per-channel additive shift (LAB-safe)
    out_size: Optional[int] = None  # default: keep input H
    # growth_v3 (opt-in; padrão 0 = histórico bit a bit: nenhum sorteio extra no gerador)
    blur_p: float = 0.0           # prob. de blur gaussiano por amostra
    blur_sigma: tuple = (0.1, 2.0)  # sigma uniforme por amostra (pixels)
    blur_kernel: int = 5          # tamanho (ímpar) do kernel separável
    rot90_p: float = 0.0          # prob. de rot90 por amostra (k uniforme em {1,2,3}; só imagem quadrada)
    # growth v3.1 (BYOL/Barlow/DINO): solarização em LAB [0,1] = 1 - v onde v >= limiar (adaptação de 128/255)
    solarize_p: float = 0.0
    solarize_thr: float = 0.5


# growth_v3: força da DA do SSL como gene (registrada por ``aug_record``).
AUG_PRESETS: Dict[str, Dict[str, Any]] = {
    "weak": {"crop_scale": (0.6, 1.0), "hflip_p": 0.5, "vflip_p": 0.0, "brightness": 0.1, "contrast": 0.1,
             "channel_jitter": 0.02, "blur_p": 0.0, "rot90_p": 0.0},
    "medium": {"crop_scale": (0.3, 1.0), "hflip_p": 0.5, "vflip_p": 0.5, "brightness": 0.2, "contrast": 0.2,
               "channel_jitter": 0.05, "blur_p": 0.2, "blur_sigma": (0.1, 2.0), "rot90_p": 0.25},
    "strong": {"crop_scale": (0.08, 1.0), "hflip_p": 0.5, "vflip_p": 0.5, "brightness": 0.4, "contrast": 0.4,
               "channel_jitter": 0.1, "blur_p": 0.5, "blur_sigma": (0.1, 2.0), "rot90_p": 0.5},
}


def aug_config(spec: Any = None) -> "AugConfig":
    """None / "default" -> ``AugConfig()`` histórico; "weak"|"medium"|"strong" -> preset; dict -> campos
    (com ``preset`` opcional como base); ``AugConfig`` -> ele mesmo."""
    if spec is None or isinstance(spec, AugConfig):
        return spec if spec is not None else AugConfig()
    if isinstance(spec, str):
        if spec.lower() in ("default", "historical", "none"):
            return AugConfig()
        if spec.lower() not in AUG_PRESETS:
            raise ValueError(f"aug {spec!r}: use default|{'|'.join(AUG_PRESETS)}")
        return AugConfig(**AUG_PRESETS[spec.lower()])
    if isinstance(spec, dict):
        d = dict(spec)
        base = d.pop("preset", None)
        kw = dict(AUG_PRESETS[base]) if base else {}
        kw.update(d)
        return AugConfig(**kw)
    raise TypeError(f"aug: {type(spec).__name__}")


def aug_record(cfg: Any) -> Dict[str, Any]:
    """Registro json-safe da DA efetiva (tuplas -> listas)."""
    import dataclasses
    d = dataclasses.asdict(aug_config(cfg))
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _gaussian_blur(v: Tensor, sigma: Tensor, k: int) -> Tensor:
    """Blur gaussiano por amostra (sigma [N]) por conv separável agrupada; borda reflect."""
    N, C, H, W = v.shape
    r = k // 2
    t = torch.arange(k, device=v.device, dtype=v.dtype) - r
    g = torch.exp(-(t[None, :] ** 2) / (2 * sigma[:, None].clamp_min(1e-3) ** 2))
    g = g / g.sum(1, keepdim=True)                                       # [N, k]
    w = g.repeat_interleave(C, 0)                                        # [N*C, k]
    x = v.reshape(1, N * C, H, W)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, w[:, None, None, :], groups=N * C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, w[:, None, :, None], groups=N * C)
    return x.reshape(N, C, H, W)


def make_views(x: Tensor, V: int, cfg: Optional[AugConfig | dict] = None,
               seed: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> List[Tensor]:
    """Return V augmented views of x [N,C,H,W] in [0,1] (each [N,C,S,S], clamped).

    Vectorised random-resized-crop via affine_grid, flips, brightness/contrast
    and mild per-channel jitter. Deterministic given ``seed`` (or ``generator``).
    """
    if cfg is None:
        cfg = AugConfig()
    elif isinstance(cfg, dict):
        cfg = AugConfig(**cfg)
    if generator is None:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()
    N, C, H, W = x.shape
    S = cfg.out_size or H
    dev, dt = x.device, x.dtype

    # growth_fix8: os sorteios saem do MESMO gerador de CPU, nas MESMAS chamadas/formas/ordem de antes
    # (9 por vista, nenhum depende de valor calculado), mas vão para o device numa ÚNICA cópia por lote em vez
    # de 9·V cópias síncronas (cada ``.to(cuda)`` de memória paginável sincroniza o stream). Mesmos números,
    # mesmas operações elementares -> vistas bit a bit iguais.
    shapes = ((N,),) * 6 + ((N, 1, 1, 1), (N, 1, 1, 1), (N, C, 1, 1))
    # growth_v3: sorteios extras SÓ quando ligados, depois dos 9 históricos de cada vista
    use_blur, use_rot = float(cfg.blur_p) > 0, float(cfg.rot90_p) > 0
    # rot90: as vistas são sempre S x S (quadradas), então a rotação preserva a forma
    use_sol = float(getattr(cfg, "solarize_p", 0.0)) > 0
    shapes = shapes + (((N,), (N,)) if use_blur else ()) + (((N,), (N,)) if use_rot else ()) + (((N,),) if use_sol else ())
    cpu = [torch.rand(*s, generator=generator) for _ in range(V) for s in shapes]
    flat = torch.cat([t.reshape(-1) for t in cpu])
    if dev.type == "cuda":
        flat = flat.pin_memory()
    flat = flat.to(dev, non_blocking=True).to(dt)
    draws, off = [], 0
    for t in cpu:
        draws.append(flat[off:off + t.numel()].view(t.shape))
        off += t.numel()
    it = iter(draws)

    def rnd(*shape):
        t = next(it)
        assert t.shape == torch.Size(shape), (t.shape, shape)
        return t

    views = []
    for _ in range(V):
        area = cfg.crop_scale[0] + rnd(N) * (cfg.crop_scale[1] - cfg.crop_scale[0])
        lr0, lr1 = math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])
        ratio = torch.exp(lr0 + rnd(N) * (lr1 - lr0))
        sw = torch.sqrt(area * ratio).clamp(max=1.0)   # fraction of width
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
        v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)
        mean = v.mean(dim=(1, 2, 3), keepdim=True)
        con = 1 + (rnd(N, 1, 1, 1) * 2 - 1) * cfg.contrast
        bri = (rnd(N, 1, 1, 1) * 2 - 1) * cfg.brightness
        ch = (rnd(N, C, 1, 1) * 2 - 1) * cfg.channel_jitter
        v = (v - mean) * con + mean + bri + ch
        v = v.clamp(0.0, 1.0)
        if use_blur:            # growth_v3: blur gaussiano por amostra (sigma por amostra)
            on, u = rnd(N), rnd(N)
            sig = cfg.blur_sigma[0] + u * (cfg.blur_sigma[1] - cfg.blur_sigma[0])
            vb = _gaussian_blur(v, sig, int(cfg.blur_kernel))
            v = torch.where((on < cfg.blur_p)[:, None, None, None], vb, v)
        if use_rot:             # growth_v3: rot90 por amostra, k uniforme em {1,2,3}
            on, u = rnd(N), rnd(N)
            kk = torch.where(on < cfg.rot90_p, 1 + (u * 3).floor().clamp(max=2), torch.zeros_like(u)).long()
            out = v.clone()
            for k in (1, 2, 3):
                m = kk == k
                if bool(m.any()):
                    out[m] = torch.rot90(v[m], k, dims=(2, 3))
            v = out
        if use_sol:             # growth v3.1: solarização por amostra (LAB [0,1], limiar 0,5)
            on = rnd(N)
            v = torch.where((on < cfg.solarize_p)[:, None, None, None], torch.where(v >= cfg.solarize_thr, 1.0 - v, v), v)
        views.append(v)
    return views


# growth v3.1: vistas ASSIMÉTRICAS do BYOL/Barlow (T: blur 1,0 / solarize 0; T': blur 0,1 / solarize 0,2), em LAB
ASYM_VIEWS = ({"crop_scale": (0.08, 1.0), "hflip_p": 0.5, "vflip_p": 0.0, "brightness": 0.4, "contrast": 0.4,
               "channel_jitter": 0.1, "blur_p": 1.0, "blur_sigma": (0.1, 2.0), "blur_kernel": 5, "solarize_p": 0.0},
              {"crop_scale": (0.08, 1.0), "hflip_p": 0.5, "vflip_p": 0.0, "brightness": 0.4, "contrast": 0.4,
               "channel_jitter": 0.1, "blur_p": 0.1, "blur_sigma": (0.1, 2.0), "blur_kernel": 5, "solarize_p": 0.2})


def make_views_asym(x: Tensor, cfgs, generator: torch.Generator) -> List[Tensor]:
    """Uma vista por config (mesmo gerador, em ordem)."""
    return [make_views(x, 1, AugConfig(**c) if isinstance(c, dict) else c, generator=generator)[0] for c in cfgs]


def barlow_loss(z1: Tensor, z2: Tensor, lambd: float = 0.0051, eps: float = 1e-5):
    """Barlow Twins (Zbontar et al. 2021, código oficial): c = bn(z1)^T bn(z2) / N com BN sem afim (estatística do
    batch, viesada); perda = sum (diag(c)-1)^2 + lambd * sum offdiag(c)^2 (SOMA sobre D), em fp32."""
    z1, z2 = z1.float(), z2.float()
    N = z1.shape[0]
    bn = lambda z: (z - z.mean(0)) / torch.sqrt(z.var(0, unbiased=False) + eps)          # noqa: E731
    c = bn(z1).T @ bn(z2) / N
    on = (torch.diagonal(c) - 1).pow(2).sum()
    off = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on + lambd * off, {"on": on.detach(), "off": off.detach()}


def byol_loss(p: Tensor, z: Tensor) -> Tensor:
    """2 - 2 cos(p, sg(z)), média no batch (BYOL, Grill et al. 2020)."""
    return (2 - 2 * F.cosine_similarity(p.float(), z.detach().float(), dim=-1)).mean()


def _make_barlow_projector(d: int, dims: Sequence[int] = (8192, 8192, 8192)) -> nn.Module:
    """Projetor oficial do Barlow: (Linear sem bias + BN1d + ReLU) x2 + Linear sem bias."""
    dims = [int(x) for x in dims]
    layers: List[nn.Module] = []
    prev = int(d)
    for h in dims[:-1]:
        layers += [nn.Linear(prev, h, bias=False), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
        prev = h
    layers.append(nn.Linear(prev, dims[-1], bias=False))
    return nn.Sequential(*layers)


class _WNLinear(nn.Module):
    """weight_norm(Linear(bias=False)) com g = 1 congelado (norm_last_layer do DINO): W = v / ||v|| por linha.
    Igual ao oficial; escrito explícito porque o ``nn.utils.weight_norm`` não suporta deepcopy (professora)."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        lin = nn.Linear(din, dout, bias=False)
        self.weight_v = nn.Parameter(lin.weight.detach().clone())
        self.register_buffer("weight_g", torch.ones(dout, 1))

    def forward(self, x):
        return F.linear(x, self.weight_g * self.weight_v / self.weight_v.norm(dim=1, keepdim=True).clamp_min(1e-12))


class DINOHead(nn.Module):
    """Cabeça do DINO (Caron et al. 2021, código oficial): MLP 3 camadas (hidden, BN p/ CNN, GELU) -> bottleneck ->
    L2 -> weight_norm(Linear(bottleneck, K, bias=False)) com weight_g = 1 congelado (norm_last_layer)."""

    def __init__(self, d: int, K: int = 4096, hidden: int = 2048, bottleneck: int = 256, use_bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(d, hidden)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, hidden))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.GELU(), nn.Linear(hidden, bottleneck)]
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last_layer = _WNLinear(bottleneck, int(K))

    def forward(self, x):
        return self.last_layer(F.normalize(self.mlp(x), dim=-1, p=2))


def dino_views(x: Tensor, generator: torch.Generator, n_local: int, local_size: int,
               global_scale=(0.14, 1.0), local_scale=(0.05, 0.14)) -> Tuple[List[Tensor], List[Tensor]]:
    """2 globais assimétricas (blur 1,0 / blur 0,1 + solarize 0,2) e ``n_local`` locais (blur 0,5) em ``local_size``."""
    base = {"hflip_p": 0.5, "vflip_p": 0.0, "brightness": 0.4, "contrast": 0.4, "channel_jitter": 0.1,
            "blur_sigma": (0.1, 2.0), "blur_kernel": 5}
    g1 = AugConfig(**base, crop_scale=tuple(global_scale), blur_p=1.0)
    g2 = AugConfig(**base, crop_scale=tuple(global_scale), blur_p=0.1, solarize_p=0.2)
    lc = AugConfig(**base, crop_scale=tuple(local_scale), blur_p=0.5, out_size=int(local_size))
    gl = [make_views(x, 1, g1, generator=generator)[0], make_views(x, 1, g2, generator=generator)[0]]
    lo = make_views(x, int(n_local), lc, generator=generator) if n_local else []
    return gl, lo


def dino_loss(student_out: List[Tensor], teacher_out: List[Tensor], center: Tensor, tau_s: float, tau_t: float):
    """CE entre q = softmax((t - c)/tau_t) (sem grad) e log_softmax(s/tau_s), pares (iq global, v != iq)."""
    total, n = 0.0, 0
    q = [F.softmax((t.float() - center) / tau_t, dim=-1).detach() for t in teacher_out]
    for iq, qq in enumerate(q):
        for v, sv in enumerate(student_out):
            if v == iq:
                continue
            total = total + torch.sum(-qq * F.log_softmax(sv.float() / tau_s, dim=-1), dim=-1).mean()
            n += 1
    return total / max(n, 1)


def byol_tau(k: int, K: int, tau_base: float = 0.996) -> float:
    """tau_k = 1 - (1 - tau_base) (cos(pi k / K) + 1) / 2 (BYOL, por passo)."""
    return 1.0 - (1.0 - float(tau_base)) * (math.cos(math.pi * min(int(k), int(K)) / max(int(K), 1)) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SSL = ("simclr", "lejepa", "ae", "vicreg", "barlow", "byol", "dino")
_REG = ("none", "sigreg", "weak_sigreg")


def method_name(ssl: str, reg: str) -> str:
    if ssl not in _SSL:
        raise ValueError(f"ssl must be one of {_SSL}, got {ssl!r}")
    if reg not in _REG:
        raise ValueError(f"reg must be one of {_REG}, got {reg!r}")
    if ssl == "lejepa":
        if reg == "sigreg":
            return "lejepa"
        if reg == "weak_sigreg":
            return "lejepa_weak"
        raise ValueError("lejepa requires a regularizer ('sigreg' = original LeJEPA, "
                         "'weak_sigreg' = lejepa_weak); invariance-only collapses")
    return ssl if reg == "none" else f"{ssl}+{reg}"


def _is_new(name: str, new_names: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p.rstrip(".") + ".") for p in new_names)


def split_param_groups(encoder: nn.Module, head: nn.Module, extra_new: Iterable[nn.Module],
                       new_param_names: Sequence[str], lr_old: float, lr_new: float,
                       weight_decay: float = 0.0) -> List[dict]:
    """Two AdamW groups: encoder params not in new_param_names -> lr_old;
    new encoder params + head + projector/decoder -> lr_new."""
    old, new = [], []
    for n, p in encoder.named_parameters():
        (new if _is_new(n, new_param_names) else old).append(p)
    new += list(head.parameters())
    for m in extra_new:
        new += list(m.parameters())
    groups = []
    if old:
        groups.append({"params": old, "lr": lr_old, "name": "old", "weight_decay": weight_decay})
    groups.append({"params": new, "lr": lr_new, "name": "new", "weight_decay": weight_decay})
    return groups


def _scale_conv_groups(groups: List[dict], encoder: nn.Module, scale: float) -> List[dict]:
    """Hybrid SPiFiL kernel protection: encoder Conv2d weights/biases get lr * ``scale``
    (1 = unchanged, 0.01 = low lr, 0 = frozen: removed from the optimizer, no grad)."""
    scale = float(scale)
    if scale == 1.0:
        return groups
    conv_ids = {id(p) for m in encoder.modules() if isinstance(m, nn.Conv2d) for p in m.parameters()}
    out = []
    for g in groups:
        convs = [p for p in g["params"] if id(p) in conv_ids]
        rest = [p for p in g["params"] if id(p) not in conv_ids]
        if rest:
            out.append({**g, "params": rest})
        if convs and scale > 0:
            out.append({**g, "params": convs, "lr": g["lr"] * scale, "name": g.get("name", "") + "_spifil"})
        elif convs:
            for p in convs:
                p.requires_grad_(False)
    return out


def _layer_key(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 and parts[1].isdigit() else parts[0]


def _per_layer_norms(module: nn.Module, prefix: str, grad: bool) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    for n, p in module.named_parameters():
        t = p.grad if grad else p
        if t is None:
            continue
        k = f"{prefix}/{_layer_key(n)}"
        acc[k] = acc.get(k, 0.0) + float(t.detach().float().pow(2).sum())
    return {k: math.sqrt(v) for k, v in acc.items()}


def _make_expander(d: int, dims: Sequence[int] = (2048, 2048, 2048)) -> nn.Module:
    """growth_v3: expander do VICReg (paper: 3 camadas, Linear-BN-ReLU-Linear-BN-ReLU-Linear)."""
    dims = [int(x) for x in dims]
    layers: List[nn.Module] = []
    prev = int(d)
    for h in dims[:-1]:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
        prev = h
    layers.append(nn.Linear(prev, dims[-1]))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# growth v3.1 bloco 2: LARS (cópia do oficial VICReg main_vicreg.py) + agenda warmup/cosseno por passo
# ---------------------------------------------------------------------------
def exclude_bias_and_norm(p) -> bool:
    """Filtro oficial (VICReg/SimCLR): parâmetros 1-D (bias, BN) fora do wd e da adaptação LARS."""
    return p.ndim == 1


class LARS(torch.optim.Optimizer):
    """LARS do código oficial do VICReg (``main_vicreg.py``): wd e adaptação de confiança por camada (eta),
    momentum clássico; ``weight_decay_filter``/``lars_adaptation_filter`` True = parâmetro EXCLUÍDO."""

    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta,
                        weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])
                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


def schedule_factor(step: int, total_steps: int, warmup_steps: int, end_ratio: float = 1e-3,
                    start_ratio: float = 0.0) -> float:
    """Fator do lr no passo GLOBAL ``step``: warmup linear de ``start_ratio`` a 1 em ``warmup_steps``, depois
    cosseno de 1 até ``end_ratio`` em ``total_steps`` (VICReg: end 1e-3; SimCLR: 0; LeJEPA: 1e-3)."""
    step, total_steps, warmup_steps = int(step), max(1, int(total_steps)), max(0, int(warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return start_ratio + (1.0 - start_ratio) * step / warmup_steps
    t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def _make_projector(d: int, hidden: int = 128, out: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                         nn.ReLU(inplace=True), nn.Linear(hidden, out))


class _ConvDecoder(nn.Module):
    """Light decoder: feature map [N,C,h,w] -> image [N,out_ch,H,W] in [0,1]."""

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, out_ch, 3, padding=1))

    def forward(self, feat: Tensor, size: tuple) -> Tensor:
        return torch.sigmoid(F.interpolate(self.net(feat), size=size, mode="bilinear",
                                           align_corners=False))


def _unpack(batch: Any) -> Tensor:
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    if isinstance(batch, dict):
        batch = batch.get("image", batch.get("x"))
    if isinstance(batch, (list, tuple)):          # pre-made views: take the first
        batch = batch[0]
    return batch


def _alert(logger, title: str, text: str) -> None:
    if logger is not None and hasattr(logger, "alert"):
        try:
            logger.alert(title, text, "ERROR")
        except Exception:  # noqa: BLE001 — never mask the original failure
            pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train_global_ssl(encoder: nn.Module, head: nn.Module, loader: Iterable,
                     ssl: str, reg: str, epochs: int, lr_old: float, lr_new: float,
                     new_param_names: Sequence[str], device: str | torch.device,
                     logger=None, seed: int = 0, sigreg_lambda: float = 0.05, *,
                     tau: float = 0.2, n_views: Optional[int] = None,
                     aug_cfg: Optional[AugConfig | dict] = None, weight_decay: float = 0.0,
                     grad_clip: Optional[float] = None, log_every: int = 10,
                     step_offset: int = 0, reg_kwargs: Optional[dict] = None,
                     max_steps_per_epoch: Optional[int] = None,
                     max_skip_frac: float = 1.0, conv_lr_scale: float = 1.0,
                     projector: bool = False, amp_dtype: Optional[torch.dtype] = None,
                     resume: Optional[Dict[str, Any]] = None, return_state: bool = False,
                     progress: Optional[str] = None, freeze_head: bool = False,
                     projector_dims: Optional[Sequence[int]] = None,
                     optimizer: str = "adamw", lars_kwargs: Optional[dict] = None,
                     lr_schedule: Optional[dict] = None, wd_exclude_1d: bool = False,
                     barlow_kwargs: Optional[dict] = None, byol_kwargs: Optional[dict] = None,
                     asym_views: bool = False, dino_kwargs: Optional[dict] = None,
                     vicreg_kwargs: Optional[dict] = None) -> Dict[str, Any]:
    """Global SSL on the full encoder + head. Learning rates are fixed (never tuned).

    growth_fix6 (opt-in; defaults = historical behaviour, bit for bit):
      ``return_state=True`` -> result["state"] = everything needed to CONTINUE this run exactly
        (AdamW state, projector/decoder weights, RNG states: torch CPU/CUDA, the views generator and the
        loader's generator, global step), all cloned to CPU;
      ``resume=<that state>`` -> continue it for ``epochs`` MORE epochs: same lr (fixed), same optimizer
        moments (``resume["opt"] is None`` -> fresh AdamW), RNG streams restored, and the optimizer-building
        probe forward / extra-module init no longer touch the BN running stats nor the RNG. On CPU,
        run(e1, return_state) + run(e2, resume) == run(e1 + e2) bit for bit.

    growth_fix8: ``progress="<prefixo>"`` imprime 1 linha por época (perda, tempo, passos/s, espera do loader);
    None (padrão) = silencioso como antes. A checagem de gradiente não finito virou UMA sincronização por passo
    (antes: uma por parâmetro); a decisão é a mesma. ``result["augment_device"]`` = device das vistas.

    Losses:
      simclr : L = nt_xent(proj(h1), proj(h2)) + lam * mean_v R(h_v)   (R on head
               embedding h BEFORE any L2 norm; projector is discarded at the end)
      lejepa : L = lam * SIGReg(H) + (1-lam) * inv(H), H=[V,B,D] head embeddings
               (reg='sigreg' -> RealSIGReg; reg='weak_sigreg' -> 'lejepa_weak')
      ae     : L = MSE(dec(feat), view) + lam * R(h)

    growth_v3 (opt-in; padrões = histórico): ``ssl="vicreg"`` (2 vistas, expander do paper em
    ``extra["expander"]``, ``vicreg_kwargs``: expander_dims, sim_coeff, std_coeff, cov_coeff);
    ``freeze_head=True``: a cabeça (projeção antiga) fica CONGELADA no SSL (fora do otimizador, sem grad,
    em eval(): BN da cabeça não atualiza); o gradiente atravessa a cabeça até o encoder; requires_grad dos
    params da cabeça restaurado no fim. ``aug_cfg`` aceita preset ("weak"|"medium"|"strong") ou dict.

    ``amp_dtype`` (opt-in, e.g. torch.bfloat16): forward passes (encoder, head, projector, decoder)
    under autocast; every loss / SIGReg statistic / reduction in fp32 (inputs cast with .float(),
    autocast off); no GradScaler (bf16 range). None = fp32 everywhere (default, unchanged).
    """
    name = method_name(ssl, reg)
    torch.manual_seed(seed)
    device = torch.device(device)
    encoder.to(device).train()
    head.to(device).train()
    head_rg = [p.requires_grad for p in head.parameters()]
    if freeze_head:                 # growth_v3: cabeça antiga congelada (eval, sem grad, fora do otimizador)
        head.eval()
        for p in encoder.parameters():
            p.requires_grad_(True)
        for p in head.parameters():
            p.requires_grad_(False)
    else:
        for p in list(encoder.parameters()) + list(head.parameters()):
            p.requires_grad_(True)
    if not isinstance(aug_cfg, (AugConfig, type(None))):
        aug_cfg = aug_config(aug_cfg)
    vk = dict(vicreg_kwargs or {})

    dk = {"K": 4096, "hidden": 2048, "bottleneck": 256, "use_bn": True, "n_local": 6, "local_frac": 96 / 224,
          "tau_s": 0.1, "tau_t0": 0.04, "tau_t": 0.07, "tau_t_warmup_epochs": 30, "center_m": 0.9,
          "momentum_base": 0.996, "wd_start": None, "wd_end": None, "freeze_last_layer_epochs": 1,
          **(dino_kwargs or {})}
    if ssl == "dino":
        n_views = 2 + int(dk["n_local"])
    V = n_views or {"simclr": 2, "lejepa": 4, "ae": 1, "vicreg": 2, "barlow": 2, "byol": 2}[ssl]
    bk, yk = dict(barlow_kwargs or {}), dict(byol_kwargs or {})
    target: Dict[str, nn.Module] = {}           # growth v3.1: rede-alvo do BYOL (EMA; fora do otimizador)
    if ssl in ("simclr", "vicreg", "barlow", "byol") and V != 2:
        raise ValueError(f"{ssl} uses exactly 2 views")

    # Probe output dims with a dummy batch from the loader-independent shape later.
    extra: Dict[str, nn.Module] = {}
    reg_mod: Optional[nn.Module] = None
    if ssl == "lejepa" and reg == "sigreg":
        reg_mod = RealSIGReg(**(reg_kwargs or {}))
    elif reg != "none":
        from ga_search.modehb.regularizers import build_regularizer  # sibling (I4)
        reg_mod = build_regularizer(reg, **(reg_kwargs or {}))
    if reg_mod is not None:
        reg_mod.to(device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if amp_dtype is not None:
        amp = lambda: torch.autocast(device.type, dtype=amp_dtype)            # noqa: E731
        fp32 = lambda: torch.autocast(device.type, enabled=False)            # noqa: E731
    else:
        amp = fp32 = contextlib.nullcontext
    opt: Optional[torch.optim.Optimizer] = None
    if resume is not None:
        step_offset = int(resume.get("step", step_offset))
        if resume.get("dino_student") is not None:   # v3.1 DINO: o modelo devolvido é a professora; a aluna continua
            encoder.load_state_dict(resume["dino_student"]["encoder"])
            head.load_state_dict(resume["dino_student"]["head"])
    step = step_offset
    reg_key = "ssl/loss_sigreg" if reg == "sigreg" else "ssl/loss_weak_sigreg"
    history: List[Dict[str, float]] = []
    total_imgs, total_time = 0, 0.0
    skipped: List[int] = []           # non-finite loss/grad steps: update skipped (AMP-style)

    def build_opt(x0: Tensor) -> None:
        nonlocal opt
        with torch.no_grad():
            feat = encoder(x0[:2])
            d = head(feat).shape[1]
        if ssl == "simclr":           # growth v3: projector_dims = receita do artigo (ex.: d->d->128 com BN)
            extra["projector"] = (_make_projector(d) if projector_dims is None
                                  else _make_expander(d, [d if int(x) <= 0 else int(x) for x in projector_dims])).to(device)
        elif ssl == "vicreg":
            extra["expander"] = _make_expander(d, vk.get("expander_dims", (2048, 2048, 2048))).to(device)
        elif ssl == "barlow":       # growth v3.1: projetor oficial (sem bias), perda no projetor
            extra["projector"] = _make_barlow_projector(d, bk.get("projector_dims", (8192, 8192, 8192))).to(device)
        elif ssl == "byol":         # growth v3.1: projetor g e preditor q (Linear-BN-ReLU-Linear), alvo = EMA
            pd = [int(v) for v in yk.get("projector_dims", (4096, 256))]
            extra["projector"] = _make_expander(d, pd).to(device)
            extra["predictor"] = _make_expander(pd[-1], yk.get("predictor_dims", (4096, pd[-1]))).to(device)
        elif ssl == "dino":         # growth v3.1: DINOHead (descartada no fim; a professora é devolvida)
            extra["dino_head"] = DINOHead(d, int(dk["K"]), int(dk["hidden"]), int(dk["bottleneck"]),
                                          bool(dk["use_bn"])).to(device)
        elif ssl == "ae":
            extra["decoder"] = _ConvDecoder(feat.shape[1], x0.shape[1]).to(device)
        if projector and ssl not in ("simclr", "vicreg"):   # simclr already has its own projector; Hybrid SPiFiL: SSL losses on a separate projector (Guillotine), not on
            extra["ssl_proj"] = (_make_projector(d, hidden=512, out=128) if projector_dims is None   # the SVM embedding
                                 else _make_expander(d, [d if int(x) <= 0 else int(x) for x in projector_dims])).to(device)
        groups = split_param_groups(encoder, head, extra.values(), new_param_names,
                                    lr_old, lr_new, weight_decay)
        if freeze_head:             # growth_v3: params da cabeça fora do otimizador
            hid = {id(p) for p in head.parameters()}
            groups = [{**g, "params": [p for p in g["params"] if id(p) not in hid]} for g in groups]
            groups = [g for g in groups if g["params"]]
        groups = _scale_conv_groups(groups, encoder, conv_lr_scale)
        if wd_exclude_1d and optimizer == "adamw":     # growth v3.1: bias/BN sem wd (receita dos artigos)
            split = []
            for g_ in groups:
                nd = [p for p in g_["params"] if p.ndim == 1]
                dd = [p for p in g_["params"] if p.ndim != 1]
                if dd:
                    split.append({**g_, "params": dd})
                if nd:
                    split.append({**g_, "params": nd, "weight_decay": 0.0, "name": g_.get("name", "") + "_nowd"})
            groups = split
        if ssl == "barlow" and optimizer == "lars":       # v3.1: Barlow = 2 grupos (pesos 0,2*lr; bias/BN 0,0048*lr)
            split = []
            for g_ in groups:
                nd = [p for p in g_["params"] if p.ndim == 1]
                dd = [p for p in g_["params"] if p.ndim != 1]
                if dd:
                    split.append({**g_, "params": dd, "lr": g_["lr"] * float(bk.get("lr_w_scale", 0.2))})
                if nd:
                    split.append({**g_, "params": nd, "lr": g_["lr"] * float(bk.get("lr_b_scale", 0.0048)),
                                  "name": g_.get("name", "") + "_1d"})
            groups = split
        for g_ in groups:                                # lr de referência do grupo (a agenda multiplica)
            g_["base_lr"] = g_["lr"]
        if optimizer == "lars":                          # growth v3.1: LARS oficial (VICReg/SimCLR)
            lk = {"momentum": 0.9, "eta": 0.001, **(lars_kwargs or {})}
            opt = LARS(groups, lr=0.0, weight_decay=weight_decay, momentum=lk["momentum"], eta=lk["eta"],
                       weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)
        elif optimizer == "adamw":
            opt = torch.optim.AdamW(groups)
        else:
            raise ValueError(f"optimizer {optimizer!r} (adamw|lars)")
        if ssl in ("byol", "dino"):  # alvo/professora = cópia idêntica, sem grad, modo train
            import copy as _copy
            mods_ = (("encoder", encoder), ("head", head),
                     ("projector", extra["projector"]) if ssl == "byol" else ("dino_head", extra["dino_head"]))
            for k_, m_ in mods_:
                t_ = _copy.deepcopy(m_)
                for p_ in t_.parameters():
                    p_.requires_grad_(False)
                target[k_] = t_.train()

    if resume is not None and resume.get("rng") is not None:
        _rng_restore(resume["rng"], device, gen, loader)

    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        sums: Dict[str, float] = {}
        n_batches, ep_imgs, wait_total = 0, 0, 0.0
        t_wait = time.perf_counter()
        for bi, batch in enumerate(loader):
            if max_steps_per_epoch is not None and bi >= max_steps_per_epoch:
                break
            wait = time.perf_counter() - t_wait
            wait_total += wait
            t_step = time.perf_counter()
            x = _unpack(batch).to(device, non_blocking=True).float()
            if opt is None:
                if resume is None:
                    build_opt(x)
                else:                       # growth_fix6: no side effect (BN stats / RNG) at the resume point
                    with _no_side_effects(encoder, head, device):
                        build_opt(x)
                    for k, sd in (resume.get("extra") or {}).items():
                        if k in extra:
                            extra[k].load_state_dict(sd)
                    for k, sd in (resume.get("byol_target") or {}).items():
                        if k in target:
                            target[k].load_state_dict(sd)
                        elif k == "center":
                            target["center"] = sd.to(device).clone()
                    if resume.get("opt") is not None:
                        opt.load_state_dict(resume["opt"])
            if ssl == "dino":
                ls_ = max(8, int(round(x.shape[-1] * float(dk["local_frac"]))))
                gviews, lviews = dino_views(x, gen, int(dk["n_local"]), ls_)
                views = gviews + lviews
            else:
                views = make_views_asym(x, ASYM_VIEWS, gen) if asym_views else make_views(x, V, aug_cfg, generator=gen)
            B = x.shape[0]
            parts: Dict[str, Tensor] = {}
            if ssl == "dino":               # growth v3.1: DINO (aluna por resolução; professora só nas globais)
                spe_d = int(dk.get("steps_per_epoch") or 1)
                ep_glob = step // max(spe_d, 1)
                wte = int(dk["tau_t_warmup_epochs"])
                tau_t = float(dk["tau_t0"]) + (float(dk["tau_t"]) - float(dk["tau_t0"])) * min(ep_glob, wte) / max(wte, 1) \
                    if wte > 0 else float(dk["tau_t"])
                with amp():
                    sg = extra["dino_head"](head(encoder(torch.cat(gviews, 0))))
                    s_out = list(sg.split(B, 0))
                    if lviews:
                        sl = extra["dino_head"](head(encoder(torch.cat(lviews, 0))))
                        s_out += list(sl.split(B, 0))
                with torch.no_grad(), amp():
                    tg = target["dino_head"](target["head"](target["encoder"](torch.cat(gviews, 0)))).float()
                t_out = list(tg.split(B, 0))
                if "center" not in target:
                    target["center"] = torch.zeros(1, tg.shape[1], device=device)
                with fp32():
                    total = dino_loss(s_out, t_out, target["center"], float(dk["tau_s"]), tau_t)
                with torch.no_grad():       # centro DEPOIS da perda (média das globais da professora)
                    target["center"].mul_(float(dk["center_m"])).add_(tg.mean(0, keepdim=True), alpha=1 - float(dk["center_m"]))
                parts["ssl/loss_dino"] = total
                parts["ssl/dino_tau_t"] = torch.tensor(tau_t)
                pos, neg = 0.0, 0.0
            elif ssl == "ae":
                v = views[0]
                with amp():
                    feat = encoder(v)
                    h = head(feat)
                    rec = extra["decoder"](feat, v.shape[-2:])
                    hr = extra["ssl_proj"](h) if (projector and reg_mod is not None) else h
                with fp32():
                    parts["ssl/loss_ae"] = F.mse_loss(rec.float(), v.float())
                    total = parts["ssl/loss_ae"]
                    if reg_mod is not None:
                        parts[reg_key] = reg_mod(hr.float())
                        total = total + sigreg_lambda * parts[reg_key]
            else:
                with amp():
                    h_all = head(encoder(torch.cat(views, 0)))       # [V*B, D]
                    if projector and ssl not in ("simclr", "vicreg"):
                        h_all = extra["ssl_proj"](h_all)
                h_all = h_all.float()
                hs = list(h_all.split(B, 0))
                if ssl == "simclr":
                    proj = extra["projector"]
                    with amp():
                        z1, z2 = proj(hs[0]), proj(hs[1])
                    z1, z2 = z1.float(), z2.float()
                    parts["ssl/loss_simclr"] = nt_xent(z1, z2, tau)
                    total = parts["ssl/loss_simclr"]
                    if reg_mod is not None:
                        # regularizer on the raw head embedding (pre-L2, pre-projector)
                        reg_in = [z1, z2] if projector else hs   # projector gene: reg off the SVM embedding
                        parts[reg_key] = torch.stack([reg_mod(hv) for hv in reg_in]).mean()
                        total = total + sigreg_lambda * parts[reg_key]
                    pos, neg = _pos_neg_sim(z1.detach(), z2.detach())
                elif ssl == "barlow":       # growth v3.1
                    with amp():
                        z1, z2 = extra["projector"](hs[0]), extra["projector"](hs[1])
                    with fp32():
                        total, bp = barlow_loss(z1, z2, float(bk.get("lambd", 0.0051)))
                    parts["ssl/loss_barlow_on"], parts["ssl/loss_barlow_off"] = bp["on"], bp["off"]
                    pos, neg = _pos_neg_sim(z1.detach().float(), z2.detach().float())
                elif ssl == "byol":         # growth v3.1: online (g, q) x alvo EMA (modo train, sem grad)
                    with amp():
                        p1 = extra["predictor"](extra["projector"](hs[0]))
                        p2 = extra["predictor"](extra["projector"](hs[1]))
                    with torch.no_grad(), amp():
                        tz = target["projector"](target["head"](target["encoder"](torch.cat(views, 0)))).float()
                    t1, t2 = tz.split(B, 0)
                    with fp32():
                        total = byol_loss(p1, t2) + byol_loss(p2, t1)
                    pos, neg = _pos_neg_sim(p1.detach().float(), t2.detach())
                elif ssl == "vicreg":       # growth_v3
                    exp_ = extra["expander"]
                    with amp():
                        z1, z2 = exp_(hs[0]), exp_(hs[1])
                    z1, z2 = z1.float(), z2.float()
                    with fp32():
                        total, vp = vicreg_loss(z1, z2, sim_coeff=float(vk.get("sim_coeff", 25.0)),
                                                std_coeff=float(vk.get("std_coeff", 25.0)),
                                                cov_coeff=float(vk.get("cov_coeff", 1.0)))
                    parts["ssl/loss_vicreg_inv"] = vp["inv"]
                    parts["ssl/loss_vicreg_var"] = vp["var"]
                    parts["ssl/loss_vicreg_cov"] = vp["cov"]
                    if reg_mod is not None:
                        parts[reg_key] = torch.stack([reg_mod(hv) for hv in hs]).mean()
                        total = total + sigreg_lambda * parts[reg_key]
                    pos, neg = _pos_neg_sim(z1.detach(), z2.detach())
                else:  # lejepa
                    H = torch.stack(hs, 0)                          # [V, B, D]
                    parts["ssl/loss_inv"] = invariance_loss(H)
                    if reg == "sigreg":
                        parts[reg_key] = reg_mod(H)
                    else:
                        parts[reg_key] = torch.stack([reg_mod(hv) for hv in hs]).mean()
                    total = sigreg_lambda * parts[reg_key] + (1 - sigreg_lambda) * parts["ssl/loss_inv"]
                    pos, neg = _pos_neg_sim(hs[0].detach(), hs[1].detach())
            parts["ssl/loss_total"] = total

            opt.zero_grad(set_to_none=True)
            msg = None
            if not torch.isfinite(total):
                msg = (f"{name}: non-finite loss at epoch {ep} step {step}: "
                       + ", ".join(f"{k}={float(v):.4g}" for k, v in parts.items()))
            else:
                total.backward()
                named = [(n, p) for n, p in list(encoder.named_parameters()) + list(head.named_parameters())
                         + [(f"extra.{n}", p) for m in extra.values() for n, p in m.named_parameters()]
                         if p.grad is not None]
                # growth_fix8: 1 sincronização (antes: 1 por parâmetro); só procura os nomes se algo não é finito
                all_fin = bool(torch.stack([torch.isfinite(p.grad).all() for _, p in named]).all()) if named else True
                gnorm_bad = [] if all_fin else [n for n, p in named if not torch.isfinite(p.grad).all()]
                if gnorm_bad:
                    msg = f"{name}: non-finite gradients at step {step} in {gnorm_bad[:5]}"
            if msg is not None:          # skip this update; weights stay as they were
                opt.zero_grad(set_to_none=True)
                skipped.append(step)
                if len(skipped) == 1:     # alert once per run (W&B alerts are rate-limited)
                    _alert(logger, "SSL NaN/Inf (step skipped)", msg)
                step += 1
                t_wait = time.perf_counter()
                continue
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                               grad_clip)
            if ssl == "dino":               # freeze_last_layer + wd em cosseno só no grupo regularizado
                if step // max(int(dk.get("steps_per_epoch") or 1), 1) < int(dk["freeze_last_layer_epochs"]):
                    for p_ in extra["dino_head"].last_layer.parameters():
                        p_.grad = None
                if dk.get("wd_start") is not None and lr_schedule is not None:
                    tt = min(1.0, step / max(1, int(lr_schedule["total_steps"])))
                    wd_t = float(dk["wd_end"]) + (float(dk["wd_start"]) - float(dk["wd_end"])) * 0.5 * (1 + math.cos(math.pi * tt))
                    for g in opt.param_groups:
                        if not g.get("name", "").endswith("_nowd"):     # 1-D (bias/BN) seguem sem wd
                            g["weight_decay"] = wd_t
            if lr_schedule is not None:                  # growth v3.1: agenda por passo GLOBAL (retomada continua)
                fac = schedule_factor(step, lr_schedule["total_steps"], lr_schedule.get("warmup_steps", 0),
                                      float(lr_schedule.get("end_ratio", 1e-3)), float(lr_schedule.get("start_ratio", 0.0)))
                for g in opt.param_groups:
                    g["lr"] = g.get("base_lr", g["lr"]) * fac
            log_now = logger is not None and (step % max(log_every, 1) == 0)
            if log_now:
                gn = _per_layer_norms(encoder, "grad_norm/encoder", grad=True)
                gn.update(_per_layer_norms(head, "grad_norm/head", grad=True))
            opt.step()
            if ssl == "dino":               # v3.1: EMA da professora (momentum 0,996 -> 1 em cosseno, por passo)
                K = int((lr_schedule or {}).get("total_steps") or dk.get("total_steps") or 1)
                m_k = byol_tau(step, K, float(dk["momentum_base"]))
                with torch.no_grad():
                    for k_, online_ in (("encoder", encoder), ("head", head), ("dino_head", extra["dino_head"])):
                        for pt, po in zip(target[k_].parameters(), online_.parameters()):
                            pt.mul_(m_k).add_(po.detach(), alpha=1.0 - m_k)
                parts["ssl/dino_momentum"] = torch.tensor(m_k)
            if ssl == "byol":               # v3.1: EMA DEPOIS do passo, por passo (tau_k em cosseno até 1)
                K = yk.get("total_steps") or (lr_schedule or {}).get("total_steps")
                if not K:
                    try:
                        spe_ = len(loader) if max_steps_per_epoch is None else min(len(loader), int(max_steps_per_epoch))
                    except TypeError:
                        spe_ = int(max_steps_per_epoch or 1)
                    K = max(1, int(epochs) * int(spe_) + int(step_offset))
                tau_k = byol_tau(step, K, float(yk.get("tau_base", 0.996)))
                with torch.no_grad():
                    for k_, online_ in (("encoder", encoder), ("head", head), ("projector", extra["projector"])):
                        for pt, po in zip(target[k_].parameters(), online_.parameters()):
                            pt.mul_(tau_k).add_(po.detach(), alpha=1.0 - tau_k)
                parts["ssl/byol_tau"] = torch.tensor(tau_k)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t_step

            n_batches += 1
            ep_imgs += B
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            if ssl != "ae":
                sums["ssl/pos_sim"] = sums.get("ssl/pos_sim", 0.0) + pos
                sums["ssl/neg_sim"] = sums.get("ssl/neg_sim", 0.0) + neg
            if log_now:
                rec = {k: float(v.detach()) for k, v in parts.items()}
                if ssl != "ae":
                    rec["ssl/pos_sim"], rec["ssl/neg_sim"] = pos, neg
                rec.update(gn)
                rec.update(_per_layer_norms(encoder, "weight_norm/encoder", grad=False))
                rec.update(_per_layer_norms(head, "weight_norm/head", grad=False))
                for g in opt.param_groups:
                    rec[f"lr/{g['name']}"] = g["lr"]
                rec["perf/throughput_img_s"] = B * V / max(dt, 1e-9)
                rec["perf/dataloader_wait_s"] = wait
                rec["ssl/method"] = name
                rec["ssl/epoch"] = ep
                logger.log(rec, step=step)
            step += 1
            t_wait = time.perf_counter()

        ep_time = time.perf_counter() - ep_t0
        total_imgs += ep_imgs
        total_time += ep_time
        ep_rec = {k: v / max(n_batches, 1) for k, v in sums.items()}
        ep_rec.update({"perf/epoch_time_s": ep_time,
                       "perf/epoch_throughput_img_s": ep_imgs * V / max(ep_time, 1e-9),
                       "perf/epoch_dataloader_wait_s": wait_total,
                       "ssl/epoch": ep})
        history.append(ep_rec)
        if progress is not None:          # growth_fix8: o log não fica mudo durante o SSL
            print(f"{progress} {name} época {ep + 1}/{epochs} passos={n_batches} "
                  f"perda={ep_rec.get('ssl/loss_total', float('nan')):.5g} t={ep_time:.1f}s "
                  f"passos/s={n_batches / max(ep_time, 1e-9):.3f} espera_loader={wait_total:.2f}s "
                  f"pulados={len(skipped)}", flush=True)
        if logger is not None:
            logger.log({f"epoch/{k}" if not k.startswith("perf/epoch") else k: v
                        for k, v in ep_rec.items()}, step=step)

    n_steps = step - step_offset
    if n_steps and len(skipped) > max_skip_frac * n_steps:
        msg = f"{name}: {len(skipped)}/{n_steps} SSL steps non-finite (> {max_skip_frac:.0%})"
        _alert(logger, "SSL NaN/Inf", msg)
        raise FloatingPointError(msg)
    state = None
    if return_state:                 # growth_fix6: continuation state (CPU clones), BEFORE discarding extras
        state = {"format": "ssl_resume/v1", "ssl": ssl, "reg": reg, "n_views": V, "step": int(step),
                 "opt": _cpu_clone(opt.state_dict()) if opt is not None else None,
                 "extra": {k: _cpu_clone(m.state_dict()) for k, m in extra.items()},
                 **({"byol_target": {k: (_cpu_clone(m.state_dict()) if isinstance(m, nn.Module) else m.detach().cpu().clone())
                                     for k, m in target.items()}} if target else {}),
                 "rng": _rng_capture(device, gen, loader)}
    # Discard projector / decoder; encoder + head stay trained.
    if ssl == "dino" and target and state is not None:     # retomada: a ALUNA continua de onde parou
        state["dino_student"] = {"encoder": _cpu_clone(encoder.state_dict()), "head": _cpu_clone(head.state_dict())}
    if ssl == "dino" and target:     # DINO: devolve a PROFESSORA (encoder + cabeça), como checkpoint_key=teacher
        with torch.no_grad():
            encoder.load_state_dict(target["encoder"].state_dict())
            head.load_state_dict(target["head"].state_dict())
    extra.clear()
    target.clear()                   # BYOL: devolve o encoder ONLINE (o alvo é descartado)
    encoder.eval()
    head.eval()
    if freeze_head:                 # growth_v3: requires_grad original da cabeça
        for p, rg in zip(head.parameters(), head_rg):
            p.requires_grad_(rg)
    final = history[-1] if history else {}
    return {
        "method": name,
        "ssl": ssl,
        "reg": reg,
        "n_views": V,
        "epochs": epochs,
        "steps": step - step_offset,
        "skipped_steps": len(skipped),
        "final_loss_total": final.get("ssl/loss_total", float("nan")),
        "final": final,
        "history": history,
        "throughput_img_s": total_imgs * V / max(total_time, 1e-9),
        "train_time_s": total_time,
        "lr_old": lr_old,
        "lr_new": lr_new,
        "sigreg_lambda": sigreg_lambda,
        "augment_device": str(device),
        **({"freeze_head": True} if freeze_head else {}),
        **({"aug": aug_record(aug_cfg)} if aug_cfg is not None else {}),
        **({"state": state} if return_state else {}),
    }


# ───────────────────────────────────────────────────── growth_fix6: exact continuation helpers
def _loader_generator(loader) -> Optional[torch.Generator]:
    """The ``torch.Generator`` that drives the loader's shuffle (DataLoader or a wrapper with ``.loader``)."""
    seen = 0
    while loader is not None and seen < 8:
        g = getattr(loader, "generator", None)
        if isinstance(g, torch.Generator):
            return g
        loader, seen = getattr(loader, "loader", None), seen + 1
    return None


def _cpu_clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_clone(v) for v in obj)
    return obj


def _rng_capture(device, gen: torch.Generator, loader) -> Dict[str, Any]:
    st: Dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "views": gen.get_state()}
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device))
    lg = _loader_generator(loader)
    st["loader"] = lg.get_state() if lg is not None else None
    return st


def _rng_restore(st: Dict[str, Any], device, gen: torch.Generator, loader) -> None:
    torch.set_rng_state(st["torch_cpu"])
    gen.set_state(st["views"])
    if st.get("torch_cuda") is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["torch_cuda"], torch.device(device))
    lg = _loader_generator(loader)
    if lg is not None and st.get("loader") is not None:
        lg.set_state(st["loader"])


@contextlib.contextmanager
def _no_side_effects(encoder: nn.Module, head: nn.Module, device):
    """Buffers (BN running stats / num_batches_tracked) and RNG states restored on exit."""
    bufs = [(b, b.detach().clone()) for m in (encoder, head) for b in m.buffers()]
    cpu = torch.get_rng_state()
    dev = torch.device(device)
    cuda = torch.cuda.get_rng_state(dev) if dev.type == "cuda" and torch.cuda.is_available() else None
    try:
        yield
    finally:
        with torch.no_grad():
            for b, v in bufs:
                b.copy_(v)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, dev)
