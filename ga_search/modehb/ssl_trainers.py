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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.lejepa_loss import RealSIGReg, invariance_loss

__all__ = [
    "AugConfig",
    "nt_xent",
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

    def rnd(*shape):
        return torch.rand(*shape, generator=generator).to(dev, dt)

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
        views.append(v.clamp(0.0, 1.0))
    return views


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SSL = ("simclr", "lejepa", "ae")
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
                     projector: bool = False, amp_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    """Global SSL on the full encoder + head. Learning rates are fixed (never tuned).

    Losses:
      simclr : L = nt_xent(proj(h1), proj(h2)) + lam * mean_v R(h_v)   (R on head
               embedding h BEFORE any L2 norm; projector is discarded at the end)
      lejepa : L = lam * SIGReg(H) + (1-lam) * inv(H), H=[V,B,D] head embeddings
               (reg='sigreg' -> RealSIGReg; reg='weak_sigreg' -> 'lejepa_weak')
      ae     : L = MSE(dec(feat), view) + lam * R(h)

    ``amp_dtype`` (opt-in, e.g. torch.bfloat16): forward passes (encoder, head, projector, decoder)
    under autocast; every loss / SIGReg statistic / reduction in fp32 (inputs cast with .float(),
    autocast off); no GradScaler (bf16 range). None = fp32 everywhere (default, unchanged).
    """
    name = method_name(ssl, reg)
    torch.manual_seed(seed)
    device = torch.device(device)
    encoder.to(device).train()
    head.to(device).train()
    for p in list(encoder.parameters()) + list(head.parameters()):
        p.requires_grad_(True)

    V = n_views or {"simclr": 2, "lejepa": 4, "ae": 1}[ssl]
    if ssl == "simclr" and V != 2:
        raise ValueError("simclr uses exactly 2 views")

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
        if ssl == "simclr":
            extra["projector"] = _make_projector(d).to(device)
        elif ssl == "ae":
            extra["decoder"] = _ConvDecoder(feat.shape[1], x0.shape[1]).to(device)
        if projector and ssl != "simclr":   # simclr already has its own projector; Hybrid SPiFiL: SSL losses on a separate projector (Guillotine), not on
            extra["ssl_proj"] = _make_projector(d, hidden=512, out=128).to(device)   # the SVM embedding
        groups = split_param_groups(encoder, head, extra.values(), new_param_names,
                                    lr_old, lr_new, weight_decay)
        groups = _scale_conv_groups(groups, encoder, conv_lr_scale)
        opt = torch.optim.AdamW(groups)

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
                build_opt(x)
            views = make_views(x, V, aug_cfg, generator=gen)
            B = x.shape[0]
            parts: Dict[str, Tensor] = {}
            if ssl == "ae":
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
                    if projector and ssl != "simclr":
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
                gnorm_bad = [n for n, p in list(encoder.named_parameters()) + list(head.named_parameters())
                             + [(f"extra.{n}", p) for m in extra.values() for n, p in m.named_parameters()]
                             if p.grad is not None and not torch.isfinite(p.grad).all()]
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
            log_now = logger is not None and (step % max(log_every, 1) == 0)
            if log_now:
                gn = _per_layer_norms(encoder, "grad_norm/encoder", grad=True)
                gn.update(_per_layer_norms(head, "grad_norm/head", grad=True))
            opt.step()
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
        if logger is not None:
            logger.log({f"epoch/{k}" if not k.startswith("perf/epoch") else k: v
                        for k, v in ep_rec.items()}, step=step)

    n_steps = step - step_offset
    if n_steps and len(skipped) > max_skip_frac * n_steps:
        msg = f"{name}: {len(skipped)}/{n_steps} SSL steps non-finite (> {max_skip_frac:.0%})"
        _alert(logger, "SSL NaN/Inf", msg)
        raise FloatingPointError(msg)
    # Discard projector / decoder; encoder + head stay trained.
    extra.clear()
    encoder.eval()
    head.eval()
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
    }
