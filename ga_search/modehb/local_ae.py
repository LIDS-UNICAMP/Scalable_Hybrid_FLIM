"""Local autoencoder for the newly added layer (progressive MO-DEHB, Fase B).

The prefix (previously accepted blocks) is frozen; only ``new_block`` and a
temporary decoder are trained. The decoder is discarded at the end: the caller
keeps ``new_block`` (trained in place) and receives a metrics dict.

Targets:
  pixel   -> reconstruct the input image from new_block(prefix(x))
  feature -> reconstruct prefix(x) (detached) from new_block(prefix(x))
  masked  -> zero 2x2-patch blocks of the input, reconstruct masked patches
             (src.losses.masked_object_mse.masked_patch_mse, per-patch normalised target)

The ResNetDecoder used by methods/autoencoder/progressive.py needs a full
architecture dict + 1280-d projection; for a single local block a minimal
conv/upsample decoder is enough and works for any target shape.
"""
from __future__ import annotations

import contextlib
import math
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.losses.masked_object_mse import masked_patch_mse, patch_to_pixel, sample_block_mask

TARGETS = ("pixel", "feature", "masked")


class _LocalDecoder(nn.Module):
    """conv3x3 -> ReLU -> bilinear resize to target size -> conv1x1 to target channels."""

    def __init__(self, in_ch: int, out_ch: int, out_hw: tuple[int, int], hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(16, min(128, in_ch))
        self.out_hw = tuple(int(v) for v in out_hw)
        self.body = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, z):
        h = self.body(z)
        if tuple(h.shape[-2:]) != self.out_hw:
            h = F.interpolate(h, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.head(h)


def _batch_x(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        for key in ("image", "images", "x", "input"):
            if key in batch:
                return _batch_x(batch[key])
        raise KeyError(f"no image key in batch dict: {list(batch)}")
    if isinstance(batch, (list, tuple)):
        return _batch_x(batch[0])
    raise TypeError(f"unsupported batch type {type(batch)}")


def _freeze(prefix: nn.Module) -> None:
    for p in prefix.parameters():
        p.requires_grad_(False)
    prefix.eval()
    assert all(not p.requires_grad for p in prefix.parameters()), "prefix not frozen"
    assert not any(m.training for m in prefix.modules()), "prefix not in eval mode"


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy & Vetterli effective rank over channels of z [B,C,H,W] (or [N,C])."""
    if z.dim() == 4:
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    z = torch.nan_to_num(z.double(), nan=0.0, posinf=0.0, neginf=0.0)   # diagnostic: never raise
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    if s.sum() <= eps:
        return 0.0
    p = s / s.sum()
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def _patch_size(h: int, w: int) -> int:
    return max(1, min(16, h // 4, w // 4))


def _fail(logger, text: str, raise_: bool = True):
    if logger is not None:
        try:
            logger.alert("local_ae non-finite", text, "ERROR" if raise_ else "WARN")
        except Exception:  # pragma: no cover - alerting must never hide the real error
            pass
    if raise_:
        raise FloatingPointError(text)


def train_local_ae(
    prefix: nn.Module,
    new_block: nn.Module,
    loader,
    target: str = "pixel",
    epochs: int = 1,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    logger=None,
    seed: int = 0,
    mask_ratio: float = 0.4,
    rank_batches: int = 4,
    max_skip_frac: float = 1.0,
    conv_lr_scale: float = 1.0,
    decoder_warmup_frac: float = 0.0,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Train ``new_block`` as a local AE on top of the frozen ``prefix``.

    ``prefix`` stays frozen afterwards (requires_grad False, eval). ``new_block``
    is updated in place and left in eval mode. The temporary decoder is deleted.
    ``amp_dtype`` (opt-in, e.g. torch.bfloat16): forward passes under autocast, the reconstruction
    loss in fp32; None (default) = fp32 everywhere, unchanged.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    device = torch.device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    prefix.to(device)
    new_block.to(device)
    _freeze(prefix)
    prefix_snapshot = [p.detach().clone() for p in prefix.parameters()]

    # probe shapes to build the decoder
    x0 = _batch_x(next(iter(loader))).to(device)
    with torch.no_grad():
        f0 = prefix(x0)
        z0 = new_block(f0)
    tgt0 = f0 if target == "feature" else x0
    decoder = _LocalDecoder(z0.shape[1], tgt0.shape[1], tuple(tgt0.shape[-2:])).to(device)

    # Hybrid SPiFiL kernel protection: SPiFiL conv weights get lr * conv_lr_scale (0 = frozen)
    conv_ids = {id(p) for m in new_block.modules() if isinstance(m, nn.Conv2d) for p in m.parameters()}
    if float(conv_lr_scale) == 0.0:
        for p in new_block.parameters():
            if id(p) in conv_ids:
                p.requires_grad_(False)
    rest = [p for p in new_block.parameters() if p.requires_grad and id(p) not in conv_ids]
    convs = [p for p in new_block.parameters() if p.requires_grad and id(p) in conv_ids]
    trainable = rest + convs + list(decoder.parameters())
    # Hybrid SPiFiL layer schedule: first ``decoder_warmup_frac`` of the epochs train ONLY the
    # decoder (SPiFiL block frozen, decoder adapts to it); then the block joins (convs at lr*scale)
    groups = [{"params": list(decoder.parameters()), "lr": lr, "block": False}]
    if rest:
        groups.append({"params": rest, "lr": lr, "block": True})
    if convs:
        groups.append({"params": convs, "lr": lr * float(conv_lr_scale), "block": True})
    base_lr = [g["lr"] for g in groups]
    n_warm = int(round(float(decoder_warmup_frac) * epochs))
    opt = torch.optim.Adam(groups, lr=lr)
    named = [(f"new_block.{n}", p) for n, p in new_block.named_parameters() if p.requires_grad]

    amp = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if amp_dtype is not None else contextlib.nullcontext
    step, initial_loss, epoch_losses = 0, None, []
    skipped = []                     # steps whose loss/grad was non-finite: no update (AMP-style skip)
    t_start = time.perf_counter()
    for epoch in range(epochs):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = 0.0 if (g.get("block") and epoch < n_warm) else b
        new_block.train()
        decoder.train()
        prefix.eval()
        losses = []
        t_ep = time.perf_counter()
        for batch in loader:
            x = _batch_x(batch).to(device, non_blocking=True)
            if target == "masked":
                H, W = x.shape[-2:]
                patch = _patch_size(H, W)
                obj = torch.ones(x.shape[0], H // patch, W // patch, dtype=torch.bool)
                masked = sample_block_mask(obj, mask_ratio, generator=gen).to(device)
                vis = 1.0 - patch_to_pixel(masked, H, W, patch).to(x.dtype)
                with torch.no_grad(), amp():
                    feats = prefix(x * vis)
                with amp():
                    rec = decoder(new_block(feats))
                loss = masked_patch_mse(rec.float(), x, masked, patch=patch)
            else:
                with torch.no_grad(), amp():
                    feats = prefix(x)
                tgt = feats.detach().float() if target == "feature" else x
                with amp():
                    rec = decoder(new_block(feats))
                loss = F.mse_loss(rec.float(), tgt)

            opt.zero_grad(set_to_none=True)
            bad = None
            if not torch.isfinite(loss):
                bad = f"recon loss non-finite ({loss.item()}) at epoch {epoch} step {step}"
            else:
                loss.backward()
            log = {"ae/recon_loss": float(loss.item()), "ae/lr": opt.param_groups[0]["lr"]}
            for name, p in named:
                if bad is None and p.grad is not None:
                    g = float(p.grad.detach().norm())
                    if not math.isfinite(g):
                        bad = f"grad norm non-finite in {name} at step {step}"
                    log[f"ae/grad_norm/{name}"] = g
            dec_g = [p.grad.detach().norm() for p in decoder.parameters() if p.grad is not None]
            if dec_g:
                log["ae/grad_norm/decoder"] = float(torch.stack(dec_g).norm())
                if bad is None and not math.isfinite(log["ae/grad_norm/decoder"]):
                    bad = f"decoder grad norm non-finite at step {step}"
            if bad is not None:      # skip the update (weights untouched), keep training
                skipped.append(step)
                opt.zero_grad(set_to_none=True)
                if len(skipped) == 1:     # alert once per layer
                    _fail(logger, bad + " -> step skipped", raise_=False)
                log["ae/skipped_steps"] = len(skipped)
                if logger is not None:
                    logger.log(log, step=step)
                step += 1
                continue
            opt.step()
            if initial_loss is None:
                initial_loss = log["ae/recon_loss"]
            losses.append(log["ae/recon_loss"])
            if logger is not None:
                logger.log(log, step=step)
            step += 1
        ep_time = time.perf_counter() - t_ep
        epoch_losses.append(losses)
        if logger is not None:
            logger.log({"ae/epoch": epoch, "ae/epoch_time_s": ep_time,
                        "ae/epoch_loss": sum(losses) / max(1, len(losses))}, step=step)
    time_s = time.perf_counter() - t_start

    # feature quality: effective rank of new_block output channels (clean input)
    new_block.eval()
    zs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= rank_batches:
                break
            zs.append(new_block(prefix(_batch_x(batch).to(device))).float().cpu())
    z = torch.cat(zs)
    erank = effective_rank(z)

    # discard decoder; verify prefix untouched
    del opt, decoder, trainable
    for before, p in zip(prefix_snapshot, prefix.parameters()):
        assert torch.equal(before, p.detach()), "prefix weights changed during local AE"
        assert p.grad is None, "prefix received gradients"
    del prefix_snapshot

    if step and len(skipped) > max_skip_frac * step:
        _fail(logger, f"local AE: {len(skipped)}/{step} steps non-finite (> {max_skip_frac:.0%})")
    last = torch.tensor(epoch_losses[-1] or [float("nan")], dtype=torch.float64)
    final_loss = float(last.mean())
    return {
        "target": target,
        "initial_recon_loss": float("nan") if initial_loss is None else float(initial_loss),
        "final_recon_loss": final_loss,
        "recon_ratio": final_loss / initial_loss if initial_loss else float("nan"),
        "stability": float(last.std(unbiased=False)) if last.numel() > 1 else 0.0,
        "time_s": time_s,
        "effective_rank": erank,
        "effective_rank_ratio": erank / z.shape[1],
        "out_channels": int(z.shape[1]),
        "steps": step,
        "skipped_steps": len(skipped),
        "epochs": epochs,
    }
