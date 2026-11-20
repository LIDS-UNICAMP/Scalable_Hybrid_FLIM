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
    """conv3x3 -> ReLU -> bilinear resize to target size -> conv1x1 to target channels.

    growth_v31_hotfix_ae_diverge: ``in_scale`` (opcional, buffer FIXO) = 1/desvio por canal de z no lote-sonda;
    o decoder lê ``(z - in_mu) * in_scale``. É uma reparametrização da 1ª conv do decoder (mesma classe de
    funções; o decoder é descartado), que tira a escala de z (e o fan-in C) do passo do Adam."""

    def __init__(self, in_ch: int, out_ch: int, out_hw: tuple[int, int], hidden: int | None = None,
                 in_mu: torch.Tensor | None = None, in_scale: torch.Tensor | None = None):
        super().__init__()
        hidden = hidden or max(16, min(128, in_ch))
        self.out_hw = tuple(int(v) for v in out_hw)
        self.body = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(hidden, out_ch, 1)
        self.normed = in_scale is not None
        if self.normed:
            self.register_buffer("in_mu", in_mu.detach().reshape(1, -1, 1, 1).clone())
            self.register_buffer("in_scale", in_scale.detach().reshape(1, -1, 1, 1).clone())

    def forward(self, z):
        if self.normed:
            z = (z - self.in_mu.to(z.dtype)) * self.in_scale.to(z.dtype)
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


def ae_guard_verdict(losses: list, prev_mean, initial_loss, ratio: float, last_epoch: bool):
    """hotfix ae_diverge: veredito da guarda numa época (razão + tendência; nunca o valor absoluto).
    Devolve (diverged, motivo, média da época)."""
    ep_mean = sum(losses) / len(losses) if losses else float("nan")
    if not losses or not math.isfinite(ep_mean):
        return True, "perda não finita", ep_mean
    if not initial_loss:
        return False, None, ep_mean
    q = max(1, len(losses) // 4)
    head_m, tail_m = sum(losses[:q]) / q, sum(losses[-q:]) / q
    ratio_ep = ep_mean / float(initial_loss)
    rising = (prev_mean is not None and ep_mean > prev_mean) or (prev_mean is None and tail_m > head_m)
    if ratio_ep > float(ratio) and rising:
        return True, (f"média/inicial = {ratio_ep:.4g} > {float(ratio):g} e subindo (média {ep_mean:.5g}, "
                      f"anterior {prev_mean}, início/fim da época {head_m:.5g}/{tail_m:.5g})"), ep_mean
    final_ratio = tail_m / float(initial_loss)
    if last_epoch and final_ratio > float(ratio):
        return True, (f"fim do AE: perda final (último quarto) / inicial = {final_ratio:.4g} > {float(ratio):g} "
                      f"({tail_m:.5g} vs {float(initial_loss):.5g})"), ep_mean
    return False, None, ep_mean


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
    progress: str | None = None,
    grad_clip: float | None = None,
    dec_in_norm: bool = False,
    lr_scale_fanin: bool = False,
    warmup_steps: int = 0,
    diverge_ratio: float | None = None,
) -> dict:
    """Train ``new_block`` as a local AE on top of the frozen ``prefix``.

    ``prefix`` stays frozen afterwards (requires_grad False, eval). ``new_block``
    is updated in place and left in eval mode. The temporary decoder is deleted.
    ``amp_dtype`` (opt-in, e.g. torch.bfloat16): forward passes under autocast, the reconstruction
    loss in fp32; None (default) = fp32 everywhere, unchanged.
    growth_fix8: ``progress="<prefixo>"`` imprime 1 linha por época (perda, tempo, passos/s); None = silencioso.
    As normas de gradiente por parâmetro saem numa única sincronização (mesmos valores, mesma ordem).

    growth_v31_hotfix_ae_diverge (todos desligados por padrão = histórico bit a bit):
      ``dec_in_norm``: entrada do decoder padronizada por canal com média/desvio FIXOS do lote-sonda (buffer,
        sem gradiente; reparametrização do decoder descartável). Causa medida da divergência: com z de RMS
        ~16 e fan-in 9*C (C=455..500), o 1º passo do Adam (|Δw| ~ lr por peso, coerente em sinal) move a saída
        do decoder em ~lr*9C*|z| >> |x| -> perda 20 -> 6000 no passo 2.
      ``grad_clip``: clip da norma global (bloco + decoder) antes do passo.
      ``lr_scale_fanin``: lr dos pesos do DECODER * min(1, sqrt(ref/fan_in)), ref = 9*16 (a 1ª conv do decoder tem
        fan-in 9*C; mesmo com a entrada padronizada, o passo coerente do Adam move a saída ~lr*9C). O bloco não muda.
      ``warmup_steps``: lr linear de 1/(n+1) até 1 nos primeiros n passos.
      ``diverge_ratio``: guarda por RAZÃO e TENDÊNCIA (nunca o valor absoluto): diverge se (a) perda não
        finita; (b) média da época / perda inicial > ratio E subindo (média > a da época anterior; na 1ª época,
        último quarto > primeiro quarto); (c) no fim, último quarto da última época / inicial > ratio.
        "Começa alto e cai" passa. ``diverged=True`` -> o chamador restaura o bloco. None = sem checagem.
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
    dec_kw = {}
    if dec_in_norm:                      # hotfix ae_diverge: escala fixa da entrada do decoder (lote-sonda)
        zc = z0.detach().float().transpose(0, 1).reshape(z0.shape[1], -1)
        zc = torch.nan_to_num(zc, nan=0.0, posinf=0.0, neginf=0.0)
        mu = zc.mean(1)
        sd = zc.std(1, unbiased=False)
        floor = sd.max().clamp_min(1e-6) * 1e-3 + 1e-6
        dec_kw = {"in_mu": mu, "in_scale": 1.0 / sd.clamp_min(floor)}
    if dec_kw:
        decoder = _LocalDecoder(z0.shape[1], tgt0.shape[1], tuple(tgt0.shape[-2:]), **dec_kw).to(device)
    else:
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
    if lr_scale_fanin:                   # hotfix ae_diverge: pesos do DECODER com fan-in grande -> lr * sqrt(ref/fan_in)
        ref = 9.0 * 16.0                 # (só o decoder descartável; o lr do bloco novo/herdado não muda)
        split = []
        for g in groups:
            if g.get("block"):
                split.append(g)
                continue
            keep, big = [], {}
            for p in g["params"]:
                fan = p[0].numel() if p.dim() > 1 else 0
                f = min(1.0, math.sqrt(ref / fan)) if fan > ref else 1.0
                (keep.append(p) if f == 1.0 else big.setdefault(f, []).append(p))
            if keep:
                split.append({**g, "params": keep})
            for f, ps in big.items():
                split.append({**g, "params": ps, "lr": g["lr"] * f})
        groups = split
    base_lr = [g["lr"] for g in groups]
    n_warm = int(round(float(decoder_warmup_frac) * epochs))
    opt = torch.optim.Adam(groups, lr=lr)
    clip_params = [p for g in groups for p in g["params"]]
    warm_n = max(0, int(warmup_steps or 0))
    diverged, diverge_reason = False, None
    guard_means: list = []
    clip_hits = 0
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
            # growth_fix8: todas as normas numa sincronização (antes: uma por parâmetro); mesmos valores/ordem
            with_grad = [(name, p) for name, p in named if p.grad is not None]
            norms = torch.stack([p.grad.detach().norm() for _, p in with_grad]).tolist() if (bad is None and with_grad) else []
            for (name, p), g in zip(with_grad, norms):
                if bad is None:
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
            if grad_clip is not None and float(grad_clip) > 0:
                tn = torch.nn.utils.clip_grad_norm_(clip_params, float(grad_clip))
                log["ae/grad_total_norm"] = float(tn)
                clip_hits += int(float(tn) > float(grad_clip))
            if warm_n and step < warm_n:
                for g, b in zip(opt.param_groups, base_lr):
                    if g["lr"] != 0.0:
                        g["lr"] = b * float(step + 1) / float(warm_n + 1)
            opt.step()
            if warm_n and step < warm_n:
                for g, b in zip(opt.param_groups, base_lr):
                    if g["lr"] != 0.0:
                        g["lr"] = b
            if initial_loss is None:
                initial_loss = log["ae/recon_loss"]
            losses.append(log["ae/recon_loss"])
            if logger is not None:
                logger.log(log, step=step)
            step += 1
        ep_time = time.perf_counter() - t_ep
        epoch_losses.append(losses)
        if diverge_ratio is not None and not diverged:     # hotfix ae_diverge: guarda por época (razão + tendência)
            # nunca o valor absoluto: "começa alto e cai" (escala) passa; "sobe" (razão > guard E subindo) diverge.
            prev = guard_means[-1] if guard_means else None
            dv, why, ep_mean = ae_guard_verdict(losses, prev, initial_loss, float(diverge_ratio), epoch == epochs - 1)
            guard_means.append(ep_mean)
            if dv:
                diverged, diverge_reason = True, f"época {epoch}: {why}"
        if progress is not None:             # growth_fix8: o log não fica mudo durante o AE local
            n_ep = len(losses)
            print(f"{progress} época {epoch + 1}/{epochs} passos={n_ep} "
                  f"perda={sum(losses) / max(1, n_ep):.5g} t={ep_time:.1f}s passos/s={n_ep / max(ep_time, 1e-9):.3f} "
                  f"pulados={len(skipped)}", flush=True)
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
        **({"diverged": bool(diverged), "diverge_reason": diverge_reason,
            "epoch_mean_losses": [float(v) for v in guard_means]} if diverge_ratio is not None else {}),
        **({"grad_clip": float(grad_clip), "grad_clip_hits": int(clip_hits)}
           if grad_clip is not None and float(grad_clip) > 0 else {}),
        **({"dec_in_norm": True} if dec_in_norm else {}),
        **({"lr_scale_fanin": True} if lr_scale_fanin else {}),
        **({"warmup_steps": warm_n} if warm_n else {}),
    }
