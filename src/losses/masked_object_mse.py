# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""
Masked reconstruction restricted to the object (Especificação 1 §4).

Pipeline (CNN, no Transformer):
    obj    = object_patches(mask)                 # [B,gh,gw] patch is object if >= 25% of its pixels are mask
    masked = sample_block_mask(obj, ratio=0.20)   # only object patches, in 2x2-patch blocks, floor of 8
    vis    = 1 - patch_to_pixel(masked, H, W)     # pixel mask fed to the encoder (sparse-conv emulation)
    loss   = masked_patch_mse(pred, x, masked)    # MSE on masked patches only, target normalised per patch

Floor patch grid: gh = H // patch, gw = W // patch (T=200, patch=16 -> 12x12 over pixels [0,192)).
The border strip beyond gh*patch / gw*patch belongs to no patch: never masked, never in the loss.
Background patches are ignored completely (not masked, out of the loss, no weight).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def object_patches(mask: Tensor, patch: int = 16, threshold: float = 0.25) -> Tensor:
    """mask [B,1,H,W] in {0,1} -> [B,gh,gw] bool, True where >= threshold of the patch's pixels are object."""
    gh, gw = mask.shape[-2] // patch, mask.shape[-1] // patch
    frac = F.avg_pool2d(mask[..., : gh * patch, : gw * patch].float(), patch)
    return frac[:, 0] >= threshold


def sample_block_mask(
    obj: Tensor,
    ratio: float,
    block: int = 2,
    min_masked: int = 8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Choose which object patches to mask, in blocks of block x block patches.

    Per image: n_mask = min(n_obj, max(min_masked, round(ratio * n_obj))).
    Blocks (on the gh/block x gw/block grid) that contain object patches are visited in a
    random order and their object patches are masked until n_mask is reached; the last block
    may be partially masked so the count is exact. Background patches are never masked.
    Images without object patches get an all-False mask. Fully vectorised over the batch.
    """
    B, gh, gw = obj.shape
    dev = obj.device
    nbh, nbw = -(-gh // block), -(-gw // block)          # ceil: partial blocks at the edge if gh % block

    n_obj = obj.flatten(1).sum(1)
    n_mask = torch.floor(ratio * n_obj.float() + 0.5).long().clamp(min=min_masked)
    n_mask = torch.minimum(n_mask, n_obj)

    # random rank of every block, broadcast to its patches
    block_rank = torch.rand(B, nbh * nbw, generator=generator, device=dev).argsort(1).argsort(1)
    block_rank = block_rank.view(B, nbh, 1, nbw, 1).expand(B, nbh, block, nbw, block)
    block_rank = block_rank.reshape(B, nbh * block, nbw * block)[:, :gh, :gw]

    # key = block order + random order inside the block; background patches never selected
    intra = torch.rand(B, gh, gw, generator=generator, device=dev)
    key = torch.where(obj, block_rank.float() + intra, torch.full_like(intra, float("inf")))
    rank = key.flatten(1).argsort(1).argsort(1)
    masked = (rank < n_mask[:, None]) & obj.flatten(1)
    return masked.view(B, gh, gw)


def patch_to_pixel(masked: Tensor, H: int, W: int, patch: int = 16) -> Tensor:
    """[B,gh,gw] bool -> [B,1,H,W] float, 1 = masked pixel; the border strip outside the grid is 0."""
    gh, gw = masked.shape[-2:]
    pix = masked.float().repeat_interleave(patch, 1).repeat_interleave(patch, 2)
    return F.pad(pix, (0, W - gw * patch, 0, H - gh * patch)).unsqueeze(1)


def _to_patches(x: Tensor, gh: int, gw: int, patch: int) -> Tensor:
    """[B,C,H,W] -> [B,gh,gw,C*patch*patch] over the floor grid (border strip dropped)."""
    B, C = x.shape[:2]
    x = x[..., : gh * patch, : gw * patch].reshape(B, C, gh, patch, gw, patch)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, gh, gw, C * patch * patch)


def masked_patch_mse(
    pred: Tensor,
    target: Tensor,
    masked: Tensor,
    patch: int = 16,
    eps: float = 1e-6,
) -> Tensor:
    """
    MSE only over masked patches; target normalised per patch (mean 0, std 1 over its C*p*p values).

    Averaged over all masked patches of the batch. If nothing is masked, returns a 0 that is
    still attached to the graph of `pred` (no NaN).
    """
    gh, gw = masked.shape[-2:]
    p = _to_patches(pred, gh, gw, patch)
    t = _to_patches(target, gh, gw, patch).detach()
    t = (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=False, keepdim=True) + eps).sqrt()
    per_patch = (p - t).pow(2).mean(-1)                  # [B,gh,gw]
    m = masked.to(per_patch.dtype)
    return (per_patch * m).sum() / m.sum().clamp(min=1.0)
