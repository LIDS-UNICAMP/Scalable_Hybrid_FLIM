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
Augmentation for LAB[0, 1] tensors, applied only where there is backpropagation.

Two families, both safe for LAB because they never mix the L, a, b channels:

  geometric   flips H/V, rot90, free rotation + translation + scale (one affine), light
              elastic deformation — they only move pixels; interpolation is per channel.
  L only      brightness/contrast ``L' = c*L + b`` and gamma ``L' = L**g`` on channel 0;
              a and b (the colour) are never touched. ColorJitter/Grayscale/Solarize are
              absent on purpose: they assume RGB and scramble a/b.

Borders opened by rotation/translation/elastic are filled with each image's own
background (per-channel median of its border pixels), not with 0 — 0 is L=0, a black
that never occurs on a slide. Done by transforming ``x - bg`` with fill 0 and adding
``bg`` back, which is exact for the bilinear interpolation used here.

Every image of the batch draws its own parameters (``torch`` RNG, so ``seed_everything``
governs it). The caller feeds the result as both input and reconstruction target.
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
from torch import Tensor


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high))


class LabAugment:
    """``__call__(x: [B, 3, H, W]) -> [B, 3, H, W]``, configured by the YAML ``augment`` block."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        el = cfg.get("elastic") or {}
        self.elastic = (v2.ElasticTransform(alpha=el["alpha"], sigma=el["sigma"], fill=0.0)
                        if el.get("p", 0) > 0 else None)

    def __call__(self, x: Tensor) -> Tensor:
        return torch.stack([self._one(img) for img in x])

    @staticmethod
    def _background(img: Tensor) -> Tensor:
        border = torch.cat([img[:, 0], img[:, -1], img[:, :, 0], img[:, :, -1]], dim=1)
        return border.median(dim=1).values[:, None, None]

    def _one(self, img: Tensor) -> Tensor:
        c = self.cfg
        if torch.rand(1) < c.get("hflip", 0):
            img = img.flip(-1)
        if torch.rand(1) < c.get("vflip", 0):
            img = img.flip(-2)
        if c.get("rot90"):
            img = torch.rot90(img, int(torch.randint(4, (1,))), dims=(-2, -1))

        bg = self._background(img)
        z = img - bg
        deg, tr, (s0, s1) = c.get("rotation_deg", 0), c.get("translate", 0), c.get("scale", (1, 1))
        if deg or tr or (s0, s1) != (1, 1):
            h, w = img.shape[-2:]
            z = TF.affine(z, angle=_uniform(-deg, deg),
                          translate=[_uniform(-tr, tr) * w, _uniform(-tr, tr) * h],
                          scale=_uniform(s0, s1), shear=[0.0, 0.0],
                          interpolation=v2.InterpolationMode.BILINEAR, fill=0.0)
        if self.elastic is not None and torch.rand(1) < c["elastic"]["p"]:
            z = self.elastic(z)
        img = z + bg

        L = img[0]
        if c.get("l_contrast") or c.get("l_brightness"):
            lo, hi = c.get("l_contrast", (1, 1))
            b = c.get("l_brightness", 0)
            L = _uniform(lo, hi) * L + _uniform(-b, b)
        if c.get("l_gamma"):
            L = L.clamp(0, 1) ** _uniform(*c["l_gamma"])
        return torch.cat([L[None], img[1:]]).clamp(0.0, 1.0)
