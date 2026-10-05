# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
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

"""Cross-attention heads (spec 4.2): each head turns the normalised tokens into prototype logits.

Heads are plain classes registered with ``@register_head(name)`` and chosen by name in the config,
so a new head never requires editing the model. A head returns one or more sub-heads
``(sub_name, logits (B, N, K_h), kernel_index (K_h,))``; kernel_index points into the K prototypes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

__all__ = [
    "HeadContext",
    "CrossHead",
    "HEAD_REGISTRY",
    "register_head",
    "build_heads",
]


@dataclass
class HeadContext:
    z: Tensor                     # (B, N, 2K) normalised updated tokens: [:K] mean part, [K:] std part
    adj: Tensor                   # (B, N, N) bool
    valid: Tensor                 # (B, N) bool
    kernel_class: Tensor          # (K,) int64
    kernel_source: Tensor | None  # (K,) int64

    @property
    def K(self) -> int:
        return self.kernel_class.numel()

    def part(self, which: int) -> Tensor:
        """Mean (0) or std (1) half of z, zeroed on padding so no NaN can leak out."""
        x = self.z[..., which * self.K:(which + 1) * self.K]
        return x.masked_fill(~self.valid[..., None], 0.0)

    def all_kernels(self) -> Tensor:
        return torch.arange(self.K, device=self.z.device)


class CrossHead(Protocol):
    name: str

    def __call__(self, ctx: HeadContext) -> list[tuple[str, Tensor, Tensor]]: ...


HEAD_REGISTRY: dict[str, type] = {}


def register_head(name: str):
    def deco(cls: type) -> type:
        cls.name = name
        HEAD_REGISTRY[name] = cls
        return cls

    return deco


def build_heads(names: Sequence[str]) -> list[CrossHead]:
    unknown = [n for n in names if n not in HEAD_REGISTRY]
    if unknown:
        raise ValueError(f"cabeca(s) desconhecida(s) {unknown}; disponiveis: {sorted(HEAD_REGISTRY)}")
    return [HEAD_REGISTRY[n]() for n in names]


@register_head("appearance")
class AppearanceHead:
    def __call__(self, ctx: HeadContext):
        return [(self.name, ctx.part(0), ctx.all_kernels())]


@register_head("texture")
class TextureHead:
    def __call__(self, ctx: HeadContext):
        return [(self.name, ctx.part(1), ctx.all_kernels())]


@register_head("context")
class ContextHead:
    def __call__(self, ctx: HeadContext):
        a = (ctx.adj & ctx.valid[:, None, :]).to(ctx.z.dtype)
        deg = a.sum(-1, keepdim=True).clamp(min=1.0)  # no neighbour -> sum 0 / 1 = 0
        return [(self.name, (a @ ctx.part(0)) / deg, ctx.all_kernels())]


@register_head("per_image")
class PerImageHead:
    def __call__(self, ctx: HeadContext):
        if ctx.kernel_source is None:
            raise ValueError(
                "per_image precisa de kernel_source (imagem de origem de cada kernel), que nao foi fornecido"
            )
        zm = ctx.part(0)
        src = ctx.kernel_source.to(zm.device)
        out = []
        for r in torch.unique(src).tolist():
            idx = torch.nonzero(src == r).flatten()
            out.append((f"{self.name}:{r}", zm[..., idx], idx))
        return out
