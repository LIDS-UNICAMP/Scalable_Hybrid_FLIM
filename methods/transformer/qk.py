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

"""Pluggable Q/K builders: contexts, protocol, registry and ``shared_patch``.

A builder turns the normalised tokens of a stage into Q and K ``(B, N, d)``,
L2-normalised per row. Add one by writing a class with ``@register_qk``; the
attention block never changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "QKContext",
    "QKFitContext",
    "QKBuilder",
    "QK_REGISTRY",
    "register_qk",
    "build_qk",
    "check_qk",
    "SharedPatch",
]


@dataclass
class QKContext:
    Z: Tensor  # (B, D, Gh, Gw) grid; superpixel: (B, D, 1, Nmax) padded
    valid: Tensor  # (B, N) bool, False on padding
    grid: bool  # True for the grid tokenizer
    stage: int = 0
    stages_Z: dict[int, Tensor] = field(default_factory=dict)
    # SPiFiL layer l tokenised on the SAME regions, token means, same layout as Z, not normalised.
    layer_feats: dict[int, Tensor] = field(default_factory=dict)
    adjacency: Tensor | None = None  # (B, N, N) bool, diagonal False


@dataclass
class QKFitContext(QKContext):
    token_labels: Tensor | None = None  # (B, N) int64, 0 = unmarked / outside mask
    image_labels: Tensor | None = None  # (B,) int64 image classes


class QKBuilder(Protocol):
    name: str

    def fit(self, ctx_train: QKFitContext) -> None: ...

    def __call__(self, ctx: QKContext) -> tuple[Tensor, Tensor]: ...


QK_REGISTRY: dict[str, type] = {}


def register_qk(name: str):
    """Class decorator: register ``cls`` under ``name`` and set ``cls.name``."""

    def deco(cls: type) -> type:
        cls.name = name
        QK_REGISTRY[name] = cls
        return cls

    return deco


def build_qk(name: str, **params) -> QKBuilder:
    if name not in QK_REGISTRY:
        raise ValueError(f"qk_builder desconhecido: {name!r}; registrados: {sorted(QK_REGISTRY)}")
    return QK_REGISTRY[name](**params)


def check_qk(q: Tensor, k: Tensor) -> None:
    if q.ndim != 3 or q.shape != k.shape:
        raise ValueError(f"Q e K precisam ter o mesmo shape (B, N, d): Q {tuple(q.shape)}, K {tuple(k.shape)}")


@register_qk("shared_patch")
class SharedPatch:
    """Q = K = r x r patch of Z around each token (reflect padding), d = r*r*D. Symmetric S."""

    def __init__(self, r: int = 3) -> None:
        self.r = r

    def fit(self, ctx_train: QKFitContext) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass

    @torch.no_grad()
    def __call__(self, ctx: QKContext) -> tuple[Tensor, Tensor]:
        if not ctx.grid:
            raise ValueError(f"{self.name} precisa do tokenizador grid (recebeu superpixel)")
        z, r = ctx.Z, self.r
        if r > 1:
            mode = "reflect" if min(z.shape[-2:]) > r // 2 else "replicate"
            z = F.pad(z, (r // 2,) * 4, mode=mode)
        q = F.normalize(F.unfold(z, r).transpose(1, 2), dim=-1)  # (B, N, D r r)
        return q, q


from methods.transformer import qk_builders  # noqa: F401, E402  registers the other builders
