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

"""Q/K builders beyond ``shared_patch`` (USER_SPEC_V4 2.2 to 2.5).

Each builder is a plain class registered with ``@register_qk``: ``fit`` reads a
``QKFitContext``, ``__call__`` returns L2-normalised ``Q, K`` of shape
``(B, N, d)``. Fitted tensors live in ``self.state`` (``state_dict`` /
``load_state_dict``). Batched, device-agnostic, no grad.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from spifil.metrics import Euclidean
from spifil.scoring import FisherScorer, per_class_ranks
from spifil.selection import DiversitySelector
from spifil.types import PatchSet
from torch import Tensor

from methods.transformer.qk import QKContext, QKFitContext, register_qk

__all__ = ["ContentContext", "CrossScale", "PrototypeProfile"]

# ponytail: Fisher builds an M x M distance matrix; marked tokens are subsampled to this cap.
_MAX_SELECT = 4096


def _flat(z: Tensor) -> Tensor:
    """``(B, D, Gh, Gw)`` or ``(B, D, 1, Nmax)`` -> ``(B, N, D)``."""
    return z.flatten(2).transpose(1, 2)


def _need_grid(ctx: QKContext, name: str) -> None:
    if not ctx.grid:
        raise ValueError(
            f"{name} precisa do tokenizador grid (com superpixels nao ha janela de vizinhos)"
        )


def _marked(ctx: QKFitContext) -> tuple[Tensor, Tensor]:
    """Mask ``(B, N)`` of tokens inside the training masks (all valid if none) and their class."""
    m = ctx.valid
    if ctx.token_labels is not None and (m & (ctx.token_labels > 0)).any():
        m = m & (ctx.token_labels > 0)
    b = torch.arange(m.shape[0], device=m.device)[:, None].expand_as(m)
    if ctx.image_labels is not None:
        lab = ctx.image_labels.to(m.device)[b]
    elif ctx.token_labels is not None:
        lab = ctx.token_labels
    else:
        lab = torch.zeros_like(b)
    return m, lab[m]


def select_tokens(feats: Tensor, labels: Tensor, p: int, method: str, seed: int) -> Tensor:
    """Indices into ``feats (M, D)`` of up to ``p`` prototype tokens.

    ``kmeans``: the token nearest each k-means centroid. ``fisher``: SPiFiL's
    Fisher score between ``labels`` classes, then its greedy diversity selection.
    """
    g = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(feats), generator=g)[:_MAX_SELECT].sort().values.to(feats.device)
    x, y = feats[sub], labels[sub].long()
    p = min(p, len(x))
    if method == "kmeans":
        km = KMeans(n_clusters=p, n_init=4, random_state=seed).fit(x.double().cpu().numpy())
        cent = torch.as_tensor(km.cluster_centers_, dtype=x.dtype, device=x.device)
        return sub[torch.cdist(cent, x).argmin(1)]
    if method == "fisher":
        n = torch.arange(len(x), device=x.device)
        ps = PatchSet(feats=x, labels=y, seed_rows=n, image_ids=torch.zeros_like(n))
        ranks = per_class_ranks(FisherScorer()(ps, Euclidean()), y)
        cls = torch.unique(y).tolist()
        budget = {c: p // len(cls) + (i < p % len(cls)) for i, c in enumerate(cls)}
        return sub[DiversitySelector()(ps, ranks, budget).nonzero(as_tuple=True)[0]]
    raise ValueError(f"prototype_method desconhecido: {method} (use kmeans ou fisher)")


def _profile(x: Tensor, anchors: Tensor) -> Tensor:
    """Cosine of each token ``(B, N, C)`` with ``anchors (P, C)``, L2-normalised: ``(B, N, P)``."""
    a = F.normalize(anchors.to(x.device), dim=-1)
    return F.normalize(F.normalize(x, dim=-1) @ a.T, dim=-1)


class _Builder:
    state: dict[str, Tensor]

    def fit(self, ctx_train: QKFitContext) -> None:
        self.state = {}

    def state_dict(self) -> dict[str, Tensor]:
        return dict(getattr(self, "state", {}))

    def load_state_dict(self, d: dict[str, Tensor]) -> None:
        self.state = dict(d)


@register_qk("content_context")
class ContentContext(_Builder):
    """Q = z_i, K = mean of Z over the s x s window around j. Asymmetric, d = D."""

    def __init__(self, s: int = 3) -> None:
        self.s = int(s)

    @torch.no_grad()
    def __call__(self, ctx: QKContext) -> tuple[Tensor, Tensor]:
        _need_grid(ctx, self.name)
        z = ctx.Z
        k = F.avg_pool2d(z, self.s, 1, self.s // 2, count_include_pad=False)
        k = k[..., : z.shape[-2], : z.shape[-1]]  # even s pads one extra row/col
        return F.normalize(_flat(z), dim=-1), F.normalize(_flat(k), dim=-1)


@register_qk("cross_scale")
class CrossScale(_Builder):
    """Q = z_i, K = the token of j's parent block at half resolution. Asymmetric, d = D."""

    @torch.no_grad()
    def __call__(self, ctx: QKContext) -> tuple[Tensor, Tensor]:
        _need_grid(ctx, self.name)
        z = ctx.Z
        gh, gw = z.shape[-2:]
        k = F.adaptive_avg_pool2d(z, (math.ceil(gh / 2), math.ceil(gw / 2)))
        k = F.interpolate(k, size=(gh, gw), mode="nearest")
        return F.normalize(_flat(z), dim=-1), F.normalize(_flat(k), dim=-1)


@register_qk("prototype_profile")
class PrototypeProfile(_Builder):
    """Q = K = cosine profile against P prototype tokens from the training masks. d = P."""

    def __init__(self, n_prototypes: int = 32, prototype_method: str = "kmeans", seed: int = 42) -> None:
        self.p, self.method, self.seed = int(n_prototypes), prototype_method, int(seed)

    @torch.no_grad()
    def fit(self, ctx_train: QKFitContext) -> None:
        m, lab = _marked(ctx_train)
        t = _flat(ctx_train.Z)[m]
        self.state = {"prototypes": t[select_tokens(t, lab, self.p, self.method, self.seed)].clone()}

    @torch.no_grad()
    def __call__(self, ctx: QKContext) -> tuple[Tensor, Tensor]:
        if "prototypes" not in self.state_dict():
            raise RuntimeError("prototype_profile.fit precisa ser chamado antes")
        q = _profile(_flat(ctx.Z), self.state["prototypes"])
        return q, q

