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

"""Pluggable tokenizers: a ``(C, h, w)`` map -> tokens, adjacency and layout.

Superpixels and P x P windows share one reduction: a window grid is just a
region map. All reductions are scatter ops over pixels, no Python loops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "Tokens",
    "Tokenizer",
    "TOKENIZER_REGISTRY",
    "register_tokenizer",
    "grid_regions",
    "tokenize",
    "token_labels",
    "TokenBatch",
    "tokenize_batch",
]


@dataclass
class Tokens:
    t: Tensor  # (N, D) token stats
    pos: Tensor  # (N, 2) medoid position in [0, 1]
    pix: Tensor  # (h*w,) token index per pixel, -1 = ignored
    region: Tensor  # (h, w) region map used, to tokenise other maps on the same regions
    adjacency: Tensor  # (N, N) bool, diagonal False
    shape: tuple[int, int]  # (Gh, Gw) grid layout, (1, N) for superpixels


@dataclass
class TokenBatch:
    t: Tensor  # (B, N, D) raw tokens [mean ; std], D = 2K
    valid: Tensor  # (B, N) bool
    adj: Tensor  # (B, N, N) bool, symmetric, diagonal False, False on padding
    area: Tensor  # (B, N) float pixel counts, 0 on padding
    pos: Tensor  # (B, N, 2) centroid in [0, 1]
    labels: Tensor  # (B, H', W') int64 token id per feature pixel, contiguous 0..N_b-1
    interest: Tensor  # (B, N) bool


class Tokenizer(Protocol):
    name: str
    grid: bool

    def __call__(self, x: Tensor, region: Tensor | None, cfg) -> Tokens: ...

    def batch(self, F: Tensor, segs, masks, cfg) -> TokenBatch: ...


TOKENIZER_REGISTRY: dict[str, type] = {}


def register_tokenizer(name: str):
    def deco(cls: type) -> type:
        cls.name = name
        TOKENIZER_REGISTRY[name] = cls
        return cls

    return deco


def grid_regions(h: int, w: int, window: int, device=None) -> Tensor:
    """Region map of non-overlapping ``window`` x ``window`` cells, row-major."""
    if h % window or w % window:
        raise ValueError(f"mapa {h}x{w} nao e divisivel pela janela {window}")
    r = torch.arange(h, device=device).unsqueeze(1) // window
    c = torch.arange(w, device=device).unsqueeze(0) // window
    return r * (w // window) + c


def tokenize(x: Tensor, region: Tensor, stats: str) -> tuple[Tensor, Tensor, Tensor]:
    """Return tokens ``(N, D)``, medoids ``(N, 2)`` in [0, 1] and the per-pixel token index.

    Pixels with region id < 0 are ignored (index -1). Region ids are remapped
    to contiguous token indices in increasing id order. Medoid = the region
    pixel closest to the region centroid.
    """
    c, h, w = x.shape
    dev = x.device
    flat = x.reshape(c, -1).T.float()  # (h*w, C)
    reg = region.reshape(-1).long().to(dev)
    valid = reg >= 0
    _, inv = torch.unique(reg[valid], return_inverse=True)
    n = int(inv.max()) + 1 if inv.numel() else 0
    xv = flat[valid]
    cnt = torch.bincount(inv, minlength=n).float().unsqueeze(1)
    mean = torch.zeros(n, c, device=dev).index_add_(0, inv, xv) / cnt
    if stats == "mean":
        t = mean
    elif stats == "mean_std":
        var = torch.zeros(n, c, device=dev).index_add_(0, inv, (xv - mean[inv]) ** 2) / cnt
        t = torch.cat([mean, var.sqrt()], 1)
    elif stats == "mean_max":
        mx = torch.full((n, c), -torch.inf, device=dev).scatter_reduce_(
            0, inv.unsqueeze(1).expand_as(xv), xv, "amax"
        )
        t = torch.cat([mean, mx], 1)
    else:
        raise ValueError(f"token_stats desconhecido: {stats}")
    rows, cols = torch.meshgrid(
        (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w, indexing="ij"
    )
    rc = torch.stack([rows.reshape(-1), cols.reshape(-1)], 1)[valid]
    centroid = torch.zeros(n, 2, device=dev).index_add_(0, inv, rc) / cnt
    d = ((rc - centroid[inv]) ** 2).sum(1)
    dmin = torch.full((n,), torch.inf, device=dev).scatter_reduce_(0, inv, d, "amin")
    near = d <= dmin[inv]
    first = torch.full((n,), len(rc), device=dev).scatter_reduce_(
        0, inv[near], torch.arange(len(rc), device=dev)[near], "amin"
    )
    pix = torch.full_like(reg, -1)
    pix[valid] = inv
    return t, rc[first], pix


def token_labels(pix: Tensor, marker: Tensor, n: int, min_marked_frac: float) -> Tensor:
    """Majority marker class among marked pixels of each token, 0 if under-marked."""
    m = marker.reshape(-1).long().to(pix.device)
    valid = pix >= 0
    inv, m = pix[valid], m[valid]
    n_cls = int(m.max()) + 1 if m.numel() else 1
    hist = torch.bincount(inv * n_cls + m, minlength=n * n_cls).reshape(n, n_cls)
    marked = hist[:, 1:].sum(1)
    if n_cls == 1:
        return torch.zeros(n, dtype=torch.long, device=pix.device)
    lab = hist[:, 1:].argmax(1) + 1
    ok = (marked > 0) & (marked >= min_marked_frac * hist.sum(1))
    return torch.where(ok, lab, torch.zeros_like(lab))


@register_tokenizer("grid")
class GridTokenizer:
    """P x P windows; adjacency = Chebyshev distance <= neighborhood // 2 on the grid."""

    grid = True

    def __call__(self, x: Tensor, region: Tensor | None, cfg) -> Tokens:
        h, w = x.shape[-2:]
        win = cfg.window
        region = grid_regions(h, w, win, x.device)
        t, pos, pix = tokenize(x, region, cfg.token_stats)
        gh, gw = h // win, w // win
        idx = torch.arange(gh * gw, device=x.device)
        rr, cc = idx // gw, idx % gw
        cheb = torch.maximum((rr[:, None] - rr).abs(), (cc[:, None] - cc).abs())
        adj = cheb <= cfg.neighborhood // 2
        adj.fill_diagonal_(False)
        return Tokens(t, pos, pix, region, adj, (gh, gw))

    def batch(self, F: Tensor, segs, masks, cfg) -> TokenBatch:
        # ponytail: edge windows may be partial, so any (H', W') works (the old __call__ demands divisibility).
        b, _, h, w = F.shape
        p, dev = cfg.window, F.device
        gw = -(-w // p)
        idx = torch.arange(-(-h // p) * gw, device=dev)
        cheb = torch.maximum((idx[:, None] // gw - idx // gw).abs(), (idx[:, None] % gw - idx % gw).abs())
        adj = (cheb <= cfg.neighborhood // 2).fill_diagonal_(False)
        lab = torch.arange(h, device=dev)[:, None] // p * gw + torch.arange(w, device=dev) // p
        return _token_batch(F, lab.expand(b, h, w), masks, cfg, adj.expand(b, -1, -1).clone())


@register_tokenizer("superpixel")
class SuperpixelTokenizer:
    """Given regions; adjacency = regions sharing a pixel edge."""

    grid = False

    def __call__(self, x: Tensor, region: Tensor | None, cfg) -> Tokens:
        if region is None:
            raise ValueError("tokenizador superpixel precisa do mapa de regioes")
        region = region.to(x.device)
        t, pos, pix = tokenize(x, region, cfg.token_stats)
        n = len(t)
        p = pix.reshape(region.shape)
        a = torch.cat([p[:, :-1].reshape(-1), p[:-1, :].reshape(-1)])
        b = torch.cat([p[:, 1:].reshape(-1), p[1:, :].reshape(-1)])
        ok = (a >= 0) & (b >= 0) & (a != b)
        adj = torch.zeros(n, n, dtype=torch.bool, device=x.device)
        adj[a[ok], b[ok]] = True
        adj[b[ok], a[ok]] = True
        return Tokens(t, pos, pix, region, adj, (1, n))

    def batch(self, F: Tensor, segs, masks, cfg) -> TokenBatch:
        b, _, h, w = F.shape
        if segs is None or len(segs) != b:
            raise ValueError("tokenizador superpixel precisa de um mapa de superpixels por imagem")
        n = 0
        labs = []
        for seg in segs:  # vanished labels drop out of unique, the inverse is contiguous 0..N_b-1
            lab = torch.unique(_nearest(seg, h, w, F.device), return_inverse=True)[1]
            labs.append(lab)
            n = max(n, int(lab.max()) + 1)
        lab = torch.stack(labs)
        adj = torch.zeros(b, n, n, dtype=torch.bool, device=F.device)
        u = torch.cat([lab[:, :, :-1].reshape(b, -1), lab[:, :-1, :].reshape(b, -1)], 1)
        v = torch.cat([lab[:, :, 1:].reshape(b, -1), lab[:, 1:, :].reshape(b, -1)], 1)
        bi = torch.arange(b, device=F.device)[:, None].expand_as(u)
        ok = u != v
        adj[bi[ok], u[ok], v[ok]] = True
        adj[bi[ok], v[ok], u[ok]] = True
        return _token_batch(F, lab, masks, cfg, adj)


def _nearest(a, h: int, w: int, device) -> Tensor:
    """Nearest-neighbour resize of a full-resolution int/bool map to (h, w), as torch 'nearest' does."""
    a = a.long().to(device) if isinstance(a, Tensor) else torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    hh, ww = a.shape[-2:]
    return a[torch.arange(h, device=device) * hh // h][:, torch.arange(w, device=device) * ww // w]


def _token_batch(F: Tensor, lab: Tensor, masks, cfg, adj: Tensor) -> TokenBatch:
    """Scatter-reduce F (B, K, h, w) over the contiguous token labels (B, h, w) of every image at once."""
    b, k, h, w = F.shape
    dev = F.device
    n = adj.shape[-1]
    gid = (lab + torch.arange(b, device=dev)[:, None, None] * n).reshape(-1)
    x = F.permute(0, 2, 3, 1).reshape(-1, k).float()
    cnt = torch.bincount(gid, minlength=b * n).float()
    c1 = cnt.clamp_min(1)[:, None]
    mean = torch.zeros(b * n, k, device=dev).index_add_(0, gid, x) / c1
    var = torch.zeros(b * n, k, device=dev).index_add_(0, gid, (x - mean[gid]) ** 2) / c1
    rows, cols = torch.meshgrid(
        (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w, indexing="ij"
    )
    rc = torch.stack([rows.reshape(-1), cols.reshape(-1)], 1).repeat(b, 1)
    pos = torch.zeros(b * n, 2, device=dev).index_add_(0, gid, rc) / c1
    area = cnt.reshape(b, n)
    valid = area > 0
    adj &= valid[:, :, None] & valid[:, None, :]
    m = torch.stack([
        torch.ones(h, w, device=dev) if mk is None else _nearest(mk, h, w, dev).gt(0).float()
        for mk in (masks if masks is not None else [None] * b)
    ])
    frac = torch.zeros(b * n, device=dev).index_add_(0, gid, m.reshape(-1)).reshape(b, n) / area.clamp_min(1)
    return TokenBatch(
        t=torch.cat([mean, var.sqrt()], 1).reshape(b, n, 2 * k),
        valid=valid,
        adj=adj,
        area=area,
        pos=pos.reshape(b, n, 2),
        labels=lab.long(),
        interest=valid & (frac >= cfg.min_mask_frac),
    )


def tokenize_batch(
    F: Tensor, segs: Sequence[np.ndarray | Tensor] | None, masks: Sequence[np.ndarray | None] | None, cfg
) -> TokenBatch:
    """Padded token batch of ``F`` (B, K, H', W'); no segs means P x P grid windows."""
    name = "grid" if segs is None else cfg.tokenizer
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"tokenizer desconhecido: {name!r}; registrados: {sorted(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name]().batch(F, segs, masks, cfg)
