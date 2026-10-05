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

"""Single-stage FLIM transformer, every parameter estimated in closed form.

Block: tokens T (pluggable tokenizer) -> marker normalisation Z -> Q, K from a
pluggable builder (methods.transformer.qk) -> S = Q K^T -> A: softmax(S / tau)
over the neighbours N(i) when the best neighbour reaches delta, else over the
top-k most similar tokens outside N(i) U {i} -> T + alpha A T -> readout g.
Images run in padded batches; no backprop, no grad, any device.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from methods.transformer.config import FlimTransformerConfig
from methods.transformer.qk import QKContext, QKFitContext, build_qk, check_qk
from methods.transformer.tokenizers import TOKENIZER_REGISTRY, Tokens, token_labels, tokenize

__all__ = ["FlimTransformer", "row_blocks", "best_neighbour", "graph_attention"]

_BATCH = 16  # ponytail: fixed images per batch; move to cfg if memory ever demands it

Item = tuple[Tokens, dict[int, Tensor]]


def row_blocks(q: Tensor, k: Tensor, chunk_threshold: int, chunk_size: int) -> Iterator[tuple[slice, Tensor]]:
    """Row blocks of S = Q K^T (B, rows, N); one block unless N > chunk_threshold."""
    n = q.shape[1]
    step = n if n <= chunk_threshold else chunk_size
    kt = k.transpose(1, 2)
    for i in range(0, n, step):
        sl = slice(i, i + step)
        yield sl, q[:, sl] @ kt


def best_neighbour(q: Tensor, k: Tensor, adj: Tensor, chunk_threshold: int, chunk_size: int) -> Tensor:
    """(B, N) best similarity to a neighbour, -inf for tokens without neighbours."""
    return torch.cat(
        [torch.where(adj[:, sl], s, -torch.inf).amax(-1) for sl, s in row_blocks(q, k, chunk_threshold, chunk_size)], 1
    )


def graph_attention(
    q: Tensor, k: Tensor, adj: Tensor, valid: Tensor, delta: float, topk: int, tau: float,
    chunk_threshold: int, chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Return ``(A (B, N, N), fallback (B, N))``: softmax(S / tau) over the neighbours when the best one
    reaches delta, else over the top-k most similar valid tokens outside N(i) U {i}. Masks, no token loops."""
    b, n = valid.shape
    a = q.new_zeros(b, n, n)
    fallback = torch.zeros_like(valid)
    eye = torch.eye(n, dtype=torch.bool, device=q.device)
    kk = min(topk, n)
    for sl, s in row_blocks(q, k, chunk_threshold, chunk_size):
        nb = adj[:, sl]
        cand = ~nb & ~eye[sl] & valid[:, None, :]
        fb = (torch.where(nb, s, -torch.inf).amax(-1) < delta) & valid[:, sl]
        top = torch.zeros_like(cand)
        if kk:
            top.scatter_(-1, torch.where(cand, s, -torch.inf).topk(kk, -1).indices, True)
        keep = torch.where(fb[..., None], top & cand, nb) & valid[:, sl, None]
        # ponytail: a row with no candidate is all -inf -> NaN -> zeros, it composes nothing.
        a[:, sl] = torch.nan_to_num(torch.softmax(torch.where(keep, s / tau, -torch.inf), -1), nan=0.0)
        fallback[:, sl] = fb
    return a, fallback


class FlimTransformer:
    """One neighbourhood composition block over FLIM tokens, fitted from markers."""

    def __init__(self, cfg: FlimTransformerConfig) -> None:
        if cfg.tokenizer not in TOKENIZER_REGISTRY:
            raise ValueError(f"tokenizer desconhecido: {cfg.tokenizer!r}; registrados: {sorted(TOKENIZER_REGISTRY)}")
        self.cfg = cfg
        self.tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer]()
        params = dict(cfg.qk_params)
        if cfg.qk_builder == "shared_patch":
            params.setdefault("r", cfg.patch_size)
        self.builder = build_qk(cfg.qk_builder, **params)
        self.mu: Tensor | None = None
        self.sigma: Tensor | None = None
        self.delta = float("inf")
        self.n_marked = 0
        self.d = 0
        self.warnings: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Tokens and contexts
    # ------------------------------------------------------------------

    def _item(self, x: Tensor, region: Tensor | None, feats: dict[int, Tensor] | None) -> Item:
        """Tokens of ``x`` and each SPiFiL layer map resized to (h, w), meaned on the same regions."""
        tok = self.tokenizer(x, region, self.cfg)
        lf = {}
        for layer, f in (feats or {}).items():
            f = F.interpolate(f[None].float().to(x.device), size=x.shape[-2:], mode="bilinear", align_corners=False)
            lf[layer] = tokenize(f[0], tok.region, "mean")[0]
        return tok, lf

    @staticmethod
    def _batches(items: Sequence[Item]) -> list[list[int]]:
        """Indices grouped by token layout (grid shape; any for superpixels), at most _BATCH each."""
        groups: dict[object, list[int]] = defaultdict(list)
        for i, (tok, _) in enumerate(items):
            groups[tok.shape if tok.shape[0] > 1 else None].append(i)
        return [g[j : j + _BATCH] for g in groups.values() for j in range(0, len(g), _BATCH)]

    def _context(
        self, items: Sequence[Item], labels: Sequence[Tensor] | None = None, image_labels=None
    ) -> tuple[QKContext, Tensor]:
        """Padded context of ``items`` and their raw tokens ``(B, N, D)``."""
        toks = [tok for tok, _ in items]
        b, n = len(toks), max(len(tok.t) for tok in toks)
        dev = toks[0].t.device
        grid = self.tokenizer.grid
        gh, gw = toks[0].shape if grid else (1, n)

        def pad(rows: Sequence[Tensor], fill=0) -> Tensor:
            out = rows[0].new_full((b, n, *rows[0].shape[1:]), fill)
            for i, r in enumerate(rows):
                out[i, : len(r)] = r
            return out

        def layout(v: Tensor) -> Tensor:  # (B, N, C) -> (B, C, Gh, Gw)
            return v.transpose(1, 2).reshape(b, -1, gh, gw)

        t = pad([tok.t for tok in toks])
        valid = pad([torch.ones(len(tok.t), dtype=torch.bool, device=dev) for tok in toks], False)
        adj = torch.zeros(b, n, n, dtype=torch.bool, device=dev)
        for i, tok in enumerate(toks):
            adj[i, : len(tok.t), : len(tok.t)] = tok.adjacency
        kw = dict(
            Z=layout(pad([(tok.t - self.mu) / self.sigma for tok in toks])),
            valid=valid,
            grid=grid,
            layer_feats={k: layout(pad([lf[k] for _, lf in items])) for k in items[0][1]},
            adjacency=adj,
        )
        if labels is None:
            return QKContext(**kw), t
        img = None if image_labels is None else torch.as_tensor(list(image_labels), dtype=torch.long, device=dev)
        return QKFitContext(**kw, token_labels=pad(list(labels)), image_labels=img), t

    def _qk(self, ctx: QKContext) -> tuple[Tensor, Tensor]:
        q, k = self.builder(ctx)
        check_qk(q, k)
        return q, k

    # ------------------------------------------------------------------
    # Fit and block
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(
        self,
        maps: Sequence[Tensor],
        regions: Sequence[Tensor | None],
        markers: Sequence[Tensor],
        layer_feats: Sequence[dict[int, Tensor] | None] | None = None,
        image_labels: Sequence[int] | None = None,
    ) -> "FlimTransformer":
        cfg = self.cfg
        feats = layer_feats if layer_feats is not None else [None] * len(maps)
        items = [self._item(x, r, f) for x, r, f in zip(maps, regions, feats)]
        labels = [
            token_labels(tok.pix, m, len(tok.t), cfg.min_marked_frac) for (tok, _), m in zip(items, markers)
        ]
        warns: list[str] = []
        t_m = torch.cat([tok.t[lab > 0] for (tok, _), lab in zip(items, labels)])
        self.n_marked = len(t_m)
        if self.n_marked < 2:
            # ponytail: 1 marked token gives sigma = 0; all tokens are the safer stats.
            warns.append(f"{self.n_marked} tokens marcados: mu e sigma usam todos os tokens")
            t_m = torch.cat([tok.t for tok, _ in items])
        self.mu = t_m.mean(0)
        self.sigma = t_m.std(0, unbiased=False) + cfg.eps
        # ponytail: one fit context with every train image; mixed grid shapes are not supported here.
        self.builder.fit(self._context(items, labels, image_labels)[0])
        best = []
        for idx in self._batches(items):
            ctx, _ = self._context([items[i] for i in idx])
            q, k = self._qk(ctx)
            self.d = int(q.shape[-1])
            bn = best_neighbour(q, k, ctx.adjacency, cfg.chunk_threshold, cfg.chunk_size)
            best.append(bn[ctx.valid])
        best = torch.cat(best)
        best = best[torch.isfinite(best)]
        if best.numel():
            self.delta = float(torch.quantile(best.float(), cfg.delta_percentile / 100.0))
        else:
            warns.append("nenhum token com vizinhos: todos usam o fallback nao local")
        self.warnings = tuple(warns)
        return self

    def _block(self, t: Tensor, ctx: QKContext, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(T + alpha A T, A, fallback)`` per batch; A built with masks, no token loops."""
        c = self.cfg
        a, fallback = graph_attention(
            q, k, ctx.adjacency, ctx.valid, self.delta, c.global_topk, c.tau, c.chunk_threshold, c.chunk_size
        )
        return t + self.cfg.residual_alpha * (a @ t), a, fallback

    def _readout(self, out: Tensor, a: Tensor, valid: Tensor) -> Tensor:
        if self.cfg.readout == "gap":
            w = valid.to(out.dtype)
        elif self.cfg.readout == "attention_weighted":
            w = a.sum(1)
        else:
            raise ValueError(f"readout desconhecido: {self.cfg.readout}")
        w = w / w.sum(1, keepdim=True).clamp_min(self.cfg.eps)
        return (w[:, None] @ out)[:, 0]

    def _run(self, items: Sequence[Item]) -> tuple[QKContext, Tensor, Tensor, Tensor]:
        if self.mu is None:
            raise RuntimeError("FlimTransformer.fit precisa ser chamado antes")
        ctx, t = self._context(items)
        return (ctx, *self._block(t, ctx, *self._qk(ctx)))

    @torch.no_grad()
    def block(
        self, x: Tensor, region: Tensor | None, layer_feats: dict[int, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(T' (N, D), A (N, N), fallback (N,), pix (h, w))``, pix -1 = ignored."""
        item = self._item(x, region, layer_feats)
        _, out, a, fb = self._run([item])
        return out[0], a[0], fb[0], item[0].pix.reshape(x.shape[-2:])

    @torch.no_grad()
    def forward(
        self, x: Tensor, region: Tensor | None, layer_feats: dict[int, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        out, a, fb, _ = self.block(x, region, layer_feats)
        return self._readout(out[None], a[None], torch.ones_like(fb)[None])[0], a, fb

    @torch.no_grad()
    def encode(
        self,
        maps: Sequence[Tensor],
        regions: Sequence[Tensor | None],
        layer_feats: Sequence[dict[int, Tensor] | None] | None = None,
    ) -> np.ndarray:
        feats = layer_feats if layer_feats is not None else [None] * len(maps)
        items = [self._item(x, r, f) for x, r, f in zip(maps, regions, feats)]
        g: list[np.ndarray | None] = [None] * len(items)
        for idx in self._batches(items):
            ctx, out, a, _ = self._run([items[i] for i in idx])
            for i, row in zip(idx, self._readout(out, a, ctx.valid).cpu().numpy()):
                g[i] = row
        return np.stack(g)

    def state_dict(self) -> dict:
        sd = getattr(self.builder, "state_dict", dict)
        return {"mu": self.mu, "sigma": self.sigma, "delta": self.delta, "n_marked": self.n_marked,
                "d": self.d, "builder": sd()}

    def load_state_dict(self, d: dict) -> None:
        self.mu, self.sigma, self.delta = d["mu"], d["sigma"], float(d["delta"])
        self.n_marked, self.d = int(d.get("n_marked", 0)), int(d.get("d", 0))
        if hasattr(self.builder, "load_state_dict"):
            self.builder.load_state_dict(d["builder"])
