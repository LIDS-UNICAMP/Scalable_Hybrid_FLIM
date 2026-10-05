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

"""Training-free transformer over superpixel tokens (USER_SPEC_V5 sections 2 to 5).

Each layer: graph self-attention (neighbours first, global top-k fallback, T + alpha A T) then
multi-head cross-attention to the K SPiFiL prototypes (softmax over the prototype logits of each
registered head, value = onehot of the prototype class, background gate). Readout: area and
confidence weighted mean of the last T and E. Closed-form statistics only, no grad, any device.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as tnf
from torch import Tensor

from methods.transformer.config import CROSS_READOUTS, CrossTransformerConfig
from methods.transformer.heads import HeadContext, build_heads
from methods.transformer.model import best_neighbour, graph_attention
from methods.transformer.tokenizers import TOKENIZER_REGISTRY, TokenBatch, tokenize_batch

__all__ = ["CrossOutput", "CrossTransformer"]

_BATCH = 16  # ponytail: fixed images per batch, as in model.py


@dataclass
class CrossOutput:
    tokens: TokenBatch
    T: list[Tensor]  # len L+1: T[0] input tokens, T[l] after layer l self-attention, (B, N, D)
    A_self: list[Tensor]  # len L, (B, N, N), zeros when self_attention is off
    fallback: list[Tensor]  # len L, (B, N) bool
    A_cross: list[list[Tensor]]  # [layer][sub-head] (B, N, K_h)
    E: list[Tensor]  # len L, (B, N, H, C) gated evidence
    weights: Tensor  # (B, N) readout weights, sum 1 over valid tokens
    g: Tensor  # (B, D + H * C)


def _quantile(x: Tensor, pct: float) -> float:
    x = x[torch.isfinite(x)].float()
    return float(torch.quantile(x, pct / 100.0)) if x.numel() else float("nan")


class CrossTransformer:
    def __init__(self, cfg: CrossTransformerConfig) -> None:
        if cfg.tokenizer not in TOKENIZER_REGISTRY:
            raise ValueError(f"tokenizer desconhecido: {cfg.tokenizer!r}; registrados: {sorted(TOKENIZER_REGISTRY)}")
        if cfg.readout not in CROSS_READOUTS:
            raise ValueError(f"readout desconhecido: {cfg.readout!r}; opcoes: {CROSS_READOUTS}")
        self.cfg = cfg
        self.heads = build_heads(cfg.heads)
        self.mu: Tensor | None = None
        self.sigma: Tensor | None = None
        self.delta: list[float] = []
        self.gamma: list[list[float]] = []
        self.head_names: tuple[str, ...] = ()
        self.classes: tuple[int, ...] = ()
        self.kernel_class: Tensor | None = None
        self.kernel_source: Tensor | None = None
        self.n_interest = 0
        self.warnings: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Pieces of a layer
    # ------------------------------------------------------------------

    def _z(self, t: Tensor) -> Tensor:
        return (t - self.mu.to(t.device)) / (self.sigma.to(t.device) + self.cfg.eps)

    def _phi(self, tb: TokenBatch, t: Tensor) -> Tensor:
        """phi_s = L2-normalised [z_s ; mean of z_u over adj(s)] (0 for no neighbour)."""
        z = self._z(t).masked_fill(~tb.valid[..., None], 0.0)
        a = tb.adj.to(z.dtype)
        nb = (a @ z) / a.sum(-1, keepdim=True).clamp_min(1.0)
        return tnf.normalize(torch.cat([z, nb], -1), dim=-1)

    def _self(self, tb: TokenBatch, t: Tensor, delta: float) -> tuple[Tensor, Tensor, Tensor]:
        c = self.cfg
        b, n = tb.valid.shape
        if not c.self_attention:
            return t, t.new_zeros(b, n, n), torch.zeros_like(tb.valid)
        phi = self._phi(tb, t)
        a, fb = graph_attention(
            phi, phi, tb.adj, tb.valid, delta, c.global_topk, c.tau_self, c.chunk_threshold, c.chunk_size
        )
        return t + c.alpha * (a @ t), a, fb

    def _subheads(self, tb: TokenBatch, t: Tensor) -> list[tuple[str, Tensor, Tensor]]:
        dev = t.device
        src = None if self.kernel_source is None else self.kernel_source.to(dev)
        ctx = HeadContext(self._z(t), tb.adj, tb.valid, self.kernel_class.to(dev), src)
        return [sub for head in self.heads for sub in head(ctx)]

    def _cross(self, tb: TokenBatch, subs, gamma: Sequence[float]) -> tuple[list[Tensor], Tensor]:
        """Per sub-head attention over its prototypes and gated class evidence (B, N, H, C)."""
        onehot = (self.kernel_class.to(tb.t.device)[:, None] == torch.tensor(self.classes, device=tb.t.device)).float()
        keep = tb.valid[..., None]
        att, ev = [], []
        for (_, logits, idx), gm in zip(subs, gamma):
            a = torch.softmax(logits / self.cfg.tau_cross, -1) * keep
            e = a @ onehot[idx.to(onehot.device)]
            if self.cfg.gate:
                e = e * (logits.amax(-1, keepdim=True) >= gm)
            att.append(a)
            ev.append(e)
        return att, torch.stack(ev, 2)

    def _readout(self, tb: TokenBatch, t: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        area = tb.area * tb.valid
        w = area
        if self.cfg.readout == "weighted":
            cw = e.mean(2).amax(-1) * area
            w = torch.where(cw.sum(1, keepdim=True) > 0, cw, area)  # all conf 0 -> area only
        w = w / w.sum(1, keepdim=True).clamp_min(self.cfg.eps)
        g = torch.cat([(w[..., None] * t).sum(1), (w[..., None] * e.flatten(2)).sum(1)], 1)
        return w, g

    # ------------------------------------------------------------------
    # Fit / forward
    # ------------------------------------------------------------------

    @staticmethod
    def _groups(F: Sequence[Tensor]) -> list[list[int]]:
        """Indices of same-shape feature maps, at most _BATCH each."""
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, f in enumerate(F):
            groups[tuple(f.shape)].append(i)
        return [g[j : j + _BATCH] for g in groups.values() for j in range(0, len(g), _BATCH)]

    @torch.no_grad()
    def fit(
        self,
        F: Sequence[Tensor],
        segs,
        masks,
        kernel_class: Tensor,
        kernel_source: Tensor | None = None,
    ) -> "CrossTransformer":
        """mu, sigma on layer-0 interest tokens; then delta[l] and gamma[l][h] layer by layer."""
        if kernel_class is None:
            raise ValueError("kernel_class (classe de cada kernel) e obrigatorio")
        c = self.cfg
        self.kernel_class = torch.as_tensor(kernel_class).long().flatten()
        self.kernel_source = None if kernel_source is None else torch.as_tensor(kernel_source).long().flatten()
        self.classes = tuple(int(x) for x in torch.unique(self.kernel_class).tolist())
        pick = lambda xs, idx: None if xs is None else [xs[i] for i in idx]  # noqa: E731
        batches = [
            tokenize_batch(torch.stack([F[i] for i in idx]), pick(segs, idx), pick(masks, idx), c)
            for idx in self._groups(F)
        ]
        warns: list[str] = []
        self.n_interest = int(sum(int(tb.interest.sum()) for tb in batches))
        sel = [tb.interest for tb in batches]
        if self.n_interest < 2:
            # ponytail: 1 token gives sigma = 0; all valid tokens are the safer stats.
            warns.append(f"{self.n_interest} tokens de interesse: mu, sigma, delta e gamma usam todos os tokens")
            sel = [tb.valid for tb in batches]
        t0 = torch.cat([tb.t[s] for tb, s in zip(batches, sel)])
        self.mu, self.sigma = t0.mean(0), t0.std(0, unbiased=False)
        ts = [tb.t for tb in batches]
        self.delta, self.gamma = [], []
        for layer in range(c.num_layers):
            delta = float("nan")
            if c.self_attention:
                best = []
                # Interest tokens only, like gamma: near-identical background
                # superpixels would otherwise pull the percentile to 1.
                for tb, t, s in zip(batches, ts, sel):
                    phi = self._phi(tb, t)
                    best.append(best_neighbour(phi, phi, tb.adj, c.chunk_threshold, c.chunk_size)[s])
                best = torch.cat(best)
                delta = _quantile(best[torch.isfinite(best)], c.delta_percentile)
                if np.isnan(delta):
                    delta = float("inf")
                    warns.append(f"camada {layer + 1}: nenhum token com vizinhos, todos usam o fallback global")
            self.delta.append(delta)
            ts = [self._self(tb, t, delta)[0] for tb, t in zip(batches, ts)]
            subs = [self._subheads(tb, t) for tb, t in zip(batches, ts)]
            self.head_names = tuple(name for name, _, _ in subs[0])
            self.gamma.append([
                _quantile(torch.cat([s[h][1].amax(-1)[m] for s, m in zip(subs, sel)]), c.gate_percentile)
                for h in range(len(self.head_names))
            ])
        self.warnings = tuple(warns)
        return self

    @torch.no_grad()
    def forward(self, F: Tensor, segs, masks=None) -> CrossOutput:
        """Full per-layer output for a batch F (B, K, H', W'); segs len B, or None for the grid."""
        if self.mu is None:
            raise RuntimeError("CrossTransformer.fit precisa ser chamado antes")
        tb = tokenize_batch(F, segs, masks, self.cfg)
        ts, a_self, fbs, a_cross, es = [tb.t], [], [], [], []
        for layer in range(self.cfg.num_layers):
            t, a, fb = self._self(tb, ts[-1], self.delta[layer])
            att, e = self._cross(tb, self._subheads(tb, t), self.gamma[layer])
            ts.append(t)
            a_self.append(a)
            fbs.append(fb)
            a_cross.append(att)
            es.append(e)
        w, g = self._readout(tb, ts[-1], es[-1])
        return CrossOutput(tb, ts, a_self, fbs, a_cross, es, w, g)

    __call__ = forward

    @torch.no_grad()
    def encode(self, F: Sequence[Tensor], segs) -> np.ndarray:
        """Global vectors g (n, D + H*C), batched by feature-map shape."""
        g: list[np.ndarray | None] = [None] * len(F)
        for idx in self._groups(F):
            out = self.forward(torch.stack([F[i] for i in idx]), None if segs is None else [segs[i] for i in idx])
            for i, row in zip(idx, out.g.cpu().numpy()):
                g[i] = row
        return np.stack(g)

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "mu", "sigma", "delta", "gamma", "head_names", "classes", "kernel_class", "kernel_source",
            "n_interest", "warnings",
        )}

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
