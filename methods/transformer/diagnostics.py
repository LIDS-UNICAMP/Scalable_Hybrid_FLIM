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

"""Per-image diagnostics of a ``CrossOutput`` (spec section 7). Pure functions, no fitting."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import torch
from torch import Tensor

__all__ = ["fallback_fraction", "token_variance", "head_diversity", "paint"]


def _count(valid: Tensor) -> Tensor:
    return valid.sum(-1).clamp(min=1).float()


def fallback_fraction(out) -> Tensor:
    """(L, B): share of valid tokens whose self-attention fell back to the global top-k."""
    valid = out.tokens.valid
    return torch.stack([(f & valid).sum(-1).float() / _count(valid) for f in out.fallback])


def token_variance(out) -> Tensor:
    """(L+1, B): variance across the image's valid tokens, averaged over channels.

    A sharp drop from one layer to the next means the layers are homogenising the tokens.
    """
    m = out.tokens.valid[..., None].float()
    n = _count(out.tokens.valid)[:, None]
    rows = []
    for t in out.T:
        mean = (t * m).sum(1) / n
        rows.append((((t - mean[:, None]) ** 2) * m).sum(1).div(n).mean(-1))
    return torch.stack(rows)


def head_diversity(out, head_names: Sequence[str]) -> Tensor:
    """(L, B): mean pairwise Jensen-Shannon divergence between the full-support sub-heads' A_h.

    Full support = the sub-head attends over all K prototypes; per_image sub-heads are left out.
    Averaged over valid tokens. NaN when fewer than two such sub-heads exist.
    """
    valid = out.tokens.valid
    k = out.tokens.t.shape[-1] // 2
    rows = []
    for layer in out.A_cross:
        full = [a for name, a in zip(head_names, layer) if not name.startswith("per_image") and a.shape[-1] == k]
        pairs = list(combinations(full, 2))
        if not pairs:
            rows.append(torch.full(valid.shape[:1], float("nan"), device=valid.device))
            continue
        js = torch.stack([_js(p, q) for p, q in pairs]).mean(0)  # (B, N)
        rows.append((js * valid).sum(-1) / _count(valid))
    return torch.stack(rows)


def _js(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    m = 0.5 * (p + q)
    kl = lambda a: (a * (torch.log(a + eps) - torch.log(m + eps))).sum(-1)
    return 0.5 * kl(p) + 0.5 * kl(q)


def paint(values: Tensor, labels: Tensor) -> Tensor:
    """Per-token ``values`` (B, N) -> (B, H', W') through the token id map ``labels``."""
    return values.gather(1, labels.flatten(1)).view(labels.shape)
