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

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.epps_pulley import EppsPulley


class SimpleSIGReg(nn.Module):
    """
    Simplified SIGReg using moment-matching as test statistic T.

    For each random projection direction a_m, the test statistic is:
        T(s_m) = mean(s_m)^2 + (std(s_m) - 1)^2

    which penalizes deviation of the projected distribution from zero mean
    and unit variance — the first two moments of N(0,1).

    SIGReg_simple = (1/|A|) * sum_{a in A} T({a^T z_{n,v}}_{n=1}^B)

    This is cheaper than the Epps-Pulley test and requires no integration,
    but is less powerful as a Gaussianity test.

    Args:
        n_proj: Number of random projection directions. Default: 256.
    """

    def __init__(self, n_proj: int = 256) -> None:
        super().__init__()
        self.n_proj = n_proj

    def forward(self, proj: Tensor) -> Tensor:
        """
        Args:
            proj: Projected embeddings [V, B, D].

        Returns:
            Scalar SimpleSIGReg loss.
        """
        A = torch.randn(proj.size(-1), self.n_proj, device=proj.device)
        A = A / A.norm(p=2, dim=0)            # unit-norm columns [D, n_proj]
        s = proj @ A                           # [V, B, n_proj]

        # T(x) = mean(x)^2 + (std(x) - 1)^2, computed over batch dim (dim 1)
        mean_sq  = s.mean(dim=1).square()      # [V, n_proj]
        std_term = (s.std(dim=1, correction=0) - 1.0).square()  # [V, n_proj]
        return (mean_sq + std_term).mean()


class RealSIGReg(nn.Module):
    """
    Full SIGReg (Sketched Isotropic Gaussian Regularization) using the
    Epps-Pulley characteristic function test statistic T, as recommended
    in the LeJEPA paper.

    Follows the exact paper definition (Def. 2):
        SIGReg_T(A, {z_{n,v}}_{n=1}^B) = (1/|A|) * sum_{a in A} T({a^T z_{n,v}}_{n=1}^B)

    where T is the Epps-Pulley statistic:
        T(x) = N * integral |ECF(x,t) - exp(-t^2/2)|^2 * exp(-t^2/2) dt

    The full training objective averages SIGReg over all V views:
        lambda * (1/V) * sum_v SIGReg_T(A, {f(x_{n,v})}_{n=1}^B)

    This is handled here by averaging over both views and directions.

    Args:
        n_proj:   Number of random projection directions. Default: 256.
        t_max:    Upper integration limit for the EP statistic. Default: 3.0.
        n_points: Quadrature nodes for the EP integral (must be odd). Default: 17.

    Reference: LeJEPA paper, Def. 2 and Sec. 4.2.3 (Epps-Pulley).
    """

    def __init__(
        self, n_proj: int = 256, t_max: float = 3.0, n_points: int = 17
    ) -> None:
        super().__init__()
        self.n_proj = n_proj
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)

    def forward(self, proj: Tensor) -> Tensor:
        """
        Args:
            proj: Projected embeddings [V, B, D].

        Returns:
            Scalar RealSIGReg loss.
        """
        _, _, D = proj.shape
        A = torch.randn(D, self.n_proj, device=proj.device)
        A = A / A.norm(p=2, dim=0)             # unit-norm columns [D, n_proj]
        s = proj @ A                            # [V, B, n_proj]

        # EppsPulley accepts (*, B, M) — V is treated as a leading batch dim,
        # so all views are processed in one vectorised call instead of a Python loop.
        return self.ep(s).mean()               # [V, n_proj] → scalar


# Alias kept for backward compatibility with existing configs/code
SIGReg = RealSIGReg


def invariance_loss(proj: Tensor) -> Tensor:
    """
    Invariance (prediction) loss: pulls each view's embedding toward the
    global-view mean, following the LeJEPA alignment objective.

        L_pred = (1/V) * sum_v || mean_v(z) - z_v ||^2

    Args:
        proj: Stacked projected embeddings [V, B, D].

    Returns:
        Scalar invariance loss.
    """
    return (proj.mean(0) - proj).square().mean()
