# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""
Loss SIGReg completa (Epps-Pulley) do LeJEPA.

Movida verbatim, sem tocar em nenhuma constante numerica:
  - src/losses/base.py:24-53        is_dist_avail_and_initialized, UnivariateTest
  - src/losses/epps_pulley.py:26-87 all_reduce, EppsPulley
  - src/losses/lejepa_loss.py:69-116 RealSIGReg

A arvore-alvo nao tem arquivo proprio para EppsPulley nem para UnivariateTest,
entao os dois viajam junto com a loss que os usa, no mesmo arquivo fechado.
t_max=3.0, n_points=17 e a quadratura trapezoidal sao a definicao da
estatistica: copiados como estao.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed.nn
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if is_dist_avail_and_initialized():
            torch.distributed.nn.functional.all_reduce(
                x, torch.distributed.ReduceOp.AVG
            )
        return x

    @property
    def world_size(self):
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, op)
    else:
        return x


class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley test statistic for normality via empirical characteristic function.

    Compares the empirical CF to the standard Gaussian CF in a weighted L2 norm:
        T(x) = N * integral |ECF(x, t) - exp(-t^2/2)|^2 * w(t) dt

    where w(t) = exp(-t^2/2) and integration is over [0, t_max] using the
    trapezoid rule (symmetry of |ECF|^2 halves the domain).

    Args:
        t_max:    Upper integration limit. Default: 3.
        n_points: Number of quadrature nodes (must be odd). Default: 17.
        integration: Integration method (only 'trapezoid' supported). Default: 'trapezoid'.
    """

    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # half-weight at boundaries
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (*, B, M) where B is the batch dimension (-2)
               and M is the number of directions/projections (-1).

        Returns:
            Tensor of shape (*, M) — one EP statistic per direction.
        """
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t   # (*, B, M, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        cos_mean = cos_vals.mean(-3)      # mean over B → (*, M, n_points)
        sin_mean = sin_vals.mean(-3)

        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size   # (*, M)


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
