"""Distribution regularizers for the progressive MO-DEHB NAS (prompt section 4).

Two *distinct* regularizers, both with the interface ``forward(z: Tensor[N, D]) -> scalar``:

* ``SIGReg``      -- full Sketched Isotropic Gaussian Regularization from LeJEPA
                     (Balestriero & LeCun, arXiv:2511.08544, Def. 2). Re-uses the
                     repo's ``src.losses.lejepa_loss.RealSIGReg`` (no reimplementation).
* ``WeakSIGReg``  -- covariance-only sketched variant from Akbar, "Weak-SIGReg:
                     Covariance Regularization for Stable Deep Learning",
                     arXiv:2603.05924 (GRaM @ ICLR 2026), Sec. 2.2 / Fig. 3.

Usage notes
-----------
* SimCLR: the regularizer MUST be applied to the representation BEFORE the L2
  normalization used by the InfoNCE/NT-Xent loss. After L2 normalization every
  sample lies on the unit sphere, so (a) per-sample norms are fixed and an
  isotropic N(0, I_D) target (E||z||^2 = D) is unreachable, and (b) the
  covariance trace is pinned to 1, so Weak-SIGReg's ``Cov -> I`` target is
  unreachable for D > 1. Pass the un-normalized projector output ``z``, then
  normalize separately for the contrastive term.
* LeJEPA + Weak-SIGReg is an EXPERIMENTAL variant, not the original LeJEPA.
  Original LeJEPA = invariance loss + full SIGReg (Epps-Pulley on random slices).
  Swapping in Weak-SIGReg only constrains second moments, so it does not
  guarantee isotropic Gaussianity (any non-Gaussian distribution with identity
  covariance has zero Weak-SIGReg loss).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.lejepa_loss import RealSIGReg

__all__ = ["RealSIGReg", "SIGReg", "WeakSIGReg", "build_regularizer", "REGULARIZER_NAMES"]

REGULARIZER_NAMES = ("none", "sigreg", "weak_sigreg")


def _check_2d(z: Tensor) -> None:
    if z.dim() != 2:
        raise ValueError(f"expected z of shape [N, D], got {tuple(z.shape)}")
    if z.size(0) < 2:
        raise ValueError("regularizer needs a batch with N >= 2 samples")


class SIGReg(nn.Module):
    """Full SIGReg on ``[N, D]`` embeddings -- thin wrapper over ``RealSIGReg``.

    LeJEPA Def. 2:  SIGReg_T(A, Z) = (1/|A|) sum_{a in A} T({a^T z_n}_{n=1}^N),
    with T the Epps-Pulley statistic
        T(x) = N * int |phi_hat_x(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt,
    trapezoid on [0, t_max] with doubled weights (symmetry), unit-norm Gaussian
    slice directions a ~ N(0, I_D)/||.||, redrawn every call.

    Verification against the official code (github.com/galilai-group/lejepa,
    ``lejepa/univariate/epps_pulley.py`` + ``lejepa/multivariate/slicing.py``,
    fetched 2026-11-13):
      * ``src/losses/epps_pulley.py::EppsPulley`` is line-for-line identical to
        the official ``EppsPulley`` (t grid, half-weight endpoints, phi-weighted
        weights, ``err @ weights * N * world_size``, DDP all_reduce of the ECF).
      * Slicing formula identical (``A = randn(D, K); A /= A.norm(dim=0)``; mean
        over slices).
      * Discrepancy 1 (RNG only, not formula): the official
        ``SlicingUnivariateTest`` draws A from a private ``torch.Generator``
        seeded by a ``global_step`` buffer synchronised (MAX all_reduce) across
        ranks; ``RealSIGReg`` uses the global torch RNG. Single-process results
        are therefore deterministic under ``torch.manual_seed`` but, under DDP,
        ranks only share A if their global RNG states agree.
      * Discrepancy 2: ``RealSIGReg`` creates A in the default dtype (float32);
        the official code matches it as well (``randn(..., device=x.device)``),
        so for bf16/fp16 inputs both rely on autocast. No numerical difference.
      * Official default in the README example is ``num_slices=1024``; this
        project uses ``n_proj=256`` (prompt section 4), a hyper-parameter, not a
        formula change.
    """

    def __init__(self, n_proj: int = 256, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        self.impl = RealSIGReg(n_proj=n_proj, t_max=t_max, n_points=n_points)

    def forward(self, z: Tensor) -> Tensor:
        _check_2d(z)
        # RealSIGReg expects [V, B, D]; a single "view" of N samples.
        return self.impl(z.unsqueeze(0))


class WeakSIGReg(nn.Module):
    """Weak-SIGReg (Akbar, arXiv:2603.05924, Sec. 2.2, Fig. 3 ``sigreg_weak_loss``).

    The paper gives no numbered equation; the formal definition is the Fig. 3
    listing ("Forces Covariance(x) ~ Identity via Sketching. Approximates the
    2nd Moment constraint efficiently."). Written as math, for Z in R^{N x C}:

      (1) Sketch (only if C > K):  S in R^{K x C}, S_ij ~ N(0, 1/C) i.i.d.,
          redrawn each call;  Z' = Z S^T  in R^{N x K}.  If C <= K: Z' = Z, K := C.
      (2) Center:      Z_c = Z' - 1_N mean_n(Z')
      (3) Covariance:  Sigma = Z_c^T Z_c / (N - 1 + 1e-6)
      (4) Loss:        L_weak = || Sigma - I_K ||_F      (Frobenius, NOT squared)

    Defaults follow the paper: K = ``sketch_dim`` = 64 (Fig. 3 default and
    Appendix A.1). The paper uses regularization strength alpha = 0.1 (App. A.1)
    but does not state how alpha enters the total loss; weighting is left to the
    caller (``L = L_task + alpha * L_weak``).

    Differences vs SIGReg (LeJEPA):
      * Matches only the 2nd moment (Cov -> I) of the sketched representation,
        not the full characteristic function; blind to higher moments / shape.
      * Sketch is a K x C Gaussian matrix scaled by 1/sqrt(C) (approximately
        orthonormal rows, Johnson-Lindenstrauss), not K unit-norm 1-D slices
        tested independently; cross-dimension covariances are penalised jointly.
      * Mean is removed (centered covariance) -> no penalty on the mean, whereas
        Epps-Pulley penalises a non-zero mean.
      * Cost O(N C K + N K^2), no trig/quadrature; no N scaling of the statistic.
      * Note: because S S^T != I exactly, a perfectly isotropic Z still yields a
        positive floor when C > K: ||S S^T - I||_F ~ K/sqrt(C) (~2 at C=1024,
        ~5.7 at C=128, vs ~sqrt(K)=8 for full collapse). Contrast is good at the
        paper's C=1024/K=64 or when C <= K (e.g. the 48-d FLIM GAP: no sketch),
        but weak for C just above K -- consider sketch_dim >= C there.
    """

    def __init__(self, sketch_dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        if sketch_dim < 1:
            raise ValueError("sketch_dim must be >= 1")
        self.sketch_dim = sketch_dim
        self.eps = eps  # paper: (N - 1 + 1e-6)

    def forward(self, z: Tensor) -> Tensor:
        _check_2d(z)
        N, C = z.shape
        K = self.sketch_dim
        if C > K:  # step (1)
            S = torch.randn(K, C, device=z.device, dtype=z.dtype) / (C ** 0.5)
            z = z @ S.T
        else:
            K = C
        z = z - z.mean(dim=0, keepdim=True)  # step (2)
        cov = (z.T @ z) / (N - 1 + self.eps)  # step (3)
        target = torch.eye(K, device=z.device, dtype=z.dtype)
        return torch.linalg.matrix_norm(cov - target, ord="fro")  # step (4)


def build_regularizer(name: str, **kw) -> Optional[nn.Module]:
    """Factory: ``"none"`` -> None, ``"sigreg"`` -> SIGReg(**kw), ``"weak_sigreg"`` -> WeakSIGReg(**kw).

    The returned module maps ``z: Tensor[N, D] -> scalar``. For SimCLR apply it
    to the representation BEFORE L2 normalization. LeJEPA + ``weak_sigreg`` is an
    experimental variant, not the original LeJEPA (which uses ``sigreg``).
    """
    key = (name or "none").lower().replace("-", "_")
    if key == "none":
        return None
    if key == "sigreg":
        return SIGReg(**kw)
    if key == "weak_sigreg":
        return WeakSIGReg(**kw)
    raise ValueError(f"unknown regularizer {name!r}; expected one of {REGULARIZER_NAMES}")
