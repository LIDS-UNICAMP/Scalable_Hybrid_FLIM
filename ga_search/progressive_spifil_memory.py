"""Memory-bounded Fisher scoring over the existing SPiFiL metric protocol.

The external SPiFiL package is unchanged. The covariance is fitted once to all
training patches, and each row block is scored against the complete reference
set. This removes the dense N-by-N distance/mask allocations without changing
the Fisher definition, its epsilon, or which samples participate in scoring.
"""
from __future__ import annotations

import torch

from spifil.metrics import Euclidean, Mahalanobis
from spifil.scoring import FisherScorer


class StreamingFisherScorer:
    """SPiFiL-compatible Fisher scorer retaining at most ``chunk_size * N`` distances.

    By default use the metric's own row chunk size, preserving its distance
    kernel shapes. For the validated standard Mahalanobis implementation, full
    features are whitened once with its fitted matrix, matching the upstream
    full-matrix multiplication shape exactly. Other metric implementations use
    the original dense scorer rather than an unvalidated approximation.
    """

    def __init__(self, chunk_size: int | None = None):
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer or None")
        self.chunk_size = chunk_size

    def __call__(self, patches, metric):
        if type(metric) not in (Euclidean, Mahalanobis):
            return FisherScorer()(patches, metric)
        feats, labels = patches.feats, patches.labels
        n = feats.shape[0]
        if n == 0:
            raise ValueError("Fisher scoring requires a nonempty patch set")
        chunk_size = self.chunk_size
        if chunk_size is None:
            chunk_size = getattr(metric, "chunk_size", 4096)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("metric chunk_size must be a positive integer")
        metric.fit(feats)
        reference = feats
        if type(metric) is Mahalanobis and not metric._use_euclidean:
            if metric._whitener is None:
                raise RuntimeError("Mahalanobis.fit() did not produce a fitted metric")
            # Upstream pairwise(feats, feats) performs this full-shaped product
            # on both operands before chunking cdist. Computing sliced products
            # instead may change floating-point rounding and filter rankings.
            reference = feats @ metric._whitener
        scores = []
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            # Match the upstream metric's cdist row shapes, including the final
            # partial chunk and torch.cdist's size-dependent algorithm choice.
            dist = torch.cdist(reference[start:stop], reference)
            same = labels[start:stop, None] == labels[None, :]
            diff = ~same
            # These are global column indices, not a local block diagonal.
            rows = torch.arange(stop - start, device=dist.device)
            same[rows, rows + start] = False
            cnt_intra, cnt_inter = same.sum(dim=1), diff.sum(dim=1)
            sum_intra = (dist * same).sum(dim=1)
            sum_inter = (dist * diff).sum(dim=1)
            zero = torch.zeros_like(sum_intra)
            avg_intra = torch.where(cnt_intra > 0, sum_intra / cnt_intra.clamp(min=1), zero)
            avg_inter = torch.where(cnt_inter > 0, sum_inter / cnt_inter.clamp(min=1), zero)
            scores.append(avg_inter / (avg_intra + 1e-6))
            # Release large buffers before allocating the next row block.
            del dist, same, diff
        return torch.cat(scores, dim=0)
