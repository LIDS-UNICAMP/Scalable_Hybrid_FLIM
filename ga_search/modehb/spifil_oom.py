"""OOM fallback for the SPiFiL Fisher scorer (library untouched, process-local patch).

``spifil.scoring.FisherScorer`` materializes the full ``N x N`` distance matrix plus
``N x N`` masks (and an int64 ``N x N`` temporary in ``same.sum``): with N ~ 42k seeds
that is > 25 GB at peak, so on a shared GPU the fit OOMs even though the candidate is
otherwise cheap (remote_sensing hybrid, job 43, 417 images x 100 seeds).

``install()`` wraps ``FisherScorer.__call__``: the ORIGINAL library code runs first, so
every candidate that fits today gets bit-identical scores (same code path, nothing
changes). ONLY when it raises a CUDA OOM, the cache is freed and the same scores are
recomputed in row blocks:

* distances: the exact ``torch.cdist`` calls of ``metrics._chunked_cdist`` (same
  ``chunk_size`` row blocks, same whitened features for ``Mahalanobis``), so every
  distance row is the one the library would have produced;
* Fisher ratio: the same expressions as ``scoring._fisher_ratio`` per row block; the
  counts come from the class histogram (exact integers). Only the CUDA reduction
  order of the two row sums may differ (shape-dependent kernel config) -> fp rounding.

Peak memory is bounded by ``chunk_size x N`` (dist block) + ``sub_rows x N`` (temps). If
the blocked GPU pass still OOMs, it is repeated on CPU and the scores are moved back.
"""
from __future__ import annotations

import gc

import torch

_ORIG = None
STATS = {"fallbacks": 0, "cpu_fallbacks": 0}
SUB_ROWS = 1024


def is_oom(exc: BaseException) -> bool:
    """CUDA OOM, directly or wrapped (torch 2.2 and 2.7)."""
    while exc is not None:
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _row_source(metric, feats):
    """``(x, chunk)`` so that ``cdist(x[s:s+chunk], x)`` == ``metric.pairwise(feats, feats)[s:s+chunk]``,
    or ``(None, chunk)`` for an unknown metric (generic row-block ``pairwise``)."""
    from spifil.metrics import Euclidean, Mahalanobis
    chunk = int(getattr(metric, "chunk_size", 4096) or 4096)
    if isinstance(metric, Euclidean):
        return feats, chunk
    if isinstance(metric, Mahalanobis):
        if metric._use_euclidean:
            return feats, chunk
        if metric._whitener is not None:
            return feats @ metric._whitener, chunk
    return None, chunk


def fisher_blocked(patches, metric, device=None):
    """Same scores as ``FisherScorer()(patches, metric)`` with bounded peak memory."""
    feats, labels = patches.feats, patches.labels
    if device is not None:
        feats, labels = feats.to(device), labels.to(device)
    metric.fit(feats)                        # deterministic: same whitener as the failed call
    x, chunk = _row_source(metric, feats)
    n = feats.shape[0]
    _, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    own = counts[inv]                        # class size incl. self (exact ints)
    cnt_intra_all, cnt_inter_all = own - 1, n - own
    out = torch.empty(n, dtype=feats.dtype, device=feats.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        if x is not None:
            dist = torch.cdist(x[s:e], x)
        else:
            dist = metric.pairwise(feats[s:e], feats)
        for a in range(0, e - s, SUB_ROWS):
            b = min(a + SUB_ROWS, e - s)
            d = dist[a:b]
            lab = labels[s + a:s + b]
            same = lab[:, None] == labels[None, :]
            rows = torch.arange(b - a, device=d.device)
            same[rows, rows + s + a] = False             # ~eye
            diff = lab[:, None] != labels[None, :]
            sum_intra = (d * same).sum(dim=1)
            sum_inter = (d * diff).sum(dim=1)
            del same, diff
            cnt_intra, cnt_inter = cnt_intra_all[s + a:s + b], cnt_inter_all[s + a:s + b]
            zero = torch.zeros_like(sum_intra)
            avg_intra = torch.where(cnt_intra > 0, sum_intra / cnt_intra.clamp(min=1), zero)
            avg_inter = torch.where(cnt_inter > 0, sum_inter / cnt_inter.clamp(min=1), zero)
            out[s + a:s + b] = avg_inter / (avg_intra + 1e-6)
        del dist
    return out


def _patched_call(self, patches, metric):
    try:
        return _ORIG(self, patches, metric)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_oom(exc):
            raise
    # outside the except block: the failed frames (full N x N tensors) are released
    _free()
    STATS["fallbacks"] += 1
    dev = patches.feats.device
    try:
        return fisher_blocked(patches, metric)
    except Exception as exc:  # noqa: BLE001
        if not is_oom(exc) or dev.type == "cpu":
            raise
    _free()
    STATS["cpu_fallbacks"] += 1
    scores = fisher_blocked(patches, metric, device="cpu")
    metric.fit(patches.feats)                # metric state back on the original device
    return scores.to(dev)


def install() -> None:
    """Idempotent, process-local: wraps ``spifil.scoring.FisherScorer.__call__``."""
    global _ORIG
    from spifil.scoring import FisherScorer
    if getattr(FisherScorer.__call__, "_modehb_oom_fallback", False):
        return
    _ORIG = FisherScorer.__call__
    _patched_call._modehb_oom_fallback = True
    FisherScorer.__call__ = _patched_call
