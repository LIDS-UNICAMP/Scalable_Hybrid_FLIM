"""Label-free embedding evaluator for the progressive MO-DEHB NAS (prompt §8, §9).

``EmbeddingEvaluator(cfg).evaluate(views)`` takes ``views: Tensor[V, N, D]``
(V augmented views of the same N samples, D-dim embeddings) and returns a flat
dict of label-free metrics:

* spectral: RankMe (Garrido et al., ICML 2023), effective rank (participation
  ratio), effective_rank_ratio = effective_rank / min(N, D), rank_limit,
  top-k singular spectrum, covariance condition number;
* per-dimension: variance mean/min, fraction of near-constant dims, mean
  |off-diagonal correlation| (redundancy);
* view geometry: alignment & uniformity (Wang & Isola, 2020), kNN view
  consistency (k=10), k-means clustering stability (ARI between views);
* drift: linear CKA vs ``prev`` (previous stage); CKA + mean cosine vs
  ``other_init`` (SPiFiL vs random twin);
* norm distribution stats.

``quality(metrics)`` is a FIXED fitness in [0, 1]:
``0.5 * knn_view_consistency + 0.5 * clustering_stability`` — it never reads
labels or loss values.

Linear CKA is computed in feature space (||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)),
which is mathematically identical to the Gram-matrix HSIC form used by
``analysis/distill/embedding_analysis.py::linear_cka`` but O(N D^2) instead of
O(N^2), and in torch (no import of that heavy analysis script).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

__all__ = ["EmbeddingEvalConfig", "EmbeddingEvaluator", "linear_cka", "rankme", "effective_rank"]


# ─────────────────────────────────────────────────────────────── config
@dataclass
class EmbeddingEvalConfig:
    eps: float = 1e-7                       # RankMe eps (Garrido et al.)
    top_k_spectrum: int = 16                # singular values logged
    max_full_dim: int = 2048                # D above this -> randomized low-rank SVD
    lowrank_q: int = 256                    # rank for torch.svd_lowrank
    spectral_every: int = 1                 # compute spectra every n evaluate() calls
    near_constant_tau: float = 1e-6         # per-dim variance threshold
    knn_k: int = 10
    cluster_k: int = 8
    cluster_n_init: int = 4
    max_samples: int = 4096                 # subsample N for O(N^2) kNN / k-means
    uniformity_t: float = 2.0
    seed: int = 0
    # collapse
    tau_total: float = 1e-6                 # mean var below -> total collapse
    tau_dimensional: float = 0.05           # effective_rank_ratio below -> dimensional collapse
    # alerts
    redundancy_tau: float = 0.8             # mean |offdiag corr| above -> excessive redundancy
    saturation_n: int = 5                   # quality saturation window
    saturation_delta: float = 1e-3
    nonconv_n: int = 5                      # loss non-convergence window
    nonconv_rel_delta: float = 1e-3
    divergence_factor: float = 3.0          # loss > factor * min(history) -> divergence

    @classmethod
    def from_any(cls, cfg: Any) -> "EmbeddingEvalConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        names = {f.name for f in fields(cls)}
        if isinstance(cfg, Mapping):
            src = dict(cfg)
        else:
            src = {n: getattr(cfg, n) for n in names if hasattr(cfg, n)}
        return cls(**{k: v for k, v in src.items() if k in names})


# ─────────────────────────────────────────────────────────────── primitives
def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def rankme(s: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe = exp(H(p)), p_k = s_k/||s||_1 + eps (Garrido et al. 2023)."""
    s = s.to(torch.float64)
    if s.numel() == 0 or float(s.sum()) <= 0:
        return 0.0
    p = s / s.sum() + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


def effective_rank(s: torch.Tensor) -> float:
    """Participation ratio of covariance eigenvalues λ=s²: (Σλ)² / Σλ²."""
    lam = s.to(torch.float64) ** 2
    den = float((lam ** 2).sum())
    if den <= 0:
        return 0.0
    return float(lam.sum() ** 2 / den)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019), feature-space form. x:[N,Dx], y:[N,Dy]."""
    x = _center(x.to(torch.float64))
    y = _center(y.to(torch.float64))
    hxy = torch.linalg.matrix_norm(y.T @ x) ** 2
    hxx = torch.linalg.matrix_norm(x.T @ x)
    hyy = torch.linalg.matrix_norm(y.T @ y)
    den = float(hxx * hyy)
    if den <= 0 or not math.isfinite(den):
        return float("nan")
    return float(hxy / den)


def _singular_values(xc: torch.Tensor, cfg: EmbeddingEvalConfig) -> torch.Tensor:
    n, d = xc.shape
    if d > cfg.max_full_dim:
        q = min(cfg.lowrank_q, n, d)
        _, s, _ = torch.svd_lowrank(xc, q=q, niter=4)
        return s
    return torch.linalg.svdvals(xc)


def _knn_indices(z: torch.Tensor, k: int) -> torch.Tensor:
    zn = torch.nn.functional.normalize(z, dim=1)
    sim = zn @ zn.T
    sim.fill_diagonal_(-float("inf"))
    return sim.topk(k, dim=1).indices


# ─────────────────────────────────────────────────────────────── evaluator
class EmbeddingEvaluator:
    def __init__(self, cfg: Any = None):
        self.cfg = EmbeddingEvalConfig.from_any(cfg)
        self._calls = 0
        self._last_spectral: Dict[str, Any] = {}

    # ---------------------------------------------------------------- main
    @torch.no_grad()
    def evaluate(self, views: torch.Tensor, prev: Optional[torch.Tensor] = None,
                 other_init: Optional[torch.Tensor] = None,
                 structural_rank: Optional[int] = None) -> Dict[str, Any]:
        """``structural_rank``: rank cap imposed by the architecture (e.g. a 1x1 head
        from C channels gives rank <= n_blocks*C). effective_rank_ratio is computed
        against min(N, D, structural_rank) so a narrow trunk is not mistaken for
        dimensional collapse."""
        cfg = self.cfg
        views = torch.as_tensor(views).detach()
        if views.dim() == 2:
            views = views.unsqueeze(0)
        if views.dim() != 3:
            raise ValueError(f"views must be [V,N,D], got {tuple(views.shape)}")
        V, N, D = views.shape
        views = views.to(torch.float32)
        z = views[0]
        m: Dict[str, Any] = {"n_views": V, "n_samples": N, "dim": D,
                             "rank_limit": min(N, D, int(structural_rank) if structural_rank else D),
                             "structural_rank": int(structural_rank) if structural_rank else None}
        self._structural = m["rank_limit"]
        m["finite"] = bool(torch.isfinite(views).all())
        if not m["finite"]:
            views = torch.nan_to_num(views, nan=0.0, posinf=0.0, neginf=0.0)
            z = views[0]

        # ---- spectral (float64), every `spectral_every` calls
        do_spec = (self._calls % max(1, cfg.spectral_every) == 0) or not self._last_spectral
        self._calls += 1
        if do_spec:
            self._last_spectral = self._spectral(z)
            self._last_spectral["spectral_fresh"] = True
        else:
            self._last_spectral["spectral_fresh"] = False
        m.update(self._last_spectral)

        # ---- per-dimension stats
        z64 = z.to(torch.float64)
        var = z64.var(dim=0, unbiased=False)
        m["var_mean"] = float(var.mean())
        m["var_min"] = float(var.min())
        m["frac_near_constant_dims"] = float((var < cfg.near_constant_tau).double().mean())
        m["mean_abs_offdiag_corr"] = self._mean_abs_offdiag_corr(z64, var)

        # ---- norms
        norms = z64.norm(dim=1)
        m["norm_mean"] = float(norms.mean())
        m["norm_std"] = float(norms.std(unbiased=False))
        m["norm_min"] = float(norms.min())
        m["norm_max"] = float(norms.max())

        # ---- subsample for O(N^2) metrics
        gen = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(N, generator=gen)[: cfg.max_samples] if N > cfg.max_samples else torch.arange(N)
        vs = views[:, idx]

        m["uniformity"] = self._uniformity(vs[0])
        if V >= 2 and m["var_mean"] < cfg.tau_total:
            # totally collapsed: neighbourhoods are tie-broken noise, not structure
            m["alignment"] = self._alignment(vs[0], vs[1])
            m["knn_view_consistency"] = 0.0
            m["clustering_stability"] = 0.0
        elif V >= 2:
            m["alignment"] = self._alignment(vs[0], vs[1])
            m["knn_view_consistency"] = self._knn_consistency(vs[0], vs[1])
            m["clustering_stability"] = self._cluster_stability(vs[0], vs[1])
        else:
            m["alignment"] = float("nan")
            m["knn_view_consistency"] = float("nan")
            m["clustering_stability"] = float("nan")

        # ---- drift / twin comparison
        if prev is not None:
            p = torch.as_tensor(prev).detach().float()
            p = p[0] if p.dim() == 3 else p
            m["cka_prev"] = linear_cka(z, p) if p.shape[0] == N else float("nan")
        if other_init is not None:
            o = torch.as_tensor(other_init).detach().float()
            o = o[0] if o.dim() == 3 else o
            if o.shape[0] == N:
                m["cka_other_init"] = linear_cka(z, o)
                if o.shape[1] == D:
                    cos = torch.nn.functional.cosine_similarity(z, o, dim=1)
                    m["cos_other_init"] = float(cos.mean())
                    m["dist_other_init"] = float((z - o).norm(dim=1).mean())
            else:
                m["cka_other_init"] = float("nan")
        m["quality"] = self.quality(m)
        c = self.collapse(m)
        m["collapse_total"], m["collapse_dimensional"] = c["total"], c["dimensional"]
        return m

    # ---------------------------------------------------------------- pieces
    def _spectral(self, z: torch.Tensor) -> Dict[str, Any]:
        cfg = self.cfg
        xc = _center(z.to(torch.float64))
        n, d = xc.shape
        s = _singular_values(xc, cfg)
        lim = min(n, d, getattr(self, "_structural", d) or d)
        er = effective_rank(s)
        out: Dict[str, Any] = {
            "rankme": rankme(s, cfg.eps),
            "effective_rank": er,
            "effective_rank_ratio": min(1.0, er / lim) if lim > 0 else 0.0,
            "spectrum_topk": s[: cfg.top_k_spectrum].cpu().numpy().tolist(),
        }
        lam = s ** 2 / max(n - 1, 1)
        lam_min = float(lam[-1]) if (s.numel() == lim and lim == d) else 0.0
        lam_max = float(lam[0]) if lam.numel() else 0.0
        out["cov_condition_number"] = (lam_max / lam_min) if lam_min > 0 else float("inf")
        return out

    @staticmethod
    def _mean_abs_offdiag_corr(z64: torch.Tensor, var: torch.Tensor) -> float:
        d = z64.shape[1]
        if d < 2:
            return 0.0
        keep = var > 1e-12
        if int(keep.sum()) < 2:
            return 0.0
        x = _center(z64[:, keep])
        x = x / x.std(dim=0, unbiased=False, keepdim=True)
        c = (x.T @ x) / x.shape[0]
        k = c.shape[0]
        off = c[~torch.eye(k, dtype=torch.bool)]
        return float(off.abs().mean())

    def _uniformity(self, z: torch.Tensor) -> float:
        zn = torch.nn.functional.normalize(z.double(), dim=1)
        if zn.shape[0] < 2:
            return float("nan")
        sq = torch.pdist(zn).pow(2)
        return float(torch.logsumexp(-self.cfg.uniformity_t * sq, 0) - math.log(sq.numel()))

    @staticmethod
    def _alignment(a: torch.Tensor, b: torch.Tensor) -> float:
        an = torch.nn.functional.normalize(a.double(), dim=1)
        bn = torch.nn.functional.normalize(b.double(), dim=1)
        return float((an - bn).norm(dim=1).pow(2).mean())

    def _knn_consistency(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Mean Jaccard-free overlap |kNN_a(i) ∩ kNN_b(i)| / k, averaged over i."""
        n = a.shape[0]
        k = min(self.cfg.knn_k, n - 1)
        if k < 1:
            return float("nan")
        ia, ib = _knn_indices(a, k), _knn_indices(b, k)
        ma = torch.zeros(n, n, dtype=torch.bool)
        ma.scatter_(1, ia, True)
        overlap = ma.gather(1, ib).sum(dim=1).double() / k
        return float(overlap.mean())

    def _cluster_stability(self, a: torch.Tensor, b: torch.Tensor) -> float:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        n = a.shape[0]
        k = min(self.cfg.cluster_k, n)
        if k < 2:
            return float("nan")
        res = []
        for x in (a, b):
            xn = np.asarray(torch.nn.functional.normalize(x.double(), dim=1).cpu().numpy())
            if np.unique(xn.round(12), axis=0).shape[0] < k:
                res.append(np.zeros(n, dtype=int))  # degenerate: single cluster
                continue
            km = KMeans(n_clusters=k, n_init=self.cfg.cluster_n_init, random_state=self.cfg.seed)
            res.append(km.fit_predict(xn))
        if len(np.unique(res[0])) == 1 and len(np.unique(res[1])) == 1:
            return 0.0  # collapsed: no structure to be stable about
        return float(max(0.0, adjusted_rand_score(res[0], res[1])))

    # ---------------------------------------------------------------- fitness
    @staticmethod
    def quality(metrics: Mapping[str, Any]) -> float:
        """FIXED label-free fitness: 0.5*kNN consistency + 0.5*cluster stability."""
        def _g(k):
            v = metrics.get(k, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                return 0.0
            return v if math.isfinite(v) else 0.0
        q = 0.5 * _g("knn_view_consistency") + 0.5 * _g("clustering_stability")
        return float(min(1.0, max(0.0, q)))

    def collapse(self, metrics: Mapping[str, Any]) -> Dict[str, bool]:
        vm = float(metrics.get("var_mean", float("nan")))
        er = float(metrics.get("effective_rank_ratio", float("nan")))
        total = (not math.isfinite(vm)) or vm < self.cfg.tau_total
        dim = total or (math.isfinite(er) and er < self.cfg.tau_dimensional)
        return {"total": bool(total), "dimensional": bool(dim)}

    # ---------------------------------------------------------------- alerts
    def alerts(self, metrics: Mapping[str, Any], logger: Any = None,
               history: Optional[Mapping[str, List[float]]] = None) -> List[Dict[str, str]]:
        """Return (and send to ``logger.alert``) a list of {title,text,level}.

        ``metrics`` may carry flags ``nonfinite_grads`` (bool) and ``loss`` (float).
        ``history`` may carry lists ``loss`` and ``quality`` (oldest first).
        """
        cfg = self.cfg
        history = history or {}
        out: List[Dict[str, str]] = []

        def add(title, text, level):
            out.append({"title": title, "text": text, "level": level})

        c = self.collapse(metrics)
        if c["total"]:
            add("total_collapse", f"mean var={metrics.get('var_mean')} < {cfg.tau_total}", "ERROR")
        elif c["dimensional"]:
            add("dimensional_collapse",
                f"effective_rank_ratio={metrics.get('effective_rank_ratio'):.4f} < {cfg.tau_dimensional}", "WARN")
        if metrics.get("nonfinite_grads") or metrics.get("finite") is False:
            add("non_finite", "non-finite gradients or embeddings", "ERROR")
        red = metrics.get("mean_abs_offdiag_corr")
        if red is not None and math.isfinite(float(red)) and float(red) > cfg.redundancy_tau:
            add("excessive_redundancy", f"mean |offdiag corr|={float(red):.3f} > {cfg.redundancy_tau}", "WARN")

        losses = [float(x) for x in history.get("loss", [])]
        cur = metrics.get("loss")
        if cur is not None:
            losses = losses + [float(cur)]
        if losses:
            last = losses[-1]
            finite_prev = [x for x in losses[:-1] if math.isfinite(x)]
            if not math.isfinite(last):
                add("loss_divergence", f"loss={last}", "ERROR")
            elif finite_prev and last > cfg.divergence_factor * max(min(finite_prev), 1e-12):
                add("loss_divergence", f"loss={last:.4g} > {cfg.divergence_factor}x min={min(finite_prev):.4g}", "ERROR")
            w = losses[-(cfg.nonconv_n + 1):]
            if len(w) == cfg.nonconv_n + 1 and all(math.isfinite(x) for x in w):
                rel = (w[0] - min(w[1:])) / max(abs(w[0]), 1e-12)
                if rel < cfg.nonconv_rel_delta:
                    add("non_convergence", f"loss improved {rel:.2e} (rel) over last {cfg.nonconv_n}", "WARN")

        qs = [float(x) for x in history.get("quality", [])]
        if "quality" in metrics:
            qs = qs + [float(metrics["quality"])]
        w = qs[-(cfg.saturation_n + 1):]
        if len(w) == cfg.saturation_n + 1 and max(w[1:]) - w[0] <= cfg.saturation_delta:
            add("quality_saturation", f"quality gain <= {cfg.saturation_delta} over last {cfg.saturation_n}", "INFO")

        if logger is not None:
            for a in out:
                try:
                    logger.alert(a["title"], a["text"], a["level"])
                except Exception:  # logger must never break the search
                    pass
        return out
