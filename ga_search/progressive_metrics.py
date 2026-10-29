"""Validation-only clustering and embedding artifacts for progressive NAS.

Hoyer uses raw embeddings. HDBSCAN/DBI optionally use standardized embeddings.
The optional ``hdbscan`` package supplies DBCV; sklearn supplies HDBSCAN when
that package is absent. No cluster count or classification labels are used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def hoyer_sparsity(embeddings) -> np.ndarray:
    """Per-sample Hoyer; zero vectors/nonfinite values are explicitly invalid."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) == 0:
        raise ValueError("Hoyer requires a nonempty [samples, dimensions>=2] array")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite embeddings")
    scale = np.max(np.abs(x), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero embedding: Hoyer is undefined")
    # Scale invariance avoids overflow while preserving exact zero handling.
    x = np.abs(x) / scale[:, None]
    ratio = x.sum(axis=1) / np.sqrt(np.square(x).sum(axis=1))
    root = np.sqrt(x.shape[1])
    return np.clip((root - ratio) / (root - 1), 0.0, 1.0)


def scalar_fitness(dbi: float, hoyer: float, sparsity_lambda: float = 0.004) -> float:
    """Minimized J; increasing sparsity decreases this objective."""
    if not (np.isfinite(dbi) and dbi >= 0 and np.isfinite(hoyer) and 0 <= hoyer <= 1):
        raise ValueError("invalid DBI or Hoyer")
    if not np.isfinite(sparsity_lambda) or not 0 <= sparsity_lambda <= 1:
        raise ValueError("sparsity_lambda must be in [0,1]")
    return float((1 - sparsity_lambda) * (dbi / (1 + dbi)) + sparsity_lambda * (1 - hoyer))


def evaluate_embeddings(embeddings, config: dict) -> dict:
    """Return finite J or the configured deterministic penalty and a reason.

    Flat keys override the optional ``clustering`` mapping. Keys are
    ``sparsity_lambda`` (0.004), ``min_coverage`` (0.8), ``penalty`` (1e12),
    ``min_cluster_size`` (5), ``min_samples`` (None), ``metric`` (euclidean),
    ``cluster_selection_method`` (eom), and ``standardize`` (False).
    ``min_samples`` includes the point itself (sklearn convention), defaults to
    min_cluster_size, and must be >=2 for exact backend parameter translation.
    DBCV failure is complementary information and never replaces primary J.
    """
    cfg = {**config.get("clustering", {}), **config}
    penalty = float(cfg.get("penalty", 1e12))
    if not np.isfinite(penalty) or penalty <= 1:
        raise ValueError("penalty must be finite and greater than one")
    row = dict(status="invalid", reason="", dbi=None, dbi_norm=None,
               hoyer_mean=None, hoyer_std=None, coverage=0.0, n_clusters=0,
               n_samples=0, n_clustered=0, dbcv=None, dbcv_status="unavailable",
               dbcv_reason="", J=penalty, clustering_backend=None,
               min_samples_including_self=None, min_samples_backend=None)
    try:
        weight = float(cfg.get("sparsity_lambda", 0.004))
        coverage_min = float(cfg.get("min_coverage", 0.8))
        if not 0 <= coverage_min <= 1:
            raise ValueError("min_coverage must be in [0,1]")
        scalar_fitness(1.0, 0.5, weight)
        x = np.asarray(embeddings, dtype=np.float64)
        h = hoyer_sparsity(x)
        row.update(n_samples=len(x), hoyer_mean=float(h.mean()), hoyer_std=float(h.std()))
        if len(x) < 3:
            raise ValueError("insufficient samples for clustering")
        z = StandardScaler().fit_transform(x) if cfg.get("standardize", False) else x
        kwargs = {key: cfg[key] for key in ("min_cluster_size", "min_samples", "metric",
                  "cluster_selection_method", "cluster_selection_epsilon", "alpha") if key in cfg}
        kwargs.setdefault("min_cluster_size", 5)
        kwargs.setdefault("metric", "euclidean")
        min_samples = kwargs.get("min_samples")
        min_samples = kwargs["min_cluster_size"] if min_samples is None else min_samples
        if not isinstance(min_samples, (int, np.integer)) or min_samples < 2:
            raise ValueError("min_samples must be an integer >=2, including self")
        kwargs["min_samples"] = int(min_samples)
        row["min_samples_including_self"] = int(min_samples)
        if kwargs["metric"] == "precomputed":
            raise ValueError("precomputed distances are not embedding coordinates")
        try:
            from hdbscan import HDBSCAN
            row["clustering_backend"] = "hdbscan"
            # sklearn includes the sample itself in min_samples; contrib does not.
            kwargs["min_samples"] -= 1
        except ImportError:
            from sklearn.cluster import HDBSCAN
            row["clustering_backend"] = "sklearn.cluster.HDBSCAN"
            kwargs["copy"] = True
        row["min_samples_backend"] = kwargs["min_samples"]
        labels = HDBSCAN(**kwargs).fit_predict(z)
        keep = labels >= 0
        k = len(np.unique(labels[keep]))
        count = int(keep.sum())
        row.update(n_clusters=k, n_clustered=count, coverage=count / len(x))
        if k < 2 or count <= k:
            raise ValueError("fewer than two valid clusters or insufficient clustered samples")
        dbi = float(davies_bouldin_score(z[keep], labels[keep]))
        if not np.isfinite(dbi) or dbi < 0:
            raise ValueError("DBI is nonfinite or invalid")
        row.update(dbi=dbi, dbi_norm=dbi / (1 + dbi))
        try:
            from hdbscan.validity import validity_index
            dbcv = float(validity_index(z, labels, metric=kwargs["metric"]))
            if not np.isfinite(dbcv):
                raise ValueError("nonfinite DBCV")
            row.update(dbcv=dbcv, dbcv_status="ok")
        except ImportError:
            row["dbcv_reason"] = "optional hdbscan package is not installed"
        except Exception as exc:
            row.update(dbcv_status="invalid", dbcv_reason=str(exc))
        if row["coverage"] < coverage_min:
            raise ValueError(f"coverage {row['coverage']:.6f} below {coverage_min:.6f}")
        row.update(status="ok", J=scalar_fitness(dbi, row["hoyer_mean"], weight))
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        row["reason"] = str(exc)
    return row


def extract_embeddings(model, loader, output_path=None, checkpoint_id="", device=None):
    """Extract [N,1280] via model.embed with gradients disabled.

    Each loader batch is ``(images, ids)`` or ``(images, labels, ids)``; labels
    are ignored. IDs must be a sequence matching the batch. Returns
    ``(embeddings_numpy, ids_numpy)``. NPZ stores ``embeddings``, ``ids`` and
    scalar ``checkpoint_id``. Every module's original train/eval mode is restored.
    """
    modes = [(module, module.training) for module in model.modules()]
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    vectors, names = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                    raise ValueError("loader must yield (images, ids) or (images, labels, ids)")
                images, ids = batch[0], batch[-1]
                if isinstance(ids, str) or len(ids) != len(images):
                    raise ValueError("sample IDs must match batch length")
                embedding = model.embed(images.to(device))
                if embedding.ndim != 2 or embedding.shape != (len(images), 1280):
                    raise ValueError("model.embed must return [batch,1280]")
                if not torch.isfinite(embedding).all():
                    raise ValueError("nonfinite extracted embeddings")
                vectors.append(embedding.detach().cpu().numpy())
                names.extend(str(value) for value in ids)
    finally:
        for module, training in modes:
            module.training = training
    result = np.concatenate(vectors) if vectors else np.empty((0, 1280), dtype=np.float32)
    ids = np.asarray(names, dtype=str)
    if len(set(names)) != len(names):
        raise ValueError("duplicate sample IDs in embedding extraction")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embeddings=result, ids=ids, checkpoint_id=str(checkpoint_id))
    return result, ids
