"""Equivalent DBCV acceleration, without changing the experiment definition.

This opt-in helper is not imported by the baseline experiment. A runtime wrapper
may replace ``hdbscan.validity.validity_index`` with this function. It retains
the upstream distances, dimension, noise denominator, MST ties and summation
order. Unsupported diagnostic/raw-MST options use the captured serial function.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from threadpoolctl import threadpool_limits

from hdbscan.validity import (
    validity_index as _serial_validity_index,
    distances_between_points,
    density_separation,
    mst_linkage_core,
)


def fast_internal_minimum_spanning_tree(mr_distances):
    """Return upstream-identical internal nodes/edges, including its tie rules.

    In upstream's reconstruction, intersection with endpoints of edges before i
    is precisely membership ``first_seen[vertex] < i``. ``flatnonzero`` preserves
    the sorted ordering of ``intersect1d``, so the smallest candidate is identical.
    The linkage input remains immutable while reconstructed edges are modified.
    """
    single_linkage_data = mst_linkage_core(mr_distances)
    min_span_tree = single_linkage_data.copy()
    first_seen = np.full(mr_distances.shape[0], len(single_linkage_data), dtype=np.intp)
    endpoints = single_linkage_data[:, :2].astype(int)
    np.minimum.at(first_seen, endpoints.ravel(), np.repeat(np.arange(len(endpoints)), 2))
    for index, row in enumerate(min_span_tree[1:], 1):
        candidates = np.flatnonzero(np.isclose(mr_distances[int(row[1])], row[2]) &
                                    (first_seen < index))
        candidates = candidates[candidates != row[1]]
        assert len(candidates) > 0
        row[0] = candidates[0]
    vertices = np.arange(mr_distances.shape[0])[
        np.bincount(min_span_tree.T[:2].flatten().astype(np.intp)) > 1]
    if not len(vertices):
        vertices = [0]
    edge_selection = np.prod(np.isin(min_span_tree.T[:2], vertices), axis=0).astype(bool)
    edges = min_span_tree[edge_selection] if np.any(edge_selection) else min_span_tree.copy()
    return vertices, edges


def parallel_validity_index(X, labels, metric="euclidean", d=None, workers=24,
                            progress=False, **kwargs):
    """Thread-parallel default DBCV with the same scalar as contrib hdbscan.

    Jobs first process independent clusters, then independent cluster pairs.
    Native BLAS is limited to one thread during this call to prevent nested
    oversubscription. ``workers`` affects execution only. Diagnostic options
    ``verbose``, ``per_cluster_scores`` and ``mst_raw_dist`` fall back to the
    original captured function when true. Metric kwargs are passed unchanged.
    """
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if any(kwargs.get(key, False) for key in ("verbose", "per_cluster_scores", "mst_raw_dist")):
        return _serial_validity_index(X, labels, metric=metric, d=d, **kwargs)
    kwargs = dict(kwargs)
    for key in ("verbose", "per_cluster_scores", "mst_raw_dist"):
        kwargs.pop(key, None)
    start = time.perf_counter()
    max_cluster_id = labels.max() + 1
    cluster_ids = [i for i in range(max_cluster_id) if np.sum(labels == i) != 0]
    if progress:
        print(f"DBCV start samples={X.shape[0]} clusters={len(cluster_ids)} workers={workers}", flush=True)
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id), dtype=np.float64)
    cores, nodes, sparseness = {}, {}, {}

    def cluster_job(cluster_id):
        distances, core = distances_between_points(X, labels, cluster_id, metric, d,
                                                   no_coredist=False, **kwargs)
        vertices, edges = fast_internal_minimum_spanning_tree(distances)
        return cluster_id, core, vertices, edges.T[2].max()

    def pair_job(pair):
        i, j = pair
        separation = density_separation(X, labels, i, j, nodes[i], nodes[j],
                                        cores[i], cores[j], metric=metric,
                                        no_coredist=False, **kwargs)
        return i, j, separation

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for cluster_id, core, vertices, spread in executor.map(cluster_job, cluster_ids):
            cores[cluster_id], nodes[cluster_id], sparseness[cluster_id] = core, vertices, spread
        if progress:
            print(f"DBCV clusters={len(cluster_ids)} workers={workers} "
                  f"MST_seconds={time.perf_counter() - start:.3f}", flush=True)
        pairs = [(i, j) for i in cluster_ids for j in cluster_ids if j > i]
        for i, j, separation in executor.map(pair_job, pairs):
            density_sep[i, j] = density_sep[j, i] = separation
    # Keep the exact scalar expression and ascending reduction order upstream.
    result = 0
    n_samples = float(X.shape[0])
    for i in cluster_ids:
        minimum = density_sep[i].min()
        validity = (minimum - sparseness[i]) / max(minimum, sparseness[i])
        result += (np.sum(labels == i) / n_samples) * validity
    if progress:
        print(f"DBCV pairs={len(pairs)} elapsed_seconds={time.perf_counter() - start:.3f}", flush=True)
    return result
