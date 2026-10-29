"""Opt-in CPU preprocessing cache for deterministic progressive data loaders.

Imports remain lazy so loky workers import this module without initializing a
training process. Workers load CPU images only, using one numerical thread.
The caller must invoke cache_loader from a protected main entry point.
"""
from __future__ import annotations

import time


class CachedDataset:
    """Map dataset retaining original sample metadata and exact CPU float pixels."""

    def __init__(self, records, original):
        self.records = records
        if hasattr(original, "samples"):
            self.samples = original.samples

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        pixels, label, sample_id = self.records[index]
        return torch.from_numpy(pixels), label, sample_id


def _load_cpu_item(dataset, index):
    """Runs in a loky worker; never moves tensors to CUDA."""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    pixels, label, sample_id = dataset[index]
    if isinstance(pixels, torch.Tensor):
        if pixels.device.type != "cpu":
            raise ValueError("preprocessing cache workers require CPU dataset tensors")
        pixels = pixels.detach().numpy()
    pixels = np.asarray(pixels)
    if pixels.dtype != np.float32:
        raise ValueError("preprocessing cache requires original float32 pixels; refusing conversion")
    return np.ascontiguousarray(pixels), label, sample_id


def cache_loader(loader, workers=24, device="cuda:0"):
    """Preload a deterministic map dataset once with loky CPU processes.

    Return ``(cached_loader, stats)``. Dataset values, IDs and order are retained,
    as are batch size, sampler, collator, generator and drop_last. The cached
    loader has num_workers=0, preserving the original iterator's Torch base-seed
    draw without persistent worker state. Preprocessing itself preserves the
    parent's Torch RNG. CUDA device strings enable optional pinned CPU memory;
    this function never creates CUDA tensors or starts a training process.

    Use only for deterministic preprocessing (the progressive loader's LAB/test
    transform); caching random augmentation would intentionally fix its outputs.
    """
    import torch
    from torch.utils.data import DataLoader, IterableDataset
    from joblib import Parallel, delayed, parallel_config

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >=1")
    if isinstance(loader.dataset, IterableDataset) or loader.batch_size is None:
        raise ValueError("cache_loader requires a map dataset and explicit batch size")
    started = time.perf_counter()
    rng = torch.get_rng_state()
    old_threads = torch.get_num_threads()
    try:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            records = Parallel(n_jobs=workers, max_nbytes=None)(
                delayed(_load_cpu_item)(loader.dataset, index)
                for index in range(len(loader.dataset)))
    finally:
        # n_jobs=1 executes locally, so restore its thread limit as well.
        if torch.get_num_threads() != old_threads:
            torch.set_num_threads(old_threads)
        torch.set_rng_state(rng)
    cached = CachedDataset(records, loader.dataset)
    result = DataLoader(cached, batch_size=loader.batch_size, sampler=loader.sampler,
                        num_workers=0, collate_fn=loader.collate_fn,
                        drop_last=loader.drop_last, generator=loader.generator,
                        pin_memory=str(device).startswith("cuda"))
    stats = {"load_seconds": time.perf_counter() - started,
             "samples": len(records), "bytes": sum(row[0].nbytes for row in records),
             "workers": workers}
    return result, stats
