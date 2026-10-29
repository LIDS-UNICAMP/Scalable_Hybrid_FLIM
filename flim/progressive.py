"""One-block, in-memory adapter over the existing SPiFiL growth protocol.

Only the supplied training samples are read. The caller owns split selection;
production callers should obtain them with :func:`flim.spifil.train_samples`.
No learned prefix or encoder filter is passed through an auxiliary initializer.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import random
import resource
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from torch import nn

from flim.spifil import (ArchSpec, LayerSpec, Learner, Trunk, Cropped,
                         GrowOneLayer, BudgetExhausted, build_layer0)


class InfeasibleCandidate(RuntimeError):
    """Deterministic rejection; the runner records this reason and a penalty."""

    def __init__(self, reason, *, details=None):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.details = details or {}

    def to_dict(self):
        return {"status": "infeasible", "reason": self.reason, **self.details}


def _fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_new_block(prefix, samples, block_spec, config, *, prefix_id=None):
    """Fit exactly one conventional 3/5 convolution on frozen prefix features.

    Returns ``(block, provenance)``; actual width can be smaller than requested.
    ``samples`` must contain training-only ``spifil.data.Sample`` records.
    A new block is trainable; the supplied prefix remains frozen and in eval.
    """
    spec = dict(block_spec)
    cfg = dict(config)
    if spec.get("op", "conv") != "conv" or spec.get("connection", "sequential") != "sequential":
        raise InfeasibleCandidate("unsupported operation or connection for SPiFiL")
    kernel = int(spec.get("kernel", 3))
    stride = int(spec.get("pool_stride", 1))
    channels = int(spec.get("channels", 16))
    activation = spec.get("activation", "relu")
    dropout = float(spec.get("dropout", 0))
    if (kernel not in (3, 5) or stride not in (1, 2) or channels < 1
            or activation not in ("relu", "relu6", "silu")
            or not 0 <= dropout < 1 or int(spec.get("groups", 1)) != 1):
        raise InfeasibleCandidate("unsupported kernel/channels/groups/pooling/activation/dropout")
    if cfg.get("spifil_in_feature", False) and cfg.get("spifil_in_image", True):
        raise InfeasibleCandidate("spifil_in_image and spifil_in_feature are exclusive")
    if cfg.get("superpixels", "slic") not in ("slic", "disf"):
        raise InfeasibleCandidate("unknown superpixel algorithm")
    samples = list(samples)
    if not samples or len({s.label for s in samples}) < 2:
        raise InfeasibleCandidate("SPiFiL Fisher initialization requires training samples from >=2 classes")
    if len({s.name for s in samples}) != len(samples):
        raise InfeasibleCandidate("duplicate SPiFiL sample IDs")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(cfg.get("image_size", 200))
    if str(device).startswith("cuda"):
        # An empty growing encoder has not allocated any CUDA tensor yet.
        # Peak-memory counters require the allocator to be initialized first.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    prefix = nn.Identity() if prefix is None else prefix
    before = _fingerprint(prefix)
    started = time.perf_counter()
    try:
        trunk = Trunk(prefix, size, device)
        data = Cropped(samples, size, use_mask=not cfg.get("impurities", False))
        layer = LayerSpec(kernel_size=kernel, out_channels=channels,
                          pool_type="max" if stride > 1 else "none",
                          pool_stride=stride, relu=False)
        learner = Learner(data, ArchSpec(layers=[layer]), color=trunk,
                          strategy=GrowOneLayer(cfg.get("budget_check", True)), device=device)
        learner.state[0] = build_layer0(
            data, trunk, samples, int(cfg.get("n_superpixels", 100)), size,
            cfg.get("imagenet_norm", False), device,
            cfg.get("spifil_in_feature", False), cfg.get("superpixels", "slic"))
        # Existing GrowOneLayer checks again after preparation. Do the same check
        # before the quadratic scoring/covariance work, using its exact N and D.
        learner.extract_layer_patches(0)
        patches = learner.state[0].patches
        n, d = len(patches), int(patches.feats.shape[1])
        if cfg.get("budget_check", True) and n <= d:
            raise BudgetExhausted(f"N={n} <= D={d}: insufficient covariance budget")
        learner.fit()
        fitted = learner.model.block(1)
        got = int(fitted.out_channels)
        if got < 1 or not all(torch.isfinite(p).all() for p in fitted.parameters()):
            raise InfeasibleCandidate("SPiFiL produced empty or non-finite filters")
        modules = OrderedDict(conv=fitted.conv)
        if spec.get("bn", False):
            bn = nn.BatchNorm2d(got).to(device)
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            modules["bn"] = bn
        modules["activation"] = {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU}[activation]()
        if fitted.pool is not None:
            modules["pool"] = fitted.pool
        if dropout:
            modules["dropout"] = nn.Dropout2d(dropout)
        block = nn.Sequential(modules).to(device)
        block.out_channels = got
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        if _fingerprint(prefix) != before:
            raise InfeasibleCandidate("SPiFiL fitting modified the trained prefix")
    except InfeasibleCandidate:
        raise
    except (RuntimeError, ValueError, IndexError, OSError) as exc:
        raise InfeasibleCandidate(f"SPiFiL fit failed: {type(exc).__name__}: {exc}") from exc
    try:
        spifil_version = version("spifil")
    except PackageNotFoundError:
        spifil_version = "editable/unpackaged"
    provenance = {
        "origin": "spifil", "requested_channels": channels, "actual_channels": got,
        "channel_repair": "allocator_actual_width" if channels != got else None,
        "sample_ids": [s.name for s in samples], "sample_labels": [int(s.label) for s in samples],
        "sample_paths": [str(s.image_path) for s in samples],
        "mask_paths": [str(s.mask_path) if s.mask_path else None for s in samples],
        "prefix_id": prefix_id or before, "prefix_state_sha256": before,
        "seed": seed, "patch_count": n, "patch_dimension": d,
        "fit_seconds": time.perf_counter() - started,
        "fit_peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "superpixel_mode": "feature" if cfg.get("spifil_in_feature", False) else "image",
        "config": cfg, "requested_spec": spec,
        "effective_spec": dict(spec, channels=got),
        "modules": {"conv": "spifil", **({"bn": "auxiliary_scale1_bias0"} if spec.get("bn", False) else {})},
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "spifil_version": spifil_version,
        "nondeterminism": "CUDA kernels may be nondeterministic" if str(device).startswith("cuda") else None,
    }
    block.initialization_provenance = provenance
    return block, provenance
