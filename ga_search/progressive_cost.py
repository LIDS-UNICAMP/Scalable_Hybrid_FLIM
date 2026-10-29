"""Reproducible inference cost measurements for progressive encoder + current P_i.

MACs count only Conv2d and Linear multiply-accumulates (one kernel term is one
MAC). Pooling, activation, normalization, interpolation and tensor movement are
excluded. Decoder and historical embedding heads are excluded from inference.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import torch
from torch import nn


def profile_model(model, config, checkpoint=None):
    """Measure an independent inference copy without mutating trained model state."""
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    batch_size = int(config.get("latency_batch_size", 1))
    warmup = int(config.get("latency_warmup", 2))
    repeats = int(config.get("latency_repeats", 5))
    if batch_size < 1 or warmup < 0 or repeats < 1:
        raise ValueError("latency batch/repeats must be positive and warmup nonnegative")
    inference = model.inference_model().to(device=device, dtype=torch.float32).eval()
    image_size = int(config.get("image_size", 200))
    x = torch.zeros(batch_size, int(config.get("in_channels", 3)), image_size, image_size,
                    device=device, dtype=torch.float32)
    macs = 0

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel_terms = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += output.numel() * kernel_terms
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    handles = [module.register_forward_hook(count) for module in inference.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            embedding = inference(x)
    finally:
        for handle in handles:
            handle.remove()
    if embedding.shape != (batch_size, 1280):
        raise ValueError(f"inference embedding has unexpected shape {tuple(embedding.shape)}")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings = []
    cuda_baseline = None
    if device.type == "cuda":
        cuda_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            inference(x)
        synchronize()
        for _ in range(repeats):
            start = time.perf_counter()
            inference(x)
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    latency = torch.tensor(timings, dtype=torch.float64)
    result = {
        "parameters_total": sum(p.numel() for p in parameters),
        "parameters_trainable": sum(p.numel() for p in parameters if p.requires_grad),
        "inference_parameters": sum(p.numel() for p in inference.parameters()),
        "inference_macs": int(macs // batch_size),
        "inference_macs_per_batch": int(macs),
        "mac_methodology": "Conv2d/Linear kernel terms; 1 multiply-accumulate = 1 MAC; biases and other operations excluded",
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_std": float(latency.std(unbiased=False)),
        "latency_ms_samples": timings,
        "latency_batch_size": batch_size,
        "latency_warmup": warmup,
        "latency_repeats": repeats,
        "precision": "float32",
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or platform.machine()),
        "cpu_threads": torch.get_num_threads(),
        "embedding_dim": 1280,
        "checkpoint_bytes": Path(checkpoint).stat().st_size if checkpoint is not None else None,
        "cuda_memory_allocated_baseline_bytes": cuda_baseline,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    return result
