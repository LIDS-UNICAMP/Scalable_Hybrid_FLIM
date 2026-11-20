"""Per-block initialization for the progressive MO-DEHB NAS (Fase A).

SPiFiL is the PRIMARY initializer; ``random`` exists only as a baseline twin
that differs from SPiFiL in the weight values and nothing else.

SPiFiL support gap (why ``repair_for_spifil`` exists)
-----------------------------------------------------
``flim.progressive.fit_new_block`` (reused, never modified) only fits ONE
*sequential, conventional* convolution:

* ``conv`` must be ``"conv"`` -- depthwise / pointwise / dwsep are not fittable
  (Fisher filters are dense patch projections, ``groups == 1``);
* ``kernel`` must be 3 or 5 -- 1x1 and 7x7 are rejected;
* ``residual`` must be False -- ``connection`` must be ``"sequential"``;
* ``activation`` must be relu / relu6 / silu;
* ``pool`` must be none or max (stride 2, SPiFiL's own zero-padded 3x3 max
  pool); ``avg`` is not offered by the progressive adapter;
* the actual width can be SMALLER than requested (class allocator), see
  ``provenance["effective_spec"]["channels"]``.

``norm`` (only ``batch`` maps to fit_new_block's ``bn`` flag), ``softhebb`` and
the residual wrapper are applied by the sibling builders on top of the block
returned here; they do not affect what SPiFiL can fit.

``repair_for_spifil`` maps an unsupported genome layer to the nearest
supported one (kernel 1->3, 7->5; any conv->conv; residual->False;
leaky_relu/prelu/selu->relu, gelu->silu; avg->max) and lists every change, so
the search can log/penalize the genotype-phenotype mismatch.

Width < number of classes (default path): ``fit_new_block`` would fail ("build_filters
needs at least one selected"), so ``init_block`` fits that (already repaired) layer through
``spifil_ops.fit_ops_block`` instead, whose ``AutoAllocator`` builds exactly the requested
width (``provenance["small_width_route"]``; ``effective_spec["builder"] == "spifil_ops"`` so
``random_twin_block`` builds the identical module).

The random path builds the *same* topology (``spifil.SpifilConvBlock`` conv +
pool, wrapped exactly like ``fit_new_block`` wraps it) and He-initializes it
(``src.models.models.init_weights_he``) under a seeded, forked RNG.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn

SPIFIL_KERNELS = (3, 5)
SPIFIL_ACTIVATIONS = ("relu", "relu6", "silu")
SPIFIL_POOLS = ("none", "max")
_KERNEL_REPAIR = {1: 3, 3: 3, 5: 5, 7: 5}
_ACT_REPAIR = {"leaky_relu": "relu", "prelu": "relu", "selu": "relu", "gelu": "silu",
               "relu": "relu", "relu6": "relu6", "silu": "silu"}
_ACT = {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU}


# ─── SPiFiL support / repair ──────────────────────────────────────────────────

def spifil_supported(layer_spec: dict, *, spifil_ops: bool = False) -> bool:
    """True when ``fit_new_block`` can fit this genome layer as-is.

    ``spifil_ops=True`` (opt-in, ``config["spifil_ops"]``): the extended operator set
    of ``ga_search.modehb.spifil_ops`` instead.
    """
    if spifil_ops:
        from ga_search.modehb.spifil_ops import ops_supported
        return ops_supported(layer_spec)
    return (layer_spec.get("conv", "conv") == "conv"
            and int(layer_spec.get("kernel", 3)) in SPIFIL_KERNELS
            and not bool(layer_spec.get("residual", False))
            and layer_spec.get("activation", "relu") in SPIFIL_ACTIVATIONS
            and layer_spec.get("pool", "none") in SPIFIL_POOLS)


def repair_for_spifil(layer_spec: dict, *, spifil_ops: bool = False) -> dict:
    """Return a copy of the genome layer mapped to the nearest SPiFiL-fittable one.

    The copy carries ``"spifil_repairs"``: a list of ``"field:old->new"``
    strings (empty when the layer was already supported). See module docstring.
    ``spifil_ops=True`` keeps the extended operators (``spifil_ops.ops_repair``).
    """
    if spifil_ops:
        from ga_search.modehb.spifil_ops import ops_repair
        return ops_repair(layer_spec)
    out = dict(layer_spec)
    repairs = []

    def _set(key, new):
        old = out.get(key)
        if old != new:
            repairs.append(f"{key}:{old}->{new}")
            out[key] = new

    k = int(out.get("kernel", 3))
    _set("kernel", _KERNEL_REPAIR.get(k, 3 if k < 4 else 5))
    _set("conv", "conv")
    _set("residual", False)
    _set("activation", _ACT_REPAIR.get(out.get("activation", "relu"), "relu"))
    if out.get("pool", "none") not in SPIFIL_POOLS:
        _set("pool", "max")
    out["spifil_repairs"] = repairs
    return out


def to_block_spec(layer_spec: dict) -> dict:
    """Genome layer (must be SPiFiL-supported) -> ``fit_new_block`` block_spec."""
    return {
        "op": "conv", "connection": "sequential",
        "kernel": int(layer_spec.get("kernel", 3)),
        "channels": int(layer_spec.get("channels", 16)),
        "activation": layer_spec.get("activation", "relu"),
        "pool_stride": 2 if layer_spec.get("pool", "none") == "max" else 1,
        "bn": layer_spec.get("norm", "none") == "batch",
        "groups": 1, "dropout": float(layer_spec.get("dropout", 0.0)),
    }


# ─── Random twin ──────────────────────────────────────────────────────────────

def _infer_in_channels(prefix, config) -> int:
    if config and config.get("in_channels"):
        return int(config["in_channels"])
    if prefix is None:
        return 3
    if getattr(prefix, "out_channels", None):
        return int(prefix.out_channels)
    convs = [m for m in prefix.modules() if isinstance(m, nn.Conv2d)]
    return int(convs[-1].out_channels) if convs else 3


def random_twin_block(effective_spec: dict, seed: int, *, in_channels: int | None = None,
                      device=None) -> nn.Module:
    """He-initialized block with EXACTLY the SPiFiL topology of ``effective_spec``.

    ``effective_spec`` is a ``fit_new_block`` block_spec (e.g.
    ``provenance["effective_spec"]``). The conv/pool come from
    ``spifil.SpifilConvBlock`` (same padding, bias, pool geometry) and are
    wrapped in the same ``conv[/bn]/activation[/pool][/dropout]`` Sequential
    as ``fit_new_block``, so state_dict keys and shapes match one-to-one.
    An ``effective_spec`` produced by ``spifil_ops`` (``builder == "spifil_ops"``)
    is delegated to ``spifil_ops.random_twin_ops`` (same module structure).
    """
    if effective_spec.get("builder") == "spifil_ops":
        from ga_search.modehb.spifil_ops import random_twin_ops
        return random_twin_ops(effective_spec, seed, in_channels=in_channels, device=device)
    from spifil import LayerSpec, SpifilConvBlock
    from src.models.models import init_weights_he

    spec = dict(effective_spec)
    in_ch = int(in_channels or spec.get("in_channels") or 3)
    out_ch = int(spec.get("channels", 16))
    stride = int(spec.get("pool_stride", 1))
    activation = spec.get("activation", "relu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        core = SpifilConvBlock(LayerSpec(kernel_size=int(spec.get("kernel", 3)), out_channels=out_ch,
                                         pool_type="max" if stride > 1 else "none",
                                         pool_stride=stride, relu=False), in_ch)
        init_weights_he(core)
    modules = OrderedDict(conv=core.conv)
    if spec.get("bn", False):
        bn = nn.BatchNorm2d(out_ch)
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)
        modules["bn"] = bn
    modules["activation"] = _ACT[activation]()
    if core.pool is not None:
        modules["pool"] = core.pool
    if float(spec.get("dropout", 0) or 0):
        modules["dropout"] = nn.Dropout2d(float(spec["dropout"]))
    block = nn.Sequential(modules)
    if device is not None:
        block = block.to(device)
    block.out_channels = out_ch
    return block


# ─── Entry point ──────────────────────────────────────────────────────────────

def init_block(genome: dict, layer_idx: int, prefix, samples, config: dict,
               prefix_id=None):
    """Initialize layer ``layer_idx`` of ``genome`` on top of the frozen ``prefix``.

    Returns ``(block, provenance)``. Unsupported layers are first passed through
    ``repair_for_spifil`` (both modes, so random and SPiFiL share a topology);
    ``provenance["spifil_repairs"]`` lists the changes. The layer seed is
    ``genome["seed"] + layer_idx`` for both modes.
    Raises ``flim.progressive.InfeasibleCandidate`` from SPiFiL unchanged.
    ``config["spifil_ops"]`` (default False) switches to ``_init_block_ops``.
    """
    if genome.get("init", "spifil") == "spifil":
        from ga_search.modehb.spifil_oom import install as _install_oom_fallback
        _install_oom_fallback()      # Fisher scorer: blocked recompute only on CUDA OOM
    if (config or {}).get("spifil_ops", False):
        return _init_block_ops(genome, layer_idx, prefix, samples, config, prefix_id)
    layer = repair_for_spifil(genome["layers"][layer_idx])
    repairs = layer.pop("spifil_repairs")
    block_spec = to_block_spec(layer)
    seed = int(genome.get("seed", 0)) + int(layer_idx)
    cfg = dict(config or {})
    cfg["seed"] = seed
    mode = genome.get("init", "spifil")
    n_classes = len({getattr(s, "label", None) for s in (samples or [])})
    if mode == "spifil" and 0 < block_spec["channels"] < n_classes:
        # fit_new_block's UniformAllocator would give 0 filters per class ("build_filters needs
        # at least one selected"). Same (already repaired) layer through spifil_ops, whose
        # AutoAllocator builds EXACTLY the requested width (APPROXIMATION of the allocator).
        from ga_search.modehb import spifil_ops as so
        ops_spec = so.to_ops_spec(layer, _infer_in_channels(prefix, cfg))
        block, prov = so.fit_ops_block(prefix, samples, ops_spec, cfg, prefix_id=prefix_id)
        prov = dict(prov, small_width_route="spifil_ops_exact_alloc")
    elif mode == "spifil":
        from flim.progressive import fit_new_block
        block, prov = fit_new_block(prefix, samples, block_spec, cfg, prefix_id=prefix_id)
        prov = dict(prov)
    elif mode == "random":
        in_ch = _infer_in_channels(prefix, cfg)
        device = cfg.get("device")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        block = random_twin_block(block_spec, seed, in_channels=in_ch, device=device)
        prov = {"origin": "random_he", "requested_channels": block_spec["channels"],
                "actual_channels": block_spec["channels"], "seed": seed,
                "prefix_id": prefix_id, "requested_spec": block_spec,
                "effective_spec": dict(block_spec), "in_channels": in_ch,
                "modules": {"conv": "kaiming_normal_fan_out_relu"}}
    else:
        raise ValueError(f"unknown init {mode!r}; expected 'spifil' or 'random'")
    prov.update(layer_idx=int(layer_idx), init=mode, spifil_repairs=repairs,
                spifil_repaired=bool(repairs))
    block.initialization_provenance = prov
    return block, prov


def _init_block_ops(genome, layer_idx, prefix, samples, config, prefix_id):
    """Opt-in branch: extended operators via ``ga_search.modehb.spifil_ops``.

    SPiFiL -> ``spifil_ops.fit_ops_block``; random -> ``spifil_ops.random_twin_ops``
    on the requested spec (same module structure as the SPiFiL block).
    """
    from ga_search.modehb import spifil_ops as so
    layer = repair_for_spifil(genome["layers"][layer_idx], spifil_ops=True)
    repairs = layer.pop("spifil_repairs")
    seed = int(genome.get("seed", 0)) + int(layer_idx)
    cfg = dict(config or {})
    cfg["seed"] = seed
    in_ch = _infer_in_channels(prefix, cfg)
    ops_spec = so.to_ops_spec(layer, in_ch)
    mode = genome.get("init", "spifil")
    if mode == "spifil":
        block, prov = so.fit_ops_block(prefix, samples, ops_spec, cfg, prefix_id=prefix_id)
        prov = dict(prov)
    elif mode == "random":
        device = cfg.get("device")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        block = so.random_twin_ops(ops_spec, seed, in_channels=in_ch, device=device)
        prov = {"origin": "random_he", "builder": so.BUILDER,
                "requested_channels": ops_spec["requested_channels"],
                "actual_channels": ops_spec["channels"], "seed": seed, "prefix_id": prefix_id,
                "requested_spec": ops_spec, "effective_spec": dict(ops_spec), "in_channels": in_ch,
                "modules": {"conv": "kaiming_normal_fan_out_relu"}}
    else:
        raise ValueError(f"unknown init {mode!r}; expected 'spifil' or 'random'")
    prov.update(layer_idx=int(layer_idx), init=mode, spifil_repairs=repairs,
                spifil_repaired=bool(repairs), spifil_ops=True)
    block.initialization_provenance = prov
    return block, prov


# ─── Diagnostics ──────────────────────────────────────────────────────────────

def _erank(mat: torch.Tensor) -> float:
    """exp(entropy of normalized singular values) of a (N, D) matrix (centered)."""
    mat = torch.nan_to_num(mat.double(), nan=0.0, posinf=0.0, neginf=0.0)   # diagnostic: never raise
    mat = mat - mat.mean(0, keepdim=True)
    s = torch.linalg.svdvals(mat)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _pair_cos(x: torch.Tensor) -> torch.Tensor:
    """Off-diagonal pairwise cosines of the rows of ``x``."""
    x = nn.functional.normalize(x, dim=1, eps=1e-12)
    c = x @ x.T
    iu = torch.triu_indices(c.shape[0], c.shape[0], offset=1)
    return c[iu[0], iu[1]]


@torch.no_grad()
def kernel_diagnostics(weight: torch.Tensor, feats: torch.Tensor | None = None,
                       *, max_rows: int = 8192) -> dict:
    """Filter-bank / activation diagnostics as plain floats (W&B-ready).

    weight: (C_out, C_in, k, k). feats: optional (B, C_out, H, W) activations.
    Keys:
      filter_diversity      mean pairwise cosine distance (1 - cos) of filters
      filter_redundancy     fraction of filter pairs with |cos| > 0.9
      weight_effective_rank exp-entropy erank of the flattened filter bank
      featmap_similarity    mean |cos| between channel activation maps
      pattern_coverage      fraction of channels that are the (std-normalized)
                            arg-max responder at >= 1 position (specialization)
      dead_fraction         fraction of channels with ~zero activation variance
      activation_effective_rank  erank of activations (positions x channels)
    Feature keys are NaN when ``feats`` is None; single-filter banks give NaN
    pairwise metrics.
    """
    nan = float("nan")
    w = weight.detach().float().cpu().reshape(weight.shape[0], -1)
    out = {"n_filters": float(w.shape[0])}
    if w.shape[0] > 1:
        cos = _pair_cos(w)
        out["filter_diversity"] = float((1 - cos).mean())
        out["filter_redundancy"] = float((cos.abs() > 0.9).float().mean())
    else:
        out["filter_diversity"] = out["filter_redundancy"] = nan
    out["weight_effective_rank"] = _erank(w)
    out.update(featmap_similarity=nan, pattern_coverage=nan, dead_fraction=nan,
               activation_effective_rank=nan)
    if feats is None:
        return out
    f = feats.detach().float().cpu()
    c = f.shape[1]
    rows = f.permute(0, 2, 3, 1).reshape(-1, c)          # (positions, C)
    if rows.shape[0] > max_rows:                          # deterministic stride subsample
        rows = rows[:: math.ceil(rows.shape[0] / max_rows)]
    std = rows.std(0)
    dead = std < 1e-8
    out["dead_fraction"] = float(dead.float().mean())
    if c > 1:
        out["featmap_similarity"] = float(_pair_cos(rows.T).abs().mean())
    z = (rows - rows.mean(0)) / std.clamp_min(1e-8)
    z[:, dead] = -math.inf
    if not bool(dead.all()):
        winners = torch.unique(z.argmax(1))
        out["pattern_coverage"] = float(winners.numel() / c)
    else:
        out["pattern_coverage"] = 0.0
    out["activation_effective_rank"] = _erank(rows)
    return out
