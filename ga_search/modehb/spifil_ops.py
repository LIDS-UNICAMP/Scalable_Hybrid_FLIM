"""SPiFiL initialization for the operators ``flim.progressive.fit_new_block`` rejects.

Opt-in (``config["spifil_ops"] = True``, routed by ``init.init_block``). The SPiFiL
library (``spifil``) and the repo adapters ``flim/spifil.py`` / ``flim/progressive.py``
are imported and composed here, NEVER modified.

What the library does (verified against /dados/home/moliveira/SPiFiL/src/spifil)
-------------------------------------------------------------------------------
* ``patches.extract_patches``: for each superpixel seed, the channel-major
  ``C * span**2`` vector of the feature map around it (``span = 2*(k//2)+1``,
  zero padding) -- any odd ``k >= 1`` works, so 1x1 and 7x7 are native.
* ``scoring.FisherScorer`` + ``metrics.Mahalanobis`` (Ledoit-Wolf shrinkage):
  score = mean distance to other classes / mean distance to own class; then
  ``per_class_ranks``.
* ``allocation.UniformAllocator``: ``out_channels // n_classes`` filters per class
  (so widths can shrink); ``selection.DiversitySelector``: greedy rank/diversity pick.
* ``nn.builder.build_filters``: z-score the selected patches (population std +
  ``stdev_factor``), unit-normalize, fold into ``weight``/``bias``.
* Budget: ``flim.spifil.GrowOneLayer`` raises ``BudgetExhausted`` when N (seeds) <= D
  (patch dimension); we pre-check with the same N/D before scoring, like
  ``fit_new_block`` does, and convert it into ``InfeasibleCandidate``.

Per operator (``provenance["spifil_variant"]`` lists the parts used):
* ``conv`` k in {1,3,5,7} and ``pointwise`` (= dense 1x1): FAITHFUL -- the library
  ``Learner`` + ``GrowOneLayer`` exactly as ``fit_new_block`` uses them, only the
  kernel-size validation is relaxed (the library accepts any odd kernel).
* ``depthwise`` (channel multiplier m): APPROXIMATION. For each input channel the
  library scores/ranks/selects on single-channel k x k patches (same seeds), but
  (a) ``ExactBudgetAllocator`` gives exactly m filters per channel (Uniform would
  give 0 when m < n_classes) and (b) the z-score statistics of ``build_filters``
  are taken over ALL the channel's seed patches, not only the m selected ones
  (with m=1 the library's statistics make the filter identically zero).
  A constant input channel (e.g. a dead ReLU map) still yields a zero filter
  (zero z-score), which is reported, not rejected.
* ``dwsep``: APPROXIMATION = depthwise above (m=1) followed by a FAITHFUL dense
  1x1 SPiFiL fitted on the depthwise output (linear, no activation in between,
  like ``ops_norm_act.make_conv("dwsep")``), same seeds (no pooling in between).
* activations (all of ``ops_norm_act.ACTIVATIONS``), ``max``/``avg`` pooling and
  norms (``make_norm``): applied AFTER the SPiFiL conv; the conv itself is fitted
  linear (``relu=False``), exactly what ``fit_new_block`` already does for
  relu6/silu. The pool geometry is the library's (``SpifilConvBlock._make_pool``,
  3x3 stride 2; avg with ``count_include_pad=False``). SPiFiL's internal feature
  computation for the NEXT layer DOES use the real activation/norm/skip/pool:
  the caller appends this whole block to the prefix and the next fit encodes
  layer 0 through it (``flim.spifil.Trunk`` / ``encoder_features``). The norm
  itself is not learned by SPiFiL (BN: scale 1 / bias 0; ZCA: fitted later by the
  runner) -- marked ``norm:post_hoc``.
* ``residual``: ``y = pool(dropout(act(norm(conv(x)))) + skip(x))`` (same order as
  ``ops_norm_act.ConvBlock``). Identity skip when in == effective out channels,
  else a dense 1x1 SPiFiL projection fitted on the block input with
  ``ExactBudgetAllocator`` (exact width) -- APPROXIMATION of the allocator only.

* ``inception`` (``ops_norm_act.InceptionConv``, branches 1x1/3x3/5x5/maxpool->1x1):
  each conv branch is a FAITHFUL dense SPiFiL fit (``fit_dense``) of its own kernel size
  and channel share on the block input; the pool branch's 1x1 is a faithful dense 1x1
  fit on the max-pooled (3x3, stride 1, pad 1 -- same grid, same seeds) input features.
  Branches are fitted independently (no cross-branch diversity), so each branch may shrink
  to its allocator width; the effective per-branch widths go to
  ``effective_spec["inception_branches"]`` (the twin rebuilds exactly that).
  ``spifil_variant`` contains ``"inception_per_branch"``.
* Width < number of classes (``UniformAllocator`` would give 0 per class and
  ``build_filters`` raises "needs at least one selected"): every dense fit uses
  ``AutoAllocator`` = library ``UniformAllocator`` when ``out >= n_classes`` (identical
  result), else ``ExactBudgetAllocator`` (exactly ``out`` filters, one per class for the
  classes with the best Fisher score, deterministic). APPROXIMATION of the allocator only;
  recorded as ``allocator:exact_small_width`` in ``spifil_variant`` / ``stages``.
* norms ``batch_zca`` / ``batch_dbn`` (BN then ZCA/DBN) are post-hoc like the other norms.

Random twins: ``random_twin_ops(effective_spec, ...)`` builds the identical module
(same class, names, shapes) and He-initializes it under a forked, seeded RNG.
"""
from __future__ import annotations

import contextlib
import io
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ga_search.modehb.ops_norm_act import ACTIVATIONS, NORMS, InceptionConv, make_act, make_norm

SPIFIL_OPS_KERNELS = (1, 3, 5, 7)
SPIFIL_OPS_CONVS = ("conv", "pointwise", "depthwise", "dwsep", "inception")
SPIFIL_OPS_ACTIVATIONS = tuple(ACTIVATIONS)
SPIFIL_OPS_POOLS = ("none", "max", "avg")
SPIFIL_OPS_NORMS = tuple(NORMS)
BUILDER = "spifil_ops"


# ─── genome-level support / repair / spec ────────────────────────────────────

def ops_supported(layer_spec: dict) -> bool:
    conv = layer_spec.get("conv", "conv")
    return (conv in SPIFIL_OPS_CONVS
            and (conv == "inception" or int(layer_spec.get("kernel", 3)) in SPIFIL_OPS_KERNELS)
            and layer_spec.get("activation", "relu") in SPIFIL_OPS_ACTIVATIONS
            and layer_spec.get("pool", "none") in SPIFIL_OPS_POOLS
            and layer_spec.get("norm", "none") in SPIFIL_OPS_NORMS)


def ops_repair(layer_spec: dict) -> dict:
    """Copy mapped to the extended operator set, with ``spifil_repairs`` (usually [])."""
    out = dict(layer_spec)
    repairs = []

    def _set(key, new):
        if out.get(key) != new:
            repairs.append(f"{key}:{out.get(key)}->{new}")
            out[key] = new

    k = int(out.get("kernel", 3))
    if k not in SPIFIL_OPS_KERNELS:
        _set("kernel", min(SPIFIL_OPS_KERNELS, key=lambda c: (abs(c - k), c)))
    for key, allowed, default in (("conv", SPIFIL_OPS_CONVS, "conv"),
                                  ("activation", SPIFIL_OPS_ACTIVATIONS, "relu"),
                                  ("pool", SPIFIL_OPS_POOLS, "max"),
                                  ("norm", SPIFIL_OPS_NORMS, "none")):
        if out.get(key, default) not in allowed:
            _set(key, default)
    out["spifil_repairs"] = repairs
    return out


def to_ops_spec(layer_spec: dict, in_channels: int) -> dict:
    """Genome layer -> requested spifil_ops spec (``effective_spec`` after a fit)."""
    conv = layer_spec.get("conv", "conv")
    cin = int(in_channels)
    ch = int(layer_spec.get("channels", 16))
    k = 1 if conv == "pointwise" else int(layer_spec.get("kernel", 3))
    mult = max(1, ch // cin) if conv == "depthwise" else None
    out_ch = cin * mult if conv == "depthwise" else ch
    residual = bool(layer_spec.get("residual", False))
    return {
        "builder": BUILDER, "norm_in_block": True,
        "conv": conv, "kernel": k, "in_channels": cin, "channels": out_ch,
        "requested_channels": ch, "dw_multiplier": mult,
        "activation": layer_spec.get("activation", "relu"),
        "norm": layer_spec.get("norm", "none"),
        "pool": layer_spec.get("pool", "none") or "none",
        "pool_stride": 1 if (layer_spec.get("pool", "none") or "none") == "none" else 2,
        "residual": residual,
        "skip": (("identity" if cin == out_ch else "project") if residual else None),
        "dropout": float(layer_spec.get("dropout", 0.0) or 0.0),
    }


def _as_layer(spec: dict) -> dict:
    """Ops spec -> genome-layer dict (inverse of ``to_ops_spec`` up to in_channels)."""
    return {"conv": spec["conv"], "kernel": spec["kernel"],
            "channels": spec.get("requested_channels", spec["channels"]),
            "activation": spec["activation"], "norm": spec["norm"], "pool": spec["pool"],
            "residual": spec["residual"], "dropout": spec.get("dropout", 0.0)}


# ─── the block (shared by SPiFiL and the random twin) ────────────────────────

def _library_pool(kind: str):
    """The SPiFiL library's own pool geometry (3x3, stride 2, pad 1)."""
    if kind == "none":
        return None
    from spifil import LayerSpec, SpifilConvBlock
    return SpifilConvBlock._make_pool(LayerSpec(pool_type=kind, pool_size=3, pool_stride=2))


def _dense(cin, cout, k):
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=True)


class SpifilOpsBlock(nn.Module):
    """conv -> norm -> act [-> dropout] (+ skip(x)) -> pool, built from an ops spec.

    ``conv`` is a Conv2d (conv / pointwise / depthwise with ``groups=cin``) or a
    Sequential(depthwise, pointwise) for dwsep. All convs keep a bias (SPiFiL folds
    the z-score mean into it).
    """

    def __init__(self, spec: dict, in_channels: int | None = None):
        super().__init__()
        self.spec = dict(spec)
        cin = int(in_channels or spec["in_channels"])
        cout, k, kind = int(spec["channels"]), int(spec["kernel"]), spec["conv"]
        if kind in ("conv", "pointwise"):
            self.conv = _dense(cin, cout, k)
        elif kind == "depthwise":
            if cout % cin:
                raise ValueError(f"depthwise needs channels % in_channels == 0, got {cout}/{cin}")
            self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, groups=cin, bias=True)
        elif kind == "inception":
            self.conv = InceptionConv(cin, cout, bias=True, branches=spec.get("inception_branches"))
        elif kind == "dwsep":
            self.conv = nn.Sequential()
            self.conv.add_module("depthwise", nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=True))
            self.conv.add_module("pointwise", _dense(cin, cout, 1))
        else:
            raise ValueError(f"unknown conv {kind!r}")
        self.norm = make_norm(spec.get("norm", "none"), cout)
        self.activation = make_act(spec.get("activation", "relu"))
        p = float(spec.get("dropout", 0.0) or 0.0)
        self.dropout = nn.Dropout2d(p) if p > 0 else nn.Identity()
        skip = spec.get("skip")
        if skip == "identity":
            if cin != cout:
                raise ValueError("identity skip needs in_channels == channels")
            self.skip = nn.Identity()
        elif skip == "project":
            self.skip = _dense(cin, cout, 1)
        else:
            self.skip = None
        if spec.get("pool_padding") is not None and spec.get("pool", "none") == "max":
            # external FLIM encoders (external_seed.py): MaxPool2d(size, stride) WITHOUT padding
            self.pool = nn.MaxPool2d(int(spec.get("pool_size", 3)), int(spec.get("pool_stride", 2)),
                                     padding=int(spec["pool_padding"]))
        else:
            self.pool = _library_pool(spec.get("pool", "none")) or nn.Identity()
        self.out_channels = cout

    def forward(self, x):
        y = self.dropout(self.activation(self.norm(self.conv(x))))
        if self.skip is not None:
            y = y + self.skip(x)
        return self.pool(y)


def random_twin_ops(effective_spec: dict, seed: int, *, in_channels: int | None = None,
                    device=None) -> SpifilOpsBlock:
    """He-initialized block with EXACTLY the module structure of ``effective_spec``."""
    from src.models.models import init_weights_he
    spec = dict(effective_spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        block = SpifilOpsBlock(spec, in_channels or spec.get("in_channels"))
        init_weights_he(block)
    return block if device is None else block.to(device)


# ─── SPiFiL fitting pieces (library components, composed) ────────────────────

class _Bands:
    """Stand-in ``ColorTransform``: the Learner only reads ``out_channels`` when
    layer 0 is injected (same role as ``flim.spifil.Trunk``)."""

    def __init__(self, n):
        self.out_channels = int(n)

    def __call__(self, image):
        raise NotImplementedError("features come from the injected state[0]")


class ExactBudgetAllocator:
    """APPROXIMATION of ``spifil.UniformAllocator``: exactly ``out_channels`` filters.

    ``base = out // n_classes`` per class (identical to Uniform when out divides
    evenly); the remainder goes to the classes with the highest best Fisher score
    (ties -> lower class id). Used only for depthwise channels and the residual
    projection, where the width is fixed by the topology.
    """

    def __call__(self, layer, spec, patches, scores):
        classes = sorted(torch.unique(patches.labels).tolist())
        if not classes:
            return {}
        base, rem = divmod(int(spec.out_channels), len(classes))
        best = {c: float(scores[patches.labels == c].max()) if scores.numel() else 0.0
                for c in classes}
        bonus = set(sorted(classes, key=lambda c: (-best[c], c))[:rem])
        return {c: base + (1 if c in bonus else 0) for c in classes}


class AutoAllocator:
    """Library ``UniformAllocator`` unless ``out_channels < n_classes`` (it would allocate 0
    filters and ``build_filters`` would raise); then ``ExactBudgetAllocator`` (exact width).
    ``used_exact`` records which branch was taken (APPROXIMATION flag)."""

    def __init__(self):
        self.used_exact = False

    def __call__(self, layer, spec, patches, scores):
        from spifil import UniformAllocator
        n_cls = len(torch.unique(patches.labels))
        if 0 < int(spec.out_channels) < n_cls:
            self.used_exact = True
            return ExactBudgetAllocator()(layer, spec, patches, scores)
        return UniformAllocator()(layer, spec, patches, scores)


class _SkipEncode:
    """Callback: stop ``Learner._fit_layer`` right after the filters are installed
    (``after_model_build``), via the library's own ``CancelLayerException``.

    ``fit_dense`` reads only ``model.block(1).conv`` / ``selected`` / ``state[0]``; the
    skipped steps 5-7 encode EVERY donor image at full resolution and keep it on the device
    (417 imgs x 281 ch x 224^2 fp32 = 23.5 GB for remote_sensing job 43) and, with a single
    layer, never re-score. Weights/bias are bit-identical (installed before the cut)."""

    def after_model_build(self, learner):
        from spifil.learner import CancelLayerException
        raise CancelLayerException


def _learner(features, seeds, data, in_ch, kernel, out_ch, device, budget_check, allocator=None,
             cbs=()):
    from spifil import ArchSpec, LayerSpec, Learner
    from spifil.learner import LayerState
    from flim.spifil import GrowOneLayer
    spec = LayerSpec(kernel_size=int(kernel), out_channels=int(out_ch),
                     pool_type="none", pool_stride=1, relu=False)
    learner = Learner(data, ArchSpec(layers=[spec]), color=_Bands(in_ch), allocator=allocator,
                      strategy=GrowOneLayer(budget_check), device=device, cbs=cbs)
    learner.state[0] = LayerState(features=[f.to(device) for f in features], seeds=list(seeds))
    return learner


def _budget(learner, budget_check, where):
    from flim.spifil import BudgetExhausted
    learner.extract_layer_patches(0)
    patches = learner.state[0].patches
    n, d = len(patches), int(patches.feats.shape[1])
    if budget_check and n <= d:
        raise BudgetExhausted(f"{where}: N={n} <= D={d}: insufficient covariance budget")
    return n, d


def fit_dense(features, seeds, data, in_ch, kernel, out_ch, *, device, budget_check=True,
              allocator=None):
    """FAITHFUL dense SPiFiL layer (library Learner + GrowOneLayer, as fit_new_block).

    Returns ``(weight, bias, n, d)``; ``weight`` is ``(got, in_ch, k, k)``.
    """
    if allocator is None:
        allocator = AutoAllocator()
    learner = _learner(features, seeds, data, in_ch, kernel, out_ch, device, budget_check, allocator,
                       cbs=[_SkipEncode()])
    n, d = _budget(learner, budget_check, f"dense k{kernel}")
    learner.fit()
    conv = learner.model.block(1).conv
    fit_dense.last_used_exact = bool(getattr(allocator, "used_exact", False))
    w, b = conv.weight.detach().clone(), conv.bias.detach().clone()
    fit_dense.last_single_filter_fix = False
    if int(w.shape[0]) == 1:
        # build_filters z-scores over the SELECTED patches only: with one selected patch
        # (mean == patch, std == 0) the filter AND bias are identically zero -> dead channel
        # (then ZCA/DBN on all-zero features -> NaN). Same APPROXIMATION as fit_depthwise:
        # z-score statistics over all ranked candidate patches; the selected patch is the
        # library's own pick. Only taken for out_channels == 1.
        from spifil.nn.builder import build_filters
        from spifil.types import PatchSet
        st = learner.state[0]
        cands = st.patches.select(st.ranks > 0)
        bank = build_filters(PatchSet.cat([learner.selected, cands]), learner.spec(1),
                             stdev_factor=learner.arch.stdev_factor)
        w = bank.weight[:1].detach().clone().reshape(w.shape).to(w)
        b = bank.bias[:1].detach().clone().reshape(b.shape).to(b)
        fit_dense.last_single_filter_fix = True
    return w, b, n, d


fit_dense.last_used_exact = False
fit_dense.last_single_filter_fix = False


def fit_depthwise(features, seeds, data, in_ch, kernel, mult, *, device, budget_check=True):
    """APPROXIMATION: per-channel SPiFiL on single-channel patches (see module doc).

    Library scoring/ranking/selection per channel with ``ExactBudgetAllocator``;
    ``build_filters`` statistics over all of the channel's seed patches. Returns
    ``(weight (in_ch*mult, 1, k, k), bias, n, d)`` in grouped-conv channel order.
    """
    from spifil.nn.builder import build_filters
    from spifil.selection import candidate_order
    from spifil.types import PatchSet
    ws, bs, n, d = [], [], 0, 0
    for c in range(int(in_ch)):
        feats_c = [f[c:c + 1] for f in features]
        learner = _learner(feats_c, seeds, data, 1, kernel, mult, device, budget_check,
                           ExactBudgetAllocator())
        n, d = _budget(learner, budget_check, f"depthwise channel {c} k{kernel}")
        with contextlib.redirect_stdout(io.StringIO()):
            learner.prepare()                       # scores layer 0 (Fisher/Mahalanobis)
        st, spec = learner.state[0], learner.spec(1)
        ranked = st.ranks > 0
        cands, cranks, scores = st.patches.select(ranked), st.ranks[ranked], st.scores[ranked]
        n_per_class = learner.component_for(1, "allocator")(1, spec, cands, scores)
        mask = learner.component_for(1, "selector")(cands, cranks, n_per_class)
        picks = mask.nonzero(as_tuple=True)[0]
        if picks.numel() != mult:
            from flim.progressive import InfeasibleCandidate
            raise InfeasibleCandidate(f"depthwise channel {c}: selected {picks.numel()} != {mult}")
        selected = cands.select(picks[candidate_order(cands.select(picks), cranks[picks])])
        bank = build_filters(PatchSet.cat([selected, cands]), spec,
                             stdev_factor=learner.arch.stdev_factor)
        ws.append(bank.weight[:mult])
        bs.append(bank.bias[:mult])
    return torch.cat(ws).detach(), torch.cat(bs).detach(), n, d


@torch.no_grad()
def _apply(conv_w, conv_b, features, groups, device):
    k = conv_w.shape[-1]
    return [F.conv2d(f[None].to(device), conv_w.to(device), conv_b.to(device),
                     padding=k // 2, groups=groups)[0] for f in features]


# ─── entry point ─────────────────────────────────────────────────────────────

def _variant(spec):
    parts = {"conv": "faithful", "pointwise": "faithful",
             "inception": "inception_per_branch(faithful_dense_per_branch,pool_branch_on_maxpooled)",
             "depthwise": "approx:depthwise_per_channel(exact_alloc,stats_all_seeds)",
             "dwsep": "approx:depthwise_per_channel(m=1)+faithful:pointwise_on_dw_output"}
    out = [f"conv:{spec['conv']}:k{spec['kernel']}:{parts[spec['conv']]}",
           f"activation:{spec['activation']}:post_hoc(linear_fit,as_fit_new_block)",
           f"pool:{spec['pool']}:library_geometry"]
    if spec.get("exact_small_width"):
        out.append("allocator:exact_small_width(approx)")
    if spec.get("norm", "none") != "none":
        out.append(f"norm:{spec['norm']}:post_hoc(not_spifil)")
    if spec.get("skip") == "identity":
        out.append("residual:identity")
    elif spec.get("skip") == "project":
        out.append("residual:approx:spifil_1x1_projection(exact_alloc)")
    res = {"faithful": (spec["conv"] in ("conv", "pointwise", "inception")
                        and spec.get("skip") != "project" and not spec.get("exact_small_width")),
           "parts": out}
    if spec["conv"] == "inception":
        res["name"] = "inception_per_branch"
    return res


def fit_ops_block(prefix, samples, ops_spec, config, *, prefix_id=None, exact_width=False):
    """SPiFiL-initialize one extended-operator block on the frozen ``prefix``.

    ``exact_width`` (opt-in, external_seed growth): dense conv/pointwise fits use
    ``ExactBudgetAllocator`` (exactly ``channels`` filters) instead of ``AutoAllocator``.

    Mirrors ``flim.progressive.fit_new_block`` (same validation, seeding, layer-0
    construction, budget check, prefix fingerprint, InfeasibleCandidate contract).
    Returns ``(SpifilOpsBlock, provenance)``; ``provenance["effective_spec"]`` is the
    spec ``random_twin_ops`` needs for an exact twin.
    """
    from flim.progressive import InfeasibleCandidate, _fingerprint
    from flim.spifil import Cropped, Trunk, build_layer0
    from ga_search.modehb.spifil_oom import install as _install_oom_fallback
    _install_oom_fallback()          # Fisher scorer: blocked recompute only on CUDA OOM
    spec, cfg = dict(ops_spec), dict(config)
    if not ops_supported(spec):
        raise InfeasibleCandidate(f"unsupported spifil_ops spec {spec}")
    if not 0 <= float(spec.get("dropout", 0)) < 1:
        raise InfeasibleCandidate("unsupported dropout")
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
    budget = bool(cfg.get("budget_check", True))
    prefix = nn.Identity() if prefix is None else prefix
    before = _fingerprint(prefix)
    started = time.perf_counter()
    stages = {}
    try:
        trunk = Trunk(prefix, size, device)
        cin = int(trunk.out_channels)
        if int(spec.get("in_channels", cin)) != cin:     # caller guessed cin wrong: re-derive
            spec = to_ops_spec(_as_layer(spec), cin)
        data = Cropped(samples, size, use_mask=not cfg.get("impurities", False))
        st0 = build_layer0(data, trunk, samples, int(cfg.get("n_superpixels", 100)), size,
                           cfg.get("imagenet_norm", False), device,
                           cfg.get("spifil_in_feature", False), cfg.get("superpixels", "slic"))
        feats, seeds = st0.features, st0.seeds
        kind, k = spec["conv"], int(spec["kernel"])
        state = {}
        exact_small = False
        if kind in ("conv", "pointwise"):
            w, b, n, d = fit_dense(feats, seeds, data, cin, k, spec["channels"],
                                   device=device, budget_check=budget,
                                   allocator=ExactBudgetAllocator() if exact_width else None)
            got = int(w.shape[0])
            exact_small = fit_dense.last_used_exact
            state["conv.weight"], state["conv.bias"] = w, b
            stages["conv"] = {"N": n, "D": d, "exact_small_width": exact_small,
                              "single_filter_stats_all_seeds": fit_dense.last_single_filter_fix}
        elif kind == "inception":
            from ga_search.modehb.ops_norm_act import inception_split
            branches = []
            pooled = None
            for name, bk, share in inception_split(spec["channels"]):
                if name == "pool":
                    if pooled is None:
                        with torch.no_grad():
                            pooled = [F.max_pool2d(f[None].to(device), 3, 1, 1)[0] for f in feats]
                    w, b, n, d = fit_dense(pooled, seeds, data, cin, 1, share,
                                           device=device, budget_check=budget)
                    key = "conv.branches.pool.proj"
                else:
                    w, b, n, d = fit_dense(feats, seeds, data, cin, bk, share,
                                           device=device, budget_check=budget)
                    key = f"conv.branches.{name}"
                exact_small = exact_small or fit_dense.last_used_exact
                if int(w.shape[0]) < 1:
                    raise InfeasibleCandidate(f"inception branch {name}: SPiFiL produced no filters")
                branches.append([name, int(bk), int(w.shape[0])])
                state[f"{key}.weight"], state[f"{key}.bias"] = w, b
                stages[f"inception_{name}"] = {"N": n, "D": d, "requested": int(share),
                                               "got": int(w.shape[0]),
                                               "exact_small_width": fit_dense.last_used_exact,
                                               "single_filter_stats_all_seeds": fit_dense.last_single_filter_fix}
            got = sum(br[2] for br in branches)
        elif kind == "depthwise":
            m = int(spec["dw_multiplier"])
            w, b, n, d = fit_depthwise(feats, seeds, data, cin, k, m, device=device, budget_check=budget)
            got = int(w.shape[0])
            state["conv.weight"], state["conv.bias"] = w, b
            stages["depthwise"] = {"N": n, "D": d, "multiplier": m}
        else:                                              # dwsep
            wd, bd, n, d = fit_depthwise(feats, seeds, data, cin, k, 1, device=device, budget_check=budget)
            stages["depthwise"] = {"N": n, "D": d, "multiplier": 1}
            dw_feats = _apply(wd, bd, feats, cin, device)   # linear dw output, same grid/seeds
            wp, bp, n, d = fit_dense(dw_feats, seeds, data, cin, 1, spec["channels"],
                                     device=device, budget_check=budget)
            got = int(wp.shape[0])
            exact_small = fit_dense.last_used_exact
            stages["pointwise"] = {"N": n, "D": d, "exact_small_width": exact_small}
            state.update({"conv.depthwise.weight": wd, "conv.depthwise.bias": bd,
                          "conv.pointwise.weight": wp, "conv.pointwise.bias": bp})
        if got < 1:
            raise InfeasibleCandidate("SPiFiL produced empty filters")
        eff = dict(spec, channels=got, in_channels=cin)
        if exact_small:
            eff["exact_small_width"] = True
        if kind == "inception":
            eff["inception_branches"] = branches
        if kind == "depthwise":
            eff["dw_multiplier"] = got // cin
        if spec.get("residual"):
            eff["skip"] = "identity" if cin == got else "project"
            if eff["skip"] == "project":
                ws, bs, n, d = fit_dense(feats, seeds, data, cin, 1, got, device=device,
                                         budget_check=budget, allocator=ExactBudgetAllocator())
                if int(ws.shape[0]) != got:
                    raise InfeasibleCandidate(f"residual projection width {ws.shape[0]} != {got}")
                state["skip.weight"], state["skip.bias"] = ws, bs
                stages["skip"] = {"N": n, "D": d}
        block = SpifilOpsBlock(eff, cin).to(device)
        with torch.no_grad():
            own = block.state_dict()
            for name, value in state.items():
                own[name].copy_(value.to(own[name].device))
        if not all(torch.isfinite(p).all() for p in block.parameters()):
            raise InfeasibleCandidate("SPiFiL produced non-finite filters")
        for p in block.parameters():
            p.requires_grad_(True)
        if _fingerprint(prefix) != before:
            raise InfeasibleCandidate("SPiFiL fitting modified the trained prefix")
    except InfeasibleCandidate:
        raise
    except (RuntimeError, ValueError, IndexError, OSError) as exc:
        raise InfeasibleCandidate(f"SPiFiL fit failed: {type(exc).__name__}: {exc}") from exc
    first = next(iter(stages.values()))
    provenance = {
        "origin": "spifil", "builder": BUILDER,
        "requested_channels": int(spec.get("requested_channels", spec["channels"])),
        "actual_channels": got,
        "channel_repair": "allocator_actual_width" if int(spec["channels"]) != got else None,
        "sample_ids": [s.name for s in samples], "sample_labels": [int(s.label) for s in samples],
        "prefix_id": prefix_id or before, "prefix_state_sha256": before, "seed": seed,
        "patch_count": first["N"], "patch_dimension": first["D"], "stages": stages,
        "fit_seconds": time.perf_counter() - started,
        "superpixel_mode": "feature" if cfg.get("spifil_in_feature", False) else "image",
        "config": cfg, "requested_spec": dict(ops_spec), "effective_spec": eff,
        "in_channels": cin, "spifil_variant": _variant(eff),
        "modules": {name.rsplit(".", 1)[0]: "spifil" for name in state},
    }
    block.initialization_provenance = provenance
    return block, provenance
