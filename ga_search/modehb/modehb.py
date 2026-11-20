"""MO-DEHB (Awad et al. 2023, arXiv 2305.04502) for the progressive SPiFiL NAS.

Encoding
--------
A genome is a fixed-length continuous vector ``x in [0,1]^DIM`` (``DIM = 8 + 8*8``):
8 global genes (``GLOBAL_GENES``) followed by 8 layers x 8 genes (``LAYER_GENES``).
``spifil_images_per_class`` (training images per class SPiFiL learns kernels from) is a
global gene whose choices come from the search config ``spifil_images_choices`` (default
``SPIFIL_IMAGES``); ``decode``/``encode``/``repair`` take them as ``spifil_images_choices``.
Categorical/ordinal genes decode as ``choices[min(int(v*n), n-1)]``; ``encode`` writes
the bin centre ``(i+0.5)/n``. ``depth`` masks layers (``active = i < depth``); inactive
layer genes keep evolving in the vector but are canonicalised away from ``arch_hash``.
``seed`` is NOT a gene (fixed per search, identical for twins).

``repair`` (idempotent) enforces: depth in 1..max_depth, choices snapped to the grid,
embed_dim >= 1280, pointwise => kernel 1, dwsep/depthwise not on layer 0 (cin=3),
kernel 1 only with conv/pointwise, residual only when cin == cout and no pool, number of
stride-2 pools bounded so the final map is >= ``min_spatial``; when ``init == "spifil"``
every active layer goes through ``init.repair_for_spifil`` (never spifil -> random).

Channel space (opt-in, ``channel_space`` in the search config; absent -> the discrete
``CHANNELS`` grid, unchanged): ``{type: int_range, min: 1, max: 1024, scale: log|linear,
monotonic: true}``. The channel gene decodes to ANY integer in ``[min, max]`` (log scale:
``round(min * (max/min)**v)``, so small widths are explored as densely as large ones;
``encode`` is the exact inverse on integers). ``repair`` clamps to the range and, with
``monotonic``, enforces ``c1 <= c2 <= ... <= c_depth`` over ACTIVE layers by a running max
(inactive layers are only clamped); a depthwise layer (cout = cin * multiplier) is floored
to a multiple of its cin (still >= cin, so monotonicity holds). Idempotent; the random
twin copies channels exactly. NOTE: the monotonic rule applies to the GENOME. SPiFiL may
build FEWER channels than requested (class allocator, ``provenance["effective_spec"]``),
so the built widths may not be monotonic; ``run_search`` logs ``effective_monotonic``.
Today's 24/48/96/192/384 widths come from the genome grid itself (``CHANNELS`` below),
not only from SPiFiL's effective_spec shrinking.

Objectives (all minimised, normalised to [0,1])
-----------------------------------------------
``f0 = 1 - quality`` (EmbeddingEvaluator.quality after collapse penalty),
``f1 = clip(log10(flops/FLOPS_LO) / log10(FLOPS_HI/FLOPS_LO), 0, 1)`` with defaults
``FLOPS_LO=1e6, FLOPS_HI=1e11`` MACs, optional ``f2 = clip(latency_ms / LAT_HI, 0, 1)``
(``LAT_HI=100``). Failed evaluations get ``FAIL_OBJ = 1.0`` in every objective.
Fixed hypervolume reference point: ``REF_POINT = 1.1`` per objective (documented, never
data-dependent, so HV is comparable across brackets, datasets and runs).

MO-DEHB
-------
* DE rand/1 + binomial crossover (``dehb.optimizers.de.DE`` semantics), parent pool =
  top of the lower-fidelity sub-population ranked by non-dominated sorting, filled with
  rand/1 mutants drawn from the global population (DEHB ``_acquire_config``).
* SH promotion: ``select_promotions`` = whole non-dominated fronts (pymoo
  ``NonDominatedSorting``), last front broken by crowding distance (or EpsNet).
* DE selection (``mo_select``): offspring replaces target if it dominates it; if they are
  mutually non-dominated, the population+offspring is sorted and the member of the last
  front with the least HV contribution is removed (offspring takes its slot); else no update.
* Initial population (>= 16): channels {16,32,64,128} x depths x {spifil, random}; random
  twins identical except ``init``. Stage 1 mutates only channels/depth/embed_dim/spifil_images_per_class; stage 2
  unlocks all operators; with probability ``epsilon`` an offspring ignores the stage lock.
  ``spifil_fraction`` (default .75) of new offspring belong to SPiFiL lineages, and every
  SPiFiL candidate evaluated at max fidelity gets its random twin evaluated along the same
  fidelity path, with the same seed (fair comparison). Twins are controls: they are kept in
  the archive but do not enter the DE sub-populations (``twins_in_population``).
* search_v3 switches (all opt-in; defaults keep the behaviour above): ``random_baseline=False``
  -> SPiFiL only (no twins, no random offspring, SPiFiL-only seeds, ``initial_size`` seeds);
  ``stage1_brackets=0`` -> every gene mutable from bracket 0; ``search_space`` (YAML
  ``stages.stage2.space``) -> per-gene vocabularies for decode/encode/repair (``SPACE_UNIVERSE``
  adds conv ``inception`` and norms ``batch_zca``/``batch_dbn``); ``initial_operators="sampled"``
  -> seeds' per-layer operators + final pooling stratified-sampled from that space.
* Restart: every finished job is appended to ``search_state.json`` (atomic write). On
  resume, the search is replayed from the same seed and the logged results are returned
  instead of re-training (genome hash checked), so the state reproduces bit-for-bit.
  Checkpoints/state are never deleted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# ────────────────────────────────────────────────────────────────── search space
N_LAYERS = 8
CHANNELS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
KERNELS = (1, 3, 5, 7)
CONVS = ("conv", "depthwise", "pointwise", "dwsep")
ACTIVATIONS = ("relu", "relu6", "silu", "gelu", "leaky_relu", "prelu", "selu")
NORMS = ("none", "batch", "group", "layer", "instance", "zca", "dbn")
POOLS = ("none", "max", "avg")
BOOLS = (False, True)
INITS = ("spifil", "random")
SSLS = ("simclr", "lejepa", "ae")
REGS = ("none", "sigreg", "weak_sigreg")
EMBED_DIMS = (1280, 1536, 2048, 2560)
POOLINGS = ("gap", "gem", "gem_learn", "meanstd", "moments", "combo")
AE_TARGETS = ("pixel", "feature", "masked")
DEPTHS = tuple(range(1, N_LAYERS + 1))
SPIFIL_IMAGES = (1, 2, 4, 8, 16)   # default ``spifil_images_choices``
# Hybrid SPiFiL: how backprop treats the SPiFiL conv kernels (lr multiplier 1 / 0.01 / 0)
PROTECTS = ("full", "lowlr", "freeze")
PROTECT_LR_SCALE = {"full": 1.0, "lowlr": 0.01, "freeze": 0.0}
# per-layer local AE: "joint" (block+decoder together) | "decoder_first" (block frozen for the
# first half while the decoder adapts to SPiFiL, then the block joins)
AE_SCHEDULES = ("joint", "decoder_first")
AE_WARMUP_FRAC = {"joint": 0.0, "decoder_first": 0.5}

GLOBAL_GENES = {"init": INITS, "depth": DEPTHS, "ssl": SSLS, "reg": REGS,
                "embed_dim": EMBED_DIMS, "pooling": POOLINGS, "ae_target": AE_TARGETS,
                "spifil_images_per_class": SPIFIL_IMAGES,
                # appended last (Hybrid SPiFiL, 14/11): SSL losses on a separate projector, kernel protection
                "projector": BOOLS, "protect": PROTECTS, "ae_schedule": AE_SCHEDULES}
LAYER_GENES = {"channels": CHANNELS, "kernel": KERNELS, "conv": CONVS, "residual": BOOLS,
               "activation": ACTIVATIONS, "norm": NORMS, "pool": POOLS, "softhebb": BOOLS}
N_GLOBAL, N_LAYER_GENES = len(GLOBAL_GENES), len(LAYER_GENES)
DIM = N_GLOBAL + N_LAYERS * N_LAYER_GENES

# Extended operator universes accepted in a YAML ``search_space`` (opt-in). The default
# tuples above are untouched so default decoding (bin layout) is unchanged.
SPACE_UNIVERSE = {"conv": CONVS + ("inception",), "kernel": KERNELS, "activation": ACTIVATIONS,
                  "norm": NORMS + ("batch_zca", "batch_dbn"), "pool": POOLS, "residual": BOOLS,
                  "softhebb": BOOLS, "pooling": POOLINGS, "projector": BOOLS, "protect": PROTECTS,
                  "ae_schedule": AE_SCHEDULES, "ssl": SSLS, "reg": REGS}
# YAML names (stages.stage2.space) -> gene names
SPACE_ALIASES = {"operator": "conv", "conv": "conv", "kernel": "kernel", "activation": "activation",
                 "norm": "norm", "pool": "pool", "residual": "residual", "softhebb": "softhebb",
                 "softhebb_conv": "softhebb", "pooling": "pooling", "projector": "projector",
                 "ssl_projector": "projector", "protect": "protect", "spifil_protect": "protect",
                 "ae_schedule": "ae_schedule", "ssl": "ssl", "reg": "reg"}

STAGE1_GENES = ("depth", "embed_dim", "channels", "spifil_images_per_class")   # stage 1: only these mutate
NEVER_MUTATED = ("init",)                            # lineage is set by allocation, not DE

FLOPS_LO, FLOPS_HI, LAT_HI = 1e6, 1e11, 100.0
REF_POINT = 1.1
FAIL_OBJ = 1.0


def gene_index(name: str, layer: Optional[int] = None) -> int:
    if layer is None:
        return list(GLOBAL_GENES).index(name)
    return N_GLOBAL + layer * N_LAYER_GENES + list(LAYER_GENES).index(name)


def _dec(v: float, choices: Sequence):
    n = len(choices)
    return choices[min(max(int(float(v) * n), 0), n - 1)]


def _enc(value, choices: Sequence) -> float:
    if value not in choices:
        raise ValueError(f"{value!r} not in {choices}")
    return (list(choices).index(value) + 0.5) / len(choices)


def _choices(name: str, spifil_images_choices: Optional[Sequence] = None,
             search_space: Optional[dict] = None) -> tuple:
    """Choices of global gene ``name`` (``spifil_images_per_class`` is config-driven)."""
    if name == "spifil_images_per_class" and spifil_images_choices:
        return tuple(sorted({int(c) for c in spifil_images_choices}))
    if search_space and name in search_space:
        return search_space[name]
    return GLOBAL_GENES[name]


def _layer_choices(name: str, search_space: Optional[dict] = None) -> tuple:
    if search_space and name in search_space:
        return search_space[name]
    return LAYER_GENES[name]


def normalize_search_space(space) -> Optional[dict]:
    """YAML per-gene space -> ``{gene: tuple}`` (validated against ``SPACE_UNIVERSE``).

    Keys: operator|conv, kernel, activation, norm, pool, residual, softhebb|softhebb_conv,
    pooling (final). Unknown keys (e.g. legacy batchnorm/whitening) are ignored; absent
    genes keep the hardcoded vocabulary. ``None``/empty -> None (default behaviour)."""
    if not space:
        return None
    out = {}
    for key, vals in dict(space).items():
        gene = SPACE_ALIASES.get(key)
        if gene is None:
            continue
        uni = SPACE_UNIVERSE[gene]
        if gene in ("residual", "softhebb", "projector"):
            vals = [bool(v) for v in vals]
        elif gene == "kernel":
            vals = [int(v) for v in vals]
        bad = [v for v in vals if v not in uni]
        if bad or not vals:
            raise ValueError(f"search_space.{key}: {bad or 'empty'} not in {uni}")
        seen = []
        for v in vals:
            if v not in seen:
                seen.append(v)
        out[gene] = tuple(seen)
    return out or None


def _channel_space(cs) -> Optional[dict]:
    """Normalised opt-in integer channel space, or None (discrete ``CHANNELS`` grid)."""
    if not cs:
        return None
    if cs.get("type", "int_range") != "int_range":
        raise ValueError(f"channel_space.type must be 'int_range', got {cs.get('type')!r}")
    lo, hi = int(cs.get("min", 1)), int(cs.get("max", 1024))
    scale = cs.get("scale", "log")
    if not 1 <= lo <= hi or scale not in ("log", "linear"):
        raise ValueError(f"bad channel_space {cs!r}")
    return {"type": "int_range", "min": lo, "max": hi, "scale": scale,
            "monotonic": bool(cs.get("monotonic", True))}


def _dec_channels(v: float, cs: dict) -> int:
    lo, hi, v = cs["min"], cs["max"], min(max(float(v), 0.0), 1.0)
    c = lo * (hi / lo) ** v if cs["scale"] == "log" else lo + v * (hi - lo)
    return int(min(max(int(round(c)), lo), hi))


def _enc_channels(c, cs: dict) -> float:
    lo, hi = cs["min"], cs["max"]
    c = min(max(int(c), lo), hi)
    if hi == lo:
        return 0.5
    if cs["scale"] == "log":
        return math.log(c / lo) / math.log(hi / lo)
    return (c - lo) / (hi - lo)


def _snap(value, choices):
    if value in choices and not (isinstance(value, bool) != isinstance(choices[0], bool)):
        return value
    if isinstance(choices[0], bool):
        return bool(value) if bool(value) in choices else choices[0]
    if all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices):
        return min(choices, key=lambda c: (abs(c - float(value)), c))
    return choices[0]


# ────────────────────────────────────────────────────────────────── genome ↔ vector
def decode(vector, seed: int = 0, *, image_size: int = 200, min_spatial: int = 6,
           max_depth: int = N_LAYERS, spifil_repair: Optional[Callable] = None,
           spifil_images_choices: Optional[Sequence] = None,
           channel_space: Optional[dict] = None, search_space: Optional[dict] = None) -> dict:
    """[0,1]^DIM -> repaired genome dict. ``search_space`` restricts operator vocabularies."""
    cs = _channel_space(channel_space)
    ss = normalize_search_space(search_space) if search_space and not _is_norm(search_space) else search_space
    v = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    if v.shape != (DIM,):
        raise ValueError(f"vector must have shape ({DIM},), got {v.shape}")
    g: Dict[str, Any] = {name: _dec(v[gene_index(name)], _choices(name, spifil_images_choices, ss))
                         for name in GLOBAL_GENES}
    g["seed"] = int(seed)
    g["layers"] = []
    for i in range(N_LAYERS):
        layer = {name: _dec(v[gene_index(name, i)], _layer_choices(name, ss)) for name in LAYER_GENES}
        if cs is not None:
            layer["channels"] = _dec_channels(v[gene_index("channels", i)], cs)
        g["layers"].append(layer)
    return repair(g, image_size=image_size, min_spatial=min_spatial, max_depth=max_depth,
                  spifil_repair=spifil_repair, spifil_images_choices=spifil_images_choices,
                  **({"channel_space": cs} if cs is not None else {}),
                  **({"search_space": ss} if ss else {}))


def _is_norm(space) -> bool:
    """True when ``space`` is already a normalised ``{gene: tuple}`` dict."""
    return all(k in SPACE_UNIVERSE and isinstance(v, tuple) for k, v in space.items())


def encode(genome: dict, spifil_images_choices: Optional[Sequence] = None,
           channel_space: Optional[dict] = None, search_space: Optional[dict] = None) -> np.ndarray:
    """Genome -> bin-centre vector (inverse of ``decode`` on repaired genomes)."""
    cs = _channel_space(channel_space)
    ss = normalize_search_space(search_space) if search_space and not _is_norm(search_space) else search_space
    enc = _enc if not ss else (lambda val, ch: _enc(_snap(val, ch), ch))   # repair may leave space
    v = np.zeros(DIM)
    for name in GLOBAL_GENES:
        v[gene_index(name)] = enc(genome[name], _choices(name, spifil_images_choices, ss))
    for i, layer in enumerate(genome["layers"][:N_LAYERS]):
        for name in LAYER_GENES:
            if name == "channels" and cs is not None:
                v[gene_index(name, i)] = _enc_channels(layer[name], cs)
            else:
                v[gene_index(name, i)] = enc(layer[name], _layer_choices(name, ss))
    return v


def _default_spifil_repair(layer):
    from ga_search.modehb.init import repair_for_spifil
    return repair_for_spifil(layer)


def repair(genome: dict, *, image_size: int = 200, min_spatial: int = 6,
           max_depth: int = N_LAYERS, spifil_repair: Optional[Callable] = None,
           spifil_images_choices: Optional[Sequence] = None,
           channel_space: Optional[dict] = None, search_space: Optional[dict] = None) -> dict:
    """Return a valid copy of ``genome`` (idempotent). Never changes ``init``.

    ``search_space`` (opt-in, YAML ``stages.stage2.space``): operator genes are snapped to
    its per-gene choices before the structural/SPiFiL repairs (which may still fall back to
    conv / kernel 3 / pool none / residual False).

    ``channel_space`` (opt-in, see module docstring): integer widths in [min, max],
    non-decreasing over active layers when ``monotonic``; None = discrete ``CHANNELS``."""
    cs = _channel_space(channel_space)
    ss = normalize_search_space(search_space) if search_space and not _is_norm(search_space) else search_space
    g = copy.deepcopy(genome)
    if g.get("init") not in INITS:
        raise ValueError(f"init must be one of {INITS}")
    g["seed"] = int(g.get("seed", 0))
    g["depth"] = int(min(max(int(g.get("depth", 1)), 1), max_depth))
    for key in GLOBAL_GENES:
        if key not in ("init", "depth"):
            choices = _choices(key, spifil_images_choices, ss)
            missing = 1 if key == "spifil_images_per_class" else choices[0]   # legacy: 1 image/class
            g[key] = _snap(g.get(key, missing), choices)
    g["embed_dim"] = max(int(g["embed_dim"]), 1280)
    if g.get("ssl") == "lejepa" and g.get("reg") == "none":
        g["reg"] = "sigreg"         # LeJEPA without a regulariser is undefined (ssl_trainers raises)
    layers = list(g.get("layers", []))[:N_LAYERS]
    while len(layers) < N_LAYERS:
        layers.append({})
    defaults = {"channels": 16, "kernel": 3, "conv": "conv", "residual": False,
                "activation": "relu", "norm": "none", "pool": "none", "softhebb": False}
    fix = spifil_repair or _default_spifil_repair
    out_layers = []
    cin, spatial = 3, int(image_size)
    for i, raw in enumerate(layers):
        L = {k: raw.get(k, d) for k, d in defaults.items()}
        if cs is None:
            L["channels"] = int(_snap(int(L["channels"]), CHANNELS))
        else:
            L["channels"] = int(min(max(int(round(float(L["channels"]))), cs["min"]), cs["max"]))
        L["kernel"] = int(_snap(int(L["kernel"]), _layer_choices("kernel", ss)))
        for key in ("conv", "activation", "norm", "pool"):
            L[key] = _snap(L[key], _layer_choices(key, ss))
        L["residual"], L["softhebb"] = bool(L["residual"]), bool(L["softhebb"])
        if ss:
            for key in ("residual", "softhebb"):
                L[key] = _snap(L[key], _layer_choices(key, ss))
        L["active"] = i < g["depth"]
        if L["active"]:
            if g["init"] == "spifil":
                L = _apply_spifil(L, fix)
            if cs is not None and cs["monotonic"] and i > 0:
                L["channels"] = max(L["channels"], cin)          # running max (cin = previous width)
                if L["conv"] == "depthwise":                     # cout = cin * multiplier
                    L["channels"] = (L["channels"] // cin) * cin
            L, cin, spatial = _repair_dims(L, i, cin, spatial, min_spatial,
                                           project_residual=search_space is not None)
            if g["init"] == "spifil":   # dims repair only narrows; re-check support
                L, cin2, _ = _repair_dims(_apply_spifil(L, fix), i, None, None, min_spatial)
        out_layers.append(L)
    g["layers"] = out_layers
    return g


def _apply_spifil(L, fix):
    active = L["active"]
    R = dict(fix({k: v for k, v in L.items() if k != "active"}))
    R.pop("spifil_repairs", None)
    R = {k: R.get(k, L[k]) for k in L if k != "active"}
    R["active"] = active
    R["kernel"] = int(R["kernel"])
    return R


def _repair_dims(L, i, cin, spatial, min_spatial, project_residual=False):
    L = dict(L)
    if L["conv"] == "pointwise":
        L["kernel"] = 1
    if L["conv"] in ("dwsep", "inception") and L["kernel"] == 1:
        L["kernel"] = 3
    if i == 0 and L["conv"] in ("depthwise", "dwsep"):
        L["conv"] = "conv"          # cin=3 input: grouped convs are degenerate
    if L["kernel"] == 1 and L["conv"] == "depthwise":
        L["kernel"] = 3
    if cin is None:                 # second pass (support re-check only)
        return L, None, None
    if L["pool"] != "none":
        if spatial // 2 < min_spatial:
            L["pool"] = "none"
        else:
            spatial //= 2
    if L["residual"] and (i == 0 or (not project_residual
                                     and (cin != L["channels"] or L["pool"] != "none"))):
        # project_residual (YAML search space + spifil_ops): width mismatch is bridged by a
        # 1x1 SPiFiL projection skip (spifil_ops "project"), so residual is allowed for i>0
        L["residual"] = False
    return L, L["channels"], spatial


def canonical(genome: dict, include_init: bool = True) -> dict:
    """Effective genome: active layers only, no seed, no bookkeeping keys."""
    keys = [k for k in GLOBAL_GENES if include_init or k != "init"]
    c = {k: genome[k] for k in keys}
    c["layers"] = [{k: L[k] for k in LAYER_GENES} for L in genome["layers"][:genome["depth"]]]
    if genome.get("external"):      # external-seed genomes (external_seed.py) never alias SPiFiL ones
        c["external"] = genome["external"]
    return c


def arch_hash(genome: dict, include_init: bool = True) -> str:
    blob = json.dumps(canonical(genome, include_init), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_twin(genome: dict) -> dict:
    """Random twin: identical genome (incl. seed and repaired SPiFiL topology), init=random."""
    t = copy.deepcopy(genome)
    t["init"] = "random"
    return t


def gene_mask(genome: dict, stage: int) -> np.ndarray:
    """Boolean mask of genes DE may change: active-gene mask x stage lock (init never)."""
    m = np.zeros(DIM, dtype=bool)
    for name in GLOBAL_GENES:
        if name in NEVER_MUTATED:
            continue
        m[gene_index(name)] = stage >= 2 or name in STAGE1_GENES
    for i in range(N_LAYERS):
        # inactive layers stay mutable: depth growth re-activates them (masked in decode)
        for name in LAYER_GENES:
            m[gene_index(name, i)] = stage >= 2 or name in STAGE1_GENES
    return m


def active_mask(genome: dict) -> np.ndarray:
    """Genes that influence the phenotype (depth masks layers)."""
    m = np.zeros(DIM, dtype=bool)
    m[:N_GLOBAL] = True
    for i in range(genome["depth"]):
        m[gene_index("channels", i):gene_index("channels", i) + N_LAYER_GENES] = True
    return m


def initial_population(seed: int, channels=(16, 32, 64, 128), depths=(2, 3), *,
                       embed_dim: int = 1280, ssl: str = "lejepa", reg: str = "sigreg",
                       pooling: str = "gap", ae_target: str = "pixel", pyramid_growth: float = 1.0,
                       twins: bool = True, size: Optional[int] = None,
                       sample_operators: bool = False, **repair_kw) -> List[dict]:
    """Small valid seeds: channels x depths x {spifil, random}; twins differ only in init.

    ``twins=False`` (``random_baseline: false``): SPiFiL seeds only; the channels x depths
    grid is cycled until ``size`` seeds exist. ``sample_operators=True``: each seed keeps
    its pyramid widths/depth, but every layer's operator genes (conv, kernel, activation,
    norm, pool, residual, softhebb) and the final ``pooling`` are drawn by a seeded
    stratified (Latin-hypercube-like) sample over ``repair_kw["search_space"]`` (default
    vocabularies when absent): per gene and layer, the choices are tiled over the seeds and
    permuted, so every choice appears ~equally often. Then repaired as usual.

    ``pyramid_growth`` > 1 builds pyramid seeds: layer i gets round(ch * growth**i) filters
    (capped at 1024), e.g. 16 -> 32 -> 64 with growth 2; 1.0 = uniform widths (old behaviour).

    ``spifil_images_per_class`` cycles through its choices across the twin pairs."""
    img_choices = _choices("spifil_images_per_class", repair_kw.get("spifil_images_choices"))
    grid = [(depth, ch) for depth in depths for ch in channels]
    n = len(grid) if twins else max(len(grid), int(size or 0))
    plan = [grid[j % len(grid)] for j in range(n)]
    ops = None
    if sample_operators:
        ss = repair_kw.get("search_space")
        ss = normalize_search_space(ss) if ss and not _is_norm(ss) else ss
        rng = np.random.default_rng([int(seed), 0x5EED])

        def strat(choices):
            tiled = [choices[j % len(choices)] for j in range(n)]
            return [tiled[j] for j in rng.permutation(n)]
        ops = {"pooling": strat(_choices("pooling", None, ss)),
               "projector": strat(_choices("projector", None, ss)),
               "protect": strat(_choices("protect", None, ss)),
               "ae_schedule": strat(_choices("ae_schedule", None, ss)),
               "layers": [{k: strat(_layer_choices(k, ss)) for k in
                           ("conv", "kernel", "activation", "norm", "pool", "residual", "softhebb")}
                          for _ in range(N_LAYERS)]}
    pop = []
    for j, (depth, ch) in enumerate(plan):
        layers = [{"channels": min(1024, max(1, int(round(ch * float(pyramid_growth) ** i)))), "kernel": 3, "conv": "conv", "residual": False,
                   "activation": "relu", "norm": "batch", "pool": "max", "softhebb": False}
                  for i in range(N_LAYERS)]
        if ops is not None:
            for i, L in enumerate(layers):
                L.update({k: v[j] for k, v in ops["layers"][i].items()})
        k_img = len(pop) // 2 if twins else j
        base = {"init": "spifil", "seed": int(seed), "depth": depth, "layers": layers,
                "ssl": ssl, "reg": reg, "embed_dim": embed_dim,
                "pooling": ops["pooling"][j] if ops is not None else pooling,
                **({"projector": ops["projector"][j], "protect": ops["protect"][j],
                    "ae_schedule": ops["ae_schedule"][j]} if ops is not None else {}),
                "ae_target": ae_target,
                "spifil_images_per_class": img_choices[k_img % len(img_choices)]}
        sp = repair(base, **repair_kw)
        pop.append(sp)
        if twins:
            pop.append(make_twin(sp))
    return pop


def widths_monotonic(widths) -> bool:
    """True when ``widths`` (e.g. SPiFiL effective_spec channels per layer) is non-decreasing."""
    w = [int(c) for c in widths]
    return all(a <= b for a, b in zip(w, w[1:]))


# ────────────────────────────────────────────────────────────────── objectives / MO
def objectives(result: dict, use_latency: bool = False, bounds: Optional[dict] = None) -> List[float]:
    b = {"flops_lo": FLOPS_LO, "flops_hi": FLOPS_HI, "latency_hi_ms": LAT_HI, **(bounds or {})}
    n = 3 if use_latency else 2
    if result.get("status", "ok") != "ok":
        return [FAIL_OBJ] * n
    q = float(result.get("quality", 0.0))
    q = min(max(q if math.isfinite(q) else 0.0, 0.0), 1.0)
    fl = max(float(result.get("flops", b["flops_lo"])), b["flops_lo"])
    f1 = math.log10(fl / b["flops_lo"]) / math.log10(b["flops_hi"] / b["flops_lo"])
    out = [1.0 - q, min(max(f1, 0.0), 1.0)]
    if use_latency:
        out.append(min(max(float(result.get("latency_ms", b["latency_hi_ms"])) / b["latency_hi_ms"], 0.0), 1.0))
    return out


def dominates(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_fronts(F) -> List[np.ndarray]:
    F = np.asarray(F, float)
    if len(F) == 0:
        return []
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    return [np.asarray(f, dtype=int) for f in NonDominatedSorting().do(F)]


def front_ranks(F) -> np.ndarray:
    ranks = np.zeros(len(F), dtype=int)
    for r, front in enumerate(nondominated_fronts(F)):
        ranks[front] = r
    return ranks


def hypervolume(F, ref=None) -> float:
    """Exact HV of the minimisation set ``F`` w.r.t. ``ref`` (default REF_POINT per dim)."""
    F = np.atleast_2d(np.asarray(F, float))
    if F.size == 0:
        return 0.0
    ref = np.full(F.shape[1], REF_POINT) if ref is None else np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        P = F[np.lexsort((F[:, 1], F[:, 0]))]
        hv, best_y = 0.0, ref[1]
        for x, y in P:
            if y < best_y:
                hv += (ref[0] - x) * (best_y - y)
                best_y = y
        return float(hv)
    from pymoo.indicators.hv import HV
    return float(HV(ref_point=ref)(F))


def hv_contributions(F, ref=None) -> np.ndarray:
    """Exclusive HV contribution of each point: HV(F) - HV(F without i)."""
    F = np.atleast_2d(np.asarray(F, float))
    total = hypervolume(F, ref)
    return np.array([total - hypervolume(np.delete(F, i, axis=0), ref) for i in range(len(F))])


def crowding_distance(F) -> np.ndarray:
    F = np.asarray(F, float)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span <= 0:
            continue
        d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


def epsnet(F, k: int) -> List[int]:
    """Greedy EpsNet (MO-DEHB option): start at best f0, add farthest-from-selected point."""
    F = np.asarray(F, float)
    if k >= len(F):
        return list(range(len(F)))
    chosen = [int(np.lexsort(F.T[::-1])[0])]
    while len(chosen) < k:
        dist = np.min(np.linalg.norm(F[:, None, :] - F[chosen][None], axis=2), axis=1)
        dist[chosen] = -1
        chosen.append(int(np.argmax(dist)))
    return chosen


def select_promotions(F, n: int, tie_breaker: str = "crowding") -> List[int]:
    """Indices of the ``n`` promoted points: whole fronts, last front by crowding/EpsNet."""
    F = np.asarray(F, float)
    chosen: List[int] = []
    for front in nondominated_fronts(F):
        if len(chosen) + len(front) <= n:
            chosen.extend(int(i) for i in front)
            if len(chosen) == n:
                break
            continue
        k = n - len(chosen)
        if tie_breaker == "epsnet":
            pick = epsnet(F[front], k)
        else:
            cd = crowding_distance(F[front])
            pick = list(np.argsort(-cd, kind="stable")[:k])
        chosen.extend(int(front[i]) for i in pick)
        break
    return chosen


def mo_select(pop_F, target: int, child_F, ref=None) -> Optional[int]:
    """MO-DEHB DE selection. Returns the population slot the child replaces, or None."""
    pop_F = np.atleast_2d(np.asarray(pop_F, float))
    child = np.asarray(child_F, float)
    if dominates(child, pop_F[target]):
        return int(target)
    if dominates(pop_F[target], child):
        return None
    allF = np.vstack([pop_F, child])
    last = nondominated_fronts(allF)[-1]
    if len(last) == 1:
        worst = int(last[0])
    else:
        contrib = hv_contributions(allF[last], ref)
        worst = int(last[int(np.argmin(contrib))])
    return None if worst == len(pop_F) else worst


# ────────────────────────────────────────────────────────────────── state helpers
def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _dehb_base():
    from dehb.optimizers.dehb import DEHBBase
    return DEHBBase


_FATAL_CUDA = ("illegal memory access", "CUSOLVER_STATUS", "CUBLAS_STATUS", "cuDNN error",
               "device-side assert", "CUDA error: unspecified launch failure", "misaligned address")


def _fatal_cuda(exc) -> bool:
    """Errors after which the CUDA context is unusable (out-of-memory is NOT one of them)."""
    msg = str(exc)
    return any(k in msg for k in _FATAL_CUDA) and "out of memory" not in msg


OOM_RETRIES = 10           # re-runs of a job that hit CUDA out-of-memory before it counts as failed (shared GPU: be patient)
OOM_RETRY_SLEEP_S = 300


_OOM_LIKE = ("out of memory", "unable to find an engine",       # cuDNN: no algorithm fits the free workspace
             "cudnn_status_alloc_failed", "cublas_status_alloc_failed")


def _is_oom(exc) -> bool:
    msg = str(exc).lower()
    return type(exc).__name__ == "OutOfMemoryError" or any(k in msg for k in _OOM_LIKE)


class ReplayMismatch(RuntimeError):
    pass


DEFAULTS = dict(fidelities=[5, 15, 45], eta=3, n_configs_scale=1, max_brackets=6,
                max_evaluations=None, gpu_hours_budget=None, mutation_factor=0.5,
                crossover_prob=0.5, epsilon=0.1, spifil_fraction=0.75, stage1_brackets=2,
                tie_breaker="crowding", use_latency=False, objective_bounds={},
                ref_point=REF_POINT, image_size=200, min_spatial=6, max_depth=N_LAYERS,
                initial_channels=[16, 32, 64, 128], initial_depths=[2, 3],
                initial_ssl="lejepa", initial_reg="sigreg", initial_pooling="gap", initial_pyramid_growth=1.0,
                twins_in_population=False, boundary_fix_type="random",
                random_baseline=True, initial_size=None, initial_operators="plain", search_space=None,
                spifil_images_choices=list(SPIFIL_IMAGES), seed=0)


class MODEHB:
    """Synchronous MO-DEHB. ``evaluate(genome, fidelity, ctx) -> result dict``.

    ``result`` keys: status ("ok"|"failed"), quality [0,1], flops (MACs), latency_ms,
    checkpoint (path|None), gpu_hours, plus anything else (stored verbatim).
    ``ctx`` keys: cand_id, parent_id, generation, bracket, rung, role ("search"|"twin"),
    twin_of, resume (record of a lower-fidelity evaluation of the same genome or None),
    twin_source (SPiFiL record at the same fidelity for twins, else None).

    Hyperband schedule, DE operators and seeding follow ``dehb`` 0.1.2 (``DEHBBase`` for the
    bracket spacing, ``DE.mutation_rand1``/``crossover_bin``/``boundary_check`` semantics).
    """

    def __init__(self, config: dict, evaluate: Callable, output_dir, *, logger=None,
                 spifil_repair: Optional[Callable] = None, param_fn: Optional[Callable] = None):
        self.cfg = {**DEFAULTS, **(config or {})}
        from ga_search.modehb import strata as S
        sc = self.cfg.get("stratified") or {}
        # opt-in parameter-stratified search (YAML ``stratified:``); None = original MO-DEHB
        self.strat = S.strat_config(sc) if sc.get("enabled") else None
        self.param_fn = param_fn or S.param_estimate
        # v4 ``stratified.cap_count: spifil_ops``: param_cap on strata.param_count_built (>= the measured
        # params) BEFORE training; absent = the ConvBlock estimate, cap checked only after training (v3)
        self._cap_built = (self.strat or {}).get("cap_count") == "spifil_ops"
        self.cap_fn = S.param_count_built if self._cap_built else self.param_fn
        self.evaluate_fn = evaluate
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "search_state.json"
        self.logger = logger
        self.spifil_repair = spifil_repair
        self.fidelities = [float(f) for f in self.cfg["fidelities"]]
        if sorted(self.fidelities) != self.fidelities or len(set(self.fidelities)) < 2:
            raise ValueError("fidelities must be strictly increasing, >= 2 values")
        self.n_obj = 3 if self.cfg["use_latency"] else 2
        self.ref = np.full(self.n_obj, float(self.cfg["ref_point"]))
        self.repair_kw = dict(image_size=int(self.cfg["image_size"]),
                              min_spatial=int(self.cfg["min_spatial"]),
                              max_depth=int(self.cfg["max_depth"]), spifil_repair=spifil_repair,
                              spifil_images_choices=tuple(int(c) for c in self.cfg["spifil_images_choices"]))
        if self.cfg.get("channel_space"):   # opt-in integer channel space (absent: discrete grid)
            self.repair_kw["channel_space"] = _channel_space(self.cfg["channel_space"])
        ss = normalize_search_space(self.cfg.get("search_space"))
        if ss:   # opt-in per-gene operator vocabularies (YAML stages.stage2.space)
            self.repair_kw["search_space"] = ss
        self.random_baseline = bool(self.cfg.get("random_baseline", True))
        if self.cfg.get("initial_operators", "plain") not in ("plain", "sampled"):
            raise ValueError("initial_operators must be 'plain' or 'sampled'")
        self.enc_kw = dict(spifil_images_choices=self.repair_kw["spifil_images_choices"],
                           **({"channel_space": self.repair_kw["channel_space"]}
                              if "channel_space" in self.repair_kw else {}),
                           **({"search_space": ss} if ss else {}))
        self._init_cache = None
        self._last_op = None
        from ga_search.modehb import lemonade as LM
        # opt-in LEMONADE warm start + network morphisms + Lamarckian inheritance (YAML ``lemonade:``)
        self.lemonade = LM.lemonade_config(self.cfg.get("lemonade"))
        self._lineage = {}          # arch_hash(initial morphed child) -> (parent genome, morphism info)
        self._pending_inherit = None
        from ga_search.modehb import pnas_cs as PC
        # opt-in PNAS-CS (YAML ``pnas_cs:``): compound-scaled families P0..P4 = the strata, growth by the
        # function-preserving morphisms of external_seed.py, structural mutations, Phase-0 AE ablation
        self.pnas = PC.pnas_config(self.cfg.get("pnas_cs"))
        self.dim = DIM
        self._families = {}
        if self.pnas is not None:
            if not sc.get("enabled"):
                raise ValueError("pnas_cs needs stratified.enabled: true (the P-levels are the strata)")
            names = list(self.pnas["level_names"])
            ex = [s for s in (sc.get("explore_strata") or []) if s in names] or names[2:]
            self.strat = S.strat_config({**sc, "strata": PC.level_strata(self.pnas), "param_cap": PC.cap(self.pnas),
                                         "explore_strata": ex})
            self.dim = DIM + PC.N_EXTRA_GENES
        self._schedule = self._hb_schedule()
        self._max_pop = self._max_pop_sizes()

    # ---- Hyperband spacing (reuses dehb.DEHBBase._get_next_iteration)
    def _hb_schedule(self):
        base = _dehb_base()
        eta = float(self.cfg["eta"])
        stub = base.__new__(base)
        stub.max_SH_iter, stub.fidelities, stub.eta = len(self.fidelities), np.array(self.fidelities), eta
        stub.min_clip = stub.max_clip = None
        sched = []
        for it in range(len(self.fidelities)):
            ns, fids = base._get_next_iteration(stub, it)
            ns = [max(1, int(round(n * float(self.cfg["n_configs_scale"])))) for n in ns]
            sched.append((ns, [float(f) for f in fids]))
        return sched

    def bracket(self, b: int):
        ns, fids = self._schedule[b % len(self._schedule)]
        ns = list(ns)
        if b == 0:
            ns[0] = max(ns[0], len(self._initial()))
        return ns, fids

    def _max_pop_sizes(self):
        m = {}
        for b in range(len(self._schedule)):
            ns, fids = self.bracket(b)
            for n, f in zip(ns, fids):
                m[f] = max(m.get(f, 0), n)
        return m

    def _initial(self):
        if self._init_cache is None:
            self._init_cache = self._initial_uncached()
        return [copy.deepcopy(g) for g in self._init_cache]

    def _initial_uncached(self):
        c = self.cfg
        pop = self._initial_grid()
        if getattr(self, "pnas", None) is not None:
            return self._pnas_initial(pop)
        if self.strat is not None:
            from ga_search.modehb import strata as S
            names = S.names(self.strat)   # seeds round-robin over the categories (projected widths)
            pop = [S.fit_to_stratum(g, names[j % len(names)], self.strat, self._rep,
                                    np.random.default_rng([int(c["seed"]), j, 0x57A7]), count=self.param_fn)
                   for j, g in enumerate(pop)]
        ext = self._external_seeds()          # [] unless search.external_seed has a converted encoder
        if self.lemonade is None:
            if not ext:
                return pop
            hs = {arch_hash(g) for g in ext}
            return ext + [g for g in pop if arch_hash(g) not in hs]
        return self._lemonade_initial(pop, ext)

    def _external_seeds(self):
        """External pretrained encoder + its function-preserving growth grid (external_seed.py),
        first in the initial population (calibration representatives of their categories). Seeds are
        only ``repair``-ed: a variant the repair, the param cap or the memory budget would change is
        dropped (warning), never projected/shrunk (its blocks are loaded, not re-fitted)."""
        es = self.cfg.get("external_seed")
        if not es or not es.get("layers"):
            return []
        import warnings
        from ga_search.modehb import external_seed as ES
        from ga_search.modehb import strata as S
        spp = int((self.strat or {}).get("spifil_seeds_per_image", 100))
        n_cls = int(self.cfg.get("n_classes") or 0)
        pools = self.cfg.get("spifil_pool_sizes")
        out, seen = [], set()
        for g0 in ES.seed_genomes(es, int(self.cfg["seed"]), n_classes=n_cls,
                                  choices=self.repair_kw["spifil_images_choices"], seeds_per_image=spp,
                                  pool_sizes=pools, max_depth=int(self.cfg["max_depth"])):
            g = repair(g0, **self.repair_kw)
            why = None
            if ES.active_layers(g) != ES.active_layers(g0):
                why = "changed by repair"
            elif self.strat is not None and self.cap_fn(g) > self.strat["param_cap"]:
                why = f"params {self.cap_fn(g)} > cap {self.strat['param_cap']}"
            elif self.strat is not None and self.strat.get("memory_budget_gb") and S.mem_estimate_gb(
                    g, int(self.cfg["image_size"]), n_classes=n_cls or None, seeds_per_image=spp,
                    pool_sizes=pools) > float(self.strat["memory_budget_gb"]):
                why = "memory estimate above budget"
            if why or arch_hash(g) in seen:
                if why:
                    warnings.warn(f"[external_seed] variant {g0['external']} skipped: {why}")
                continue
            seen.add(arch_hash(g))
            out.append(g)
        return out

    # ---- LEMONADE (opt-in, lemonade.py)
    def _morph_repair(self, g):
        """Repair of a morphed child: no projection to a category (it would rescale every width
        and destroy the reusable prefix), only the cap + the memory/SPiFiL repairs."""
        g = repair(g, **self.repair_kw)
        if self.strat is None:
            return g
        from ga_search.modehb import strata as S
        g = S.enforce_cap(g, self.strat, self._rep, self.rng if hasattr(self, "rng") else
                          np.random.default_rng([int(self.cfg["seed"]), 0x1E30]), count=self.param_fn)
        return self._rep(g)

    def _lemonade_initial(self, pop, ext=()):
        """[seeds] + pyramid seeds + the seeds' morphed descendants: the warm-start seeds go first
        (calibration representatives of their categories, trained before their descendants, which
        then inherit their blocks); the descendants (round-robin over the seeds) go last.
        ``ext``: external-seed genomes (external_seed.py), put before the YAML seeds (also parents)."""
        from ga_search.modehb import lemonade as LM
        lc = self.lemonade
        rep = self._morph_repair
        seeds, seen = [], set()
        for g in ext:
            if arch_hash(g) not in seen:
                seen.add(arch_hash(g))
                seeds.append(g)
        for spec in lc["seeds"]:
            g = LM.seed_genome(lc, int(self.cfg["seed"]), rep, n_layers=N_LAYERS, spec=spec)
            if arch_hash(g) not in seen:
                seen.add(arch_hash(g))
                seeds.append(g)
        rng = np.random.default_rng([int(self.cfg["seed"]), 0x1E30])
        n = int(lc["n_descendants"])
        kids = []
        for j, sd in enumerate(seeds):
            share = n // len(seeds) + (1 if j < n % len(seeds) else 0)
            off = j if j < len(ext) else 0         # external seeds: operators rotate across the seeds
            for child, info in LM.descendants(sd, share, rng, lc, rep, max_depth=int(self.cfg["max_depth"]),
                                              op_offset=off):
                if arch_hash(child) not in seen:
                    seen.add(arch_hash(child))
                    self._lineage[arch_hash(child)] = (sd, info)
                    kids.append(child)
        return seeds + [g for g in pop if arch_hash(g) not in seen] + kids

    def _inherit_for(self, parent, child, fid, info=None):
        """``ctx["inherit"]`` for a morphed child (None: nothing reusable / parent not trained).
        ``lemonade.function_preserving`` (opt-in) + morphism ``info``: Net2Net-style inheritance
        (external_seed.grow_blocks), ``layer_map`` = child layer -> parent layer / identity / fresh."""
        from ga_search.modehb import lemonade as LM
        if self.lemonade is None or not self.lemonade.get("inherit"):
            return None
        if self.lemonade.get("function_preserving") and info is not None:
            from ga_search.modehb import external_seed as ES
            lm = ES.morph_layer_map(parent, child, info)
            job = LM.pick_parent_checkpoint(self._find(parent, role="search"), fid,
                                            self.lemonade["inherit_fidelity"])
            spec = LM.inherit_spec(job, max(1, sum(isinstance(x, int) for x in lm)))
            if spec is not None:
                spec.update(layer_map=lm, parent_layers=ES.active_layers(parent))
            return spec
        n = LM.reusable_prefix(parent, child)
        if n < 1:
            return None
        job = LM.pick_parent_checkpoint(self._find(parent, role="search"), fid,
                                        self.lemonade["inherit_fidelity"])
        return LM.inherit_spec(job, n)

    def _morph_child(self, parent_member, fid, tstrat, kind):
        """(vector, genome, op) of a morphism of ``parent_member``, or None."""
        from ga_search.modehb import lemonade as LM
        child, info = LM.random_morph(parent_member["genome"], self.rng, self.lemonade, self._morph_repair,
                                      max_depth=int(self.cfg["max_depth"]))
        if child is None:
            return None
        child["init"] = parent_member["genome"]["init"]
        self._pending_inherit = self._inherit_for(parent_member["genome"], child, fid, info)
        op = {"type": "morph", "via": kind, "target_stratum": tstrat, "parent_stratum": parent_member.get("stratum"),
              "parent": parent_member["cand_id"], **info,
              "inherit": ({k: v for k, v in self._pending_inherit.items() if k != "effective_specs"}
                          if self._pending_inherit else None)}
        return encode(child, **self.enc_kw).tolist(), child, op

    def _initial_grid(self):
        c = self.cfg
        return initial_population(int(c["seed"]), tuple(c["initial_channels"]), tuple(c["initial_depths"]),
                                  ssl=c["initial_ssl"], reg=c["initial_reg"], pooling=c["initial_pooling"],
                                  pyramid_growth=float(c.get("initial_pyramid_growth", 1.0)),
                                  twins=self.random_baseline, size=c.get("initial_size"),
                                  sample_operators=c.get("initial_operators", "plain") == "sampled",
                                  **self.repair_kw)

    # ---- state
    def _fresh_state(self):
        return {"version": 1, "config": self.cfg, "jobs": [], "subpops": {str(f): [] for f in self.fidelities},
                "parent_counter": {str(f): 0 for f in self.fidelities}, "bracket": 0,
                "stage": 1, "n_candidates": 0, "gpu_hours": 0.0, "done": False, "brackets_log": [],
                "strata_rr": 0, "promo_counts": {str(f): {} for f in self.fidelities}, "calibration": {}}

    def _save(self):
        if len(self.state["jobs"]) < len(getattr(self, "_replay", [])):
            return   # replaying a restart: never shrink the logged state on disk
        self.state["updated"] = time.time()
        atomic_json(self.state_path, self.state)

    def run(self, resume: bool = False) -> dict:
        logged: List[dict] = []
        if resume and self.state_path.exists():
            old = json.loads(self.state_path.read_text())
            logged = old.get("jobs", [])
        elif self.state_path.exists() and not resume:
            raise FileExistsError(f"{self.state_path} exists; pass resume=True (never overwritten)")
        self._replay = logged
        self.state = self._fresh_state()
        if self.pnas is not None:
            self.state.update({"pnas_hv_history": [], "pnas_pareto_history": [], "pnas_grown": {},
                               "pnas_mut_rr": 0, "phase0": {}})
        self.rng = np.random.default_rng(int(self.cfg["seed"]))
        self._t0 = time.time()
        self.init_queue = self._initial()
        if self.pnas is not None and self.pnas["phase0"].get("enabled"):
            self._bracket_job0 = 0
            self._pnas_phase0()
        if self.strat is not None and self.strat.get("calibration"):
            self._bracket_job0 = 0
            self._calibrate()
        b = 0
        while not self._budget_exhausted(b):
            self.state["bracket"] = b
            self.state["stage"] = 1 if b < int(self.cfg["stage1_brackets"]) else 2
            self._bracket_job0 = len(self.state["jobs"])
            self._run_bracket(b)
            self._log_bracket(b)
            b += 1
            self._save()
        self.state["done"] = True
        self._save()
        return self.state

    def _budget_exhausted(self, b):
        c = self.cfg
        if c["max_brackets"] is not None and b >= int(c["max_brackets"]):
            return True
        if c["max_evaluations"] is not None and len(self.state["jobs"]) >= int(c["max_evaluations"]):
            return True
        if c["gpu_hours_budget"] is not None and self.state["gpu_hours"] >= float(c["gpu_hours_budget"]):
            return True
        return False

    # ---- bracket
    def _run_bracket(self, b):
        ns, fids = self.bracket(b)
        first_hb = b < len(self.fidelities)
        rung_records: List[List[dict]] = []
        for k, (n, fid) in enumerate(zip(ns, fids)):
            recs = []
            if k > 0 and first_hb:
                prev = [r for r in rung_records[-1] if r["status"] == "ok"]   # failures never promoted
                F = np.array([r["F"] for r in prev]) if prev else np.zeros((0, self.n_obj))
                for idx, explore in self._promote(prev, F, min(n, len(prev)), fids[k - 1], b):
                    src = prev[idx]
                    rec = self._evaluate(src["genome"], fid, parent_id=src["cand_id"], generation=b,
                                         rung=k, cand_id=src["cand_id"], vector=src["vector"])
                    rec["op"] = {"type": "promotion", "exploratory": bool(explore),
                                 "from_fidelity": fids[k - 1], "stratum": src.get("stratum")}
                    self._insert(fid, rec, target=None)
                    recs.append(rec)
                    self._log_live(b)
                    if fid == self.fidelities[-1]:
                        self._extend(rec, b)
            else:
                for _ in range(n):
                    if self._budget_exhausted(b) and recs:
                        break
                    vec, genome, target, parent_id = self._acquire(fid, fids, k, ns)
                    inherit, self._pending_inherit = self._pending_inherit, None
                    rec = self._evaluate(genome, fid, parent_id=parent_id, generation=b, rung=k,
                                         vector=vec, **({"inherit": inherit} if inherit else {}))
                    if self._last_op is not None:
                        rec["op"] = {**self._last_op, "child_stratum": rec.get("stratum")}
                    self._insert(fid, rec, target=target)
                    recs.append(rec)
                    self._log_live(b)
                    if fid == self.fidelities[-1]:
                        self._extend(rec, b)
            rung_records.append(recs)
            self._log_live(b)

    def _acquire(self, fid, fids, k, ns):
        """Next config for rung ``k`` at ``fid``: seeds / random fill, else DE (DEHB)."""
        sp = self.state["subpops"][str(fid)]
        self._last_op = None
        self._pending_inherit = None
        if fid == self.fidelities[0] and self.init_queue:
            g = self.init_queue.pop(0)
            self._last_op = {"type": "external_seed" if g.get("external") else "seed",
                             "target_stratum": self._stratum_est(g),
                             **({"external": g["external"]} if g.get("external") else {})}
            lin = self._lineage.get(arch_hash(g)) if self.lemonade is not None else None
            if lin is not None:                    # LEMONADE: morphed descendant of the warm-start seed
                self._pending_inherit = self._inherit_for(lin[0], g, fid, lin[1])
                self._last_op.update({"morphism": lin[1].get("morphism"), "n_reuse": lin[1].get("n_reuse"),
                                      "inherited": self._pending_inherit is not None})
            if self.pnas is not None:              # PNAS-CS: GA-checkpoint seeds import their blocks 1:1
                info = g.get("pnas") or {}
                self._pending_inherit = dict(info["import"]) if info.get("import") else None
                self._last_op.update({"origin": info.get("origin"), "family": info.get("family")})
                return self._encode(g).tolist(), g, None, None
            return encode(g, **self.enc_kw).tolist(), g, None, None
        if self.pnas is not None:
            return self._pnas_acquire(fid, fids, k, ns)
        want = self._draw_init()
        stage = 2 if self.rng.random() < float(self.cfg["epsilon"]) else self.state["stage"]
        glob = self._global_members()
        tstrat = self._next_stratum()          # None unless stratified: round-robin category
        if (len(sp) < self._max_pop[fid] and fid == self.fidelities[0]) or not glob:
            if (self.lemonade is not None and glob and self.lemonade.get("morph_random_fill")
                    and self.rng.random() < float(self.lemonade["morph_prob"])):
                gl = [m for m in glob if m.get("stratum") == tstrat] or glob
                front = nondominated_fronts(np.array([m["F"] for m in gl]))[0]
                m = self._morph_child(gl[int(self.rng.choice(front))], fid, tstrat, "random_fill")
                if m is not None:
                    self._last_op = m[2]
                    return m[0], m[1], None, m[2]["parent"]
            vec = self._random_vector(stage)
            g = self._decode(vec, want, stratum=tstrat, force=True)
            self._last_op = {"type": "random", "target_stratum": tstrat}
            return self._vec_of(vec, g), g, None, None
        if sp and len(sp) >= self._max_pop[fid]:
            target = self._next_target(fid, want, tstrat)
            tgt = sp[target]
        else:   # higher-fidelity sub-population still filling: target from global population
            target = None
            gl = [m for m in glob if m.get("stratum") == tstrat] or glob if tstrat else glob
            F = np.array([m["F"] for m in gl])
            tgt = gl[select_promotions(F, 1, self.cfg["tie_breaker"])[0]]
        if self.lemonade is not None and self.rng.random() < float(self.lemonade["morph_prob"]):
            m = self._morph_child(tgt, fid, tstrat, "de_target")
            if m is not None:              # LEMONADE: child = DE target + one network morphism
                self._last_op = m[2]
                return m[0], m[1], target, tgt["cand_id"]
        lower = fids[k - 1] if k > 0 else fid
        cross = tstrat is None or self.rng.random() < float(self.strat["cross_category_prob"])
        pool = self._parent_pool(lower, ns[k], None if cross else tgt.get("stratum"))
        self._last_op = {"type": "de", "target_stratum": tstrat, "parent_stratum": tgt.get("stratum"),
                         "pool": "cross" if (cross and tstrat is not None) else "same"}
        mask = gene_mask(tgt["genome"], stage)
        tvec = np.asarray(tgt["vector"], float)
        if len(pool) < 3:   # DEHB fallback: rand/1 mutants from the global population
            gpop = self._global_pop()
            pool = pool + [self._rand1(gpop, tvec) for _ in range(3 - len(pool) + 1)]
        mutant = self._rand1(pool, tvec)
        child = self._boundary(self._crossover(tvec, mutant, mask))
        g = self._decode(child, want, stratum=tstrat)
        return self._vec_of(child, g), g, target, tgt["cand_id"]

    def _decode(self, vec, init, stratum=None, force=False):
        if self.pnas is not None:
            return self._pnas_decode(vec, init, stratum)
        g = decode(vec, int(self.cfg["seed"]), **self.repair_kw)
        g["init"] = init
        g = repair(g, **self.repair_kw)
        if self.strat is None:
            return g
        from ga_search.modehb import strata as S
        # offspring keep their own category unless projected into the round-robin target
        if stratum is not None and (force or self.rng.random() < float(self.strat["projection_prob"])):
            g = S.fit_to_stratum(g, stratum, self.strat, self._rep, self.rng, count=self._count)
        g = S.enforce_cap(g, self.strat, self._rep, self.rng, count=self.param_fn)   # cap: raw (conservative)
        return self._rep(g)

    def _vec_of(self, vec, genome):
        """Stored DE vector: the projected genome's encoding when stratified."""
        if self.strat is None or self.pnas is not None:   # PNAS-CS: the family vector (base + alpha, beta)
            return list(np.asarray(vec, float))
        return encode(genome, **self.enc_kw).tolist()

    # ---- parameter strata (opt-in)
    def _rep(self, g):
        """repair (+ GPU-memory repair when ``stratified.memory_budget_gb`` is set)."""
        g = repair(g, **self.repair_kw)
        mb = (self.strat or {}).get("memory_budget_gb")
        from ga_search.modehb import strata as S
        n_cls = int(self.cfg.get("n_classes") or 0)
        spp = int((self.strat or {}).get("spifil_seeds_per_image", 100))
        pools = self.cfg.get("spifil_pool_sizes")

        def _images(x):
            if not (n_cls and x.get("init") == "spifil"):
                return x
            kern = [k for k in (self.repair_kw.get("search_space") or {}).get("kernel", KERNELS) if int(k) > 1]
            return S.spifil_images_repair(x, n_cls, self.repair_kw["spifil_images_choices"], spp,
                                          pool_sizes=pools, repair=lambda y: repair(y, **self.repair_kw),
                                          kernels=kern or (3,))
        # images FIRST (only raises k until N > D), THEN memory: SPiFiL's N x N Fisher matrix grows
        # with k^2 (aid, 30 classes x 32 imgs: ~150 GB), mem_repair lowers k / D back into the budget.
        g = _images(g)
        if mb:
            g = S.mem_repair(g, float(mb), lambda x: repair(x, **self.repair_kw),
                             image_size=int(self.cfg["image_size"]), min_spatial=int(self.cfg["min_spatial"]),
                             n_classes=n_cls or None, seeds_per_image=spp, pool_sizes=pools,
                             spifil_choices=self.repair_kw["spifil_images_choices"],
                             scale=float((self.strat or {}).get("mem_scale", 1.0) or 1.0))
            g = _images(g)                            # no-op unless a width repair broke N > D
        return g

    def _count(self, genome):
        """Structural estimate corrected by the median estimate/measured ratio of finished
        jobs (SPiFiL may build fewer channels than the genome asks)."""
        r = [j["params_est"] / j["result"]["params"] for j in getattr(self, "state", {}).get("jobs", [])
             if j.get("params_est") and j["status"] == "ok" and j["result"].get("params")]
        ratio = float(np.median(r)) if len(r) >= 5 else 1.0
        return int(self.param_fn(genome) / max(ratio, 1e-6))

    def _stratum_est(self, genome):
        if self.strat is None:
            return None
        from ga_search.modehb import strata as S
        return S.stratum_of(self._count(genome), self.strat)

    def _next_stratum(self):
        if self.strat is None:
            return None
        from ga_search.modehb import strata as S
        names = S.names(self.strat)
        k = int(self.state.get("strata_rr", 0))
        self.state["strata_rr"] = k + 1
        if self.strat.get("band_quota"):      # v4: most under-represented band of THIS bracket
            counts = {}
            for j in self.state["jobs"][getattr(self, "_bracket_job0", 0):]:
                op = j.get("op") or {}
                if j["role"] == "search" and j["status"] == "ok" and op.get("type") not in ("promotion", "extension"):
                    counts[j.get("stratum")] = counts.get(j.get("stratum"), 0) + 1
            return S.next_band(counts, self.strat)
        return names[k % len(names)]

    def _quality_path(self, cand_id):
        return {float(j["fidelity"]): float(j["result"].get("quality", 0.0)) for j in self.state["jobs"]
                if j["cand_id"] == cand_id and j["role"] == "search" and j["status"] == "ok"}

    def _promote(self, prev, F, n, fid_prev, b):
        """[(index into prev, exploratory?)]. Stratified: protected quotas (strata.py)."""
        sc = self.strat
        pb = None if sc is None else sc.get("protected_brackets")
        if sc is not None and sc.get("promotion") == "per_band":   # v4: Pareto inside each band, always
            from ga_search.modehb import strata as S
            from ga_search.modehb.strata import learning_slope
            cands = [{"F": r["F"], "stratum": r.get("stratum") or S.names(sc)[0],
                      "slope": learning_slope(self._quality_path(r["cand_id"]))} for r in prev]
            counts = self.state.setdefault("promo_counts", {}).setdefault(str(fid_prev), {})
            idx = S.band_promotions(cands, n, sc, lambda Fs, k: select_promotions(Fs, k, self.cfg["tie_breaker"]))
            for i in idx:
                counts[cands[i]["stratum"]] = counts.get(cands[i]["stratum"], 0) + 1
            return [(i, j >= n) for j, i in enumerate(idx)]
        if sc is None or (pb is not None and b >= int(pb)):
            return [(i, False) for i in select_promotions(F, n, self.cfg["tie_breaker"])]
        from ga_search.modehb import strata as S
        from ga_search.modehb.strata import learning_slope
        cands = [{"F": r["F"], "stratum": r.get("stratum") or "tiny",
                  "slope": learning_slope(self._quality_path(r["cand_id"]))} for r in prev]
        counts = self.state.setdefault("promo_counts", {}).setdefault(str(fid_prev), {})
        idx = S.stratified_promotions(cands, n, sc, lambda Fs, k: select_promotions(Fs, k, self.cfg["tie_breaker"]),
                                      counts=counts)
        for i in idx:
            counts[cands[i]["stratum"]] = counts.get(cands[i]["stratum"], 0) + 1
        return [(i, j >= n) for j, i in enumerate(idx)]

    def _extend(self, rec, b, budget=None):
        """Adaptive extra training past the max fidelity while quality still improves.
        Extension jobs (fidelity > max) never enter sub-populations nor the main HV."""
        sc = self.strat
        if sc is None or rec["status"] != "ok" or not sc.get("extension_fidelities"):
            return
        n_ext = sum(1 for j in self.state["jobs"][self._bracket_job0:]
                    if (j.get("op") or {}).get("type") == "extension")
        cap = int(sc["extension_max_per_bracket"]) if budget is None else budget
        if n_ext >= cap:
            return
        cur = rec
        for f in sorted(float(x) for x in sc["extension_fidelities"]):
            if f <= float(cur["fidelity"]) or self._budget_exhausted(b):
                break
            path = self._quality_path(cur["cand_id"])
            lower = [q for fl, q in path.items() if fl < float(cur["fidelity"])]
            gain = float(cur["result"].get("quality", 0.0)) - (max(lower) if lower else 0.0)
            if gain < float(sc["extension_min_gain"]):
                break
            nxt = self._evaluate(cur["genome"], f, parent_id=cur["cand_id"], generation=b, rung=-1,
                                 cand_id=cur["cand_id"], vector=cur["vector"])
            nxt["op"] = {"type": "extension", "from_fidelity": cur["fidelity"], "gain": gain}
            if nxt["status"] != "ok":
                break
            cur = nxt

    def _calibrate(self):
        """One representative per category along every fidelity (+ extensions): learning speed
        and saturation per category. Jobs are normal search jobs (generation -1)."""
        from ga_search.modehb import strata as S
        names = S.names(self.strat)
        reps = {}
        for g in self.init_queue:
            s = self._stratum_est(g)
            if s is not None and s not in reps:
                reps[s] = g
        for s in names:
            g = reps.get(s)
            if g is None:
                continue
            self._bracket_job0 = len(self.state["jobs"])   # extension budget per representative
            cand = None
            for k, f in enumerate(self.fidelities):
                imp = ((g.get("pnas") or {}).get("import") if self.pnas is not None and k == 0 else None)
                rec = self._evaluate(g, f, parent_id=None, generation=-1, rung=k, cand_id=cand,
                                     **({"inherit": dict(imp)} if imp else {}))
                rec["op"] = {"type": "calibration", "target_stratum": s,
                             **({"external": g["external"]} if g.get("external") else {})}
                cand = rec["cand_id"]
                self._insert(f, rec, target=None)
                self._log_live(-1)
                if rec["status"] != "ok":
                    break
                if f == self.fidelities[-1]:
                    self._extend(rec, -1, budget=len(self.strat["extension_fidelities"]))
            path = self._quality_path(cand) if cand else {}
            fl = sorted(path)
            sat = next((fl[i] for i in range(len(fl) - 1)
                        if path[fl[i + 1]] - path[fl[i]] < float(self.strat["extension_min_gain"])), None)
            measured = [j for j in self.state["jobs"] if j["cand_id"] == cand and j["status"] == "ok"]
            self.state["calibration"][s] = {
                "cand_id": cand, "quality_by_epochs": {str(int(k)): v for k, v in path.items()},
                "params": measured[-1]["result"].get("params") if measured else None,
                "measured_stratum": measured[-1].get("stratum") if measured else None,
                "saturation_epochs": sat}
            self._log_strata_now(-1, tables=True)
        self._save()

    def _random_vector(self, stage):
        """Uniform sample of the genes unlocked at ``stage``; the rest from a random seed genome."""
        seeds = self._initial()
        base = self._encode(seeds[int(self.rng.integers(len(seeds)))])
        mask = self._gene_mask(seeds[0], stage)
        base[mask] = self.rng.uniform(size=int(mask.sum()))
        return base

    def _global_members(self):
        return [m for f in self.fidelities for m in self.state["subpops"][str(f)] if m["status"] == "ok"]

    def _draw_init(self):
        if not self.random_baseline:   # SPiFiL-only experiment: no random offspring
            return "spifil"
        return "spifil" if self.rng.random() < float(self.cfg["spifil_fraction"]) else "random"

    def _next_target(self, fid, want, stratum=None):
        sp = self.state["subpops"][str(fid)]
        idx = [i for i, m in enumerate(sp) if m["genome"]["init"] == want] or list(range(len(sp)))
        if stratum is not None:   # stratified: mutate a member of the target category
            idx = [i for i in idx if sp[i].get("stratum") == stratum] or idx
        c = self.state["parent_counter"][str(fid)]
        self.state["parent_counter"][str(fid)] = c + 1
        return idx[c % len(idx)]

    def _parent_pool(self, fid, n, stratum=None):
        sp = [m for m in self.state["subpops"][str(fid)] if m["status"] == "ok"]
        if stratum is not None:   # within-category DE (falls back to all when < 3 parents)
            same = [m for m in sp if m.get("stratum") == stratum]
            sp = same if len(same) >= 3 else sp
        if not sp:
            return []
        F = np.array([m["F"] for m in sp])
        if (self.strat or {}).get("parent_select") == "quality":   # v4: no FLOPs pressure on DE parents
            return [np.asarray(sp[i]["vector"], float) for i in np.argsort(F[:, 0], kind="stable")[:min(n, len(sp))]]
        return [np.asarray(sp[i]["vector"], float) for i in select_promotions(F, min(n, len(sp)),
                                                                              self.cfg["tie_breaker"])]

    def _global_pop(self):
        pop = [np.asarray(m["vector"], float) for f in self.fidelities for m in self.state["subpops"][str(f)]]
        while len(pop) < 3:
            pop.append(self.rng.uniform(size=self.dim))
        return pop

    def _rand1(self, pool, target):
        pool = [p for p in pool if not np.array_equal(p, target)] or list(pool)
        while len(pool) < 3:
            pool.append(self.rng.uniform(size=len(target)))
        r1, r2, r3 = (pool[i] for i in self.rng.choice(len(pool), 3, replace=False))
        return r1 + float(self.cfg["mutation_factor"]) * (r2 - r3)

    def _crossover(self, target, mutant, mask):
        cross = self.rng.random(len(target)) < float(self.cfg["crossover_prob"])
        cross &= mask
        if not cross.any() and mask.any():
            cross[self.rng.choice(np.flatnonzero(mask))] = True
        return np.where(cross, mutant, target)

    def _boundary(self, v):
        bad = (v < 0) | (v > 1)
        if self.cfg["boundary_fix_type"] == "random":
            v = v.copy()
            v[bad] = self.rng.uniform(size=int(bad.sum()))
            return v
        return np.clip(v, 0, 1)

    # ---- evaluation (+ replay + twins)
    def _new_id(self):
        self.state["n_candidates"] += 1
        return f"c{self.state['n_candidates']:05d}"

    def _find(self, genome, fidelity=None, role=None):
        h = arch_hash(genome)
        out = [j for j in self.state["jobs"] if j["arch_hash"] == h and j["genome"]["seed"] == genome["seed"]
               and (fidelity is None or j["fidelity"] == fidelity) and (role is None or j["role"] == role)]
        return out

    def _cap_project(self, genome, vector):
        """v4 hard cap BEFORE training: a genome whose ``cap_fn`` count is above ``param_cap`` is shrunk
        by ``strata.enforce_cap`` (top band, then widths x0.9) with an rng drawn from the genome, so a
        restart replays it; a job logged with the raw genome (search started before this fix, e.g.
        remote_sensing c00015) replays raw. Still above the cap: rejected untrained in ``_call``."""
        from ga_search.modehb import strata as S
        if self.cap_fn(genome) <= self.strat["param_cap"]:
            return genome, vector
        k, log = len(self.state["jobs"]), getattr(self, "_replay", [])
        if k < len(log) and log[k]["arch_hash"] == arch_hash(genome):
            return genome, vector
        rng = np.random.default_rng([int(self.cfg["seed"]), int(arch_hash(genome)[:8], 16), 0xCA9])
        g = S.enforce_cap(genome, self.strat, self._rep, rng, count=self.cap_fn)
        print(f"[modehb] cap: {self.cap_fn(genome)} -> {self.cap_fn(g)} params (built count), reprojected", flush=True)
        return g, None

    def _evaluate(self, genome, fid, *, parent_id, generation, rung, cand_id=None, vector=None,
                  role="search", twin_of=None, twin_source=None, inherit=None, extra_ctx=None):
        genome = repair(genome, **self.repair_kw)
        if getattr(self, "_cap_built", False) and inherit is None:   # v4: hard cap before training
            genome, vector = self._cap_project(genome, vector)
        cached = self._find(genome, fid, role=role)
        lower = [j for j in self._find(genome, role=role) if j["fidelity"] < fid and j["status"] == "ok"
                 and j["result"].get("checkpoint")]
        resume = max(lower, key=lambda j: j["fidelity"]) if lower else None
        if cand_id is None:
            cand_id = resume["cand_id"] if resume else (cached[0]["cand_id"] if cached else self._new_id())
        ctx = {"cand_id": cand_id, "parent_id": parent_id, "generation": generation, "bracket": generation,
               "rung": rung, "role": role, "twin_of": twin_of, "resume": resume, "twin_source": twin_source,
               "front_rank": lambda result, _f=fid: self.rank_of(result, _f)}
        if extra_ctx:
            ctx.update(extra_ctx)            # PNAS-CS Phase 0 (ae_mode)
        if inherit and resume is None and not cached and role == "search":
            ctx["inherit"] = inherit           # LEMONADE: reuse the parent's trained leading blocks
        if cached:
            result = {k: v for k, v in cached[0]["result"].items()}
            result["cached_from"] = cached[0]["job"]
            cost_free = True
        else:
            result = self._call(genome, fid, ctx)
            cost_free = False
        stratum, p_est = None, None
        if self.strat is not None:
            from ga_search.modehb import strata as S
            p_est = self.param_fn(genome)
            p_meas = result.get("params") if result.get("status", "ok") == "ok" or result.get("not_trained") else None
            stratum = S.stratum_of(p_meas if p_meas is not None else p_est, self.strat)
            if result.get("status", "ok") == "ok" and p_meas is not None and int(p_meas) > self.strat["param_cap"]:
                result = {**result, "status": "failed", "infeasible": "param_cap",
                          "error": f"params {p_meas} > cap {self.strat['param_cap']}"}
        F = objectives(result, self.cfg["use_latency"], self.cfg["objective_bounds"])
        rec = {"job": len(self.state["jobs"]), "cand_id": cand_id, "parent_id": parent_id,
               "generation": generation, "rung": rung, "fidelity": fid, "role": role, "twin_of": twin_of,
               "genome": genome, "arch_hash": arch_hash(genome),
               "twin_key": arch_hash(genome, include_init=False),
               "vector": list(vector) if vector is not None else self._encode(genome).tolist(),
               "status": result.get("status", "ok"), "F": F, "result": result,
               "resumed_from": resume["job"] if resume else None, "cached": cost_free,
               "time": time.time(),
               **({"inherit": {k: v for k, v in ctx["inherit"].items() if k != "effective_specs"}}
                  if ctx.get("inherit") else {}),
               **({"stratum": stratum, "params_est": p_est} if self.strat is not None else {})}
        self.state["jobs"].append(rec)
        if not cost_free:
            self.state["gpu_hours"] += float(result.get("gpu_hours", 0.0) or 0.0)
        self._save()
        if role == "search" and genome["init"] == "spifil" and fid == self.fidelities[-1] and rec["status"] == "ok" \
                and (self.random_baseline or (self.pnas is not None and self.pnas.get("random_twins"))):
            self._evaluate_twin(rec)
        return rec

    def _call(self, genome, fid, ctx):
        k = len(self.state["jobs"])
        if k < len(self._replay):
            old = self._replay[k]
            if old["arch_hash"] != arch_hash(genome) or float(old["fidelity"]) != float(fid) \
                    or old["role"] != ctx["role"]:
                raise ReplayMismatch(f"job {k}: logged {old['arch_hash']}@{old['fidelity']} "
                                     f"vs {arch_hash(genome)}@{fid}")
            return old["result"]
        if getattr(self, "_cap_built", False) and self.cap_fn(genome) > self.strat["param_cap"]:
            n = int(self.cap_fn(genome))              # v4: never trained above the cap
            return {"status": "failed", "infeasible": "param_cap", "not_trained": True, "params": n,
                    "gpu_hours": 0.0, "error": f"params {n} > cap {self.strat['param_cap']} (built count, before training)"}
        ladder = list((getattr(self, "strat", None) or {}).get("oom_fallback") or [])
        if ladder:
            return self._call_oom_ladder(genome, fid, ctx, ladder)
        attempt = 0
        while True:
            try:
                result = dict(self.evaluate_fn(genome, fid, ctx))
                break
            except Exception as exc:  # an evaluation failure is an infeasible candidate, not a crash
                if _fatal_cuda(exc):  # poisoned CUDA context: crash unrecorded, resume re-runs the job
                    raise
                if _is_oom(exc) and attempt < OOM_RETRIES:
                    # shared GPU: an OOM is usually other processes' transient peak, not the candidate.
                    # Wait and re-run the same job instead of logging a failure (zero-failure rule).
                    attempt += 1
                    print(f"[modehb] OOM on job {k} ({type(exc).__name__}); retry {attempt}/{OOM_RETRIES} "
                          f"in {OOM_RETRY_SLEEP_S}s", flush=True)
                    time.sleep(OOM_RETRY_SLEEP_S)
                    continue
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                if attempt:
                    result["oom_retries"] = attempt
                break
        result.setdefault("status", "ok")
        return result

    def _call_oom_ladder(self, genome, fid, ctx, ladder):
        """v4 OOM handling: ``oom_fast_retries`` transient retries (shared GPU), then the
        ``oom_fallback`` levels in order (ctx["mem_fallback"] = level, e.g. bf16, then bf16 + batch/2);
        every attempt is recorded in ``result["oom_attempts"]`` and the level that worked in
        ``result["mem_fallback"]``. Still OOM after the ladder: ``status=failed``,
        ``infeasible="oom"``, ``penalize=False`` (not a bad model: never a sub-population member, the
        band stays under-represented and is re-targeted)."""
        sc = self.strat or {}
        fast = int(sc.get("oom_fast_retries") if sc.get("oom_fast_retries") is not None else 1)
        sleep_s = float(sc.get("oom_retry_sleep_s") if sc.get("oom_retry_sleep_s") is not None else 60)
        levels = [None] * (1 + fast) + [dict(l) for l in ladder]
        attempts = []
        for lvl in levels:
            c = dict(ctx)
            if lvl is not None:
                c["mem_fallback"] = lvl
            try:
                result = dict(self.evaluate_fn(genome, fid, c))
                result.setdefault("status", "ok")
                if attempts:
                    result["oom_attempts"] = attempts
                if lvl is not None:
                    result["mem_fallback"] = lvl
                return result
            except Exception as exc:
                if _fatal_cuda(exc):
                    raise
                if not _is_oom(exc):
                    return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            **({"oom_attempts": attempts} if attempts else {})}
                attempts.append({"mem_fallback": lvl, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                print(f"[modehb] OOM on {ctx.get('cand_id')} (level {lvl}); next level", flush=True)
                if lvl is None and sleep_s > 0:
                    time.sleep(sleep_s)
        return {"status": "failed", "infeasible": "oom", "penalize": False, "oom_attempts": attempts,
                "error": attempts[-1]["error"] if attempts else "OOM"}

    def _evaluate_twin(self, rec):
        """Random twin along the SPiFiL candidate's fidelity path, same seed and budgets."""
        twin = make_twin(rec["genome"])
        path = sorted({j["fidelity"] for j in self._find(rec["genome"], role="search")
                       if j["status"] == "ok"})
        twin_id = None
        for f in path:
            done = [j for j in self._find(twin, f, role="twin") if j["twin_of"] == rec["cand_id"]]
            if done:
                twin_id = done[0]["cand_id"]
                continue
            src = self._find(rec["genome"], f, role="search")[0]
            t = self._evaluate(twin, f, parent_id=rec["cand_id"], generation=rec["generation"],
                               rung=src["rung"], cand_id=twin_id, role="twin",
                               twin_of=rec["cand_id"], twin_source=src)
            twin_id = t["cand_id"]

    def _insert(self, fid, rec, target):
        if rec["role"] != "search":
            return
        if fid not in self.fidelities:      # extension fidelities never enter sub-populations
            return
        sp = self.state["subpops"][str(fid)]
        member = {k: rec[k] for k in ("cand_id", "vector", "F", "status", "genome", "job")}
        if self.strat is not None:
            member["stratum"] = rec.get("stratum")
        if any(m["cand_id"] == rec["cand_id"] for m in sp):
            return
        if rec["status"] != "ok" and (self.strat or {}).get("skip_failed_in_subpop"):
            return   # v4: an OOM/infeasible big model is not a "bad" member (no FAIL_OBJ slot, no DE target)
        if len(sp) < self._max_pop[fid]:
            sp.append(member)
            return
        if rec["status"] != "ok":
            return
        if self.strat is not None:
            self._insert_stratified(sp, member)
            return
        if target is None:
            target = self._next_target(fid, rec["genome"]["init"])
        slot = mo_select(np.array([m["F"] for m in sp]), target, rec["F"], self.ref)
        if slot is not None:
            sp[slot] = member

    def _insert_stratified(self, sp, member):
        """Diversity-preserving replacement: a child competes only inside its category; an
        under-represented category (< fair share) evicts the worst of the most populated one,
        so no category (e.g. Tiny) can take over the sub-population."""
        from ga_search.modehb import strata as S
        names = S.names(self.strat)
        cnt = {s: sum(1 for m in sp if m.get("stratum") == s) for s in names}
        s = member.get("stratum")
        fair = len(sp) / len(names)
        if s is None:
            return
        if cnt.get(s, 0) < fair:
            big = max(names, key=lambda x: cnt.get(x, 0))
            idx = [i for i, m in enumerate(sp) if m.get("stratum") == big] or list(range(len(sp)))
            Fs = np.array([sp[i]["F"] for i in idx])
            last = nondominated_fronts(Fs)[-1]
            worst = last[int(np.argmin(hv_contributions(Fs[last], self.ref)))] if len(last) > 1 else last[0]
            sp[idx[int(worst)]] = member
            return
        idx = [i for i, m in enumerate(sp) if m.get("stratum") == s]
        c = self.state["parent_counter"].get("strata", 0)
        self.state["parent_counter"]["strata"] = c + 1
        t = c % len(idx)
        slot = mo_select(np.array([sp[i]["F"] for i in idx]), t, member["F"], self.ref)
        if slot is not None:
            sp[idx[slot]] = member

    # ---- reporting
    def rank_of(self, result, fidelity) -> int:
        """Non-dominated front rank (0 = Pareto) of ``result`` among ok jobs at ``fidelity``."""
        F = [j["F"] for j in self.state["jobs"] if j["status"] == "ok" and j["fidelity"] == fidelity]
        F.append(objectives(result, self.cfg["use_latency"], self.cfg["objective_bounds"]))
        return int(front_ranks(np.array(F))[-1])

    def front(self, fidelity=None, role=None, init=None):
        jobs = [j for j in self.state["jobs"] if j["status"] == "ok"
                and (fidelity is None or j["fidelity"] == fidelity)
                and (role is None or j["role"] == role) and (init is None or j["genome"]["init"] == init)]
        if not jobs:
            return []
        F = np.array([j["F"] for j in jobs])
        return [jobs[i] for i in nondominated_fronts(F)[0]]

    def _log_live(self, b):
        """Controller update after every evaluation (W&B), so the dashboard moves before the
        bracket ends. Does not touch brackets_log; logging never kills the search."""
        if getattr(self, "pnas", None) is not None:
            self._pnas_log(b)
        if self.logger is None or not hasattr(self.logger, "log_generation"):
            return
        start = self._bracket_job0
        pts = [{"objectives": j["F"], "fidelity": j["fidelity"], "id": j["cand_id"],
                "quality": j["result"].get("quality"), "flops": j["result"].get("flops"),
                "init": j["genome"]["init"], "ssl": j["genome"]["ssl"], "reg": j["genome"]["reg"],
                "depth": j["genome"]["depth"], "role": j["role"], "kappa": j["result"].get("fitness_kappa"),
                "params": j["result"].get("params"), "stratum": j.get("stratum")}
               for j in self.state["jobs"][start:] if j["status"] == "ok" and j["fidelity"] in self.fidelities]
        try:
            self.logger.log_generation(pts, self.fidelities[-1], b, self.state["gpu_hours"],
                                       time.time() - self._t0, len(self.state["jobs"]),
                                       meta={"live": 1, "stage": self.state.get("stage")},
                                       **({"dry": True} if self._replaying() else {}))
        except Exception as exc:
            print(f"[modehb] live log failed: {exc}")
        self._log_strata_now(b, tables=False)

    def _replaying(self):
        """True while a restarted search is re-walking its logged jobs (nothing new to log)."""
        return len(self.state["jobs"]) < len(getattr(self, "_replay", []))

    def _log_strata_now(self, b, tables=True):
        """Per-category panels: scalars after every evaluation, tables/plots when ``tables``."""
        if self.strat is None or self.logger is None or not hasattr(self.logger, "log_strata") or self._replaying():
            return
        try:
            rep = self.strata_report(b, record_history=tables)
            self.logger.log_strata(b, self.state["gpu_hours"], len(self.state["jobs"]), rep["scalars"],
                                   rep["tables"] if tables else {})
        except Exception as exc:
            print(f"[modehb] log_strata failed: {exc}")

    def _log_bracket(self, b):
        top = self.fidelities[-1]
        summary = {"bracket": b, "stage": self.state["stage"], "n_jobs": len(self.state["jobs"]),
                   "gpu_hours": self.state["gpu_hours"]}
        for f in self.fidelities:
            sp = [m for m in self.state["subpops"][str(f)] if m["status"] == "ok"]
            summary[f"hv/subpop_f{int(f)}"] = hypervolume([m["F"] for m in sp], self.ref) if sp else 0.0
        for init in INITS:
            fr = self.front(top, init=init)
            summary[f"hv/max_fidelity_{init}"] = hypervolume([j["F"] for j in fr], self.ref) if fr else 0.0
        self.state["brackets_log"].append(summary)
        if self.logger is not None and hasattr(self.logger, "log_generation"):
            start = self._bracket_job0
            points = [{"objectives": j["F"], "fidelity": j["fidelity"], "id": j["cand_id"],
                       "quality": j["result"].get("quality"), "flops": j["result"].get("flops"),
                       "init": j["genome"]["init"], "ssl": j["genome"]["ssl"], "reg": j["genome"]["reg"],
                       "depth": j["genome"]["depth"], "role": j["role"],
                       "spifil_images_per_class": j["genome"].get("spifil_images_per_class"),
                       "kappa": j["result"].get("fitness_kappa"), "params": j["result"].get("params"),
                       "stratum": j.get("stratum")}
                      for j in self.state["jobs"][start:] if j["status"] == "ok" and j["fidelity"] in self.fidelities]
            n_prom = sum(1 for j in self.state["jobs"][start:] if j["resumed_from"] is not None)
            try:   # wandb_logger.ControllerRun.log_generation (I10)
                self.logger.log_generation(points, top, b, self.state["gpu_hours"],
                                           time.time() - self._t0, len(self.state["jobs"]),
                                           meta={"n_promotions": n_prom, "stage": self.state["stage"], **summary},
                                           **({"dry": True} if self._replaying() else {}))
            except Exception as exc:  # logging must never kill the search
                print(f"[modehb] log_generation failed: {exc}")
        if self.strat is not None:
            rep = self.strata_report(b, record_history=False)
            summary.update({k: v for k, v in rep["scalars"].items() if k.startswith(("hv_strata/", "strata/"))})
            self._log_strata_now(b, tables=True)
        if self.pnas is not None:
            self._pnas_log(b, tables=True)

    def strata_report(self, b, record_history=True):
        """Per-category search statistics + W&B table rows (plots 1-12 of the stratified spec).
        Main/global HV only over max-fidelity jobs (comparable protocol); extension jobs are
        reported in the learning curves only."""
        from ga_search.modehb import strata as S
        names = S.names(self.strat)
        top = self.fidelities[-1]
        jobs = [j for j in self.state["jobs"] if j["role"] == "search" and j["status"] == "ok"]
        last_stratum = {}
        for j in jobs:
            last_stratum[j["cand_id"]] = j.get("stratum")
        fids_of = {}
        for j in jobs:
            fids_of.setdefault(j["cand_id"], set()).add(float(j["fidelity"]))
        sc, tables = {}, {}
        sc["hv_strata/global"] = hypervolume([j["F"] for j in jobs if j["fidelity"] == top], self.ref)
        for s in names:
            cs = [c for c, st in last_stratum.items() if st == s]
            sc[f"strata/{s}/n_candidates"] = len(cs)
            Ft = [j["F"] for j in jobs if j["fidelity"] == top and j.get("stratum") == s]
            sc[f"hv_strata/{s}"] = hypervolume(Ft, self.ref) if Ft else 0.0
            entered = [c for c in cs if self.fidelities[0] in fids_of[c]]
            sc[f"strata/{s}/survival_rate"] = (sum(1 for c in entered if top in fids_of[c]) / len(entered)
                                              if entered else 0.0)
            for f0, f1 in zip(self.fidelities, self.fidelities[1:]):
                at = [c for c in cs if f0 in fids_of[c]]
                sc[f"strata/{s}/promotion_rate_f{int(f0)}"] = (sum(1 for c in at if f1 in fids_of[c]) / len(at)
                                                              if at else 0.0)
            for f in self.fidelities:
                q = [j["result"].get("quality", 0.0) for j in jobs if j["fidelity"] == f and j.get("stratum") == s]
                sc[f"strata/{s}/n_evals_f{int(f)}"] = len(q)
                if q:
                    sc[f"strata/{s}/quality_max_f{int(f)}"] = float(max(q))
                    sc[f"strata/{s}/quality_mean_f{int(f)}"] = float(np.mean(q))
        ops = {}
        for j in self.state["jobs"]:
            op = j.get("op") or {}
            key = None
            if op.get("type") == "de":
                key = f"ops/{op.get('parent_stratum')}->{op.get('child_stratum')}"
                pk = f"ops/pool_{op.get('pool')}"
                ops[pk] = ops.get(pk, 0) + 1
            elif op.get("type") in ("random", "promotion", "extension", "calibration", "seed", "morph",
                                    "external_seed", "grow", "mutation", "phase0"):
                key = f"ops/{op['type']}" + ("_exploratory" if op.get("exploratory") else "")
            if key:
                ops[key] = ops.get(key, 0) + 1
        sc.update(ops)
        for init in INITS:
            fr = self.front(top, init=init)
            sc[f"hv_strata/init_{init}"] = hypervolume([j["F"] for j in fr], self.ref) if fr else 0.0
        tables["quality_vs_epochs"] = (["cand_id", "stratum", "epochs", "quality", "params"],
                                       [[j["cand_id"], str(last_stratum.get(j["cand_id"])), float(j["fidelity"]),
                                         float(j["result"].get("quality", 0.0)), j["result"].get("params")]
                                        for j in jobs], ("line", "epochs", "quality", "stroke:stratum"))
        tables["params_vs_quality"] = (["cand_id", "stratum", "fidelity", "params", "flops", "quality", "init"],
                                       [[j["cand_id"], str(j.get("stratum")), float(j["fidelity"]),
                                         j["result"].get("params"), j["result"].get("flops"),
                                         float(j["result"].get("quality", 0.0)), j["genome"]["init"]]
                                        for j in jobs if j["fidelity"] == top], ("scatter", "params", "quality", None))
        front = sorted(self.front(top, role="search"), key=lambda j: j["result"].get("params") or 0)
        marg = []
        for a, c in zip(front, front[1:]):
            dp = (c["result"].get("params") or 0) - (a["result"].get("params") or 0)
            dq = float(c["result"].get("quality", 0)) - float(a["result"].get("quality", 0))
            marg.append([f"{a['cand_id']}->{c['cand_id']}", str(c.get("stratum")), c["result"].get("params"),
                         dp, dq, dq / (dp / 1e5) if dp > 0 else None])
        tables["marginal_gain"] = (["step", "stratum", "params", "delta_params", "delta_quality",
                                    "dq_per_100k_params"], marg, ("bar", "step", "dq_per_100k_params", None))
        hist = self.state.setdefault("pareto_history", [])
        if record_history:
            hist.extend([[b, j["cand_id"], str(j.get("stratum")), j["result"].get("params"), j["result"].get("flops"),
                          float(j["result"].get("quality", 0.0))] for j in front])
        tables["pareto_evolution"] = (["bracket", "cand_id", "stratum", "params", "flops", "quality"], hist,
                                      ("scatter", "params", "quality", None))
        tables["category_operations"] = (["operation", "count"], [[k[4:], v] for k, v in sorted(ops.items())],
                                         ("bar", "operation", "count", None))
        tables["calibration"] = (["stratum", "epochs", "quality"],
                                 [[s, float(e), q] for s, c in self.state.get("calibration", {}).items()
                                  for e, q in c.get("quality_by_epochs", {}).items()],
                                 ("line", "epochs", "quality", "stroke:stratum"))
        return {"scalars": sc, "tables": tables}

    # ---- PNAS-CS (opt-in, pnas_cs.py): never reached unless ``pnas_cs.enabled``
    def _encode(self, genome):
        """DE vector of ``genome``: its encoding, or (PNAS-CS) its family's base encoding + (alpha, beta)."""
        if getattr(self, "pnas", None) is None:
            return encode(genome, **self.enc_kw)
        from ga_search.modehb import pnas_cs as PC
        info = genome.get("pnas") or {}
        base = (self._families.get(info.get("family")) or {}).get("base") or genome
        a = info.get("alpha", self.pnas["seed_alpha"])
        b = info.get("beta", self.pnas["seed_beta"])
        return np.concatenate([encode(base, **self.enc_kw), PC.encode_coeffs(a, b, self.pnas)])

    def _gene_mask(self, genome, stage):
        m = gene_mask(genome, stage)
        return m if self.dim == DIM else np.concatenate([m, np.ones(self.dim - DIM, dtype=bool)])

    def _pnas_register(self, base, alpha, beta, origin, parent_family=None):
        from ga_search.modehb import pnas_cs as PC
        base = {k: v for k, v in copy.deepcopy(base).items() if k not in ("pnas", "external")}
        fam = PC.family_id(base, alpha, beta)
        if fam not in self._families:
            self._families[fam] = {"base": base, "alpha": float(alpha), "beta": float(beta), "origin": origin,
                                   "parent_family": parent_family}
        return fam

    def _pnas_rep(self, g):
        """repair + SPiFiL-images repair, WITHOUT the GPU-memory repair (which would add pools /
        shrink widths and break the family): memory is a feasibility constraint (``_pnas_mem_ok``)."""
        from ga_search.modehb import strata as S
        g = repair(g, **self.repair_kw)
        n_cls = int(self.cfg.get("n_classes") or 0)
        if not (n_cls and g.get("init") == "spifil"):
            return g
        kern = [k for k in (self.repair_kw.get("search_space") or {}).get("kernel", KERNELS) if int(k) > 1]
        return S.spifil_images_repair(g, n_cls, self.repair_kw["spifil_images_choices"],
                                      int(self.strat.get("spifil_seeds_per_image", 100)),
                                      pool_sizes=self.cfg.get("spifil_pool_sizes"),
                                      repair=lambda y: repair(y, **self.repair_kw), kernels=kern or (3,))

    def _pnas_mem_ok(self, g):
        from ga_search.modehb import strata as S
        mb = self.strat.get("memory_budget_gb")
        if not mb:
            return True
        n_cls = int(self.cfg.get("n_classes") or 0)
        est = S.mem_estimate_gb(g, int(self.cfg["image_size"]), n_classes=n_cls or None,
                                seeds_per_image=int(self.strat.get("spifil_seeds_per_image", 100)),
                                pool_sizes=self.cfg.get("spifil_pool_sizes"))
        return est * float(self.pnas.get("mem_scale", 1.0)) <= float(mb)

    def _pnas_fit(self, base, alpha, beta, level, min_params=None, min_mults=(None, None), fresh=False):
        """``fresh`` (grid / DE / random member, no parent): full repairs incl. the GPU-memory repair
        (it may add early pools); growth: repairs WITHOUT structural memory repair + memory as a
        feasibility constraint, so the child keeps the parent's structure."""
        from ga_search.modehb import pnas_cs as PC
        pc = self.pnas
        return PC.fit_level(base, alpha, beta, PC.budget_of(pc, level), self._count,
                            self._rep if fresh else self._pnas_rep,
                            cap_params=int(self.strat["param_cap"]), cap_count=self.cap_fn,
                            feasible=None if fresh else self._pnas_mem_ok, tol=float(pc["tolerance"]),
                            max_depth=int(self.cfg["max_depth"]),
                            cmax=int((self.repair_kw.get("channel_space") or {}).get("max", 1024)),
                            deepen=pc["deepen"], min_params=min_params, min_mults=min_mults, n_layers=N_LAYERS,
                            residual_activation=pc["residual_activation"])

    def _pnas_fresh(self, base, alpha, beta, level, origin):
        """Member of a NEW family at ``level``: compound scaling of ``base`` (full repairs); the family
        is then anchored at the member itself (its own base, multipliers 1), so later growth is a
        pure compound scaling of a trained network."""
        from ga_search.modehb import strata as S
        fit = self._pnas_fit(base, alpha, beta, level, fresh=True)
        if fit is None:                       # nothing admissible: the old repairs (memory + cap) on the base
            g = S.enforce_cap(self._rep(base), self.strat, self._rep, self.rng, count=self.cap_fn)
            g, info = self._rep(g), {"phi": 0.0, "within_tol": False, "params_est": int(self._count(g))}
        else:
            g, info = fit
        fam = self._pnas_register(g, alpha, beta, origin)
        g["pnas"] = self._pnas_info(fam, {**info, "d_mult": 1.0, "w_mult": 1.0}, level, origin,
                                    scaled_from={"phi": info.get("phi"), "d_mult": info.get("d_mult"),
                                                 "w_mult": info.get("w_mult")})
        return g

    def _pnas_info(self, fam, info, level, origin, **extra):
        F = self._families[fam]
        return {"family": fam, "alpha": F["alpha"], "beta": F["beta"], "phi": info.get("phi", 0.0),
                "d_mult": info.get("d_mult", 1.0), "w_mult": info.get("w_mult", 1.0), "level_target": level,
                "within_tol": info.get("within_tol"), "params_est": info.get("params_est"), "origin": origin, **extra}

    def _pnas_initial(self, pop):
        """[external seed (external_seed.py)] + [GA-checkpoint seeds] + grid bases compound-scaled to the
        levels (round-robin), each grid base its own family with a stratified (alpha, beta);
        ``plain_grid``: grid bases use plain dense operators (conv, norm none, no skip/softhebb) so that
        their growth is function-preserving end to end (external_seed morphisms)."""
        from ga_search.modehb import pnas_cs as PC
        pc, names = self.pnas, list(self.pnas["level_names"])
        out, seen = [], set()
        for g in self._external_seeds():          # converted external encoder; evaluator imports it itself
            fam = self._pnas_register(g, pc["seed_alpha"], pc["seed_beta"], "seed_external")
            g["pnas"] = {**self._pnas_info(fam, {"within_tol": None}, None, "seed_external"),
                         "name": (g.get("external") or {}).get("name")}
            if arch_hash(g) not in seen:
                seen.add(arch_hash(g))
                out.append(g)
        for sd in PC.discover_seeds(pc, pc.get("dataset"), self.out, int(self.cfg["seed"]), n_layers=N_LAYERS):
            g0 = copy.deepcopy(sd["genome"])
            g0["seed"] = int(self.cfg["seed"])
            g = repair(g0, **self.repair_kw)
            if PC._structure(g) != PC._structure(g0):
                print(f"[pnas_cs] WARNING seed {sd['name']}: repair changes its architecture -> skipped", flush=True)
                continue
            mb = (self.strat or {}).get("memory_budget_gb")
            if mb and sd.get("src_peak_gb") is not None:   # measured peak, scaled to LeJEPA's 4 views
                need = float(sd["src_peak_gb"]) * 4.0 / max(1, int(sd.get("src_views") or 4))
                if need > float(mb):
                    print(f"[pnas_cs] WARNING seed {sd['name']}: measured {need:.1f} GB > budget {mb} -> skipped",
                          flush=True)
                    continue
            fam = self._pnas_register(g, pc["seed_alpha"], pc["seed_beta"], "seed_checkpoint")
            g["pnas"] = {**self._pnas_info(fam, {"within_tol": None}, None, "seed_checkpoint"),
                         "name": sd["name"], "import": PC.checkpoint_inherit({**sd, "genome": g})}
            if arch_hash(g) not in seen:
                seen.add(arch_hash(g))
                out.append(g)
        rng = np.random.default_rng([int(self.cfg["seed"]), 0xC0FF])
        n = len(pop)
        A = [pc["alphas"][j % len(pc["alphas"])] for j in rng.permutation(n)]
        B = [pc["betas"][j % len(pc["betas"])] for j in rng.permutation(n)]
        for j, base in enumerate(pop):
            if pc.get("plain_grid"):
                base = copy.deepcopy(base)
                for L in base["layers"]:     # relu: depth growth = identity layers, which stay widenable
                    L.update(conv="conv", norm="none", residual=False, softhebb=False, activation="relu")
                base = repair(base, **self.repair_kw)
            g = self._pnas_fresh(base, A[j], B[j], names[j % len(names)], "grid")
            if arch_hash(g) not in seen:
                seen.add(arch_hash(g))
                out.append(g)
        return out

    def _pnas_decode(self, vec, init, stratum=None):
        """Family vector -> member at the target level (compound scaling of the decoded base)."""
        from ga_search.modehb import pnas_cs as PC
        from ga_search.modehb import strata as S
        v = np.asarray(vec, float)
        base = decode(v[:DIM], int(self.cfg["seed"]), **self.repair_kw)
        base["init"] = init
        base = repair(base, **self.repair_kw)
        a, b = PC.decode_coeffs(v[DIM:], self.pnas)
        level = stratum or self._stratum_est(base) or self.pnas["level_names"][0]
        return self._pnas_fresh(base, a, b, level, "de")

    def _pnas_acquire(self, fid, fids, k, ns):
        """grow (P_k -> P_k+1 of a family, morphism + inheritance) | structural mutation | DE / random
        fill over family vectors; the level is the round-robin target stratum."""
        pc = self.pnas
        sp = self.state["subpops"][str(fid)]
        want = self._draw_init()
        stage = 2 if self.rng.random() < float(self.cfg["epsilon"]) else self.state["stage"]
        glob = self._global_members()
        tstrat = self._next_stratum()
        u = self.rng.random()
        if glob and u < float(pc["grow_prob"]):
            m = self._pnas_grow(fid, tstrat)
            if m is not None:
                return m
        elif glob and u < float(pc["grow_prob"]) + float(pc["mutate_prob"]):
            m = self._pnas_mutate(fid, tstrat)
            if m is not None:
                return m
        if (len(sp) < self._max_pop[fid] and fid == self.fidelities[0]) or not glob:
            vec = self._random_vector(stage)
            g = self._decode(vec, want, stratum=tstrat, force=True)
            g["pnas"]["origin"] = "random"
            self._last_op = {"type": "random", "target_stratum": tstrat, "family": g["pnas"]["family"]}
            return list(np.asarray(vec, float)), g, None, None
        if sp and len(sp) >= self._max_pop[fid]:
            target = self._next_target(fid, want, tstrat)
            tgt = sp[target]
        else:
            target = None
            gl = [m for m in glob if m.get("stratum") == tstrat] or glob
            tgt = gl[select_promotions(np.array([m["F"] for m in gl]), 1, self.cfg["tie_breaker"])[0]]
        lower = fids[k - 1] if k > 0 else fid
        cross = self.rng.random() < float(self.strat["cross_category_prob"])
        pool = self._parent_pool(lower, ns[k], None if cross else tgt.get("stratum"))
        mask = self._gene_mask(tgt["genome"], stage)
        tvec = np.asarray(tgt["vector"], float)
        if len(pool) < 3:
            gpop = self._global_pop()
            pool = pool + [self._rand1(gpop, tvec) for _ in range(3 - len(pool) + 1)]
        mutant = self._rand1(pool, tvec)
        child = self._boundary(self._crossover(tvec, mutant, mask))
        g = self._decode(child, want, stratum=tstrat, force=True)
        self._last_op = {"type": "de", "target_stratum": tstrat, "parent_stratum": tgt.get("stratum"),
                         "pool": "cross" if cross else "same", "family": g["pnas"]["family"]}
        return list(child), g, target, tgt["cand_id"]

    def _pnas_inherit(self, parent, child, fid):
        """(ctx["inherit"], mode): function-preserving growth (external_seed.grow_blocks through
        run_search._grow_fp, ``layer_map``) from the parent's checkpoint; else LEMONADE prefix reuse."""
        from ga_search.modehb import external_seed as ES
        from ga_search.modehb import lemonade as LM
        from ga_search.modehb import pnas_cs as PC
        job = LM.pick_parent_checkpoint(self._find(parent, role="search"), fid, self.pnas["inherit_fidelity"])
        if job is None:
            return None, "none"
        lm = PC.layer_map(parent, child)
        if lm is not None:
            return {"checkpoint": job["result"]["checkpoint"], "layer_map": lm,
                    "parent_layers": ES.active_layers(parent), "n_blocks": sum(isinstance(x, int) for x in lm),
                    "parent_cand_id": job["cand_id"], "parent_fidelity": float(job["fidelity"]),
                    "parent_job": int(job["job"])}, "net2net"
        spec = LM.inherit_spec(job, LM.reusable_prefix(parent, child))
        return spec, ("prefix" if spec else "none")

    def _pnas_grow(self, fid, tstrat):
        """Child = next level of a family, grown from its best trained member (one growth per
        family and level). Prefers parents one level below the target stratum."""
        from ga_search.modehb import pnas_cs as PC
        pc, names = self.pnas, list(self.pnas["level_names"])
        grown = self.state.setdefault("pnas_grown", {})
        cand = {}
        for j in self.state["jobs"]:
            if j["role"] != "search" or j["status"] != "ok" or not j["result"].get("checkpoint"):
                continue
            info = j["genome"].get("pnas") or {}
            fam, lv = info.get("family"), PC.level_index(pc, j.get("stratum"))
            if fam not in self._families or lv is None or lv >= len(names) - 1 or f"{fam}|{lv + 1}" in grown:
                continue
            q = float(j["result"].get("quality", 0.0))
            c = cand.get(j["cand_id"])
            if c is None or q > c[3]:
                cand[j["cand_id"]] = (fam, lv, j["genome"], q)
        if not cand:
            return None
        tl = PC.level_index(pc, tstrat)
        lvls = {c[1] for c in cand.values()}
        lv = tl - 1 if tl is not None and tl - 1 in lvls else min(lvls)
        cid = max((c for c, v in cand.items() if v[1] == lv), key=lambda c: (cand[c][3], c))
        fam, _, parent, _ = cand[cid]
        grown[f"{fam}|{lv + 1}"] = cid
        F = self._families[fam]
        pp = parent.get("pnas") or {}
        fit = self._pnas_fit(F["base"], F["alpha"], F["beta"], names[lv + 1], min_params=int(self._count(parent)),
                             min_mults=(pp.get("d_mult", 1.0), pp.get("w_mult", 1.0)))
        if fit is None:
            return None
        child, info = fit
        if arch_hash(child) == arch_hash(parent):
            return None
        inherit, mode = self._pnas_inherit(parent, child, fid)
        child["pnas"] = self._pnas_info(fam, info, names[lv + 1], "grow", parent=cid, from_level=names[lv],
                                        to_level=names[lv + 1])
        self._pending_inherit = inherit
        self._last_op = {"type": "grow", "target_stratum": tstrat, "family": fam, "from_level": names[lv],
                         "to_level": names[lv + 1], "parent": cid, "morph": mode, "phi": info["phi"]}
        return self._encode(child).tolist(), child, None, cid

    def _pnas_mutate(self, fid, tstrat):
        """Structural mutation, type in round-robin over ``pnas_cs.mutations``: parent = a first-front
        member of the target level (then any other member, so the type is honoured when some parent
        supports it); the child starts a new family and inherits (net2net or prefix)."""
        from ga_search.modehb import pnas_cs as PC
        pc = self.pnas
        allm = self._global_members()
        glob = [m for m in allm if m.get("stratum") == tstrat] or allm
        if not glob:
            return None
        front = list(nondominated_fronts(np.array([m["F"] for m in glob]))[0])
        first = glob[int(self.rng.choice(front))]
        rest = [allm[i] for i in self.rng.permutation(len(allm)) if allm[i]["cand_id"] != first["cand_id"]]
        parents = [first] + rest[:7]
        ss = self.repair_kw.get("search_space")
        poolings = list(_choices("pooling", None, ss))
        pools = [p for p in _layer_choices("pool", ss) if p != "none"]
        ops = list(pc["mutations"])
        rr = int(self.state.get("pnas_mut_rr", 0))
        self.state["pnas_mut_rr"] = rr + 1
        found = None
        for t in range(len(ops)):
            op = ops[(rr + t) % len(ops)]
            for par in parents:
                m = PC.mutate(par["genome"], op, self.rng, pc, poolings=poolings, pools=pools,
                              max_depth=int(self.cfg["max_depth"]), n_layers=N_LAYERS)
                if m is None:
                    continue
                child = self._pnas_rep(m[0])
                if (arch_hash(child) != arch_hash(par["genome"]) and int(self.cap_fn(child)) <= int(self.strat["param_cap"])
                        and self._pnas_mem_ok(child)):
                    found = (op, par, m[1], child)
                    break
            if found:
                break
        if found is None:
            return None
        op, par, minfo, child = found
        pf = par["genome"].get("pnas") or {}
        a, b = pf.get("alpha", pc["seed_alpha"]), pf.get("beta", pc["seed_beta"])
        fam = self._pnas_register(child, a, b, "mutation:" + op, parent_family=pf.get("family"))
        inherit, mode = self._pnas_inherit(par["genome"], child, fid)
        child["pnas"] = self._pnas_info(fam, {"within_tol": None}, self._stratum_est(child), "mutation",
                                        mutation=op, parent=par["cand_id"], parent_family=pf.get("family"))
        self._pending_inherit = inherit
        self._last_op = {"type": "mutation", "mutation": op, "target_stratum": tstrat, "parent": par["cand_id"],
                         "parent_stratum": par.get("stratum"), "family": fam, "morph": mode,
                         **{k: v for k, v in minfo.items() if k != "mutation"}}
        return self._encode(child).tolist(), child, None, par["cand_id"]

    def _pnas_phase0(self):
        """Phase 0 AE ablation: the same level-P0 grid seeds with the per-layer local AE off /
        decoder_first / joint, equal epoch budget (AE off: every epoch goes to SSL). The search then
        runs with the AE always on and the schedule with the best mean VALIDATION quality."""
        pc = self.pnas
        p0 = pc["phase0"]
        lv = p0.get("level") or pc["level_names"][0]
        seeds = [g for g in self.init_queue if (g.get("pnas") or {}).get("origin") == "grid"
                 and g["pnas"].get("level_target") == lv][:int(p0.get("n_seeds", 2))]
        fid = float(p0.get("fidelity", 45))
        rows, by = [], {}
        for si, g in enumerate(seeds):
            for variant in ("noae", "decoder_first", "joint"):
                gv = copy.deepcopy(g)
                if variant != "noae":
                    gv["ae_schedule"] = variant
                gv["pnas"] = {**gv["pnas"], "origin": "phase0", "phase0_variant": variant}
                rec = self._evaluate(gv, fid, parent_id=None, generation=-2, rung=0,
                                     role="phase0_noae" if variant == "noae" else "phase0_ae",
                                     extra_ctx={"ae_mode": "off" if variant == "noae" else "on"})
                rec["op"] = {"type": "phase0", "variant": variant, "seed_index": si}
                self._log_live(-2)
                if rec["status"] == "ok":
                    r = rec["result"]
                    kap = float(r.get("fitness_kappa", r.get("quality", 0.0)))
                    rows.append({"seed_index": si, "variant": variant, "cand_id": rec["cand_id"], "fidelity": fid,
                                 "kappa": kap, "quality": float(r.get("quality", 0.0))})
                    by.setdefault(variant, []).append(float(r.get("quality", 0.0)))
        means = {v: float(np.mean(q)) for v, q in by.items()}
        ae = {v: means[v] for v in ("decoder_first", "joint") if v in means}
        chosen = max(ae, key=lambda v: (ae[v], v == "decoder_first")) if ae else None
        self.state["phase0"] = {"rows": rows, "mean_kappa": means, "chosen_ae_schedule": chosen, "fidelity": fid,
                                "ae_minus_noae": (ae[chosen] - means["noae"]) if chosen and "noae" in means else None,
                                "level": lv, "n_seeds": len(seeds)}
        if chosen:                          # AE always on, schedule fixed by the ablation (validation)
            ss = self.repair_kw.get("search_space")
            if ss is None:
                ss = {"ae_schedule": (chosen,)}
                self.repair_kw["search_space"] = ss
                self.enc_kw["search_space"] = ss
            else:
                ss["ae_schedule"] = (chosen,)
        print(f"[pnas_cs] phase0: mean val quality {means} -> ae_schedule={chosen}", flush=True)
        self._save()

    def _pnas_log(self, b, tables=False):
        """History (HV x GPU-hours, Pareto snapshots) after every new job + local JSON (always) +
        W&B panels (Pareto every evaluation, the other tables every ``tables_every`` jobs / bracket)."""
        from ga_search.modehb import pnas_cs as PC
        n = len(self.state["jobs"])
        if n != getattr(self, "_pnas_n", -1):
            self._pnas_n = n
            ok = [j for j in self.state["jobs"] if j["status"] == "ok" and j["role"] == "search"]
            ok = list({(j["cand_id"], float(j["fidelity"])): j for j in ok}.values())   # cached repeats once
            top = self.fidelities[-1]
            hv_top = hypervolume([j["F"] for j in ok if j["fidelity"] == top], self.ref) if ok else 0.0
            best = max((j["fidelity"] for j in ok), default=None)
            hv_best = hypervolume([j["F"] for j in ok if j["fidelity"] == best], self.ref) if ok else 0.0
            self.state.setdefault("pnas_hv_history", []).append([n, float(self.state["gpu_hours"]), hv_top, hv_best])
            last = self.state["jobs"][-1] if self.state["jobs"] else None
            if last is not None and last["status"] == "ok" and last["role"] == "search":
                f = float(last["fidelity"])
                sub = [j for j in ok if float(j["fidelity"]) == f]
                fr = PC._front([(1.0 - PC._kappa(j), float(j["result"].get("flops") or 0)) for j in sub])
                ids = sorted(sub[i]["cand_id"] for i in fr)
                prev = getattr(self, "_pnas_front_ids", {})
                if prev.get(f) != ids:
                    prev[f] = ids
                    self._pnas_front_ids = prev
                    self.state.setdefault("pnas_pareto_history", []).extend(
                        [[n, b, f, sub[i]["cand_id"], float(sub[i]["result"].get("flops") or 0) / 1e9,
                          PC._kappa(sub[i])] for i in fr])
        if self._replaying():
            return
        try:
            rep = PC.report(self.state, self.fidelities, self.pnas, float(self.cfg["ref_point"]))
            d = self.out / "pnas"
            atomic_json(d / "report.json", {"bracket": b, "n_jobs": n, "time": time.time(), **rep})
            pf = rep["tables"]["pareto_kappa_flops"]
            atomic_json(d / "pareto_front.json", {"n_jobs": n, "columns": pf[0], "rows": pf[1]})
        except Exception as exc:
            print(f"[modehb] pnas report failed: {exc}", flush=True)
            return
        if self.logger is None or not hasattr(self.logger, "log_pnas"):
            return
        every = max(1, int(self.pnas.get("tables_every", 5)))
        full = tables or n % every == 0
        tabs = rep["tables"] if full else {k: rep["tables"][k] for k in ("pareto_kappa_flops", "hv_vs_gpu_hours")}
        try:
            self.logger.log_pnas(b, self.state["gpu_hours"], n, rep["scalars"], tabs)
        except Exception as exc:
            print(f"[modehb] log_pnas failed: {exc}", flush=True)
