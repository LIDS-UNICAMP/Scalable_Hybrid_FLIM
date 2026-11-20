"""Parameter-stratified MO-DEHB helpers (opt-in YAML block ``stratified:``).

Categories by TOTAL encoder parameters, including the final 1x1 projection of the
EmbeddingHead (never MACs/FLOPs, which stay as recorded metrics):

    Tiny <= 300k < Small <= 800k < Medium <= 1.5M < Huge <= 3.5M   (nothing above the cap)

Pieces used by ``modehb.MODEHB`` when ``stratified.enabled``:

* ``param_estimate(genome)``: structural count (ConvBlock per active layer + EmbeddingHead,
  built on the ``meta`` device, no memory). Used BEFORE evaluation to place a genome in a
  category / enforce the cap; after evaluation the measured ``result["params"]`` reclassifies.
* ``fit_to_stratum``: rescales all active widths by one factor (monotonic order kept, then the
  usual ``repair``) so the estimate falls inside the requested category.
* ``stratified_promotions``: protected successive-halving promotion: per-category quota
  (>= ``min_share`` of the slots for each category with valid candidates), in-category
  non-dominated ranking, plus exploratory promotions of Medium/Huge that are still improving.
* ``learning_slope``: quality gain per log-epoch between a candidate's last two fidelities.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_STRATA = [("tiny", 0, 300_000), ("small", 300_000, 800_000),
                  ("medium", 800_000, 1_500_000), ("huge", 1_500_000, 3_500_000)]
DEFAULT_STRAT = dict(enabled=False, strata=None, param_cap=3_500_000, tiny_floor=5_000,
                     min_share=0.25, protected_brackets=None, cross_category_prob=0.3,
                     projection_prob=0.7, explore_extra=1, explore_min_slope=0.0,
                     explore_strata=["medium", "huge"], calibration=True,
                     extension_fidelities=[], extension_min_gain=0.01, extension_max_per_bracket=2,
                     # --- v4 (search_v4.yaml; all opt-in, the defaults keep the v3 behaviour) ---
                     band_weights=None,          # {band: weight}: generation quota AND promotion share per band
                     band_quota=False,           # True: next target band = most under-represented (count/weight)
                     promotion="quota",          # "per_band": Pareto INSIDE each band, every bracket (no global FLOPs)
                     promote_min_per_band=0,     # >=1: every band with valid candidates gets this many slots
                     parent_select="front",      # "quality": DE parent pool ranked by quality only (no FLOPs)
                     skip_failed_in_subpop=False,  # True: failed/infeasible never occupy a sub-population slot
                     mem_scale=1.0,              # calibration of mem_estimate_gb (measured peak/est, 1699 jobs)
                     oom_fallback=[],            # OOM ladder: [{amp_dtype: bf16}, {amp_dtype: bf16, batch_div: 2}]
                     oom_fast_retries=None,      # with oom_fallback: transient retries before the ladder
                     oom_retry_sleep_s=None,
                     cap_count="estimate")       # "spifil_ops": param_cap checked BEFORE training on param_count_built


def strat_config(cfg: Optional[dict]) -> dict:
    out = {**DEFAULT_STRAT, **(cfg or {})}
    raw = out.get("strata") or DEFAULT_STRATA
    if isinstance(raw, dict):   # YAML {tiny: [0, 300000], ...}
        raw = [(k, v[0], v[1]) for k, v in raw.items()]
    strata = sorted([(str(n), int(lo), int(hi)) for n, lo, hi in raw], key=lambda t: t[1])   # JSON sorts keys
    for (_, _, hi), (_, lo, _) in zip(strata, strata[1:]):
        if hi != lo:
            raise ValueError("strata must be contiguous (hi of one = lo of the next)")
    out["strata"] = strata
    out["param_cap"] = int(min(int(out["param_cap"]), strata[-1][2]))
    return out


def names(sc: dict) -> List[str]:
    return [n for n, _, _ in sc["strata"]]


def stratum_of(params, sc: dict) -> Optional[str]:
    """Category of a parameter count; None above the cap (infeasible)."""
    if params is None:
        return None
    p = int(params)
    if p > sc["param_cap"]:
        return None
    for n, lo, hi in sc["strata"]:
        if lo < p <= hi or (lo == 0 and p <= hi):
            return n
    return None


def param_estimate(genome: dict) -> int:
    """Encoder (active layers) + EmbeddingHead parameters, built on the meta device."""
    import torch
    from ga_search.modehb.ops_norm_act import ConvBlock
    from ga_search.modehb.pooling_head import EmbeddingHead
    n, cin = 0, 3
    with torch.device("meta"):
        for i in range(int(genome["depth"])):
            L = {k: v for k, v in genome["layers"][i].items() if k != "active"}
            block = ConvBlock({**L, "in_channels": cin})
            n += sum(p.numel() for p in block.parameters())
            cin = int(L["channels"])
        head = EmbeddingHead(cin, int(genome["embed_dim"]), genome["pooling"])
        n += sum(p.numel() for p in head.parameters())
    return int(n)


def param_count_built(genome: dict) -> int:
    """Upper bound of the MEASURED ``result["params"]`` (encoder + EmbeddingHead) with ``spifil_ops``:
    the blocks the evaluator builds (SpifilOpsBlock: conv bias + 1x1 residual projection) at the
    requested widths (SPiFiL only builds fewer), every residual counted as a projection (when SPiFiL
    narrows the previous layer an identity skip becomes one). ``param_estimate`` (ConvBlock) skips
    both: remote_sensing v4 c00015 = 5,251,641 estimated, 5,612,872 measured, 5,797,782 here."""
    import torch
    from ga_search.modehb.pooling_head import EmbeddingHead
    from ga_search.modehb.spifil_ops import SpifilOpsBlock, to_ops_spec
    n, cin = 0, 3
    with torch.device("meta"):
        for i in range(int(genome["depth"])):
            spec = to_ops_spec(genome["layers"][i], cin)
            if spec["residual"]:
                spec["skip"] = "project"
            n += sum(p.numel() for p in SpifilOpsBlock(spec).parameters())
            cin = int(spec["channels"])
        n += sum(p.numel() for p in EmbeddingHead(cin, int(genome["embed_dim"]), genome["pooling"]).parameters())
    return int(n)


def _scaled(genome: dict, s: float, cmax: int) -> dict:
    g = {**genome, "layers": [dict(L) for L in genome["layers"]]}
    for i in range(int(g["depth"])):
        g["layers"][i]["channels"] = int(min(max(round(float(genome["layers"][i]["channels"]) * s), 1), cmax))
    return g


def fit_to_stratum(genome: dict, stratum: str, sc: dict, repair: Callable[[dict], dict],
                   rng: np.random.Generator, *, count: Callable[[dict], int] = param_estimate,
                   cmax: int = 1024, iters: int = 24, max_depth: int = 8) -> dict:
    """``_fit_widths``; when widths alone cannot reach the category (e.g. the GPU-memory
    repair caps high-resolution layers), add layers (depth + 1, up to ``max_depth``) and retry."""
    g = genome
    for _ in range(max_depth):
        best = _fit_widths(g, stratum, sc, repair, rng, count=count, cmax=cmax, iters=iters)
        lo = next(lo for n, lo, hi in sc["strata"] if n == stratum)
        if stratum_of(count(best), sc) == stratum or count(best) > lo or int(best["depth"]) >= max_depth:
            return best
        g = repair({**best, "depth": int(best["depth"]) + 1})
    return best


def _fit_widths(genome: dict, stratum: str, sc: dict, repair: Callable[[dict], dict],
                rng: np.random.Generator, *, count: Callable[[dict], int] = param_estimate,
                cmax: int = 1024, iters: int = 24) -> dict:
    """Rescale widths (one factor for every active layer) so ``count`` lands in ``stratum``.

    The target is drawn log-uniformly inside the category (Tiny: from ``tiny_floor``), so
    projected offspring keep spreading over the whole band. Returns the closest repaired
    genome found (it may stay outside the band when the depth/operators cannot reach it,
    e.g. a 1-layer net can't be Huge; the measured count reclassifies it afterwards)."""
    lo, hi = next((lo, hi) for n, lo, hi in sc["strata"] if n == stratum)
    lo_t = max(lo, int(sc.get("tiny_floor", 5_000))) if lo == 0 else lo
    target = math.exp(rng.uniform(math.log(lo_t + 1), math.log(hi)))
    g0 = repair(genome)
    best, best_err = g0, abs(math.log(max(count(g0), 1) / target))
    a, b = math.log(1.0 / 1024), math.log(1024.0)
    for _ in range(iters):
        m = 0.5 * (a + b)
        g = repair(_scaled(g0, math.exp(m), cmax))
        p = count(g)
        err = abs(math.log(max(p, 1) / target))
        if err < best_err or (stratum_of(p, sc) == stratum and stratum_of(count(best), sc) != stratum):
            best, best_err = g, err
        if p < target:
            a = m
        else:
            b = m
        if b - a < 1e-3:
            break
    return best


def enforce_cap(genome: dict, sc: dict, repair: Callable[[dict], dict], rng, *,
                count: Callable[[dict], int] = param_estimate) -> dict:
    """Never generate a model above ``param_cap``: shrink into the top category if needed."""
    if count(genome) <= sc["param_cap"]:
        return genome
    top = sc["strata"][-1][0]
    g = fit_to_stratum(genome, top, sc, repair, rng, count=count)
    s = 0.9
    while count(g) > sc["param_cap"] and s > 1e-3:   # depth/ops floor: keep shrinking
        g = repair(_scaled(g, s, 1024))
        s *= 0.9
    return g


def learning_slope(quality_by_fid: Dict[float, float]) -> Optional[float]:
    """Quality gain per log2(epoch) between the two highest fidelities seen (None if <2)."""
    if len(quality_by_fid) < 2:
        return None
    (f0, q0), (f1, q1) = sorted(quality_by_fid.items())[-2:]
    return float((q1 - q0) / max(math.log2(f1 / f0), 1e-9))


def stratified_promotions(cands: Sequence[dict], n: int, sc: dict, select: Callable,
                          *, counts: Optional[Dict[str, int]] = None) -> List[int]:
    """Protected promotion. ``cands``: dicts with ``F``, ``stratum``, ``slope``.

    1. quota q = max(1, floor(min_share * n)) per category that has valid candidates; when
       n < #categories, the categories least promoted so far (``counts``) go first;
    2. inside each category: ``select(F_cat, k)`` (non-dominated sort + crowding);
    3. remaining slots: global ``select`` over the not-yet-chosen candidates;
    4. ``explore_extra`` extra slots for still-improving Medium/Huge (slope > min).
    Returns indices into ``cands`` (base slots first, exploratory last)."""
    if n <= 0 or not cands:
        return []
    order = names(sc)
    present = [s for s in order if any(c["stratum"] == s for c in cands)]
    counts = counts or {}
    present.sort(key=lambda s: (counts.get(s, 0), order.index(s)))
    quota = max(1, int(math.floor(float(sc["min_share"]) * n)))
    chosen: List[int] = []
    for s in present:
        if len(chosen) >= n:
            break
        idx = [i for i, c in enumerate(cands) if c["stratum"] == s]
        k = min(quota, len(idx), n - len(chosen))
        F = np.array([cands[i]["F"] for i in idx])
        chosen.extend(idx[j] for j in select(F, k))
    rest = [i for i in range(len(cands)) if i not in chosen]
    if len(chosen) < n and rest:
        F = np.array([cands[i]["F"] for i in rest])
        chosen.extend(rest[j] for j in select(F, min(n - len(chosen), len(rest))))
    extra = int(sc.get("explore_extra", 0))
    if extra > 0:
        thr = float(sc.get("explore_min_slope", 0.0))
        pool = [i for i, c in enumerate(cands) if i not in chosen and c["stratum"] in sc["explore_strata"]
                and c.get("slope") is not None and c["slope"] > thr]
        pool.sort(key=lambda i: -cands[i]["slope"])
        chosen.extend(pool[:extra])
    return chosen


def band_weights(sc: dict) -> Dict[str, float]:
    """Per-band weight (v4). Absent = equal weights (v3 round-robin)."""
    w = sc.get("band_weights") or {}
    return {n: float(w.get(n, 1.0)) for n in names(sc)}


def next_band(counts: Dict[str, int], sc: dict) -> str:
    """Directed sampling (v4): the band with the smallest count/weight; ties -> the LARGER band.
    ``counts`` = valid candidates already generated per band in the current bracket, so a band
    whose offspring failed or landed in another band stays under-represented and is re-targeted."""
    w = band_weights(sc)
    order = names(sc)
    return min(order, key=lambda n: (counts.get(n, 0) / max(w[n], 1e-9), -order.index(n)))


def band_promotions(cands: Sequence[dict], n: int, sc: dict, select: Callable) -> List[int]:
    """v4 promotion BY BAND. ``cands``: dicts with ``F`` (1-quality, flops) and ``stratum``.

    * slots per band proportional to ``band_weights`` over the bands that have candidates
      (largest remainder, larger bands first), at least ``promote_min_per_band`` each, so the
      large bands get >= the small ones and n < #bands never starves a band;
    * inside a band: ``select`` (non-dominated sort on (1-q, flops) + crowding): the Pareto is
      taken WITHIN the band, so FLOPs only compares models of similar size;
    * slots a band cannot fill go to the other bands by in-band quality (largest band first);
    * ``explore_extra`` still-improving large models (slope > min) as in v3.
    Returns indices into ``cands``; may exceed ``n`` by the per-band minimum."""
    if n <= 0 or not cands:
        return []
    order = names(sc)
    w = band_weights(sc)
    present = [s for s in order if any(c["stratum"] == s for c in cands)]
    mn = int(sc.get("promote_min_per_band", 0) or 0)
    n_eff = max(n, mn * len(present))
    tot = sum(w[s] for s in present) or 1.0
    raw = {s: n_eff * w[s] / tot for s in present}
    k = {s: max(mn, int(math.floor(raw[s]))) for s in present}
    rem = n_eff - sum(k.values())
    for s in sorted(present, key=lambda s: (-(raw[s] - math.floor(raw[s])), -order.index(s))):
        if rem <= 0:
            break
        k[s] += 1
        rem -= 1
    chosen: List[int] = []
    left = 0
    for s in sorted(present, key=lambda s: -order.index(s)):
        idx = [i for i, c in enumerate(cands) if c["stratum"] == s]
        kk = min(k[s], len(idx))
        left += k[s] - kk
        if kk:
            F = np.array([cands[i]["F"] for i in idx])
            chosen.extend(idx[j] for j in select(F, kk))
    while left > 0:
        rest = [i for i, c in enumerate(cands) if i not in chosen]
        if not rest:
            break
        rest.sort(key=lambda i: (cands[i]["F"][0], -order.index(cands[i]["stratum"])
                                 if cands[i]["stratum"] in order else 0))
        chosen.append(rest[0])
        left -= 1
    extra = int(sc.get("explore_extra", 0))
    if extra > 0:
        thr = float(sc.get("explore_min_slope", 0.0))
        pool = [i for i, c in enumerate(cands) if i not in chosen and c["stratum"] in sc["explore_strata"]
                and c.get("slope") is not None and c["slope"] > thr]
        pool.sort(key=lambda i: -cands[i]["slope"])
        chosen.extend(pool[:extra])
    return chosen


# ─── GPU-memory model (measured on jaci 14/11/2026: Evaluator._evaluate instrumented per phase,
# batch 64 @224, torch.cuda.max_memory_allocated; small widths, linear fit in channels) ─────────
# Unit: GB per OUTPUT channel of a stride-1 layer at FULL 224x224 input resolution; a layer whose
# input is H x H costs coef * C * (H/224)^2. The evaluation peak is the MAX over its phases,
# not their sum (each phase frees its tensors):
#   Fase C  global SSL : backprop through the whole encoder -> SUM over layers, x views*batch
#                        (lejepa 4x64=256 imgs = 1.0; simclr 2x64 = 0.43; ae 1x64 = 0.18)
#   Fase B  per layer  : local AE (64 imgs, block + decoder at the target size), SoftHebb,
#                        ZCA fit, eval views -> MAX over layers
#   Fase A  SPiFiL     : Fisher scoring materialises the N x N patch-distance matrix
#                        (N = seeds x donors, ~16 bytes/entry) + Mahalanobis D x D / N x D
MEM_SSL_NORM = {"none": 0.132, "batch": 0.184, "zca": 0.206, "batch_zca": 0.257,
                "dbn": 0.308, "batch_dbn": 0.360}
MEM_SSL_IN = {"dwsep": 0.051, "inception": 0.154}   # per INPUT channel (depthwise map / maxpool branch)
MEM_SSL_ACT = 0.037                                   # gelu/silu/selu/... vs relu
MEM_SSL_RES = 0.038                                   # residual skip (identity or projection)
MEM_SSL_SCALE = {"lejepa": 1.0, "simclr": 0.43, "ae": 0.18}
MEM_SSL_BASE = {"lejepa": 1.0, "simclr": 1.3, "ae": 1.3}
MEM_HEAD_EXTRA_GB = 6.0                               # gem/moments/...: pooling temporaries of one head
MEM_HEAD_BYTES = 6.0 * 256                            #   chunk (pooling_head.CHUNK_ELEMS) = min(6 GB,
                                                      #   1.5 kB x embed_dim x h_final^2), any ssl; gap: 0
MEM_HEAD_EVAL_EXTRA = 2.0                             # same, no-grad eval views (TTA) / SVM embedding
MEM_LAYER_NORM = {"none": 0.060, "batch": 0.072, "zca": 0.065, "batch_zca": 0.065,
                  "dbn": 0.090, "batch_dbn": 0.090}   # max(local AE, ZCA fit, eval views), 64 imgs
MEM_LAYER_DEC = 0.032                                 # local-AE decoder, per hidden ch (<=128) at target
MEM_LAYER_BASE = 1.8                                  # head/views/SVM embedding constants
MEM_SOFTHEBB = 0.065                                  # per out-ch of the FIRST conv (conv C, inception C/4)
MEM_SOFTHEBB_CHUNKED = 1.6                            # constant when softhebb.SOFTHEBB_CHUNK_ELEMS is used
MEM_SPIFIL_BYTES = {"n2": 16.0, "nd": 6.0, "d2": 12.0, "feat": 12.0}
MEM_MARGIN = 1.15                                     # fit error (<=15% on scaled replays) + allocator
# Back-compat names (old lejepa-only coefficients, kept for callers that import them)
MEM_COEF = {"conv": 0.22, "dwsep": 0.36, "depthwise": 0.30, "pointwise": 0.22, "inception": 0.75}
MEM_NORM_EXTRA = {"dbn": 0.08, "zca": 0.08, "batch_dbn": 0.12, "batch_zca": 0.12}
MEM_BASE_GB = 2.0


def _active(genome: dict) -> list:
    return genome["layers"][:int(genome["depth"])]


def _resolutions(genome: dict, image_size: int) -> List[float]:
    """Input resolution of every active layer (SPiFiL pools: 3x3 s2 p1 -> ceil(H/2))."""
    h, res = float(image_size), []
    for L in _active(genome):
        res.append(h)
        if L.get("pool", "none") != "none":
            h = float(math.ceil(h / 2))
    return res


def _layer_ssl_gb(L: dict, cin: int, s: float) -> float:
    c = int(L["channels"])
    coef = MEM_SSL_NORM.get(L.get("norm", "none"), max(MEM_SSL_NORM.values()))
    coef += MEM_SSL_ACT if L.get("activation", "relu") != "relu" else 0.0
    coef += MEM_SSL_RES if L.get("residual") else 0.0
    return (coef * c + MEM_SSL_IN.get(L.get("conv"), 0.0) * cin) * s


def mem_phases_gb(genome: dict, image_size: int = 224, ref_size: int = 224, *,
                  n_classes: Optional[int] = None, seeds_per_image: int = 100,
                  pool_sizes: Optional[Sequence[int]] = None,
                  softhebb_chunked: bool = False) -> Dict[str, float]:
    """Estimated peak (GB, real allocated) of every phase of one evaluation (no margin)."""
    ssl = genome.get("ssl", "lejepa")
    res = _resolutions(genome, image_size)
    ssl_gb = MEM_SSL_BASE.get(ssl, 1.3)
    layer_gb, sh_gb, cin = 0.0, 0.0, 3
    for L, h in zip(_active(genome), res):
        s = (h / ref_size) ** 2
        ssl_gb += MEM_SSL_SCALE.get(ssl, 1.0) * _layer_ssl_gb(L, cin, s)
        c = int(L["channels"])
        tgt = 1.0 if genome.get("ae_target", "pixel") != "feature" else s
        lay = (MEM_LAYER_NORM.get(L.get("norm", "none"), 0.09) * c * s
               + 0.3 * MEM_SSL_IN.get(L.get("conv"), 0.0) * cin * s
               + MEM_LAYER_DEC * min(128, c) * tgt)
        layer_gb = max(layer_gb, lay)
        if L.get("softhebb"):
            first = {"conv": c, "pointwise": c, "inception": max(1, c // 4)}.get(L.get("conv"), 0)
            sh_gb = max(sh_gb, MEM_SOFTHEBB_CHUNKED if softhebb_chunked else MEM_SOFTHEBB * first * s)
        cin = c
    h_f = res[-1] if res else float(image_size)
    if _active(genome) and _active(genome)[-1].get("pool", "none") != "none":
        h_f = float(math.ceil(h_f / 2))
    if genome.get("pooling", "gap") != "gap":
        elems = int(genome.get("embed_dim", 1280)) * h_f * h_f
        ssl_gb += min(MEM_HEAD_EXTRA_GB, MEM_HEAD_BYTES * elems / 1e9)
        head_eval = min(MEM_HEAD_EVAL_EXTRA, MEM_HEAD_BYTES * elems / 4 / 1e9)
    else:
        head_eval = 0.0
    out = {"ssl": ssl_gb, "layer": MEM_LAYER_BASE + head_eval + layer_gb, "softhebb": MEM_LAYER_BASE + sh_gb}
    if n_classes and genome.get("init", "spifil") == "spifil":
        n, _ = spifil_budget(int(genome.get("spifil_images_per_class", 1)), int(n_classes),
                             seeds_per_image, pool_sizes)
        n_img = n / seeds_per_image
        b = MEM_SPIFIL_BYTES
        sp, cin = 0.0, 3
        for (L, h), d in zip(zip(_active(genome), res), spifil_layer_dims(genome)):
            sp = max(sp, b["n2"] * n * n + b["nd"] * n * d + b["d2"] * d * d + b["feat"] * n_img * cin * h * h)
            cin = int(L["channels"])
        out["spifil"] = 0.3 + sp / 1e9
    return out


def mem_estimate_gb(genome: dict, image_size: int = 224, ref_size: int = 224, *,
                    n_classes: Optional[int] = None, seeds_per_image: int = 100,
                    pool_sizes: Optional[Sequence[int]] = None, softhebb_chunked: bool = False,
                    scale: float = 1.0) -> float:
    """Estimated peak GPU memory (GB, allocated, x MEM_MARGIN) of one evaluation: max over phases.
    ``scale`` (v4 ``stratified.mem_scale``): calibration measured on 1699 finished jobs of the v3
    searches (jaci, 18/11): real peak / estimate = 0.62 median, 0.77 p90, 1.11 p99 -> 0.8 keeps the
    big models that fit; the rare under-estimates are caught by the OOM fallback ladder."""
    ph = mem_phases_gb(genome, image_size, ref_size, n_classes=n_classes, seeds_per_image=seeds_per_image,
                       pool_sizes=pool_sizes, softhebb_chunked=softhebb_chunked)
    return float(MEM_MARGIN * float(scale) * max(ph.values()))


def mem_repair(genome: dict, budget_gb: float, repair: Callable[[dict], dict], *,
               image_size: int = 224, min_spatial: int = 6, max_iter: int = 64,
               n_classes: Optional[int] = None, seeds_per_image: int = 100,
               spifil_choices: Optional[Sequence[int]] = None, pool_sizes: Optional[Sequence[int]] = None,
               softhebb_chunked: bool = False, scale: float = 1.0) -> dict:
    """Make the genome fit ``budget_gb`` (real GB). Training phases over budget: (1) add
    max-pooling to the earliest active layer without pooling that still has room (downsampling
    before widening, as a CNN stem); (2) if no layer can pool, shrink the costliest layer (and
    every earlier wider layer, to keep widths non-decreasing). SPiFiL over budget: (a) fewer
    donor images per class while ``spifil_feasible`` (N > 1.05 D); (b) smaller kernel on the
    dense conv with the largest D; (c) shrink the layers that feed it. Call it AFTER
    ``spifil_images_repair`` (which only raises k)."""
    kw = dict(n_classes=n_classes, seeds_per_image=seeds_per_image, pool_sizes=pool_sizes,
              softhebb_chunked=softhebb_chunked)
    g = repair(genome)
    budget_gb = float(budget_gb) / max(float(scale), 1e-6)   # v4 calibration (scale 1.0 = v3)
    for _ in range(max_iter):
        ph = mem_phases_gb(g, image_size, **kw)
        if MEM_MARGIN * max(ph.values()) <= budget_gb:
            return g
        d = int(g["depth"])
        res = _resolutions(g, image_size)
        g = {**g, "layers": [dict(L) for L in g["layers"]]}
        if max(ph, key=ph.get) == "spifil":
            g = _spifil_mem_step(g, budget_gb / MEM_MARGIN, kw, spifil_choices, seeds_per_image)
            g = repair(g)
            continue
        pooled = False
        for i in range(d - 1):                       # pooling at layer i shrinks layers > i
            L = g["layers"][i]
            if L.get("pool", "none") == "none" and res[i] // 2 >= min_spatial:
                L["pool"] = "max"
                pooled = True
                break
        if not pooled:
            costs, cin = [], 3
            for L, h in zip(g["layers"][:d], res):
                costs.append(_layer_ssl_gb(L, cin, (h / image_size) ** 2))
                cin = int(L["channels"])
            j = int(np.argmax(costs))
            new = max(1, int(int(g["layers"][j]["channels"]) * 0.8))
            for i in range(j + 1):
                g["layers"][i]["channels"] = min(int(g["layers"][i]["channels"]), new)
        g = repair(g)
    return g


def _spifil_mem_step(g: dict, budget: float, kw: dict, choices, seeds_per_image: int) -> dict:
    n_cls, pools = int(kw["n_classes"]), kw.get("pool_sizes")
    seeds = 0.9 * float(seeds_per_image)              # = spifil_images_repair's seed_floor
    k0 = int(g.get("spifil_images_per_class", 1))
    ok = [k for k in sorted({int(c) for c in (choices or [k0])}) if k < k0
          and spifil_feasible({**g, "spifil_images_per_class": k}, k, n_cls, seeds, pools)]
    if ok:                                            # largest k that fits, else the smallest feasible
        fit = [k for k in ok if mem_phases_gb({**g, "spifil_images_per_class": k}, **kw)["spifil"] <= budget]
        return {**g, "spifil_images_per_class": max(fit) if fit else min(ok)}
    dims = spifil_layer_dims(g)
    j = int(np.argmax(dims))
    L = g["layers"][j]
    if L.get("conv") == "conv" and int(L.get("kernel", 3)) > 3:
        L["kernel"] = int(L["kernel"]) - 2               # 7 -> 5 -> 3: D / 2-2.8
    elif j > 0:
        new = max(1, int(int(g["layers"][j - 1]["channels"]) * 0.8))
        for i in range(j):
            g["layers"][i]["channels"] = min(int(g["layers"][i]["channels"]), new)
    else:                                             # layer 0 (D = 3 k^2): only k is left
        g["spifil_images_per_class"] = min(int(c) for c in (choices or [k0]))
    return g


def _spifil_kk(L: dict) -> int:
    """Spatial factor of the widest SPiFiL fit of a layer: D = cin * kk (dwsep pointwise: kk=1,
    its depthwise fits are k^2 on 1 channel). Inception ALWAYS fits a 5x5 branch when
    channels >= 3 (ops_norm_act.inception_split), whatever the genome's ``kernel`` gene."""
    from ga_search.modehb.ops_norm_act import inception_split
    conv, k = L["conv"], int(L["kernel"])
    if conv == "inception":
        return max(int(bk) for _, bk, _ in inception_split(int(L["channels"]))) ** 2
    if conv == "pointwise":
        return 1
    if conv in ("dwsep", "depthwise"):
        return 1
    return k * k


def spifil_layer_dims(genome: dict) -> List[int]:
    """Per active layer, the largest covariance dimension D its SPiFiL fit needs (upper bound:
    SPiFiL never builds MORE channels than requested, so the real cin <= the genome width).
    conv: cin*k^2 | inception: cin*25 (b5) | dwsep: max(k^2, cin) | depthwise: k^2 |
    residual projection: cin (dominated)."""
    out, cin = [], 3
    for L in genome["layers"][:int(genome["depth"])]:
        k = int(L["kernel"])
        d = cin * _spifil_kk(L)
        if L["conv"] in ("dwsep", "depthwise"):
            d = max(k * k, d if L["conv"] == "dwsep" else 0)
        out.append(int(d))
        cin = int(L["channels"])
    return out


def spifil_dim(genome: dict) -> int:
    """Largest covariance dimension D the SPiFiL fit needs over the active layers."""
    return max(spifil_layer_dims(genome) or [0])


def spifil_budget(k: int, n_classes: int, seeds_per_image: float,
                  pool_sizes: Optional[Sequence[int]] = None):
    """(N total, N of the smallest class) with k donors/class; a class with fewer training
    images than k contributes only what it has (run_search.build_spifil_pool)."""
    sizes = list(pool_sizes) if pool_sizes else [None] * int(n_classes)
    per = [seeds_per_image * (int(k) if s is None else min(int(k), int(s))) for s in sizes]
    return float(sum(per)), float(min(per))


def spifil_feasible(genome: dict, k: int, n_classes: int, seeds_per_image: float,
                    pool_sizes=None, margin: float = 1.05) -> bool:
    """N > margin*D for every fit, and every residual layer's 1x1 projection (ExactBudget
    allocator, ~channels/n_classes per class) has enough seed patches in the smallest class."""
    N, n_min = spifil_budget(k, n_classes, seeds_per_image, pool_sizes)
    if N <= margin * spifil_dim(genome):
        return False
    return all(not (i and L.get("residual")) or math.ceil(int(L["channels"]) / n_classes) + 1 <= n_min
               for i, L in enumerate(genome["layers"][:int(genome["depth"])]))


def spifil_images_repair(genome: dict, n_classes: int, choices: Sequence[int], seeds_per_image: int = 100,
                         margin: float = 1.05, *, pool_sizes: Optional[Sequence[int]] = None,
                         seed_floor: float = 0.9, repair: Optional[Callable[[dict], dict]] = None,
                         kernels: Sequence[int] = (3, 5, 7), max_iter: int = 64) -> dict:
    """Smallest ``spifil_images_per_class`` choice making every SPiFiL fit feasible
    (``spifil_feasible``). When even the largest choice is not enough, the genome is made
    feasible (deterministically, monotonic widths kept): drop the residual of a layer whose
    projection cannot be built; else reduce the kernel of an offending dense conv
    (7 -> 5 -> 3); else cap the widths of all earlier layers (= that layer's cin) to
    (N/margin)/kk. ``seed_floor``: masked SLIC gives 93..101 seeds/image (measured)."""
    if not n_classes:
        return genome
    seeds = float(seeds_per_image) * float(seed_floor)
    ks = sorted({int(c) for c in choices})
    kmin = min(int(x) for x in kernels)
    g = genome
    for _ in range(max_iter):
        k0 = int(g.get("spifil_images_per_class", 1))
        for k in ks:
            if k >= k0 and spifil_feasible(g, k, n_classes, seeds, pool_sizes, margin):
                return g if k == k0 else {**g, "spifil_images_per_class": k}
        N, n_min = spifil_budget(ks[-1], n_classes, seeds, pool_sizes)
        g = {**g, "spifil_images_per_class": ks[-1], "layers": [dict(L) for L in g["layers"]]}
        layers, dims = g["layers"], spifil_layer_dims(g)
        for i, L in enumerate(layers[:int(g["depth"])]):
            if i and L.get("residual") and math.ceil(int(L["channels"]) / n_classes) + 1 > n_min:
                L["residual"] = False
                break
            if dims[i] < N / margin:
                continue
            if L["conv"] == "conv" and int(L["kernel"]) > kmin:
                L["kernel"] = max(int(x) for x in kernels if int(x) < int(L["kernel"]))
                break
            cap = max(1, int((N / margin - 1) // _spifil_kk(L)))
            for j in range(i):
                layers[j]["channels"] = min(int(layers[j]["channels"]), cap)
            break
        else:
            return g
        if repair is not None:
            g = repair(g)
    return g
