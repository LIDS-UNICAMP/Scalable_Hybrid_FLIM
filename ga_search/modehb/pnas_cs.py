"""Progressive NAS with Compound Scaling (PNAS-CS) on top of MO-DEHB (opt-in YAML ``pnas_cs:``).

Absent / ``enabled: false`` -> MO-DEHB exactly as before (no code path below is reached).

Families and compound scaling
-----------------------------
A *family* = a base architecture (a genome: stem -> stages of conv/residual blocks -> 1x1 head +
pooling) + compound coefficients ``alpha`` (depth) and ``beta`` (width); the input resolution
stays at the dataset's 224 (same protocol for every candidate). The member of level ``k`` is
``scale_genome(base, alpha**phi_k, beta**phi_k)``; ``fit_level`` picks ``phi_k`` so that the
parameter count (encoder + final 1x1 projection, the measure ``strata`` uses) is ~ budget_k:
P0..P4 = 0.5M / 1M / 2M / 3.5M / 5.5M (tolerance 10 %, hard cap 5.5M x 1.05). The MO-DEHB vector
of a member is ``[base genome genes | alpha gene | beta gene]``: DE explores base architectures
and (alpha, beta); the level (the round-robin target stratum) fixes phi. Levels are the strata
(bands = geometric mid-points between budgets), so the protected per-stratum promotions of
``strata.py`` are protected promotions BY SIZE.

Stages: consecutive active layers up to and including the next pooled layer (a SpifilOpsBlock
pools at its END); the stem (layer 0) is never repeated. Depth scaling inserts layers right before
a stage's pooled layer (cin = cout = width of the layer that feeds it, pool none) or at the end of
the final un-pooled stage; width scaling multiplies every width (monotonic repair kept).

Network morphisms (function preserving, weight inheritance) = ``external_seed.py`` (reused, not duplicated)
---------------------------------------------------------------------------------------------------------
P_{k+1} is GROWN from the trained P_k: ``layer_map(parent, child)`` aligns the two genomes (child layer ->
parent layer | "identity" | "residual_zero") and the child is evaluated with
``ctx["inherit"] = {"checkpoint", "layer_map", "parent_layers", ...}``, i.e. ``run_search._grow_fp`` ->
``external_seed.grow_blocks``: kept filters copied, NEW filters SPiFiL-fitted (exact-width allocator) with
ZERO consumer columns (Net2Wider), Net2Deeper identity layers (relu input >= 0) or zero-init residual blocks
(x + silu(0*x)), local AE only on blocks with new elements, inherited elements moving at protect x lr
(``ElementProtect``), the step-0 preservation error logged; the first layer that is not mappable exactly
(non-plain block: norm / skip / grouped conv) and the following ones are re-fitted (SPiFiL + AE).
Depth growth inserts identity layers where the feeding layer ends in relu (no skip), else zero-init
residual blocks with silu (``deepen: auto``). ``external_seed.widen_block`` widens plain blocks only (no
norm / skip), so a later growth re-fits from the first zero-residual (or BN/ZCA/inception...) layer on;
``plain_grid`` keeps the grid families plain + relu, whose whole P0 -> P4 chain is function preserving.

Structural mutations (``mutate``): add_layer (identity), add_residual (zero-init residual, silu),
widen (one stage x factor), pooling (final pooling gene -> new head, encoder kept 1:1; or the pool type
of one stage: max <-> avg, not function preserving -> LEMONADE prefix inheritance).

Seeds: the external "SPiFiL pequeno + LeJEPA" encoder via ``external_seed`` (YAML ``external_seed:``, grid
= the seed itself; growth is the compound scaling) and our best GA checkpoints (``discover_seeds``: genome +
encoder imported 1:1 through ``_grow_fp`` with the identity layer map), frozen in ``<out>/pnas_seeds.json``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LEVEL_NAMES = ("P0", "P1", "P2", "P3", "P4")
DEFAULT_BUDGETS = (500_000, 1_000_000, 2_000_000, 3_500_000, 5_500_000)
ALPHAS = (1.0, 1.1, 1.2, 1.3, 1.4)
BETAS = (1.05, 1.1, 1.15, 1.2, 1.3)
MUTATIONS = ("add_layer", "add_residual", "widen", "pooling")
SPEC_KEYS = ("kernel", "conv", "residual", "activation", "norm", "pool")
NONNEG_ACTS = ("relu", "relu6")
N_EXTRA_GENES = 2          # alpha, beta appended to the MO-DEHB vector

DEFAULTS = dict(enabled=False, dataset=None, budgets=list(DEFAULT_BUDGETS), level_names=list(LEVEL_NAMES),
                tolerance=0.10, cap_factor=1.05, alphas=list(ALPHAS), betas=list(BETAS),
                seed_alpha=1.2, seed_beta=1.1, deepen="auto", residual_activation="silu", plain_grid=True,
                grow_prob=0.4, mutate_prob=0.3,
                mutations=list(MUTATIONS), widen_factors=[1.25, 1.5], inherit_fidelity="le_child",
                random_twins=False, tables_every=5, saturation_gain=0.01, mem_scale=1.0,
                phase0=dict(enabled=True, n_seeds=2, fidelity=45, level="P0"),
                seeds=dict(checkpoints={}, best_from=[], best_k=0))


# ─────────────────────────────────────────────────────────────── config / levels
def pnas_config(cfg: Optional[dict]) -> Optional[dict]:
    """Normalised config, or None when absent/disabled (original MO-DEHB)."""
    if not cfg or not cfg.get("enabled"):
        return None
    out = copy.deepcopy(DEFAULTS)
    for k, v in cfg.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    b = [int(x) for x in out["budgets"]]
    if len(b) < 2 or any(x >= y for x, y in zip(b, b[1:])) or b[0] <= 0:
        raise ValueError(f"pnas_cs.budgets must be strictly increasing and > 0, got {b}")
    out["budgets"] = b
    names = list(out["level_names"])[:len(b)]
    if len(names) != len(b) or len(set(names)) != len(names):
        raise ValueError("pnas_cs.level_names must name every budget once")
    out["level_names"] = names
    out["alphas"] = [float(a) for a in out["alphas"]]
    out["betas"] = [float(x) for x in out["betas"]]
    if not out["alphas"] or not out["betas"] or min(out["alphas"]) < 1.0 or min(out["betas"]) <= 1.0:
        raise ValueError("pnas_cs: alphas must be >= 1 and betas > 1")
    if out["deepen"] not in ("auto", "residual", "identity"):
        raise ValueError("pnas_cs.deepen must be 'auto', 'residual' or 'identity'")
    from ga_search.modehb.external_seed import ZERO_ACTS
    if out["residual_activation"] not in ZERO_ACTS:
        raise ValueError(f"pnas_cs.residual_activation must be one of {ZERO_ACTS} (zero-init residual stays trainable)")
    bad = [m for m in out["mutations"] if m not in MUTATIONS]
    if bad:
        raise ValueError(f"pnas_cs.mutations: {bad} not in {MUTATIONS}")
    if float(out["grow_prob"]) + float(out["mutate_prob"]) > 1.0 + 1e-9:
        raise ValueError("pnas_cs: grow_prob + mutate_prob must be <= 1")
    if out["phase0"].get("level", names[0]) not in names:
        raise ValueError("pnas_cs.phase0.level must be a level name")
    return out


def cap(pc: dict) -> int:
    """Hard parameter cap (budget of the last level x cap_factor)."""
    return int(round(pc["budgets"][-1] * float(pc["cap_factor"])))


def level_strata(pc: dict) -> Dict[str, list]:
    """Contiguous bands {name: [lo, hi]}: geometric mid-points between budgets, last hi = cap."""
    b, names = pc["budgets"], pc["level_names"]
    edges = [0] + [int(round(math.sqrt(b[i] * b[i + 1]))) for i in range(len(b) - 1)] + [cap(pc)]
    return {n: [edges[i], edges[i + 1]] for i, n in enumerate(names)}


def level_index(pc: dict, name) -> Optional[int]:
    return pc["level_names"].index(name) if name in pc["level_names"] else None


def budget_of(pc: dict, name) -> int:
    return int(pc["budgets"][level_index(pc, name)])


# ─────────────────────────────────────────────────────────────── genes alpha / beta
def _dec(v, choices):
    n = len(choices)
    return choices[min(max(int(float(v) * n), 0), n - 1)]


def _enc(value, choices) -> float:
    i = min(range(len(choices)), key=lambda j: (abs(choices[j] - float(value)), j))
    return (i + 0.5) / len(choices)


def decode_coeffs(tail, pc: dict) -> Tuple[float, float]:
    return float(_dec(tail[0], pc["alphas"])), float(_dec(tail[1], pc["betas"]))


def encode_coeffs(alpha, beta, pc: dict) -> List[float]:
    return [_enc(alpha, pc["alphas"]), _enc(beta, pc["betas"])]


def family_id(base: dict, alpha: float, beta: float) -> str:
    from ga_search.modehb.modehb import arch_hash
    blob = f"{arch_hash(base)}|{float(alpha):g}|{float(beta):g}"
    return "F" + hashlib.sha256(blob.encode()).hexdigest()[:10]


# ─────────────────────────────────────────────────────────────── stages / compound scaling
def active(g: dict) -> List[dict]:
    return [dict(L) for L in g["layers"][:int(g["depth"])]]


def stages(g: dict) -> List[List[int]]:
    """Active layer indices grouped by stage (a stage ends at a pooled layer)."""
    out, cur = [], []
    for i, L in enumerate(g["layers"][:int(g["depth"])]):
        cur.append(i)
        if L.get("pool", "none") != "none":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _identity_ok(feed: dict) -> bool:
    """The feeding layer's output is non-negative (relu/relu6 last, no skip added after it)."""
    return feed.get("activation") in NONNEG_ACTS and not feed.get("residual")


def slots(g: dict) -> List[Optional[dict]]:
    """Per stage the insertion slot of depth growth (None: the stage is only the stem).
    ``pos`` = index the new layer takes, ``feed`` = layer feeding it, ``template`` = layer whose
    operator the new layer copies, ``n`` = repeatable layers of the stage (stem excluded)."""
    out, layers = [], g["layers"]
    for s, st in enumerate(stages(g)):
        rep = [i for i in st if i != 0]
        if not rep:
            out.append(None)
            continue
        last = st[-1]
        pooled = layers[last].get("pool", "none") != "none"
        pos = last if pooled else last + 1
        out.append({"stage": s, "pos": pos, "feed": pos - 1, "template": last, "n": len(rep), "pooled": pooled})
    return out


def new_layer(layers: List[dict], sl: dict, mode: str, residual_activation: str = "silu") -> dict:
    """Genome layer inserted at slot ``sl`` (width = feeding width, pool none), shaped so that
    ``external_seed.grow_blocks`` builds it function-preserving. ``mode``: identity (Net2Deeper relu
    conv, needs a relu-ended feeding layer) | residual (zero-init residual block, ``residual_activation``)
    | auto (identity when possible, else residual)."""
    t, f = layers[sl["template"]], layers[sl["feed"]]
    if mode in ("identity", "auto") and _identity_ok(f):
        k = int(t.get("kernel", 3)) if t.get("conv") == "conv" else 3
        return {"channels": int(f["channels"]), "kernel": k, "conv": "conv", "residual": False,
                "activation": "relu", "norm": "none", "pool": "none", "softhebb": False}
    k = int(t.get("kernel", 3)) if t.get("conv") == "conv" else 3
    return {"channels": int(f["channels"]), "kernel": k, "conv": "conv", "residual": True,
            "activation": residual_activation, "norm": "none", "pool": "none", "softhebb": False}


def _pad_layers(g: dict, layers: List[dict], n_layers: int) -> dict:
    g = copy.deepcopy(g)
    d = len(layers)
    full = [dict(L) for L in layers]
    while len(full) < n_layers:
        full.append({**dict(layers[-1]), "pool": "none"})
    g["layers"] = full[:max(n_layers, d)]
    g["depth"] = d
    return g


def _round(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def stage_additions(base: dict, d_mult: float, max_depth: int = 8) -> List[int]:
    """Layers added (< 0: removed) per stage for depth multiplier ``d_mult``: round(n_s * d) - n_s.
    Under the depth cap, the added units are taken in the order of the multiplier at which each
    appears (ties: later stage first), so the additions are MONOTONE in ``d_mult`` per stage (a
    grown member always contains its parent)."""
    sl = slots(base)
    d = float(d_mult)
    if d < 1.0:
        return [0 if s is None else max(1, _round(s["n"] * d)) - s["n"] for s in sl]
    units = []
    for k, s in enumerate(sl):
        if s is None:
            continue
        n = s["n"]
        for u in range(1, max(0, _round(n * d) - n) + 1):
            units.append(((n + u - 0.5) / n, -k, k))
    room = max(0, int(max_depth) - int(base["depth"]))
    adds = [0] * len(sl)
    for _, _, k in sorted(units)[:room]:
        adds[k] += 1
    return adds


def scale_genome(base: dict, d_mult: float, w_mult: float, *, max_depth: int = 8, cmax: int = 1024,
                 deepen: str = "auto", n_layers: int = 8, residual_activation: str = "silu") -> dict:
    """Compound-scaled copy of ``base`` (NOT repaired): per stage round(n * d_mult) repeatable
    layers (``stage_additions``), every width x ``w_mult`` (inserted layers: cin = cout)."""
    layers = active(base)
    st, sl = stages(base), slots(base)
    adds = stage_additions(base, d_mult, max_depth)
    seq: List[Tuple[str, object]] = []
    for s_idx, idx in enumerate(st):
        part = [("orig", i) for i in idx]
        s, a = sl[s_idx], adds[s_idx]
        if s is not None and a:
            k = len(part) - 1 if s["pooled"] else len(part)
            if a > 0:
                part = part[:k] + [("new", s)] * a + part[k:]
            else:
                removable = [j for j in range(k) if part[j][1] != 0]
                keep_min = 1 if not s["pooled"] else 0        # pooled stage keeps its pooled layer
                drop = set(removable[::-1][:max(0, min(-a, len(removable) - keep_min))])
                part = [p for j, p in enumerate(part) if j not in drop]
        seq.extend(part)
    out: List[dict] = []
    for kind, ref in seq:
        if kind == "orig":
            L = dict(layers[ref])
            L["channels"] = int(min(cmax, max(1, _round(int(L["channels"]) * float(w_mult)))))
        else:
            L = new_layer(layers, ref, deepen, residual_activation)
            L["channels"] = int(out[-1]["channels"]) if out else int(L["channels"])
        out.append(L)
    g = _pad_layers(base, out, max(n_layers, len(out)))
    g.pop("external", None)            # a scaled member is a normal genome (inherits through ctx["inherit"])
    return g


def fit_level(base: dict, alpha: float, beta: float, budget: int, count: Callable[[dict], int],
              rep: Callable[[dict], dict], *, cap_params: int, cap_count: Optional[Callable] = None,
              feasible: Optional[Callable[[dict], bool]] = None, tol: float = 0.10, max_depth: int = 8,
              cmax: int = 1024, deepen: str = "auto", min_params: Optional[int] = None,
              min_mults: Tuple[Optional[float], Optional[float]] = (None, None),
              phi_bounds=(-30.0, 30.0), n_scan: int = 41, iters: int = 30,
              n_layers: int = 8, residual_activation: str = "silu") -> Optional[Tuple[dict, dict]]:
    """(genome, info): repaired member of the family (base, alpha, beta) whose ``count`` is the
    closest (log scale) to ``budget``, subject to ``cap_count`` <= ``cap_params``, ``feasible``
    (GPU memory) and, for a growth, ``count`` > ``min_params`` with multipliers >= ``min_mults``
    (the child contains the parent). 1) scan phi: d = alpha**phi, w = beta**phi; 2) for the best
    depth multipliers, bisection on the width multiplier alone (depth is discrete: the parameter
    count jumps when a layer is added; the width refinement lands inside the tolerance). None if
    nothing is admissible."""
    cap_count = cap_count or count
    d_min, w_min = (float(min_mults[0]) if min_mults[0] else 0.0), (float(min_mults[1]) if min_mults[1] else 0.0)
    cache = {}

    def build(d, w):
        key = (round(math.log(max(d, 1e-9)), 9), round(math.log(max(w, 1e-9)), 9))
        if key not in cache:
            g = rep(scale_genome(base, d, w, max_depth=max_depth, cmax=cmax, deepen=deepen, n_layers=n_layers,
                                 residual_activation=residual_activation))
            p = int(count(g))
            okk = (int(cap_count(g)) <= cap_params and (min_params is None or p > int(min_params))
                   and (feasible is None or bool(feasible(g))))
            cache[key] = (g, p, okk)
        return cache[key]

    def err(p):
        return abs(math.log(max(p, 1) / float(budget)))

    tried = []                                    # (err, d, w, g, p)
    for phi in np.linspace(float(phi_bounds[0]), float(phi_bounds[1]), int(n_scan)):
        d, w = max(float(alpha) ** phi, d_min), max(float(beta) ** phi, w_min)
        g, p, okk = build(d, w)
        if okk:
            tried.append((err(p), d, w, g, p, float(phi)))
    if not tried and d_min == 0.0 and w_min == 0.0:
        return None
    if not tried:                                 # growth: start the refinement at the parent's multipliers
        g, p, okk = build(max(d_min, 1e-9), max(w_min, 1e-9))
        tried = [(err(p), max(d_min, 1e-9), max(w_min, 1e-9), g, p, 0.0)]
    tried.sort(key=lambda t: t[0])
    best = tried[0] if build(tried[0][1], tried[0][2])[2] else None
    seen_depth = []
    for e0, d, w0, g0, p0, phi in tried:
        depth = int(g0["depth"])
        if depth in seen_depth:
            continue
        seen_depth.append(depth)
        if len(seen_depth) > 3:
            break
        lo = math.log(max(w_min, w0 / 16.0, 1e-6))
        hi = math.log(w0 * 16.0)
        for _ in range(int(iters)):
            m = 0.5 * (lo + hi)
            g, p, okk = build(d, math.exp(m))
            if okk and (best is None or err(p) < best[0] - 1e-12):
                best = (err(p), d, math.exp(m), g, p, phi)
            if not okk:                           # too small for a growth -> wider; cap/memory -> narrower
                lo, hi = (m, hi) if (min_params is not None and p <= int(min_params)) else (lo, m)
            elif p < budget:
                lo = m
            else:
                hi = m
            if hi - lo < 1e-4:
                break
    if best is None:
        return None
    e, d, w, g, p, phi = best
    info = {"phi": round(float(phi), 4), "alpha": float(alpha), "beta": float(beta),
            "d_mult": float(d), "w_mult": float(w), "budget": int(budget), "params_est": int(p),
            "rel_err": float(p / float(budget) - 1.0), "within_tol": bool(abs(p / float(budget) - 1.0) <= float(tol))}
    return g, info


# ─────────────────────────────────────────────────────────────── morph plan / mutations
def align(parent: dict, child: dict) -> Optional[List[Optional[int]]]:
    """Child layer -> parent layer index (None = inserted), by the insertion convention of
    ``scale_genome`` / ``mutate``; None when the child is not a growth of the parent."""
    ps, cs = stages(parent), stages(child)
    if len(ps) != len(cs):
        return None
    P = parent["layers"]
    out: List[Optional[int]] = []
    for sp, sc in zip(ps, cs):
        extra = len(sc) - len(sp)
        if extra < 0:
            return None
        pooled = P[sp[-1]].get("pool", "none") != "none"
        k = len(sp) - 1 if pooled else len(sp)
        if extra and not [i for i in sp if i != 0]:
            return None                       # the stem stage never grows
        out.extend(sp[:k] + [None] * extra + sp[k:])
    return out


def layer_map(parent: dict, child: dict) -> Optional[list]:
    """``external_seed.grow_blocks`` layer map parent -> child (int = parent layer, "identity",
    "residual_zero"), or None when the child is not a pure growth of the parent (a layer shrinks,
    an operator changes, or an inserted layer is not identity/zero-residual shaped). Aligned layers
    may be non-plain: grow_blocks then copies them verbatim if unchanged, else re-fits from there."""
    from ga_search.modehb.external_seed import ZERO_ACTS
    m = align(parent, child)
    if m is None:
        return None
    P, C = parent["layers"], child["layers"]
    out, changed = [], False
    for i, q in enumerate(m):
        L = C[i]
        if q is None:
            if (i == 0 or L.get("pool", "none") != "none" or int(L["channels"]) != int(C[i - 1]["channels"])
                    or L.get("conv") != "conv" or L.get("norm", "none") != "none" or L.get("softhebb")):
                return None
            if L.get("residual") and L.get("activation") in ZERO_ACTS:
                out.append("residual_zero")
            elif not L.get("residual") and L.get("activation") == "relu" and _identity_ok(C[i - 1]):
                out.append("identity")
            else:
                return None
            changed = True
            continue
        A = P[q]
        if any(A.get(k) != L.get(k) for k in SPEC_KEYS) or bool(A.get("softhebb")) != bool(L.get("softhebb")):
            return None
        if int(L["channels"]) < int(A["channels"]):
            return None
        changed = changed or int(L["channels"]) != int(A["channels"])
        out.append(int(q))
    head_changed = (parent.get("pooling") != child.get("pooling")
                    or int(parent.get("embed_dim", 1280)) != int(child.get("embed_dim", 1280)))
    return out if (changed or head_changed) else None


def insert_layer(g: dict, sl: dict, mode: str, n_layers: int = 8, residual_activation: str = "silu") -> dict:
    layers = active(g)
    L = new_layer(layers, sl, mode, residual_activation)
    layers = layers[:sl["pos"]] + [L] + layers[sl["pos"]:]
    return _pad_layers(g, layers, max(n_layers, len(layers)))


def mutate(genome: dict, op: str, rng, pc: dict, *, poolings: Sequence[str] = ("gap", "gem", "moments"),
           pools: Sequence[str] = ("max", "avg"), max_depth: int = 8, n_layers: int = 8):
    """``genome`` + one structural mutation ``op`` (NOT repaired), or None if not applicable."""
    g = copy.deepcopy(genome)
    g.pop("external", None)
    d = int(g["depth"])
    if op in ("add_layer", "add_residual"):
        if d >= max_depth:
            return None
        cand = [s for s in slots(g) if s is not None]
        if op == "add_layer":
            cand = [s for s in cand if _identity_ok(g["layers"][s["feed"]])]
        if not cand:
            return None
        s = cand[int(rng.integers(len(cand)))]
        mode = "identity" if op == "add_layer" else "residual"
        return (insert_layer(g, s, mode, n_layers, pc.get("residual_activation", "silu")),
                {"mutation": op, "stage": s["stage"], "position": s["pos"]})
    if op == "widen":
        st = stages(g)
        s = int(rng.integers(len(st)))
        f = float(rng.choice([float(x) for x in pc["widen_factors"]]))
        for i in st[s]:
            c = int(g["layers"][i]["channels"])
            g["layers"][i]["channels"] = int(min(1024, max(c + 1, math.ceil(c * f))))
        return g, {"mutation": op, "stage": s, "factor": f}
    if op == "pooling":
        finals = [p for p in poolings if p != g.get("pooling")]
        pooled = [i for i in range(d) if g["layers"][i].get("pool", "none") != "none"
                  and any(p != g["layers"][i]["pool"] for p in pools)]
        kinds = (["final"] if finals else []) + (["stage"] if pooled else [])
        if not kinds:
            return None
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "final":
            new = finals[int(rng.integers(len(finals)))]
            info = {"mutation": op, "kind": "final", "from": g.get("pooling"), "to": new}
            g["pooling"] = new
            return g, info
        i = pooled[int(rng.integers(len(pooled)))]
        alt = [p for p in pools if p != g["layers"][i]["pool"]]
        new = alt[int(rng.integers(len(alt)))]
        info = {"mutation": op, "kind": "stage", "layer": i, "from": g["layers"][i]["pool"], "to": new}
        g["layers"][i]["pool"] = new
        return g, info
    raise ValueError(f"unknown mutation {op!r}")


# ─────────────────────────────────────────────────────────────── seeds (GA checkpoints)
def _structure(g: dict) -> list:
    from ga_search.modehb.modehb import LAYER_GENES
    return [{k: L.get(k) for k in LAYER_GENES} for L in g["layers"][:int(g["depth"])]] + \
           [g.get("pooling"), int(g.get("embed_dim", 1280))]


def checkpoint_inherit(sd: dict) -> dict:
    """``ctx["inherit"]`` that imports a GA checkpoint 1:1 through ``_grow_fp`` (identity layer map:
    every block copied verbatim, all elements inherited -> protect x lr in SSL; new head)."""
    from ga_search.modehb.external_seed import active_layers
    g = sd["genome"]
    return {"checkpoint": sd["checkpoint"], "layer_map": list(range(int(g["depth"]))),
            "parent_layers": active_layers(g), "n_blocks": int(g["depth"]),
            "parent_cand_id": sd["name"], "parent_fidelity": 0.0}


def discover_seeds(pc: dict, dataset: Optional[str], out_dir, seed: int, n_layers: int = 8,
                   log=print) -> List[dict]:
    """GA-checkpoint seed manifest (frozen in ``<out_dir>/pnas_seeds.json`` on the first call): explicit
    paths and/or the best max-fidelity jobs of finished searches by VALIDATION quality (never test).
    A missing file only drops that seed (warning). The external encoder is ``external_seed``'s."""
    path = Path(out_dir) / "pnas_seeds.json"
    if path.exists():
        return json.loads(path.read_text())
    sc, out = pc["seeds"], []
    paths = list((sc.get("checkpoints") or {}).get(dataset, []) or [])
    for root in sc.get("best_from") or []:
        paths += _best_checkpoints(Path(str(root).format(dataset=dataset)), int(sc.get("best_k", 0)), log)
    seen = set()
    for p in paths:
        p = Path(p)
        if str(p) in seen:
            continue
        seen.add(str(p))
        if not p.exists():
            log(f"[pnas_cs] WARNING checkpoint seed missing: {p}")
            continue
        try:
            import torch
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            g = copy.deepcopy(ck["genome"])
            g["seed"] = int(seed)
            g.pop("external", None)
            if int(g["depth"]) > n_layers or len(ck["encoder"]) != int(g["depth"]):
                raise ValueError("depth / encoder length mismatch")
            peak = None                               # measured peak of the source job (fp32, its own SSL)
            rj = p.parent / "result.json"
            if rj.exists():
                try:
                    peak = json.loads(rj.read_text()).get("peak_gpu_mem_gb")
                except Exception:
                    peak = None
            views = {"lejepa": 4, "simclr": 2, "ae": 1}.get(ck["genome"].get("ssl", "lejepa"), 4)
            out.append({"kind": "checkpoint", "name": f"ckpt:{p.parent.parent.name}/{p.parent.name}",
                        "checkpoint": str(p), "genome": g,
                        "src_peak_gb": float(peak) if peak is not None else None, "src_views": views})
        except Exception as exc:
            log(f"[pnas_cs] checkpoint seed {p} skipped: {type(exc).__name__}: {exc}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=str))
    tmp.replace(path)
    return out


def _best_checkpoints(state_path: Path, k: int, log=print) -> List[str]:
    """Top-k max-fidelity search jobs of a finished search by VALIDATION quality (never test)."""
    if k <= 0:
        return []
    try:
        s = json.loads(state_path.read_text())
    except Exception as exc:
        log(f"[pnas_cs] best_from {state_path} unreadable: {exc}")
        return []
    jobs = [j for j in s.get("jobs", []) if j.get("status") == "ok" and j.get("role") == "search"
            and (j.get("result") or {}).get("checkpoint")]
    if not jobs:
        return []
    top = max(float(j["fidelity"]) for j in jobs)
    jobs = sorted((j for j in jobs if float(j["fidelity"]) == top),
                  key=lambda j: -float(j["result"].get("quality", 0.0)))
    out = []
    for j in jobs[:k]:
        p = Path(j["result"]["checkpoint"])
        if not p.exists() and "candidates" in p.parts:
            p = state_path.parent.joinpath(*p.parts[p.parts.index("candidates"):])
        out.append(str(p))
    return out


# ─────────────────────────────────────────────────────────────── reporting (pure python)
def _kappa(j):
    r = j.get("result") or {}
    k = r.get("fitness_kappa")
    return float(k if k is not None else r.get("quality", 0.0))


def _front(points):
    """Indices of the non-dominated points of [(1-kappa, flops)] (minimisation)."""
    idx = []
    for i, (a0, a1) in enumerate(points):
        if not any((b0 <= a0 and b1 <= a1) and (b0 < a0 or b1 < a1) for j, (b0, b1) in enumerate(points) if j != i):
            idx.append(i)
    return idx


def fit_scaling_law(params: Sequence[float], kappas: Sequence[float], sat_gain: float = 0.01) -> Optional[dict]:
    """kappa = a + b log10(params) (least squares) + saturation: the first point after which the
    gain per doubling of params falls below ``sat_gain``."""
    p = np.asarray(params, float)
    k = np.asarray(kappas, float)
    ok = np.isfinite(p) & np.isfinite(k) & (p > 0)
    p, k = p[ok], k[ok]
    if len(p) < 2 or np.ptp(np.log10(p)) <= 0:
        return None
    x = np.log10(p)
    b, a = np.polyfit(x, k, 1)
    pred = a + b * x
    ss = float(np.sum((k - k.mean()) ** 2))
    r2 = 1.0 - float(np.sum((k - pred) ** 2)) / ss if ss > 0 else 1.0
    order = np.argsort(p)
    sat = None
    for i0, i1 in zip(order, order[1:]):
        doublings = math.log2(p[i1] / p[i0]) if p[i1] > p[i0] else 0.0
        if doublings > 0 and (k[i1] - k[i0]) / doublings < sat_gain:
            sat = float(p[i0])
            break
    return {"slope_per_decade": float(b), "intercept": float(a), "r2": float(r2), "n": int(len(p)),
            "saturation_params": sat, "best_kappa": float(k.max())}


def report(state: dict, fidelities: Sequence[float], pc: dict, ref_point: float = 1.1) -> dict:
    """W&B scalars + tables (cols, rows, plot spec) of the PNAS-CS panels, from the job log only."""
    from ga_search.modehb.modehb import hypervolume
    top = float(fidelities[-1])
    jobs = state.get("jobs", [])
    ok = [j for j in jobs if j.get("status") == "ok" and j.get("role") == "search"]
    ok = list({(j["cand_id"], float(j["fidelity"])): j for j in ok}.values())   # cached repeats counted once
    sc, tables = {}, {}
    ref = np.full(2, float(ref_point))
    hv_top = hypervolume([j["F"] for j in ok if float(j["fidelity"]) == top], ref) if ok else 0.0
    best_f = max((float(j["fidelity"]) for j in ok), default=None)
    hv_best = hypervolume([j["F"] for j in ok if float(j["fidelity"]) == best_f], ref) if ok else 0.0
    sc.update({"pnas/hv_max_fidelity": hv_top, "pnas/hv_highest_reached": hv_best,
               "pnas/highest_fidelity_reached": best_f or 0.0, "pnas/n_jobs": len(jobs),
               "pnas/gpu_hours": float(state.get("gpu_hours", 0.0))})
    tables["hv_vs_gpu_hours"] = (["n_jobs", "gpu_hours", "hv_max_fidelity", "hv_highest_reached"],
                                 [list(r) for r in state.get("pnas_hv_history", [])],
                                 ("line", "gpu_hours", "hv_highest_reached", None))
    fam = lambda j: ((j.get("genome") or {}).get("pnas") or {}).get("family")
    rows = []
    for j in ok:
        r = j["result"]
        rows.append([j["cand_id"], str(fam(j)), str(j.get("stratum")), float(j["fidelity"]), r.get("params"),
                     float(r.get("flops") or 0) / 1e9, _kappa(j), float(r.get("quality", 0.0)),
                     str((j.get("op") or {}).get("type"))])
    tables["kappa_vs_params"] = (["cand_id", "family", "level", "fidelity", "params", "gflops", "kappa", "quality", "op"],
                                 rows, ("scatter", "params", "kappa", None))
    # live Pareto kappa x FLOPs, per fidelity (fidelities are not comparable)
    prow = []
    for f in sorted({float(j["fidelity"]) for j in ok}):
        sub = [j for j in ok if float(j["fidelity"]) == f]
        for i in _front([(1.0 - _kappa(j), float(j["result"].get("flops") or 0)) for j in sub]):
            j = sub[i]
            prow.append([f, j["cand_id"], str(j.get("stratum")), float(j["result"].get("flops") or 0) / 1e9,
                         _kappa(j), j["result"].get("params")])
    prow.sort(key=lambda r: (r[0], r[3]))
    tables["pareto_kappa_flops"] = (["fidelity", "cand_id", "level", "gflops", "kappa", "params"], prow,
                                    ("line", "gflops", "kappa", "stroke:fidelity"))
    tables["pareto_evolution"] = (["n_jobs", "bracket", "fidelity", "cand_id", "gflops", "kappa"],
                                  [list(r) for r in state.get("pnas_pareto_history", [])],
                                  ("scatter", "gflops", "kappa", None))
    # growth curves per family: best (highest-fidelity) record of each member
    mem = {}
    for j in ok:
        key = (fam(j), j["cand_id"])
        if key not in mem or float(j["fidelity"]) >= float(mem[key]["fidelity"]):
            mem[key] = j
    grow_rows = sorted([[str(f), str(j.get("stratum")), c, j["result"].get("params"), _kappa(j), float(j["fidelity"])]
                        for (f, c), j in mem.items() if f], key=lambda r: (r[0], r[3] or 0))
    tables["growth_curves"] = (["family", "level", "cand_id", "params", "kappa", "fidelity"], grow_rows,
                               ("line", "params", "kappa", "stroke:family"))
    dw = []
    for (f, c), j in mem.items():
        L = j["genome"]["layers"][:int(j["genome"]["depth"])]
        w = [int(x["channels"]) for x in L]
        dw.append([c, int(j["genome"]["depth"]), float(np.mean(w)), int(max(w)), j["result"].get("params"),
                   _kappa(j), str(j.get("stratum"))])
    tables["depth_x_width"] = (["cand_id", "depth", "mean_width", "max_width", "params", "kappa", "level"], dw,
                               ("scatter", "depth", "mean_width", None))
    # marginal gain per expansion (grow jobs vs their parent at the SAME fidelity when available)
    by_cf = {(j["cand_id"], float(j["fidelity"])): j for j in ok}
    marg = []
    for j in ok:
        op = j.get("op") or {}
        if op.get("type") != "grow" or j.get("resumed_from") is not None:
            continue
        par = by_cf.get((op.get("parent"), float(j["fidelity"])))
        if par is None:
            pf = [k for (c, k) in by_cf if c == op.get("parent")]
            par = by_cf[(op.get("parent"), max(pf))] if pf else None
        if par is None or not par["result"].get("params") or not j["result"].get("params"):
            continue
        dk = _kappa(j) - _kappa(par)
        dl = math.log10(float(j["result"]["params"]) / float(par["result"]["params"]))
        marg.append([str(op.get("family")), f"{op.get('from_level')}->{op.get('to_level')}", op.get("parent"),
                     j["cand_id"], float(j["fidelity"]), dk, dl, dk / dl if dl > 0 else None])
    tables["marginal_gain"] = (["family", "expansion", "parent", "child", "fidelity", "delta_kappa",
                                "delta_log10_params", "dkappa_per_decade"], marg,
                               ("bar", "expansion", "delta_kappa", None))
    if marg:
        sc["pnas/marginal/mean_delta_kappa"] = float(np.mean([r[5] for r in marg]))
    # mutation / operator survival: promoted (reached a higher fidelity) and on the current front
    fids_of = {}
    for j in ok:
        fids_of.setdefault(j["cand_id"], set()).add(float(j["fidelity"]))
    front_ids = {r[1] for r in prow}
    surv = {}
    for j in ok:
        op = j.get("op") or {}
        if j.get("resumed_from") is not None or op.get("type") in (None, "promotion", "extension"):
            continue
        t = op.get("mutation") or op.get("type")
        s = surv.setdefault(t, {"n": 0, "promoted": 0, "front": 0})
        s["n"] += 1
        s["promoted"] += int(any(f > float(j["fidelity"]) for f in fids_of[j["cand_id"]]))
        s["front"] += int(j["cand_id"] in front_ids)
    srow = []
    for t, s in sorted(surv.items()):
        pr, fr = s["promoted"] / s["n"], s["front"] / s["n"]
        sc[f"pnas/survival/{t}/promoted_rate"] = pr
        sc[f"pnas/survival/{t}/front_rate"] = fr
        sc[f"pnas/survival/{t}/n"] = s["n"]
        srow.append([t, s["n"], s["promoted"], pr, s["front"], fr])
    tables["mutation_survival"] = (["operator", "n", "n_promoted", "promoted_rate", "n_front", "front_rate"], srow,
                                   ("bar", "operator", "promoted_rate", None))
    # scaling laws per family (members at the highest fidelity they share) + global (best per level)
    law = []
    fams = {}
    for (f, c), j in mem.items():
        if f:
            fams.setdefault(f, []).append(j)
    for f, js in sorted(fams.items()):
        common = min(float(j["fidelity"]) for j in js)
        pts = [(by_cf.get((j["cand_id"], common)) or j) for j in js]
        res = fit_scaling_law([p["result"].get("params") or 0 for p in pts], [_kappa(p) for p in pts],
                              float(pc.get("saturation_gain", 0.01)))
        if res:
            law.append([f, res["n"], common, res["slope_per_decade"], res["intercept"], res["r2"],
                        res["saturation_params"], res["best_kappa"]])
    best_lv = {}
    for j in ok:
        if float(j["fidelity"]) != (best_f or top):
            continue
        s = j.get("stratum")
        if s not in best_lv or _kappa(j) > _kappa(best_lv[s]):
            best_lv[s] = j
    g = fit_scaling_law([j["result"].get("params") or 0 for j in best_lv.values()],
                        [_kappa(j) for j in best_lv.values()], float(pc.get("saturation_gain", 0.01)))
    if g:
        law.append(["GLOBAL(best per level)", g["n"], best_f, g["slope_per_decade"], g["intercept"], g["r2"],
                    g["saturation_params"], g["best_kappa"]])
        sc["pnas/scaling/slope_per_decade"] = g["slope_per_decade"]
        if g["saturation_params"]:
            sc["pnas/scaling/saturation_params"] = g["saturation_params"]
    tables["scaling_law"] = (["family", "n", "fidelity", "slope_per_decade", "intercept", "r2", "saturation_params",
                              "best_kappa"], law, ("bar", "family", "slope_per_decade", None))
    for s, j in best_lv.items():
        sc[f"pnas/best_kappa/{s}"] = _kappa(j)
    sc["pnas/n_families"] = len(fams)
    lv = [level_index(pc, j.get("stratum")) for j in ok if level_index(pc, j.get("stratum")) is not None]
    sc["pnas/max_level_reached"] = max(lv) if lv else -1
    fp = [((j["result"].get("pnas") or {}).get("fp") or {}) for j in ok]
    errs = [m["rel_err"] for m in fp if m.get("rel_err") is not None]
    sc["pnas/morph/n_fp_children"] = sum(1 for m in fp if m.get("n_fp_layers"))
    if errs:
        sc["pnas/morph/max_rel_err"] = float(max(errs))
    ph = state.get("phase0") or {}
    prow0 = [[r["seed_index"], r["variant"], r["cand_id"], r["fidelity"], r["kappa"], r["quality"]]
             for r in ph.get("rows", [])]
    tables["phase0_ae_ablation"] = (["seed", "variant", "cand_id", "fidelity", "val_kappa", "quality"], prow0,
                                    ("bar", "variant", "val_kappa", None))
    for v, q in (ph.get("mean_kappa") or {}).items():
        sc[f"pnas/phase0/{v}_mean_val_kappa"] = q
    return {"scalars": sc, "tables": tables}
