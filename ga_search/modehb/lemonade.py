"""LEMONADE-style warm start + network morphisms for MO-DEHB (opt-in YAML ``lemonade:``).

Elsken, Metzen & Hutter, "Efficient Multi-objective Neural Architecture Search via Lamarckian
Evolution" (ICLR 2019, arXiv 1804.09081). Three pieces, all inactive unless
``lemonade.enabled: true`` (search.yaml top level, translated to ``search.lemonade``):

1. **Warm start** (``seed_genome`` + ``descendants``): the initial population gets one coherent
   hybrid SPiFiL seed (default: the best small remote_sensing SPiFiL, widths
   40-80-152-152-152-312, conv k3, relu, batch norm, max-pool on layers 1-4 = pools
   2-2-2-2-1-1 of ``hyb_ssl_encoder_remote_sensing_jaci.yaml``) plus ``n_descendants`` children,
   each = seed + ONE morphism.
2. **Network morphisms as mutation operators** (``morph`` / ``random_morph``): child = parent
   architecture + one of

   * ``deepen``  : insert a width-preserving layer (cout = cin, no pool, kernel 3, no residual:
                   the identity-shaped insertion of LEMONADE's "insert layer") after layer p-1;
   * ``widen``   : multiply the width of one layer by a factor (monotonic repair widens the next);
   * ``residual``: add a skip connection to one layer (i > 0; SPiFiL 1x1 projection on mismatch);
   * ``kernel``  : change the kernel size of one layer.

   With probability ``morph_prob`` an offspring is a morphism of the DE target instead of a DE
   mutant (``morph_random_fill``: also for the random fill of the lowest-fidelity rung, parent =
   a random member of the first non-dominated front).
3. **Lamarckian inheritance** (``inherit_spec``): the child reuses the parent's TRAINED blocks
   for the unchanged leading layers (``reusable_prefix``); only new/changed layers (and every
   layer after the first change, whose input changed) get SPiFiL init + the per-layer local AE.
   Passed to the evaluator as ``ctx["inherit"] = {"checkpoint", "n_blocks", ...}``
   (``run_search.Evaluator._grow(inherit=...)``); absent = original behaviour.
"""
from __future__ import annotations

import copy
import math
from typing import Callable, List, Optional, Sequence, Tuple

LAYER_KEYS = ("channels", "kernel", "conv", "residual", "activation", "norm", "pool", "softhebb")
MORPHISMS = ("deepen", "widen", "residual", "kernel")

DEFAULT_SEED = dict(widths=[40, 80, 152, 152, 152, 312], conv="conv", kernel=3, activation="relu",
                    norm="batch", pool=["max", "max", "max", "max", "none", "none"], residual=False,
                    softhebb=False, pooling="gem", ssl="lejepa", reg="sigreg", embed_dim=1280,
                    ae_target="pixel", projector=False, protect="full", ae_schedule=None,
                    spifil_images_per_class=None)
DEFAULTS = dict(enabled=False, seed=None, seeds=None, n_descendants=5, morph_prob=0.5, morph_random_fill=True,
                operators=list(MORPHISMS), widen_factors=[1.25, 1.5, 2.0], kernels=[3, 5, 7],
                inherit=True, inherit_fidelity="le_child", max_tries=8, cmax=1024,
                # opt-in: Net2Net-style inheritance (external_seed.grow_blocks): widen keeps the trained
                # filters + new SPiFiL filters (consumer columns 0), deepen inserts an identity layer,
                # kernel growth zero-pads; layers after a change are reused instead of re-fitted
                function_preserving=False)


def lemonade_config(cfg: Optional[dict]) -> Optional[dict]:
    """Normalised config, or None when absent/disabled (original MO-DEHB)."""
    if not cfg or not cfg.get("enabled"):
        return None
    out = {**DEFAULTS, **cfg}
    # ``seeds`` (list) = several warm-start seeds; ``seed`` (one mapping) kept as a shorthand
    raw = list(cfg.get("seeds") or []) or [cfg.get("seed") or {}]
    out["seeds"] = [{**DEFAULT_SEED, **(s or {})} for s in raw]
    out["seed"] = out["seeds"][0]
    bad = [o for o in out["operators"] if o not in MORPHISMS]
    if bad or not out["operators"]:
        raise ValueError(f"lemonade.operators: {bad or 'empty'} not in {MORPHISMS}")
    if out["inherit_fidelity"] not in ("le_child", "max"):
        raise ValueError("lemonade.inherit_fidelity must be 'le_child' or 'max'")
    return out


def _per_layer(v, n):
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


def seed_genome(lc: dict, seed: int, repair: Callable[[dict], dict], n_layers: int = 8,
                spec: Optional[dict] = None) -> dict:
    """A hybrid SPiFiL warm-start genome (``spec``, default the first seed), repaired with ``repair``."""
    s = spec if spec is not None else lc["seed"]
    w = [int(c) for c in s["widths"]]
    d = len(w)
    if not 1 <= d <= n_layers:
        raise ValueError(f"lemonade.seed.widths: depth {d} not in 1..{n_layers}")
    per = {k: _per_layer(s[k], d) for k in ("conv", "kernel", "activation", "norm", "pool", "residual", "softhebb")}
    layers = []
    for i in range(n_layers):
        j = min(i, d - 1)       # inactive layers: copy of the last active one
        layers.append({"channels": w[j], "kernel": int(per["kernel"][j]), "conv": per["conv"][j],
                       "residual": bool(per["residual"][j]), "activation": per["activation"][j],
                       "norm": per["norm"][j], "pool": per["pool"][j] if i < d else "none",
                       "softhebb": bool(per["softhebb"][j])})
    g = {"init": "spifil", "seed": int(seed), "depth": d, "layers": layers,
         **{k: s[k] for k in ("ssl", "reg", "embed_dim", "pooling", "ae_target", "projector", "protect")}}
    if s.get("ae_schedule"):
        g["ae_schedule"] = s["ae_schedule"]
    if s.get("spifil_images_per_class"):
        g["spifil_images_per_class"] = int(s["spifil_images_per_class"])
    return repair(g)


def _plain_genome(genome: dict) -> dict:
    """``genome`` without the external-seed marker (what a morphed child is compared against)."""
    return {k: v for k, v in genome.items() if k != "external"}


def widths(genome: dict) -> List[int]:
    return [int(L["channels"]) for L in genome["layers"][:int(genome["depth"])]]


def _layer(L: dict) -> dict:
    return {k: L.get(k) for k in LAYER_KEYS}


def reusable_prefix(parent: dict, child: dict) -> int:
    """Number of leading active layers identical in parent and child (same input chain, so the
    parent's trained blocks can be reused as-is). Capped at child depth - 1: the last block and
    the head are always rebuilt."""
    n = 0
    for a, b in zip(parent["layers"][:int(parent["depth"])], child["layers"][:int(child["depth"])]):
        if _layer(a) != _layer(b):
            break
        n += 1
    return min(n, int(child["depth"]) - 1)


def morph(genome: dict, op: str, rng, lc: dict, *, max_depth: int = 8) -> Optional[Tuple[dict, dict]]:
    """``genome`` + one morphism ``op`` (NOT repaired). None when ``op`` is not applicable."""
    g = copy.deepcopy(genome)
    g.pop("external", None)          # a morphed child is a normal genome (inherits via ctx["inherit"])
    d = int(g["depth"])
    layers = g["layers"]
    if op == "deepen":
        if d >= max_depth:
            return None
        p = int(rng.integers(1, d + 1))          # insert at position p (after layer p-1)
        prev = layers[p - 1]
        new = {**prev, "channels": int(prev["channels"]), "pool": "none", "residual": False,
               "kernel": 3, "softhebb": False}
        g["layers"] = layers[:p] + [new] + layers[p:]
        g["layers"] = g["layers"][:len(layers)]
        g["depth"] = d + 1
        return g, {"morphism": "deepen", "layer": p}
    if op == "widen":
        j = int(rng.integers(d))
        c = int(layers[j]["channels"])
        f = float(rng.choice([float(x) for x in lc["widen_factors"]]))
        nc = min(int(lc["cmax"]), max(c + 1, int(math.ceil(c * f))))
        if nc <= c:
            return None
        layers[j]["channels"] = nc
        return g, {"morphism": "widen", "layer": j, "from": c, "to": nc}
    if op == "residual":
        cands = [i for i in range(1, d) if not layers[i].get("residual")]
        if not cands:
            return None
        j = int(rng.choice(cands))
        layers[j]["residual"] = True
        return g, {"morphism": "residual", "layer": j}
    if op == "kernel":
        j = int(rng.integers(d))
        k0 = int(layers[j]["kernel"])
        ks = [int(k) for k in lc["kernels"] if int(k) != k0]
        if not ks:
            return None
        k = int(rng.choice(ks))
        layers[j]["kernel"] = k
        return g, {"morphism": "kernel", "layer": j, "from": k0, "to": k}
    raise ValueError(f"unknown morphism {op!r}")


def random_morph(parent: dict, rng, lc: dict, repair: Callable[[dict], dict], *,
                 max_depth: int = 8) -> Tuple[Optional[dict], Optional[dict]]:
    """Repaired child = parent + one random applicable morphism that SURVIVES the repair
    (the architecture must change). Returns (child, info) or (None, None) after ``max_tries``."""
    from ga_search.modehb.modehb import arch_hash
    h0 = arch_hash(_plain_genome(parent))
    ops = list(lc["operators"])
    for _ in range(int(lc["max_tries"])):
        op = ops[int(rng.integers(len(ops)))]
        out = morph(parent, op, rng, lc, max_depth=max_depth)
        if out is None:
            continue
        child = repair(out[0])
        if arch_hash(child) == h0:
            continue
        info = {**out[1], "n_reuse": reusable_prefix(parent, child),
                "parent_widths": widths(parent), "child_widths": widths(child)}
        return child, info
    return None, None


def descendants(seed: dict, n: int, rng, lc: dict, repair: Callable[[dict], dict], *,
                max_depth: int = 8, op_offset: int = 0) -> List[Tuple[dict, dict]]:
    """``n`` distinct children of the seed, each = seed + one morphism (operators cycled from
    ``operators[op_offset]``)."""
    from ga_search.modehb.modehb import arch_hash
    out, seen = [], {arch_hash(_plain_genome(seed))}
    ops = list(lc["operators"])
    tries = 0
    while len(out) < int(n) and tries < 8 * max(int(n), 1):
        op = ops[(tries + int(op_offset)) % len(ops)]
        tries += 1
        m = morph(seed, op, rng, lc, max_depth=max_depth)
        if m is None:
            continue
        child = repair(m[0])
        h = arch_hash(child)
        if h in seen:
            continue
        seen.add(h)
        out.append((child, {**m[1], "n_reuse": reusable_prefix(seed, child),
                            "parent_widths": widths(seed), "child_widths": widths(child)}))
    return out


def pick_parent_checkpoint(jobs: Sequence[dict], fid: float, rule: str = "le_child") -> Optional[dict]:
    """Parent job whose checkpoint is inherited: highest fidelity <= ``fid`` (``le_child``; the
    lowest one when none is <= fid), or the highest fidelity overall (``max``)."""
    ok = [j for j in jobs if j.get("status") == "ok" and (j.get("result") or {}).get("checkpoint")]
    if not ok:
        return None
    if rule == "max":
        return max(ok, key=lambda j: float(j["fidelity"]))
    le = [j for j in ok if float(j["fidelity"]) <= float(fid)]
    return max(le, key=lambda j: float(j["fidelity"])) if le else min(ok, key=lambda j: float(j["fidelity"]))


def inherit_spec(parent_job: dict, n_blocks: int) -> Optional[dict]:
    """``ctx["inherit"]`` payload for ``run_search.Evaluator``."""
    if parent_job is None or int(n_blocks) < 1:
        return None
    return {"checkpoint": parent_job["result"]["checkpoint"], "n_blocks": int(n_blocks),
            "parent_cand_id": parent_job["cand_id"], "parent_fidelity": float(parent_job["fidelity"]),
            "parent_job": int(parent_job["job"]),
            "effective_specs": list(parent_job["result"].get("effective_specs") or [])[:int(n_blocks)]}


def load_inherited_blocks(inherit: dict, device) -> Tuple[list, list]:
    """(blocks, effective_specs) of the first ``n_blocks`` blocks of the parent's checkpoint
    (the evaluated, i.e. WiSE-selected, encoder)."""
    import torch
    ck = torch.load(inherit["checkpoint"], map_location=device, weights_only=False)
    enc = ck["encoder"]
    n = int(inherit["n_blocks"])
    if n > len(enc):
        raise ValueError(f"inherit: parent has {len(enc)} blocks < n_blocks={n}")
    blocks = [copy.deepcopy(enc[i]).to(device) for i in range(n)]
    specs = list(ck.get("effective_specs") or inherit.get("effective_specs") or [])[:n]
    while len(specs) < n:
        specs.append({"origin": "inherited"})
    return blocks, specs
