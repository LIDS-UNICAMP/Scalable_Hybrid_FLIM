"""External pretrained encoder as a seed of MO-DEHB / LEMONADE + function-preserving growth.

Opt-in (YAML top-level ``external_seed: {enabled: true, ...}``; absent/disabled = nothing changes).

Public API (also used by other experiments)
-------------------------------------------
* ``load_external_seed(base, ec=None)`` -> ``(genome, block_states)``: reads the other team's
  encoder ("SPiFiL pequeno + LeJEPA": ``L0/architecture.json`` + the Lightning checkpoint after
  LeJEPA, projector/decoder discarded) and maps it to OUR representation. ``genome`` is a normal
  MO-DEHB genome (conv k3 / relu / norm none / no residual / pool max on the pooled layers) with the
  marker ``genome["external"]``; ``block_states[i] = {"spec": ops spec, "state_dict": {...}}``.
  ``build_blocks(block_states)`` -> ``nn.Sequential`` of ``spifil_ops.SpifilOpsBlock``.
* Function-preserving morphisms (Net2Net-style), reusable on any parent made of plain dense
  ``SpifilOpsBlock`` s (conv/pointwise, norm none, no skip):
  ``widen_block`` (keep / cut filters, new filters appended, consumer columns for NEW input
  channels = 0), ``identity_block`` (Net2Deeper: centre-1 kernel, bias 0, relu; exact because its
  input is >= 0), ``residual_zero_block`` (x + silu(0 * x) = x; silu/gelu: trainable at 0) and
  ``grow_blocks(parent_blocks, parent_layers, child_layers, layer_map, new_filters=...)``, which
  builds a whole child and returns the masks of the INHERITED elements.
* ``ElementProtect(module, masks, scale)``: per-ELEMENT learning-rate multiplier for the inherited
  elements (``protect`` gene: full 1 / lowlr 0.01 / freeze 0), exact for Adam/AdamW
  (post-step delta scaling == a param group with lr*scale), new elements train at full lr.

Mapping of their encoder (``flim.arch.build_encoder_from_arch`` of the discovery repo)
------------------------------------------------------------------------------------
Each FLIM layer is ``Conv2d(k, padding=k//2, bias) -> ReLU -> [MaxPool2d(size, stride)]``. It maps
exactly onto ``SpifilOpsBlock`` (conv -> Identity norm -> ReLU -> pool) EXCEPT the pool geometry:
FLIM uses ``MaxPool2d(3, stride=2, padding=0)`` (224 -> 111) while our SPiFiL pool is 3x3 s2 pad 1
(224 -> 112). The only new op is therefore the spec key ``pool_padding`` (+ ``pool_size``) of
``SpifilOpsBlock`` (absent = library pool, unchanged). The genome gene stays ``pool: max`` (params
identical, the memory/size model is off by one pixel). Unsupported FLIM features (inception
branches, dilation != 1, extra tensors in the checkpoint such as whitening buffers) raise
``NotImplementedError`` instead of being silently dropped.

Growth grid (MO-DEHB seeds)
---------------------------
Variants = width multiplier m (all layers) x extra depth d (+ an optional variant whose extra layer
is a zero-init residual block). Kept filters keep their trained weights (cut = drop the least-used
filters, by the L2 norm of the consumer's input weights; the remaining ones are NOT retrained),
NEW filters get SPiFiL init (exact width allocator) + local AE, new layers are identity / zero-init
residual blocks, so the grown net at step 0 reproduces the seed's features on the old channels
(``preservation_error``). Only inherited elements are protected (``protect`` + WiSE anchor); new
elements train at full lr.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

LAYER_KEYS = ("channels", "kernel", "conv", "residual", "activation", "norm", "pool", "softhebb")
DEFAULT_GENES = dict(ssl="lejepa", reg="sigreg", embed_dim=1280, pooling="gem", ae_target="pixel",
                     projector=False, protect="lowlr", ae_schedule="decoder_first", spifil_images_per_class=1)
DEFAULT_GRID = dict(width_mults=[1.0], extra_depths=[0], residual_extra=False, residual_activation="silu")
DEFAULTS = dict(enabled=False, name="ssl_pequeno_lejepa",
                arch="artifacts/external_ssl_pequeno/{dataset}_split{split}_pct{pct}/L0/architecture.json",
                checkpoint="artifacts/external_ssl_pequeno/{dataset}_split{split}_pct{pct}/lejepa/checkpoints/best_kappa.ckpt",
                state_prefix="model.encoder.", genes=DEFAULT_GENES, grid=DEFAULT_GRID)
ZERO_ACTS = ("silu", "gelu")          # act(0) = 0 with act'(0) != 0: zero-init residual stays trainable


# ─────────────────────────────────────────────────────────────── config
def resolve_config(raw: Optional[dict], dataset: str, split: int = 1, pct: int = 100) -> Optional[dict]:
    """Normalised config with the per-dataset paths filled in, or None when absent/disabled."""
    if not raw or not raw.get("enabled"):
        return None
    ec = {**DEFAULTS, **raw}
    ec["genes"] = {**DEFAULT_GENES, **(raw.get("genes") or {})}
    ec["grid"] = {**DEFAULT_GRID, **(raw.get("grid") or {})}
    fmt = dict(dataset=dataset, split=int(split), pct=int(pct))
    ec["arch"], ec["checkpoint"] = str(ec["arch"]).format(**fmt), str(ec["checkpoint"]).format(**fmt)
    ec["dataset"] = dataset
    return ec


# ─────────────────────────────────────────────────────────────── loader / converter
def _flim_layers(arch: dict) -> List[dict]:
    out = []
    for n in range(1, int(arch["nlayers"]) + 1):
        d = arch[f"layer{n}"]
        conv, pool = d["conv"], d.get("pooling", {"type": "none"})
        if "branches" in conv:
            raise NotImplementedError(f"layer{n}: FLIM inception branches are not mapped")
        k = [int(v) for v in conv["kernel_size"][:2]]
        if k[0] != k[1] or k[0] % 2 == 0:
            raise NotImplementedError(f"layer{n}: kernel {k} (only square odd kernels)")
        if [int(v) for v in conv.get("dilation_rate", [1, 1])[:2]] != [1, 1]:
            raise NotImplementedError(f"layer{n}: dilation != 1")
        if not d.get("relu", False):
            raise NotImplementedError(f"layer{n}: FLIM layer without relu")
        ptype = pool.get("type", "none")
        if ptype not in ("none", "max_pool"):
            raise NotImplementedError(f"layer{n}: pooling {ptype!r}")
        out.append({"kernel": k[0], "pool": "max" if ptype == "max_pool" else "none",
                    "pool_size": int(pool.get("size", [3])[0]), "pool_stride": int(pool.get("stride", 2))})
    return out


def _state(ckpt: str, prefix: str, n_layers: int) -> List[Dict[str, torch.Tensor]]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    out = []
    for n in range(1, n_layers + 1):
        for p in (f"{prefix}conv{n}.", f"{prefix}blocks.conv{n}."):
            keys = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            if keys:
                break
        if set(keys) != {"0.weight", "0.bias"}:
            raise NotImplementedError(f"conv{n}: unexpected tensors {sorted(keys)} (only conv weight/bias "
                                      "are mapped; e.g. whitening buffers would need a new op)")
        out.append({"conv.weight": keys["0.weight"].float().clone(), "conv.bias": keys["0.bias"].float().clone()})
    return out


def load_external_seed(base: str, ec: Optional[dict] = None, *, split: int = 1, pct: int = 100,
                       seed: int = 42) -> Tuple[dict, List[dict]]:
    """``(genome, block_states)`` of the external encoder of dataset ``base`` (see module doc).

    ``ec``: ``external_seed`` config (paths may contain {dataset}/{split}/{pct}); missing files
    raise ``FileNotFoundError``."""
    ec = resolve_config({**DEFAULTS, **(ec or {}), "enabled": True}, base, split, pct) \
        if not (ec or {}).get("dataset") else ec
    for p in (ec["arch"], ec["checkpoint"]):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    arch = json.loads(Path(ec["arch"]).read_text())
    geo = _flim_layers(arch)
    states = _state(ec["checkpoint"], ec["state_prefix"], len(geo))
    blocks, layers, cin = [], [], 3
    for i, (g, st) in enumerate(zip(geo, states)):
        w = st["conv.weight"]
        cout, wc, kh = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
        if wc != cin or kh != g["kernel"] or tuple(st["conv.bias"].shape) != (cout,):
            raise ValueError(f"conv{i + 1}: weight {tuple(w.shape)} does not match arch (cin={cin}, k={g['kernel']})")
        spec = {"builder": "spifil_ops", "norm_in_block": True, "conv": "conv", "kernel": kh, "in_channels": cin,
                "channels": cout, "requested_channels": cout, "dw_multiplier": None, "activation": "relu",
                "norm": "none", "pool": g["pool"], "pool_stride": g["pool_stride"] if g["pool"] != "none" else 1,
                "residual": False, "skip": None, "dropout": 0.0, "origin": f"external:{ec['name']}"}
        if g["pool"] != "none":
            spec.update(pool_size=g["pool_size"], pool_padding=0)        # FLIM MaxPool2d(size, stride), no pad
        blocks.append({"spec": spec, "state_dict": st})
        layers.append({"channels": cout, "kernel": kh, "conv": "conv", "residual": False, "activation": "relu",
                       "norm": "none", "pool": g["pool"], "softhebb": False})
        cin = cout
    genome = _genome(layers, ec["genes"], seed, {"name": ec["name"], "width": 1.0, "extra_depth": 0,
                                                  "residual": False, "n_imported": len(layers)})
    return genome, blocks


def _genome(active: List[dict], genes: dict, seed: int, marker: dict, n_layers: int = 8, protect=None) -> dict:
    layers = [dict(active[min(i, len(active) - 1)]) for i in range(max(n_layers, len(active)))]
    for L in layers[len(active):]:
        L["pool"] = "none"
    g = {"init": "spifil", "seed": int(seed), "depth": len(active), "layers": layers,
         **{k: genes[k] for k in ("ssl", "reg", "embed_dim", "pooling", "ae_target", "projector", "ae_schedule",
                                  "spifil_images_per_class")},
         "protect": protect or (genes["protect"][0] if isinstance(genes["protect"], (list, tuple)) else genes["protect"]),
         "external": dict(marker)}
    return g


def build_blocks(block_states: Sequence[dict], device=None) -> nn.Sequential:
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    blocks = []
    for b in block_states:
        blk = SpifilOpsBlock(b["spec"], b["spec"]["in_channels"])
        blk.load_state_dict(b["state_dict"])
        blocks.append(blk)
    enc = nn.Sequential(*blocks)
    return enc if device is None else enc.to(device)


def _sha(enc: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in enc.state_dict().items():
        h.update(k.encode()), h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def prepare(ec: Optional[dict], out_dir) -> Optional[dict]:
    """Convert once into ``<out_dir>/external_seed/<name>.pt`` (an encoder checkpoint like ours) and
    return the search-side info (JSON-safe). Missing files -> warning + ``{..., "missing": path}``
    (no seeds: the search runs exactly as without the block)."""
    if not ec:
        return ec
    try:
        genome, states = load_external_seed(ec["dataset"], ec)
    except FileNotFoundError as exc:
        msg = f"[external_seed] WARNING: {ec['name']} not found for {ec['dataset']} ({exc}); search runs WITHOUT the seed"
        warnings.warn(msg)
        print(msg, flush=True)
        return {**ec, "missing": str(exc)}
    enc = build_blocks(states)
    sha = _sha(enc)
    path = Path(out_dir) / "external_seed" / f"{ec['name']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    if path.exists():
        try:
            ok = torch.load(path, map_location="cpu", weights_only=False).get("sha256") == sha
        except Exception:
            ok = False
    if not ok:
        tmp = path.with_suffix(".tmp")
        torch.save({"encoder": enc, "effective_specs": [b["spec"] for b in states], "genome": genome,
                    "sha256": sha, "source": {"arch": ec["arch"], "checkpoint": ec["checkpoint"]},
                    "cand_id": f"external:{ec['name']}"}, tmp)
        os.replace(tmp, path)
    print(f"[external_seed] {ec['name']} ({ec['dataset']}): widths {[b['spec']['channels'] for b in states]} "
          f"-> {path} sha {sha}", flush=True)
    return {**ec, "converted": str(path), "sha256": sha, "layers": active_layers(genome)}


def active_layers(genome: dict) -> List[dict]:
    return [{k: L[k] for k in LAYER_KEYS} for L in genome["layers"][:int(genome["depth"])]]


# ─────────────────────────────────────────────────────────────── growth grid (seeds)
def grid_variants(grid: dict) -> List[Tuple[float, int, bool]]:
    """(width multiplier, extra depth, residual) ordered by distance to the seed (pure seed first)."""
    out = [(float(m), int(d), False) for m in grid["width_mults"] for d in grid["extra_depths"]]
    if grid.get("residual_extra"):
        out += [(float(m), 1, True) for m in grid["width_mults"]]
    out = list(dict.fromkeys(out))
    return sorted(out, key=lambda v: (abs(math.log(v[0])) + 0.5 * v[1], v[2], v[0], v[1]))


def variant_genome(info: dict, m: float, d: int, residual: bool, seed: int, protect: str,
                   n_layers: int = 8) -> dict:
    par = [dict(L) for L in info["layers"]]
    widths, w = [], 0
    for L in par:
        w = max(w, max(1, int(round(int(L["channels"]) * float(m)))))   # monotonic (running max)
        widths.append(w)
    act = [{**L, "channels": c} for L, c in zip(par, widths)]
    for j in range(int(d)):
        new = {**act[-1], "pool": "none", "residual": False, "kernel": 3, "softhebb": False}
        if residual and j == int(d) - 1:
            new.update(residual=True, activation=info["grid"].get("residual_activation", "silu"))
        act.append(new)
    return _genome(act, info["genes"], seed, {"name": info["name"], "width": float(m), "extra_depth": int(d),
                                              "residual": bool(residual), "n_imported": len(par)},
                   n_layers=n_layers, protect=protect)


def new_filter_dims(genome: dict, parent_widths: Sequence[int]) -> List[int]:
    """Covariance dimension D = cin * k^2 of every SPiFiL fit the growth needs (NEW filters only:
    mapped layers wider than the parent)."""
    dims, cin = [], 3
    for i, L in enumerate(active_layers(genome)):
        if i < len(parent_widths) and int(L["channels"]) > int(parent_widths[i]):
            dims.append(cin * int(L["kernel"]) ** 2)
        cin = int(L["channels"])
    return dims


def seed_genomes(info: Optional[dict], seed: int, *, n_classes: int = 0, choices: Sequence[int] = (1,),
                 seeds_per_image: float = 100, pool_sizes=None, max_depth: int = 8,
                 base_widths: Optional[Sequence[int]] = None) -> List[dict]:
    """Growth-grid genomes (NOT repaired). ``spifil_images_per_class`` is raised to the smallest
    choice whose SPiFiL budget covers the NEW-filter fits only (the inherited layers fit nothing);
    variants deeper than ``max_depth`` or without a feasible k are skipped (warning)."""
    if not info or not info.get("layers"):
        return []
    from ga_search.modehb.strata import spifil_budget
    prot = info["genes"]["protect"]
    prot = list(prot) if isinstance(prot, (list, tuple)) else [prot]
    out = []
    for j, (m, d, r) in enumerate(grid_variants(info["grid"])):
        if len(info["layers"]) + d > max_depth:
            continue
        g = variant_genome(info, m, d, r, seed, prot[j % len(prot)], n_layers=max_depth)
        dims = new_filter_dims(g, [int(L["channels"]) for L in info["layers"]])
        if dims and n_classes:
            dmax = max(dims)
            ks = sorted(int(c) for c in choices if int(c) >= int(g["spifil_images_per_class"]))
            ok = [k for k in ks if spifil_budget(k, n_classes, 0.9 * seeds_per_image, pool_sizes)[0] > 1.05 * dmax]
            if not ok:
                warnings.warn(f"[external_seed] variant x{m}+{d} skipped: no SPiFiL budget for D={dmax}")
                continue
            g["spifil_images_per_class"] = ok[0]
        out.append(g)
    return out


def external_layer_map(genome: dict) -> list:
    ext = genome["external"]
    n = int(ext["n_imported"])
    extra = int(genome["depth"]) - n
    tail = ["identity"] * extra
    if ext.get("residual") and extra:
        tail[-1] = "residual_zero"
    return list(range(n)) + tail


def external_inherit(es: dict, genome: dict, out_dir=None) -> dict:
    """``ctx["inherit"]``-like spec of an external-seed genome (function-preserving growth)."""
    if not es.get("converted") and out_dir is not None:
        es.update(prepare(es, out_dir) or {})
    if not es.get("converted"):
        raise FileNotFoundError(f"external seed {es.get('name')} not prepared ({es.get('missing')})")
    n = int(genome["external"]["n_imported"])
    return {"checkpoint": es["converted"], "n_blocks": n, "layer_map": external_layer_map(genome),
            "parent_layers": es["layers"], "parent_cand_id": f"external:{es['name']}", "parent_fidelity": 0.0}


def morph_layer_map(parent: dict, child: dict, info: dict) -> list:
    """LEMONADE morphism -> child layer -> parent layer index | "identity" | "fresh"."""
    pd, cd = int(parent["depth"]), int(child["depth"])
    if info.get("morphism") == "deepen" and cd == pd + 1:
        p = int(info["layer"])
        return list(range(p)) + ["identity"] + list(range(p, pd))
    return list(range(min(pd, cd))) + ["fresh"] * max(0, cd - pd)


# ─────────────────────────────────────────────────────────────── function-preserving morphisms
def _plain(block) -> bool:
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    return (isinstance(block, SpifilOpsBlock) and isinstance(block.conv, nn.Conv2d) and block.conv.groups == 1
            and block.conv.bias is not None and isinstance(block.norm, nn.Identity) and block.skip is None)


def _nonneg(block) -> bool:
    """Output >= 0 (relu, then max/avg pool or none), so an identity relu layer is exact on it."""
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    return isinstance(block, SpifilOpsBlock) and isinstance(block.activation, nn.ReLU) and block.skip is None


def _compatible(pL: dict, L: dict) -> bool:
    same = all(pL.get(k) == L.get(k) for k in ("conv", "residual", "activation", "norm", "pool", "softhebb"))
    dk = int(L["kernel"]) - int(pL["kernel"])
    return same and dk >= 0 and dk % 2 == 0


def keep_filters(parent_block, n_keep: int, consumer=None) -> List[int]:
    """Indices (sorted) of the ``n_keep`` filters kept on a cut: largest L2 norm of the consumer's
    input weights for that channel (least-used channels dropped); last layer: own filter norm."""
    w = parent_block.conv.weight.detach()
    if n_keep >= w.shape[0]:
        return list(range(int(w.shape[0])))
    if consumer is not None and _plain(consumer) and consumer.conv.in_channels == w.shape[0]:
        imp = consumer.conv.weight.detach().transpose(0, 1).reshape(w.shape[0], -1).norm(dim=1)
    else:
        imp = w.reshape(w.shape[0], -1).norm(dim=1)
    top = torch.argsort(imp, descending=True, stable=True)[:int(n_keep)]
    return sorted(int(i) for i in top)


def widen_block(parent_block, in_map: Sequence[int], cout: int, kernel: Optional[int] = None, *,
                keep: Optional[Sequence[int]] = None, new_weight=None, new_bias=None, consumer=None):
    """Net2Wider / cut of a plain dense ``SpifilOpsBlock``.

    ``in_map[j]`` = parent input channel feeding child input channel j (-1 = NEW channel: its
    columns are 0, so the kept filters compute exactly the parent's function). Kept filters (``keep``,
    default = all, or the cut chosen by ``keep_filters``) copy the parent's weights (zero-padded when
    ``kernel`` grows); the ``cout - len(keep)`` NEW filters take ``new_weight``/``new_bias``
    (``(n_new, cin, k, k)``; default zeros). Returns ``(block, keep, masks)`` where ``masks`` marks the
    INHERITED elements ({"conv.weight": bool, "conv.bias": bool})."""
    if not _plain(parent_block):
        raise ValueError("widen_block needs a plain dense SpifilOpsBlock (norm none, no skip)")
    pw, pb = parent_block.conv.weight.detach(), parent_block.conv.bias.detach()
    kp = int(pw.shape[-1])
    kc = int(kernel or kp)
    if kc < kp or (kc - kp) % 2:
        raise ValueError(f"kernel {kp} -> {kc} is not function preserving")
    keep = list(keep) if keep is not None else keep_filters(parent_block, min(int(cout), int(pw.shape[0])), consumer)
    cin, nk, n_new = len(in_map), len(keep), int(cout) - len(keep)
    if n_new < 0:
        raise ValueError("cout < len(keep)")
    W = pw.new_zeros(int(cout), cin, kc, kc)
    b = pb.new_zeros(int(cout))
    mw = torch.zeros(W.shape, dtype=torch.bool, device=W.device)
    cols = [j for j, p in enumerate(in_map) if p >= 0]
    pcols = [int(in_map[j]) for j in cols]
    o = (kc - kp) // 2
    idx_k = torch.as_tensor(keep, dtype=torch.long, device=pw.device)
    if nk and cols:
        ci = torch.as_tensor(cols, dtype=torch.long, device=pw.device)
        sub = pw[idx_k][:, torch.as_tensor(pcols, dtype=torch.long, device=pw.device)]
        W[:nk, ci, o:o + kp, o:o + kp] = sub
        mw[:nk, ci, o:o + kp, o:o + kp] = True
    if nk:
        b[:nk] = pb[idx_k]
    mb = torch.zeros(int(cout), dtype=torch.bool, device=b.device)
    mb[:nk] = True
    if n_new:
        if new_weight is not None:
            W[nk:] = new_weight.to(W)
        if new_bias is not None:
            b[nk:] = new_bias.to(b)
    blk = copy.deepcopy(parent_block)
    blk.conv = nn.Conv2d(cin, int(cout), kc, padding=kc // 2, bias=True).to(device=pw.device, dtype=pw.dtype)
    with torch.no_grad():
        blk.conv.weight.copy_(W)
        blk.conv.bias.copy_(b)
    blk.spec = {**parent_block.spec, "in_channels": cin, "channels": int(cout), "requested_channels": int(cout),
                "kernel": kc}
    blk.out_channels = int(cout)
    return blk, keep, {"conv.weight": mw, "conv.bias": mb}


def _new_ops_block(L: dict, cin: int, device=None):
    from ga_search.modehb.spifil_ops import SpifilOpsBlock, to_ops_spec
    blk = SpifilOpsBlock(to_ops_spec(L, cin), cin)
    return blk if device is None else blk.to(device)


def identity_block(L: dict, cin: int, device=None):
    """Net2Deeper: conv k (centre = I, bias 0) -> relu; exact on a >= 0 input."""
    blk = _new_ops_block({**L, "channels": cin, "residual": False, "pool": "none", "norm": "none"}, cin, device)
    k = blk.conv.kernel_size[0]
    with torch.no_grad():
        blk.conv.weight.zero_()
        blk.conv.bias.zero_()
        blk.conv.weight[torch.arange(cin), torch.arange(cin), k // 2, k // 2] = 1.0
    return blk


def residual_zero_block(L: dict, cin: int, device=None):
    """y = act(conv(x)) + x with conv = 0 and act in ZERO_ACTS (act(0) = 0, act'(0) = 0.5) -> y = x."""
    blk = _new_ops_block({**L, "channels": cin, "residual": True, "pool": "none", "norm": "none"}, cin, device)
    with torch.no_grad():
        for p in blk.conv.parameters():
            p.zero_()
    return blk


class Grown:
    """Result of ``grow_blocks``: fp-built blocks (+ masks of inherited elements, logs)."""

    def __init__(self):
        self.blocks, self.masks, self.specs, self.logs = [], [], [], []
        self.in_map, self.last_src, self.cut = None, None, False


def grow_blocks(parent_blocks: Sequence[nn.Module], parent_layers: Sequence[dict], child_layers: Sequence[dict],
                layer_map: Sequence, *, new_filters: Optional[Callable] = None, device=None,
                parent_channels: Optional[Sequence[int]] = None) -> Grown:
    """Function-preserving child from a parent (see module doc). Builds the leading child layers
    that ``layer_map`` maps (int = parent layer | "identity" | "residual_zero"); stops at the first
    layer that cannot be mapped exactly (incompatible genes / non-plain parent block / "fresh"):
    that layer and the following ones are left to the caller (fresh SPiFiL init).

    ``new_filters(i, prefix_blocks, layer, cin, n_new, kernel) -> (weight, bias)`` initialises the
    NEW filters of mapped layer i (default: zeros). ``parent_channels``: ACTUAL output channels of
    every parent block (SPiFiL may build fewer than the genome width); default ``block.out_channels``
    or the genome width. Child widths are genome widths; a child layer whose genes all equal the
    parent's, on an unchanged input, is copied verbatim (whatever its block type)."""
    out = Grown()
    n_par = len(parent_blocks)
    pch = list(parent_channels) if parent_channels is not None else \
        [int(getattr(b, "out_channels", None) or L["channels"]) for b, L in zip(parent_blocks, parent_layers)]
    in_map = list(range(3))
    prev_w = None
    for i, L in enumerate(child_layers):
        src = layer_map[i] if i < len(layer_map) else "fresh"
        L = {k: L[k] for k in LAYER_KEYS}
        cin = len(in_map)
        if isinstance(src, int) and 0 <= src < n_par and _compatible(parent_layers[src], L):
            pb = parent_blocks[src]
            pcin = int(pch[src - 1]) if src else 3
            same_in = in_map == list(range(pcin))
            if same_in and L == {k: parent_layers[src][k] for k in LAYER_KEYS}:
                blk = copy.deepcopy(pb)                                  # verbatim (any block type)
                masks = {n: torch.ones_like(p, dtype=torch.bool) for n, p in blk.named_parameters()}
                keep, n_new, origin = list(range(int(pch[src]))), 0, "inherited"
            elif _plain(pb) and pb.conv.in_channels == pcin and max(in_map) < pcin:
                cout = int(L["channels"])
                cons = parent_blocks[src + 1] if src + 1 < n_par else None
                keep = keep_filters(pb, min(cout, pb.conv.out_channels), cons)
                n_new = cout - len(keep)
                nw = nb = None
                if n_new and new_filters is not None:
                    nw, nb = new_filters(i, list(out.blocks), L, cin, n_new, int(L["kernel"]))
                blk, keep, masks = widen_block(pb, in_map, cout, int(L["kernel"]), keep=keep,
                                               new_weight=nw, new_bias=nb)
                origin = "fp_widen" if n_new or cin > len([p for p in in_map if p >= 0]) else "fp_cut"
                out.cut = out.cut or len(keep) < pb.conv.out_channels or len({p for p in in_map if p >= 0}) < pcin
            else:
                break
            out.last_src = src
            in_map = list(keep) + [-1] * n_new
        elif src == "identity" and out.blocks and _nonneg(out.blocks[-1]) and L["conv"] == "conv" \
                and L["norm"] == "none" and not L["residual"] and L["activation"] == "relu" and L["pool"] == "none" \
                and int(L["channels"]) == prev_w and int(L["kernel"]) % 2 == 1 and not L["softhebb"]:
            blk, masks, origin = identity_block(L, cin, device), {}, "fp_identity"
        elif src == "residual_zero" and out.blocks and L["conv"] == "conv" and L["norm"] == "none" and L["residual"] \
                and L["activation"] in ZERO_ACTS and L["pool"] == "none" and int(L["channels"]) == prev_w \
                and not L["softhebb"]:
            blk, masks, origin = residual_zero_block(L, cin, device), {}, "fp_residual_zero"
        else:
            break
        blk = blk if device is None else blk.to(device)
        out.blocks.append(blk)
        params = dict(blk.named_parameters())
        out.masks.append({n: m.to(params[n].device) for n, m in masks.items() if n in params})
        out.specs.append(json.loads(json.dumps(blk.spec, default=str)))
        n_inh = int(sum(int(m.sum()) for m in out.masks[-1].values()))
        out.logs.append({"layer": i, "prov": {"origin": origin, "parent_layer": src if isinstance(src, int) else None,
                                              "n_inherited_elements": n_inh,
                                              "n_new_elements": int(sum(p.numel() for p in blk.parameters())) - n_inh},
                         "ae": {}})
        prev_w = int(L["channels"])
    out.in_map = in_map
    return out


@torch.no_grad()
def preservation_error(parent_blocks: Sequence[nn.Module], grown: Grown, x: torch.Tensor) -> Optional[float]:
    """max |child(x)[old channels] - parent(x)| at the end of the fp-built prefix (None after a cut)."""
    if grown.cut or grown.last_src is None or not grown.blocks:
        return None
    par = nn.Sequential(*parent_blocks[:grown.last_src + 1]).eval()
    child = nn.Sequential(*grown.blocks).eval()
    ref, got = par(x), child(x)
    cols = [j for j, p in enumerate(grown.in_map) if p >= 0]
    pcols = [grown.in_map[j] for j in cols]
    return float((got[:, cols] - ref[:, pcols]).abs().max())


def fit_new_filters(prefix, samples, L: dict, cin: int, n_new: int, kernel: int, cfg: dict):
    """SPiFiL (spifil_ops.fit_ops_block, exact-width allocator) for ``n_new`` new filters on ``prefix``."""
    from ga_search.modehb.spifil_ops import fit_ops_block, to_ops_spec
    spec = to_ops_spec({**L, "channels": int(n_new), "kernel": int(kernel), "pool": "none", "residual": False,
                        "norm": "none"}, cin)
    blk, _ = fit_ops_block(prefix, samples, spec, cfg, exact_width=True)
    w, b = blk.conv.weight.detach(), blk.conv.bias.detach()
    if int(w.shape[0]) != int(n_new):
        from flim.progressive import InfeasibleCandidate
        raise InfeasibleCandidate(f"SPiFiL new filters: got {w.shape[0]} != {n_new}")
    return w, b


def load_parent_blocks(checkpoint: str, device=None) -> Tuple[List[nn.Module], List[dict]]:
    ck = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)
    enc = ck["encoder"]
    blocks = [copy.deepcopy(enc[i]) for i in range(len(enc))]
    if device is not None:
        blocks = [b.to(device) for b in blocks]
    return blocks, list(ck.get("effective_specs") or [])


# ─────────────────────────────────────────────────────────────── per-element protection
class ElementProtect:
    """Context manager: during every optimizer step of the parameters of ``module`` whose name is in
    ``masks``, the masked (inherited) elements move by ``scale`` x their update (0 = frozen, 0.01 =
    lowlr). Exact for Adam/AdamW: the moments only depend on the gradients, and scaling the step delta
    equals a param group with ``lr * scale`` (also for AdamW's decoupled weight decay). Uses the global
    ``torch.optim`` step hooks (removed on exit); no-op when ``masks`` is empty or ``scale == 1``."""

    def __init__(self, module: nn.Module, masks: Optional[Dict[str, torch.Tensor]], scale: float):
        self.scale = float(scale)
        params = dict(module.named_parameters())
        self.entries = [(params[n], m.to(params[n].device).bool()) for n, m in (masks or {}).items()
                        if n in params and bool(m.any())] if self.scale != 1.0 else []
        self._hooks, self._snap = [], {}

    def __enter__(self):
        if self.entries:
            from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                               register_optimizer_step_pre_hook)
            self._hooks = [register_optimizer_step_pre_hook(self._pre), register_optimizer_step_post_hook(self._post)]
        return self

    def _pre(self, opt, args, kwargs):
        ids = {id(q) for g in opt.param_groups for q in g["params"]}
        # growth_fix8: snapshot do tensor inteiro + torch.where (sem sincronizar). A indexação booleana
        # ``p[m]`` fazia um nonzero (sincronização GPU->CPU) por parâmetro, 2x por passo. Mesma conta por
        # elemento mascarado (old + scale*(new - old)); os não mascarados ficam intocados -> bit a bit igual.
        self._snap[id(opt)] = [(p, m, p.detach().clone()) for p, m in self.entries if id(p) in ids]

    def _post(self, opt, args, kwargs):
        with torch.no_grad():
            for p, m, old in self._snap.pop(id(opt), []):
                p.data.copy_(torch.where(m, old + self.scale * (p.data - old), p.data))

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks, self._snap = [], {}
        return False
