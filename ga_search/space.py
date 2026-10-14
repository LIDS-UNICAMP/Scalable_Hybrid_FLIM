# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Espaco de busca do GA: genes do search_space.yaml -> variaveis pymoo -> config ativa (§3) -> SPiFiL."""

import copy
import hashlib
import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from pymoo.core.variable import Binary, Choice, Integer, Real
from spifil.config import SpifilConfig

from ga_search.derived import layer_geometry


def _read(path):
    """YAML do espaco; com `extends`, parte do base e substitui os genes de `override`."""
    with open(path) as f:
        s = yaml.safe_load(f)
    if "extends" not in s:
        return s
    base = _read(Path(path).parent / s.pop("extends"))
    base["genes"].update(s.pop("override", {}))
    return {**base, **s}


def load_space(space_yaml, ds_cfg):
    """Le o YAML, aplica os overrides do dataset e expande os genes atomicos ate d_max."""
    space = _read(space_yaml)
    genes = space["genes"]
    for name, over in (ds_cfg.get("search_space_overrides") or {}).items():
        genes[name].update(over)
    for base, spec in space.pop("per_layer").items():
        for l in range(spec.get("from_layer", 1), space["d_max"] + 1):
            s = {**copy.deepcopy(genes[spec["same_as"]]), "default": spec["default"]} if "same_as" in spec else copy.deepcopy(spec)
            s.pop("from_layer", None)
            s["when"] = {**{k.format(l=l): v for k, v in spec["when"].items()}, "d": {"min": l}}
            genes[f"{base}_l{l}"] = s
    for name in [n for n in genes if n == "w" or n.startswith("N_l")]:
        genes[name]["low"] = max(genes[name]["low"], ds_cfg["n_classes"])
    if not ds_cfg["has_mask"]:
        genes.pop("impurities", None)
    if not space["use_flatten"]:
        genes["head"]["values"] = [v for v in genes["head"]["values"] if v != "flatten"]
    return space


def pymoo_vars(space):
    """Um Variable do pymoo por gene, inclusive os que podem ficar dormentes."""
    mk = {"real": lambda g: Real(bounds=(g["low"], g["high"])),
          "integer": lambda g: Integer(bounds=(g["low"], g["high"])),
          "choice": lambda g: Choice(options=g["values"]),
          "binary": lambda g: Binary()}
    return {n: mk[g["type"]](g) for n, g in space["genes"].items()}


def default_genome(space):
    """Os defaults do YAML, presos ao dominio (overrides podem subir o low)."""
    return {n: min(max(g["default"], g["low"]), g["high"]) if "low" in g else g["default"]
            for n, g in space["genes"].items()}


def _cast(g, x, dec):
    """Valor do pymoo (tipos numpy, Choice mista vira str) -> tipo python do YAML."""
    if g["type"] == "real":
        return round(float(x), dec)
    if g["type"] == "integer":
        return int(x)
    if g["type"] == "binary":
        return bool(x)
    return next(o for o in g["values"] if str(o) == str(x))


def _ok(x, c):
    if isinstance(c, list):
        return x in c
    if isinstance(c, dict):
        return x != c["not"] if "not" in c else x >= c["min"]
    return x == c


def active_config(X, space, ds_cfg):
    """Genoma -> config ativa canonica (§3); genes dormentes nao entram."""
    genes, dec = space["genes"], space["round_decimals"]
    X = {n: _cast(g, X[n], dec) for n, g in genes.items()}
    on = lambda n: n in genes and all(_ok(X.get(k), c) for k, c in genes[n].get("when", {}).items())
    v = lambda n: X[n] if on(n) else None
    n_eff = len(ds_cfg["paper_classes"]) if X["paper_classes"] else ds_cfg["n_classes"]
    feats, mode, d = X["spifil_in"] == "features", X["layer_mode"], X["d"]
    res = X["residual"]
    if res:  # §7.7: N_l novos, arredondados para cima ao multiplo de n_eff; C_l = C_(l-1) + N_l
        up = lambda x: -(-x // n_eff) * n_eff
        C, N = [], []
        for l in range(1, d + 1):
            g = f"growth_l{l}"  # sem o gene (l > residual_d_max, ja invalido): N_l = C_(l-1), o minimo da lei
            N.append(up(max(X["N_1"], n_eff)) if l == 1 else up(round(C[-1] * (X[g] - 1))) if g in X else C[-1])
            C.append(sum(N))

    layers = []
    for l in range(1, d + 1):
        if mode == "atomico":
            s = f"_l{l}"
            layer = {"kernel_size": X["k" + s], "out_channels": None if res else X["N" + s], "dilation": X["dilation" + s],
                     "activation": X["activation" + s], "pool_type": X["pool_type" + s],
                     "pool_size": v("pool_size" + s), "pool_stride": v("pool_stride" + s),
                     "n_superpixels": v("n_superpixels" + s)}
        else:
            grow, e = mode == "crescimento", l - 1
            pooled = not grow or {"todas": True, "primeira": l == 1, "alternadas": l % 2 == 1,
                                  "ultima": l == d}[X["pool_schedule"]]
            pt = X["pool_type"] if pooled else "none"
            nsp = None
            if feats and l >= 2:
                nsp = max(1, round(X["n_superpixels"] * X["superpixel_growth"] ** e)) if grow else X["n_superpixels"]
            layer = {"kernel_size": max(1, X["k"] + 2 * X["kernel_growth"] * e) if grow else X["k"],
                     "out_channels": None if res else max(n_eff, round(X["w"] * X["width_growth"] ** e)) if grow else X["w"],
                     "dilation": X["dilation"] * X["dilation_growth"] ** e if grow else X["dilation"],
                     "activation": X["activation"], "pool_type": pt,
                     "pool_size": X["pool_size"] if pt != "none" else None,
                     "pool_stride": X["pool_stride"] if pt != "none" else None,
                     "n_superpixels": nsp}
        if res:
            layer["out_channels"] = N[l - 1]
        layers.append({**layer, "channels": C[l - 1] if res else layer["out_channels"]})

    sp = ("compactness", "sigma", "adjacency_radius", "n_init_seeds")
    fixed = space["fixed"]
    return {
        "r": round(ds_cfg["native_size"] * X["r_factor"]), "r_factor": X["r_factor"], "head": X["head"],
        "layer_mode": mode, "d": d, "residual": res,
        **({"residual_d_max": space["residual_d_max"]} if res else {}),
        "stdev_factor": float(f"{10 ** X['stdev_factor_log10']:.{dec}g}"),
        "superpixels": {"name": X["superpixels"],
                        **{p: genes[p].get("map", {}).get(X[p], X[p]) for p in sp if on(p)}},
        "n_superpixels": X["n_superpixels"],
        "seed_extractor": X["seed_extractor"], "metric": X["metric"], "estimator": fixed["estimator"],
        "scorer": X["scorer"],
        "selector": {"name": X["selector"], **{p: X[p] for p in ("alpha", "pool_factor") if on(p)}},
        "allocator": fixed["allocator"], "color": fixed["color"], "budget_check": fixed["budget_check"],
        "one_per_class": X["one_per_class"], "n_images": v("n_images"),
        "paper_classes": X["paper_classes"], "impurities": bool(v("impurities")),
        "spifil_in": X["spifil_in"],
        "layers": layers,
        "growth": {p: v(p) for p in ("width_growth", "kernel_growth", "dilation_growth",
                                     "superpixel_growth", "pool_schedule")} if mode == "crescimento" else None,
    }


def is_valid(cfg, ctx):
    """Restricoes da §4, em ordem; devolve (ok, motivo da primeira que falha)."""
    L, n_eff = cfg["layers"], ctx["n_classes_eff"]
    for i, l in enumerate(L, 1):
        if l["pool_type"] != "none" and l["pool_stride"] > l["pool_size"]:
            return False, f"pool_stride > pool_size na camada {i}"
    for i, l in enumerate(L, 1):
        if l["out_channels"] < n_eff:
            return False, f"out_channels {l['out_channels']} < n_classes_eff {n_eff} na camada {i}"
    if cfg["residual"] and cfg["d"] > cfg["residual_d_max"]:
        return False, f"residual com d {cfg['d']} > {cfg['residual_d_max']}"
    if cfg["residual"]:
        for i, (a, b) in enumerate(zip(L, L[1:]), 2):
            if b["out_channels"] < a["channels"]:
                return False, f"N {b['out_channels']} < C_(l-1) {a['channels']} na camada {i}"
    geo = layer_geometry(cfg)
    for i, g in enumerate(geo, 1):
        if g["rf_layer"] >= g["H_in"]:
            return False, f"RF {g['rf_layer']} >= lado {g['H_in']} na camada {i}"
    if geo[-1]["H_out"] < 1:
        return False, "grade final < 1"
    if not cfg["one_per_class"] and cfg["n_images"] > ctx["n_available"]:
        return False, f"n_images {cfg['n_images']} > {ctx['n_available']} disponiveis"
    n_imgs = n_eff if cfg["one_per_class"] else cfg["n_images"]
    nsp = [l["n_superpixels"] or cfg["n_superpixels"] for l in L]
    if cfg["budget_check"]:
        for i, (g, n) in enumerate(zip(geo, nsp), 1):
            if n_imgs * n <= g["D"]:
                return False, f"seeds {n_imgs * n} <= D {g['D']} na camada {i}"
    if n_imgs * max(nsp) > ctx["max_seeds"]:
        return False, f"seeds {n_imgs * max(nsp)} > max_seeds {ctx['max_seeds']}"
    if (cfg["superpixels"]["name"] == "disf" or cfg["seed_extractor"] == "geodesic") \
            and importlib.util.find_spec("pyift") is None:
        return False, "disf/geodesic exigem pyift"
    return True, ""


def config_hash(cfg):
    """sha1 do json ordenado, 16 hex (os reais ja vem arredondados de active_config)."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def to_spifil_yaml(cfg, ds_cfg, pct, split):
    """Config ativa -> {spifil: SpifilConfig de spifil.cli.build_learner, ga_extra: o resto}.

    Os nos de componente partem do conf/<grupo>/<nome>.yaml do proprio SPiFiL.
    """
    conf = files("spifil") / "conf"
    node = lambda group, name, **args: {**yaml.safe_load((conf / group / f"{name}.yaml").read_text()), **args}
    strategy = next(d["strategy"] for d in yaml.safe_load((conf / "config.yaml").read_text())["defaults"]
                    if isinstance(d, dict) and "strategy" in d)
    lspec = ("kernel_size", "out_channels", "dilation", "pool_type", "pool_size", "pool_stride")
    sp = {k: v for k, v in cfg["superpixels"].items() if k != "name"}
    sel = {k: v for k, v in cfg["selector"].items() if k != "name"}
    metric = {"estimator": cfg["estimator"]} if cfg["metric"] == "mahalanobis" else {}
    spifil = OmegaConf.merge(OmegaConf.structured(SpifilConfig), {
        # sem mascara: o schema (str) nao aceita null; um diretorio inexistente da mask_path=None
        # (spifil.data._find_mask, com um warning por imagem) = imagem inteira.
        "data": {"images_dir": ds_cfg["images_dir"], "masks_dir": ds_cfg["masks_dir"] or f"{ds_cfg['images_dir']}/.no_masks"},
        "arch": {"layers": [{**{k: l[k] for k in lspec if l[k] is not None}, "relu": l["activation"] == "relu"}
                            for l in cfg["layers"]],
                 "stdev_factor": cfg["stdev_factor"]},
        "n_superpixels": cfg["n_superpixels"],
        "superpixels": node("superpixels", cfg["superpixels"]["name"], **sp),
        "seed_extractor": node("seed_extractor", cfg["seed_extractor"]),
        "metric": node("metric", cfg["metric"], **metric),
        "scorer": node("scorer", cfg["scorer"]),
        "selector": node("selector", cfg["selector"]["name"], **sel),
        "allocator": node("allocator", cfg["allocator"]),
        "color": node("color", cfg["color"]),
        "strategy": node("strategy", strategy),
    })
    extra = {k: cfg[k] for k in ("r", "head", "one_per_class", "n_images", "paper_classes", "impurities",
                                 "spifil_in", "budget_check", "layer_mode", "growth", "residual")}
    extra.update(dataset=ds_cfg["name"], pct=pct, split=split,
                 activation=[l["activation"] for l in cfg["layers"]],
                 n_superpixels=[l["n_superpixels"] for l in cfg["layers"]])
    return {"spifil": OmegaConf.to_container(spifil), "ga_extra": extra}
