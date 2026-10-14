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
"""Métricas derivadas (geometria, custo, orçamento) de uma config ativa; só math, convenções do SPiFiL."""

import math

from ga_search.constants import NAN, PROFILE, STR_KEYS


def layer_geometry(cfg: dict, c_in_image: int = 3) -> list[dict]:
    """Geometria por camada, como em spifil.nn.blocks (conv preserva H; pool janela ímpar, padding size//2).

    Residual (§7): n_new = out_channels (filtros novos), c_out = C_l = C_(l-1) + n_new, com C_1 = N_1.
    """
    out, H, c_in, rf, jump = [], cfg["r"], c_in_image, 1, 1
    for l, L in enumerate(cfg["layers"]):
        span, dil, n_new = 2 * (L["kernel_size"] // 2) + 1, L["dilation"], L["out_channels"]
        c_out = n_new + (c_in if cfg.get("residual") and l else 0)
        rf_layer = (span - 1) * dil + 1
        rf += (span - 1) * dil * jump
        H_out = H
        if L["pool_type"] != "none":
            p, s = L["pool_size"] // 2, L["pool_stride"]
            H_out = (H + 2 * p - (2 * p + 1)) // s + 1
            rf += 2 * p * jump
            jump *= s
        out.append({"span": span, "c_in": c_in, "c_out": c_out, "n_new": n_new, "H_in": H, "H_conv": H, "H_out": H_out,
                    "rf_layer": rf_layer, "rf_cum": rf, "jump": jump, "D": span * span * c_in})
        H, c_in = H_out, c_out
    return out


def derived(cfg: dict, info: dict, actual_out: list[int] | None = None) -> dict:
    """Chaves da §4 do contrato; actual_out (filtros realmente obtidos) prevalece sobre out_channels."""
    layers = cfg["layers"]
    if actual_out is not None:
        layers = [dict(L, out_channels=a) for L, a in zip(layers, actual_out)]
    geo = layer_geometry(dict(cfg, layers=layers))
    d, r, ncls, nsp = len(layers), cfg["r"], info["n_classes_eff"], cfg["n_superpixels"]
    n_imgs = info["n_train_images"]
    res = bool(cfg.get("residual"))
    fl = [g["H_conv"] ** 2 * g["n_new"] * g["c_in"] * g["span"] ** 2 for g in geo]
    per = [{"N": g["n_new"], "k": L["kernel_size"], "dilation": L["dilation"], "activation": L["activation"],
            "pool_type": L["pool_type"], "D": g["D"], "rf_layer": g["rf_layer"], "flops_frac": f / sum(fl),
            "expansion": g["c_out"] / g["c_in"], "budget_ratio": g["n_new"] / g["D"],
            "C": g["c_out"] if res else NAN, "Nnew": g["n_new"] if res else NAN,
            "inherited_frac": g["c_in"] / g["c_out"] if res and i else NAN,
            "growth_c": g["c_out"] / g["c_in"] if res and i else NAN}
           for i, (L, g, f) in enumerate(zip(layers, geo, fl))]
    down = math.prod(L["pool_stride"] for L in layers if L["pool_type"] != "none")
    H_f, N_d, N_1 = geo[-1]["H_out"], geo[-1]["c_out"], geo[0]["n_new"]
    feat = N_d * H_f ** 2 if cfg["head"] == "flatten" else N_d
    sp_area = r * r / nsp
    gmean = lambda key: (per[-1][key] / per[0][key]) ** (1 / (d - 1)) if d > 1 else 1.0
    row = {
        "D": geo[0]["D"], "budget_ratio": per[0]["budget_ratio"], "rf_cum": geo[-1]["rf_cum"],
        "downsampling": down, "out_res": r / down, "sp_area": sp_area,
        "rf_sp_ratio": geo[-1]["rf_cum"] / math.sqrt(sp_area), "flops": sum(fl),
        "activations": sum(g["H_conv"] ** 2 * g["n_new"] for g in geo),
        "params": sum(g["span"] ** 2 * g["c_in"] * g["n_new"] for g in geo),
        "filters_per_class": N_1 / ncls, "train_images_per_class": n_imgs / ncls,
        "candidates_per_filter": nsp * n_imgs / N_1, "data_fraction": info["pct"] / 100,
        "time_s": info.get("time_s"), "layer_mode": cfg["layer_mode"], "residual": int(res),
        "growth_c_geo": (geo[-1]["c_out"] / geo[0]["c_out"]) ** (1 / (d - 1)) if res and d > 1 else NAN,
    }
    for l in range(1, info.get("d_max", 7) + 1):
        for key in PROFILE:
            row[f"{key}_l{l}"] = per[l - 1][key] if l <= d else ("" if key in STR_KEYS else NAN)
    for key in ("N", "k", "dilation", "rf_layer", "budget_ratio"):
        v = [p[key] for p in per]
        row.update({f"{key}_mean": sum(v) / d, f"{key}_min": min(v), f"{key}_max": max(v)})
    row.update({
        "growth_w_obs": gmean("N"), "growth_k_obs": gmean("k"), "total_filters": sum(p["N"] for p in per),
        "kernel_area": layers[0]["kernel_size"] ** 2, "net_shape": d / row["N_mean"], "feat_dim": feat,
        "compression": feat / (r * r * 3), "rf_coverage": geo[-1]["rf_cum"] / r,
        "sp_per_out_pixel": nsp / H_f ** 2,
    })
    return row
