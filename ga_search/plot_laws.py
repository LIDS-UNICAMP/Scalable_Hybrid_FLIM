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
"""Figuras das leis do GA: lê os CSVs de results_dir (sem rodar SPiFiL) e salva em laws/figures/."""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ga_search.analyze_laws import bootstrap_ci, fit_power, fit_saturation
from ga_search.constants import DERIVED, GA_CONFIG, MARKERS, METRICS, MODES, PLOT_RC, RUN

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned|No artists with labels")

plt.rcParams.update(PLOT_RC)


class Skip(Exception):
    """Gráfico sem dados: pulado com aviso."""


def need(df, cols, what):
    if df is None or df.empty or any(c not in df for c in cols):
        raise Skip(f"sem dados ({what}: faltam {[c for c in cols if df is None or c not in df] or 'linhas'})")


def ylab(m):
    return f"{METRICS[m][0]}\n({METRICS[m][1]})"


def islog(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return len(s) > 0 and s.min() > 0 and s.max() / s.min() > 10


def best(D, m, ok=None):
    """Melhor avaliação ok de cada run pela métrica m; sem results.csv, cai no aggregated.csv."""
    ok = D["ok"] if ok is None else ok
    if ok.empty or m not in ok:
        return D["agg"] if ok is D["ok"] else ok
    g = ok.dropna(subset=[m]).groupby(RUN)[m]
    return ok.loc[g.idxmax() if METRICS[m][2] else g.idxmin()]


def boot(D, df):
    """Bootstrap sobre splits: n_boot reamostras com reposição, como chaves (multiconjunto ordenado de splits)."""
    rng, s = np.random.default_rng(D["L"]["seed"]), np.sort(df["split"].unique())
    return [tuple(sorted(rng.choice(s, len(s)))) for _ in range(D["L"]["n_boot"])]


def q(D):
    return [50 * (1 - D["L"]["ci"]), 50 * (1 + D["L"]["ci"])]


def pick(df, key):
    return pd.concat([df[df["split"] == v] for v in key])


def panels(D, nrows=1, sharey=True, extra=0):
    n = len(D["datasets"]) + extra
    fig, ax = plt.subplots(nrows, n, figsize=(4.4 * n, 3.6 * nrows), sharey=sharey, squeeze=False)
    return fig, ax


def dots(ax, D, df, x, y):
    for (p, s), g in df.groupby(["pct", "split"]):
        ax.scatter(g[x], g[y], color=D["pc"].get(p, "k"), marker=MARKERS.get(s, "D"), edgecolor="k", lw=.4, s=34, zorder=3)


def gray(ax, df, x, y):
    if not df.empty and x in df and y in df:
        ax.scatter(df[x], df[y], s=5, color="0.85", zorder=1, rasterized=True)


def save(D, fig, name, keys=True, tight=True):
    if keys:
        h = [plt.Line2D([], [], ls="", marker="o", color=c, label=f"{p}%") for p, c in D["pc"].items()]
        h += [plt.Line2D([], [], ls="", marker=mk, color="0.4", label=f"split {s}") for s, mk in MARKERS.items()]
        h += [plt.Line2D([], [], ls="", marker=".", color="0.8", label="todas as avaliações")]
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, 0), ncol=len(h), frameon=False)
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D["out"] / f"{name}.{ext}")
    plt.close(fig)
    print(f"[ok] {name}")


def fit_both(D, ax, df, x, y, logx, key, color=None, tag=""):
    """Potência vs saturação; a de menor AIC em linha cheia, banda IC95% por bootstrap sobre splits."""
    df = df[[x, y, "split"]].dropna()
    if len(df) < 3 or df[x].nunique() < 3:
        return
    lo, hi = df[x].min(), df[x].max()
    xs = np.geomspace(lo, hi, 100) if logx else np.linspace(lo, hi, 100)
    fits = {}
    for name, fn in (("potência", fit_power), ("saturação", fit_saturation)):
        f = fn(df[x].values, df[y].values)
        if np.isfinite(f["aic"]):
            fits[name] = (fn, f)
    if not fits:
        return
    win = min(fits, key=lambda n: fits[n][1]["aic"])
    for name, (fn, f) in fits.items():
        if color and name != win:
            continue
        c = color or ("C3" if name == "potência" else "C4")
        cache = {}   # com 3 splits só há 10 reamostras distintas: ajusta cada uma uma vez
        for k in boot(D, df):
            if k not in cache:
                b = pick(df, k)
                cache[k] = fn(b[x].values, b[y].values)["predict"](xs)
        band = [cache[k] for k in boot(D, df) if np.isfinite(cache[k]).all()]
        if band:  # banda cortada a ±50% da faixa observada para não esmagar o eixo y
            pad = .5 * np.ptp(df[y]) + 1e-12
            ax.fill_between(xs, *np.clip(np.nanpercentile(band, q(D), axis=0), df[y].min() - pad, df[y].max() + pad), color=c, alpha=.15, lw=0,
                            label=f"IC{D['L']['ci']:.0%} (bootstrap splits)" if name == win and not color else None)
        ax.plot(xs, f["predict"](xs), color=c, ls="-" if name == win else "--", lw=1.6, zorder=4,
                label=f"{tag}{name}: {f['formula']}  R²={f['r2']:.2f}  AIC={f['aic']:.1f}" + ("  ← menor AIC" if name == win else ""))
    D["summary"].append({**key, "forma": win, "formula": fits[win][1]["formula"], "r2": fits[win][1]["r2"]})


def law_scatter(D, law, m, xcols, pareto=False):
    """Métrica × variável(is), um painel por dataset; gray = todas, coloridos = melhores de cada run."""
    b, ok = best(D, m), D["ok"]
    need(b, xcols + [m], law)
    fig, ax = panels(D, len(xcols))
    for i, x in enumerate(xcols):
        for j, ds in enumerate(D["datasets"]):
            a, bd, od = ax[i, j], b[b["dataset"] == ds], ok[ok["dataset"] == ds] if not ok.empty else ok
            logx = islog(pd.concat([bd[x], od[x]]) if x in od else bd[x])
            gray(a, od, x, m)
            dots(a, D, bd, x, m)
            if pareto and not od.empty:
                p = od[[x, m]].dropna().sort_values(x)
                v = p[m] if METRICS[m][2] else -p[m]
                p = p[v >= v.cummax()]
                a.step(p[x], p[m], where="post", color="k", lw=1, zorder=2, label="fronteira de Pareto")
            fit_both(D, a, bd, x, m, logx, {"lei": law if len(xcols) == 1 else f"{law}:{x}", "metrica": m, "dataset": ds})
            if logx:
                a.set_xscale("log")
            a.set(xlabel=x + (" (log)" if logx else ""), title=ds if i == 0 else "")
            a.legend(loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False)
        ax[i, 0].set_ylabel(ylab(m))
    save(D, fig, f"{law}_{m}")


def flops(D, m):
    law_scatter(D, "flops", m, ["flops", "activations"], pareto=True)


def growth(D, m):
    law_scatter(D, "growth", m, ["growth_w_obs", "growth_k_obs"])


def superpixels_r2(D, m):
    """n_superpixels dos melhores × r² em log-log; inclinação 1 = área de superpixel constante."""
    b = best(D, m).copy()
    need(b, ["sp_area", "g_n_superpixels"], "superpixels_r2")
    b["r2"], b["n_sp"] = b["sp_area"] * b["g_n_superpixels"], b["g_n_superpixels"]
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        a, bd = ax[0, j], b[(b["dataset"] == ds)].dropna(subset=["r2", "n_sp"])
        dots(a, D, bd, "r2", "n_sp")
        a.set(xscale="log", yscale="log", xlabel="r² (log)", title=ds)
        if bd["r2"].nunique() < 2:
            continue
        f = bootstrap_ci(bd["r2"], bd["n_sp"], fit=fit_power, param="b", groups=bd["split"], n_boot=D["L"]["n_boot"],
                         ci=D["L"]["ci"], seed=D["L"]["seed"], loglog=True)
        s, r2, lo, hi = f["params"]["b"], f["r2"], f["params"]["lo"], f["params"]["hi"]
        xs = np.geomspace(bd["r2"].min(), bd["r2"].max(), 50)
        a.plot(xs, f["predict"](xs), "C3", label=f"inclinação={s:.2f} IC{D['L']['ci']:.0%}[{lo:.2f}, {hi:.2f}]  R²(log)={r2:.2f}")
        g = np.exp(np.log(bd[["r2", "n_sp"]]).mean())
        a.plot(xs, g["n_sp"] * xs / g["r2"], "k:", label="referência inclinação 1")
        a.legend(loc="best")
        D["summary"].append({"lei": "superpixels_r2", "metrica": m, "dataset": ds, "forma": "log-log",
                             "formula": f"n_sp ∝ (r²)^{s:.2f}", "r2": r2})
    ax[0, 0].set_ylabel(f"n_superpixels (log)\nmelhores por {METRICS[m][0]}")
    save(D, fig, f"superpixels_r2_{m}")


def rf_superpixel(D, m):
    """rf_cum × lado do superpixel (√sp_area), reta pela origem; boxplot de rf_sp_ratio por dataset."""
    b = best(D, m).copy()
    need(b, ["sp_area", "rf_cum", "rf_sp_ratio"], "rf_superpixel")
    b["sp_side"] = np.sqrt(b["sp_area"])
    fig, ax = panels(D, sharey=False, extra=1)
    for j, ds in enumerate(D["datasets"]):
        a, bd = ax[0, j], b[b["dataset"] == ds].dropna(subset=["sp_side", "rf_cum"])
        if j:
            a.sharey(ax[0, 0])
        dots(a, D, bd, "sp_side", "rf_cum")
        a.set(xlabel="√sp_area (lado do superpixel, px)", title=ds)
        if bd.empty:
            continue
        slope = lambda d: (d["sp_side"] * d["rf_cum"]).sum() / (d["sp_side"] ** 2).sum()
        c, ci = slope(bd), np.percentile([slope(pick(bd, k)) for k in boot(D, bd)], q(D))
        r2 = 1 - ((bd["rf_cum"] - c * bd["sp_side"]) ** 2).sum() / max(((bd["rf_cum"] - bd["rf_cum"].mean()) ** 2).sum(), 1e-12)
        xs = np.linspace(0, bd["sp_side"].max() * 1.05, 20)
        a.plot(xs, c * xs, "C3", label=f"rf = {c:.2f}·√sp_area  IC{D['L']['ci']:.0%}[{ci[0]:.2f}, {ci[1]:.2f}]  R²={r2:.2f}")
        a.legend(loc="best")
        D["summary"].append({"lei": "rf_superpixel", "metrica": m, "dataset": ds, "forma": "origem",
                             "formula": f"rf = {c:.2f}·√sp_area", "r2": r2})
    ax[0, 0].set_ylabel(f"rf_cum (px)\nmelhores por {METRICS[m][0]}")
    a = ax[0, -1]
    a.boxplot([b.loc[b["dataset"] == ds, "rf_sp_ratio"].dropna() for ds in D["datasets"]], tick_labels=D["datasets"])
    a.axhline(1, color="k", ls=":", lw=1)
    a.set(ylabel="rf_sp_ratio", title="razão rf/√sp_area")
    a.tick_params(axis="x", rotation=30)
    save(D, fig, f"rf_superpixel_{m}")


def budget_ratio(D, m):
    """N/D ótimo e filters_per_class por dataset: box + strip, média global e CV."""
    b = best(D, m)
    need(b, ["budget_ratio", "filters_per_class"], "budget_ratio")
    fig, ax = plt.subplots(1, 2, figsize=(5 + 1.2 * len(D["datasets"]), 4), squeeze=False)
    rng = np.random.default_rng(0)
    for a, col in zip(ax[0], ["budget_ratio", "filters_per_class"]):
        vals = [b.loc[b["dataset"] == ds, col].dropna() for ds in D["datasets"]]
        a.boxplot(vals, tick_labels=D["datasets"], showfliers=False)
        for i, (ds, v) in enumerate(zip(D["datasets"], vals), 1):
            g = b.loc[v.index]
            a.scatter(i + rng.uniform(-.15, .15, len(v)), v, c=[D["pc"].get(p, "k") for p in g["pct"]],
                      marker="o", edgecolor="k", lw=.3, s=22, zorder=3)
            if len(v):
                a.text(i, v.max(), f"CV={v.std() / v.mean():.2f}", ha="center", va="bottom", fontsize=8)
        a.axhline(b[col].mean(), color="C3", ls="--", lw=1, label=f"média global = {b[col].mean():.3g}")
        if islog(b[col]):
            a.set_yscale("log")
        a.set(ylabel=col, title=f"{col} ótimo ({METRICS[m][0]})")
        a.tick_params(axis="x", rotation=30)
        a.legend(loc="best")
    save(D, fig, f"budget_ratio_{m}")


def pct(D, m):
    """Melhor métrica × pct: média ± desvio entre splits, saturação marcada."""
    b = best(D, m)
    need(b, ["pct", m], "pct")
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        a, bd = ax[0, j], b[b["dataset"] == ds]
        s = bd.groupby("pct")[m].agg(["mean", "std"])
        dots(a, D, bd, "pct", m)
        a.errorbar(s.index, s["mean"], s["std"], color="k", capsize=3, lw=1.2, zorder=4, label="média ± desvio (splits)")
        fit_both(D, a, bd, "pct", m, islog(bd["pct"]), {"lei": "pct", "metrica": m, "dataset": ds})
        if len(s):
            top = s["mean"].max() if METRICS[m][2] else s["mean"].min()
            f = D["L"]["sat_frac"]
            hit = s.index[(s["mean"] >= top * f) if METRICS[m][2] else (s["mean"] <= top / f)]
            a.axvline(hit.min(), color="0.3", ls=":", label=f"satura em {hit.min()}% ({f:.0%} do melhor)")
        if islog(bd["pct"]):
            a.set_xscale("log")
        a.set(xlabel="pct de treino", title=ds)
        a.legend(loc="best")
    ax[0, 0].set_ylabel(ylab(m))
    save(D, fig, f"pct_{m}")


def genes(df):
    """Genes não atômicos com algum valor ativo."""
    return [c for c in df if c.startswith("g_") and not c.rsplit("_", 1)[-1].startswith("l") and df[c].notna().any()]


def cell(v, rng, stab):
    """(valor 0..1, texto, categórico?) de uma célula gene × grupo."""
    v = v.dropna()
    if v.empty:
        return np.nan, "", False
    if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
        span = max(rng[1] - rng[0], 1e-12)
        return (v.std(ddof=0) / span, f"{v.std(ddof=0):.2g}", False) if stab else ((v.mean() - rng[0]) / span, f"{v.mean():.3g}", False)
    f = v.astype(str).value_counts(normalize=True)
    return (1 - f.iloc[0], f"{f.index[0]}", True) if stab else (f.iloc[0], f"{f.index[0]}\n{f.iloc[0]:.0%}", True)


def heat(ax, M, T, cat, rows, cols, cmaps):
    for mask, cm in zip((cat, ~cat), cmaps):
        ax.imshow(np.ma.masked_where(mask | np.isnan(M), M), cmap=cm, vmin=0, vmax=1, aspect="auto")
    for (i, k), t in np.ndenumerate(T):
        ax.text(k, i, t, ha="center", va="center", fontsize=5.5)
    ax.set(xticks=range(len(cols)), xticklabels=cols, yticks=range(len(rows)), yticklabels=rows)


def gene_table(b, ok, gs, by, stab):
    cols = sorted(b.groupby(by).groups)
    M, T, C = (np.full((len(gs), len(cols)), np.nan), np.full((len(gs), len(cols)), "", dtype=object),
               np.zeros((len(gs), len(cols)), bool))
    src, bi = (ok if not ok.empty else b), b.set_index(by)
    for i, g in enumerate(gs):
        num = pd.to_numeric(src[g] if g in src else b[g], errors="coerce")
        for k, key in enumerate(cols):
            M[i, k], T[i, k], C[i, k] = cell(bi.loc[[key], g], (num.min(), num.max()), stab)
    return M, T, C, cols


def genes_pct(D, m):
    """Heatmap gene × pct dos melhores: numérico = média normalizada; categórico = moda + frequência."""
    b = best(D, m)
    need(b, ["pct"], "genes_pct")
    gs = genes(b)
    fig, ax = plt.subplots(1, len(D["datasets"]), figsize=(3.2 * len(D["datasets"]) + 2, .3 * len(gs) + 2),
                           sharey=True, squeeze=False)
    for j, ds in enumerate(D["datasets"]):
        bd = b[b["dataset"] == ds]
        if bd.empty:
            continue
        M, T, C, cols = gene_table(bd, D["ok"], gs, "pct", False)
        heat(ax[0, j], M, T, C, gs, [f"{c}%" for c in cols], ("Blues", "Oranges"))
        ax[0, j].set_title(ds)
    for cm, loc, lab in (("Blues", "right", "numérico: média normalizada no domínio observado"),
                         ("Oranges", "bottom", "categórico: frequência da moda")):
        fig.colorbar(plt.cm.ScalarMappable(cmap=cm), ax=ax[0, :].tolist(), location=loc, fraction=.04, pad=.08, label=lab)
    fig.suptitle(f"Genes dos melhores por {METRICS[m][0]} ({METRICS[m][1]})")
    save(D, fig, f"genes_pct_{m}", keys=False, tight=False)


def stability(D, m):
    """Heatmap gene × (dataset, pct): desvio entre splits normalizado (num.) ou 1 − concordância (cat.)."""
    b = best(D, m)
    need(b, ["pct", "dataset"], "stability")
    b = b[b["dataset"].isin(D["datasets"])]
    gs = genes(b)
    M, T, C, cols = gene_table(b, D["ok"], gs, ["dataset", "pct"], True)
    fig, ax = plt.subplots(figsize=(.7 * len(cols) + 3, .3 * len(gs) + 2))
    heat(ax, M, T, C, gs, [f"{d}\n{p}%" for d, p in cols], ("Reds", "Reds"))
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax, label="instabilidade entre splits (0 = estável)\n"
                 "num.: desvio/faixa do domínio · cat.: 1 − concordância")
    ax.set_title(f"Estabilidade dos melhores por {METRICS[m][0]} ({METRICS[m][1]})")
    save(D, fig, f"stability_{m}", keys=False)


def act_pool_head(D, m):
    """Boxplot da métrica por activation, pool_type e head, por dataset (todas as avaliações ok)."""
    ok = D["ok"].copy()
    need(ok, [m, "g_head"], "act_pool_head")
    for g in ("activation", "pool_type"):
        ok[g] = ok[f"g_{g}"].fillna(ok.get(f"{g}_l1")) if f"g_{g}" in ok else ok.get(f"{g}_l1")
    ok["head"] = ok["g_head"]
    fig, ax = panels(D, 3)
    for i, g in enumerate(("activation", "pool_type", "head")):
        for j, ds in enumerate(D["datasets"]):
            od = ok[(ok["dataset"] == ds) & ok[g].notna()]
            lv = sorted(od[g].astype(str).unique())
            if lv:
                ax[i, j].boxplot([od.loc[od[g].astype(str) == v, m].dropna() for v in lv], tick_labels=lv, showfliers=False)
            ax[i, j].set(xlabel=g, title=ds if i == 0 else "")
        ax[i, 0].set_ylabel(ylab(m))
    save(D, fig, f"act_pool_head_{m}", keys=False)


def compound_vs_single(D, m):
    """Métrica × FLOPs no scaling: curva composta e uma por dimensão."""
    s = D["scal"]
    s = s[s["status"] == "ok"] if "status" in s else s
    need(s, ["scale_mode", "phi", "flops", m], "scaling/results.csv")
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        a, sd = ax[0, j], s[s["dataset"] == ds]
        for k, (mode, g) in enumerate(sorted(sd.groupby("scale_mode"))):
            c = g.groupby("phi")[["flops", m]].mean().sort_values("flops")
            a.plot(c["flops"], c[m], marker="o", lw=2.6 if mode.startswith("compound") else 1.3, color=f"C{k}", label=mode)
        if islog(sd["flops"]):
            a.set_xscale("log")
        a.set(xlabel="flops (log)" if islog(sd["flops"]) else "flops", title=ds)
        a.legend(loc="best")
    ax[0, 0].set_ylabel(ylab(m))
    save(D, fig, f"compound_vs_single_{m}", keys=False)


def layer_mode(D, m):
    """Lei de FLOPs sobreposta nos três modos (melhor de cada run em cada modo)."""
    ok = D["ok"]
    need(ok, ["g_layer_mode", "flops", m], "layer_mode")
    g = ok.dropna(subset=[m]).groupby(RUN + ["g_layer_mode"])[m]
    b = ok.loc[g.idxmax() if METRICS[m][2] else g.idxmin()]
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        a, od, bd = ax[0, j], ok[ok["dataset"] == ds], b[b["dataset"] == ds]
        logx = islog(od["flops"])
        gray(a, od, "flops", m)
        for mode, c in MODES.items():
            bm = bd[bd["g_layer_mode"] == mode]
            a.scatter(bm["flops"], bm[m], color=c, edgecolor="k", lw=.4, s=30, zorder=3)
            fit_both(D, a, bm, "flops", m, logx, {"lei": f"layer_mode:{mode}", "metrica": m, "dataset": ds}, color=c, tag=f"{mode} ")
        if logx:
            a.set_xscale("log")
        a.set(xlabel="flops (log)" if logx else "flops", title=ds)
        a.legend(loc="best")
    ax[0, 0].set_ylabel(ylab(m))
    save(D, fig, f"layer_mode_{m}", keys=False)


def layer_profile(D, m):
    """N_l e k_l por camada dos melhores, uma linha por config; g_w ajustado em N_l = N_1·g_w^(l−1)."""
    b = best(D, m)
    need(b, ["N_l1", "k_l1"], "layer_profile")
    L = [l for l in range(1, 8) if f"N_l{l}" in b]
    fig, ax = panels(D, 2)
    for j, ds in enumerate(D["datasets"]):
        bd = b[b["dataset"] == ds]
        for i, key in enumerate(("N", "k")):
            P = bd[[f"{key}_l{l}" for l in L]].apply(pd.to_numeric, errors="coerce").to_numpy()
            for row, p in zip(P, bd["pct"]):
                ax[i, j].plot(L, row, marker="o", ms=3, color=D["pc"].get(p, "k"), alpha=.75)
            ax[i, j].set(xlabel="camada l", xticks=L, title=ds if i == 0 else "")
        P = np.log(bd[[f"N_l{l}" for l in L]].apply(pd.to_numeric, errors="coerce").to_numpy())
        dl, dy = np.broadcast_to(np.array(L) - 1, P.shape), P - P[:, :1]
        ok = ~np.isnan(dy) & (dl > 0)
        if ok.any():
            gw = np.exp((dl[ok] * dy[ok]).sum() / (dl[ok] ** 2).sum())
            n1 = np.nanmedian(np.exp(P[:, 0]))
            ax[0, j].plot(L, n1 * gw ** (np.array(L) - 1), "k--", lw=1.6, label=f"g_w ajustado = {gw:.2f}")
            ax[0, j].legend(loc="best")
            D["summary"].append({"lei": "layer_profile", "metrica": m, "dataset": ds, "forma": "geométrica",
                                 "formula": f"N_l = N_1·{gw:.2f}^(l−1)", "r2": np.nan})
    ax[0, 0].set_ylabel(f"N_l (filtros)\nmelhores por {METRICS[m][0]}")
    ax[1, 0].set_ylabel("k_l (kernel)")
    save(D, fig, f"layer_profile_{m}")


def transfer(D, m):
    """Matriz origem × destino: métrica relativa ao melhor do destino (1 = melhor)."""
    t, own = D["transfer"], D["agg"]
    need(t, ["source", "target", "pct", "split", m], "laws/transfer.csv")
    need(own, [f"best_{m}"], "aggregated.csv")
    t = t[t["status"] == "ok"] if "status" in t else t
    t = t.merge(own[RUN + [f"best_{m}"]].rename(columns={"dataset": "target"}), on=["target", "pct", "split"])
    t["rel"] = t[m] / t[f"best_{m}"] if METRICS[m][2] else t[f"best_{m}"] / t[m]
    R = t.groupby(["source", "target"])["rel"].mean().unstack()
    names = sorted(set(R.index) | set(R.columns))
    R = R.reindex(index=names, columns=names)
    for n in names:
        R.loc[n, n] = 1.0   # diagonal = ótimo próprio
    fig, ax = plt.subplots(figsize=(1.1 * R.shape[1] + 3, .8 * R.shape[0] + 2))
    im = ax.imshow(R.to_numpy(float), cmap="viridis", vmin=0, vmax=1)
    for (i, k), v in np.ndenumerate(R.to_numpy(float)):
        ax.text(k, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", color="w" if v < .6 else "k", fontsize=8)
    ax.set(xticks=range(R.shape[1]), xticklabels=R.columns, yticks=range(R.shape[0]), yticklabels=R.index,
           xlabel="destino", ylabel="origem", title=f"Transferência ({METRICS[m][0]})")
    ax.tick_params(axis="x", rotation=30)
    fig.colorbar(im, ax=ax, label="relativo ao ótimo do destino" + (" (ótimo/valor)" if not METRICS[m][2] else " (valor/ótimo)"))
    save(D, fig, f"transfer_{m}", keys=False)


def correlation(D):
    """Spearman derivadas × métricas, um painel por dataset (laws/correlation.csv se existir)."""
    c, ok, ms = D["corr"], D["ok"], D["metrics"]
    if not c.empty and {"dataset", "metric", "derived", "rho"} <= set(c):
        c = c.pivot_table(index=["dataset", "derived"], columns="metric", values="rho", aggfunc="median")
    else:   # mesmo sinal do analyze_laws: ρ > 0 = a métrica melhora (DBI com sinal trocado)
        need(ok, ms, "correlation")
        der = [v for v in DERIVED if v in ok and pd.to_numeric(ok[v], errors="coerce").notna().any()]
        sgn = pd.Series({x: 1 if METRICS[x][2] else -1 for x in ms})
        c = pd.concat({ds: (g[der + ms].apply(pd.to_numeric, errors="coerce").corr("spearman").loc[der, ms] * sgn)
                       for ds, g in ok.groupby(RUN[0]) if ds in D["datasets"]}, names=["dataset", "derived"])
    c = c.reset_index().rename(columns={"derived": "variable"})
    ms = [x for x in ms if x in c]
    fig, ax = plt.subplots(1, len(D["datasets"]), figsize=(2.6 * len(D["datasets"]) + 2, .3 * c["variable"].nunique() + 2),
                           sharey=True, squeeze=False)
    for j, ds in enumerate(D["datasets"]):
        M = c[c["dataset"] == ds].set_index("variable")[ms]
        im = ax[0, j].imshow(M.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for (i, k), v in np.ndenumerate(M.to_numpy(float)):
            ax[0, j].text(k, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6)
        ax[0, j].set(xticks=range(len(ms)), xticklabels=ms,
                     yticks=range(len(M)), yticklabels=M.index, title=ds)
    fig.colorbar(im, ax=ax[0, -1], label="ρ de Spearman (+ = a métrica melhora;\nDBI com sinal trocado)")
    save(D, fig, "correlation", keys=False)


def metric_agreement(D):
    """Dispersão entre pares de métricas nos best_config, com correlação de postos."""
    b = D["agg"] if not D["agg"].empty else best(D, D["metrics"][0])
    ms = [x for x in D["metrics"] if x in b]
    need(b, ms if len(ms) > 1 else ["2 métricas"], "metric_agreement")
    pairs = [(p, q) for i, p in enumerate(ms) for q in ms[i + 1:]]
    fig, ax = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.8), squeeze=False)
    for a, (p, q) in zip(ax[0], pairs):
        for k, ds in enumerate(D["datasets"]):
            bd = b[b["dataset"] == ds]
            a.scatter(bd[p], bd[q], color=f"C{k}", edgecolor="k", lw=.4, s=30, label=ds)
        a.set(xlabel=ylab(p), ylabel=ylab(q), title=f"ρ Spearman = {b[p].corr(b[q], method='spearman'):.2f}")
        a.legend(loc="best")
    save(D, fig, "metric_agreement", keys=False)


def convergence(D):
    """Melhor (cheio) e média (tracejado) por geração para as métricas, com stop_reason no fim."""
    g = D["gen"]
    need(g, ["gen"] + [f"{x}_best" for x in D["metrics"]], "generations.csv")
    fig, ax = plt.subplots(len(D["metrics"]), len(D["datasets"]), figsize=(4.4 * len(D["datasets"]), 3 * len(D["metrics"])),
                           sharex=True, squeeze=False)
    for (ds, p, s), r in g.sort_values("gen").groupby(RUN):
        if ds not in D["datasets"]:
            continue
        j, c = D["datasets"].index(ds), D["pc"].get(p, "k")
        for i, x in enumerate(D["metrics"]):
            ax[i, j].plot(r["gen"], r[f"{x}_best"], color=c, lw=1.2, alpha=.8)
            ax[i, j].plot(r["gen"], r[f"{x}_mean"], color=c, lw=.8, ls="--", alpha=.6)
            sr = r["stop_reason"].dropna().astype(str) if "stop_reason" in r else []
            if len(sr) and sr.iloc[-1]:
                ax[i, j].annotate(sr.iloc[-1], (r["gen"].iloc[-1], r[f"{x}_best"].iloc[-1]), fontsize=6, color=c)
    for i, x in enumerate(D["metrics"]):
        ax[i, 0].set_ylabel(ylab(x))
    for j, ds in enumerate(D["datasets"]):
        ax[0, j].set_title(ds)
        ax[-1, j].set_xlabel("geração")
    h = [plt.Line2D([], [], color=c, label=f"{p}%") for p, c in D["pc"].items()]
    h += [plt.Line2D([], [], color="k", label="melhor"), plt.Line2D([], [], color="k", ls="--", label="média")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, 0), ncol=len(h), frameon=False)
    save(D, fig, "convergence", keys=False)


def residual(D, m):
    """§7.6: residual_profile (C_l dos melhores), residual_growth (métrica × growth_l), residual_vs_plain (residual 1 × 0)."""
    rs, plain = D["res_ok"], D["plain_ok"]
    need(rs, [m, "C_l1"], "residual/results.csv")
    b = best(D, m, rs)
    C = [l for l in range(1, 6) if f"C_l{l}" in b]
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        bd = b[b["dataset"] == ds]
        for row, p in zip(bd[[f"C_l{l}" for l in C]].apply(pd.to_numeric, errors="coerce").to_numpy(), bd["pct"]):
            ax[0, j].plot(C, row, marker="o", ms=3, color=D["pc"].get(p, "k"), alpha=.75)
        ax[0, j].set(xlabel="camada l", xticks=C, title=ds, yscale="log")
    ax[0, 0].set_ylabel(f"C_l (canais totais, log)\nmelhores por {METRICS[m][0]}")
    save(D, fig, f"residual_profile_{m}")

    G = [l for l in range(2, 6) if f"g_growth_l{l}" in rs]
    fig, ax = panels(D)
    for j, ds in enumerate(D["datasets"]):
        od, bd = rs[rs["dataset"] == ds], b[b["dataset"] == ds]
        for l in G:
            gray(ax[0, j], od, f"g_growth_l{l}", m)
            ax[0, j].scatter(bd[f"g_growth_l{l}"], bd[m], color=f"C{l}", edgecolor="k", lw=.4, s=30, zorder=3, label=f"growth_l{l}")
        ax[0, j].axvline(2, color="k", ls=":", lw=1, label="2× (C_l = 2·C_(l−1))")
        ax[0, j].set(xlabel="growth_l", title=ds)
        ax[0, j].legend(loc="best")
    ax[0, 0].set_ylabel(ylab(m))
    save(D, fig, f"residual_growth_{m}", keys=False)

    pb = best(D, m, plain)
    fig, ax = panels(D, 2)
    for i, x in enumerate(("flops", "feat_dim")):
        for j, ds in enumerate(D["datasets"]):
            a = ax[i, j]
            for tag, df, src, c in (("sem resíduo", pb, plain, "C0"), ("resíduo", b, rs, "C1")):
                df, src = df[df["dataset"] == ds], src[src["dataset"] == ds] if not src.empty else src
                if src.empty or x not in src:
                    continue
                a.scatter(src[x], src[m], s=5, color=c, alpha=.15, zorder=1, rasterized=True)
                a.scatter(df[x], df[m], color=c, edgecolor="k", lw=.4, s=32, zorder=3, label=f"{tag} (melhores)")
                fit_both(D, a, df, x, m, islog(src[x]), {"lei": f"residual:{x}:{'res' if c == 'C1' else 'plain'}", "metrica": m, "dataset": ds},
                         color=c, tag=f"{tag} ")
            if islog(pd.concat([plain.get(x, pd.Series(dtype=float)), rs[x]])):
                a.set_xscale("log")
            a.set(xlabel=x, title=ds if i == 0 else "")
            a.legend(loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False)
        ax[i, 0].set_ylabel(ylab(m))
    save(D, fig, f"residual_vs_plain_{m}", keys=False)


LAWS = {f.__name__: f for f in (flops, superpixels_r2, rf_superpixel, budget_ratio, pct, genes_pct, compound_vs_single,
                                layer_mode, layer_profile, growth, act_pool_head, stability, transfer, residual)}
SUPPORT = {f.__name__: f for f in (correlation, metric_agreement, convergence)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--laws", nargs="+", default=[*LAWS, *SUPPORT], choices=[*LAWS, *SUPPORT])
    ap.add_argument("--results-dir")
    a = ap.parse_args()
    ga = yaml.safe_load(open(GA_CONFIG))
    root = Path(a.results_dir or ga["results_dir"])
    L = ga["laws"]
    datasets = a.datasets or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "laws")
    read = lambda f: pd.read_csv(f) if f.exists() else pd.DataFrame()
    cat = lambda pat: pd.concat([pd.read_csv(f) for d in datasets for f in sorted(root.glob(f"{d}/{pat}"))] or [pd.DataFrame()],
                                ignore_index=True)
    res = cat("pct_*/split_*/results.csv")
    # §7.7: residual é gene (residual / g_residual = 0/1) nos mesmos runs; a pasta residual/ é opcional
    ok_all = pd.concat([res, cat("residual/pct_*/split_*/results.csv")], ignore_index=True)
    ok_all = ok_all[ok_all["status"] == "ok"] if "status" in ok_all else ok_all
    flag = pd.to_numeric(ok_all.get("residual", ok_all.get("g_residual", pd.Series(0, index=ok_all.index))),
                         errors="coerce").fillna(0)
    agg = read(root / "laws" / "aggregated.csv")
    agg = agg[agg["dataset"].isin(datasets)] if "dataset" in agg else agg
    pcts = sorted(set(res.get("pct", pd.Series(dtype=int))) | set(agg.get("pct", pd.Series(dtype=int))))
    D = {"datasets": datasets, "metrics": a.metrics, "agg": agg, "gen": cat("pct_*/split_*/generations.csv"),
         "ok": res[res["status"] == "ok"] if "status" in res else res, "scal": cat("scaling/results.csv"),
         "res_ok": ok_all[flag == 1], "plain_ok": ok_all[flag != 1],
         "transfer": read(root / "laws" / "transfer.csv"), "corr": read(root / "laws" / "correlation.csv"),
         "out": root / "laws" / "figures", "summary": [], "L": L,
         "pc": dict(zip(pcts, plt.cm.viridis(np.linspace(0, .9, max(len(pcts), 1)))))}
    D["out"].mkdir(parents=True, exist_ok=True)
    for law in a.laws:
        for m in ([None] if law in SUPPORT else a.metrics):
            try:
                SUPPORT[law](D) if m is None else LAWS[law](D, m)
            except Skip as e:
                print(f"[aviso] {law}{'_' + m if m else ''} pulado: {e}")
    s = pd.DataFrame(D["summary"])
    if s.empty:
        return print("[aviso] nenhum ajuste para resumir")
    thr = L["r2_min"]
    print(f"\n{'lei':<30}{'métrica':<12}{'forma':<12}{'fórmula (melhor R²)':<44}{'R² med.':<9}datasets onde vale (R²≥{thr})")
    for (law, m), g in s.groupby(["lei", "metrica"], sort=False):
        top = g.loc[g["r2"].fillna(-np.inf).idxmax()]
        ok = g.loc[g["r2"] >= thr, "dataset"]
        print(f"{law:<30}{m:<12}{top['forma']:<12}{str(top['formula'])[:42]:<44}{g['r2'].median():<9.2f}{', '.join(ok) or '-'}")


if __name__ == "__main__":
    main()
