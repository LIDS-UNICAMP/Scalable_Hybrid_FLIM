"""Figuras da etapa 3 (por camada) a partir de reports/lejepa_diagnostics/data/layer_metrics.csv.

03_kappa_by_layer, 04_eigenvalue_spectrum (GAP; _cov para COV), 05_cka_heatmap -> PNG + SVG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C0, C1 = "#2a78d6", "#eb6834"         # P0 (anchor), P1 (ssl_state)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROLE = {"delta_pos": "maior Δκ positivo", "delta_neg": "maior Δκ negativo"}
DSNAME = {"remote_sensing": "remote_sensing", "larvae": "larvae"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
                     "svg.fonttype": "none"})


def _short(pid):
    _ds, search, cand = pid.split(":")
    return f"{search.replace('modehb_', '')} {cand}"


def _pairs(df):
    order = {"delta_pos": 0, "delta_neg": 1}
    p = df[["pair_id", "role", "delta_kappa_manifest"]].drop_duplicates()
    return sorted(p.itertuples(index=False), key=lambda r: order[r.role])


def _title(r):
    return f"{_short(r.pair_id)} ({ROLE[r.role]}: Δκ_val={r.delta_kappa_manifest:+.3f})"


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_kappa(df, ds, out):
    pairs = _pairs(df)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.3 * len(pairs)), squeeze=False, sharey=True)
    for i, pr in enumerate(pairs):
        d = df[df.pair_id == pr.pair_id]
        for j, feat in enumerate(("gap", "cov")):
            ax = axes[i, j]
            for m, c, lab in (("P0", C0, "P0 (anchor, antes do LeJEPA)"), ("P1", C1, "P1 (ssl_state, depois)")):
                s = d[(d.model == m) & (d.feature == feat)].sort_values("layer_idx")
                ax.plot(s.layer_idx, s.kappa, "-o", color=c, lw=2, ms=6, label=lab,
                        mec="white", mew=1.2)
            s0 = d[(d.model == "P0") & (d.feature == feat)].sort_values("layer_idx")
            s1 = d[(d.model == "P1") & (d.feature == feat)].sort_values("layer_idx")
            for x, a, b in zip(s0.layer_idx, s0.kappa, s1.kappa):
                ax.annotate(f"{b - a:+.2f}", (x, max(a, b)), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=INK2)
            dims = s0.dim.tolist()
            ax.set_xticks(s0.layer_idx)
            ax.set_xticklabels([f"L{k}\n(d={dd})" for k, dd in zip(s0.layer_idx, dims)], fontsize=7)
            name = "GAP do mapa" if feat == "gap" else f"COV entre canais (PCA treino, k≤{int(d.pca_k_requested.iloc[0])})"
            ax.set_title(f"{name}\n{_title(pr)}", fontsize=8.5)
            ax.set_xlabel("camada (bloco)")
            if j == 0:
                ax.set_ylabel("κ de validação (SVM linear C=100, sem scaler)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    n_tr, n_va = int(df.n_train.iloc[0]), int(df.n_val.iloc[0])
    fig.suptitle(f"{DSNAME[ds]}: κ por camada, P0 × P1 (treino n={n_tr}, validação n={n_va}; rótulos = Δκ P1−P0)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    _save(fig, out, "03_kappa_by_layer")


def fig_spectrum(df, ds, out, feat="gap"):
    pairs = _pairs(df)
    nl = int(df.layer_idx.max())
    fig, axes = plt.subplots(len(pairs), nl, figsize=(2.3 * nl, 2.9 * len(pairs)), squeeze=False)
    for i, pr in enumerate(pairs):
        d = df[(df.pair_id == pr.pair_id) & (df.feature == feat)]
        for L in range(1, nl + 1):
            ax = axes[i, L - 1]
            row0 = d[(d.model == "P0") & (d.layer_idx == L)]
            if row0.empty:
                ax.axis("off")
                continue
            for m, c in (("P0", C0), ("P1", C1)):
                r = d[(d.model == m) & (d.layer_idx == L)].iloc[0]
                p = np.asarray(json.loads(r.eig_norm_json))
                p = p[p > 1e-12]
                ax.semilogy(np.arange(1, len(p) + 1), p, color=c, lw=2,
                            label=f"{m}: erank={r.erank:.1f}")
                ax.axvline(r.erank, color=c, ls="--", lw=1.2)
            ax.set_title(f"L{L} (d={int(row0.dim.iloc[0])})", fontsize=8.5)
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
            ax.set_xlabel("componente", fontsize=7.5)
            if L == 1:
                ax.set_ylabel(f"{_short(pr.pair_id)}\n({ROLE[pr.role]})\nautovalor normalizado", fontsize=7.5)
    n = int(df.geom_n.iloc[0])
    what = "GAP do mapa" if feat == "gap" else "COV entre canais"
    fig.suptitle(f"{DSNAME[ds]}: espectro normalizado de autovalores ({what}), P0 (azul) × P1 (laranja); "
                 f"tracejado = rank efetivo exp(H); validação n={n} pareado", fontsize=10, y=1.0)
    fig.tight_layout()
    _save(fig, out, "04_eigenvalue_spectrum" + ("" if feat == "gap" else "_cov"))


def fig_cka(df, ds, out):
    pairs = _pairs(df)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(9.2, 4.2 * len(pairs)), squeeze=False)
    im = None
    for i, pr in enumerate(pairs):
        for j, feat in enumerate(("gap", "cov")):
            d = df[(df.pair_id == pr.pair_id) & (df.feature == feat) & (df.model == "P0")].sort_values("layer_idx")
            Ls = [f"L{k}" for k in d.layer_idx]
            M = np.array([[d[d.layer == a][f"cka_P0{a}_vs_P1{b}"].iloc[0] for b in Ls] for a in Ls])
            ax = axes[i, j]
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            ax.grid(False)
            for a in range(len(Ls)):
                for b in range(len(Ls)):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[a, b] > 0.6 else INK,
                            fontweight="bold" if a == b else "normal")
            ax.set_xticks(range(len(Ls)), Ls)
            ax.set_yticks(range(len(Ls)), Ls)
            ax.set_xlabel("camada de P1 (depois do LeJEPA)")
            ax.set_ylabel("camada de P0 (antes do LeJEPA)")
            name = "GAP" if feat == "gap" else "COV"
            ax.set_title(f"CKA linear ({name}): {_short(pr.pair_id)}\n{ROLE[pr.role]}, Δκ_val={pr.delta_kappa_manifest:+.3f}",
                         fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("CKA linear (1 = mesma geometria; baixo = representação mudou)")
    n = int(df.geom_n.iloc[0])
    fig.suptitle(f"{DSNAME[ds]}: CKA linear camadas P0 × camadas P1 (validação, n={n} amostras pareadas)",
                 fontsize=10.5)
    _save(fig, out, "05_cka_heatmap")


def plot_all(ds, data_dir, out):
    df = pd.read_csv(Path(data_dir) / "layer_metrics.csv")
    df = df[df.dataset == ds]
    if df.empty or "eig_norm_json" not in df:
        print(f"[plots] sem linhas para {ds}")
        return
    fig_kappa(df, ds, out)
    fig_spectrum(df, ds, out, "gap")
    fig_spectrum(df, ds, out, "cov")
    fig_cka(df, ds, out)
    print(f"[plots] {ds} -> {out}")
