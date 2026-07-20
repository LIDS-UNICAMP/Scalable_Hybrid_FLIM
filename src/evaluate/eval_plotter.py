# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""eval_plotter.py — Composite metric plots for the unified evaluation pipeline.

Follows the visual style established in src/utils/plot_svm_results.py:
  - seaborn whitegrid, font_scale=2.0, #FAFAFA axes background
  - Saves PDF + PNG, dpi=300

Generates per-(dataset, pct) horizontal composite figures with three panels:
  kappa | accuracy | F1-score

Usage (internal):
    from src.evaluate.eval_plotter import plot_composite
    plot_composite(df, dataset="eggs", pct=5, out_dir=Path("artifacts/plots/eggs/lejepa_pct_5"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Visual config (mirroring plot_svm_results.py) ─────────────────────────────

INIT_ORDER = ["flim", "he", "xavier", "random", "trunc_normal"]

# One colour per method — same palette as plot_svm_results.py init colours
METHOD_COLOR: dict[str, str] = {
    "SVM":          "#0072B2",   # blue
    "MLP_freeze":   "#D55E00",   # orange-red
    "MLP_unfreeze": "#009E73",   # green
}

METHOD_LABEL: dict[str, str] = {
    "SVM":          "SVM",
    "MLP_freeze":   "MLP freeze",
    "MLP_unfreeze": "MLP unfreeze",
}

METHOD_HATCH: dict[str, str] = {
    "SVM":          "",
    "MLP_freeze":   "//",
    "MLP_unfreeze": "..",
}

DATASET_LABEL: dict[str, str] = {
    "eggs":      "Helminth Eggs",
    "larvae":    "Helminth Larvae",
    "protozoan": "Protozoan Cysts",
    "helminth-eggs":   "Helminth Eggs",
    "helminth-larvae": "Helminth Larvae",
    "protozoan-cysts": "Protozoan Cysts",
}

METRIC_LABELS: dict[str, str] = {
    "kappa": "Cohen's Kappa ($\\kappa$)",
    "acc":   "Accuracy",
    "f1":    "F1-score",
}

METRICS = ["kappa", "acc", "f1"]


# ── Core plotting function ─────────────────────────────────────────────────────

def plot_composite(
    df: pd.DataFrame,
    dataset: str,
    pct: int,
    out_dir: Path,
    methods: Optional[list[str]] = None,
) -> None:
    """Generate a composite metric figure for a (dataset, pct) group.

    The figure contains three panels — kappa, accuracy, F1 — arranged
    horizontally.  X-axis shows initialization type; bars are grouped by
    evaluation method.  Error bars represent ± std across splits.

    Expected DataFrame columns:
        method    — one of "SVM", "MLP_freeze", "MLP_unfreeze"
        init      — one of "flim", "he", "xavier", "random"
        kappa     — mean Cohen's kappa
        kappa_std — std of kappa across splits
        acc       — mean accuracy
        acc_std   — std of accuracy across splits
        f1        — mean F1-score
        f1_std    — std of F1-score across splits

    Args:
        df:      Aggregated data for this (dataset, pct) group.
        dataset: Short or full dataset name.
        pct:     SSL pretraining percentage.
        out_dir: Directory to save outputs (created if needed).
        methods: Method subset to include (defaults to all with data).
    """
    if df.empty:
        return

    if methods is None:
        methods = [m for m in METHOD_COLOR if m in df["method"].unique()]
    else:
        methods = [m for m in methods if m in df["method"].unique()]

    inits = [i for i in INIT_ORDER if i in df["init"].unique()]

    if not inits or not methods:
        return

    sns.set_theme(
        style="whitegrid",
        font_scale=2.0,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(10 * n_metrics, 9),
        squeeze=False,
    )

    dataset_label = DATASET_LABEL.get(dataset, dataset)
    fig.suptitle(
        f"{dataset_label}  —  LeJEPA {pct}% pre-training",
        fontsize=26, fontweight="bold", y=1.01,
    )

    x = np.arange(len(inits))
    n_methods = len(methods)
    bar_width = min(0.22, 0.7 / max(n_methods, 1))
    half = (n_methods - 1) / 2.0
    offsets = np.array([(i - half) * bar_width for i in range(n_methods)])

    legend_patches: list[mpatches.Patch] = []

    for col, metric in enumerate(METRICS):
        ax = axes[0][col]
        std_col = f"{metric}_std"

        ax.set_title(METRIC_LABELS[metric], fontsize=22, fontweight="bold", pad=10)

        for method, offset in zip(methods, offsets):
            color = METHOD_COLOR.get(method, "#888888")
            hatch = METHOD_HATCH.get(method, "")

            method_df = df[df["method"] == method].set_index("init")

            vals = np.array([
                float(method_df.loc[init, metric]) if init in method_df.index else np.nan
                for init in inits
            ])
            errs = np.array([
                float(method_df.loc[init, std_col])
                if (init in method_df.index and std_col in method_df.columns)
                else 0.0
                for init in inits
            ])
            errs = np.nan_to_num(errs)

            ax.bar(
                x + offset, vals, bar_width,
                color=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.88,
                yerr=errs,
                error_kw={
                    "elinewidth": 1.5,
                    "capsize": 5,
                    "capthick": 1.5,
                    "alpha": 0.75,
                    "ecolor": color,
                },
            )

            if col == 0:
                legend_patches.append(
                    mpatches.Patch(
                        facecolor=color,
                        hatch=hatch,
                        edgecolor="#555555",
                        linewidth=0.5,
                        label=METHOD_LABEL.get(method, method),
                    )
                )

        ax.set_xlabel("Initialization", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels([i.upper() for i in inits], fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.55, len(inits) - 0.45)
        ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)

    axes[0][0].set_ylabel("Score", fontsize=20)

    fig.legend(
        handles=legend_patches,
        title="Method",
        title_fontsize=20,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=20,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        handlelength=2.2,
        handletextpad=0.6,
        bbox_to_anchor=(0.5, 0.05),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{dataset}_pct{pct}_composite"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [PLOT] {png_path}")


# ── SVM-style line plot (across all percentages, per init) ────────────────────

PERCENTAGES = [1, 5, 25, 50, 75, 100]
PCT_TO_POS = {p: i for i, p in enumerate(PERCENTAGES)}

INIT_LINE_STYLE: dict[str, dict] = {
    "flim":         {"color": "#0072B2", "ls": "-",   "marker": "o", "zorder": 5},
    "random":       {"color": "#009E73", "ls": "-",   "marker": "s", "zorder": 4},
    "he":           {"color": "#D55E00", "ls": "--",  "marker": "^", "zorder": 3},
    "xavier":       {"color": "#CC79A7", "ls": "--",  "marker": "D", "zorder": 2},
    "trunc_normal": {"color": "#F0E442", "ls": "-.",  "marker": "P", "zorder": 6},
}


def plot_metric_vs_pct(
    df: pd.DataFrame,
    dataset: str,
    method: str,
    metric: str,
    out_dir: Path,
) -> None:
    """Line plot: metric vs. pretraining percentage, one line per init type.

    Follows the exact style of plot_svm_results.py for a single method+metric.

    Args:
        df:      Aggregated DataFrame with columns init, pretrained_pct,
                 {metric}, {metric}_std.
        dataset: Short or full dataset name.
        method:  Method label, e.g. "SVM", "MLP_freeze", "MLP_unfreeze".
        metric:  One of "kappa", "acc", "f1".
        out_dir: Output directory.
    """
    if df.empty:
        return

    inits = [i for i in INIT_LINE_STYLE if i in df["init"].unique()]
    if not inits:
        return

    pct_col = "pretrained_pct" if "pretrained_pct" in df.columns else "pct"

    sns.set_theme(
        style="whitegrid",
        font_scale=2.0,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    dataset_label = DATASET_LABEL.get(dataset, dataset)
    method_label = METHOD_LABEL.get(method, method)
    ax.set_title(
        f"{dataset_label} — {method_label}",
        fontsize=26, fontweight="bold", pad=12,
    )

    legend_handles = []
    std_col = f"{metric}_std"

    for init in inits:
        sub = df[df["init"] == init].copy().sort_values(pct_col)
        if sub.empty:
            continue

        pcts = sub[pct_col].values
        valid = [p for p in pcts if p in PCT_TO_POS]
        if not valid:
            continue

        x = [PCT_TO_POS[p] for p in valid]
        y = sub.set_index(pct_col).loc[valid, metric].values
        yerr = (
            sub.set_index(pct_col).loc[valid, std_col].values
            if std_col in sub.columns else np.zeros(len(valid))
        )

        sty = INIT_LINE_STYLE[init]
        line, = ax.plot(
            x, y,
            color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
            linewidth=3.0, markersize=12,
            markeredgecolor="white", markeredgewidth=0.8,
            zorder=sty["zorder"], label=init.upper(),
        )
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="none", ecolor=sty["color"],
            elinewidth=1.2, capsize=5, capthick=1.5,
            alpha=0.7, zorder=sty["zorder"] - 1,
        )
        legend_handles.append(line)

    pcts_present = sorted({p for p in df[pct_col].unique() if p in PCT_TO_POS})
    tick_pos = [PCT_TO_POS[p] for p in pcts_present]

    ax.set_xlabel("Pre-training Data", fontsize=24)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{p}%" for p in pcts_present], fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=24)

    fig.legend(
        handles=legend_handles,
        title="Initialization",
        title_fontsize=22,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=22,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        handlelength=2.5,
        handletextpad=0.6,
        bbox_to_anchor=(0.5, 0.06),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{dataset}_{method}_{metric}_vs_pct"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [PLOT] {out_dir / f'{stem}.png'}")


# ── Multi-source method comparison (FLIM vs LeJEPA vs I-JEPA) ─────────────────

# Modelos oficiais do experimento — apenas estas chaves aparecem nas curvas/legenda
# (8 originais + as 2 variantes FLIM Residual, eggs-only).
# Ordem = ordem da legenda. Cores contrastivas.
METHOD_COMPARE_STYLE: dict[str, dict] = {
    "SVM_FLIM":                                {"color": "#0072B2", "ls": "-",  "marker": "o", "zorder": 7},
    "SVM_LeJEPA_trunc_normal":                 {"color": "#9467BD", "ls": "-.", "marker": "P", "zorder": 8},
    "SVM_IJEPA":                               {"color": "#E69F00", "ls": "-.", "marker": "*", "zorder": 9},
    "SVM_Distill_Proj1280":                    {"color": "#D62728", "ls": "--", "marker": "v", "zorder": 10},
    "SVM_Distill_3x3BN":                       {"color": "#E377C2", "ls": "--", "marker": "^", "zorder": 11},
    "SVM_Distill_1x1BN":                        {"color": "#17BECF", "ls": "-",  "marker": "s", "zorder": 12},
    "SVM_Distill_2l400K":                       {"color": "#2CA02C", "ls": "--", "marker": "D", "zorder": 13},
    "SVM_Distill_1x1BN_flim_frozen_eval_loss": {"color": "#000000", "ls": ":",  "marker": ">", "zorder": 14},
    "SVM_FLIMResidual_1_3":                    {"color": "#008080", "ls": "-",  "marker": "d", "zorder": 15},
    "SVM_FLIMResidual_2_3":                    {"color": "#B22222", "ls": "--", "marker": "H", "zorder": 16},
}

# Rótulos curtos (inglês) — uma entrada por modelo oficial, mesma ordem da legenda.
# Procedência de cada curva (qual CSV gera cada valor) está documentada no override
# desta dict em scripts/plot_comparison_flim.py e em
# metrics_distillation/data_provenance.md (como cada CSV de origem é gerado).
METHOD_COMPARE_LABEL: dict[str, str] = {
    "SVM_FLIM":                                 "FLIM (59.504)",               # data/reports_felipe/svm/report_svm_*.csv
    "SVM_LeJEPA_trunc_normal":                  "LeJEPA (59.504)",             # artifacts/SVM/*/*/metrics_SVM_*.csv (init=trunc_normal)
    "SVM_IJEPA":                                "I-JEPA (632M)",               # results/ijepa_svm_aggregated.csv
    "SVM_Distill_Proj1280":                     "Distill 4 (889K)",            # results/svm_distill_proj1280_results.csv
    "SVM_Distill_3x3BN":                        "Distill 3 (615K)",            # results/svm_proj1280_3x3_BN2d_results.csv (trunc_normal)
    "SVM_Distill_1x1BN":                        "Distill 1 (123K)",            # results/svm_proj1280_1x1_BN2d_results.csv (trunc_normal)
    "SVM_Distill_2l400K":                       "Distill 2 (402K)",            # results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv
    "SVM_Distill_1x1BN_flim_frozen_eval_loss":  "Distill 1 — FLIM init (123K)", # results/svm_distillation_conv_flim_frozen_results.csv (ckpt best-loss)
    "SVM_FLIMResidual_1_3":                     "FLIM Residual 1→3",           # results/svm_flim_residual_eggs.csv
    "SVM_FLIMResidual_2_3":                     "FLIM Residual 2→3",           # results/svm_flim_residual_eggs.csv
}


def plot_method_comparison(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    out_dir: Path,
    *,
    show_title:          bool = True,
    show_xlabel:         bool = True,
    legend_fontsize:     int  = 22,
    legend_ncol:         Optional[int] = None,
    tick_fontsize:       int  = 24,
    ylabel_fontsize:     int  = 28,
    stem_override:       Optional[str] = None,
) -> None:
    """Line plot: metric vs pct, one line per (method, init) combination.

    Compares SVM_FLIM, SVM_LeJEPA (all inits), and SVM_IJEPA on the same axes.

    Expected DataFrame columns:
        method         — "SVM_FLIM", "SVM_LeJEPA", or "SVM_IJEPA"
        init           — "flim", "he", "xavier", "random", or "ijepa"
        pretrained_pct — integer percentage
        {metric}       — mean metric value
        {metric}_std   — std of metric across splits

    Args:
        df:               Aggregated data for this dataset (all methods).
        dataset:          Short dataset name (e.g. "eggs", "larvae", "protozoan").
        metric:           One of "kappa", "acc", "f1".
        out_dir:          Output directory.
        show_title:       Whether to draw the dataset title above the axes.
        show_xlabel:      Whether to draw the "Pre-training Data" x-axis label.
        legend_fontsize:  Font size for legend item text.
        legend_ncol:      Number of legend columns. None → min(n_handles, 3).
                          Pass -1 to put all items in one horizontal row.
        tick_fontsize:    Font size for x-axis percentage labels.
        ylabel_fontsize:  Font size for the y-axis label.
        stem_override:    Override the output filename stem (without extension).
    """
    if df.empty:
        return

    std_col = f"{metric}_std"
    pct_col = "pretrained_pct"

    def _line_key(method: str, init: str) -> str:
        if method in ("SVM_IJEPA", "SVM_FLIM", "SVM_Distillation_Conv",
                      "SVM_Distill_Proj1280", "SVM_Distill_3x3BN", "SVM_Distill_3x3BN_flim",
                      "SVM_Distill_1x1BN", "SVM_Distill_1x1BN_flim", "SVM_Distill_1x1BN_nonorm",
                      "SVM_Distill_2l400K", "SVM_Distill_2l400K_flim",
                      "SVM_Distill_2l400K_flim_nonorm",
                      "SVM_Distill_1x1BN_flim_frozen_eval_knn",
                      "SVM_Distill_1x1BN_flim_frozen_eval_loss",
                      "SVM_FLIMResidual_1_3", "SVM_FLIMResidual_2_3"):
            return method
        return f"SVM_LeJEPA_{init}"

    line_df = df.copy()
    line_df["_line_key"] = [
        _line_key(m, i) for m, i in zip(line_df["method"], line_df["init"])
    ]

    line_keys = [k for k in METHOD_COMPARE_STYLE if k in line_df["_line_key"].unique()]
    if not line_keys:
        return

    sns.set_theme(
        style="whitegrid",
        font_scale=2.5,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, ax = plt.subplots(figsize=(13, 10))

    if show_title:
        dataset_label = DATASET_LABEL.get(dataset, dataset)
        ax.set_title(
            f"{dataset_label} — SVM Method Comparison",
            fontsize=32, fontweight="bold", pad=14,
        )

    legend_handles = []

    for lkey in line_keys:
        sub = line_df[line_df["_line_key"] == lkey].sort_values(pct_col)
        if sub.empty:
            continue

        pcts = sub[pct_col].values
        valid_pcts = [p for p in pcts if p in PCT_TO_POS]
        if not valid_pcts:
            continue

        x = [PCT_TO_POS[p] for p in valid_pcts]
        y = sub.set_index(pct_col).loc[valid_pcts, metric].values
        yerr = (
            sub.set_index(pct_col).loc[valid_pcts, std_col].values
            if std_col in sub.columns else np.zeros(len(valid_pcts))
        )

        sty = METHOD_COMPARE_STYLE[lkey]
        line, = ax.plot(
            x, y,
            color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
            linewidth=3.5, markersize=14,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=sty["zorder"],
            label=METHOD_COMPARE_LABEL.get(lkey, lkey),
        )
        ax.errorbar(
            x, y, yerr=np.nan_to_num(yerr),
            fmt="none", ecolor=sty["color"],
            elinewidth=2.5, capsize=8, capthick=2.0,
            alpha=0.9, zorder=sty["zorder"] + 1,
        )
        legend_handles.append(line)

    pcts_present = sorted({p for p in df[pct_col].unique() if p in PCT_TO_POS})
    tick_pos = [PCT_TO_POS[p] for p in pcts_present]

    if show_xlabel:
        ax.set_xlabel("Pre-training Data", fontsize=28)
    else:
        ax.set_xlabel("")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{p}%" for p in pcts_present], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=ylabel_fontsize)

    n_handles = len(legend_handles)
    ncol = (n_handles if legend_ncol == -1 else legend_ncol) or min(n_handles, 3)

    fig.legend(
        handles=legend_handles,
        title=None,
        loc="lower center",
        ncol=ncol,
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        borderpad=0.5,
        handlelength=2.5,
        handletextpad=0.6,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout()
    legend_rows = max(1, -(-n_handles // ncol))
    bottom_pad  = 0.10 + legend_rows * 0.08
    fig.subplots_adjust(bottom=min(bottom_pad, 0.42))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = stem_override or f"{dataset}_{metric}_comparison"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [PLOT] {out_dir / f'{stem}.png'}")


# ── Merged comparison: all datasets in one figure ─────────────────────────────

def plot_merge_comparison(
    df: pd.DataFrame,
    datasets: list,
    metrics: list,
    out_dir: Path,
    *,
    show_xlabel:           bool  = False,
    subtitle_fontsize:     int   = 30,
    tick_fontsize:         int   = 28,
    ytick_fontsize:        int   = 24,
    ylabel_fontsize:       int   = 28,
    legend_fontsize:       int   = 24,
    legend_ncol:           int   = -1,
    legend_y:              float = 0.0,
    linewidth:             float = 3.5,
    markersize:            int   = 14,
    fig_height:            int   = 10,
    fig_width_per_dataset: int   = 13,
) -> None:
    """One subplot-figure per metric, with one panel per dataset.

    Produces ``len(metrics)`` files named ``{metric}.png`` (e.g. ``kappa.png``),
    each containing ``len(datasets)`` side-by-side subplots.  The dataset name
    is the title above each subplot.  A single horizontal legend (no title) is
    shared below all panels.

    Args:
        df:              Unified filtered DataFrame.
        datasets:        Datasets → one subplot column each.
        metrics:         Metrics → one output file each.
        out_dir:         Output directory (``merge_plots/``).
        show_xlabel:           Show "Pre-training Data" xlabel (default False).
        subtitle_fontsize:     Font size for dataset name above each subplot.
        tick_fontsize:         Font size for x-axis percentage labels.
        ytick_fontsize:        Font size for y-axis tick numbers.
        ylabel_fontsize:       Font size for y-axis label (metric name).
        legend_fontsize:       Font size for legend item text.
        legend_ncol:           Legend columns. -1 = all items in one row.
        legend_y:              Vertical anchor of legend in figure coords.
                               0.0 = figure bottom; negative moves it further down.
        linewidth:             Line thickness.
        markersize:            Marker size.
        fig_height:            Figure height in inches.
        fig_width_per_dataset: Figure width per dataset column in inches.
    """
    if df.empty:
        return

    pct_col = "pretrained_pct"

    def _line_key(method: str, init: str) -> str:
        if method in ("SVM_IJEPA", "SVM_FLIM", "SVM_Distillation_Conv",
                      "SVM_Distill_Proj1280", "SVM_Distill_3x3BN", "SVM_Distill_3x3BN_flim",
                      "SVM_Distill_1x1BN", "SVM_Distill_1x1BN_flim", "SVM_Distill_1x1BN_nonorm",
                      "SVM_Distill_2l400K", "SVM_Distill_2l400K_flim",
                      "SVM_Distill_2l400K_flim_nonorm",
                      "SVM_Distill_1x1BN_flim_frozen_eval_knn",
                      "SVM_Distill_1x1BN_flim_frozen_eval_loss",
                      "SVM_FLIMResidual_1_3", "SVM_FLIMResidual_2_3"):
            return method
        return f"SVM_LeJEPA_{init}"

    df = df.copy()
    df["_line_key"] = [_line_key(m, i) for m, i in zip(df["method"], df["init"])]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        std_col = f"{metric}_std"

        sns.set_theme(
            style="whitegrid", font_scale=2.4,
            rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
        )

        fig, axes = plt.subplots(
            1, len(datasets),
            figsize=(fig_width_per_dataset * len(datasets), fig_height),
            sharey=True, squeeze=False,
        )

        legend_handles: list = []
        legend_labels:  list = []

        for col, dataset in enumerate(datasets):
            ax    = axes[0][col]
            ds_df = df[df["dataset_short"] == dataset]

            # Dataset name as subplot title
            ax.set_title(DATASET_LABEL.get(dataset, dataset),
                         fontsize=subtitle_fontsize, fontweight="bold", pad=10)

            line_keys = [k for k in METHOD_COMPARE_STYLE
                         if k in ds_df["_line_key"].unique()]

            for lkey in line_keys:
                sub = ds_df[ds_df["_line_key"] == lkey].sort_values(pct_col)
                if sub.empty:
                    continue
                valid_pcts = [p for p in sub[pct_col].values if p in PCT_TO_POS]
                if not valid_pcts:
                    continue

                x    = [PCT_TO_POS[p] for p in valid_pcts]
                y    = sub.set_index(pct_col).loc[valid_pcts, metric].values
                yerr = (sub.set_index(pct_col).loc[valid_pcts, std_col].values
                        if std_col in sub.columns else np.zeros(len(valid_pcts)))

                sty   = METHOD_COMPARE_STYLE[lkey]
                label = METHOD_COMPARE_LABEL.get(lkey, lkey)
                line, = ax.plot(
                    x, y,
                    color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
                    linewidth=linewidth, markersize=markersize,
                    markeredgecolor="white", markeredgewidth=0.9,
                    zorder=sty["zorder"], label=label,
                )
                ax.errorbar(
                    x, y, yerr=np.nan_to_num(yerr),
                    fmt="none", ecolor=sty["color"],
                    elinewidth=2.5, capsize=8, capthick=2.0,
                    alpha=0.9, zorder=sty["zorder"] + 1,
                )
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            pcts_present = sorted({p for p in ds_df[pct_col].unique() if p in PCT_TO_POS})
            ax.set_xticks([PCT_TO_POS[p] for p in pcts_present])
            ax.set_xticklabels([f"{p}%" for p in pcts_present], fontsize=tick_fontsize)
            ax.tick_params(axis="y", labelsize=ytick_fontsize)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)

            if show_xlabel:
                ax.set_xlabel("Pre-training Data", fontsize=ylabel_fontsize)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric),
                              fontsize=ylabel_fontsize)

        n_handles = len(legend_handles)
        ncol = (n_handles if legend_ncol == -1 else legend_ncol) or n_handles

        # Place the legend below the figure boundary (y < 0 in figure coords).
        # loc="upper center" means the TOP of the legend box is at the anchor
        # point, so the legend grows downward — entirely outside the axes area.
        # bbox_inches="tight" on savefig then expands the canvas to capture it.
        fig.legend(
            handles=legend_handles, labels=legend_labels,
            title=None,
            loc="upper center",
            ncol=ncol,
            fontsize=legend_fontsize,
            frameon=True, fancybox=True, edgecolor="#CCCCCC",
            borderpad=0.5, handlelength=2.5, handletextpad=0.6,
            bbox_to_anchor=(0.5, legend_y),
        )

        plt.tight_layout()

        dst = out_dir / f"{metric}.png"
        fig.savefig(dst, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [MERGE] {dst}")
