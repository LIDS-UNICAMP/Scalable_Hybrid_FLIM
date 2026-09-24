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

"""plot_classical_classifiers.py — Adaptive learning curves for classical classifiers.

Reads results/classical_classifiers_results.csv and plots kappa vs. pre-training
percentage for each (dataset, init) combination.  Missing (dataset, pct, split)
combinations are filled with NaN — the plot works with partial data and updates
automatically each time more results arrive.

Three plot types:
  1. classifiers_per_init/   — X=pct, one line per classifier (kNN k5/10/15, RF,
                                LGBM, GP, SVM-RBF) + SVM-linear baseline.
                                One file per (dataset, init, metric).
  2. inits_per_classifier/   — X=pct, one line per init.
                                One file per (dataset, classifier, metric).
  3. plots_compare_to_flim/merge_plots/
       classifiers_per_init/ — Grid: one subplot per init, all datasets merged.
                                One file per (metric, dataset).
       inits_per_classifier/ — Grid: one subplot per classifier, all datasets merged.
                                One file per (metric, dataset).

Outputs saved to: artifacts/plots/classical_classifiers/

Usage:
    python scripts/plot_classical_classifiers.py                  # all modes
    python scripts/plot_classical_classifiers.py --mode merge     # only merged grids
    python scripts/plot_classical_classifiers.py --mode classifiers_per_init
    python scripts/plot_classical_classifiers.py --mode inits_per_classifier
    python scripts/plot_classical_classifiers.py --metrics kappa f1
    python scripts/plot_classical_classifiers.py --datasets eggs larvae
    python scripts/plot_classical_classifiers.py --out /custom/path
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# scripts/ é o sys.path[0] quando se roda `python scripts/plot_classical_classifiers.py`;
# tem de vir antes do bloco abaixo, que usa PROJECT_ROOT.
from constants import (  # noqa: E402
    ARTIFACTS_PLOTS_DIR,
    DATASETS as ALL_DATASETS,
    DATASET_LONG_TO_SHORT as DATASET_NAME_MAP,
    METRICS,
    PERCENTAGES as ALL_PCTS,
    PROJECT_ROOT as _ROOT_STR,
    RESULTS_DIR,
    SPLITS as ALL_SPLITS,
)

if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Constants ──────────────────────────────────────────────────────────────────

_ROOT = Path(_ROOT_STR)

# A ordem vira ordem de legenda e diverge da de outros scripts: fica local.
ALL_INITS     = ["flim", "he", "random", "xavier", "trunc_normal"]

DATASET_LABEL = {
    "eggs":      "Helminth Eggs",
    "larvae":    "Helminth Larvae",
    "protozoan": "Protozoan Cysts",
}

METRIC_LABEL = {
    "kappa": "Cohen's Kappa (κ)",
    "acc":   "Accuracy",
    "f1":    "F1-score",
}

PCT_TO_POS = {p: i for i, p in enumerate(ALL_PCTS)}

# ── Visual style — one colour+style per classifier ─────────────────────────────

CLASSIFIER_STYLE: dict[str, dict] = {
    "svm_linear": {"color": "#0072B2", "ls": "-",   "marker": "o",  "zorder": 10, "label": "SVM Linear (baseline)"},
    "svm_rbf":    {"color": "#E69F00", "ls": "-.",  "marker": "D",  "zorder": 9,  "label": "SVM RBF"},
    "knn_k5":     {"color": "#009E73", "ls": "-",   "marker": "s",  "zorder": 8,  "label": "kNN k=5"},
    "knn_k10":    {"color": "#56B4E9", "ls": "--",  "marker": "s",  "zorder": 7,  "label": "kNN k=10"},
    "knn_k15":    {"color": "#0096FF", "ls": ":",   "marker": "s",  "zorder": 6,  "label": "kNN k=15"},
    "rf":         {"color": "#CC79A7", "ls": "--",  "marker": "^",  "zorder": 5,  "label": "Random Forest"},
    "lgbm":       {"color": "#D55E00", "ls": "--",  "marker": "v",  "zorder": 4,  "label": "LightGBM"},
    "gp":         {"color": "#F0E442", "ls": "-.",  "marker": "P",  "zorder": 3,  "label": "Gaussian Process"},
    "qda":        {"color": "#999999", "ls": ":",   "marker": "X",  "zorder": 2,  "label": "QDA"},
}

INIT_LINE_STYLE: dict[str, dict] = {
    "flim":         {"color": "#0072B2", "ls": "-",   "marker": "o", "zorder": 5},
    "random":       {"color": "#009E73", "ls": "-",   "marker": "s", "zorder": 4},
    "he":           {"color": "#D55E00", "ls": "--",  "marker": "^", "zorder": 3},
    "xavier":       {"color": "#CC79A7", "ls": "--",  "marker": "D", "zorder": 2},
    "trunc_normal": {"color": "#F0E442", "ls": "-.",  "marker": "P", "zorder": 6},
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _classifier_key(row: pd.Series) -> str:
    """Collapse classifier + variant into a single key."""
    clf = str(row["classifier"])
    var = str(row.get("classifier_variant", ""))
    if clf == "knn" and var in ("knn_k5", "knn_k10", "knn_k15"):
        return var
    return clf


def load_classical(csv_path: Path) -> pd.DataFrame:
    """Load and aggregate classical classifiers results.

    Returns a DataFrame with columns:
        dataset_short, init, pct, clf_key,
        kappa, kappa_std, acc, acc_std, f1, f1_std, n_splits
    Rows with status != 'ok' are dropped before aggregation.
    Missing (dataset, pct, init, clf) combinations are added with NaN.
    """
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[WARN] No ok rows in classical_classifiers_results.csv")
        return pd.DataFrame()

    # Normalise dataset name
    df["dataset_short"] = df["dataset_name"].map(DATASET_NAME_MAP).fillna(df["dataset_name"])
    df["pct"] = df["percentage"].astype(int)
    df["clf_key"] = df.apply(_classifier_key, axis=1)

    # Aggregate across splits
    rows = []
    for (ds, init, pct, clf), grp in df.groupby(
        ["dataset_short", "initialization_type", "pct", "clf_key"], sort=False
    ):
        n = len(grp)
        rows.append({
            "dataset_short": ds,
            "init":          init,
            "pct":           pct,
            "clf_key":       clf,
            "kappa":         grp["kappa"].mean(),
            "kappa_std":     grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":           grp["acc"].mean(),
            "acc_std":       grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":            grp["f1"].mean(),
            "f1_std":        grp["f1"].std(ddof=1) if n > 1 else 0.0,
            "n_splits":      n,
        })

    return pd.DataFrame(rows)


def load_svm_baseline(csv_path: Path) -> pd.DataFrame:
    """Load existing SVM-linear results and aggregate to (dataset, init, pct).

    Returns same schema as load_classical() but with clf_key='svm_linear'.
    """
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found — SVM baseline skipped")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()

    ds_col  = "dataset_short" if "dataset_short" in df.columns else "dataset"
    pct_col = "pretrained_pct" if "pretrained_pct" in df.columns else "percentage"

    df["dataset_short"] = df[ds_col].map(DATASET_NAME_MAP).fillna(df[ds_col])
    df["pct"]           = df[pct_col].astype(int)
    df["clf_key"]       = "svm_linear"

    rows = []
    for (ds, init, pct), grp in df.groupby(["dataset_short", "init", "pct"], sort=False):
        n = len(grp)
        rows.append({
            "dataset_short": ds,
            "init":          init,
            "pct":           pct,
            "clf_key":       "svm_linear",
            "kappa":         grp["kappa"].mean(),
            "kappa_std":     grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":           grp["acc"].mean(),
            "acc_std":       grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":            grp["f1"].mean(),
            "f1_std":        grp["f1"].std(ddof=1) if n > 1 else 0.0,
            "n_splits":      n,
        })

    return pd.DataFrame(rows)


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the metric columns are all NaN (nothing to plot)."""
    if df.empty:
        return df
    metric_cols = [c for c in ["kappa", "acc", "f1"] if c in df.columns]
    return df.dropna(subset=metric_cols, how="all").reset_index(drop=True)


# ── Plot 1: classifiers_per_init ───────────────────────────────────────────────

def plot_classifiers_per_init(
    df: pd.DataFrame,
    dataset: str,
    init: str,
    metric: str,
    out_dir: Path,
    *,
    legend_fontsize: int = 24,
    legend_ncol: int = -1,
    legend_y: float = 0.0,
    tick_fontsize: int = 28,
    ytick_fontsize: int = 24,
    ylabel_fontsize: int = 28,
    subtitle_fontsize: int = 30,
    linewidth: float = 3.5,
    markersize: int = 14,
    fig_width: int = 13,
    fig_height: int = 10,
    show_xlabel: bool = False,
) -> None:
    """X=pct, one line per classifier. NaN → gap in line."""
    sub = df[(df["dataset_short"] == dataset) & (df["init"] == init)].copy()
    if sub.empty:
        return

    clf_keys = [k for k in CLASSIFIER_STYLE if k in sub["clf_key"].unique()]
    if not clf_keys:
        return

    sns.set_theme(
        style="whitegrid", font_scale=2.4,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_title(
        f"{DATASET_LABEL.get(dataset, dataset)} — init={init.upper()}",
        fontsize=subtitle_fontsize, fontweight="bold", pad=14,
    )

    std_col = f"{metric}_std"
    legend_handles = []

    for clf in clf_keys:
        grp = sub[(sub["clf_key"] == clf) & sub[metric].notna()].sort_values("pct")
        if grp.empty:
            continue

        valid_pcts = [p for p in grp["pct"].values if p in PCT_TO_POS]
        if not valid_pcts:
            continue

        x    = [PCT_TO_POS[p] for p in valid_pcts]
        y    = grp.set_index("pct").loc[valid_pcts, metric].values.astype(float)
        yerr = (grp.set_index("pct").loc[valid_pcts, std_col].values.astype(float)
                if std_col in grp.columns else np.zeros(len(valid_pcts)))

        sty = CLASSIFIER_STYLE[clf]
        line, = ax.plot(
            x, y,
            color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
            linewidth=linewidth, markersize=markersize,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=sty["zorder"],
            label=sty["label"],
        )
        ax.errorbar(
            x, y, yerr=np.nan_to_num(yerr),
            fmt="none", ecolor=sty["color"],
            elinewidth=1.8, capsize=6, capthick=1.8,
            alpha=0.7, zorder=sty["zorder"] - 1,
        )
        legend_handles.append(line)

    if show_xlabel:
        ax.set_xlabel("Pre-training Data (%)", fontsize=tick_fontsize)
    ax.set_xticks(list(PCT_TO_POS.values()))
    ax.set_xticklabels([f"{p}%" for p in ALL_PCTS], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=ylabel_fontsize)

    n_handles = len(legend_handles)
    ncol = n_handles if legend_ncol == -1 else legend_ncol
    fig.legend(
        handles=legend_handles,
        title="Classifier",
        title_fontsize=legend_fontsize + 4,
        loc="upper center",
        ncol=max(1, ncol),
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, legend_y),
    )

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{init}_{metric}_classifiers"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {(out_dir / f'{stem}.png').relative_to(_ROOT)}")


# ── Plot 2: inits_per_classifier ───────────────────────────────────────────────

def plot_inits_per_classifier(
    df: pd.DataFrame,
    dataset: str,
    clf_key: str,
    metric: str,
    out_dir: Path,
    *,
    legend_fontsize: int = 24,
    legend_ncol: int = -1,
    legend_y: float = 0.0,
    tick_fontsize: int = 28,
    ytick_fontsize: int = 24,
    ylabel_fontsize: int = 28,
    subtitle_fontsize: int = 30,
    linewidth: float = 3.5,
    markersize: int = 14,
    fig_width: int = 13,
    fig_height: int = 10,
    show_xlabel: bool = False,
) -> None:
    """X=pct, one line per init. Matches existing SVM style."""
    sub = df[(df["dataset_short"] == dataset) & (df["clf_key"] == clf_key)].copy()
    if sub.empty:
        return

    inits = [i for i in ALL_INITS if i in sub["init"].unique()]
    if not inits:
        return

    sns.set_theme(
        style="whitegrid", font_scale=2.4,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    clf_label = CLASSIFIER_STYLE.get(clf_key, {}).get("label", clf_key)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_title(
        f"{DATASET_LABEL.get(dataset, dataset)} — {clf_label}",
        fontsize=subtitle_fontsize, fontweight="bold", pad=14,
    )

    std_col = f"{metric}_std"
    legend_handles = []

    for init in inits:
        grp = sub[(sub["init"] == init) & sub[metric].notna()].sort_values("pct")
        if grp.empty:
            continue

        valid_pcts = [p for p in grp["pct"].values if p in PCT_TO_POS]
        if not valid_pcts:
            continue

        x    = [PCT_TO_POS[p] for p in valid_pcts]
        y    = grp.set_index("pct").loc[valid_pcts, metric].values.astype(float)
        yerr = (grp.set_index("pct").loc[valid_pcts, std_col].values.astype(float)
                if std_col in grp.columns else np.zeros(len(valid_pcts)))

        sty = INIT_LINE_STYLE[init]
        line, = ax.plot(
            x, y,
            color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
            linewidth=linewidth, markersize=markersize,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=sty["zorder"], label=init.upper(),
        )
        ax.errorbar(
            x, y, yerr=np.nan_to_num(yerr),
            fmt="none", ecolor=sty["color"],
            elinewidth=1.5, capsize=6, capthick=1.8,
            alpha=0.7, zorder=sty["zorder"] - 1,
        )
        legend_handles.append(line)

    if show_xlabel:
        ax.set_xlabel("Pre-training Data (%)", fontsize=tick_fontsize)
    ax.set_xticks(list(PCT_TO_POS.values()))
    ax.set_xticklabels([f"{p}%" for p in ALL_PCTS], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=ylabel_fontsize)

    n_handles = len(legend_handles)
    ncol = n_handles if legend_ncol == -1 else legend_ncol
    fig.legend(
        handles=legend_handles,
        title="Initialization",
        title_fontsize=legend_fontsize + 4,
        loc="upper center",
        ncol=max(1, ncol),
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, legend_y),
    )

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{clf_key}_{metric}_inits"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {(out_dir / f'{stem}.png').relative_to(_ROOT)}")


# ── Plot 3: merge_plots/classifiers_per_init ──────────────────────────────────

def plot_merge_classifiers_per_init(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    out_dir: Path,
    inits: list[str],
    *,
    legend_fontsize: int = 24,
    legend_ncol: int = -1,
    legend_y: float = 0.0,
    tick_fontsize: int = 28,
    ytick_fontsize: int = 24,
    ylabel_fontsize: int = 28,
    subtitle_fontsize: int = 30,
    linewidth: float = 3.5,
    markersize: int = 14,
    fig_width: int = 13,
    fig_height: int = 10,
    show_xlabel: bool = False,
) -> None:
    """Grid of subplots (one per init), X=pct, lines=classifiers.

    Output: {out_dir}/{metric}_{dataset}.png
    """
    inits_present = [i for i in inits if not df[
        (df["dataset_short"] == dataset) & (df["init"] == i)
    ].empty]
    if not inits_present:
        return

    n = len(inits_present)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    sns.set_theme(
        style="whitegrid", font_scale=2.2,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width * ncols, fig_height * nrows),
        sharey=True, squeeze=False,
    )
    fig.suptitle(
        f"{DATASET_LABEL.get(dataset, dataset)} — {METRIC_LABEL.get(metric, metric)}"
        f"\n(one subplot per initialisation)",
        fontsize=36, fontweight="bold", y=1.01,
    )

    std_col = f"{metric}_std"
    legend_handles: list = []
    legend_labels:  list = []

    for idx, init in enumerate(inits_present):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_title(f"init = {init.upper()}", fontsize=subtitle_fontsize, fontweight="bold", pad=10)

        sub = df[(df["dataset_short"] == dataset) & (df["init"] == init)]
        clf_keys = [k for k in CLASSIFIER_STYLE if k in sub["clf_key"].unique()]

        for clf in clf_keys:
            grp = sub[(sub["clf_key"] == clf) & sub[metric].notna()].sort_values("pct")
            if grp.empty:
                continue
            valid_pcts = [p for p in grp["pct"].values if p in PCT_TO_POS]
            if not valid_pcts:
                continue
            x    = [PCT_TO_POS[p] for p in valid_pcts]
            y    = grp.set_index("pct").loc[valid_pcts, metric].values.astype(float)
            yerr = (grp.set_index("pct").loc[valid_pcts, std_col].values.astype(float)
                    if std_col in grp.columns else np.zeros(len(valid_pcts)))
            sty  = CLASSIFIER_STYLE[clf]
            line, = ax.plot(
                x, y,
                color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
                linewidth=linewidth, markersize=markersize,
                markeredgecolor="white", markeredgewidth=0.9,
                zorder=sty["zorder"], label=sty["label"],
            )
            ax.errorbar(
                x, y, yerr=np.nan_to_num(yerr),
                fmt="none", ecolor=sty["color"],
                elinewidth=1.8, capsize=6, capthick=1.8,
                alpha=0.7, zorder=sty["zorder"] - 1,
            )
            if sty["label"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(sty["label"])

        ax.set_xticks(list(PCT_TO_POS.values()))
        ax.set_xticklabels([f"{p}%" for p in ALL_PCTS], fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
        if col == 0:
            ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=ylabel_fontsize)
        if show_xlabel:
            ax.set_xlabel("Pre-training Data (%)", fontsize=tick_fontsize)

    # Hide unused subplots
    for idx in range(len(inits_present), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    n_handles = len(legend_handles)
    ncol_leg = n_handles if legend_ncol == -1 else legend_ncol
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        title="Classifier", title_fontsize=legend_fontsize + 4,
        loc="upper center", ncol=max(1, ncol_leg),
        fontsize=legend_fontsize, frameon=True, fancybox=True, edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, legend_y),
    )

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric}_{dataset}"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MERGE] {(out_dir / f'{stem}.png').relative_to(_ROOT)}")


# ── Plot 4: merge_plots/inits_per_classifier ───────────────────────────────────

def plot_merge_inits_per_classifier(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    out_dir: Path,
    clf_keys: list[str],
    inits: list[str],
    *,
    legend_fontsize: int = 24,
    legend_ncol: int = -1,
    legend_y: float = 0.0,
    tick_fontsize: int = 28,
    ytick_fontsize: int = 24,
    ylabel_fontsize: int = 28,
    subtitle_fontsize: int = 30,
    linewidth: float = 3.5,
    markersize: int = 14,
    fig_width: int = 13,
    fig_height: int = 10,
    show_xlabel: bool = False,
) -> None:
    """Grid of subplots (one per classifier), X=pct, lines=inits.

    Output: {out_dir}/{metric}_{dataset}.png
    """
    clfs_present = [c for c in clf_keys if not df[
        (df["dataset_short"] == dataset) & (df["clf_key"] == c)
    ].empty]
    if not clfs_present:
        return

    n = len(clfs_present)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    sns.set_theme(
        style="whitegrid", font_scale=2.2,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width * ncols, fig_height * nrows),
        sharey=True, squeeze=False,
    )
    fig.suptitle(
        f"{DATASET_LABEL.get(dataset, dataset)} — {METRIC_LABEL.get(metric, metric)}"
        f"\n(one subplot per classifier)",
        fontsize=36, fontweight="bold", y=1.01,
    )

    std_col = f"{metric}_std"
    legend_handles: list = []
    legend_labels:  list = []

    for idx, clf in enumerate(clfs_present):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        clf_label = CLASSIFIER_STYLE.get(clf, {}).get("label", clf)
        ax.set_title(clf_label, fontsize=subtitle_fontsize, fontweight="bold", pad=10)

        sub = df[(df["dataset_short"] == dataset) & (df["clf_key"] == clf)]
        inits_here = [i for i in inits if i in sub["init"].unique()]

        for init in inits_here:
            grp = sub[(sub["init"] == init) & sub[metric].notna()].sort_values("pct")
            if grp.empty:
                continue
            valid_pcts = [p for p in grp["pct"].values if p in PCT_TO_POS]
            if not valid_pcts:
                continue
            x    = [PCT_TO_POS[p] for p in valid_pcts]
            y    = grp.set_index("pct").loc[valid_pcts, metric].values.astype(float)
            yerr = (grp.set_index("pct").loc[valid_pcts, std_col].values.astype(float)
                    if std_col in grp.columns else np.zeros(len(valid_pcts)))
            sty  = INIT_LINE_STYLE[init]
            line, = ax.plot(
                x, y,
                color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
                linewidth=linewidth, markersize=markersize,
                markeredgecolor="white", markeredgewidth=0.9,
                zorder=sty["zorder"], label=init.upper(),
            )
            ax.errorbar(
                x, y, yerr=np.nan_to_num(yerr),
                fmt="none", ecolor=sty["color"],
                elinewidth=1.8, capsize=6, capthick=1.8,
                alpha=0.7, zorder=sty["zorder"] - 1,
            )
            if init.upper() not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(init.upper())

        ax.set_xticks(list(PCT_TO_POS.values()))
        ax.set_xticklabels([f"{p}%" for p in ALL_PCTS], fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)
        if col == 0:
            ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=ylabel_fontsize)
        if show_xlabel:
            ax.set_xlabel("Pre-training Data (%)", fontsize=tick_fontsize)

    for idx in range(len(clfs_present), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    n_handles = len(legend_handles)
    ncol_leg = n_handles if legend_ncol == -1 else legend_ncol
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        title="Initialization", title_fontsize=legend_fontsize + 4,
        loc="upper center", ncol=max(1, ncol_leg),
        fontsize=legend_fontsize, frameon=True, fancybox=True, edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, legend_y),
    )

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric}_{dataset}"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MERGE] {(out_dir / f'{stem}.png').relative_to(_ROOT)}")


# ── Argparse ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot classical classifiers evaluation curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "classifiers_per_init", "inits_per_classifier", "merge"],
        default="all",
        help="Which plot type(s) to generate (default: all).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        metavar="METRIC",
        help=f"Metrics to plot. Choices: {METRICS}. Default: all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DS",
        help="Datasets to include (default: all present in data).",
    )
    parser.add_argument(
        "--classical-csv",
        type=Path,
        default=Path(RESULTS_DIR) / "classical_classifiers_results.csv",
        metavar="PATH",
        help="Path to classical_classifiers_results.csv.",
    )
    parser.add_argument(
        "--svm-csv",
        type=Path,
        default=_ROOT / "artifacts" / "SVM" / "svm_results.csv",
        metavar="PATH",
        help="Path to SVM baseline results CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(ARTIFACTS_PLOTS_DIR) / "classical_classifiers",
        metavar="DIR",
        help="Base output directory for all plots.",
    )
    # ── Plot style controls ───────────────────────────────────────────────────
    style = parser.add_argument_group("plot style")
    style.add_argument(
        "--subtitle-fontsize", type=int, default=30, metavar="N",
        help="Título do dataset acima de cada subplot (default: 30).",
    )
    style.add_argument(
        "--tick-fontsize", type=int, default=28, metavar="N",
        help="Porcentagens no eixo X (default: 28).",
    )
    style.add_argument(
        "--ytick-fontsize", type=int, default=24, metavar="N",
        help="Números no eixo Y (default: 24).",
    )
    style.add_argument(
        "--ylabel-fontsize", type=int, default=28, metavar="N",
        help="Label do eixo Y (kappa/acc/f1) (default: 28).",
    )
    style.add_argument(
        "--legend-fontsize", type=int, default=24, metavar="N",
        help="Texto dos itens da legenda (default: 24).",
    )
    style.add_argument(
        "--legend-ncol", type=int, default=-1, metavar="N",
        help="Colunas da legenda. -1 = tudo em uma linha (default: -1).",
    )
    style.add_argument(
        "--legend-y", type=float, default=0.0, metavar="F",
        help="Posição vertical da legenda em coordenadas da figura. "
             "0.0 = borda inferior, negativo = mais para baixo (default: 0.0).",
    )
    style.add_argument(
        "--linewidth", type=float, default=3.5, metavar="F",
        help="Espessura das linhas (default: 3.5).",
    )
    style.add_argument(
        "--markersize", type=int, default=14, metavar="N",
        help="Tamanho dos marcadores (default: 14).",
    )
    style.add_argument(
        "--fig-height", type=int, default=10, metavar="N",
        help="Altura da figura em polegadas (default: 10).",
    )
    style.add_argument(
        "--fig-width-per-dataset", type=int, default=13, metavar="N",
        help="Largura por dataset em polegadas (default: 13).",
    )
    style.add_argument(
        "--show-xlabel", action="store_true", default=False,
        help="Mostrar label 'Pre-training Data' no eixo X (default: oculto).",
    )
    return parser


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    classical_csv = args.classical_csv
    svm_csv       = args.svm_csv
    out_base      = args.out
    mode          = args.mode
    metrics       = args.metrics

    style_kwargs = dict(
        legend_fontsize=args.legend_fontsize,
        legend_ncol=args.legend_ncol,
        legend_y=args.legend_y,
        tick_fontsize=args.tick_fontsize,
        ytick_fontsize=args.ytick_fontsize,
        ylabel_fontsize=args.ylabel_fontsize,
        subtitle_fontsize=args.subtitle_fontsize,
        linewidth=args.linewidth,
        markersize=args.markersize,
        fig_width=args.fig_width_per_dataset,
        fig_height=args.fig_height,
        show_xlabel=args.show_xlabel,
    )

    print(f"\n{'=' * 70}")
    print("[LOAD] Reading classical classifiers results...")
    print(f"{'=' * 70}")

    classical = load_classical(classical_csv)
    svm_base  = load_svm_baseline(svm_csv)

    if classical.empty and svm_base.empty:
        print("[ERROR] No data available. Run the evaluation first.")
        return

    combined = pd.concat([classical, svm_base], ignore_index=True)
    full     = drop_missing(combined)

    if full.empty:
        print("[ERROR] No valid rows after filtering. Nothing to plot.")
        return

    clf_keys_present = sorted(full["clf_key"].dropna().unique())
    datasets_present = sorted(full["dataset_short"].dropna().unique())

    if args.datasets:
        datasets_present = [d for d in args.datasets if d in datasets_present]
        if not datasets_present:
            print(f"[ERROR] None of the requested datasets found in data.")
            return

    # Coverage report
    print(f"\n[INFO] Mode:     {mode}")
    print(f"[INFO] Metrics:  {metrics}")
    print(f"[INFO] Datasets: {datasets_present}")
    print(f"\n[INFO] Data coverage (kappa rows):")
    for ds in datasets_present:
        n_ok = len(full[(full["dataset_short"] == ds) & full["kappa"].notna()])
        n_total = len(ALL_INITS) * len(ALL_PCTS) * len(clf_keys_present)
        print(f"  {ds:12s}: {n_ok:4d} / {n_total} expected ({100*n_ok/n_total:.0f}%)")
    print(f"\n[INFO] Classifiers present: {clf_keys_present}")

    # ── Plot 1: classifiers per init ──────────────────────────────────────────
    if mode in ("all", "classifiers_per_init"):
        print(f"\n{'=' * 70}")
        print("[PLOTS] classifiers_per_init/")
        print(f"{'=' * 70}")
        out1 = out_base / "classifiers_per_init"
        for ds in datasets_present:
            for init in ALL_INITS:
                for metric in metrics:
                    plot_classifiers_per_init(full, ds, init, metric, out1, **style_kwargs)

    # ── Plot 2: inits per classifier ──────────────────────────────────────────
    if mode in ("all", "inits_per_classifier"):
        print(f"\n{'=' * 70}")
        print("[PLOTS] inits_per_classifier/")
        print(f"{'=' * 70}")
        out2 = out_base / "inits_per_classifier"
        for ds in datasets_present:
            for clf in clf_keys_present:
                for metric in metrics:
                    plot_inits_per_classifier(full, ds, clf, metric, out2, **style_kwargs)

    # ── Plot 3 & 4: merge_plots ───────────────────────────────────────────────
    if mode in ("all", "merge"):
        print(f"\n{'=' * 70}")
        print("[PLOTS] plots_compare_to_flim/merge_plots/")
        print(f"{'=' * 70}")
        merge_base = out_base / "plots_compare_to_flim" / "merge_plots"

        out_m1 = merge_base / "classifiers_per_init"
        print(f"\n  -> merge_plots/classifiers_per_init/")
        for ds in datasets_present:
            for metric in metrics:
                plot_merge_classifiers_per_init(
                    full, ds, metric, out_m1, ALL_INITS, **style_kwargs,
                )

        out_m2 = merge_base / "inits_per_classifier"
        print(f"\n  -> merge_plots/inits_per_classifier/")
        for ds in datasets_present:
            for metric in metrics:
                plot_merge_inits_per_classifier(
                    full, ds, metric, out_m2, clf_keys_present, ALL_INITS, **style_kwargs,
                )

    print(f"\n{'=' * 70}")
    print(f"[DONE] Plots saved to: {out_base.relative_to(_ROOT)}/")
    print(f"       Re-run after new data arrives — NaN cells auto-fill.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
