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

"""plot_svm_results.py
--------------------
Reads svm_results.csv and plots Cohen's Kappa vs. training percentage
for each initialization type (flim / random / he / xavier), averaged
across splits, with ±std error bars.

One panel per dataset present in the CSV.

Usage:
    python -m analysis.plots.plot_svm_results
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _ROOT / "svm_results.csv"
OUT_DIR = _ROOT / "plots"

# ── Visual config ─────────────────────────────────────────────────────────────

PERCENTAGES = [1, 5, 25, 50, 75, 100]
TICK_POS = list(range(len(PERCENTAGES)))
PCT_TO_POS = {p: i for i, p in enumerate(PERCENTAGES)}

INIT_STYLE: dict[str, dict] = {
    "flim":   {"color": "#0072B2", "ls": "-",  "marker": "o", "zorder": 5},
    "random": {"color": "#009E73", "ls": "-",  "marker": "s", "zorder": 4},
    "he":     {"color": "#D55E00", "ls": "--", "marker": "^", "zorder": 3},
    "xavier": {"color": "#CC79A7", "ls": "--", "marker": "D", "zorder": 2},
}

DATASET_LABEL = {
    "helminth-eggs":    "Helminth Eggs",
    "helminth-larvae":  "Helminth Larvae",
    "protozoan-cysts":  "Protozoan Cysts",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_to_pos(percentages) -> list[int]:
    return [PCT_TO_POS[p] for p in percentages]


def aggregate(df: pd.DataFrame, dataset: str, init: str) -> pd.DataFrame:
    """Mean ± std of kappa across splits, per percentage."""
    sub = df[(df["dataset_name"] == dataset) & (df["initialization_type"] == init)]
    grouped = (
        sub.groupby("percentage")["kappa"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "avg_kappa", "std": "std_kappa"})
        .reset_index()
    )
    grouped["std_kappa"] = grouped["std_kappa"].fillna(0.0)
    return grouped.sort_values("percentage")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "ok"].copy()

    datasets = [d for d in DATASET_LABEL if d in df["dataset_name"].unique()]
    inits = [i for i in INIT_STYLE if i in df["initialization_type"].unique()]

    if not datasets:
        print("[ERROR] No successful rows found in svm_results.csv")
        return

    sns.set_theme(
        style="whitegrid",
        font_scale=2.0,
        rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"},
    )

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(11 * len(datasets), 9),
        sharey=True,
        squeeze=False,
    )

    legend_handles = []

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        ax.set_title(
            DATASET_LABEL.get(dataset, dataset),
            fontsize=28, fontweight="bold", pad=12,
        )

        for init in inits:
            agg = aggregate(df, dataset, init)
            if agg.empty:
                continue

            x = pct_to_pos(agg["percentage"].values)
            y = agg["avg_kappa"].values
            yerr = agg["std_kappa"].values
            sty = INIT_STYLE[init]

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

            if col == 0:
                legend_handles.append(line)

        ax.set_xlabel("Training Data", fontsize=24)
        ax.set_xticks(TICK_POS)
        ax.set_xticklabels([f"{p}%" for p in PERCENTAGES], fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_xlim(-0.35, len(PERCENTAGES) - 0.65)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)

    axes[0][0].set_ylabel("Cohen's Kappa ($\\kappa$)", fontsize=24)

    fig.legend(
        handles=legend_handles,
        title="Initialization",
        title_fontsize=22,
        loc="lower center",
        ncol=len(inits),
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "svm_results_kappa.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
