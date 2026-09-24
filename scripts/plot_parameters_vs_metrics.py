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
"""Plot parameters vs metrics for all experiments.

Output layout:
  artifacts/parameters_metrics/pct_{N}/{metric}_{dataset}.png

X-axis: number of model parameters (log scale).
Y-axis: metric value averaged over 3 splits.

Models:
  FLIM            59,504 params  — svm_flim_aggregated.csv
  LeJEPA          59,504 params  — unified_svm_comparison.csv (trunc_normal; mesmo backbone)
  Distill 1x1    123,504 params  — svm_proj1280_1x1_BN2d_results.csv
  Distill 3x3    615,024 params  — svm_proj1280_3x3_BN2d_results.csv
  Distill Conv   889,200 params  — svm_distill_proj1280_results.csv
  I-JEPA     630,762,240 params  — ijepa_svm_aggregated.csv

Legend groups: FLIM | LeJEPA | Distill (param count annotated per point) | I-JEPA
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from adjustText import adjust_text

from constants import (
    ARTIFACTS_NORMALIZED_DIR as NORM,
    DATASETS,
    PERCENTAGES,
    PROJECT_ROOT as ROOT,
    RESULTS_DIR as RESULTS,
    UNIFIED_SVM_COMPARISON_CSV,
)

OUT = os.path.join(ROOT, "artifacts", "parameters_metrics")

# ── Model metadata ────────────────────────────────────────────────────────────
# (model_key, params, legend_label, color)
MODELS: list[tuple[str, int, str, str]] = [
    ("FLIM",         59_504,      "FLIM",          "#E63946"),
    ("LeJEPA",       59_504,      "LeJEPA",         "#F4A261"),
    ("Distill 1x1",  123_504,     "Distillation",  "#2A9D8F"),
    ("Distill 3x3",  615_024,     "Distillation",  "#2A9D8F"),
    ("Distill Conv", 889_200,     "Distillation",  "#2A9D8F"),
    ("I-JEPA",       630_762_240, "I-JEPA",         "#6A0572"),
]

DATASET_LABELS = {
    "eggs":      "Helminth Eggs",
    "larvae":    "Helminth Larvae",
    "protozoan": "Protozoan Cysts",
}
# Ordem propositalmente diferente da canonica de constants.METRICS
# (["kappa","acc","f1"]): aqui ela define a ordem em que as figuras saem.
METRICS = ["f1", "kappa", "acc"]
METRIC_LABELS = {
    "f1":    "F1 Score",
    "kappa": "Cohen's Kappa",
    "acc":   "Accuracy",
}

ModelData = dict[tuple[str, int], dict[str, float]]


# ── Data loaders ──────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_flim() -> ModelData:
    rows = _read_csv(os.path.join(NORM, "svm_flim_aggregated.csv"))
    return {
        (r["dataset_short"], int(float(r["pretrained_pct"]))): {
            "kappa": float(r["kappa"]),
            "acc":   float(r["acc"]),
            "f1":    float(r["f1"]),
        }
        for r in rows
        if r["method"] == "SVM_FLIM" and r["init"] == "flim"
    }


def load_lejepa() -> ModelData:
    rows = _read_csv(UNIFIED_SVM_COMPARISON_CSV)
    return {
        (r["dataset_short"], int(float(r["pretrained_pct"]))): {
            "kappa": float(r["kappa"]),
            "acc":   float(r["acc"]),
            "f1":    float(r["f1"]),
        }
        for r in rows
        if r["method"] == "SVM_lejepa_view" and r["init"] == "trunc_normal"
    }


def load_ijepa() -> ModelData:
    rows = _read_csv(os.path.join(RESULTS, "ijepa_svm_aggregated.csv"))
    return {
        (r["dataset_short"], int(float(r["pretrained_pct"]))): {
            "kappa": float(r["kappa"]),
            "acc":   float(r["acc"]),
            "f1":    float(r["f1"]),
        }
        for r in rows
    }


def _aggregate_raw_distill(path: str) -> ModelData:
    """Average raw distillation CSV over splits, grouping by (dataset, percentage)."""
    rows = _read_csv(path)
    groups: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        if r.get("status", "ok") != "ok":
            continue
        key = (r["dataset"], int(float(r["percentage"])))
        groups[key].append({
            "kappa": float(r["kappa"]),
            "acc":   float(r["acc"]),
            "f1":    float(r["f1"]),
        })
    return {
        key: {
            metric: float(np.mean([v[metric] for v in vals]))
            for metric in ("kappa", "acc", "f1")
        }
        for key, vals in groups.items()
        if vals
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _param_fmt(x: float, _pos=None) -> str:
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return str(int(x))


# ── Main plotting routine ─────────────────────────────────────────────────────

def make_plots() -> None:
    model_data: dict[str, ModelData] = {
        "FLIM":         load_flim(),
        "LeJEPA":       load_lejepa(),
        "Distill 1x1":  _aggregate_raw_distill(
            os.path.join(RESULTS, "svm_proj1280_1x1_BN2d_results.csv")
        ),
        "Distill 3x3":  _aggregate_raw_distill(
            os.path.join(RESULTS, "svm_proj1280_3x3_BN2d_results.csv")
        ),
        "Distill Conv": _aggregate_raw_distill(
            os.path.join(RESULTS, "svm_distill_proj1280_results.csv")
        ),
        "I-JEPA":       load_ijepa(),
    }

    os.makedirs(OUT, exist_ok=True)

    n_saved = 0
    for pct in PERCENTAGES:
        pct_dir = os.path.join(OUT, f"pct_{pct}")
        os.makedirs(pct_dir, exist_ok=True)

        for metric in METRICS:
            for ds in DATASETS:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.set_title(
                    f"{DATASET_LABELS[ds]}  |  {pct}% training data",
                    fontsize=12, fontweight="bold",
                )
                ax.set_xlabel("Parameters", fontsize=11)
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_param_fmt))
                ax.tick_params(axis="x", labelrotation=30)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.set_ylim(-0.05, 1.05)

                legend_seen: set[str] = set()
                texts = []
                pts_x, pts_y = [], []

                for model_key, params, legend_label, color in MODELS:
                    data = model_data[model_key]
                    key = (ds, pct)
                    if key not in data:
                        continue
                    val = data[key][metric]

                    # Jitter LeJEPA to avoid overlap with FLIM (same param count)
                    x = params * (1.10 if model_key == "LeJEPA" else 1.0)
                    pts_x.append(x)
                    pts_y.append(val)

                    # Add to legend only once per label
                    scatter_label = legend_label if legend_label not in legend_seen else "_nolegend_"
                    legend_seen.add(legend_label)

                    ax.scatter(
                        x, val,
                        color=color,
                        marker="o",
                        s=160,
                        label=scatter_label,
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.8,
                    )

                    texts.append(ax.text(
                        x, val, _param_fmt(params),
                        ha="center", va="bottom",
                        fontsize=7.5, color="#1a1a1a",
                        zorder=6,
                    ))

                # Only push labels away from each other vertically — no arrows
                adjust_text(
                    texts,
                    x=pts_x, y=pts_y,
                    ax=ax,
                    only_move={"text": "y", "points": "y"},
                    expand=(1.0, 1.5),
                    force_text=(0, 0.5),
                    arrowprops=None,
                )

                ax.legend(fontsize=9, frameon=True, loc="best")
                plt.tight_layout()
                save_path = os.path.join(pct_dir, f"{metric}_{ds}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {os.path.relpath(save_path, ROOT)}")
                n_saved += 1

    print(f"\nDone — {n_saved} figures saved to {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    make_plots()
