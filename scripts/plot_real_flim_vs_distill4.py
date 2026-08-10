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
"""plot_real_flim_vs_distill4.py — FLIM real (avg pool) x Distill 4 x old_flim.

Tres fontes, uma figura por metrica (kappa, acc, f1) mais uma figura 3x3 unindo tudo:

  Real FLIM (48-d, avg pool)  artifacts/real_FLIM/real_FLIM_results.csv
      rodado agora por  src/evaluate/svm_real_flim.py  — encoder FLIM raw
      (architecture.json + pesos dos marcadores, sem checkpoint) -> AdaptiveAvgPool2d(1)
      -> SVM do avaliador de destilacao. Duas variantes de pre-processamento:
      imagenet_norm=True (mesmo transform do Distill 4) e LAB[0,1] raw.

  Distill 4 (889K)            results/svm_distill_proj1280_results.csv
      encoder FLIM destilado + proj head 1280-d, mesmo SVM. Gerado por
      ``python -m src.evaluate.svm_distill_with_projection``.

  old_flim                    data/reports_felipe/svm/report_svm_{ds}_split{N}_perc{pct}.csv
      SVM do FLIM supervisionado do Felipe (encoder_mode==frozen), a mesma fonte que
      alimenta ``SVM_FLIM`` em scripts/plot_comparison_flim.py.

Todas as curvas sao media +- desvio sobre os 3 splits, por (dataset, percentage).
Cores de ``old_flim`` e ``Distill 4`` sao as mesmas de ``eval_plotter.METHOD_LINE_STYLE``,
para as figuras baterem com as ja publicadas.

Uso::

    conda run -n scalable_FLIM python scripts/plot_real_flim_vs_distill4.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import (
    DATASETS,
    DATASET_ALIASES as _FELIPE_DATASET,
    METRICS,
    PERCENTAGES as PCTS,
    PROJECT_ROOT,
    REPORTS_FELIPE_SVM_DIR,
    RESULTS_DIR,
)

_ROOT = Path(PROJECT_ROOT)

# Rotulos em sentence case, divergentes de propositio dos de constants
# (Title Case / "Cohen's Kappa"): trocar mudaria os PNGs ja publicados.
METRIC_LABEL = {"kappa": "Cohen's kappa", "acc": "Accuracy", "f1": "F1 (weighted)"}
DATASET_LABEL = {"eggs": "Helminth eggs", "larvae": "Helminth larvae",
                 "protozoan": "Protozoan cysts"}
PCT_TO_POS = {p: i for i, p in enumerate(PCTS)}

_FELIPE_RE = re.compile(r"report_svm_(\w+)_split(\d+)_perc(\d+)\.csv$")

# Cor + linestyle + marker: identidade nunca fica so na cor (CVD / impressao P&B).
# "old_flim" e "Distill 4" herdam a cor de src/evaluate/eval_plotter.METHOD_LINE_STYLE.
SERIES = {
    "old_flim": {
        "label": "old_flim",
        "color": "#0072B2", "ls": "-", "marker": "o", "zorder": 7,
    },
    "Distill4": {
        "label": "Distill 4 (889K)",
        "color": "#D62728", "ls": "--", "marker": "v", "zorder": 10,
    },
    "RealFLIM_norm": {
        "label": "Real FLIM avg-pool (48-d, imagenet_norm)",
        "color": "#009E73", "ls": "-", "marker": "s", "zorder": 9,
    },
    "RealFLIM_lab01": {
        "label": "Real FLIM avg-pool (48-d, LAB[0,1])",
        "color": "#CC79A7", "ls": ":", "marker": "D", "zorder": 8,
    },
}
SERIES_ORDER = ["old_flim", "Distill4", "RealFLIM_norm", "RealFLIM_lab01"]


def _agg(df: pd.DataFrame, series: str) -> pd.DataFrame:
    """Media +- desvio sobre splits, por (dataset, percentage)."""
    g = (df.groupby(["dataset", "percentage"])[METRICS]
           .agg(["mean", "std", "count"]).reset_index())
    g.columns = ["dataset", "percentage"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "std", "count")]
    g["series"] = series
    return g


def load_real_flim(path: Path) -> list[pd.DataFrame]:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"]
    out = []
    for key, norm in (("RealFLIM_norm", True), ("RealFLIM_lab01", False)):
        sub = df[df["imagenet_norm"].astype(str).str.lower() == str(norm).lower()]
        if not sub.empty:
            out.append(_agg(sub, key))
    return out


def load_distill4(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _agg(df[df["status"] == "ok"], "Distill4")


def load_old_flim(svm_dir: Path) -> pd.DataFrame:
    """Le os 54 relatorios do Felipe (encoder_mode==frozen) e agrega."""
    rows = []
    for csv_file in sorted(svm_dir.glob("report_svm_*.csv")):
        m = _FELIPE_RE.match(csv_file.name)
        if not m:
            continue
        ds = _FELIPE_DATASET.get(m.group(1))
        if ds is None:
            continue
        d = pd.read_csv(csv_file)
        d = d[d["encoder_mode"] == "frozen"]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append({
            "dataset": ds, "split": int(m.group(2)), "percentage": int(m.group(3)),
            "kappa": float(r["test_cohen_kappa"]),
            "acc":   float(r["test_accuracy"]),
            "f1":    float(r["test_f1_weighted"]),
        })
    if not rows:
        raise RuntimeError(f"nenhum relatorio do Felipe encontrado em {svm_dir}")
    return _agg(pd.DataFrame(rows), "old_flim")


def _draw(ax, data: pd.DataFrame, dataset: str, metric: str, handles: dict) -> None:
    """Desenha um subplot e registra em *handles* (dict serie->Line2D) o que apareceu.

    O dict e compartilhado pela figura toda: uma serie ausente de um subplot ainda
    entra na legenda se aparecer em qualquer outro.
    """
    for key in SERIES_ORDER:
        sub = data[(data["series"] == key) & (data["dataset"] == dataset)]
        sub = sub[sub["percentage"].isin(PCTS)].sort_values("percentage")
        if sub.empty:
            continue
        x    = [PCT_TO_POS[p] for p in sub["percentage"]]
        y    = sub[f"{metric}_mean"].to_numpy()
        yerr = np.nan_to_num(sub[f"{metric}_std"].to_numpy())
        sty  = SERIES[key]
        line, = ax.plot(
            x, y, color=sty["color"], linestyle=sty["ls"], marker=sty["marker"],
            linewidth=2.4, markersize=9, markeredgecolor="white", markeredgewidth=0.8,
            zorder=sty["zorder"], label=sty["label"],
        )
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=sty["color"],
                    elinewidth=1.1, capsize=4, capthick=1.3, alpha=0.65,
                    zorder=sty["zorder"] - 1)
        handles.setdefault(key, line)
    ax.set_xticks(list(PCT_TO_POS.values()))
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlim(-0.3, len(PCTS) - 0.7)
    ax.set_ylim(0, 1.02)


def _ordered(handles: dict) -> list:
    return [handles[k] for k in SERIES_ORDER if k in handles]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real-flim-csv", type=Path,
                    default=_ROOT / "artifacts" / "real_FLIM" / "real_FLIM_results.csv")
    ap.add_argument("--distill4-csv", type=Path,
                    default=Path(RESULTS_DIR) / "svm_distill_proj1280_results.csv")
    ap.add_argument("--felipe-dir", type=Path,
                    default=Path(REPORTS_FELIPE_SVM_DIR))
    ap.add_argument("--out-dir", type=Path,
                    default=_ROOT / "artifacts" / "real_FLIM")
    args = ap.parse_args()

    frames = load_real_flim(args.real_flim_csv)
    frames.append(load_distill4(args.distill4_csv))
    frames.append(load_old_flim(args.felipe_dir))
    data = pd.concat(frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "real_FLIM_vs_distill4_aggregated.csv", index=False)

    sns.set_theme(style="whitegrid", font_scale=1.15,
                  rc={"axes.facecolor": "#FAFAFA", "grid.color": "#E0E0E0"})

    # ── Uma figura por metrica (1 x 3 datasets) ──────────────────────────────
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
        handles: dict = {}
        for ax, ds in zip(axes, DATASETS):
            _draw(ax, data, ds, metric, handles)
            ax.set_title(DATASET_LABEL[ds], fontsize=17, fontweight="bold", pad=10)
            ax.set_xlabel("Labelled training data", fontsize=14)
        axes[0].set_ylabel(METRIC_LABEL[metric], fontsize=15)
        fig.suptitle(f"Real FLIM (avg pool) vs Distill 4 vs old_flim — {METRIC_LABEL[metric]}",
                     fontsize=20, fontweight="bold", y=1.0)
        fig.legend(handles=_ordered(handles), loc="lower center", ncol=2, frameon=True,
                   fontsize=13, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        out = args.out_dir / f"real_FLIM_vs_distill4_{metric}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[OK] {out}")

    # ── Figura unica 3 metricas x 3 datasets ─────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(19, 16), sharey="row", sharex=True)
    handles = {}
    for i, metric in enumerate(METRICS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            _draw(ax, data, ds, metric, handles)
            if i == 0:
                ax.set_title(DATASET_LABEL[ds], fontsize=18, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=16)
            if i == len(METRICS) - 1:
                ax.set_xlabel("Labelled training data", fontsize=14)
    fig.suptitle("Real FLIM (48-d, avg pool) vs Distill 4 (889K) vs old_flim",
                 fontsize=22, fontweight="bold", y=0.995)
    fig.legend(handles=_ordered(handles), loc="lower center", ncol=2, frameon=True,
               fontsize=15, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.01, 1, 0.985))
    out = args.out_dir / "real_FLIM_vs_distill4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
