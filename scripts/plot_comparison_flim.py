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

"""plot_comparison_flim.py — Comparison plots: FLIM vs LeJEPA vs I-JEPA vs Distillation.

Filters the unified SVM CSV to keep:
  - SVM_FLIM
  - SVM_LeJEPA (trunc_normal init)
  - SVM_IJEPA
  - SVM_Distill_Proj1280 / SVM_Distill_3x3BN / SVM_Distill_1x1BN / SVM_Distill_2l400K

Two output modes:
  individual/   — one plot per (dataset, metric)
  merge_plots/  — one merged grid per metric (all datasets as subplots)

Outputs saved to: artifacts/plots/plots_compare_to_flim/

Usage:
    python scripts/plot_comparison_flim.py                   # both modes
    python scripts/plot_comparison_flim.py --mode individual
    python scripts/plot_comparison_flim.py --mode merge
    python scripts/plot_comparison_flim.py --metrics kappa f1
    python scripts/plot_comparison_flim.py --datasets eggs larvae
    python scripts/plot_comparison_flim.py --out /custom/path
    python scripts/plot_comparison_flim.py --csv /path/to/unified.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT_STR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

import pandas as pd

import src.evaluate.eval_plotter as eval_plotter
from src.evaluate.eval_plotter import plot_method_comparison, plot_merge_comparison

# ── PROCEDÊNCIA DE CADA CURVA (de onde vem cada valor / cada CSV) ─────────────
# Toda curva sai de  artifacts/normalized/unified_svm_comparison.csv, montado por
# scripts/normalize_reports.py. Cada função _normalize_*() lê o CSV de origem,
# filtra (status==ok, encoder_init, encoder_mode...) e agrega média ± std sobre os
# splits/folds por (dataset, percentage). Mapa chave→função→CSV de origem→filtro:
#
#   FLIM
#     SVM_FLIM                       → normalize_felipe_svm()
#       data/reports_felipe/svm/report_svm_{ds}_split{N}_perc{pct}.csv   (encoder_mode==frozen)
#   LeJEPA
#     SVM_LeJEPA_trunc_normal        → load_lejepa_svm_metrics()
#       artifacts/SVM/*/*/metrics_SVM_*.csv                              (init==trunc_normal; filtro no _filter() abaixo)
#   I-JEPA
#     SVM_IJEPA                      → normalize_ijepa_svm()
#       results/ijepa_svm_aggregated.csv
#   Distill 4
#     SVM_Distill_Proj1280           → _normalize_distill_proj1280()
#       results/svm_distill_proj1280_results.csv                         (status==ok)
#   Distill 3
#     SVM_Distill_3x3BN              → _normalize_distill_3x3bn()
#       results/svm_proj1280_3x3_BN2d_results.csv                        (status==ok & encoder_init==trunc_normal)
#   Distill 1
#     SVM_Distill_1x1BN              → _normalize_distill_1x1bn()
#       results/svm_proj1280_1x1_BN2d_results.csv                        (status==ok & encoder_init==trunc_normal)
#   Distill 2
#     SVM_Distill_2l400K             → _normalize_distill_2l_400k()
#       results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv            (status==ok)
#   Distill 1 (FLIM init)
#     SVM_Distill_1x1BN_flim_frozen_eval_loss → _normalize_distill_flim_frozen()
#       results/svm_distillation_conv_flim_frozen_results.csv            (status==ok; checkpoint best-loss)
#
# Como cada CSV de origem é gerado (treino → checkpoint → SVM eval → CSV):
#   ver  metrics_distillation/data_provenance.md
# ─────────────────────────────────────────────────────────────────────────────

# Modelos oficiais do experimento — rótulos curtos (inglês), mesma ordem da legenda.
# Apenas estas 8 chaves devem aparecer nas curvas/legenda.
eval_plotter.METHOD_COMPARE_LABEL = {
    "SVM_FLIM":                                 "FLIM (59.504)",               # data/reports_felipe/svm/report_svm_*.csv
    "SVM_LeJEPA_trunc_normal":                  "LeJEPA (59.504)",             # artifacts/SVM/*/*/metrics_SVM_*.csv
    "SVM_IJEPA":                                "I-JEPA (632M)",               # results/ijepa_svm_aggregated.csv
    "SVM_Distill_Proj1280":                     "Distill 4 (889K)",            # results/svm_distill_proj1280_results.csv
    "SVM_Distill_3x3BN":                        "Distill 3 (615K)",            # results/svm_proj1280_3x3_BN2d_results.csv
    "SVM_Distill_1x1BN":                        "Distill 1 (123K)",            # results/svm_proj1280_1x1_BN2d_results.csv
    "SVM_Distill_2l400K":                       "Distill 2 (402K)",            # results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv
    "SVM_Distill_1x1BN_flim_frozen_eval_loss":  "Distill 1 — FLIM init (123K)", # results/svm_distillation_conv_flim_frozen_results.csv
}

_ROOT   = Path(__file__).resolve().parent.parent
METRICS = ["kappa", "acc", "f1"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate FLIM vs LeJEPA vs I-JEPA vs Distillation comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "individual", "merge", "row"],
        default="all",
        help="Plot type(s) to generate (default: all). "
             "'merge' = panels side-by-side (horizontal); "
             "'row' = panels stacked vertically, saved to merge_plots/row/.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        metavar="METRIC",
        help=f"Metrics to plot. Default: {METRICS}.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DS",
        help="Datasets to include (default: all present in CSV).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "artifacts" / "normalized" / "unified_svm_comparison.csv",
        metavar="PATH",
        help="Path to the unified SVM comparison CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "artifacts" / "plots" / "plots_compare_to_flim",
        metavar="DIR",
        help="Base output directory (default: artifacts/plots/plots_compare_to_flim).",
    )

    # ── Merge/row-plot style controls ─────────────────────────────────────────
    # Todos os controles de fonte abaixo valem tanto para --mode merge quanto row.
    merge = parser.add_argument_group("merge/row plot style (--mode merge/row/all)")
    merge.add_argument(
        "--subtitle-fontsize", type=int, default=30, metavar="N",
        help="Titulo do dataset acima de cada subplot (default: 30).",
    )
    merge.add_argument(
        "--tick-fontsize", type=int, default=28, metavar="N",
        help="Porcentagens no eixo X (default: 28).",
    )
    merge.add_argument(
        "--ytick-fontsize", type=int, default=24, metavar="N",
        help="Numeros no eixo Y (default: 24).",
    )
    merge.add_argument(
        "--ylabel-fontsize", type=int, default=28, metavar="N",
        help="Label do eixo Y (kappa/acc/f1) (default: 28).",
    )
    merge.add_argument(
        "--legend-fontsize", type=int, default=24, metavar="N",
        help="Texto dos itens da legenda (default: 24).",
    )
    merge.add_argument(
        "--legend-ncol", type=int, default=-1, metavar="N",
        help="Colunas da legenda. -1 = tudo em uma linha (default: -1).",
    )
    merge.add_argument(
        "--linewidth", type=float, default=3.5, metavar="F",
        help="Espessura das linhas (default: 3.5).",
    )
    merge.add_argument(
        "--markersize", type=int, default=14, metavar="N",
        help="Tamanho dos marcadores (default: 14).",
    )
    merge.add_argument(
        "--fig-height", type=int, default=10, metavar="N",
        help="Altura da figura em polegadas (default: 10).",
    )
    merge.add_argument(
        "--fig-width-per-dataset", type=int, default=13, metavar="N",
        help="Largura por dataset em polegadas (default: 13).",
    )
    merge.add_argument(
        "--legend-y", type=float, default=0.0, metavar="F",
        help="Posicao vertical da legenda em coordenadas da figura. "
             "0.0 = borda inferior, negativo = mais para baixo (default: 0.0).",
    )
    merge.add_argument(
        "--show-xlabel", action="store_true", default=False,
        help="Mostrar label 'Pre-training Data' no eixo X (default: oculto).",
    )
    return parser


def _filter(unified: pd.DataFrame) -> pd.DataFrame:
    # Apenas os 8 modelos oficiais. CSV de origem de cada um (ver bloco de
    # procedência acima e metrics_distillation/data_provenance.md):
    mask = (
        (unified["method"] == "SVM_FLIM") |                       # data/reports_felipe/svm/report_svm_*.csv
        (unified["method"] == "SVM_IJEPA") |                      # results/ijepa_svm_aggregated.csv
        (unified["method"] == "SVM_Distill_Proj1280") |           # results/svm_distill_proj1280_results.csv
        (unified["method"] == "SVM_Distill_3x3BN") |              # results/svm_proj1280_3x3_BN2d_results.csv (trunc_normal)
        (unified["method"] == "SVM_Distill_1x1BN") |              # results/svm_proj1280_1x1_BN2d_results.csv (trunc_normal)
        (unified["method"] == "SVM_Distill_1x1BN_nonorm") |
        (unified["method"] == "SVM_Distill_2l400K") |             # results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv
        (unified["method"] == "SVM_Distill_2l400K_flim_nonorm") |
        (unified["method"] == "SVM_Distill_1x1BN_flim_frozen_eval_knn") |
        (unified["method"] == "SVM_Distill_1x1BN_flim_frozen_eval_loss") |  # results/svm_distillation_conv_flim_frozen_results.csv
        ((unified["method"] == "SVM_LeJEPA") & (unified["init"] == "trunc_normal"))  # artifacts/SVM/*/*/metrics_SVM_*.csv
    )
    return unified[mask].copy()


def main() -> None:
    args     = _build_parser().parse_args()
    mode     = args.mode
    metrics  = args.metrics
    out_base = args.out

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Unified CSV not found at {args.csv}\n"
            "Run 'python scripts/normalize_reports.py' first."
        )

    unified  = pd.read_csv(args.csv)
    filtered = _filter(unified)

    datasets_present = sorted(filtered["dataset_short"].dropna().unique())
    if args.datasets:
        datasets_present = [d for d in args.datasets if d in datasets_present]
        if not datasets_present:
            print("[ERROR] None of the requested datasets found in CSV.")
            return

    print(f"\n{'=' * 70}")
    print(f"[READ]  {args.csv}  ({len(unified)} rows -> {len(filtered)} after filter)")
    print(f"[MODE]  {mode}  |  metrics: {metrics}  |  datasets: {datasets_present}")
    print(f"[OUT]   {out_base}")
    print(f"{'=' * 70}\n")

    # ── Individual plots (one per dataset x metric) ───────────────────────────
    if mode in ("all", "individual"):
        print("[PLOTS] individual/ — one plot per (dataset, metric)")
        for dataset in datasets_present:
            ds_df = filtered[filtered["dataset_short"] == dataset]
            for metric in metrics:
                plot_method_comparison(ds_df, dataset, metric, out_base)

    # ── Merged plots: {metric}.png, one file per metric ───────────────────────
    #   merge → panels side-by-side (horizontal), merge_plots/{metric}.png
    #   row   → panels stacked vertically,          merge_plots/row/{metric}.png
    merge_style = dict(
        show_xlabel=args.show_xlabel,
        subtitle_fontsize=args.subtitle_fontsize,
        tick_fontsize=args.tick_fontsize,
        ytick_fontsize=args.ytick_fontsize,
        ylabel_fontsize=args.ylabel_fontsize,
        legend_fontsize=args.legend_fontsize,
        legend_ncol=args.legend_ncol,
        legend_y=args.legend_y,
        linewidth=args.linewidth,
        markersize=args.markersize,
        fig_height=args.fig_height,
        fig_width_per_dataset=args.fig_width_per_dataset,
    )
    n = len(metrics) * len(datasets_present)

    if mode in ("all", "merge"):
        merge_dir = out_base / "merge_plots"
        print(f"\n[PLOTS] merge_plots/ (horizontal) — {len(metrics)} metrics x {len(datasets_present)} datasets = {n} plots")
        plot_merge_comparison(filtered, datasets_present, metrics, merge_dir,
                              orientation="horizontal", **merge_style)

    if mode in ("all", "row"):
        row_dir = out_base / "merge_plots" / "row"
        print(f"\n[PLOTS] merge_plots/row/ (vertical) — {len(metrics)} metrics x {len(datasets_present)} datasets = {n} plots")
        plot_merge_comparison(filtered, datasets_present, metrics, row_dir,
                              orientation="vertical", **merge_style)

    print(f"\n[DONE] Plots saved to: {out_base}/")


if __name__ == "__main__":
    main()
