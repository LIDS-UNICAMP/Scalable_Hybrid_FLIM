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
"""plot_svm_vs_mlp_pct.py — Curvas de crescimento (métrica × % de dados de treino)
para as três cabeças sobre o MESMO encoder FLIM:

    1. FLIM + SVM              (encoder congelado; SVM linear sobre os embeddings)
    2. FLIM + MLP  frozen      (encoder congelado; só a MLP treina)
    3. FLIM + MLP  unfrozen    (fine-tune end-to-end encoder + MLP)

Fonte: data/reports_felipe/{svm,flim_mlp}/*.csv — os dois conjuntos usam a MESMA
convenção de métrica (global/weighted), então as curvas são diretamente comparáveis.
Percentuais cobertos: 1, 5, 25, 50, 75, 100. Média ± desvio-padrão sobre os 3 splits.

Usa o plotter oficial do repo (src/evaluate/eval_plotter.py), o mesmo dos gráficos
de comparação FLIM/LeJEPA/I-JEPA, para manter o estilo idêntico.

Saída: artifacts/plots/svm_vs_mlp_pct/
    merge_plots/{metric}.png       painéis lado a lado (layout do artigo)
    merge_plots/row/{metric}.png   painéis empilhados na vertical
    individual/...                 um PNG por (dataset × métrica)
    aggregated.csv                 os números por trás das curvas

Uso (o LD_LIBRARY_PATH é necessário, ver scripts/README_plot_comparison_flim.md):

    LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
    /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
      scripts/plot_svm_vs_mlp_pct.py --mode all --metrics acc f1 kappa
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# scripts/ é o sys.path[0] quando se roda `python scripts/plot_svm_vs_mlp_pct.py`;
# tem de vir antes do bloco abaixo, que usa PROJECT_ROOT.
from constants import (  # noqa: E402
    ARTIFACTS_PLOTS_DIR,
    DATASETS,
    DATASET_ALIASES as DATASET_MAP,
    FELIPE_COLUMN_RENAME,
    METRICS,
    PROJECT_ROOT as _ROOT_STR,
    REPORTS_FELIPE_DIR,
)

if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

import pandas as pd

import src.evaluate.eval_plotter as eval_plotter
from src.evaluate.eval_plotter import plot_merge_comparison, plot_method_comparison

# ── As três curvas ────────────────────────────────────────────────────────────
# O _line_key do eval_plotter devolve `method` para a whitelist e `SVM_LeJEPA_{init}`
# no resto; por isso o braço SVM usa method=SVM_FLIM e as MLPs usam init frozen/unfrozen.
LINE_KEYS = ["SVM_FLIM", "SVM_LeJEPA_frozen", "SVM_LeJEPA_unfrozen"]

eval_plotter.METHOD_COMPARE_STYLE = {
    "SVM_FLIM":            {"color": "#0072B2", "ls": "-",  "marker": "o", "zorder": 7},
    "SVM_LeJEPA_frozen":   {"color": "#E69F00", "ls": "--", "marker": "s", "zorder": 8},
    "SVM_LeJEPA_unfrozen": {"color": "#009E73", "ls": "-.", "marker": "^", "zorder": 9},
}
eval_plotter.METHOD_COMPARE_LABEL = {
    "SVM_FLIM":            "FLIM + SVM (frozen)",       # data/reports_felipe/svm/*.csv
    "SVM_LeJEPA_frozen":   "FLIM + MLP (frozen)",       # data/reports_felipe/flim_mlp/*.csv
    "SVM_LeJEPA_unfrozen": "FLIM + MLP (unfrozen)",     # data/reports_felipe/flim_mlp/*.csv
}


def _load(subdir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(REPORTS_FELIPE_DIR) / subdir / "*.csv")))
    if not files:
        raise SystemExit(f"[plot] Nenhum CSV em data/reports_felipe/{subdir}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dataset_short"] = df["dataset"].map(DATASET_MAP)
    return df.dropna(subset=["dataset_short"])


def build_dataframe() -> pd.DataFrame:
    """Agrega os CSVs do Felipe no schema canônico do eval_plotter."""
    svm = _load("svm")
    mlp = _load("flim_mlp")

    # SVM não faz backprop: só existe em frozen.
    svm = svm[svm["encoder_mode"] == "frozen"].copy()
    svm["method"], svm["init"] = "SVM_FLIM", "flim"

    mlp = mlp[mlp["encoder_mode"].isin(["frozen", "unfrozen"])].copy()
    mlp["method"] = "MLP_FLIM"
    mlp["init"] = mlp["encoder_mode"]          # vira o sufixo do _line_key

    df = pd.concat([svm, mlp], ignore_index=True)
    df = df.rename(columns=FELIPE_COLUMN_RENAME)

    g = df.groupby(["method", "init", "dataset_short", "pretrained_pct"], as_index=False)
    agg = g.agg(
        n_splits=("split", "nunique"),
        kappa=("kappa", "mean"), kappa_std=("kappa", "std"),
        acc=("acc", "mean"),     acc_std=("acc", "std"),
        f1=("f1", "mean"),       f1_std=("f1", "std"),
    )
    return agg.fillna({"kappa_std": 0.0, "acc_std": 0.0, "f1_std": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--mode", choices=["all", "individual", "merge", "row"], default="all")
    ap.add_argument("--metrics", nargs="+", default=METRICS, metavar="METRIC")
    ap.add_argument("--datasets", nargs="+", default=None, metavar="DS")
    ap.add_argument("--out", default=str(Path(ARTIFACTS_PLOTS_DIR) / "svm_vs_mlp_pct"))
    args = ap.parse_args()

    df = build_dataframe()
    if args.datasets:
        df = df[df["dataset_short"].isin(args.datasets)]
    datasets = [d for d in DATASETS if d in set(df["dataset_short"])]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "aggregated.csv"
    df.sort_values(["dataset_short", "method", "init", "pretrained_pct"]).to_csv(csv_path, index=False)
    print(f"[plot] Números das curvas -> {csv_path}")

    pcts = sorted(df["pretrained_pct"].unique())
    print(f"[plot] datasets={datasets}  pcts={pcts}  curvas={len(df.groupby(['method','init']))}")

    if args.mode in ("all", "individual"):
        # plot_method_comparison não filtra por dataset — o caller precisa passar o recorte.
        for ds in datasets:
            ds_df = df[df["dataset_short"] == ds]
            for metric in args.metrics:
                plot_method_comparison(ds_df, ds, metric, out / "individual")

    if args.mode in ("all", "merge"):
        plot_merge_comparison(df, datasets, args.metrics, out / "merge_plots")

    if args.mode in ("all", "row"):
        plot_merge_comparison(df, datasets, args.metrics, out / "merge_plots" / "row",
                              orientation="vertical")

    print(f"[plot] OK -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
