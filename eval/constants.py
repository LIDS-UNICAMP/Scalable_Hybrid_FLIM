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

"""constants.py — Nomes e dados so do pacote eval/.

Arquivo PLANO: so nomes e dados. Sem class, sem def, sem valor computado por
chamada de funcao, sem os.environ, sem leitura de disco.

O que era global (IMAGE_SIZE, DATASET_NUM_CLASSES, SPLITS, PERCENTAGES, DATASETS,
METRICS) subiu para core.constants e NAO se repete aqui (anti-shadowing). Local nao
importa de outro local: este arquivo NAO importa analysis/constants.py.
"""

__all__ = [
    "CLASS_NAMES",
    "INIT_ORDER",
    "METHOD_COLOR",
    "METHOD_LABEL",
    "METHOD_HATCH",
    "DATASET_LABEL",
    "METRIC_LABELS",
    "INIT_LINE_STYLE",
]


# ── Nomes de classe por dataset (origem: src/evaluate/constants.py:20) ────────
# Consumidor: src/evaluate/tsne_analysis.py:75.

CLASS_NAMES: dict[str, list[str]] = {
    "eggs": [
        "Hymenolepis nana",
        "Hymenolepis diminuta",
        "Ancylostoma",
        "Enterobius vermicularis",
        "Ascaris lumbricoides",
        "Trichuris trichiura",
        "Schistosoma mansoni",
        "Taenia spp",
        "Impurities",
    ],
    "larvae": [
        "Strongyloides stercoralis",
        "Impurities",
    ],
    "protozoan": [
        "Entamoeba coli",
        "Entamoeba histolytica",
        "Endolimax nana",
        "Giardia intestinalis",
        "Iodamoeba bütschlii",
        "Blastocystis hominis",
        "Impurities",
    ],
}


# ── Estilo de plot (origem: src/evaluate/constants.py:74-109) ─────────────────
# Consumidor em eval/: eval/eval_plotter.py:45-55, unico arquivo do pacote que
# usa os seis nomes abaixo.
#
# CONFLITO ABERTO, registrado e NAO resolvido aqui: spec_refactor.md:672 manda
# esses mesmos seis nomes para analysis/constants.py. Dois pacotes consumindo o
# mesmo nome e, pela tabela de spec_refactor.md:700, definicao de nome GLOBAL —
# ele deveria subir para core/constants.py, nao atravessar de local para local
# (regra 3 de spec_refactor.md:706). Ficam aqui porque este e o arquivo de
# origem deles e core/constants.py nao e deste agente. Ver MIGRATION.md.

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


# ── Estilo de linha por init (origem: src/evaluate/constants.py:112) ──────────
# Consumidor: src/evaluate/eval_plotter.py:45. NAO unificar com INIT_STYLE de
# src/utils/plot_svm_results.py:51 (mesmos hex, sem a entrada "trunc_normal").

INIT_LINE_STYLE: dict[str, dict] = {
    "flim":         {"color": "#0072B2", "ls": "-",   "marker": "o", "zorder": 5},
    "random":       {"color": "#009E73", "ls": "-",   "marker": "s", "zorder": 4},
    "he":           {"color": "#D55E00", "ls": "--",  "marker": "^", "zorder": 3},
    "xavier":       {"color": "#CC79A7", "ls": "--",  "marker": "D", "zorder": 2},
    "trunc_normal": {"color": "#F0E442", "ls": "-.",  "marker": "P", "zorder": 6},
}
