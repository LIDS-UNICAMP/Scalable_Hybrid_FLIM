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
"""constants.py — Shared constants for the evaluation pipeline: labels, experiment matrix and plot styling."""

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

# ── Data / experiment matrix ──────────────────────────────────────────────────

# Student (FLIM) input size. NOT the same quantity as IJEPAEncoder.IMAGE_SIZE = 224
# (src/models/ijepa_encoder.py:245), which is the teacher input size.
IMAGE_SIZE: int = 200

# src/analysis/tsne_flim.py:105 keeps a 3-key copy of this dict, missing "parasito".
DATASET_NUM_CLASSES: dict[str, int] = {
    "helminth-eggs": 9,
    "helminth-larvae": 2,
    "protozoan-cysts": 7,
    "parasito": 9,
}

SPLITS: list[int] = [1, 2, 3]

# Spelled PCTS in some call sites; PERCENTAGES is the canonical name here.
PERCENTAGES: list[int] = [1, 5, 25, 50, 75, 100]

# Short dataset keys, used by 12+ sites. src/evaluate/svm_ijepa.py:68 and
# check_experiments/_common.py:40 use the long form
# ["helminth-eggs", "helminth-larvae", "protozoan-cysts"] and are NOT consumers of this name.
DATASETS: list[str] = ["eggs", "larvae", "protozoan"]

METRICS: list[str] = ["kappa", "acc", "f1"]


# ── Plot styling (canonical copies live here; src/evaluate/eval_plotter.py) ────

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

INIT_LINE_STYLE: dict[str, dict] = {
    "flim":         {"color": "#0072B2", "ls": "-",   "marker": "o", "zorder": 5},
    "random":       {"color": "#009E73", "ls": "-",   "marker": "s", "zorder": 4},
    "he":           {"color": "#D55E00", "ls": "--",  "marker": "^", "zorder": 3},
    "xavier":       {"color": "#CC79A7", "ls": "--",  "marker": "D", "zorder": 2},
    "trunc_normal": {"color": "#F0E442", "ls": "-.",  "marker": "P", "zorder": 6},
}


# ── Deliberately NOT unified ──────────────────────────────────────────────────
# Each entry below conflicts with a name above; unifying it would change a
# produced artifact (axis text, panel order, CSV columns), so it stays local.
#
# METRIC_LABELS in scripts/plot_parameters_vs_metrics.py:71 — "Cohen's Kappa" (no LaTeX), "F1 Score".
# METRIC_LABEL in scripts/plot_classical_classifiers.py:89 — "Cohen's Kappa (κ)" (literal κ, not LaTeX).
# METRIC_LABEL in scripts/plot_real_flim_vs_distill4.py:60 — "Cohen's kappa", "F1 (weighted)".
# DATASET_LABEL in scripts/plot_real_flim_vs_distill4.py:62 — sentence case ("Helminth eggs").
# METRICS in scripts/plot_parameters_vs_metrics.py:70 — reordered ["f1", "kappa", "acc"] (panel order).
# METRICS in scripts/build_classhead_report.py:48 — different domain: (csv_column, header) tuples.
# METRICS in statistics/tools/wilcoxon_flim_init.py:112 and wilcoxon_equivalence.py:124 — (key, label) tuples.
# SEED = 0 in statistics/tools/measure_compute_cost.py:78 vs SEED = 42 elsewhere — not unified at all.
