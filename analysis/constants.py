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

"""constants.py — Nomes e dados compartilhados pelo pacote analysis/.

Arquivo PLANO: so nomes e dados. Sem class, sem def, sem valor computado por
chamada de funcao, sem os.environ, sem leitura de disco.

Nomes globais (DATASETS, PERCENTAGES, METRICS) vivem em core.constants e aqui sao
apenas reexportados — nunca redefinidos (anti-shadowing). Local nao importa de
outro local: este arquivo NAO importa eval/constants.py.
"""

from core.constants import DATASETS, METRICS, PERCENTAGES

__all__ = [
    "DATASETS",
    "METRICS",
    "PERCENTAGES",
    "PCTS",
    "PCTS_CURRICULUM",
    "SEED",
    "DEFAULT_CSV",
    "INIT_ORDER",
    "METHOD_COLOR",
    "METHOD_LABEL",
    "METHOD_HATCH",
    "DATASET_LABEL",
    "METRIC_LABELS",
]


# ── Matriz de experimentos ────────────────────────────────────────────────────

# Segunda grafia de PERCENTAGES, usada pelos sites de analise (os 5 wilcoxon,
# check_experiments/_common.py:42, analysis_flim_distill/aggregate_nonorm_compare.py:56).
# Mesmo valor, nome diferente: alias, nao copia.
PCTS = PERCENTAGES

# Curriculo de duas porcentagens do spifil_growth. NAO e subconjunto de PERCENTAGES,
# e outra grade — por isso nome proprio. Origens:
# tools/plot_partial_train_spifil_hybrid.py:73 (PCTS = (5, 50), tupla) e
# tools/heatmap_stages.py:92 (PERCENTAGES = [5, 50], lista).
PCTS_CURRICULUM: list[int] = [5, 50]

# Semente dos bootstraps dos wilcoxon e da analise de embedding.
# statistics/tools/measure_compute_cost.py:78 usa SEED = 0 de proposito e NAO
# consome este nome: divergencia deliberada, nao unificada.
SEED: int = 42

# CSV unificado que os 5 wilcoxon leem por default. Caminho relativo a raiz do repo:
# arquivo plano nao pode montar Path (chamada de funcao). O mesmo caminho existe em
# experiments sob o nome UNIFIED_SVM_COMPARISON_CSV (scripts/constants.py:144).
DEFAULT_CSV: str = "artifacts/normalized/unified_svm_comparison.csv"


# ── Estilo de plot ────────────────────────────────────────────────────────────

INIT_ORDER = ["flim", "he", "xavier", "random", "trunc_normal"]

# Uma cor por metodo — mesma paleta das cores de init de plot_svm_results.py
METHOD_COLOR: dict[str, str] = {
    "SVM":          "#0072B2",   # azul
    "MLP_freeze":   "#D55E00",   # laranja-vermelho
    "MLP_unfreeze": "#009E73",   # verde
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


# ── Deliberadamente NAO unificado ─────────────────────────────────────────────
# Cada item abaixo colide de nome com algo daqui e continua local: unificar mudaria
# um artefato produzido (texto de eixo, ordem de painel, coluna de CSV, medicao).
#
# DATASETS forma longa em check_experiments/_common.py:40 — chaves longas, outro dominio.
# DATASETS de analysis_flim_distill/ruler_mismatch.py:52 — so ["eggs", "larvae"].
# DATASETS de tools/plot_comparacao_flim_protocolo.py:63 — lista de tuplas (chave, rotulo).
# SEED = 0 em statistics/tools/measure_compute_cost.py:78 — muda a medicao.
# METRICS como tuplas (key, label) em statistics/tools/wilcoxon_flim_init.py:112 e
#   wilcoxon_equivalence.py:124; como (csv_column, header) em scripts/build_classhead_report.py:55.
# METRIC / METRIC_LABEL string unica em statistics/tools/wilcoxon_{acc:93,f1:94,kappa:88}.py —
#   sao a identidade do script, nao duplicacao.
# FAMILY_COLORS / FAMILY_MARKERS em tools/plot_continuity_spifil_hybrid.py:72,83 — paleta por
#   familia, 1 consumidor, escolha do 9o tom justificada em comentario no proprio arquivo.
# STAGE_COLORS em tools/plot_partial_train_spifil_hybrid.py:213 — paleta por estagio; o
#   comentario em plot_continuity_spifil_hybrid.py:70 diz que uma nao serve para a outra.
# INIT_LINE_STYLE fica em eval/constants.py (consumidor src/evaluate/eval_plotter.py:45);
#   INIT_STYLE de src/utils/plot_svm_results.py:51 e o mesmo hex sem "trunc_normal".
