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
"""constants.py — fonte unica das constantes compartilhadas por scripts/.

Regra: um valor que aparece identico em DOIS OU MAIS arquivos de scripts/ mora
aqui. Um valor que existe em um arquivo so, ou que difere de propositio entre
arquivos, fica no arquivo dele — centralizar divergencia intencional e como
apagar a intencao.

## Como importar

Os scripts desta pasta rodam como ``python scripts/X.py``, entao o proprio
diretorio ``scripts/`` e o ``sys.path[0]`` do processo:

    from constants import DATASETS, NUM_CLASSES

`autoencoder_flim_ray.py` e a excecao: ele tambem e IMPORTADO como modulo por
quatro arquivos de fora (``src/evaluate/eval_avg_pooling_48d.py``,
``src/evaluate/eval_svm_flim_flatten.py``, ``src/evaluate/svm_real_flim.py`` e
``tools/check_probe_matches_evaluator.py``, todos atras de ``_arch_json`` e
``_flim_weights_path``). Nesse caminho o ``sys.path[0]`` e a raiz do repo, nao
``scripts/``, entao aquele arquivo insere o proprio diretorio no ``sys.path``
antes de importar daqui. Nao remova aquele bloco.

## Relacao com src/evaluate/constants.py

Aquele arquivo serve o pipeline de avaliacao e usa chave LONGA de dataset
(``"helminth-eggs"``); este serve os launchers e usa chave CURTA (``"eggs"``).
Os dois coexistem de proposito. `DATASET_LONG_TO_SHORT` faz a ponte.

## O que NAO foi unificado, e por que

* **Arquitetura do protozoan.** `ch24_30_48_a0.5_f5` (arch JSON) e
  `ch24_32_48_a0.5_f5` (pesos) sao diretorios diferentes para o mesmo dataset.
  Nao e erro de digitacao: viraram `FLIM_ARCH_BASE` e `FLIM_WEIGHTS_BASE`.
* **Projeto W&B.** `flim-ssl` na maioria, `journal_02_2026_hybrid_FLIM` no
  autoencoder, `flim-ssl_old` no limpador de cache. Tres constantes.
* **`--num-gpus` / `--max-concurrent-per-gpu`.** Cada launcher tem o default
  dele (1/3/4, 1/3/10) porque os modelos tem tamanhos diferentes. Ficam no
  argparse de cada script.
* **`ALL_INITS`.** Em `distillation_*_ray` e init do student
  (``["trunc_normal"]``); em `retry_protozoan_experiment` e init do SSL
  (``["xavier", "random", "he", "flim"]``). Nomes distintos aqui.
* **`freeze/unfreeze` vs `frozen/unfrozen`.** Os dois vocabularios existem em
  disco (nomes de pasta de config e coluna de CSV). Ambos estao registrados;
  unificar quebraria caminho gravado.
"""

from __future__ import annotations

import os

# ─── Raiz do repositorio ──────────────────────────────────────────────────────

# scripts/constants.py -> scripts -> raiz.
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Datasets ─────────────────────────────────────────────────────────────────

# Chave curta: a forma canonica dentro de scripts/.
DATASETS: list[str] = ["eggs", "larvae", "protozoan"]

# Chave longa: o nome da pasta em disco, e o que src/evaluate/constants.py usa.
DATASETS_LONG: list[str] = ["helminth-eggs", "helminth-larvae", "protozoan-cysts"]

# Curto -> pasta em disco. Historicamente chamado _PARASITE_DIR.
PARASITE_DIR: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

# Longo -> curto. Era _DATASET_CONFIG_ALIAS, _LONG_DATASET_MAP e
# DATASET_NAME_MAP, tres nomes para o mesmo dicionario.
DATASET_LONG_TO_SHORT: dict[str, str] = {v: k for k, v in PARASITE_DIR.items()}

# Toda grafia que aparece em nome de arquivo -> curto. "cistos" e o nome usado
# nos CSVs externos de data/reports_felipe/; "parasito" e o dataset agregado.
DATASET_ALIASES: dict[str, str] = {
    **DATASET_LONG_TO_SHORT,
    **{d: d for d in DATASETS},
    "cistos":   "protozoan",
    "parasito": "parasito",
}

NUM_CLASSES: dict[str, int] = {"eggs": 9, "larvae": 2, "protozoan": 7}

# Rotulo de plot. Title Case — a variante sentence case que existia em
# plot_real_flim_vs_distill4.py foi mantida la, porque muda o PNG.
DATASET_LABEL: dict[str, str] = {
    "eggs":      "Helminth Eggs",
    "larvae":    "Helminth Larvae",
    "protozoan": "Protozoan Cysts",
}

# ─── Grade experimental ───────────────────────────────────────────────────────

SPLITS: list[int] = [1, 2, 3]
PERCENTAGES: list[int] = [1, 5, 25, 50, 75, 100]

# ─── Caminhos ─────────────────────────────────────────────────────────────────

DATA_ROOT: str = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
ARTIFACTS_DIR: str = os.path.join(PROJECT_ROOT, "artifacts")
CONFIGS_DIR: str = os.path.join(PROJECT_ROOT, "configs")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")

DEFAULT_CONFIG_YAML: str = os.path.join(CONFIGS_DIR, "default.yaml")
MLP_CONFIGS_DIR: str = os.path.join(CONFIGS_DIR, "evaluate", "mlp")

# Splits incrementais consumidos pelos launchers.
DATASET_SPLITS_ROOT: str = os.path.join(DATA_ROOT, "to_modules", "new_split_parasito")

# CSVs do pipeline externo, nao versionado.
REPORTS_FELIPE_DIR: str = os.path.join(DATA_ROOT, "reports_felipe")
REPORTS_FELIPE_SVM_DIR: str = os.path.join(REPORTS_FELIPE_DIR, "svm")

ARTIFACTS_NORMALIZED_DIR: str = os.path.join(ARTIFACTS_DIR, "normalized")
ARTIFACTS_PLOTS_DIR: str = os.path.join(ARTIFACTS_DIR, "plots")
ARTIFACTS_DISTILLATION_DIR: str = os.path.join(ARTIFACTS_DIR, "distillation")
ARTIFACTS_CLASSIFICATION_FLIM_DIR: str = os.path.join(ARTIFACTS_DIR, "classification_flim")

UNIFIED_SVM_COMPARISON_CSV: str = os.path.join(
    ARTIFACTS_NORMALIZED_DIR, "unified_svm_comparison.csv")

# ─── Pesos FLIM entregues ─────────────────────────────────────────────────────

FLIM_MODEL_ROOT: str = os.path.join(DATA_ROOT, "to_mateus", "model")

# A tag de canais faz parte do nome do diretorio. protozoan diverge: 30 canais
# na arquitetura, 32 nos pesos. Os dois dicts abaixo NAO sao redundantes.
ARCH_TAG: dict[str, str] = {
    "eggs":      "ch24_32_48",
    "larvae":    "ch24_32_48",
    "protozoan": "ch24_30_48",
}
_FLIM_SUFFIX = "_a0.5_f5"

FLIM_ARCH_BASE: dict[str, str] = {
    ds: os.path.join(FLIM_MODEL_ROOT, ARCH_TAG[ds] + _FLIM_SUFFIX, ds)
    for ds in DATASETS
}
FLIM_WEIGHTS_BASE: dict[str, str] = {
    ds: os.path.join(FLIM_MODEL_ROOT, "ch24_32_48" + _FLIM_SUFFIX, ds)
    for ds in DATASETS
}

# ─── Nomes de arquivo e subdiretorio ──────────────────────────────────────────

ARCH_JSON_FILENAME: str = "architecture.json"
MODELS_SUBDIR: str = "models"
CHECKPOINTS_SUBDIR: str = "checkpoints"
SPLITS_INCREMENTAL_SUBDIR: str = "splits_incremental"
RUN_METADATA_FILENAME: str = "run_metadata.json"
WRITE_TEST_FILENAME: str = ".write_test"
LAST_CKPT_FILENAME: str = "last.ckpt"


def train_dir(split: int) -> str:
    """Subpasta do split dentro dos pesos FLIM: ``train1``, ``train2``, ..."""
    return f"train{split}"


def split_dir(split: int) -> str:
    """Subpasta do split dentro de ``splits_incremental``: ``split1``, ..."""
    return f"split{split}"


def data_descriptor_filename(pct: int) -> str:
    """Descritor do percentual: ``data_descriptor_perc75.json``."""
    return f"data_descriptor_perc{pct}.json"


# ─── W&B ──────────────────────────────────────────────────────────────────────

WANDB_ENTITY: str = "ophira-ai"
WANDB_PROJECT: str = "flim-ssl"
# O autoencoder loga em projeto proprio, de proposito.
WANDB_PROJECT_AUTOENCODER: str = "journal_02_2026_hybrid_FLIM"
# Projeto legado, so o limpador de cache aponta para ele.
WANDB_PROJECT_LEGACY: str = "flim-ssl_old"

WANDB_FAILED_STATES: frozenset[str] = frozenset({"crashed", "failed", "killed"})
WANDB_SKIP_STATES: frozenset[str] = frozenset({"finished", "running"})

# Env dos subprocessos de treino: console mudo, logging online.
WANDB_CHILD_ENV: dict[str, str] = {"WANDB_CONSOLE": "off", "WANDB_MODE": "online"}

# ─── Ray e subprocessos ───────────────────────────────────────────────────────

CUDA_ENV_VAR: str = "CUDA_VISIBLE_DEVICES"
OMP_ENV_VAR: str = "OMP_NUM_THREADS"

# Ray nao gerencia GPU: o pinning e feito na mao via CUDA_VISIBLE_DEVICES.
RAY_INIT_KWARGS: dict = {"ignore_reinit_error": True, "log_to_driver": True, "num_gpus": 0}

DEFAULT_CPUS_PER_EXPERIMENT: int = 4
DEFAULT_NUM_WORKERS: int = 4

# Truncamento do stderr do filho: HEAD primeiros + TAIL ultimos caracteres.
STDERR_TRUNCATE_MAX_CHARS: int = 2000
STDERR_TRUNCATE_HEAD: int = 1000
STDERR_TRUNCATE_TAIL: int = 1000

# ─── Log ──────────────────────────────────────────────────────────────────────

LOG_LEVELS: dict[str, int] = {"DEBUG": 0, "INFO": 1, "WARN": 2}
LOG_LEVEL_DEFAULT: str = "INFO"
LOG_TIME_FMT: str = "%H:%M:%S"
SEP_WIDTH: int = 70

# ─── Metricas e schema dos CSVs agregados ─────────────────────────────────────

METRICS: list[str] = ["kappa", "acc", "f1"]

# Rotulo de plot das metricas. plot_real_flim_vs_distill4.py mantem a variante
# "F1 (weighted)" / "Cohen's kappa" — trocar aqui mudaria o PNG.
METRIC_LABEL: dict[str, str] = {
    "kappa": "Cohen's Kappa",
    "acc":   "Accuracy",
    "f1":    "F1 Score",
}

# Schema do unified_svm_comparison.csv e dos agregados que alimentam os plots.
CANONICAL_COLS: list[str] = [
    "method", "init", "dataset_short", "pretrained_pct", "n_splits",
    "kappa", "kappa_std", "acc", "acc_std", "f1", "f1_std",
]

# Colunas dos CSVs externos -> nomes canonicos.
FELIPE_COLUMN_RENAME: dict[str, str] = {
    "test_cohen_kappa":  "kappa",
    "test_accuracy":     "acc",
    "test_f1_weighted":  "f1",
    "percentage":        "pretrained_pct",
}

# ─── Checkpoints de destilacao ────────────────────────────────────────────────

# O que sobrevive quando o teacher e removido do .ckpt.
KEEP_STUDENT_PREFIXES: tuple[str, ...] = ("student.", "proj_kd.")
TEACHER_PREFIX: str = "teacher."
