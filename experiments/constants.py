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

"""constants.py — constantes locais do pacote ``experiments``.

Aparato de lancador: o que so os launchers Ray, os scripts de status e os
normalizadores/plots deste pacote leem. E a metade de ``scripts/constants.py``
que NAO subiu para ``core/constants.py`` — la ficou o que dois ou mais pacotes
consomem (``PROJECT_ROOT``, ``DATASETS``, ``NUM_CLASSES``, ``SPLITS``,
``PERCENTAGES``, ``METRICS``, ``FLIM_ARCH_BASE``, ``OMP_ENV_VAR``, ...), porque
isso e contrato entre pacotes.

Arquivo PLANO: so nomes e dados. Sem ``class``, sem ``def``, sem variavel de
ambiente lida em tempo de import, sem leitura de disco. O unico import de pacote
permitido e ``core.constants`` (local pode importar global; global nunca importa
local, e local nunca importa local de outro pacote).

## Como importar

    from experiments.constants import RESULTS_DIR, WANDB_ENTITY

O antigo ``from constants import ...`` so funcionava porque o proprio diretorio
``scripts/`` era o primeiro item do caminho de import de ``python scripts/X.py``.
Com o pacote, esse hack morre.

## Relacao com ``eval/constants.py``

Aquele arquivo serve o pipeline de avaliacao e usa chave LONGA de dataset
(``"helminth-eggs"``); este serve os lancadores e usa chave CURTA (``"eggs"``).
Os dois coexistem de proposito. ``DATASET_LONG_TO_SHORT`` faz a ponte.

## O que NAO foi unificado, e por que

* **Projeto W&B.** ``flim-ssl`` na maioria (``WANDB_PROJECT``, aqui),
  ``journal_02_2026_hybrid_FLIM`` no autoencoder e ``flim-ssl_old`` no limpador
  de cache. Sao tres constantes de proposito; as duas de consumidor unico ficam
  no topo do proprio consumidor.
* **``--num-gpus`` / ``--max-concurrent-per-gpu``.** Cada launcher tem o default
  dele (1/3/4, 1/3/10) porque os modelos tem tamanhos diferentes. Ficam no
  argparse de cada script.
* **``ALL_INITS``.** Em ``distillation_*_ray`` e init do student
  (``["trunc_normal"]``); em ``retry_protozoan_experiment`` e init do SSL
  (``["xavier", "random", "he", "flim"]``). Dois significados, nomes distintos.
* **``freeze/unfreeze`` vs ``frozen/unfrozen``.** Os dois vocabularios existem em
  disco (nome de pasta de config e coluna de CSV). Ambos estao registrados;
  unificar quebraria caminho gravado.
"""

from __future__ import annotations

# Alias privado de proposito: PROJECT_ROOT e nome do global e nao pode ser
# reexportado daqui (nao pode existir `experiments.constants.PROJECT_ROOT`,
# senao ha duas moradas para o mesmo nome).
from core.constants import PROJECT_ROOT as _PROJECT_ROOT

# ─── Datasets ─────────────────────────────────────────────────────────────────

# Curto -> pasta em disco. Historicamente chamado _PARASITE_DIR.
PARASITE_DIR: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

# Longo -> curto: o inverso de PARASITE_DIR, escrito por extenso porque este
# arquivo e plano (o original era uma comprehension sobre `.items()`). Ja se
# chamou _DATASET_CONFIG_ALIAS, _LONG_DATASET_MAP e DATASET_NAME_MAP — tres
# nomes para o mesmo dicionario.
DATASET_LONG_TO_SHORT: dict[str, str] = {
    "helminth-eggs":   "eggs",
    "helminth-larvae": "larvae",
    "protozoan-cysts": "protozoan",
}

# Toda grafia que aparece em nome de arquivo -> curto. E a uniao de
# DATASET_LONG_TO_SHORT com a identidade das chaves curtas (core.DATASETS), mais
# duas grafias de fora: "cistos" e o nome usado nos CSVs externos de
# data/reports_felipe/; "parasito" e o dataset agregado. Tambem escrito por
# extenso — mesma ordem de chaves que o merge original produzia.
DATASET_ALIASES: dict[str, str] = {
    "helminth-eggs":   "eggs",
    "helminth-larvae": "larvae",
    "protozoan-cysts": "protozoan",
    "eggs":      "eggs",
    "larvae":    "larvae",
    "protozoan": "protozoan",
    "cistos":    "protozoan",
    "parasito":  "parasito",
}

# ─── Caminhos ─────────────────────────────────────────────────────────────────
# Derivados da raiz do repositorio (core.constants.PROJECT_ROOT, importado
# aqui como _PROJECT_ROOT). f-string no lugar de os.path.join porque um arquivo
# de constantes nao chama funcao — o valor resolvido e identico ao de hoje.

RESULTS_DIR: str = f"{_PROJECT_ROOT}/results"

# configs/evaluate/mlp. A base `configs/` tem um consumidor externo so
# (retry_protozoan_experiment.py) e nao virou constante deste pacote.
MLP_CONFIGS_DIR: str = f"{_PROJECT_ROOT}/configs/generated/mlp"

# Splits incrementais consumidos pelos launchers.
DATASET_SPLITS_ROOT: str = f"{_PROJECT_ROOT}/data/to_modules/new_split_parasito"

# CSVs do pipeline externo, nao versionado.
REPORTS_FELIPE_DIR: str = f"{_PROJECT_ROOT}/data/reports_felipe"
REPORTS_FELIPE_SVM_DIR: str = f"{REPORTS_FELIPE_DIR}/svm"

# artifacts/distillation nao esta aqui: tem consumidor fora deste pacote e mora
# em core.constants (ARTIFACTS_DISTILLATION_DIR).
ARTIFACTS_NORMALIZED_DIR: str = f"{_PROJECT_ROOT}/artifacts/normalized"
ARTIFACTS_PLOTS_DIR: str = f"{_PROJECT_ROOT}/artifacts/plots"
ARTIFACTS_CLASSIFICATION_FLIM_DIR: str = f"{_PROJECT_ROOT}/artifacts/classification_flim"

UNIFIED_SVM_COMPARISON_CSV: str = f"{ARTIFACTS_NORMALIZED_DIR}/unified_svm_comparison.csv"

# ─── Nomes de arquivo e subdiretorio ──────────────────────────────────────────

SPLITS_INCREMENTAL_SUBDIR: str = "splits_incremental"
WRITE_TEST_FILENAME: str = ".write_test"

# ─── W&B ──────────────────────────────────────────────────────────────────────

WANDB_ENTITY: str = "ophira-ai"
WANDB_PROJECT: str = "flim-ssl"

# frozenset(...) e a UNICA chamada de funcao deste arquivo: Python nao tem
# literal de frozenset, e trocar por `{...}` mudaria o tipo de uma constante
# compartilhada para um set mutavel. Preservar o valor venceu a regra do plano.
WANDB_FAILED_STATES: frozenset[str] = frozenset({"crashed", "failed", "killed"})

# Env dos subprocessos de treino: console mudo, logging online.
WANDB_CHILD_ENV: dict[str, str] = {"WANDB_CONSOLE": "off", "WANDB_MODE": "online"}

# ─── Ray e subprocessos ───────────────────────────────────────────────────────
# O par OMP_NUM_THREADS (OMP_ENV_VAR) tambem e lido fora deste pacote e mora em
# core.constants.

CUDA_ENV_VAR: str = "CUDA_VISIBLE_DEVICES"

# Ray nao gerencia GPU: o pinning e feito na mao via CUDA_VISIBLE_DEVICES.
RAY_INIT_KWARGS: dict = {"ignore_reinit_error": True, "log_to_driver": True, "num_gpus": 0}

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

# ─── Schema dos CSVs agregados ────────────────────────────────────────────────
# A lista de metricas em si (METRICS) e lida por mais de um pacote e mora em
# core.constants.

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

# O que sobrevive quando o teacher e removido do .ckpt. O prefixo do proprio
# teacher (TEACHER_PREFIX) tambem e lido fora deste pacote e mora em
# core.constants.
KEEP_STUDENT_PREFIXES: tuple[str, ...] = ("student.", "proj_kd.")
