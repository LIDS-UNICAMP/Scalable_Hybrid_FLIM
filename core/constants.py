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

"""
Constantes globais do projeto: os nomes consumidos por 2+ pacotes de topo.

Fase 1 do refactor estrutural. Tudo aqui foi MOVIDO (copiado) das quatro fontes
historicas, sem apagar a origem e sem religar consumidor nenhum:

    config.py                    -> registry de caminhos de dataset
    scripts/constants.py         -> grade experimental, caminhos, pesos FLIM
    src/evaluate/constants.py    -> tamanho de entrada e classes por dataset
    src/utils/constant.py        -> (orfao, nada veio de la)

Cada bloco abaixo cita a origem em file:line. Valores e nomes sao os mesmos da
origem; as unicas excecoes, todas deliberadas e registradas no relatorio:

  * o registry-dict de config.py:64 entra aqui como DATASET_REGISTRY, porque
    DATASETS ja e o nome da lista curta de scripts/constants.py:81;
  * FLIM_ARCH_BASE e FLIM_WEIGHTS_BASE eram compreensoes sobre ARCH_TAG e
    _FLIM_SUFFIX, que NAO sao globais e ficaram na origem; aqui viraram dicts
    literais com exatamente os mesmos valores.

Este arquivo e PLANO de proposito: so nomes e dados. Nada de classe, nada de
funcao, nada de variavel de ambiente, nada de leitura de disco e nenhuma
dependencia de pacote do projeto. A unica coisa importada aqui e pathlib
(stdlib), que refaz o mesmo computo de raiz do repositorio que as origens
faziam com os.path.
"""

from pathlib import Path

# --- Raiz do repositorio ----------------------------------------------------

# Origem: scripts/constants.py:76 (e config.py:52, la privado como
# _PROJECT_ROOT). Sao dois computos diferentes para o mesmo valor; aqui vira um
# so. core/constants.py -> core -> raiz, mesma profundidade de scripts/.
PROJECT_ROOT: str = str(Path(__file__).resolve().parents[1])

# Alias interno em forma de Path, so para montar os caminhos derivados abaixo
# sem repetir a conversao. Nao e constante publica: nao importe este nome.
_ROOT = Path(PROJECT_ROOT)

# --- Datasets ---------------------------------------------------------------

# Chave curta, a forma canonica do projeto.
# Origem: scripts/constants.py:81 == src/evaluate/constants.py:69 (identicos).
DATASETS: list[str] = ["eggs", "larvae", "protozoan"]

# Numero de classes por chave curta. Origem: scripts/constants.py:106
# (copia identica vive em src/modules/autoencoder_flim_module.py:850).
NUM_CLASSES: dict[str, int] = {"eggs": 9, "larvae": 2, "protozoan": 7}

# Numero de classes por chave longa, mais o agregado "parasito".
# Origem: src/evaluate/constants.py:54 (copia identica e orfa em
# src/utils/constant.py:29).
DATASET_NUM_CLASSES: dict[str, int] = {
    "helminth-eggs": 9,
    "helminth-larvae": 2,
    "protozoan-cysts": 7,
    "parasito": 9,
}

# Registry de caminhos de dataset. Origem: config.py:64, onde o nome era
# DATASETS e os caminhos saiam do helper _abs (config.py:55). Renomeado para
# DATASET_REGISTRY por anti-shadowing: convive com a lista DATASETS acima.
DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "to_modules": {
        "images": str(_ROOT / "data/to_modules/images"),
        "splits": str(_ROOT / "data/to_modules/splits"),
    },
    "parasito": {
        "classes": [
            {
                "name": "helminth-eggs_split_2",
                "images": str(_ROOT / "data/to_modules/new_split_parasito/helminth-eggs/images"),
                "masks": str(_ROOT / "data/to_modules/new_split_parasito/helminth-eggs/masks"),
                "splits": str(_ROOT / "data/to_modules/new_split_parasito/helminth-eggs/splits"),
                "splits_incremental": str(_ROOT / "data/to_modules/new_split_parasito/helminth-eggs/splits_incremental"),
            },
            {
                "name": "helminth-larvae_split_2",
                "images": str(_ROOT / "data/to_modules/new_split_parasito/helminth-larvae/images"),
                "masks": str(_ROOT / "data/to_modules/new_split_parasito/helminth-larvae/masks"),
                "splits": str(_ROOT / "data/to_modules/new_split_parasito/helminth-larvae/splits"),
                "splits_incremental": str(_ROOT / "data/to_modules/new_split_parasito/helminth-larvae/splits_incremental"),
            },
            {
                "name": "protozoan-cysts_split_2",
                "images": str(_ROOT / "data/to_modules/new_split_parasito/protozoan-cysts/images"),
                "masks": str(_ROOT / "data/to_modules/new_split_parasito/protozoan-cysts/masks"),
                "splits": str(_ROOT / "data/to_modules/new_split_parasito/protozoan-cysts/splits"),
                "splits_incremental": str(_ROOT / "data/to_modules/new_split_parasito/protozoan-cysts/splits_incremental"),
            },
        ],
    },
}

# --- Grade experimental -----------------------------------------------------

# Origem: scripts/constants.py:118 == src/evaluate/constants.py:61.
SPLITS: list[int] = [1, 2, 3]

# Grafado PCTS em alguns call sites; PERCENTAGES e o nome canonico.
# Origem: scripts/constants.py:119 == src/evaluate/constants.py:64.
PERCENTAGES: list[int] = [1, 5, 25, 50, 75, 100]

# Origem: scripts/constants.py:275 == src/evaluate/constants.py:71.
METRICS: list[str] = ["kappa", "acc", "f1"]

# Tamanho de entrada do student (FLIM). Origem: src/evaluate/constants.py:51.
# NAO e a mesma grandeza que IJEPAEncoder.IMAGE_SIZE = 224
# (src/models/ijepa_encoder.py:246), que e a entrada do teacher, nem que o
# IMAGE_SIZE = 224 de scripts/spifil_resnet_graft.py:96. Os 224 ficaram onde
# estao, de proposito.
IMAGE_SIZE: int = 200

# --- Caminhos ---------------------------------------------------------------

# Origem: scripts/constants.py:129 (la era CONFIGS_DIR / "default.yaml";
# CONFIGS_DIR:126 nao e global e ficou na origem).
DEFAULT_CONFIG_YAML: str = str(_ROOT / "configs/default.yaml")

# Origem: scripts/constants.py:141 (la era ARTIFACTS_DIR / "distillation";
# ARTIFACTS_DIR:125 nao e global e ficou na origem).
ARTIFACTS_DISTILLATION_DIR: str = str(_ROOT / "artifacts/distillation")

# --- Pesos FLIM entregues ---------------------------------------------------

# A tag de canais faz parte do nome do diretorio. protozoan diverge: 30 canais
# na arquitetura, 32 nos pesos. Os dois dicts abaixo NAO sao redundantes.
# Origem: scripts/constants.py:160 e :164, la montados por compreensao sobre
# FLIM_MODEL_ROOT:149, ARCH_TAG:153 e _FLIM_SUFFIX:158 — nenhum dos tres e
# global, entao aqui os caminhos estao escritos por extenso, com os mesmos
# valores.
FLIM_ARCH_BASE: dict[str, str] = {
    "eggs": str(_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5/eggs"),
    "larvae": str(_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5/larvae"),
    "protozoan": str(_ROOT / "data/to_mateus/model/ch24_30_48_a0.5_f5/protozoan"),
}
FLIM_WEIGHTS_BASE: dict[str, str] = {
    "eggs": str(_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5/eggs"),
    "larvae": str(_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5/larvae"),
    "protozoan": str(_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5/protozoan"),
}

# --- Nomes de arquivo e subdiretorio ----------------------------------------

ARCH_JSON_FILENAME: str = "architecture.json"       # scripts/constants.py:171
MODELS_SUBDIR: str = "models"                       # scripts/constants.py:172
CHECKPOINTS_SUBDIR: str = "checkpoints"             # scripts/constants.py:173
RUN_METADATA_FILENAME: str = "run_metadata.json"    # scripts/constants.py:175

# O que e removido do .ckpt quando o teacher e descartado.
# Origem: scripts/constants.py:303.
TEACHER_PREFIX: str = "teacher."

# --- Execucao ---------------------------------------------------------------

OMP_ENV_VAR: str = "OMP_NUM_THREADS"                # scripts/constants.py:217
DEFAULT_CPUS_PER_EXPERIMENT: int = 4                # scripts/constants.py:222

# --- Crescimento SPiFiL -----------------------------------------------------

# Dentro disso e empate, nao ganho. Origem: scripts/constants.py:258, que era
# espelho deliberado de src/modules/autoencoder_flim_module.py:137 — a
# duplicacao existia para o lancador nao pagar import de torch. Este arquivo e
# livre de torch por construcao, entao a razao do espelho deixou de valer.
KAPPA_TOLERANCE: float = 0.01

# --- Vocabulario de dataset -------------------------------------------------

# O nome longo que o loader de imagem exige: ele casa "helminth-eggs", nao
# "eggs". Era `_dataset_short_to_parasite_name`, replicado em 8 arquivos
# (src/modules/autoencoder_flim_module.py:854 e vizinhos). Nao e do SPiFiL nem
# de um metodo: e o vocabulario do projeto, e mora ao lado de NUM_CLASSES.
PARASITE_NAME: dict[str, str] = {
    "eggs": "helminth-eggs_split_2",
    "larvae": "helminth-larvae_split_2",
    "protozoan": "protozoan-cysts_split_2",
}
