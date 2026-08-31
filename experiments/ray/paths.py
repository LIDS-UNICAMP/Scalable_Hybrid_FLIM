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

"""paths.py — resolucao de caminho do launcher Ray.

Todo caminho que uma celula da grade consome sai daqui. Ha exatamente DUAS
f-strings de config no pacote e as duas moram neste arquivo::

    configs/dataset/{dataset}/split_{split}.yaml
    configs/model/{method}/{init}/{dataset}/split_{split}.yaml

Nao ha dict de excecao, nao ha config generico de fallback: se o arquivo nao
existe, a resolucao FALHA com o caminho absoluto esperado impresso. O caso
especial de protozoan de `scripts/run_ssl_ray.py:113-121` — que caia num
`lejepa_line_{init}.yaml` generico para eggs/larvae e num per-split para
protozoan — morre aqui: cada celula tem o proprio arquivo, ou nao roda.

O vocabulario de dataset da entrada e o LONGO (`helminth-eggs`), que e o do
`grid.datasets` do schema; a traducao para a chave curta que
`core.constants`/`experiments.constants` usam nos dicts de caminho fica em
`_short`, um lugar so.

Absorve, das 4 copias em `scripts/{distillation,distillation_conv,autoencoder_flim,
classification_flim}_ray.py`: `_arch_json`, `_flim_weights_path`, `_split_json`
e `_run_name`. Herda tambem `train_dir` / `split_dir` /
`data_descriptor_filename`, que eram `def` em `scripts/constants.py:184-196` e
por isso nao couberam no `experiments/constants.py` (arquivo plano, so dados).
"""

from __future__ import annotations

import os

from core.constants import (
    ARCH_JSON_FILENAME,
    FLIM_ARCH_BASE,
    FLIM_WEIGHTS_BASE,
    MODELS_SUBDIR,
    PROJECT_ROOT,
)
from experiments.constants import (
    DATASET_LONG_TO_SHORT,
    DATASET_SPLITS_ROOT,
    PARASITE_DIR,
    SPLITS_INCREMENTAL_SUBDIR,
)

# ── Subpastas e nomes de arquivo ──────────────────────────────────────────────
# VERBATIM de scripts/constants.py:184-196.


def train_dir(split: int) -> str:
    """Subpasta do split dentro dos pesos FLIM: ``train1``, ``train2``, ..."""
    return f"train{split}"


def split_dir(split: int) -> str:
    """Subpasta do split dentro de ``splits_incremental``: ``split1``, ..."""
    return f"split{split}"


def data_descriptor_filename(pct: int) -> str:
    """Descritor do percentual: ``data_descriptor_perc75.json``."""
    return f"data_descriptor_perc{pct}.json"


# ── As duas f-strings de config ───────────────────────────────────────────────


def dataset_config(dataset: str, split: int) -> str:
    """Config de dataset da celula, relativa a raiz do repo."""
    return _require(f"configs/dataset/{dataset}/split_{split}.yaml")


def model_config(method: str, init: str, dataset: str, split: int) -> str:
    """Config de modelo da celula, relativa a raiz do repo."""
    return _require(f"configs/model/{method}/{init}/{dataset}/split_{split}.yaml")


def _require(relative: str) -> str:
    """Devolve o caminho relativo; levanta com o absoluto se nao houver arquivo.

    O retorno e relativo porque e assim que o filho recebe (`--config <rel>`,
    com cwd na raiz); o erro mostra o absoluto porque e o que se cola num `ls`.
    """
    absolute = os.path.join(PROJECT_ROOT, relative)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"config ausente, esperado em: {absolute}")
    return relative


# ── Pesos e dados FLIM ────────────────────────────────────────────────────────


def arch_json(dataset: str, split: int) -> str:
    """architecture.json do encoder FLIM. Sempre de FLIM_ARCH_BASE."""
    return os.path.join(
        FLIM_ARCH_BASE[_short(dataset)], train_dir(split), ARCH_JSON_FILENAME
    )


def flim_weights_path(dataset: str, split: int) -> str:
    """Diretorio dos kernels/bias FLIM. Sempre de FLIM_WEIGHTS_BASE."""
    return os.path.join(
        FLIM_WEIGHTS_BASE[_short(dataset)], train_dir(split), MODELS_SUBDIR
    )


def split_json(dataset: str, split: int, pct: int) -> str:
    """Descritor incremental de dados do par (split, percentual)."""
    return os.path.join(
        DATASET_SPLITS_ROOT,
        PARASITE_DIR[_short(dataset)],
        SPLITS_INCREMENTAL_SUBDIR,
        split_dir(split),
        data_descriptor_filename(pct),
    )


# ── Identidade do run ─────────────────────────────────────────────────────────

# Marcadores de identidade do autoencoder, VERBATIM de
# scripts/autoencoder_flim_ray.py:288-301 — o comentario de la explica por que
# existem: os 36 runs ja gravados usaram `avgpool2d` + Normalize da ImageNet e
# ficaram com o nome pelado. Se `_lab` ou `_flat` sumirem, a variante nova cai no
# MESMO diretorio de checkpoint da antiga e o stage2 carrega o ckpt do outro braco.
_LAB_MARKER: str = "_lab"
_EMBED_MARKER: dict[str, str] = {"avgpool2d": "", "flatten": "_flat"}
_STAGE_SUFFIX: dict[int, str] = {1: "stage1_frozen", 2: "stage2_fine_tune"}

# scripts/constants.py:248, copiado como literal: `scripts/` nao esta no import
# path deste pacote.
_GROWTH_PREFIX: str = "spifil_growth"

# Eixos que entram no nome e NAO sao (dataset, split, pct, init). A lista e
# fechada para que um erro de digitacao ("embedmode") vire ValueError em vez de
# um nome silenciosamente diferente — que e exatamente o link perdido com os
# pesos que esta funcao existe para evitar.
_VARIANT_KEYS: frozenset = frozenset({
    "prefix",            # --run-prefix dos launchers antigos
    "suffix",            # cauda de CLI do SSL: _bs150, _mc8g8l, _3lproj
    "stage",             # autoencoder: 1 | 2 ; growth: nome do estagio
    "imagenet_norm",     # autoencoder: False acrescenta _lab
    "embed_mode",        # autoencoder: avgpool2d | flatten (_flat)
    "output_relu",       # classification: relu2l
    "output_softplus",   # classification: softplus2l
    "freeze_encoder",    # classification: _frozen
    "dist_type",         # distillation: direct | hybrid
    "proj_type",         # distillation_conv: as 9 cabecas de projecao
    "use_flim_init",     # distillation_conv: _flim_init
    "no_imagenet_norm",  # distillation_conv: _no_imagenet_norm
})


def run_name(
    name: str,
    dataset: str,
    split: int,
    pct: int,
    init: str,
    *,
    method: str,
    runner: str = "train",
    **variant: object,
) -> str:
    """Nome do run: a formula HISTORICA de cada familia, reproduzida byte a byte.

    ESTE E O UNICO PONTO DO PACOTE QUE NAO FOI UNIFORMIZADO, E E DE PROPOSITO.
    `run_name` resolve duas coisas ao mesmo tempo: o diretorio de artefatos
    (`<work_dir>/<run_name>/checkpoints/`) e o display name no W&B. Uma formula
    nova, por mais uniforme que fosse, deixaria TODO checkpoint ja gravado
    inalcancavel pelo motor novo e faria `skip` reportar a grade inteira como
    pendente. O vocabulario do YAML (`gpu_ids`, `percentages`, `skip`) continua
    canonico; aqui quem manda e o historico. Nao "conserte" isto de volta para
    uma f-string so.

    Isso tambem revoga, so aqui, a promessa de `schema.py:207` ("name vira
    prefixo de run_name"): a base do nome pertence a familia, nao ao YAML. Quem
    quiser prefixo passa `prefix=` (o antigo `--run-prefix`). A excecao e o
    runner `growth`, cuja formula historica (scripts/spifil_growth_loop.py:209)
    usa mesmo uma etiqueta livre — la o `name` entra.

    Despacho: `runner == "growth"` ganha de tudo (o protocolo SPiFiL nomeia por
    estagio, seja qual for o metodo); no resto, despacha por `method`.

    `variant` traz os eixos que entram no nome e nao sao (dataset, split, pct,
    init) — estagio, normalizacao, cabeca de projecao, prefixo. Os defaults
    reproduzem o nome pelado que ja esta em disco.

    `method` e obrigatorio de proposito, sem default: um default silencioso
    daria o nome de OUTRA familia a quem esquecesse de passar, e um nome de
    familia errada aponta para o diretorio de checkpoint errado — que e a perda
    de link com os pesos que este arquivo existe para impedir. Melhor TypeError
    na primeira chamada.
    """
    unknown = sorted(set(variant) - _VARIANT_KEYS)
    if unknown:
        raise ValueError(
            f"run_name: variante desconhecida {unknown}; "
            f"validas: {sorted(_VARIANT_KEYS)}"
        )

    if runner == "growth":
        body = _name_growth(name, dataset, split, pct, variant)
    elif method in _FORMULAS:
        body = _FORMULAS[method](dataset, split, pct, init, variant)
    else:
        raise ValueError(
            f"run_name: metodo {method!r} nao tem formula historica; "
            f"conhecidos: {sorted(_FORMULAS)}"
        )
    return f"{variant.get('prefix', '')}{body}{variant.get('suffix', '')}"


def _name_lejepa(dataset: str, split: int, pct: int, init: str, variant: dict) -> str:
    """SSL. Fonte: check_experiments/_common.py:66-70, usado por run_ssl_ray.py:291.

    Unica das seis com o dataset LONGO e com underscore antes de split/pct/model.
    E a grafia que `core/wandb.py:51-53` casa — nao encostar.
    """
    return f"lejepa_line_{dataset}_split_{split}_pct_{pct}_model_{init}"


def _name_autoencoder(dataset: str, split: int, pct: int, init: str, variant: dict) -> str:
    """Autoencoder FLIM. Fonte: scripts/autoencoder_flim_ray.py:303-309.

    O `init` nao entra: a familia inteira e FLIM. O que entra e o estagio e os
    dois marcadores de identidade (`_lab`, `_flat`).
    """
    return (
        f"ae_resnet_flim_{_short(dataset)}_split{split}_pct{pct}"
        f"_{_STAGE_SUFFIX[variant.get('stage', 1)]}"
        f"{'' if variant.get('imagenet_norm', True) else _LAB_MARKER}"
        f"{_EMBED_MARKER[variant.get('embed_mode', 'avgpool2d')]}"
    )


def _name_classification(dataset: str, split: int, pct: int, init: str, variant: dict) -> str:
    """Cabeca de classificacao. Fonte: scripts/classification_flim_ray.py:220-232."""
    if variant.get("output_softplus"):
        head_tag = "softplus2l"
    elif variant.get("output_relu"):
        head_tag = "relu2l"
    else:
        head_tag = "sigmoid2l"
    body = f"classhead_{_short(dataset)}_split{split}_pct{pct}_{head_tag}"
    if variant.get("freeze_encoder"):
        body += "_frozen"
    return body


def _name_distillation(dataset: str, split: int, pct: int, init: str, variant: dict) -> str:
    """Destilacao MLP. Fonte: scripts/distillation_ray.py:141.

    A cauda `model...` e o distillation_type, NAO o init — apesar da grafia
    identica a do eixo `init` das outras grades.
    """
    dist_type = variant.get("dist_type", "direct")
    return f"distillation_{_short(dataset)}_split{split}_pct{pct}_model{dist_type}"


def _name_distillation_conv(dataset: str, split: int, pct: int, init: str, variant: dict) -> str:
    """Destilacao convolucional. Fonte: scripts/distillation_conv_ray.py:329-368.

    Os nove `proj_type` VERBATIM, inclusive as duas saidas antecipadas: as
    cabecas `*_frozen` forcam a divisao de normalizacao dentro do filho, entao o
    nome delas nunca leva `_no_imagenet_norm`.
    """
    short = _short(dataset)
    dist_type = variant.get("dist_type", "direct")
    proj_type = variant.get("proj_type", "conv_next_layers")
    suffix = "_flim_init" if variant.get("use_flim_init") else ""

    if proj_type == "3x3_bn2d_1280":
        body = f"distillation_{short}_split{split}_pct{pct}_3x3_BN2d_1280_one_layer{suffix}"
    elif proj_type == "1x1_bn2d_1280":
        body = f"distillation_{short}_split{split}_pct{pct}_1x1_BN2d_1280_one_layer{suffix}"
    elif proj_type == "2l_1x1_bn2d_256_1280":
        body = f"distillation_{short}_split{split}_pct{pct}_2l_1x1_BN2d_256_1280{suffix}"
    elif proj_type == "2l_1x1_init_flim_256_1280":
        body = f"distillation_{short}_split{split}_pct{pct}_2l_1x1_init_flim_256_1280"
    elif proj_type == "3x3_init_flim_1280":
        body = f"distillation_{short}_split{split}_pct{pct}_3x3_BN2d_1280_one_layer_init_flim"
    elif proj_type == "next_layers_init_flim":
        body = f"distillation_{short}_split{split}_pct{pct}_next_layers_init_flim_{dist_type}"
    elif proj_type == "1x1_init_flim_frozen":
        return f"distillation_{short}_split{split}_pct{pct}_1x1_BN2d_1280_flim_frozen"
    elif proj_type == "2l_1x1_init_flim_frozen_256_1280":
        return f"distillation_{short}_split{split}_pct{pct}_2l_1x1_init_flim_256_1280_flim_frozen"
    else:
        body = f"distillation_{short}_split{split}_pct{pct}_next_layers_{dist_type}{suffix}"

    if variant.get("no_imagenet_norm"):
        body += "_no_imagenet_norm"
    return body


def _name_growth(name: str, dataset: str, split: int, pct: int, variant: dict) -> str:
    """Crescimento SPiFiL. Fonte: scripts/spifil_growth_loop.py:209 (+164-168, +116).

    Aqui `name` faz o papel do `_exp(args)` historico, que era o basename do
    `--work-dir`. Sem essa etiqueta, dois lacos com flags diferentes sobem com
    nome de run identico no board.
    """
    stage = variant.get("stage", "stage1")
    # `name` historico era o BASENAME do --work-dir (ex.: "grid4" em
    # artifacts/spifil_growth/grid4). Um experiment YAML que ja chame a corrida
    # de "spifil_growth_grid4" duplicaria o prefixo e apontaria para um
    # diretorio de checkpoint que nao existe. Guarda contra isso.
    tag = name[len(_GROWTH_PREFIX) + 1:] if name.startswith(_GROWTH_PREFIX + "_") else name
    return f"{_GROWTH_PREFIX}_{tag}_{_short(dataset)}_split{split}_pct{pct}_{stage}"


# Uma entrada por familia historica. Nao e registry: e um `if` escrito como dict,
# sem registro dinamico e sem ponto de extensao.
_FORMULAS: dict = {
    "lejepa": _name_lejepa,
    "autoencoder": _name_autoencoder,
    "classification": _name_classification,
    "distillation": _name_distillation,
    "distillation_conv": _name_distillation_conv,
}


def _short(dataset: str) -> str:
    """Nome longo do schema -> chave curta dos dicts de caminho."""
    return DATASET_LONG_TO_SHORT[dataset]
