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

"""preflight.py — o pre-voo do experiment YAML, antes de qualquer job.

    python -m experiments.ray.preflight <caminho.yaml> [raiz]

Roda OBRIGATORIAMENTE em `experiments/ray/launch.py:201-221`, antes de
`ray.init()` e antes dos imports pesados. Expande a grade e confere que TUDO
existe em disco; se algo falta, imprime o relatorio e levanta `SystemExit(1)`.
Nenhum job e submetido.

## READ-ONLY

O pre-voo so LE YAML, faz `stat` de caminho e importa modulo. Nao escreve, nao
move, nao mata processo, nao toca em `work_dir` — ha experimento de outra gente
rodando na maquina. Quem cria e testa a escrita do `work_dir` e
`grid.validate_output_dir`, e isso e do launcher, nao daqui.

## As duas metades do relatorio

O requisito e literal: ele **nomeia o arquivo culpado** (o YAML que quebrou, com
a checagem violada) **e lista os que faltam** — um caminho esperado por linha,
TODOS, sem `...` e sem truncar. "24 configs faltando" sem os 24 caminhos nao
serve para nada: o que se cola num `ls` e a linha, nao a contagem.

## Divergencia entre a spec e o schema REAL

A spec do pre-voo (spec_refactor.md:2233-2262) descreve um schema com `axes:`,
`path:`, `map:` e `exclude:`. Nao e o schema que este repositorio implementa.
O real esta em `experiments/ray/schema.py`:

    `axes:`    -> `grid:` com QUATRO eixos fechados, nao um dict aberto
                  (schema.py:172-190, `Grid`: init/datasets/splits/percentages).
    `path:`    -> nao existe no YAML. Os dois templates de caminho sao codigo,
                  em `paths.dataset_config` / `paths.model_config`
                  (paths.py:85-92), e sao as UNICAS duas f-strings de config do
                  pacote.
    `map:`     -> nao existe. O unico eixo que vira flag do LightningCLI e
                  `percentage`, e quem o injeta e o runner
                  (`experiments/ray/runners/train.py:91`). `overrides:` do YAML
                  e um dict PLANO de flags literais (schema.py:317-324), sem
                  referencia a eixo.
    `exclude:` -> nao existe, e um YAML que a traga e recusado antes de chegar
                  aqui por `_check_keys("topo", ...)` (schema.py:283).

Este arquivo se conforma ao schema REAL. As 16 checagens mantem a numeracao da
spec; as que nao tinham contrapartida literal foram reancoradas no que o schema
de fato tem, e cada uma diz no proprio nome onde foi parar.

Por nao duplicar os templates: os dois caminhos sao recuperados chamando as
proprias funcoes de `paths` (`_template`), inclusive com valores-sentinela para
extrair o template cru da checagem 7. Nao ha uma terceira f-string de config
neste arquivo, e nao deve haver.

## Sobreposicao com o schema e deliberada

`schema.load` levanta no PRIMEIRO problema; o pre-voo tem que listar TODOS de uma
vez. Por isso as checagens 1-9 leem o dict cru do YAML e reportam em vez de
levantar. Nao e uma segunda fonte de verdade: os valores validos continuam vindo
de `schema.py` (`RUNNERS`, `INITS`, `DATASETS`, `SPLITS`, `PERCENTAGES`,
`RUNNER_ARGS`, `METHODS_DIR`).

## Custo: a checagem 15

E a unica que importa modulo de verdade, e importar modulo de modelo puxa a
arvore inteira de dependencias. As tres mitigacoes da spec estao todas aqui:
dedup por `class_path`, import do MODULO com `hasattr` (nunca instanciar, que
exigiria peso e GPU) e cache do resultado por `class_path` na corrida
(`_IMPORT_CACHE`). Se um modulo alocar CUDA no proprio import, isso e ACHADO do
pre-voo e vira issue — nao vira excecao na checagem.
"""

from __future__ import annotations

import importlib
import itertools
import os
import string
import sys
from functools import lru_cache

import yaml

from core.constants import DEFAULT_CONFIG_YAML, PROJECT_ROOT
from experiments.constants import SEP_WIDTH
from experiments.ray import paths
from experiments.ray.schema import (
    DATASETS,
    INITS,
    METHODS_DIR,
    PERCENTAGES,
    RUNNER_ARGS,
    RUNNERS,
    SPLITS,
)
# Privado de proposito: e a MESMA listagem que a mensagem de erro do schema usa
# (schema.py:129-137). Reescrever o `os.listdir` aqui daria duas listas de
# metodos existentes que podem discordar.
from experiments.ray.schema import _existing_methods

# Nome de cada eixo do YAML e o conjunto de valores validos. Vem do `Grid` do
# schema (schema.py:176-190); a chave do YAML e plural, a do ponto e singular.
_AXIS_VALUES: dict[str, tuple] = {
    "init": INITS,
    "datasets": DATASETS,
    "splits": SPLITS,
    "percentages": PERCENTAGES,
}
_AXIS_TO_POINT: dict[str, str] = {
    "init": "init",
    "datasets": "dataset",
    "splits": "split",
    "percentages": "percentage",
}

# O unico eixo que vira flag do LightningCLI. Literal de
# experiments/ray/runners/train.py:91 — e o `map:` da spec, com uma entrada so.
_PERCENTAGE_FLAG: str = "data.init_args.percentage"

# Chaves de init_args cujo valor e caminho em disco (checagem 16).
_PATH_KEYS: tuple[str, ...] = ("arch_json", "flim_weights_path")

# Categoria de cada checagem, na ordem em que sai o relatorio.
_CATEGORIES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("ESTRUTURA", (1, 2, 3, 4, 5)),
    ("GRADE", (6, 7, 8, 9, 10, 11)),
    ("DISCO", (12, 13, 14, 15, 16)),
)

# Checagens que nao rodam quando alguem marcou `blocks=True`: sem `method`, ou
# com eixo cujo valor nao e do vocabulario, todo caminho resolvido sairia lixo e
# o relatorio viraria centenas de erros derivados de um so. Elas sao reportadas
# como PULADAS, nunca como ok.
_BLOCKED: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16)

# class_path -> "" (importa) ou a mensagem do erro. Vive a corrida inteira.
_IMPORT_CACHE: dict[str, str] = {}


def _err(check: int, title: str, *lines: str, blocks: bool = False) -> dict:
    """Um erro: a checagem violada, o titulo do bloco e as linhas dele.

    `blocks=True` diz que este erro deixa a grade sem sentido — as checagens que
    dependem dela saem PULADAS em vez de `ok`.
    """
    return {"check": check, "title": title, "lines": list(lines), "blocks": blocks}


@lru_cache(maxsize=None)
def _read_yaml(absolute: str):
    """Le um YAML uma vez por corrida. Devolve o objeto, ou a excecao como valor."""
    try:
        with open(absolute, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:  # YAML quebrado, permissao, encoding
        return exc


def _template(fn, *args) -> str:
    """Caminho relativo esperado, exista ou nao o arquivo.

    `paths.dataset_config` / `paths.model_config` LEVANTAM quando o arquivo nao
    existe (paths.py:95-104) — que e exatamente o caso que o pre-voo precisa
    reportar. A mensagem carrega o caminho absoluto esperado; aqui ele volta a
    ser relativo. Assim o pre-voo nao repete as duas f-strings de config.
    """
    try:
        return fn(*args)
    except FileNotFoundError as exc:
        absolute = str(exc).split("esperado em: ")[-1]
        return os.path.relpath(absolute, PROJECT_ROOT)


def _root(root: str) -> str:
    """A raiz do repo usada nos `stat`. `.` e a raiz que o pacote ja calculou."""
    return PROJECT_ROOT if root == "." else os.path.abspath(root)


# ─── expand / resolve ─────────────────────────────────────────────────────────


def expand(exp: dict) -> list[dict]:
    """Produto cartesiano de `grid`, na MESMA ordem de `grid.build`.

    Eixo ausente ou `null` cai no default do eixo, que e o conjunto de valores
    validos declarado no schema — a regra de `grid.py:76-81`, nao uma segunda.

    Nao ha `exclude`: o schema real nao tem a chave e recusa o YAML que a traga
    (schema.py:283). Buraco declarado de grade, hoje, e eixo mais curto.
    """
    grid = exp.get("grid") or {}
    return [
        {"dataset": dataset, "split": split, "percentage": pct, "init": init}
        for dataset, split, pct, init in itertools.product(
            grid.get("datasets") or list(DATASETS),
            grid.get("splits") or list(SPLITS),
            grid.get("percentages") or list(PERCENTAGES),
            grid.get("init") or list(INITS),
        )
    ]


def resolve(exp: dict, point: dict) -> dict:
    """O ponto + os dois configs resolvidos + os overrides do LightningCLI.

    O ponto original fica INTACTO no dict devolvido: a checagem 10 e o relatorio
    dependem de saber de qual ponto da grade cada caminho veio.
    """
    return {
        **point,
        "dataset_config": _template(
            paths.dataset_config, point["dataset"], point["split"]
        ),
        "model_config": _template(
            paths.model_config,
            exp.get("method"),
            point["init"],
            point["dataset"],
            point["split"],
        ),
        # `overrides:` do YAML e plano e igual para todo ponto; o que varia com o
        # ponto e a unica flag derivada de eixo (runners/train.py:91).
        "overrides": {
            **(exp.get("overrides") or {}),
            _PERCENTAGE_FLAG: point["percentage"],
        },
    }


# ─── ESTRUTURA (1-5) ──────────────────────────────────────────────────────────


def check_01_required(exp: dict) -> list[dict]:
    """1. Campos obrigatorios presentes.

    A lista e a do schema real (schema.py:288-292), nao a da spec: `path` e
    `axes` nao existem; `resources` e `output` sim, e sao obrigatorios.
    """
    required = ("name", "method", "runner", "resources", "output")
    missing = [key for key in required if exp.get(key) is None]
    if not missing:
        return []
    return [_err(
        1, "chave obrigatoria ausente no topo",
        f"faltam: {missing}",
        f"obrigatorias: {list(required)}  (schema.py:288-292)",
        blocks=True,
    )]


def check_02_runner(exp: dict) -> list[dict]:
    """2. runner in {train, growth, eval}."""
    runner = exp.get("runner")
    if runner is None or runner in RUNNERS:
        return []
    return [_err(
        2, "runner invalido",
        f"runner: {runner!r}",
        f"validos: {list(RUNNERS)}",
    )]


def check_03_method(exp: dict) -> list[dict]:
    """3. methods/<method>/ existe e e diretorio; se nao, LISTAR os que existem.

    A listagem nao e enfeite: o caso real e typo (`lejepaa`), e o conserto esta
    na linha de baixo do erro.
    """
    method = exp.get("method")
    if method is None:
        return []
    if isinstance(method, str) and os.path.isdir(os.path.join(METHODS_DIR, method)):
        return []
    existem = "  ".join(f"methods/{name}/" for name in _existing_methods())
    return [_err(
        3, f"methods/{method}/ nao existe ou nao e diretorio",
        f"pedido: method: {method!r}",
        f"existem: {existem or '(nenhum)'}",
        blocks=True,
    )]


def check_04_default_config(exp: dict, root: str = ".") -> list[dict]:
    """4. configs/default.yaml existe.

    E o primeiro `--config` de todo comando filho
    (`experiments/ray/runners/train.py:86`); sem ele nenhum job sobe.
    """
    relative = os.path.relpath(DEFAULT_CONFIG_YAML, PROJECT_ROOT)
    absolute = os.path.join(_root(root), relative)
    if os.path.isfile(absolute):
        return []
    return [_err(4, "config global ausente", absolute)]


def check_05_resources_output(exp: dict) -> list[dict]:
    """5. resources.gpu_ids presente e nao-vazio; output.work_dir presente."""
    errors = []
    resources = exp.get("resources") or {}
    output = exp.get("output") or {}
    gpu_ids = resources.get("gpu_ids") if isinstance(resources, dict) else None
    if not isinstance(gpu_ids, list) or not gpu_ids:
        errors.append(_err(
            5, "resources.gpu_ids ausente ou vazio",
            f"veio: {gpu_ids!r}; esperava lista nao vazia de int",
        ))
    work_dir = output.get("work_dir") if isinstance(output, dict) else None
    if not isinstance(work_dir, str) or not work_dir.strip():
        errors.append(_err(
            5, "output.work_dir ausente",
            f"veio: {work_dir!r}; esperava caminho nao vazio",
        ))
    return errors


# ─── GRADE (6-11) ─────────────────────────────────────────────────────────────


def check_06_axes(exp: dict) -> list[dict]:
    """6. Todo eixo de `grid` e lista nao-vazia de valores validos.

    `null` continua valendo: no schema real e "usa o default deste eixo"
    (schema.Grid, schema.py:176), nao "eixo vazio".
    """
    grid = exp.get("grid")
    if grid is None:
        return []
    if not isinstance(grid, dict):
        return [_err(
            6, "grid nao e mapeamento", f"veio: {type(grid).__name__}", blocks=True
        )]

    errors = []
    for axis, values in grid.items():
        if axis not in _AXIS_VALUES:
            errors.append(_err(
                6, "eixo desconhecido em grid",
                f"grid.{axis}", f"eixos: {sorted(_AXIS_VALUES)}",
                blocks=True,
            ))
        elif values is None:
            continue
        elif not isinstance(values, list) or not values:
            errors.append(_err(
                6, "eixo de grid nao e lista nao-vazia",
                f"grid.{axis}: {values!r}",
            ))
        else:
            bad = [v for v in values if v not in _AXIS_VALUES[axis]]
            if bad:
                errors.append(_err(
                    6, "valor invalido em eixo de grid",
                    f"grid.{axis}: {bad}",
                    f"validos: {list(_AXIS_VALUES[axis])}",
                    blocks=True,
                ))
    return errors


def check_07_placeholders(exp: dict) -> list[dict]:
    """7. Todo placeholder dos templates de caminho e eixo ou a palavra `method`.

    No schema real os templates sao codigo, nao YAML — entao eles sao
    RECUPERADOS chamando `paths.*` com valores-sentinela (`"{init}"` formata em
    `{init}`) e lidos com `string.Formatter().parse`, nunca com regex caseira.
    A checagem dispara no dia em que `paths.model_config` ganhar um parametro
    que a grade nao produz: placeholder orfao, com o template citado.
    """
    templates = {
        "paths.dataset_config": _template(
            paths.dataset_config, "{dataset}", "{split}"
        ),
        "paths.model_config": _template(
            paths.model_config, "{method}", "{init}", "{dataset}", "{split}"
        ),
    }
    known = set(_AXIS_TO_POINT.values()) | {"method"}

    errors = []
    for origem, template in templates.items():
        orphans = sorted({
            name for _, name, _, _ in string.Formatter().parse(template)
            if name and name not in known
        })
        if orphans:
            errors.append(_err(
                7, "placeholder orfao em template de caminho",
                f"origem: {origem}",
                f"template: {template}",
                f"orfaos: {orphans}; conhecidos: {sorted(known)}",
            ))
    return errors


def check_08_runner_args(exp: dict) -> list[dict]:
    """8. Toda chave de `runner_args` e valida para o runner declarado.

    E o `map:` da spec reancorado: no schema real quem cita nome de fora do
    proprio YAML e `runner_args`, cuja lista fechada esta em
    `schema.RUNNER_ARGS` (schema.py:56-79).
    """
    runner_args = exp.get("runner_args") or {}
    runner = exp.get("runner")
    if runner not in RUNNER_ARGS:
        return []
    if not isinstance(runner_args, dict):
        return [_err(
            8, "runner_args nao e mapeamento", f"veio: {type(runner_args).__name__}"
        )]
    unknown = sorted(set(runner_args) - set(RUNNER_ARGS[runner]))
    if not unknown:
        return []
    return [_err(
        8, f"chave desconhecida em runner_args (runner {runner})",
        f"chaves: {unknown}",
        f"validas: {list(RUNNER_ARGS[runner])}",
    )]


def check_09_overrides(exp: dict) -> list[dict]:
    """9. Toda chave de `overrides` vira `--<chave>=<valor>` literal.

    O `exclude:` da spec nao existe no schema real; o que ocupa o mesmo lugar —
    chave do YAML que tem de existir do outro lado — e `overrides`, que so
    aceita valor escalar: mapeamento aninhado nao tem traducao para a linha de
    comando (schema.py:317-324, runners/train.py:96).
    """
    overrides = exp.get("overrides") or {}
    if not isinstance(overrides, dict):
        return [_err(
            9, "overrides nao e mapeamento", f"veio: {type(overrides).__name__}"
        )]
    nested = sorted(str(k) for k, v in overrides.items() if isinstance(v, (dict, list)))
    if not nested:
        return []
    return [_err(
        9, "override com valor aninhado",
        f"chaves: {nested}",
        "ache a chave folha, ex. 'model.init_args.lr: 3.0e-3'",
    )]


def check_10_dead_axis(exp: dict, points: list[dict]) -> list[dict]:
    """10. Eixo morto: multiplica a grade sem mudar nada do que roda.

    A identidade de um ponto e (dataset_config, model_config, overrides). Um
    eixo e morto quando fixa-lo num valor NAO reduz o numero de identidades
    distintas — ou seja, ele nao toca caminho nenhum nem override nenhum.

    O falso positivo que a spec adverte esta tratado: `percentage` nao aparece
    em nenhum dos dois templates, mas entra na identidade porque vira
    `--data.init_args.percentage` (runners/train.py:91), que `resolve` poe em
    `overrides`. Nao ha eixo obrigado a inventar um `map` falso para calar o
    pre-voo.
    """
    if not points:
        return []

    def identity(point: dict) -> tuple:
        return (
            point["dataset_config"],
            point["model_config"],
            str(sorted(point["overrides"].items(), key=lambda kv: str(kv[0]))),
        )

    total = len({identity(point) for point in points})
    errors = []
    for axis in ("init", "dataset", "split", "percentage"):
        values = sorted({point[axis] for point in points}, key=str)
        if len(values) < 2:
            continue
        fixed = {identity(p) for p in points if p[axis] == values[0]}
        if len(fixed) == total:
            errors.append(_err(
                10, "eixo morto na grade",
                f"eixo {axis}: {values}",
                "nao muda dataset_config, model_config nem override nenhum; "
                f"so multiplica a grade por {len(values)}",
            ))
    return errors


def check_11_empty_grid(exp: dict, points: list[dict]) -> list[dict]:
    """11. Grade vazia = erro."""
    if points:
        return []
    return [_err(
        11, "grade vazia",
        "nenhum ponto depois de expandir `grid`; nada seria submetido",
    )]


# ─── DISCO (12-16) ────────────────────────────────────────────────────────────


def _missing(
    check: int, titulo: str, points: list[dict], key: str, root: str, template: str
) -> list[dict]:
    """UM erro por caminho ausente — deduplicado, na ordem da grade, sem elisao.

    Um erro por caminho, e nao um erro com N linhas, porque a contagem do bloco
    e do resumo tem que dizer 24 quando faltam 24 celulas. O template vai na
    cauda do ULTIMO erro: e nota do bloco, nao de cada linha.
    """
    base = _root(root)
    vistos: set[str] = set()
    errors: list[dict] = []
    for point in points:
        relative = point[key]
        if relative in vistos:
            continue
        vistos.add(relative)
        if not os.path.isfile(os.path.join(base, relative)):
            errors.append(_err(check, titulo, relative))
    if errors:
        errors[-1]["lines"].append(f"template: {template}")
    return errors


def check_12_dataset_config(exp: dict, points: list[dict], root: str = ".") -> list[dict]:
    """12. Cada dataset_config resolvido existe.

    Uma linha por caminho, TODAS. O ponto que gerou cada uma esta no proprio
    caminho — ele e o template formatado com o ponto e nada mais.
    """
    return _missing(
        12, "dataset_config resolvido nao existe", points, "dataset_config", root,
        _template(paths.dataset_config, "{dataset}", "{split}"),
    )


def check_13_model_config(exp: dict, points: list[dict], root: str = ".") -> list[dict]:
    """13. Cada model_config resolvido existe.

    E a checagem que produz a lista das celulas que ainda nao foram geradas —
    `{he,xavier,random,trunc_normal} x {helminth-eggs,helminth-larvae} x split`
    de `lejepa` sao 24 caminhos, e os 24 saem impressos, um por linha.
    """
    return _missing(
        13, "model_config resolvido nao existe", points, "model_config", root,
        _template(paths.model_config, "{method}", "{init}", "{dataset}", "{split}"),
    )


def _model_configs(points: list[dict], root: str) -> list[str]:
    """Os model_config EXISTENTES, deduplicados. Base das checagens 14-16."""
    base = _root(root)
    existentes: list[str] = []
    for point in points:
        relative = point["model_config"]
        if relative not in existentes and os.path.isfile(
            os.path.join(base, relative)
        ):
            existentes.append(relative)
    return sorted(existentes)


def _class_path(relative: str, root: str):
    """(class_path, motivo). Um dos dois e sempre vazio."""
    data = _read_yaml(os.path.join(_root(root), relative))
    if isinstance(data, Exception):
        # Numa linha so: o erro do pyyaml vem em quatro, e quebra o alinhamento
        # do bloco do relatorio.
        return "", "YAML nao parseia: " + " ".join(str(data).split())
    if not isinstance(data, dict):
        return "", f"topo nao e mapeamento: {type(data).__name__}"
    model = data.get("model")
    if not isinstance(model, dict):
        return "", "sem a chave `model` (ou ela nao e mapeamento)"
    class_path = model.get("class_path")
    if not isinstance(class_path, str) or not class_path.strip():
        return "", "sem `model.class_path`"
    return class_path, ""


def check_14_model_yaml(exp: dict, points: list[dict], root: str = ".") -> list[dict]:
    """14. Cada model_config e YAML parseavel e tem `model.class_path`.

    E pre-requisito da 15 no MESMO arquivo: o que falha aqui gera UM erro, o
    desta checagem, e a 15 pula o arquivo — a linha final diz que pulou.
    """
    errors = []
    for relative in _model_configs(points, root):
        _, motivo = _class_path(relative, root)
        if motivo:
            errors.append(_err(
                14, "model_config nao declara model.class_path",
                f"arquivo: {relative}",
                f"motivo:  {motivo}",
                "checagem 15 pulada para este arquivo",
            ))
    return errors


def check_15_class_path(exp: dict, points: list[dict], root: str = ".") -> list[dict]:
    """15. Cada class_path e importavel: import do MODULO + `hasattr`.

    As tres mitigacoes de custo juntas: dedup por class_path, import do modulo
    sem instanciar (instanciar exigiria peso e GPU, que e o que o pre-voo nao
    pode depender de ter) e cache por class_path na corrida inteira.
    """
    errors = []
    vistos: dict[str, str] = {}
    for relative in _model_configs(points, root):
        class_path, motivo = _class_path(relative, root)
        if motivo:
            continue  # ja e erro da 14; um arquivo quebrado nao rende dois erros
        if class_path in vistos:
            continue
        vistos[class_path] = relative

        if class_path not in _IMPORT_CACHE:
            _IMPORT_CACHE[class_path] = _import_error(class_path)
        if _IMPORT_CACHE[class_path]:
            module, _, klass = class_path.rpartition(".")
            errors.append(_err(
                15, "class_path nao importavel",
                f"arquivo:    {relative}",
                f"class_path: {class_path}",
                f"modulo:     {module}",
                f"classe:     {klass}",
                _IMPORT_CACHE[class_path],
            ))
    return errors


def _import_error(class_path: str) -> str:
    """"" se importa; a mensagem se nao. Nunca levanta.

    `except Exception` porque o import de um modulo de modelo pode falhar de
    qualquer jeito (CUDA, versao de lib, side effect no topo) — e isso e ACHADO
    do pre-voo, nao motivo para derrubar o pre-voo.
    """
    module, _, klass = class_path.rpartition(".")
    if not module or not klass:
        return "class_path sem ponto: nao da para separar modulo de classe"
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        return f"modulo {module} nao importou: {type(exc).__name__}: {exc}"
    if not hasattr(imported, klass):
        return f"modulo {module} importou, mas nao tem o atributo {klass}"
    return ""


def check_16_init_args_paths(exp: dict, points: list[dict], root: str = ".") -> list[dict]:
    """16. `arch_json` e `flim_weights_path` dos init_args: existem E sao relativos.

    Sao DOIS erros distintos, com bloco proprio no relatorio. Absoluto que por
    acaso existe NESTA maquina continua sendo erro: o problema e ser absoluto —
    o caminho so roda numa maquina, e o filho recebe tudo com cwd na raiz do
    repo (`experiments/ray/runners/train.py:86-90`).

    `null` nao e caminho: e o valor de quem nao inicializa com FLIM
    (configs/model/lejepa/he/protozoan-cysts/split_1.yaml:8).
    """
    base = _root(root)
    absolutos, inexistentes = [], []
    for relative in _model_configs(points, root):
        data = _read_yaml(os.path.join(base, relative))
        if not isinstance(data, dict):
            continue  # ja e erro da 14
        model = data.get("model")
        init_args = model.get("init_args") if isinstance(model, dict) else None
        if not isinstance(init_args, dict):
            continue
        for key in _PATH_KEYS:
            value = init_args.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if os.path.isabs(value):
                absolutos.append((relative, key, value))
            elif not os.path.exists(os.path.join(base, value)):
                inexistentes.append((relative, key, value))

    errors = []
    for relative, key, value in absolutos:
        errors.append(_err(
            16, "caminho absoluto em init_args",
            f"arquivo: {relative}",
            f"chave:   {key}",
            f"valor:   {value}",
            "deve ser relativo a raiz do repo",
        ))
    for relative, key, value in inexistentes:
        errors.append(_err(
            16, "caminho de init_args nao existe em disco",
            f"arquivo: {relative}",
            f"chave:   {key}",
            f"esperado em: {os.path.join(base, value)}",
        ))
    return errors


# ─── Agregador e relatorio ────────────────────────────────────────────────────


def _fill(name: str, status: str) -> str:
    """`NOME .......... status`, com a regua de SEP_WIDTH do resto do pacote."""
    return f"{name} ".ljust(SEP_WIDTH, ".") + f" {status}"


def _plural(total: int, singular: str, plural: str) -> str:
    return f"{total} {singular if total == 1 else plural}"


def _render(
    exp_path: str,
    exp: dict,
    points: list[dict],
    errors: list[dict],
    skipped: dict[int, list[int]],
) -> str:
    """O relatorio inteiro numa string. Uma saida so, sem elisao, sem truncar."""
    linhas = [
        f"preflight: {exp_path}",
        f"  method={exp.get('method')}  runner={exp.get('runner')}  "
        f"grade={_plural(len(points), 'ponto', 'pontos')}",
        "",
    ]
    for categoria, checagens in _CATEGORIES:
        da_categoria = [e for e in errors if e["check"] in checagens]
        puladas = [n for n in checagens if n in skipped]
        if puladas and not da_categoria:
            status = "nao avaliado"
        elif da_categoria:
            status = _plural(len(da_categoria), "erro", "erros")
        else:
            status = f"ok ({len(checagens)}/{len(checagens)})"
        linhas.append(_fill(categoria, status))

        # Um bloco por (checagem, titulo): e assim que a 16 mostra o absoluto e
        # o inexistente separados, cada um com a propria contagem.
        blocos: dict[tuple, list[dict]] = {}
        for erro in da_categoria:
            blocos.setdefault((erro["check"], erro["title"]), []).append(erro)
        # Ordena pelo NUMERO da checagem; entre titulos da mesma checagem vale a
        # ordem em que os erros sairam, que e a ordem da grade.
        for (numero, titulo), grupo in sorted(blocos.items(), key=lambda kv: kv[0][0]):
            linhas.append("")
            linhas.append(f"  [{numero}] {titulo} ({len(grupo)})")
            # Erro de uma linha (os caminhos que faltam) fica denso; erro de
            # varias linhas ganha respiro entre um e outro.
            solto = any(len(erro["lines"]) > 1 for erro in grupo)
            for indice, erro in enumerate(grupo):
                if solto and indice:
                    linhas.append("")
                linhas.extend(f"    {linha}" for linha in erro["lines"])
        # Uma linha por motivo: quem pulou por causa de [3] nao se mistura com
        # quem pulou por causa de [2].
        por_motivo: dict[tuple, list[int]] = {}
        for numero in puladas:
            por_motivo.setdefault(tuple(skipped[numero]), []).append(numero)
        for motivo, numeros in por_motivo.items():
            rotulo = (
                f"checagens {numeros[0]}-{numeros[-1]} puladas"
                if len(numeros) > 1
                else f"checagem {numeros[0]} pulada"
            )
            linhas.append(
                f"  {rotulo}: depende(m) de "
                + ", ".join(f"[{n}]" for n in motivo)
            )
        # Linha em branco so depois de categoria que teve o que dizer; as que
        # sairam `ok` ficam coladas, como no exemplo da spec.
        if blocos or puladas:
            linhas.append("")

    if linhas[-1] != "":
        linhas.append("")
    resumo = (
        f"preflight: {_plural(len(points), 'ponto', 'pontos')}, "
        f"{_plural(len(errors), 'erro', 'erros')} em "
        f"{_plural(len({(e['check']) for e in errors}), 'checagem', 'checagens')}"
    )
    if skipped:
        resumo += f", {_plural(len(skipped), 'checagem pulada', 'checagens puladas')}"
    linhas.append(resumo)
    return "\n".join(linhas)


def preflight(exp_path: str, root: str = ".") -> list[dict]:
    """Roda TODAS as checagens, sempre; nunca para no primeiro erro.

    Devolve a lista de pontos resolvidos quando esta tudo certo. Quando nao,
    imprime o relatorio agrupado e levanta `SystemExit(1)` — nenhum job e
    submetido.
    """
    data = _read_yaml(os.path.abspath(exp_path))
    if isinstance(data, Exception) or not isinstance(data, dict):
        motivo = data if isinstance(data, Exception) else (
            f"topo nao e mapeamento, veio {type(data).__name__}"
        )
        print(f"preflight: {exp_path}\n  [1] experiment YAML ilegivel\n    {motivo}")
        raise SystemExit(1)
    exp = data

    errors: list[dict] = []
    errors += check_01_required(exp)
    errors += check_02_runner(exp)
    errors += check_03_method(exp)
    errors += check_04_default_config(exp, root)
    errors += check_05_resources_output(exp)
    errors += check_06_axes(exp)
    errors += check_07_placeholders(exp)
    errors += check_08_runner_args(exp)
    errors += check_09_overrides(exp)

    # checagem pulada -> as checagens que a bloquearam. Nunca sai `ok` no lugar
    # de uma pulada: ou o pre-voo avaliou tudo, ou ele diz o que ficou de fora.
    skipped: dict[int, list[int]] = {}
    # A 8 e indexada por runner (`RUNNER_ARGS[runner]`): com runner invalido nao
    # ha contra o que validar, entao ela e PULADA, nao aprovada.
    if exp.get("runner") not in RUNNER_ARGS:
        skipped[8] = [2]

    blocked_by = sorted({e["check"] for e in errors if e["blocks"]})
    if blocked_by:
        points: list[dict] = []
        skipped.update({numero: blocked_by for numero in _BLOCKED})
    else:
        points = [resolve(exp, point) for point in expand(exp)]
        errors += check_10_dead_axis(exp, points)
        errors += check_11_empty_grid(exp, points)
        errors += check_12_dataset_config(exp, points, root)
        # 13-16 resolvem configs/model/{method}/{init}/{dataset}/split_{N}.yaml.
        # O runner `eval` NUNCA abre esse arquivo: os pesos vem do checkpoint
        # (configs/generated/mlp/<run_id>.yaml no probe mlp, artifacts/** nas
        # fontes distillation/growth_stages), e por isso `init` nao e eixo real
        # ali. Cobrar a existencia do config de modelo reprovava o grupo por um
        # arquivo que ninguem le. PULADAS, nao aprovadas.
        if exp.get("runner") == "eval":
            skipped.update({numero: [2] for numero in (13, 14, 15, 16)})
        else:
            errors += check_13_model_config(exp, points, root)
            errors += check_14_model_yaml(exp, points, root)
            errors += check_15_class_path(exp, points, root)
            errors += check_16_init_args_paths(exp, points, root)

    print(_render(exp_path, exp, points, errors, skipped))
    if errors:
        raise SystemExit(1)
    return points


if __name__ == "__main__":
    # Sem argparse: a unica excecao autorizada do refactor e o launcher
    # (spec_refactor.md:246-248). Aqui e caminho posicional e ponto final.
    if len(sys.argv) < 2:
        raise SystemExit(
            "uso: python -m experiments.ray.preflight <experiment.yaml> [raiz]"
        )
    preflight(*sys.argv[1:3])
