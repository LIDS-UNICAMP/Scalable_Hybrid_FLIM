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

"""Schema do experiment YAML do launcher Ray.

O launcher le o YAML, instancia `Experiment` e e a dataclass que viaja pelo resto
do pacote: nenhum dict solto circula. Validacao e `if` + mensagem clara, sem
pydantic, sem registry, sem argparse.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# experiments/ray/schema.py -> parents[2] e a raiz do repo
REPO_ROOT = Path(__file__).resolve().parents[2]
# pasta consultada por os.path.isdir para validar `method`
METHODS_DIR = REPO_ROOT / "methods"

INITS = ("flim", "he", "xavier", "random", "trunc_normal")
DATASETS = ("helminth-eggs", "helminth-larvae", "protozoan-cysts")
SPLITS = (1, 2, 3)
PERCENTAGES = (1, 5, 25, 50, 75, 100)
RUNNERS = ("train", "growth", "eval")
SKIP_MODES = ("none", "state", "wandb")
# de onde o runner `eval` tira os checkpoints que vai avaliar
EVAL_SOURCES = ("mlp_configs", "distillation", "growth_stages")

# chaves aceitas em runner_args, por runner; qualquer outra e erro
RUNNER_ARGS: dict[str, tuple[str, ...]] = {
    "train": (),
    # As 4 primeiras vem da spec. As outras 15 sao knobs que o laco de hoje ja
    # expoe e que o runner `growth` le de `runner_args`
    # (`GROW_KNOBS`, experiments/ray/runners/growth.py:151-157, e `run_arm`,
    # :272-273). Sem elas um YAML pedindo `kernel_size: 5` ou a ablacao
    # `random_layer: true` e recusado com "chave desconhecida".
    "growth": (
        "max_rounds", "kappa_tolerance", "rounds_patience", "embed_mode",
        # repassadas a `flim.spifil.grow` com o mesmo nome, defaults em
        # scripts/spifil_grow.py:594-634; o laco as encaminha em
        # scripts/spifil_growth_loop.py:441-463
        "out_channels", "kernel_size", "pool_stride", "n_superpixels", "n_images",
        "image_size", "seed", "spifil_in_feature", "spifil_in_image", "impurities",
        "one_per_class", "random_layer", "random_layer_classic",
        # mudam a FORMA da cadeia de estagios da rodada
        # (scripts/spifil_growth_loop.py:409-417, mutuamente exclusivas)
        "head_finetune", "unfrozen_after_stage_two",
    ),
    # `source` escolhe a varredura; as 4 seguintes so fazem sentido nas fontes
    # que varrem artifacts/ (distillation, growth_stages)
    "eval": (
        "probe", "freeze", "ckpt_selection",
        "source", "artifacts_dir", "run_filter", "family", "stages",
        "require_status_ok", "results_csv",
    ),
}


def _check_keys(section: str, data: dict, allowed) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(
            f"{section}: chave desconhecida {unknown}; validas: {sorted(allowed)}"
        )


def _check_axis(axis: str, values, allowed) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid.{axis}: esperava lista nao vazia, veio {values!r}")
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"grid.{axis}: valor invalido {bad}; validos: {list(allowed)}")


def _check_eval_args(args: dict) -> None:
    """Valida runner_args do runner `eval`: cada `source` exige (e recusa) chaves."""
    section = "runner_args (runner eval)"
    source = args.get("source", "mlp_configs")
    if source not in EVAL_SOURCES:
        raise ValueError(
            f"{section}.source: {source!r} invalido; validos: {list(EVAL_SOURCES)}"
        )
    for name in ("artifacts_dir", "run_filter", "family", "results_csv"):
        value = args.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"{section}.{name}: esperava str nao vazia, veio {value!r}"
            )
    stages = args.get("stages")
    if stages is not None:
        if not isinstance(stages, list) or not stages:
            raise ValueError(
                f"{section}.stages: esperava lista nao vazia de str, veio {stages!r}"
            )
        bad = [item for item in stages if not isinstance(item, str) or not item.strip()]
        if bad:
            raise ValueError(
                f"{section}.stages: item invalido {bad}; esperava str nao vazia"
            )
    status_ok = args.get("require_status_ok", True)
    if not isinstance(status_ok, bool):
        raise ValueError(
            f"{section}.require_status_ok: esperava bool, veio {status_ok!r}"
        )
    exigidas = {
        "distillation": ("artifacts_dir", "run_filter"),
        "growth_stages": ("family",),
    }.get(source, ())
    missing = [name for name in exigidas if args.get(name) is None]
    if missing:
        raise ValueError(f"{section}: source {source!r} exige {missing}")
    if source == "mlp_configs":
        # sem isso a chave entraria no YAML e seria silenciosamente ignorada
        extra = sorted(
            name
            for name in ("artifacts_dir", "run_filter", "family", "stages")
            if args.get(name) is not None
        )
        if extra:
            raise ValueError(
                f"{section}: source 'mlp_configs' nao aceita {extra}; "
                "essas chaves so valem para source distillation ou growth_stages"
            )
        if args.get("probe") == "svm":
            raise ValueError(
                f"{section}: probe 'svm' nao combina com source 'mlp_configs'; "
                "o glob de configs/generated/mlp so serve ao probe mlp; "
                "para o SVM use source distillation ou growth_stages"
            )


def _check_gpu_map(name: str, mapping, gpu_ids: list[int], minimum: int) -> None:
    """Valida um mapeamento GPU -> int: uma entrada por GPU, nem mais nem menos."""
    if not isinstance(mapping, dict):
        raise ValueError(
            f"resources.{name}: esperava mapeamento GPU->int, veio {mapping!r}"
        )
    bad_keys = [key for key in mapping if not isinstance(key, int)]
    if bad_keys:
        raise ValueError(
            f"resources.{name}: chave {bad_keys} nao e id de GPU; "
            "escreva o id sem aspas, ex. '{0: 7, 1: 3}'"
        )
    if set(mapping) != set(gpu_ids):
        raise ValueError(
            f"resources.{name}: chaves {sorted(mapping)} tem que ser exatamente "
            f"as gpu_ids {sorted(gpu_ids)}, uma entrada por GPU"
        )
    bad = sorted(
        gid for gid, value in mapping.items()
        if not isinstance(value, int) or value < minimum
    )
    if bad:
        raise ValueError(
            f"resources.{name}: GPU {bad} com valor invalido; esperava int >= {minimum}"
        )


def _existing_methods() -> list[str]:
    if not os.path.isdir(METHODS_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(METHODS_DIR)
        if not name.startswith((".", "_"))
        and os.path.isdir(os.path.join(METHODS_DIR, name))
    )


def _section(cls, data, section: str):
    """Instancia uma sub-dataclass a partir do mapeamento do YAML."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{section}: esperava mapeamento, veio {type(data).__name__}")
    _check_keys(section, data, [f.name for f in fields(cls)])
    missing = [
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING and f.name not in data
    ]
    if missing:
        raise ValueError(f"{section}: chave obrigatoria ausente: {missing}")
    return cls(**data)


@dataclass
class Grid:
    """Eixos do produto cartesiano; None significa usar o default do metodo."""

    init: list[str] | None = None
    datasets: list[str] | None = None
    splits: list[int] | None = None
    percentages: list[int] | None = None

    def __post_init__(self) -> None:
        for axis, allowed in (
            ("init", INITS),
            ("datasets", DATASETS),
            ("splits", SPLITS),
            ("percentages", PERCENTAGES),
        ):
            values = getattr(self, axis)
            if values is not None:
                _check_axis(axis, values, allowed)


@dataclass
class Resources:
    """GPUs fisicas e paralelismo; gpu_ids e obrigatorio, sem default magico."""

    gpu_ids: list[int]
    # quantos jobs rodam AO MESMO TEMPO em cada GPU. int e o mesmo limite para
    # todas; dict e um limite por GPU ({0: 7, 1: 3}), os slots heterogeneos.
    max_per_gpu: int | dict[int, int] = 1
    # quantos jobs cada GPU recebe NO TOTAL na fila inteira (nao simultaneos);
    # null e sem cota, e a fila e drenada de forma gulosa. Cota 0 tira a GPU.
    max_total_per_gpu: dict[int, int] | None = None
    cpus_per_experiment: int = 4
    num_workers: int = 4
    ray_address: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_ids, list) or not self.gpu_ids:
            raise ValueError(
                f"resources.gpu_ids: esperava lista nao vazia de int, veio {self.gpu_ids!r}"
            )
        if any(not isinstance(gpu, int) or gpu < 0 for gpu in self.gpu_ids):
            raise ValueError(f"resources.gpu_ids: so int >= 0, veio {self.gpu_ids!r}")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError(
                f"resources.gpu_ids: id repetido em {self.gpu_ids!r}; "
                "mais de um job por GPU e max_per_gpu, nao id duplicado"
            )
        # int mantem o significado de sempre: o mesmo limite para todas as GPUs
        if not isinstance(self.max_per_gpu, (int, dict)):
            raise ValueError(
                "resources.max_per_gpu: esperava int (o mesmo para todas as GPUs) "
                f"ou mapeamento GPU->int, veio {self.max_per_gpu!r}"
            )
        if isinstance(self.max_per_gpu, int):
            if self.max_per_gpu < 1:
                raise ValueError(
                    f"resources.max_per_gpu: esperava int >= 1, veio {self.max_per_gpu!r}"
                )
        else:
            _check_gpu_map("max_per_gpu", self.max_per_gpu, self.gpu_ids, 1)
        # a cota aceita 0: e assim que se deixa uma GPU da lista de fora
        if self.max_total_per_gpu is not None:
            _check_gpu_map("max_total_per_gpu", self.max_total_per_gpu, self.gpu_ids, 0)
        for name, minimum in (
            ("cpus_per_experiment", 1),
            ("num_workers", 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"resources.{name}: esperava int >= {minimum}, veio {value!r}"
                )
        if self.ray_address is not None and not isinstance(self.ray_address, str):
            raise ValueError(
                f"resources.ray_address: esperava str ou null, veio {self.ray_address!r}"
            )


@dataclass
class Output:
    """work_dir e obrigatorio; o launcher grava <log_dir>/<name>_<timestamp>.log."""

    work_dir: str
    log_dir: str = "logs"

    def __post_init__(self) -> None:
        for name in ("work_dir", "log_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"output.{name}: esperava caminho nao vazio, veio {value!r}"
                )


@dataclass
class Wandb:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"wandb.enabled: esperava bool, veio {self.enabled!r}")
        for name in ("project", "entity"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"wandb.{name}: esperava str ou null, veio {value!r}")
        if self.enabled and not self.project:
            raise ValueError("wandb: enabled true exige wandb.project")


@dataclass
class Experiment:
    """Um experiment YAML inteiro, ja validado."""

    name: str
    method: str
    runner: str
    resources: Resources
    output: Output
    grid: Grid = field(default_factory=Grid)
    overrides: dict[str, Any] = field(default_factory=dict)
    runner_args: dict[str, Any] = field(default_factory=dict)
    wandb: Wandb = field(default_factory=Wandb)
    skip: str = "state"

    def __post_init__(self) -> None:
        # name vira prefixo de run_name e nome do arquivo de log
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name: esperava str nao vazia, veio {self.name!r}")
        if "/" in self.name or any(char.isspace() for char in self.name):
            raise ValueError(f"name: sem '/' e sem espaco, veio {self.name!r}")

        if self.runner not in RUNNERS:
            raise ValueError(
                f"runner: {self.runner!r} invalido; validos: {list(RUNNERS)}"
            )

        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError(f"method: esperava str nao vazia, veio {self.method!r}")
        if not os.path.isdir(os.path.join(METHODS_DIR, self.method)):
            raise ValueError(
                f"method: {self.method!r} nao e uma pasta em {METHODS_DIR}; "
                f"metodos existentes: {_existing_methods()}"
            )

        if self.skip not in SKIP_MODES:
            raise ValueError(f"skip: {self.skip!r} invalido; validos: {list(SKIP_MODES)}")

        if not isinstance(self.overrides, dict):
            raise ValueError(
                f"overrides: esperava mapeamento, veio {type(self.overrides).__name__}"
            )
        # cada par vira --<k>=<v> literal; mapeamento aninhado nao tem traducao
        nested = sorted(k for k, v in self.overrides.items() if isinstance(v, dict))
        if nested:
            raise ValueError(
                f"overrides: {nested} tem mapeamento aninhado; ache a chave, "
                "ex. 'model.init_args.lr: 3.0e-3'"
            )

        if not isinstance(self.runner_args, dict):
            raise ValueError(
                f"runner_args: esperava mapeamento, veio {type(self.runner_args).__name__}"
            )
        _check_keys(
            f"runner_args (runner {self.runner})",
            self.runner_args,
            RUNNER_ARGS[self.runner],
        )
        if self.runner == "eval":
            _check_eval_args(self.runner_args)
        # as duas reescrevem a rodada, e de formas diferentes: o argparse de
        # scripts/spifil_growth_loop.py:407-417 recusava o par, e sem ele
        # experiments/ray/runners/growth.py:329 daria a vitoria ao head_finetune
        if self.runner_args.get("head_finetune") and self.runner_args.get(
            "unfrozen_after_stage_two"
        ):
            raise ValueError(
                "runner_args (runner growth): head_finetune e "
                "unfrozen_after_stage_two sao mutuamente exclusivas; escolha uma"
            )

    @classmethod
    def from_dict(cls, data) -> "Experiment":
        if not isinstance(data, dict):
            raise ValueError(
                f"topo: esperava mapeamento, veio {type(data).__name__}"
            )
        _check_keys("topo", data, [f.name for f in fields(cls)])
        missing = [
            key
            for key in ("name", "method", "runner", "resources", "output")
            if key not in data
        ]
        if missing:
            raise ValueError(f"topo: chave obrigatoria ausente: {missing}")
        return cls(
            name=data["name"],
            method=data["method"],
            runner=data["runner"],
            resources=_section(Resources, data["resources"], "resources"),
            output=_section(Output, data["output"], "output"),
            grid=_section(Grid, data.get("grid"), "grid"),
            overrides=data.get("overrides") or {},
            runner_args=data.get("runner_args") or {},
            wandb=_section(Wandb, data.get("wandb"), "wandb"),
            skip=data["skip"] if data.get("skip") is not None else "state",
        )


def load(path: str | os.PathLike) -> Experiment:
    """Le o experiment YAML e devolve o Experiment ja validado."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        return Experiment.from_dict(data)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from None
