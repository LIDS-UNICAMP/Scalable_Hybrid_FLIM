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

"""eval.py — runner de avaliacao: MLP/SVM sobre checkpoint ja treinado.

Nao passa por `train.py`. E o `src/evaluate/ray_mlp_queue.py` reencarnado, com a
parte de probe removida: aqui so ha orquestracao. Nenhuma metrica e calculada
neste arquivo — `core/metrics.py::compute_metrics` e a unica do projeto e quem a
chama e o probe, dentro do subprocesso.

Consome `runner_args` (validados em `experiments/ray/schema.py`):

* `probe`          — `mlp` ou `svm`; escolhe o modulo do subprocesso.
* `freeze`         — `true` so freeze, `false` so unfreeze, ausente os dois.
* `ckpt_selection` — `best` ou `last`; vira `--ckpt-selection` no probe.
* `source`         — de ONDE saem os checkpoints: `mlp_configs` (default, o glob
  historico de configs/generated/mlp), `distillation` (varredura de
  artifacts/distillation) ou `growth_stages` (glob de artifacts/spifil_growth).
* `artifacts_dir`  — raiz da varredura das duas fontes de artifacts/.
* `run_filter`     — substring do nome do run (so `distillation`).
* `family`/`stages`— familia e rotulos de estagio (so `growth_stages`).
* `require_status_ok` — descarta run cujo `run_metadata.json` nao diz
  `status: ok` (default `true`, so `distillation`).
* `results_csv`    — CSV de saida do probe; entra no YAML de job.

Desenho de GPU, preservado de `src/evaluate/ray_mlp_queue.py:322`: Ray sobe com
`num_gpus=0` e cada task fixa `CUDA_VISIBLE_DEVICES` para a GPU que o
`GpuSlotScheduler` escolheu. A task NAO declara `num_gpus` para o Ray: isso
tiraria do usuario a escolha de QUAL GPU fisica e impediria mais de um job por
GPU (`resources.max_per_gpu`).

O env vai por `subprocess.run(env=...)`, nunca por mutacao de `os.environ` do
worker: o processo do worker e reusado entre tasks e o valor vazaria para a
proxima. Quem chama o subprocesso e o `_run` de `runners/train.py:95`, o mesmo
helper que `runners/growth.py:122` ja reusa — os tres runners tem UM caminho de
subprocesso. Este arquivo tambem nao importa torch em lugar nenhum: contexto
CUDA e so do subprocesso.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from typing import Any

import yaml
from tqdm import tqdm

import ray

from core.constants import OMP_ENV_VAR, PROJECT_ROOT
from core.wandb import resolve_run
from experiments.constants import (
    CUDA_ENV_VAR,
    LOG_LEVELS,
    LOG_LEVEL_DEFAULT,
    LOG_TIME_FMT,
    MLP_CONFIGS_DIR,
    PARASITE_DIR,
    RAY_INIT_KWARGS,
    SEP_WIDTH,
    WANDB_CHILD_ENV,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from experiments.ray.execution_state import ExecutionState
from experiments.ray.gpu_slot_scheduler import GpuSlotScheduler
from experiments.ray.runners.train import DEFAULT_CPUS, _run
from experiments.ray.schema import EVAL_SOURCES, Experiment
from experiments.ray.skip import query_wandb_state

# Valores aceitos em runner_args. O schema garante que a CHAVE existe; o valor e
# validado aqui, onde ele e usado.
PROBES = ("mlp", "svm")
CKPT_SELECTIONS = ("best", "last")

# Subpastas de MLP_CONFIGS_DIR. Os YAMLs sao gerados por
# scripts/generate_mlp_configs.py:22-23 e lidos hoje por src/evaluate/mlp.py:71:
# sao saida de gerador E entrada viva do probe, nao lixo descartavel.
FREEZE_MODES = ("freeze", "unfreeze")

_current_log_level = LOG_LEVELS[LOG_LEVEL_DEFAULT]


def _log(msg: str, level: str = "INFO") -> None:
    """Uma linha de estado por evento: submetido, terminou, falhou, pulado."""
    if LOG_LEVELS.get(level, 1) >= _current_log_level:
        print(f"[{time.strftime(LOG_TIME_FMT)}][{level}] {msg}", flush=True)


# ─── runner_args ──────────────────────────────────────────────────────────────


def parse_runner_args(exp: Experiment) -> tuple[str, bool | None, str]:
    """Le e valida `probe`, `freeze` e `ckpt_selection` do experiment YAML.

    Devolve `(probe, freeze, ckpt_selection)`, onde `freeze` e `None` quando a
    chave nao aparece no YAML — nesse caso a grade cobre freeze e unfreeze.
    """
    args = exp.runner_args

    probe = args.get("probe", "mlp")
    if probe not in PROBES:
        raise ValueError(
            f"runner_args.probe: {probe!r} invalido; validos: {list(PROBES)}"
        )

    freeze = args.get("freeze")
    if freeze is not None and not isinstance(freeze, bool):
        raise ValueError(
            f"runner_args.freeze: esperava bool ou ausente, veio {freeze!r}"
        )

    ckpt_selection = args.get("ckpt_selection", "best")
    if ckpt_selection not in CKPT_SELECTIONS:
        raise ValueError(
            f"runner_args.ckpt_selection: {ckpt_selection!r} invalido; "
            f"validos: {list(CKPT_SELECTIONS)}"
        )

    return probe, freeze, ckpt_selection


# ─── Plano da grade ───────────────────────────────────────────────────────────


def _grid_reject(exp: Experiment, cfg: dict) -> str | None:
    """Eixo do `grid` que BARRA este config, ou `None` quando ele passa.

    Eixo `None` no schema significa "sem filtro". O config carrega os quatro
    eixos como campo proprio (ver configs/evaluate/mlp/**/<run_id>.yaml). Devolve
    o NOME do eixo (nao um bool) porque o diagnostico de grade vazia precisa
    dizer qual filtro esvaziou o plano.
    """
    axes = (
        ("init", exp.grid.init, cfg.get("initialization_type")),
        ("datasets", exp.grid.datasets, cfg.get("dataset_name")),
        ("splits", exp.grid.splits, cfg.get("split", cfg.get("split_id"))),
        ("percentages", exp.grid.percentages, cfg.get("percentage")),
    )
    for axis, allowed, value in axes:
        if allowed is not None and value not in allowed:
            return axis
    return None


# Gramatica do nome de run da grade distill4. `unfrozen` vem ANTES de `frozen` na
# alternancia e os dois carregam o `_` da esquerda: sem isso `frozen` casaria
# dentro de `unfrozen` e todo run destravado sairia rotulado como travado.
_DISTILL_RUN_RE = re.compile(
    r"^distill4_all_(?:cos|kd|mse)_(?:unfrozen|frozen)"
    r"_(eggs|larvae|protozoan)_s(\d+)_p(\d+)$"
)

# Braco de uma grade de crescimento: `<dataset>_split<N>_pct<P>`. Mesmo padrao de
# eval/growth_stages.py:130, reescrito aqui porque aquele modulo importa torch no
# topo e este arquivo e livre de torch de proposito.
_GROWTH_ARM_RE = re.compile(r"^(eggs|larvae|protozoan)_split(\d+)_pct(\d+)$")

GROWTH_ARTIFACTS_DIR: str = os.path.join(PROJECT_ROOT, "artifacts", "spifil_growth")


def _job_method(exp: Experiment) -> str:
    """Rotulo da coluna `method` do CSV, derivado do NOME do experiment.

    `svm_distill_cos_unfrozen` -> `SVM_Distill_Cos_Unfrozen`: cada pedaco
    capitalizado, e o pedaco que e nome de sonda (`svm`/`mlp`) em caixa alta.
    Sai do `exp.name` e nunca do `results_csv`: o CSV e destino, nao identidade.
    """
    return "_".join(
        part.upper() if part in PROBES else part.capitalize()
        for part in exp.name.split("_")
    )


def _artifact_job(
    exp: Experiment,
    probe: str,
    ckpt_selection: str,
    source: str,
    run_name: str,
    ckpt: str,
    dataset_short: str,
    split: int,
    percentage: int,
    stage_label: str = "",
) -> dict:
    """Job de um checkpoint de `artifacts/`: grava o YAML e devolve o dict.

    O YAML em `<work_dir>/jobs/` e o CONTRATO com `eval/svm.py` — ele carrega o
    checkpoint, os eixos e o CSV de saida, entao a linha de comando fica com duas
    flags so. E escrito ja no plano (inclusive em `--dry-run`) para que o comando
    impresso seja executavel como esta.
    """
    cfg = {
        "ckpt": os.path.relpath(ckpt, PROJECT_ROOT),
        "run_name": run_name,
        "dataset_name": PARASITE_DIR[dataset_short],
        "dataset_short": dataset_short,
        "split": split,
        "percentage": percentage,
        "method": _job_method(exp),
        "source": source,
    }
    if stage_label:
        cfg["stage_label"] = stage_label
    cfg["results_csv"] = (
        exp.runner_args.get("results_csv") or f"results/svm_{exp.name}.csv"
    )

    stem = f"{run_name}_{stage_label}" if stage_label else run_name
    jobs_dir = os.path.join(exp.output.work_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    config_path = os.path.join(jobs_dir, f"{stem}.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    return {
        "key": f"{exp.name}_{probe}_{source}_{stem}",
        "probe": probe,
        "mode": source,
        "ckpt_selection": ckpt_selection,
        "config_path": config_path,
        "run_id": run_name,
        "experiment_name": run_name,
        "dataset_name": cfg["dataset_name"],
        "split": split,
        "percentage": percentage,
        "initialization_type": "",
        "source": source,
    }


def _plan_mlp_configs(
    exp: Experiment, probe: str, freeze: bool | None, ckpt_selection: str
) -> tuple[list[dict], list[str]]:
    """Um job por config de `MLP_CONFIGS_DIR` que casa com a grade.

    A descoberta e puro disco: glob recursivo em `MLP_CONFIGS_DIR/<mode>/**/*.yaml`
    mais um `yaml.safe_load` por arquivo. O `**` deixa o plano indiferente a
    profundidade da pasta — os quatro eixos da grade saem dos CAMPOS do YAML, nunca
    do caminho. Sem W&B, sem torch, sem checkpoint: o veredito de peso faltando e
    do probe, que ja devolve `status: missing_weights`
    (src/evaluate/mlp.py:373-379).
    """
    modes = FREEZE_MODES if freeze is None else (FREEZE_MODES[0 if freeze else 1],)

    jobs: list[dict] = []
    rejected: dict[str, int] = {}
    seen = 0
    for mode in modes:
        pattern = os.path.join(MLP_CONFIGS_DIR, mode, "**", "*.yaml")
        paths = sorted(glob.glob(pattern, recursive=True))
        seen += len(paths)
        for config_path in tqdm(paths, desc=f"plan mlp_configs/{mode}", unit="cfg"):
            with open(config_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            axis = _grid_reject(exp, cfg)
            if axis is not None:
                rejected[axis] = rejected.get(axis, 0) + 1
                continue

            run_id = cfg.get("run_id", os.path.splitext(os.path.basename(config_path))[0])
            split = cfg.get("split", cfg.get("split_id", "?"))
            # `exp.name` prefixa o run_name (schema.py:230) e o `mode` entra na
            # chave: sem ele, freeze e unfreeze do mesmo run_id colidiriam no
            # ExecutionState, como colidem hoje em ray_mlp_queue.py:400.
            key = (
                f"{exp.name}_{probe}_{mode}_{cfg.get('dataset_name', 'unknown')}"
                f"_split{split}_pct{cfg.get('percentage', '?')}"
                f"_{cfg.get('initialization_type', 'unknown')}_{run_id}"
            )
            jobs.append({
                "key": key,
                "probe": probe,
                "mode": mode,
                "ckpt_selection": ckpt_selection,
                "config_path": config_path,
                "run_id": run_id,
                "experiment_name": cfg.get("experiment_name", ""),
                "dataset_name": cfg.get("dataset_name", ""),
                "split": split,
                "percentage": cfg.get("percentage"),
                "initialization_type": cfg.get("initialization_type", ""),
            })

    funnel = [f"{seen} config(s) em {MLP_CONFIGS_DIR}/<{'|'.join(modes)}>"]
    funnel += [f"grid.{axis} descartou {n}" for axis, n in sorted(rejected.items())]
    return jobs, funnel


def _plan_distillation(
    exp: Experiment, probe: str, ckpt_selection: str
) -> tuple[list[dict], list[str]]:
    """Um job por run de destilacao com checkpoint utilizavel.

    A escolha do `.ckpt` NAO e reimplementada aqui: e a de
    `find_distillation_runs` (eval/svm_variants/svm_distillation.py:226-288) —
    metadado `best_checkpoint`, senao o grupo `best*`, senao o grupo `last*`,
    desempatando pelo epoch/step gravado DENTRO do arquivo. Ela existe porque um
    re-run quebrado deixa um `best.ckpt` de epoch 0 ao lado de um `best-v1.ckpt`
    treinado. O import e tardio porque aquele modulo puxa torch e o topo deste
    arquivo nao pode.

    Ela e chamada um run POR VEZ e so depois de status/nome/grade: o desempate
    abre cada `.ckpt` com `torch.load`, e aqui um checkpoint tem 7 GB — resolver
    a grade inteira de uma vez levaria minutos abrindo justamente os runs que o
    filtro ia jogar fora.
    """
    from eval.svm_variants.svm_distillation import find_distillation_runs

    args = exp.runner_args
    # `artifacts_dir` e `run_filter` sao obrigatorios nesta fonte (schema.py:125-130).
    # O join com PROJECT_ROOT aceita tanto "artifacts/distillation" quanto caminho
    # absoluto — join com absoluto devolve o absoluto.
    artifacts_dir = os.path.join(PROJECT_ROOT, args["artifacts_dir"])
    run_filter = args["run_filter"]
    require_status_ok = args.get("require_status_ok", True)

    candidates = sorted(
        entry for entry in os.listdir(artifacts_dir)
        if run_filter in entry and os.path.isdir(os.path.join(artifacts_dir, entry))
    )
    jobs: list[dict] = []
    rejected: dict[str, int] = {}
    n_meta = n_status = n_ckpt = 0
    for run_name in tqdm(candidates, desc="plan distillation", unit="run"):
        run_dir = os.path.join(artifacts_dir, run_name)
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if not os.path.isfile(meta_path):
            n_meta += 1
            continue
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        if require_status_ok and meta.get("status") != "ok":
            n_status += 1
            continue

        hit = _DISTILL_RUN_RE.match(run_name)
        if hit is None:
            raise ValueError(
                f"runner eval (source=distillation): nome de run fora da gramatica "
                f"distill4_all_<cos|kd|mse>_<frozen|unfrozen>_<dataset>_s<N>_p<P>: "
                f"{run_dir}"
            )
        dataset_short, split, percentage = hit[1], int(hit[2]), int(hit[3])

        axis = _grid_reject(exp, {
            "dataset_name": PARASITE_DIR[dataset_short],
            "split": split,
            "percentage": percentage,
        })
        if axis is not None:
            rejected[axis] = rejected.get(axis, 0) + 1
            continue

        # `run_name` como filtro casa por SUBSTRING: `..._s1_p5` traz tambem
        # `..._s1_p50`. O basename exato desempata.
        picked = next(
            (m for m in find_distillation_runs(artifacts_dir, run_name)
             if os.path.basename(m["_run_dir"]) == run_name),
            None,
        )
        if picked is None:
            n_ckpt += 1
            _log(f"sem checkpoint utilizavel: {run_dir}", "WARN")
            continue

        jobs.append(_artifact_job(
            exp, probe, ckpt_selection, "distillation", run_name,
            picked["_ckpt_path"], dataset_short, split, percentage,
        ))

    if n_status:
        _log(f"{n_status} run(s) descartado(s) por status != ok (require_status_ok).")
    funnel = [
        f"{len(candidates)} diretorio(s) em {artifacts_dir} "
        f"casando run_filter={run_filter!r}",
        f"sem run_metadata.json descartou {n_meta}",
        f"require_status_ok descartou {n_status}",
        f"sem checkpoint utilizavel descartou {n_ckpt}",
    ]
    funnel += [f"grid.{axis} descartou {n}" for axis, n in sorted(rejected.items())]
    return jobs, funnel


def _plan_growth_stages(
    exp: Experiment, probe: str, ckpt_selection: str
) -> tuple[list[dict], list[str]]:
    """Um job por estagio de crescimento com `best_kappa.ckpt`.

    A existencia do checkpoint e o proprio filtro (eval/growth_stages.py:131-136):
    os diretorios `*_grow`, que so guardam a arquitetura enxertada, nao tem
    checkpoint nenhum e caem fora sozinhos. O rotulo do estagio e o NOME DO
    DIRETORIO, nunca o campo `stage` do metadado.
    """
    args = exp.runner_args
    root = os.path.join(PROJECT_ROOT, args.get("artifacts_dir") or GROWTH_ARTIFACTS_DIR)
    family = args["family"]
    stages = args.get("stages")

    pattern = os.path.join(root, family, "*", "*", "checkpoints", "best_kappa.ckpt")
    ckpts = sorted(glob.glob(pattern))
    jobs: list[dict] = []
    rejected: dict[str, int] = {}
    n_stage = n_arm = 0
    for ckpt in tqdm(ckpts, desc="plan growth_stages", unit="ckpt"):
        stage_dir = os.path.dirname(os.path.dirname(ckpt))
        arm_dir = os.path.dirname(stage_dir)
        stage_label = os.path.basename(stage_dir)
        if stages is not None and stage_label not in stages:
            n_stage += 1
            continue

        hit = _GROWTH_ARM_RE.match(os.path.basename(arm_dir))
        if hit is None:
            n_arm += 1
            _log(f"braco fora do padrao <dataset>_split<N>_pct<P>: {arm_dir}", "WARN")
            continue
        dataset_short, split, percentage = hit[1], int(hit[2]), int(hit[3])

        axis = _grid_reject(exp, {
            "dataset_name": PARASITE_DIR[dataset_short],
            "split": split,
            "percentage": percentage,
        })
        if axis is not None:
            rejected[axis] = rejected.get(axis, 0) + 1
            continue

        jobs.append(_artifact_job(
            exp, probe, ckpt_selection, "growth_stages",
            f"{os.path.basename(os.path.dirname(arm_dir))}_{os.path.basename(arm_dir)}",
            ckpt, dataset_short, split, percentage, stage_label,
        ))

    funnel = [
        f"{len(ckpts)} best_kappa.ckpt em {root} (family={family!r})",
        f"stages={stages!r} descartou {n_stage}",
        f"nome de braco invalido descartou {n_arm}",
    ]
    funnel += [f"grid.{axis} descartou {n}" for axis, n in sorted(rejected.items())]
    return jobs, funnel


def plan(exp: Experiment) -> list[dict]:
    """Enumera um job por checkpoint que casa com a grade.

    `runner_args.source` escolhe a varredura; a secao `grid` FILTRA o resultado
    das tres, sempre pelos mesmos quatro eixos. Plano vazio e erro, nunca lista
    vazia: ver o `RuntimeError` no fim.
    """
    probe, freeze, ckpt_selection = parse_runner_args(exp)
    source = exp.runner_args.get("source", "mlp_configs")
    if source not in EVAL_SOURCES:
        raise ValueError(
            f"runner_args.source: {source!r} invalido; validos: {list(EVAL_SOURCES)}"
        )

    if source == "mlp_configs":
        jobs, funnel = _plan_mlp_configs(exp, probe, freeze, ckpt_selection)
    else:
        # Checkpoint solto de artifacts/ so tem sonda SVM: `probe` ausente ja
        # vale `svm` aqui, em vez do default `mlp` do caso historico.
        probe = exp.runner_args.get("probe", "svm")
        if probe != "svm":
            raise ValueError(
                f"runner_args.probe: source={source} so aceita 'svm', veio {probe!r}"
            )
        planner = _plan_distillation if source == "distillation" else _plan_growth_stages
        jobs, funnel = planner(exp, probe, ckpt_selection)

    if not jobs:
        # Grade vazia e ERRO, nunca exit 0 silencioso: sem isso um `run_filter`
        # com erro de digitacao vira "nada a fazer" e o experimento inteiro some
        # sem ninguem notar.
        raise RuntimeError(
            "\n  ".join([
                f"{exp.name}: nenhum job casou (source={source}). Funil:",
                *funnel,
            ])
        )
    return jobs


# ─── Comando ──────────────────────────────────────────────────────────────────


def build_cmd(exp: Experiment, job: dict) -> list[str]:
    """Monta o comando do probe. Funcao pura: `--dry-run` imprime o que roda.

    Nunca ha dois caminhos de montagem — o dry-run testa exatamente a lista que
    o caminho normal executa.
    """
    cmd = [
        sys.executable, "-m", f"eval.{job['probe']}",
        "--config", job["config_path"],
        f"--ckpt-selection={job['ckpt_selection']}",
    ]
    if job.get("source"):
        # Fontes de artifacts/: so essas duas flags. O YAML de job ja carrega
        # checkpoint, eixos, method e CSV de saida — nao ha o que acrescentar.
        return cmd
    if exp.wandb.enabled:
        cmd.append("--wandb")
    # Mesma traducao literal de overrides do runner train (spec R7).
    cmd += [f"--{key}={value}" for key, value in exp.overrides.items()]
    return cmd


def _wandb_env(exp: Experiment) -> dict[str, str]:
    """Env de W&B do filho: console mudo e projeto/entidade do experiment YAML."""
    if not exp.wandb.enabled:
        return {}
    env = dict(WANDB_CHILD_ENV)
    if exp.wandb.project:
        env["WANDB_PROJECT"] = exp.wandb.project
    if exp.wandb.entity:
        env["WANDB_ENTITY"] = exp.wandb.entity
    return env


# ─── Task Ray ─────────────────────────────────────────────────────────────────


@ray.remote  # sem GPU declarada: Ray nao gerencia GPU, o pinning e CUDA_VISIBLE_DEVICES
def run_eval(
    exp: Experiment,
    job: dict,
    gpu_id: int,
    cpus: int = DEFAULT_CPUS,
) -> dict:
    """Roda UM probe na `gpu_id`, um subprocesso, e devolve o resultado.

    Mesma forma de `runners/train.py:124`: o worker so monta lista de string e
    delega ao `_run`. Nada de torch aqui — contexto CUDA no worker Ray seguraria
    centenas de MB na GPU 0 pela corrida inteira.
    """
    # env do FILHO via subprocess(env=...), NUNCA os.environ do worker: o worker
    # Ray e reusado entre tasks e o valor vazaria para o proximo job.
    env = {
        **os.environ,
        CUDA_ENV_VAR: str(gpu_id),
        OMP_ENV_VAR: str(cpus),
        **_wandb_env(exp),
    }
    return _run(build_cmd(exp, job), env=env, name=job["key"], gpu_id=gpu_id)


# ─── Skip ─────────────────────────────────────────────────────────────────────


def _wandb_display_name(job: dict, finetune_names: dict[str, str]) -> str:
    """Nome do run no W&B, identico ao que o probe usa (src/evaluate/mlp.py:355)."""
    run_id = job["run_id"]
    fallback = f"finetune_{job['experiment_name']}"
    return "X_" + finetune_names.get(run_id, fallback)


def _skip_reason(
    exp: Experiment,
    job: dict,
    state: ExecutionState,
    finetune_names: dict[str, str],
) -> str | None:
    """Motivo para pular o job, ou `None` para executar.

    `state` le so o arquivo local do `work_dir`: barato, offline, e mente se o
    `work_dir` sumir (reexecuta tudo, caro e seguro). `wandb` consulta o
    servidor e exige estado TERMINAL — `finished`; `running` tambem segura o job
    para nao duplicar corrida em outra maquina.

    `should_skip` de `experiments/ray/skip.py` nao entra aqui de proposito: o
    ramo local dele le `run_metadata.json` do layout de classification_flim
    (scripts/classification_flim_ray.py:186), que nao existe para este probe. O
    ramo local deste runner e o ExecutionState de R6.
    """
    if exp.skip == "none":
        return None
    if exp.skip == "state":
        return "state:ok" if state.is_completed(job["key"]) else None

    wb_state, wb_id = query_wandb_state(
        _wandb_display_name(job, finetune_names),
        exp.wandb.entity or WANDB_ENTITY,
        exp.wandb.project or WANDB_PROJECT,
    )
    if wb_state in ("finished", "running"):
        return f"wandb:{wb_state}({wb_id})"
    return None


# ─── Orquestracao ─────────────────────────────────────────────────────────────


def _ray_init_kwargs(exp: Experiment) -> dict[str, Any]:
    """`num_gpus=0` sempre que subimos Ray local; GPU e nossa, nao dele."""
    kwargs = dict(RAY_INIT_KWARGS)
    if exp.resources.ray_address:
        kwargs.pop("num_gpus", None)
        kwargs["address"] = exp.resources.ray_address
    else:
        # total_slots() e nao `len(gpu_ids) * max_per_gpu`: com slots
        # heterogeneos (max_per_gpu como dict) essa multiplicacao e TypeError.
        kwargs["num_cpus"] = (
            exp.resources.cpus_per_experiment
            * GpuSlotScheduler(gpu_ids=exp.resources.gpu_ids,
                               max_per_gpu=exp.resources.max_per_gpu).total_slots()
        )
    return kwargs


def run(
    exp: Experiment,
    *,
    dry_run: bool = False,
    fail_fast: bool = False,
    gpu_ids: list[int] | None = None,
) -> list[dict]:
    """Executa a grade de avaliacao e devolve um resultado por job.

    `gpu_ids` sobrescreve `resources.gpu_ids` (precedencia flag de CLI > YAML).
    Com `dry_run`, imprime o comando de cada job e sai sem tocar em Ray.
    """
    jobs = plan(exp)
    if dry_run:
        print("=" * SEP_WIDTH)
        print(f"DRY RUN — {exp.name} (runner eval) — {len(jobs)} job(s)")
        print("=" * SEP_WIDTH)
        for job in jobs:
            print(f"  {job['key']}")
            print(f"    {' '.join(build_cmd(exp, job))}")
        print("=" * SEP_WIDTH)
        return []

    # Grade vazia nao chega aqui: `plan` levanta RuntimeError com o funil.
    work_dir = exp.output.work_dir
    os.makedirs(work_dir, exist_ok=True)
    state = ExecutionState(os.path.join(work_dir, "execution_state.json"))

    # Unico acesso ao W&B deste runner, e so quando o skip precisa dele: os
    # workers nunca chamam a API.
    finetune_names = resolve_run() if exp.skip == "wandb" else {}

    pending, skipped = [], []
    for job in jobs:
        reason = _skip_reason(exp, job, state, finetune_names)
        if reason is None:
            pending.append(job)
        else:
            skipped.append(job["key"])
            _log(f"SKIP    {job['key']}  ({reason})")

    if not pending:
        _log(f"Tudo pulado ({len(skipped)} job(s)). Nada a fazer.")
        return []

    ids = gpu_ids if gpu_ids else exp.resources.gpu_ids
    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs(exp))

    scheduler = GpuSlotScheduler(gpu_ids=ids, max_per_gpu=exp.resources.max_per_gpu)
    total = len(pending)
    queue = list(pending)
    futures: dict[Any, tuple[int, dict]] = {}
    rows: list[dict] = []
    submitted = completed = 0
    had_failure = False

    _log(
        f"{total} job(s) | {len(ids)} GPU(s) | "
        f"{scheduler.total_slots()} concorrentes | "
        f"{len(skipped)} pulado(s)"
    )

    def _submit_next() -> bool:
        nonlocal submitted
        if not queue or (fail_fast and had_failure):
            return False
        gpu_id = scheduler.pick_gpu()
        if gpu_id is None:
            return False
        job = queue.pop(0)
        future = run_eval.options(
            num_cpus=exp.resources.cpus_per_experiment
        ).remote(exp, job, gpu_id, exp.resources.cpus_per_experiment)
        scheduler.acquire(gpu_id)
        futures[future] = (gpu_id, job)
        submitted += 1
        _log(
            f"[{submitted:>4}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  fila={len(queue)}  {job['key']}"
        )
        return True

    while queue and scheduler.has_free_slot():
        _submit_next()

    while futures:
        done, _ = ray.wait(list(futures), num_returns=1, timeout=None)
        future = done[0]
        gpu_id, job = futures.pop(future)
        try:
            result = ray.get(future)
        except Exception as exc:  # noqa: BLE001
            result = {
                "run_name": job["key"],
                "gpu_id": gpu_id,
                "status": "ray_error",
                "returncode": None,
                "error": str(exc),
            }
        finally:
            # release nunca no caminho feliz: um job travado nao pode segurar o slot.
            scheduler.release(gpu_id)

        completed += 1
        result.update({k: job[k] for k in ("run_id", "dataset_name", "split",
                                           "percentage", "initialization_type",
                                           "mode", "probe", "config_path")})
        rows.append(result)
        state.mark(job["key"], result)

        if result["status"] == "ok":
            _log(
                f"[{completed:>4}/{total}] OK      gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  fila={len(queue)}  {job['key']}"
            )
        else:
            _log(
                f"[{completed:>4}/{total}] {result['status'].upper():<7} gpu={gpu_id}  "
                f"{job['key']}: {result.get('error', '')}",
                "WARN",
            )
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] descartando a fila pendente.", "WARN")
                queue.clear()

        while queue and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()

    n_ok = sum(1 for row in rows if row["status"] == "ok")
    _log(f"Fim — {n_ok} ok, {len(rows) - n_ok} falha(s), {len(skipped)} pulado(s).")
    return rows
