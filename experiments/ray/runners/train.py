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

"""Runner `train` — um ponto da grade = um subprocesso LightningCLI.

O comando e IDENTICO para todo metodo: o que muda sao os dois YAMLs resolvidos
por `experiments.ray.paths` (dataset e modelo) e o dicionario `overrides` do
experiment YAML. Nao ha argparse aqui, nao ha flag por metodo e nao ha
`python -m src.modules.<modulo>`: quem quiser mexer em hparam escreve
`overrides` no YAML e ele vira `--<chave>=<valor>` do LightningCLI.

`build_cmd` e pura de proposito — e ela que o `--dry-run` do launcher imprime e
ela que o caminho normal executa. Dois caminhos de montagem fariam o dry-run
testar codigo que nao roda em producao.

Protocolo herdado de scripts/run_ssl_ray.py:404-408, que ja fazia
`fit --config <global> --config <dataset> --config <modelo>`; a unica diferenca
e que `src/main.py` virou `train.py` na raiz do repositorio.
"""

from __future__ import annotations

import os
import subprocess
import sys

import ray

from core.constants import DEFAULT_CONFIG_YAML, OMP_ENV_VAR, PROJECT_ROOT
from experiments.constants import (
    CUDA_ENV_VAR,
    STDERR_TRUNCATE_HEAD,
    STDERR_TRUNCATE_MAX_CHARS,
    STDERR_TRUNCATE_TAIL,
)
from experiments.ray.paths import dataset_config, model_config, run_name
from experiments.ray.schema import Experiment

# Entrada do LightningCLI, relativa a PROJECT_ROOT (o `cwd` do subprocesso).
ENTRYPOINT: str = "train.py"

# Default do schema (Resources.cpus_per_experiment); o launcher passa o valor do
# YAML, este numero so cobre a chamada direta.
DEFAULT_CPUS: int = 4


def _run_name(exp: Experiment, point: dict) -> str:
    """Nome canonico do ponto. A regra vive em paths.run_name, aqui so o adaptador."""
    # Ordem posicional exata de paths.run_name: name, dataset, split, pct, init.
    # `method` e keyword-only e OBRIGATORIO: sem ele o nome sairia da familia
    # errada, e nome de familia errada e diretorio de checkpoint errado.
    return run_name(
        exp.name,
        point["dataset"],
        point["split"],
        point["percentage"],
        point["init"],
        method=exp.method,
        runner=exp.runner,
    )


def build_cmd(exp: Experiment, point: dict) -> list[str]:
    """Monta o comando de UM ponto da grade. Pura: e o que o --dry-run imprime.

    Os tres `--config` vao na ordem global -> dataset -> modelo porque o
    LightningCLI resolve chave a chave e o ultimo vence.
    """
    return [
        sys.executable, ENTRYPOINT, "fit",
        "--config", DEFAULT_CONFIG_YAML,
        "--config", dataset_config(point["dataset"], point["split"]),
        "--config", model_config(
            exp.method, point["init"], point["dataset"], point["split"]
        ),
        f"--data.init_args.percentage={point['percentage']}",
        f"--trainer.logger.init_args.name={_run_name(exp, point)}",
        # 1 = a UNICA GPU visivel, que e a gpu_id do CUDA_VISIBLE_DEVICES do env
        # do filho. Nunca "a GPU 1".
        "--trainer.devices=1",
        *[f"--{key}={value}" for key, value in exp.overrides.items()],
    ]


def _run(cmd: list[str], env: dict, name: str, gpu_id: int) -> dict:
    """Executa o subprocesso na raiz do repo e devolve o resultado do ponto."""
    result: dict = {"run_name": name, "gpu_id": gpu_id}
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            stderr = (proc.stderr or "").strip()
            if len(stderr) > STDERR_TRUNCATE_MAX_CHARS:
                stderr = (stderr[:STDERR_TRUNCATE_HEAD] + "\n...[truncated]...\n"
                          + stderr[-STDERR_TRUNCATE_TAIL:])
            result["status"] = "error"
            result["error"] = f"returncode={proc.returncode}\n{stderr}"
    except Exception as exc:
        result["status"] = "error"
        result["returncode"] = -1
        result["error"] = str(exc)
    return result


@ray.remote  # sem num_gpus: Ray nao gerencia GPU, o pinning e CUDA_VISIBLE_DEVICES
def run_train(exp: Experiment, point: dict, gpu_id: int, cpus: int = DEFAULT_CPUS) -> dict:
    """Roda um ponto da grade na `gpu_id`, um subprocesso, e devolve o resultado."""
    # env do FILHO via subprocess(env=...), NUNCA os.environ do worker: o worker
    # Ray e reusado entre tasks e o valor vazaria para o proximo ponto da grade.
    env = {**os.environ, CUDA_ENV_VAR: str(gpu_id), OMP_ENV_VAR: str(cpus)}
    return _run(build_cmd(exp, point), env=env, name=_run_name(exp, point), gpu_id=gpu_id)
