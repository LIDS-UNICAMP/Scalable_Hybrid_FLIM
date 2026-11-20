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

"""runners/spifil_dense.py — uma celula da grade E1/E2/E3 da rede densa SPiFiL.

Um ponto = (experimento, dataset, split, percentual). O experimento e a chave de
`runner_args.configs` (e1/e2/e3), que ocupa o eixo `init` da grade
(experiments/ray/grid.py), e aponta para o YAML de modelo que os dois filhos leem:

    python -m src.modules.spifil_dense_growth_module --config <yaml> --dataset <ds> --split <s> --percentage <p>
    python -m src.evaluate.svm_spifil_dense          --config <yaml> --dataset <ds> --split <s> --percentage <p>

Treino e depois SVM, em sequencia, na MESMA GPU; o SVM so roda se o treino saiu
com 0. O nome do run e o `cell_name` do modulo (experiments/ray/paths.py), e o
modulo grava em `<output_root>/<cell_name>/` — por isso o `output_root` dos tres
YAMLs tem de ser o `output.work_dir` do experiment YAML (o pre-voo confere), ou o
`skip: state` nao acha o `run_metadata.json`.

`build_cmd` e pura, como em runners/train.py: e ela que o `--dry-run` imprime e
ela que `run_spifil_dense` executa. stdout+stderr dos dois filhos vao para
`<output.log_dir>/<run_name>.log` (append), um arquivo por celula.
"""

from __future__ import annotations

import os
import subprocess
import sys

import ray
import yaml

from core.constants import OMP_ENV_VAR, PROJECT_ROOT
from experiments.constants import CUDA_ENV_VAR, DATASET_LONG_TO_SHORT, STDERR_TRUNCATE_TAIL
from experiments.ray.schema import Experiment

TRAIN_MODULE: str = "src.modules.spifil_dense_growth_module"
EVAL_MODULE: str = "src.evaluate.svm_spifil_dense"

# Default do schema (Resources.cpus_per_experiment); o launcher passa o valor do YAML.
DEFAULT_CPUS: int = 4


def build_cmd(exp: Experiment, point: dict) -> list[list[str]]:
    """Os dois comandos de UM ponto, treino e SVM. Pura: e o que o --dry-run imprime."""
    cell = [
        "--config", exp.runner_args["configs"][point["init"]],
        "--dataset", DATASET_LONG_TO_SHORT[point["dataset"]],
        "--split", str(point["split"]),
        "--percentage", str(point["percentage"]),
    ]
    return [
        [sys.executable, "-m", TRAIN_MODULE, *cell],
        [sys.executable, "-m", EVAL_MODULE, *cell],
    ]


def _log_path(exp: Experiment, run_name: str) -> str:
    """`<log_dir>/<run_name>.log`; log_dir relativo e da raiz do repo (cwd do worker varia)."""
    log_dir = exp.output.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{run_name}.log")


def _tail(path: str) -> str:
    """Fim do log da celula, para o motivo da falha no log do launcher."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()[-STDERR_TRUNCATE_TAIL:].strip()


@ray.remote  # sem num_gpus: Ray nao gerencia GPU, o pinning e CUDA_VISIBLE_DEVICES
def run_spifil_dense(exp: Experiment, point: dict, gpu_id: int, cpus: int = DEFAULT_CPUS) -> dict:
    """Treino e SVM de um ponto na `gpu_id`; mesmo formato de resultado de train._run."""
    name = point["run_name"]
    config = exp.runner_args["configs"][point["init"]]
    with open(os.path.join(PROJECT_ROOT, config), encoding="utf-8") as fh:
        alloc = ((yaml.safe_load(fh) or {}).get("run") or {}).get("cuda_alloc_conf")
    # env do FILHO via subprocess(env=...), NUNCA os.environ do worker: o worker
    # Ray e reusado entre tasks e o valor vazaria para o proximo ponto da grade.
    env = {**os.environ, CUDA_ENV_VAR: str(gpu_id), OMP_ENV_VAR: str(cpus)}
    if alloc:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc

    result: dict = {"run_name": name, "gpu_id": gpu_id}
    log = _log_path(exp, name)
    try:
        for cmd in build_cmd(exp, point):
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n$ {' '.join(cmd)}\n")
                fh.flush()
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                                      stdout=fh, stderr=subprocess.STDOUT)
            result["returncode"] = proc.returncode
            if proc.returncode != 0:  # treino falhou -> sem SVM
                result["status"] = "error"
                result["error"] = (f"returncode={proc.returncode} ({cmd[2]}), log: {log}\n"
                                   f"{_tail(log)}")
                return result
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["returncode"] = -1
        result["error"] = str(exc)
    return result
