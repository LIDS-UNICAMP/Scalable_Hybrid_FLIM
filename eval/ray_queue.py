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

"""ray_queue.py — a metade NAO-ray de `ray_mlp.py` + `ray_mlp_queue.py`.

Os dois arquivos de origem misturavam duas responsabilidades. O criterio de corte
foi mecanico: **se a linha depende de `ray`, e orquestracao e foi para
`experiments/ray/runners/eval.py`; se nao depende, e sonda e esta aqui.**

O que FICOU aqui (probe, roda in-process, sem cluster):

| daqui                       | origem                              |
|-----------------------------|-------------------------------------|
| `_log`                      | `src/evaluate/ray_mlp_queue.py:97`   |
| `build_ray_run_name`        | `src/evaluate/ray_mlp.py:85`         |
| `_artifact_dirs`            | `src/evaluate/ray_mlp.py:106`        |
| `run_experiment_artifacts`  | `src/evaluate/ray_mlp.py:123` (corpo)|
| `run_experiment_inproc`     | `src/evaluate/ray_mlp_queue.py:194` (corpo) |
| `_save_csv` / `_print_summary` | `src/evaluate/ray_mlp_queue.py:446` / `:458` |

O que SAIU para `experiments/ray/runners/eval.py` (importa `ray`):

| o que                       | origem                                    |
|-----------------------------|-------------------------------------------|
| guarda de import de `ray`   | `ray_mlp.py:59-64`, `ray_mlp_queue.py:76-80` |
| decorador `@ray.remote`     | `ray_mlp.py:122`, `ray_mlp_queue.py:193`   |
| `run_queue` (init/wait/get/shutdown) | `ray_mlp_queue.py:275`            |
| `main()` com argparse       | `ray_mlp.py:331`, `ray_mlp_queue.py:504`   |
| `_weights_exist`            | `ray_mlp.py:405`                           |
| `ExecutionState`            | `ray_mlp_queue.py:106` -> `experiments/ray/execution_state.py` |
| `GpuSlotScheduler`          | `ray_mlp_queue.py:143` -> `experiments/ray/gpu_slot_scheduler.py` |

As duas funcoes `run_experiment_*` sao os CORPOS das tasks `@ray.remote`, sem o
decorador: e exatamente a linha do corte. O `gpu_id` continua opcional para que
o pinning de `ray_mlp_queue.py:221` nao se perca quando alguem chamar a sonda
fora do Ray; o runner de `experiments/` faz o pinning por `subprocess(env=...)`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import pandas as pd
import wandb
import yaml
from torch.utils.data import DataLoader

from core.constants import DATASET_NUM_CLASSES, IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.wandb import ENTITY, PROJECT
from eval.mlp import load_classification_model, run_from_yaml, train_and_evaluate
from eval.svm import parse_experiment_name

_ROOT = PROJECT_ROOT


# ─── Logging ──────────────────────────────────────────────────────────────────
#
# Origem: src/evaluate/ray_mlp_queue.py:93-103. DUPLICADO, de proposito e
# registrado: experiments/ray/runners/eval.py:91 tem o seu proprio `_log`, com
# os niveis vindos de experiments/constants.py. Fundi-los faria `eval/` importar
# de `experiments/`, o que spec_refactor.md:2587 proibe.

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)

# ─── Naming ───────────────────────────────────────────────────────────────────


def build_ray_run_name(cfg: dict) -> str:
    """Convert an experiment config dict to a Ray run name.

    Format::

        ray_finetune_lejepa_line_<dataset>_<split>_<initializer>_pct<pct>_<run_id>

    The values are taken directly from the YAML fields so they always match
    the source of truth.
    """
    dataset = cfg.get("dataset_name", "unknown")
    split = cfg.get("split", cfg.get("split_id", "?"))
    init = cfg.get("initialization_type", "unknown")
    pct = cfg.get("percentage", "?")
    run_id = cfg.get("run_id", "unknown")
    return f"ray_finetune_lejepa_line_{dataset}_split_{split}_{init}_pct{pct}_{run_id}"


# ─── Artifact layout helpers ──────────────────────────────────────────────────


def _artifact_dirs(output_dir: str, ray_run_name: str) -> dict[str, str]:
    """Return the four artifact sub-directory paths for one experiment."""
    base = os.path.join(output_dir, ray_run_name)
    return {
        "base": base,
        "config": os.path.join(base, "config"),
        "logs": os.path.join(base, "logs"),
        "metrics": os.path.join(base, "metrics"),
        "checkpoints": os.path.join(base, "checkpoints"),
        "weights": os.path.join(base, "weights"),
    }


# ─── Ray remote task ─────────────────────────────────────────────────────────



# ─── Sonda com artefatos (corpo da task @ray.remote de ray_mlp.py:123) ────────


def run_experiment_artifacts(
    cfg: dict,
    output_dir: str,
    num_workers: int = 4,
    log_to_wandb: bool = True,
    gpu_id: Optional[int] = None,
) -> dict:
    """Faz o fine-tune de UM experimento e grava config, log, metrica e peso.

    Corpo de `src/evaluate/ray_mlp.py:123` sem o decorador `@ray.remote` da
    linha 122 e sem o hack de `sys.path` da linha 147: nada aqui depende de Ray,
    entao roda direto de um teste ou de um notebook.

    Os re-imports que a origem fazia dentro da task (linhas 138-155, necessarios
    porque o worker Ray e outro processo) subiram para o topo do modulo.

    Args:
        cfg:          Config do experimento (um YAML de configs/generated/mlp/).
        output_dir:   Raiz dos artefatos; `_artifact_dirs` monta as subpastas.
        num_workers:  Workers do DataLoader.
        log_to_wandb: Liga o `wandb.init` por experimento.
        gpu_id:       Quando dado, fixa CUDA_VISIBLE_DEVICES antes de qualquer
                      contexto CUDA. `None` (default) herda o ambiente — e o
                      caso do runner de `experiments/`, que ja passa a GPU por
                      `subprocess(env=...)`.

    Returns:
        Dict com ray_run_name, status, kappa, acc, f1, error, weights_path,
        metrics_path.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    ray_run_name = build_ray_run_name(cfg)
    dirs = _artifact_dirs(output_dir, ray_run_name)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ── Persist config snapshot ───────────────────────────────────────────
    config_snap_path = os.path.join(dirs["config"], "config.yaml")
    with open(config_snap_path, "w") as fh:
        yaml.dump({**cfg, "ray_run_name": ray_run_name}, fh, default_flow_style=False)

    result: dict = {
        "ray_run_name": ray_run_name,
        "run_id": cfg.get("run_id", ""),
        "experiment_name": cfg.get("experiment_name", ""),
        "dataset_name": cfg.get("dataset_name", ""),
        "split_id": cfg.get("split", cfg.get("split_id", "")),
        "percentage": cfg.get("percentage", ""),
        "initialization_type": cfg.get("initialization_type", ""),
        "freeze_encoder": cfg.get("freeze_encoder", True),
        "config_path": cfg.get("config_path", ""),
        "kappa": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
        "status": "error",
        "error": "",
        "weights_path": "",
        "metrics_path": "",
    }

    try:
        run_name: str = cfg.get("experiment_name", "")
        info = parse_experiment_name(run_name)

        num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)
        frozen: bool = cfg.get("freeze_encoder", True)
        max_epochs: int = cfg.get("max_epochs", 50)
        lr: float = cfg.get("lr", 1e-3)
        weight_decay: float = cfg.get("weight_decay", 1e-4)
        batch_size: int = cfg.get("batch_size", 32)
        hidden_dim: int = cfg.get("hidden_dim", 256)
        dropout: float = cfg.get("dropout", 0.3)

        run_id: str = cfg["run_id"]

        # ── Init W&B (before log redirect so W&B can access real stdout) ─────
        # Disable W&B service subprocess — it conflicts with Ray's process
        # management (same root cause as num_workers=0 for DataLoaders).
        os.environ["WANDB_DISABLE_SERVICE"] = "true"

        wandb_run = None
        if log_to_wandb:
            try:
                wandb_run = wandb.init(
                    project=PROJECT,
                    entity=ENTITY,
                    name=f"X_finetune_{run_name}_{run_id}",
                    config={
                        **{k: v for k, v in cfg.items() if k != "config_path"},
                        "num_classes": num_classes,
                        "ray_run_name": ray_run_name,
                        "wandb_run_id": run_id,
                    },
                    reinit=True,
                    settings=wandb.Settings(start_method="thread"),
                )
            except Exception as wandb_err:
                # W&B still failed despite WANDB_DISABLE_SERVICE — train without
                # W&B; weights are always saved to disk regardless.
                print(
                    f"[WARN] W&B init failed ({wandb_err!r}); training without W&B.",
                    file=sys.stderr,
                )
                log_to_wandb = False

        # Redirect stdout/stderr to a per-experiment log file
        log_path = os.path.join(dirs["logs"], "train.log")
        log_fh = open(log_path, "w")
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log_fh

        try:
            # Protozoan-cysts FLIM checkpoints have 30 conv2 channels instead of 32.
            _init = cfg.get("initialization_type", "")
            _conv2_ch = 30 if (_init == "flim" and info.dataset_name == "protozoan-cysts") else None

            model, ckpt_path = load_classification_model(
                run_id, num_classes, hidden_dim=hidden_dim, dropout=dropout,
                conv2_channels=_conv2_ch,
            )

            transform = build_test(IMAGE_SIZE)
            ds_kwargs = dict(
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            loader_kwargs = dict(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_loader = DataLoader(
                ParasiteDataset(set_name="train", **ds_kwargs),
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                ParasiteDataset(set_name="validation", **ds_kwargs),
                shuffle=False,
                **loader_kwargs,
            )
            test_loader = DataLoader(
                ParasiteDataset(set_name="test", **ds_kwargs),
                shuffle=False,
                **loader_kwargs,
            )

            mode_tag = "freeze" if frozen else "unfreeze"
            weights_path = os.path.join(
                _ROOT, "results", "mlp_weights", mode_tag, run_id, "model_best.pth"
            )

            metrics = train_and_evaluate(
                model,
                train_loader,
                val_loader,
                test_loader,
                max_epochs=max_epochs,
                lr=lr,
                weight_decay=weight_decay,
                frozen=frozen,
                num_classes=num_classes,
                log_to_wandb=log_to_wandb,
                weights_path=weights_path,
            )
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
            log_fh.close()
            if wandb_run is not None:
                wandb_run.finish()

        # ── Persist metrics ───────────────────────────────────────────────
        metrics_path = os.path.join(dirs["metrics"], "test_metrics.json")
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        result.update({
            **metrics,
            "status": "ok",
            "error": "",
            "weights_path": weights_path,
            "metrics_path": metrics_path,
        })

    except Exception as exc:
        result["error"] = str(exc)

        # Write error trace to log
        import traceback
        err_path = os.path.join(dirs["logs"], "error.log")
        with open(err_path, "w") as fh:
            traceback.print_exc(file=fh)

    return result


# ─── Sonda por YAML (corpo da task @ray.remote de ray_mlp_queue.py:194) ──────


def run_experiment_inproc(
    cfg: dict,
    finetune_names: dict,
    available_experiments: dict,
    gpu_id: Optional[int] = None,
    queue_index: int = -1,
    log_to_wandb: bool = True,
) -> dict:
    """Faz o fine-tune de UM experimento delegando tudo a `eval.mlp.run_from_yaml`.

    Corpo de `src/evaluate/ray_mlp_queue.py:194` sem o decorador `@ray.remote`
    da linha 193 e sem o hack de `sys.path` das linhas 226-228. E o mesmo
    caminho de codigo de `python -m eval.mlp`; a origem dizia isso na propria
    docstring e continua verdade.

    O parametro `total` da origem (linha 202) saiu: ele so alimentava a linha de
    log da fila, que agora e do runner de `experiments/`.

    Args:
        cfg:                   Config do experimento.
        finetune_names:        `{run_id: wandb_name}` resolvido pelo chamador.
        available_experiments: `{run_id: run_name}` resolvido pelo chamador.
        gpu_id:                Quando dado, fixa CUDA_VISIBLE_DEVICES antes de
                               qualquer contexto CUDA (origem: linha 221).
                               `None` herda o ambiente.
        queue_index:           Indice de submissao, so para o CSV.
        log_to_wandb:          Liga o log de metrica no W&B.

    Returns:
        Result dict: ray_run_name, gpu_id, status, kappa, acc, f1, error, ...
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ray_run_name = build_ray_run_name(cfg)
    rows: list[dict] = []
    report: dict = {
        "executed": [],
        "skipped_missing_weights": [],
        "skipped_errors": [],
        "collision_resolved": [],
    }

    # Protozoan-cysts FLIM checkpoints were trained with 30 conv2 channels
    # (FLIM kernel selection produced 30 filters, not the default 32).
    dataset_name = cfg.get("dataset_name", "")
    init_type = cfg.get("initialization_type", "")
    if "protozoan" in dataset_name and init_type == "flim":
        conv2_channels = 30
    else:
        conv2_channels = None

    run_from_yaml(
        cfg["config_path"], rows, finetune_names, report, available_experiments,
        log_to_wandb=log_to_wandb,
        conv2_channels=conv2_channels,
    )

    result: dict = rows[0] if rows else {
        "status": "error",
        "error": "run_from_yaml returned no result",
        "kappa": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
    }
    result.update({
        "ray_run_name": ray_run_name,
        "gpu_id": gpu_id,
        "queue_index": queue_index,
    })
    return result


_META_COLS = [
    "ray_run_name", "gpu_id", "queue_index",
    "run_id", "experiment_name", "dataset_name", "split_id",
    "percentage", "initialization_type", "freeze_encoder", "config_path",
]
_METRIC_COLS = ["kappa", "acc", "f1"]
_EXTRA_COLS = ["status", "error", "weights_path", "metrics_path"]
_COL_ORDER = _META_COLS + _METRIC_COLS + _EXTRA_COLS


def _save_csv(rows: list[dict], csv_path: str) -> None:
    df = pd.DataFrame(rows)
    if not df.empty:
        remaining = [c for c in df.columns if c not in _COL_ORDER]
        df = df.reindex(columns=_COL_ORDER + remaining)
    df.to_csv(csv_path, index=False)
    _log(f"Results CSV → {csv_path}")


# ─── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(
    rows: list[dict],
    skipped: list[str],
    t_start: float,
    gpu_ids: list[int],
    max_concurrent_per_gpu: int,
) -> None:
    elapsed = time.time() - t_start
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok

    gpu_stats: dict[int, dict] = {gid: {"ok": 0, "error": 0} for gid in gpu_ids}
    for r in rows:
        gid = r.get("gpu_id")
        if gid in gpu_stats:
            gpu_stats[gid]["ok" if r.get("status") == "ok" else "error"] += 1

    print(f"\n{'=' * 70}")
    print("QUEUE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Succeeded      : {n_ok}")
    print(f"  Failed         : {n_err}")
    print(f"  Skipped (done) : {len(skipped)}")
    print(f"  Total runtime  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"\n  GPU profile    : {len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots/GPU")
    print("  Per-GPU results:")
    for gid in sorted(gpu_stats):
        s = gpu_stats[gid]
        print(f"    GPU {gid}: {s['ok']} ok, {s['error']} error")
    if skipped:
        print(f"\n  Skipped (already completed — {len(skipped)}):")
        for name in skipped[:10]:
            print(f"    - {name}")
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('ray_run_name', '?')}: {r.get('error', '')}")
    print(f"{'=' * 70}\n")
