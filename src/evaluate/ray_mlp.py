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

"""ray_mlp.py — Ray-based parallel MLP fine-tuning evaluation queue.

Submits all discovered MLP experiment configs as independent Ray remote tasks,
running up to ``--num-gpus`` experiments concurrently (one GPU each).

Each experiment:
  1. Loads the pretrained encoder from the best checkpoint.
  2. Fine-tunes the ClassificationModel (frozen or unfrozen encoder).
  3. Saves the best-val-acc weights to ``<output-dir>/<ray_run_name>/weights/model_final.pth``.
  4. Writes metrics and a config snapshot to ``<output-dir>/<ray_run_name>/metrics/`` and
     ``<output-dir>/<ray_run_name>/config/``.
  5. Appends a row to ``results/ray_mlp_results.csv`` (thread-safe via the
     main process collecting futures).

Prerequisites:
    python scripts/generate_mlp_configs.py

Usage:
    python -m src.evaluate.ray_mlp --num-gpus 8
    python -m src.evaluate.ray_mlp --num-gpus 4 --mode freeze --output-dir results/ray_finetune
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch
import yaml

# ─── Project root on sys.path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ─── Ray import guard ────────────────────────────────────────────────────────
try:
    import ray
except ImportError as _ray_err:
    raise ImportError(
        "ray is not installed. Run: pip install 'ray[tune]'"
    ) from _ray_err

from src.evaluate.mlp import (
    DEVICE,
    _CONFIGS_DIR,
    _RESULTS_DIR,
    get_experiment_configs,
    load_classification_model,
    train_and_evaluate,
)
from src.evaluate.constants import DATASET_NUM_CLASSES, IMAGE_SIZE
from src.utils.evaluate import (
    parse_experiment_name,
)
from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from torch.utils.data import DataLoader

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


@ray.remote
def run_ray_experiment(
    cfg: dict,
    output_dir: str,
    num_workers: int = 4,
    log_to_wandb: bool = True,
) -> dict:
    """Ray remote task: fine-tune one experiment and persist artifacts.

    Resource requirements are declared on the ``@ray.remote`` decorator at
    call time via ``.options(num_cpus=..., num_gpus=...)`` so a single
    definition serves both the default (4 CPU, 1 GPU) and custom allocations.

    Returns a result dict with keys: ray_run_name, status, kappa, acc, f1,
    error, weights_path, metrics_path.
    """
    # Re-import inside the remote task (Ray workers have isolated processes)
    import os as _os
    import json as _json
    import yaml as _yaml
    import torch as _torch
    import numpy as _np
    from torch.utils.data import DataLoader as _DataLoader

    # Ensure project root is importable inside the Ray worker
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from src.evaluate.mlp import load_classification_model, train_and_evaluate
    from src.evaluate.constants import IMAGE_SIZE, DATASET_NUM_CLASSES
    from src.data_modules.datasets.dataset import DatasetParasite
    from src.data_modules.datasets.lejepa_dataset import _build_test
    from src.utils.evaluate import parse_experiment_name
    from src.utils.get_names_wandb import ENTITY, PROJECT
    import wandb as _wandb

    ray_run_name = build_ray_run_name(cfg)
    dirs = _artifact_dirs(output_dir, ray_run_name)
    for d in dirs.values():
        _os.makedirs(d, exist_ok=True)

    # ── Persist config snapshot ───────────────────────────────────────────
    config_snap_path = _os.path.join(dirs["config"], "config.yaml")
    with open(config_snap_path, "w") as fh:
        _yaml.dump({**cfg, "ray_run_name": ray_run_name}, fh, default_flow_style=False)

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
        _os.environ["WANDB_DISABLE_SERVICE"] = "true"

        _wandb_run = None
        if log_to_wandb:
            try:
                _wandb_run = _wandb.init(
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
                    settings=_wandb.Settings(start_method="thread"),
                )
            except Exception as _wandb_err:
                # W&B still failed despite WANDB_DISABLE_SERVICE — train without
                # W&B; weights are always saved to disk regardless.
                print(
                    f"[WARN] W&B init failed ({_wandb_err!r}); training without W&B.",
                    file=sys.stderr,
                )
                log_to_wandb = False

        # Redirect stdout/stderr to a per-experiment log file
        log_path = _os.path.join(dirs["logs"], "train.log")
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

            transform = _build_test(IMAGE_SIZE)
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

            train_loader = _DataLoader(
                DatasetParasite(set_name="train", **ds_kwargs),
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = _DataLoader(
                DatasetParasite(set_name="validation", **ds_kwargs),
                shuffle=False,
                **loader_kwargs,
            )
            test_loader = _DataLoader(
                DatasetParasite(set_name="test", **ds_kwargs),
                shuffle=False,
                **loader_kwargs,
            )

            mode_tag = "freeze" if frozen else "unfreeze"
            weights_path = _os.path.join(
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
            if _wandb_run is not None:
                _wandb_run.finish()

        # ── Persist metrics ───────────────────────────────────────────────
        metrics_path = _os.path.join(dirs["metrics"], "test_metrics.json")
        with open(metrics_path, "w") as fh:
            _json.dump(metrics, fh, indent=2)

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
        err_path = _os.path.join(dirs["logs"], "error.log")
        with open(err_path, "w") as fh:
            traceback.print_exc(file=fh)

    return result


# ─── Main orchestrator ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray-based parallel MLP fine-tuning evaluation queue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Total GPUs available to Ray (controls max concurrency).",
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=int,
        default=4,
        help="CPU cores allocated per experiment.",
    )
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1.0,
        help="GPUs allocated per experiment (fractional values supported).",
    )
    parser.add_argument(
        "--mode",
        choices=["freeze", "unfreeze", "all"],
        default="all",
        help="Which encoder-freeze mode configs to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_ROOT, "results", "ray_finetune"),
        help="Root directory for all experiment artifacts.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        default=None,
        metavar="RUN_ID_OR_NAME",
        help=(
            "Run only experiments whose run_id or experiment_name contains "
            "one of the given substrings. Accepts one or more values. "
            "Example: --experiment abc123 def456"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N experiments (after all other filters). Useful for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even if model_best.pth already exists (bypass skip check).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # ── Discover configs ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("[RAY-MLP] Discovering experiment configs...")
    print(f"{'=' * 70}")
    all_configs = get_experiment_configs(mode=args.mode)

    # get_experiment_configs already filters finetune_ names, but be explicit
    configs = [c for c in all_configs if not c.get("experiment_name", "").startswith("finetune_")]

    # ── Skip already-trained experiments (canonical weights exist) ─────────
    def _weights_exist(cfg: dict) -> bool:
        mode_tag = "freeze" if cfg.get("freeze_encoder", True) else "unfreeze"
        wp = os.path.join(_ROOT, "results", "mlp_weights", mode_tag, cfg["run_id"], "model_best.pth")
        return os.path.isfile(wp)

    if not args.force:
        skipped = [c for c in configs if _weights_exist(c)]
        configs  = [c for c in configs if not _weights_exist(c)]
        if skipped:
            print(f"[RAY-MLP] Skipping {len(skipped)} already-trained experiment(s).")
            for c in skipped:
                print(f"  (done) {build_ray_run_name(c)}")
    else:
        print("[RAY-MLP] --force: skip check disabled, will overwrite existing weights.")

    # ── Optional experiment filter (one or more run_id / name substrings) ────
    if args.experiment:
        needles = [n.lower() for n in args.experiment]
        configs = [
            c for c in configs
            if any(
                n in c.get("run_id", "").lower() or n in c.get("experiment_name", "").lower()
                for n in needles
            )
        ]
        if not configs:
            print(f"[RAY-MLP] No experiments matched --experiment {args.experiment}. Exiting.")
            return

    if args.limit:
        configs = configs[: args.limit]

    if not configs:
        print("[RAY-MLP] No eligible configs found. Exiting.")
        return

    print(f"[RAY-MLP] {len(configs)} experiment(s) queued.")
    for cfg in configs:
        print(f"  {build_ray_run_name(cfg)}")

    # ── Initialise Ray ────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(
            num_gpus=args.num_gpus,
            ignore_reinit_error=True,
            # Suppress excessive Ray logs in the main process
            log_to_driver=True,
        )

    print(f"\n[RAY-MLP] Submitting {len(configs)} task(s) "
          f"({args.cpus_per_trial} CPUs, {args.gpus_per_trial} GPU(s) each)...")
    print(f"[RAY-MLP] Output dir : {args.output_dir}")
    print(f"{'=' * 70}\n")

    # ── Submit all tasks ──────────────────────────────────────────────────
    futures = []
    for cfg in configs:
        future = run_ray_experiment.options(
            num_cpus=args.cpus_per_trial,
            num_gpus=args.gpus_per_trial,
        ).remote(cfg, args.output_dir, num_workers=0)
        futures.append(future)

    # ── Collect results as they complete ──────────────────────────────────
    rows: list[dict] = []
    pending = list(futures)
    completed = 0

    while pending:
        done, pending = ray.wait(pending, num_returns=1, timeout=None)
        for ref in done:
            try:
                result = ray.get(ref)
            except Exception as exc:
                result = {"ray_run_name": "unknown", "status": "ray_error", "error": str(exc)}
            rows.append(result)
            completed += 1
            status = result.get("status", "?")
            name = result.get("ray_run_name", "?")
            kappa = result.get("kappa", float("nan"))
            acc = result.get("acc", float("nan"))
            f1 = result.get("f1", float("nan"))
            if status == "ok":
                print(
                    f"[{completed:>3}/{len(futures)}] OK    {name}\n"
                    f"         kappa={kappa:.4f}  acc={acc:.4f}  f1={f1:.4f}"
                )
            else:
                err = result.get("error", "")
                print(f"[{completed:>3}/{len(futures)}] ERROR {name}: {err}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(_RESULTS_DIR, "ray_mlp_results.csv")
    _meta = [
        "ray_run_name", "run_id", "experiment_name", "dataset_name", "split_id",
        "percentage", "initialization_type", "freeze_encoder", "config_path",
    ]
    _metrics = ["kappa", "acc", "f1"]
    _extra = ["status", "error", "weights_path", "metrics_path"]
    _col_order = _meta + _metrics + _extra

    df = pd.DataFrame(rows)
    if not df.empty:
        remaining = [c for c in df.columns if c not in _col_order]
        df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok

    print(f"\n{'=' * 70}")
    print(f"[RAY-MLP] DONE — {n_ok} succeeded, {n_err} failed.")
    print(f"[RAY-MLP] Results saved to: {csv_path}")
    print(f"{'=' * 70}")

    ray.shutdown()


if __name__ == "__main__":
    main()
