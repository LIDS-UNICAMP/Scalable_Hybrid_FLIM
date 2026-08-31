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

"""run_missing_mlp.py — Run MLP fine-tuning for specific run_ids via Ray.

Thin wrapper over src.evaluate.ray_mlp.run_ray_experiment.
Discovers the YAML config for each run_id, submits all as Ray tasks,
and collects results — respecting the GPU slot budget.

Usage::

    # Freeze mode, 2 GPUs, up to 4 jobs running at once
    python -m experiments.oneoff.run_missing_mlp --mode freeze --num-gpus 2 --jobs 4 \\
        --run-ids qd7ie6xu 62n3kg3i 75m92cgn

    # Unfreeze mode, pipe from file
    python -m experiments.oneoff.run_missing_mlp --mode unfreeze --num-gpus 2 --jobs 4 \\
        --run-ids 4e11iyuo p4uj8mnn 63d9iqf9 7cus9dk9 zvv3xn5o
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import yaml

# Rodado como ``python -m experiments.oneoff.run_missing_mlp`` da raiz do
# repositorio: a raiz ja e sys.path[0], entao o antigo sys.path.insert saiu.
from experiments.constants import MLP_CONFIGS_DIR, RESULTS_DIR

try:
    import ray
except ImportError as e:
    raise ImportError("ray is not installed. Run: pip install 'ray[tune]'") from e

from src.evaluate.ray_mlp import run_ray_experiment, _RESULTS_DIR


# ─── Config discovery ─────────────────────────────────────────────────────────


def find_yaml(run_id: str, mode: str) -> str | None:
    """Return the YAML path for *run_id* under configs/evaluate/mlp/{mode}/."""
    pattern = os.path.join(MLP_CONFIGS_DIR, mode, "**", f"{run_id}.yaml")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ray-parallel MLP fine-tune for a specific list of run_ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["freeze", "unfreeze"], required=True,
                        help="Fine-tune mode.")
    parser.add_argument("--run-ids", nargs="+", required=True, metavar="RUN_ID",
                        help="List of W&B run_ids to evaluate.")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Total GPUs available to Ray.")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Max experiments running simultaneously.")
    parser.add_argument("--cpus-per-job", type=int, default=4,
                        help="CPU cores per experiment.")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(RESULTS_DIR, "ray_finetune"),
                        help="Root directory for experiment artifacts.")
    args = parser.parse_args()

    gpus_per_job = args.num_gpus / args.jobs  # fractional GPU per slot

    # ── Discover configs ───────────────────────────────────────────────────
    configs: list[dict] = []
    missing_yaml: list[str] = []

    for run_id in args.run_ids:
        yaml_path = find_yaml(run_id, args.mode)
        if yaml_path is None:
            missing_yaml.append(run_id)
            continue
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["config_path"] = yaml_path
        cfg["freeze_encoder"] = (args.mode == "freeze")
        configs.append(cfg)

    if missing_yaml:
        print(f"[WARN] No YAML found for {len(missing_yaml)} run_id(s) — skipping:")
        for rid in missing_yaml:
            print(f"  {rid}")

    if not configs:
        print("[run_missing_mlp] No valid configs found. Exiting.")
        return

    print(f"\n{'=' * 60}")
    print(f"[run_missing_mlp] mode={args.mode}  jobs={args.run_ids}")
    print(f"  GPUs: {args.num_gpus}  |  max concurrent: {args.jobs}  "
          f"|  {gpus_per_job:.2f} GPU/job  |  {args.cpus_per_job} CPU/job")
    print(f"  {len(configs)} experiment(s) queued.")
    print(f"{'=' * 60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # ── Init Ray ──────────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True, log_to_driver=True)

    # ── Submit all tasks ──────────────────────────────────────────────────
    futures = [
        run_ray_experiment.options(
            num_cpus=args.cpus_per_job,
            num_gpus=gpus_per_job,
        ).remote(cfg, args.output_dir, num_workers=args.cpus_per_job, log_to_wandb=True)
        for cfg in configs
    ]

    # ── Collect as they finish ─────────────────────────────────────────────
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
            name   = result.get("ray_run_name", "?")
            kappa  = result.get("kappa", float("nan"))
            acc    = result.get("acc",   float("nan"))
            f1     = result.get("f1",    float("nan"))
            if status == "ok":
                print(f"[{completed:>3}/{len(futures)}] OK    {name}")
                print(f"         kappa={kappa:.4f}  acc={acc:.4f}  f1={f1:.4f}")
            else:
                print(f"[{completed:>3}/{len(futures)}] ERROR {name}: {result.get('error', '')}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(_RESULTS_DIR, f"missing_mlp_{args.mode}_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    n_ok  = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok
    print(f"\n{'=' * 60}")
    print(f"[run_missing_mlp] DONE — {n_ok} OK, {n_err} failed.")
    print(f"[run_missing_mlp] Results: {csv_path}")
    print(f"{'=' * 60}")

    ray.shutdown()


if __name__ == "__main__":
    main()
