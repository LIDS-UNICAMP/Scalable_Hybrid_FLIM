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

"""generate_mlp_configs.py — Generate all MLP evaluation YAML configs.

Creates one YAML file per (run_id, freeze_mode) combination under:
    configs/evaluate/mlp/freeze/{dataset}/split_{N}/pct_{pct}/{run_id}.yaml
    configs/evaluate/mlp/unfreeze/{dataset}/split_{N}/pct_{pct}/{run_id}.yaml

The path encodes dataset, split, and percentage. The file name is the run_id.
The YAML contains all hyperparameters and the experiment name for traceability.

Usage:
    python scripts/generate_mlp_configs.py
    python scripts/generate_mlp_configs.py --dry-run   # print paths only
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

# Allow running from the project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.evaluate import parse_experiment_name  # noqa: E402
from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: E402
from src.utils.wandb_cache import get_runs_dict_cached  # noqa: E402

# ─── Default hyperparameters ─────────────────────────────────────────────────

_DEFAULTS = {
    "max_epochs": 1000,
    "patience": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "hidden_dim": 256,
    "dropout": 0.3,
}

_CONFIGS_DIR = os.path.join(_ROOT, "configs", "evaluate", "mlp")


def build_yaml_content(run_id: str, run_name: str, freeze: bool) -> dict:
    info = parse_experiment_name(run_name)
    return {
        "run_id": run_id,
        "experiment_name": run_name,
        "dataset_name": info.dataset_name,
        "split": info.split_id,
        "percentage": info.percentage,
        "initialization_type": info.initialization_type,
        "freeze_encoder": freeze,
        **_DEFAULTS,
    }


def yaml_path(run_id: str, run_name: str, freeze: bool) -> str:
    info = parse_experiment_name(run_name)
    mode = "freeze" if freeze else "unfreeze"
    return os.path.join(
        _CONFIGS_DIR,
        mode,
        info.dataset_name,
        f"split_{info.split_id}",
        f"pct_{info.percentage}",
        f"{run_id}.yaml",
    )


def generate_all(dry_run: bool = False, update_wandb: bool = False) -> None:
    created = skipped = removed = 0
    # deduplicate=True keeps only the newest run per experiment name, so
    # re-ran experiments never produce stale configs for the old run_id.
    # Reads from local cache by default; pass update_wandb=True to refresh.
    experiments = get_runs_dict_cached(ENTITY, PROJECT, deduplicate=True, update=update_wandb)

    for freeze in (True, False):
        for run_id, run_name in experiments.items():
            try:
                path = yaml_path(run_id, run_name, freeze)
                content = build_yaml_content(run_id, run_name, freeze)
            except ValueError as exc:
                print(f"  [SKIP] {run_name}: {exc}")
                skipped += 1
                continue

            target_dir = os.path.dirname(path)

            if dry_run:
                print(f"  [DRY] {os.path.relpath(path, _ROOT)}")
                created += 1
                continue

            os.makedirs(target_dir, exist_ok=True)

            # Remove stale YAMLs in the same slot that belong to older run_ids.
            for old_yaml in os.listdir(target_dir):
                if old_yaml.endswith(".yaml") and old_yaml != os.path.basename(path):
                    old_path = os.path.join(target_dir, old_yaml)
                    os.remove(old_path)
                    print(f"  [DEL] {os.path.relpath(old_path, _ROOT)}  (superseded by {run_id})")
                    removed += 1

            with open(path, "w") as f:
                yaml.dump(content, f, default_flow_style=False, sort_keys=False)

            print(f"  [OK] {os.path.relpath(path, _ROOT)}")
            created += 1

    print(f"\nCreated {created} YAML(s), removed {removed} stale YAML(s), skipped {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths without writing files.")
    parser.add_argument("--update_wandb", action="store_true",
                        help="Refresh W&B cache before generating (default: use local cache).")
    args = parser.parse_args()
    generate_all(dry_run=args.dry_run, update_wandb=args.update_wandb)


if __name__ == "__main__":
    main()
