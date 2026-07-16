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

"""
Delete offline wandb run directories and matching model checkpoints that are
no longer present as online runs.

Adapted from ophira_AI/scripts/clear_wandb_cache.py.

Usage
-----
    python scripts/clear_wandb_cache.py

Set WANDB_PROJECT to the W&B project name configured in configs/default.yaml
(default: "flim-ssl_old").  Add run IDs to WHITELIST to prevent them from being
deleted even if they are not online.
"""

import os
import shutil

import wandb

# ── configuration ────────────────────────────────────────────────────────────
WANDB_PROJECT = "flim-ssl_old"
MODEL_CHECKPOINT_PATH = os.path.join("logs", WANDB_PROJECT)
LOG_PATH = os.path.join("logs", "wandb")
WHITELIST: list[str] = [
    # "<run_id>",   # add run IDs here to protect them from deletion
]
# ─────────────────────────────────────────────────────────────────────────────


def _collect_local_run_ids(path: str) -> dict[str, str]:
    """Return {run_id: dir_path} for every wandb offline-run directory."""
    mapping: dict[str, str] = {}
    if not os.path.exists(path):
        return mapping
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isdir(full) and not os.path.islink(full):
            # wandb offline dirs are named  run-<timestamp>-<id>  or  offline-run-…
            run_id = entry.split("-")[-1]
            mapping[run_id] = full
    return mapping


def _collect_checkpoint_run_ids(path: str) -> dict[str, str]:
    """Return {run_id: dir_path} for model checkpoint directories."""
    mapping: dict[str, str] = {}
    if not os.path.exists(path):
        return mapping
    for version in os.listdir(path):
        full = os.path.join(path, version)
        if os.path.isdir(full):
            mapping[version] = full
    return mapping


def _fetch_online_run_ids(project: str) -> dict[str, str]:
    """Return {run_id: run_name} from the W&B API."""
    api = wandb.Api()
    return {run.id: run.name for run in api.runs(project)}


def _purge(mapping: dict[str, str], online_ids: set[str], label: str) -> None:
    kept = deleted = 0
    for run_id, path in mapping.items():
        if run_id in WHITELIST:
            kept += 1
            continue
        if run_id not in online_ids:
            shutil.rmtree(path)
            deleted += 1
        else:
            kept += 1
    print(f"{label}: {kept} kept, {deleted} deleted")


if __name__ == "__main__":
    local_runs = _collect_local_run_ids(LOG_PATH)
    local_ckpts = _collect_checkpoint_run_ids(MODEL_CHECKPOINT_PATH)
    online_ids = set(_fetch_online_run_ids(WANDB_PROJECT))

    _purge(local_runs, online_ids, "Offline run dirs")
    _purge(local_ckpts, online_ids, "Model checkpoint dirs")
