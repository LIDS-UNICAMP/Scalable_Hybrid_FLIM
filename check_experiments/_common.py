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

"""_common.py — Shared helpers for the check_experiments suite.

Provides:
  - Expected experiment matrix (DATASETS × SPLITS × PCTS × INITS)
  - ids_wandb.json loading → {canonical_name: run_id}
  - Local artifact existence checks (SSL ckpt, MLP weights, result CSVs)
  - Common CLI argument parser factory
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

# ── Repository root ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Expected experiment matrix ─────────────────────────────────────────────────
DATASETS = ["helminth-eggs", "helminth-larvae", "protozoan-cysts"]
SPLITS   = [1, 2, 3]
PCTS     = [1, 5, 25, 50, 75, 100]
INITS    = ["xavier", "random", "he", "flim", "trunc_normal"]

# ── Artifact paths ─────────────────────────────────────────────────────────────
_CACHE_FILE  = os.path.join(_ROOT, "configs", "wandb_update", "ids_wandb.json")
_SSL_LOGS    = os.path.join(_ROOT, "logs", "flim-ssl")
_MLP_WEIGHTS = os.path.join(_ROOT, "results", "mlp_weights")
_MLP_CSV     = os.path.join(_ROOT, "artifacts", "MLP", "mlp_results.csv")
_SVM_CSV     = os.path.join(_ROOT, "artifacts", "SVM", "svm_results.csv")

# Pattern that extracts ssl_run_id from wandb_run_name (trailing 8 hex chars)
_RUN_ID_RE = re.compile(r"_([a-z0-9]{8})$")


# ── Experiment dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Experiment:
    dataset: str
    split: int
    pct: int
    init: str

    @property
    def canonical_name(self) -> str:
        return (
            f"lejepa_line_{self.dataset}_split_{self.split}"
            f"_pct_{self.pct}_model_{self.init}"
        )


def all_expected() -> list[Experiment]:
    """Return the full 216-entry expected experiment matrix."""
    return [
        Experiment(d, s, p, i)
        for d in DATASETS
        for s in SPLITS
        for p in PCTS
        for i in INITS
    ]


# ── ids_wandb.json helpers ─────────────────────────────────────────────────────

def load_ids_wandb() -> dict[str, str]:
    """Return ``{canonical_name: run_id}`` from ids_wandb.json.

    When multiple W&B runs share the same name the newest one (by
    ``created_at``) is kept — matching the deduplication logic in
    ``src.utils.wandb_cache.get_runs_dict_cached``.
    """
    with open(_CACHE_FILE, encoding="utf-8") as fh:
        cache = json.load(fh)
    runs: dict = cache.get("runs", {})

    best: dict[str, tuple[str, str]] = {}  # name → (run_id, created_at)
    for run_id, info in runs.items():
        name = info.get("name", "")
        created_at = info.get("created_at", "")
        if name not in best or created_at > best[name][1]:
            best[name] = (run_id, created_at)

    return {name: run_id for name, (run_id, _) in best.items()}


def update_wandb_cache() -> None:
    """Refresh ids_wandb.json from the live W&B API."""
    from src.utils.wandb_cache import save_cache          # noqa: PLC0415
    from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
    save_cache(ENTITY, PROJECT)


# ── SSL checkpoint check ───────────────────────────────────────────────────────

def has_ssl_checkpoint(run_id: str) -> bool:
    """Return True if *run_id* has at least one .ckpt file under logs/flim-ssl/."""
    ckpt_dir = os.path.join(_SSL_LOGS, run_id, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    return bool(glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True))


# ── MLP weight check ───────────────────────────────────────────────────────────

def has_mlp_weight(run_id: str, mode: str) -> bool:
    """Return True if results/mlp_weights/{mode}/{run_id}/model_best.pth exists."""
    path = os.path.join(_MLP_WEIGHTS, mode, run_id, "model_best.pth")
    return os.path.isfile(path)


# ── Result CSV helpers ────────────────────────────────────────────────────────

def load_mlp_ok_set() -> set[tuple[str, str]]:
    """Return ``{(ssl_run_id, freeze_status)}`` for rows with status=ok.

    Checks two CSV formats:
    - artifacts/MLP/mlp_results.csv: wandb_run_name + freeze_status columns
    - results/ray_mlp_results.csv and results/missing_mlp_*.csv: run_id + freeze_encoder columns
    """
    import pandas as pd  # noqa: PLC0415

    result: set[tuple[str, str]] = set()

    # Old format: artifacts/MLP/mlp_results.csv
    if os.path.isfile(_MLP_CSV):
        df = pd.read_csv(_MLP_CSV)
        df = df[df["status"] == "ok"]
        for _, row in df.iterrows():
            m = _RUN_ID_RE.search(str(row.get("wandb_run_name", "")))
            if m:
                result.add((m.group(1), str(row.get("freeze_status", ""))))

    # New format: results/ray_mlp_results.csv and results/missing_mlp_*.csv
    _RESULTS_DIR = os.path.join(_ROOT, "results")
    ray_csvs = [os.path.join(_RESULTS_DIR, "ray_mlp_results.csv")]
    ray_csvs += glob.glob(os.path.join(_RESULTS_DIR, "missing_mlp_*.csv"))
    for csv_path in ray_csvs:
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "ok"]
        for _, row in df.iterrows():
            run_id = str(row.get("run_id", ""))
            freeze = row.get("freeze_encoder", True)
            mode = "freeze" if str(freeze).lower() in ("true", "1") else "unfreeze"
            if run_id:
                result.add((run_id, mode))

    return result


def load_svm_ok_set() -> set[str]:
    """Return set of ssl_run_ids that have status=ok.

    Checks both artifacts/SVM/svm_results.csv (old path) and
    results/svm_results.csv (current svm.py save path).
    """
    import pandas as pd  # noqa: PLC0415

    result: set[str] = set()
    candidates = [
        _SVM_CSV,
        os.path.join(_ROOT, "results", "svm_results.csv"),
    ]
    for csv_path in candidates:
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "ok"]
        result.update(df["wandb_run_id"].dropna().astype(str).tolist())
    return result


# ── CLI argument parser factory ────────────────────────────────────────────────

def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--update_wandb", action="store_true",
        help="Refresh the local W&B metadata cache before checking.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON summary to stdout.",
    )
    parser.add_argument(
        "--fail-on-missing", action="store_true",
        help="Exit with code 1 if any missing items are detected.",
    )
    return parser
