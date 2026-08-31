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

"""classical_classifiers.py — kNN / QDA / RF / LightGBM / GP evaluation
for frozen LeJEPA Line encoder embeddings.

Adds one row per classifier variant per experiment to
results/classical_classifiers_results.csv. Saves incrementally after each
experiment so partial results survive interruptions. Resumes automatically
if the CSV already exists.

Usage:
    python -m eval.classical_classifiers
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.constants import DATASET_NUM_CLASSES, IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from eval.svm import (
    DEVICE,
    extract_features,
    find_best_checkpoint,
    parse_experiment_name,
    resolve_available_experiments,
)
from methods.lejepa.lejepa_line_module import LejepaLineModule

_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# O kNN daqui e so mais uma linha da grade sklearn; a sonda kNN sobre embeddings
# tem um unico lar, core/mixins/knn_kappa_probe_mixin.py. Nao existe eval/knn.py.
_KNN_KS = [5, 10, 15]

_META_COLS = [
    "wandb_run_id", "experiment_name", "dataset_name",
    "split_id", "percentage", "initialization_type",
]
_CLF_COLS = ["classifier", "classifier_variant"]
_METRIC_COLS = ["kappa", "acc", "f1"]
_EXTRA_COLS = ["status", "error"]
_COL_ORDER = _META_COLS + _CLF_COLS + _METRIC_COLS + _EXTRA_COLS


def _build_classifiers(n_train: int) -> list[tuple[str, str, Any]]:
    """Return list of (classifier_name, classifier_variant, fitted_estimator_factory).

    Each entry is (clf_name, variant, clf_instance). The GP entry is replaced by
    a sentinel tuple when n_train > 2000.
    """
    entries: list[tuple[str, str, Any]] = []

    for k in _KNN_KS:
        entries.append((
            "knn",
            f"knn_k{k}",
            KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1),
        ))

    entries.append(("qda", "", QuadraticDiscriminantAnalysis(reg_param=0.1)))

    entries.append(("rf", "", RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42,
    )))

    entries.append(("lgbm", "", LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        n_jobs=-1, random_state=42, verbose=-1,
    )))

    if n_train <= 2000:
        entries.append(("gp", "", GaussianProcessClassifier(kernel=RBF(), n_jobs=-1)))
    else:
        entries.append(("gp", "", None))  # None signals skip

    return entries


def _save_csv(rows: list[dict], csv_path: str) -> None:
    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _COL_ORDER]
    df = df.reindex(columns=_COL_ORDER + remaining)
    df.to_csv(csv_path, index=False)


def _nan_row(base_row: dict, clf_name: str, variant: str, status: str, error: str) -> dict:
    return {
        **base_row,
        "classifier": clf_name,
        "classifier_variant": variant,
        "kappa": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
        "status": status,
        "error": error,
    }


def main() -> None:
    transform = build_test(IMAGE_SIZE)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(_RESULTS_DIR, "classical_classifiers_results.csv")

    # ── Resume: load already-done run_ids ─────────────────────────────────────
    rows: list[dict] = []
    done_run_ids: set[str] = set()
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        done_run_ids = set(existing["wandb_run_id"].astype(str).unique())
        print(f"[RESUME] Loaded {len(rows)} existing rows ({len(done_run_ids)} runs already done).")

    print(f"\n{'=' * 70}")
    print("[FILTER] Resolving available experiments...")
    print(f"{'=' * 70}")
    available_experiments = resolve_available_experiments(update_wandb=False)
    pending = {k: v for k, v in available_experiments.items() if str(k) not in done_run_ids}
    print(f"[FILTER] {len(available_experiments)} total | {len(pending)} pending.\n")

    for run_id, run_name in pending.items():
        print(f"\n{'=' * 70}")
        print(f"  Run : {run_name}  (id={run_id})")
        print(f"{'=' * 70}")

        base_row: dict = {"wandb_run_id": run_id, "experiment_name": run_name}

        try:
            info = parse_experiment_name(run_name)
            base_row.update({
                "dataset_name": info.dataset_name,
                "split_id": info.split_id,
                "percentage": info.percentage,
                "initialization_type": info.initialization_type,
            })

            num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)

            ckpt_path = find_best_checkpoint(run_id)
            print(f"  [INFO] Checkpoint : {os.path.relpath(ckpt_path, PROJECT_ROOT)}")

            module = LejepaLineModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            module.eval()
            encoder = module.model.encoder
            for p in encoder.parameters():
                p.requires_grad_(False)

            # ── Extract train features (0-indexed labels) ──────────────────
            train_ds = ParasiteDataset(
                set_name="train",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            train_loader = DataLoader(
                train_ds, batch_size=32, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            print("  [INFO] Extracting train features...")
            X_train, y_train = extract_features(encoder, train_loader)

            # ── Extract test features ──────────────────────────────────────
            test_ds = ParasiteDataset(
                set_name="test",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            print("  [INFO] Extracting test features...")
            X_test, y_true = extract_features(encoder, test_loader)

            n_train = len(X_train)
            print(f"  [INFO] n_train={n_train}  n_test={len(X_test)}")

        except Exception as exc:
            print(f"  [ERROR] (feature extraction) {exc}")
            rows.append(_nan_row(base_row, "all", "", "error", str(exc)))
            _save_csv(rows, csv_path)
            continue

        # ── Fit and evaluate each classifier ──────────────────────────────
        for clf_name, variant, clf in _build_classifiers(n_train):
            display = variant if variant else clf_name
            print(f"\n  [{display.upper()}] fitting...")

            if clf is None:
                # GP skipped due to large dataset
                print(f"  [{display.upper()}] skipped — n_train={n_train} > 2000")
                row = _nan_row(
                    base_row, clf_name, variant,
                    "skipped_gp_too_large", f"n_train={n_train}",
                )
                rows.append(row)
                continue

            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                metrics = compute_metrics(
                    y_true=y_true, y_pred=y_pred, num_classes=num_classes,
                )
                row = {
                    **base_row,
                    "classifier": clf_name,
                    "classifier_variant": variant,
                    **metrics,
                    "status": "ok",
                    "error": "",
                }
                rows.append(row)
                print(
                    f"  [{display.upper()}] kappa={metrics['kappa']:.4f}  "
                    f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
                )

            except Exception as exc:
                print(f"  [{display.upper()}] ERROR: {exc}")
                rows.append(_nan_row(base_row, clf_name, variant, "error", str(exc)))

        # ── Partial save after each experiment ────────────────────────────
        _save_csv(rows, csv_path)
        print(f"  [SAVED] {csv_path}  ({len(rows)} rows total)")

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
