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

"""svm.py — SVM evaluation entry-point for src/evaluate/.

Re-uses all functions from src.utils.evaluate. Results saved to
results/svm_results.csv (instead of the project root).

Usage:
    python -m src.evaluate.svm
    python -m src.evaluate.svm --wandb-update   # refresh W&B cache first
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics
from src.modules.lejepa_line_module import LejepaLineModule
from src.utils.evaluate import (
    DATASET_NUM_CLASSES,
    DEVICE,
    IMAGE_SIZE,
    _OneHotDataset,
    _ROOT,
    extract_features,
    find_best_checkpoint,
    parse_experiment_name,
    resolve_available_experiments,
    train_svm,
)
from src.utils.get_names_wandb import ENTITY, PROJECT

_RESULTS_DIR = os.path.join(_ROOT, "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVM evaluation of LeJEPA Line encoders."
    )
    parser.add_argument(
        "--wandb-update", action="store_true",
        help=(
            "Refresh the local W&B metadata cache (ids_wandb.json) from the "
            "live W&B API before running.  By default the cache is used as-is."
        ),
    )
    args = parser.parse_args()

    transform = _build_test(IMAGE_SIZE)
    rows: list[dict] = []
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("[FILTER] Resolving available experiments (local weights ∩ W&B)...")
    print(f"{'=' * 70}")
    available_experiments = resolve_available_experiments(update_wandb=args.wandb_update)
    print(f"[FILTER] Will evaluate {len(available_experiments)} experiment(s).\n")

    for run_id, run_name in available_experiments.items():
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
            print(f"  [INFO] Checkpoint : {os.path.relpath(ckpt_path, _ROOT)}")

            module = LejepaLineModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            module.eval()
            encoder = module.model.encoder
            for p in encoder.parameters():
                p.requires_grad_(False)

            train_base = DatasetParasite(
                set_name="train",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            clf = train_svm(encoder, train_loader)

            test_ds = DatasetParasite(
                set_name="test",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            feats, y_true = extract_features(encoder, test_loader)
            y_pred = clf.predict(feats) - 1

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
            rows.append({**base_row, **metrics, "status": "ok", "error": ""})
            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )

            wandb_run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=f"svm_{run_name}",
                config={**base_row, "num_classes": num_classes},
                reinit=True,
            )
            wandb.log({f"svm/{k}": v for k, v in metrics.items()})
            wandb_run.finish()

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                **base_row,
                "kappa": float("nan"),
                "acc": float("nan"),
                "f1": float("nan"),
                "status": "error",
                "error": str(exc),
            })

    csv_path = os.path.join(_RESULTS_DIR, "svm_results.csv")
    _meta = ["wandb_run_id", "experiment_name", "dataset_name",
             "split_id", "percentage", "initialization_type"]
    _metrics = ["kappa", "acc", "f1"]
    _extra = ["status", "error"]
    _col_order = _meta + _metrics + _extra

    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to: {csv_path}")
    print(f"{'=' * 70}")

    # ── Upload CSV + summary metrics to W&B ──────────────────────────────────
    df_ok = df[df["status"] == "ok"]
    summary_metrics: dict = {}
    for m in _metrics:
        if m in df_ok.columns and not df_ok[m].isna().all():
            summary_metrics[f"summary/mean_{m}"] = float(df_ok[m].mean())
            summary_metrics[f"summary/std_{m}"]  = float(df_ok[m].std())

    print(f"\n[W&B] Uploading CSV and summary to project '{PROJECT}'...")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v:.4f}")

    summary_run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="svm_summary",
        job_type="evaluation_summary",
        reinit=True,
    )
    wandb.log(summary_metrics)
    # Log full results as a W&B Table
    wandb.log({"svm/results_table": wandb.Table(dataframe=df_ok[_meta + _metrics].reset_index(drop=True))})
    # Upload CSV as an artifact
    artifact = wandb.Artifact("svm_results", type="evaluation_results")
    artifact.add_file(csv_path, name="svm_results.csv")
    summary_run.log_artifact(artifact)
    summary_run.finish()
    print("[W&B] Upload complete.")


if __name__ == "__main__":
    main()
