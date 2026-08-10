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

"""svm_ijepa.py — SVM evaluation using I-JEPA (ViT-H/14) embeddings.

Runs across all datasets / splits / percentages and saves results to
``results/ijepa_svm_results.json`` and ``results/ijepa_svm_results.csv``.

Reuses:
    * ``DatasetParasite`` — dataset loading
    * ``_OneHotDataset``  — one-hot label wrapper for SVM training
    * ``train_svm``       — linear SVM fit on extracted features
    * ``compute_metrics`` — kappa / acc / f1

Usage::

    # Run all datasets × splits × percentages
    python -m src.evaluate.svm_ijepa

    # With W&B logging
    python -m src.evaluate.svm_ijepa --wandb

    # Single dataset / split / pct (for debugging)
    python -m src.evaluate.svm_ijepa --dataset helminth-eggs --split 1 --pct 25
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.evaluate.constants import DATASET_NUM_CLASSES, PERCENTAGES as PCTS, SPLITS
from src.metrics.classification import compute_metrics
from src.models.ijepa_encoder import IJEPAEncoder
from src.utils.evaluate import (
    SVM_DIAG_MISSING,
    fit_svm,
    _OneHotDataset,
    _ROOT,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DATASETS = ["helminth-eggs", "helminth-larvae", "protozoan-cysts"]

_RESULTS_DIR = os.path.join(_ROOT, "results")

# I-JEPA ViT-H/14 requires 224×224 inputs
_IMAGE_SIZE = IJEPAEncoder.IMAGE_SIZE

_DATASET_SHORT: dict[str, str] = {
    "helminth-eggs":   "eggs",
    "helminth-larvae": "larvae",
    "protozoan-cysts": "protozoan",
}


# ── Feature extraction (I-JEPA interface) ──────────────────────────────────────


@torch.no_grad()
def _extract_features_ijepa(
    encoder: IJEPAEncoder,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract I-JEPA embeddings for every sample in *dataloader*.

    Args:
        encoder:    ``IJEPAEncoder`` instance (already on device).
        dataloader: Yields ``(inputs, int_labels)``.

    Returns:
        features: ``np.ndarray`` shape ``(N, D)`` — mean-pooled ViT-H embeddings.
        y_true:   ``np.ndarray`` shape ``(N,)``  — 0-indexed class labels.
    """
    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting features"):
        feats = encoder.extract_features(inputs)       # (B, 1280) on CPU
        all_feats.append(feats.numpy())

        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


@torch.no_grad()
def _train_svm_ijepa(
    encoder: IJEPAEncoder,
    dataloader: DataLoader,
    max_iter: int = -1,
    C: float = 1e2,
) -> object:
    """Fit a linear SVM on I-JEPA embeddings.

    Labels in *dataloader* must be one-hot ``(B, num_classes)`` — use
    ``_OneHotDataset`` wrapper.  Internally converts to 1-indexed integers
    (consistent with ``train_svm`` from ``utils/evaluate``).

    ``max_iter`` defaults to ``-1`` (unbounded solver); pass the old cap
    explicitly only to reproduce a historical CSV.

    Returns:
        Fitted ``sklearn.svm.SVC`` classifier (labels are 1-indexed); solver
        diagnostics are attached as ``fit_diagnostics_``.
    """
    all_feats: list[np.ndarray] = []
    all_y: list[int] = []

    print("[INFO] Extracting train features for SVM...")
    for inputs, labels in tqdm(dataloader, desc="  Train features"):
        feats = encoder.extract_features(inputs).numpy()      # (B, 1280)
        all_feats.append(feats)
        labels_np = np.argmax(labels.cpu().numpy(), axis=1) + 1  # 1-indexed
        all_y.extend(labels_np.tolist())

    X = np.concatenate(all_feats, axis=0)   # (N, 1280)
    y = np.array(all_y, dtype=np.int64)

    return fit_svm(X, y, max_iter=max_iter, C=C, tag="SVM_IJEPA")


# ── Aggregation ───────────────────────────────────────────────────────────────


def _save_aggregated(df: pd.DataFrame) -> str:
    """Compute mean ± std per (dataset, pretrained_pct) across splits and save CSV.

    Groups the raw per-split rows by (dataset, dataset_short, pretrained_pct) and
    computes mean and std for kappa, acc, f1 across the 3 splits.

    Returns the path to the saved aggregated CSV.
    """
    agg_path = os.path.join(_RESULTS_DIR, "ijepa_svm_aggregated.csv")

    df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if df_ok.empty:
        print("[AGGREGATE] No successful rows — skipping aggregation.")
        return agg_path

    group_cols = ["dataset", "dataset_short", "pretrained_pct"]
    agg_rows: list[dict] = []

    for keys, grp in df_ok.groupby(group_cols, dropna=False):
        dataset, dataset_short, pct = keys
        row: dict = {
            "dataset": dataset,
            "dataset_short": dataset_short,
            "pretrained_pct": pct,
            "n_splits": len(grp),
            "embedding": grp["embedding"].iloc[0] if "embedding" in grp.columns else "ijepa",
        }
        for metric in ["kappa", "acc", "f1"]:
            vals = grp[metric].dropna().values
            row[metric]            = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            row[f"{metric}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        if row["n_splits"] < 3:
            print(
                f"[AGGREGATE] WARN: {dataset} pct={pct} has only "
                f"{row['n_splits']}/3 split(s) — std will be partial"
            )
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    _col_order = [
        "dataset", "dataset_short", "pretrained_pct", "n_splits", "embedding",
        "kappa", "kappa_std", "acc", "acc_std", "f1", "f1_std",
    ]
    remaining = [c for c in agg_df.columns if c not in _col_order]
    agg_df = agg_df.reindex(columns=_col_order + remaining)
    agg_df.to_csv(agg_path, index=False)

    print(f"\n[AGGREGATE] {len(agg_rows)} group(s) → {agg_path}")
    for _, r in agg_df.iterrows():
        print(
            f"  {r['dataset_short']:10s} pct={int(r['pretrained_pct']):>3}  "
            f"kappa={r['kappa']:.4f}±{r['kappa_std']:.4f}  "
            f"acc={r['acc']:.4f}±{r['acc_std']:.4f}  "
            f"f1={r['f1']:.4f}±{r['f1_std']:.4f}"
        )
    return agg_path


# ── Main evaluation loop ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVM evaluation using I-JEPA (ViT-H/14) embeddings."
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log per-experiment metrics to W&B.",
    )
    parser.add_argument(
        "--dataset", choices=DATASETS, default=None,
        help="Restrict to a single dataset (default: all).",
    )
    parser.add_argument(
        "--split", type=int, choices=SPLITS, default=None,
        help="Restrict to a single split (default: all).",
    )
    parser.add_argument(
        "--pct", type=int, choices=PCTS, default=None,
        help="Restrict to a single percentage (default: all).",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help=(
            "Skip SVM evaluation. Load existing ijepa_svm_results.csv and "
            "compute mean ± std per (dataset, pct) across splits. "
            "Use this when raw results already exist."
        ),
    )
    args = parser.parse_args()

    # ── Aggregate-only shortcut ────────────────────────────────────────────────
    if args.aggregate_only:
        csv_path = os.path.join(_RESULTS_DIR, "ijepa_svm_results.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] Raw results not found at {csv_path}. Run without --aggregate-only first.")
            return
        df = pd.read_csv(csv_path)
        _save_aggregated(df)
        return

    datasets = [args.dataset] if args.dataset else DATASETS
    splits   = [args.split]   if args.split   else SPLITS
    pcts     = [args.pct]     if args.pct     else PCTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = _build_test(_IMAGE_SIZE)

    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # Load encoder once — shared across all experiments
    encoder = IJEPAEncoder(device=device)

    rows: list[dict] = []

    for dataset in datasets:
        num_classes = DATASET_NUM_CLASSES.get(dataset, 9)

        for split in splits:
            for pct in pcts:
                exp_tag = f"{dataset}_split_{split}_pct_{pct}"
                print(f"\n{'=' * 70}")
                print(f"  I-JEPA SVM : {exp_tag}")
                print(f"{'=' * 70}")

                run_name = f"svm_ijepa_{exp_tag}"
                base_row = {
                    "wandb_run_id":  "",
                    "run_name":      run_name,
                    "model_type":    "SVM",
                    "dataset":       dataset,
                    "dataset_short": _DATASET_SHORT.get(dataset, dataset),
                    "split":         split,
                    "pretrained_pct": pct,
                    "embedding":     "ijepa",
                    "freeze_status": "n/a",
                    "ckpt_source":   "facebook/ijepa_vith14_1k",
                }

                try:
                    # ── Train ──────────────────────────────────────────────
                    train_base = DatasetParasite(
                        set_name="train",
                        split=split,
                        percentage=pct,
                        transform=transform,
                        loader="ift_lab",
                        path_dataset=dataset,
                    )
                    train_loader = DataLoader(
                        _OneHotDataset(train_base, num_classes),
                        batch_size=32,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                    )
                    clf = _train_svm_ijepa(encoder, train_loader)

                    # ── Test ───────────────────────────────────────────────
                    test_ds = DatasetParasite(
                        set_name="test",
                        split=split,
                        percentage=pct,
                        transform=transform,
                        loader="ift_lab",
                        path_dataset=dataset,
                    )
                    test_loader = DataLoader(
                        test_ds,
                        batch_size=32,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                    )
                    feats, y_true = _extract_features_ijepa(encoder, test_loader)

                    # SVM trained with 1-indexed labels → convert back to 0-indexed
                    y_pred = clf.predict(feats) - 1

                    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
                    # Solver diagnostics ride on the classifier (fit_svm_with_diagnostics).
                    diag = getattr(clf, "fit_diagnostics_", SVM_DIAG_MISSING)

                    print(
                        f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                        f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  "
                        f"fit_status={diag['svm_fit_status']}  n_sv={diag['svm_n_sv']}"
                    )

                    rows.append({**base_row, **metrics, **diag, "status": "ok", "error": ""})

                    # ── Optional W&B logging ───────────────────────────────
                    if args.wandb:
                        import wandb  # noqa: PLC0415
                        from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
                        wandb_run = wandb.init(
                            project=PROJECT,
                            entity=ENTITY,
                            name=run_name,
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
                        **SVM_DIAG_MISSING,
                        "status": "error",
                        "error": str(exc),
                    })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    _save_aggregated(pd.DataFrame(rows))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(_RESULTS_DIR, "ijepa_svm_results.json")
    json_rows = []
    for r in rows:
        json_rows.append({
            "run_name":      r["run_name"],
            "dataset":       r["dataset"],
            "dataset_short": r["dataset_short"],
            "split":         r["split"],
            "pretrained_pct": r["pretrained_pct"],
            "embedding":     r["embedding"],
            "ckpt_source":   r["ckpt_source"],
            "metrics": {
                "acc":   r.get("acc", float("nan")),
                "kappa": r.get("kappa", float("nan")),
                "f1":    r.get("f1", float("nan")),
            },
            # New keys only — existing ones above are untouched.
            "svm": {k: r.get(k, SVM_DIAG_MISSING[k]) for k in SVM_DIAG_MISSING},
        })
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_rows, fh, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(_RESULTS_DIR, "ijepa_svm_results.csv")
    _col_order = [
        "wandb_run_id", "run_name", "model_type", "dataset", "dataset_short",
        "split", "pretrained_pct", "embedding", "freeze_status", "ckpt_source",
        "kappa", "acc", "f1",
        *SVM_DIAG_MISSING,   # new columns, appended — nothing renamed
        "status", "error",
    ]
    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] JSON : {json_path}")
    print(f"[DONE] CSV  : {csv_path}")
    print(f"{'=' * 70}")

    # ── Optional W&B summary ──────────────────────────────────────────────────
    if args.wandb:
        import wandb  # noqa: PLC0415
        from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
        df_ok = df[df["status"] == "ok"]
        summary_metrics: dict = {}
        for m in ("kappa", "acc", "f1"):
            if m in df_ok.columns and not df_ok[m].isna().all():
                summary_metrics[f"summary/mean_{m}"] = float(df_ok[m].mean())
                summary_metrics[f"summary/std_{m}"]  = float(df_ok[m].std())
        summary_run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name="svm_ijepa_summary",
            job_type="evaluation_summary",
            reinit=True,
        )
        wandb.log(summary_metrics)
        _meta_cols = ["run_name", "dataset", "dataset_short", "split", "pretrained_pct", "embedding"]
        wandb.log({
            "svm_ijepa/results_table": wandb.Table(
                dataframe=df_ok[_meta_cols + ["kappa", "acc", "f1"]].reset_index(drop=True)
            )
        })
        artifact = wandb.Artifact("ijepa_svm_results", type="evaluation_results")
        artifact.add_file(csv_path, name="ijepa_svm_results.csv")
        summary_run.log_artifact(artifact)
        summary_run.finish()
        print("[W&B] Upload complete.")


if __name__ == "__main__":
    main()
